"""LTX adapter for the gradient-guided attention-steering loop.

The steering *algorithm* lives in ``steering/steering_core.py``
(``run_steering_optimization``) and is shared across all backends. This module
provides the LTX-specific seam (``LTXSteeringAdapter``) and the
``make_ltx_steering_loop`` factory whose returned callable is passed as ``loop=``
to ``DiffusionStage.run`` / ``pipeline.stage``.

At each denoising step the ``steering_model`` opts into, the core optimizes the
current-step latent ``x_t`` for ``n_iters`` iterations against the model's loss
(backprop to ``x_t``), commits the optimized latent, then takes the normal
(no-grad) denoise + scheduler step. Hooks reuse ``LTXVideoCAProbeContext`` with
``detach=False`` so the captured attention maps stay connected to ``x_t`` for
backprop.

``_make_optimizer`` and ``_latent_softmax_kl`` moved to ``steering_core``; they are
re-exported here for backward compatibility.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Iterable, Optional

import torch

from will_it_render.inference.steering_core import (  # noqa: F401  (re-exported for back-compat)
    SteeringBackendAdapter,
    _latent_softmax_kl,
    _make_optimizer,
    run_steering_optimization,
)


class LTXSteeringAdapter(SteeringBackendAdapter):
    """LTX-specific seam over the shared steering core.

    Stateful, bound to a single generation: it holds the loop-carried
    ``video_state`` / ``audio_state`` and advances them via LTX's stepper. The
    ``loop=`` closure returned by :func:`make_ltx_steering_loop` constructs one of
    these from the pipeline-provided ``(sigmas, video_state, audio_state, stepper,
    transformer, denoiser)`` and runs the core against it.
    """

    #: LTX's transformer is an X0Model, so ``grad_forward`` hands back ``video_result_g.denoised``
    #: — a genuine x̂0. ``pred_reg`` / ``MOVE_BAND(space="pred")`` mean what they say here.
    pred_is_x0 = True

    def __init__(
        self,
        *,
        sigmas: torch.Tensor,
        video_state,
        audio_state,
        stepper,
        transformer,
        denoiser,
        video_seq_len: Optional[int],
        capture_blocks: Optional[list[int]],
        opt_denoiser=None,
        step_offset: int = 0,
        source_noise_sink: Optional[dict] = None,
        sigmas_full: Optional[torch.Tensor] = None,
        source_reg_every_step: float = 0.0,
        source_trace: Optional[list] = None,
    ) -> None:
        self.sigmas = sigmas
        self.video_state = video_state
        self.audio_state = audio_state
        self.stepper = stepper
        self.transformer = transformer
        self.denoiser = denoiser
        # Cond-only denoiser for the grad forward (skips the uncond pass); falls back
        # to the real CFG denoiser when not provided.
        self.grad_denoiser = opt_denoiser if opt_denoiser is not None else denoiser
        self.video_seq_len = video_seq_len
        self.capture_blocks = capture_blocks
        self._velocity_model = None
        self._probe_cm = None
        self.probe_ctx = None
        self._post_process_latent = None
        # Video-edit (SDEdit-style) steering starts mid-schedule: ``sigmas`` is the truncated
        # tail of the full schedule. steps() then yields ABSOLUTE step numbers (what the
        # per-step probes were trained on) and every sigmas/denoiser index below converts back
        # with ``_i``. step_offset=0 (every from-scratch run) is the identity.
        self.step_offset = int(step_offset)
        # Source-video anchor (video-edit only; see source_reference). All None/0 otherwise.
        self.source_noise_sink = source_noise_sink
        self.sigmas_full = sigmas_full
        self.source_reg_every_step = float(source_reg_every_step)
        self.source_trace = source_trace
        self._source_z: Optional[torch.Tensor] = None
        self._steered_this_step = False

    def _i(self, step_idx: int) -> int:
        """Absolute step → index into this loop's (possibly truncated) ``sigmas``."""
        return step_idx - self.step_offset

    def __enter__(self) -> "LTXSteeringAdapter":
        from ltx_pipelines.utils.helpers import post_process_latent

        from will_it_render.backends.hooks.ltx_hooks import LTXVideoCAProbeContext

        self._post_process_latent = post_process_latent
        velocity_model = self.transformer.velocity_model
        self._velocity_model = velocity_model
        # Freeze model params: autograd then builds a graph only to x_t (the params are
        # in the path but their grads are neither computed nor retained), saving memory.
        for p in velocity_model.parameters():
            p.requires_grad_(False)
        # Gradient checkpointing makes backprop-to-x_t through the full 22B stack fit in
        # memory (activations recomputed in backward; weights stay on GPU). The model's
        # gate also requires train() — LTX has no dropout (dropout=0.0, SDPA dropout_p=0),
        # so train mode is behaviorally identical to eval; begin/end_optimization flip to
        # train() only around the grad forward and back to eval() for the (un-checkpointed)
        # stepping forward.
        velocity_model.set_gradient_checkpointing(True)
        self._probe_cm = LTXVideoCAProbeContext(
            velocity_model, video_seq_len=self.video_seq_len,
            capture_blocks=self.capture_blocks, detach=False,
        )
        self.probe_ctx = self._probe_cm.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self._probe_cm is not None:
            self._probe_cm.__exit__(*exc)

    def steps(self) -> Iterable[int]:
        return range(self.step_offset, self.step_offset + len(self.sigmas) - 1)

    def get_latent(self) -> torch.Tensor:
        return self.video_state.latent

    def commit_latent(self, x_t: torch.Tensor) -> None:
        self.video_state = replace(self.video_state, latent=x_t)

    def begin_optimization(self) -> None:
        self._steered_this_step = True
        self._velocity_model.train()  # enable the gradient-checkpointing gate

    def end_optimization(self) -> None:
        self._velocity_model.eval()  # un-checkpointed path for the stepping forward

    def grad_forward(self, x_t: torch.Tensor, step_idx: int) -> tuple[list, torch.Tensor]:
        self.probe_ctx.active = True
        self.probe_ctx.captures = []
        self.probe_ctx._cond_pass_complete = False
        video_result_g, _ = self.grad_denoiser(
            self.transformer,
            replace(self.video_state, latent=x_t),
            self.audio_state,
            self.sigmas,
            self._i(step_idx),
        )
        captures = list(self.probe_ctx.captures)
        return captures, video_result_g.denoised

    def velocity(self, x_t: torch.Tensor, pred: torch.Tensor, step_idx: int) -> torch.Tensor:
        """``v = (x_t − x̂0) / σ_t`` — LTX predicts x̂0, so the velocity has to be derived.

        ``grad_forward`` returns ``video_result_g.denoised``, and that same field is what
        ``denoise_step`` hands to ``stepper.step(sample, denoised_sample, sigmas, i)`` as the
        DENOISED prediction. So on LTX the second return is ``x̂0``, not ``v`` (WAN's model
        predicts ``v`` and needs no conversion — see the base-class default). This reproduces
        ``ltx_core.utils.to_velocity`` exactly, the function the sampler itself calls, so a rule
        reading ``flow`` now sees the same velocity the Euler step advances along.
        """
        return (x_t - pred) / self.sigmas[self._i(step_idx)].to(device=x_t.device, dtype=x_t.dtype)

    def estimate_x0(self, x_t: torch.Tensor, step_idx: int) -> torch.Tensor:
        """Predicted clean latent x̂0 from x_t at this step (diagnostic / decode).

        The LTX counterpart of :meth:`WanSteeringLoop.estimate_x0`, whose absence is why
        ``X0_HOOK`` / ``X0_POINTS`` -- and every x̂0 decode script built on them -- worked only on
        WAN. It is far cheaper here than on WAN: LTX's denoiser PREDICTS x̂0, so there is nothing
        to derive. WAN predicts a velocity and has to rebuild the CFG combination and apply
        ``x̂0 = x_t - sigma*v``; here ``video_result.denoised`` already IS x̂0, and it is the same
        field ``denoise_step`` hands the stepper as ``denoised_sample``.

        Uses ``self.denoiser`` (not ``grad_denoiser``) and leaves probe capture off, so this is a
        read-only forward: no graph, no captures, no scheduler-state change.
        """
        self.probe_ctx.active = False
        with torch.no_grad():
            video_result, _ = self.denoiser(
                self.transformer,
                replace(self.video_state, latent=x_t),
                self.audio_state,
                self.sigmas,
                self._i(step_idx),
            )
        return video_result.denoised.detach()

    def shadow_reference(self, step_idx: int, capture: bool) -> Optional[list]:
        """Advance a never-steered shadow trajectory one step (``attn_anchor_ref="unsteered"``).

        Implements the attention-anchor regularizer's reference trajectory. The shadow
        starts from the same initial latent and is denoised through the SAME CFG denoiser in
        eval() under no_grad, so it walks the un-steered trajectory. When ``capture`` is set
        (a steered step) it returns that forward's cond-pass captures: the reference attention
        the un-steered generation would show at this step. Called once per denoising step,
        before the step's optimization; never called unless the term is on.
        """
        if not hasattr(self, "_u_video_state"):
            self._u_video_state = (replace(self.video_state, latent=self.video_state.latent.detach().clone())
                                   if self.video_state is not None else None)
            self._u_audio_state = (replace(self.audio_state, latent=self.audio_state.latent.detach().clone())
                                   if self.audio_state is not None else None)
        post_process_latent = self._post_process_latent
        self._velocity_model.eval()
        self.probe_ctx.active = bool(capture)
        self.probe_ctx.captures = []
        self.probe_ctx._cond_pass_complete = False
        caps = None
        with torch.no_grad():
            u_vr, u_ar = self.denoiser(
                self.transformer, self._u_video_state, self._u_audio_state, self.sigmas, self._i(step_idx)
            )
            if capture:
                caps = list(self.probe_ctx.captures)
            if self._u_video_state is not None and u_vr is not None:
                u_dv = post_process_latent(
                    u_vr.denoised, self._u_video_state.denoise_mask, self._u_video_state.clean_latent,
                )
                self._u_video_state = replace(
                    self._u_video_state,
                    latent=self.stepper.step(self._u_video_state.latent, u_dv, self.sigmas, self._i(step_idx)),
                )
            if self._u_audio_state is not None and u_ar is not None:
                u_da = post_process_latent(
                    u_ar.denoised, self._u_audio_state.denoise_mask, self._u_audio_state.clean_latent,
                )
                self._u_audio_state = replace(
                    self._u_audio_state,
                    latent=self.stepper.step(self._u_audio_state.latent, u_da, self.sigmas, self._i(step_idx)),
                )
        self.probe_ctx.active = False
        return caps

    def source_reference(self, step_idx: int) -> Optional[torch.Tensor]:
        """Point on the SOURCE video's noising line at absolute step ``step_idx`` (video edit).

        ``x_orig = (1−σ)·clean_latent + σ·z`` with ``σ = sigmas_full[step_idx]``, the source
        latent ``video_state.clean_latent`` (already patchified, never changes over the loop)
        and the SAME noise ``z`` the initial re-noising drew (recorded by the backend's noiser
        into ``source_noise_sink``, keyed by latent shape). ``None`` when not configured —
        every from-scratch run — so the core's ``source_reg`` term is skipped entirely.
        """
        if self.sigmas_full is None or self.source_noise_sink is None:
            return None
        if self._source_z is None:
            self._source_z = self.source_noise_sink.get(tuple(self.video_state.latent.shape))
            if self._source_z is None:
                return None
        sigma = float(self.sigmas_full[step_idx].item())
        return (1.0 - sigma) * self.video_state.clean_latent + sigma * self._source_z

    def _source_pull(self, step_idx: int) -> None:
        """After EVERY denoising step (steered or not): log the MSE gap to the source line at
        the sigma the latent now sits at and, if ``source_reg_every_step`` > 0, blend toward
        it (no backprop). No-op unless source tracking is configured."""
        did_steer, self._steered_this_step = self._steered_this_step, False
        if self.video_state is None:
            return
        x_orig_next = self.source_reference(step_idx + 1)
        if x_orig_next is None:
            return
        gap_before = float(((self.video_state.latent - x_orig_next) ** 2).mean())
        if self.source_reg_every_step:
            self.video_state = replace(
                self.video_state,
                latent=self.video_state.latent
                + self.source_reg_every_step * (x_orig_next - self.video_state.latent),
            )
            gap_after = float(((self.video_state.latent - x_orig_next) ** 2).mean())
        else:
            gap_after = gap_before
        if self.source_trace is not None:
            self.source_trace.append({
                "step": step_idx, "did_steer": bool(did_steer),
                "source_gap_before": gap_before, "source_gap_after": gap_after,
            })

    def denoise_step(self, step_idx: int) -> None:
        self.probe_ctx.active = False
        post_process_latent = self._post_process_latent
        with torch.no_grad():
            video_result, audio_result = self.denoiser(
                self.transformer, self.video_state, self.audio_state, self.sigmas, self._i(step_idx)
            )
            if self.video_state is not None and video_result is not None:
                dv = post_process_latent(
                    video_result.denoised,
                    self.video_state.denoise_mask,
                    self.video_state.clean_latent,
                )
                # VARIATION 2 seam: let the core re-point the advance before the stepper consumes
                # it. `dv` is x̂0, which is exactly what stepper.step() calls `denoised_sample`, so
                # a hook that wants to rotate the velocity returns a modified x̂0 and the stepper
                # stays untouched. Applied AFTER post_process_latent so any clean-latent/mask
                # compositing is respected, and never set outside a steered step.
                _adj = getattr(self, "denoise_adjust", None)
                if _adj is not None:
                    dv = _adj(self.video_state.latent, dv, step_idx)
                self.video_state = replace(
                    self.video_state,
                    latent=self.stepper.step(self.video_state.latent, dv, self.sigmas, self._i(step_idx)),
                )
            if self.audio_state is not None and audio_result is not None:
                da = post_process_latent(
                    audio_result.denoised,
                    self.audio_state.denoise_mask,
                    self.audio_state.clean_latent,
                )
                self.audio_state = replace(
                    self.audio_state,
                    latent=self.stepper.step(self.audio_state.latent, da, self.sigmas, self._i(step_idx)),
                )
        if self.sigmas_full is not None:
            self._source_pull(step_idx)


def make_ltx_steering_loop(
    steering_model,
    target_words_indices: dict[str, list[int]],
    n_iters: int,
    optimizer_name: str = "adam",
    lr: float = 0.1,
    video_seq_len: Optional[int] = None,
    capture_blocks: Optional[list[int]] = None,
    opt_denoiser=None,
    magnitude_reg: float = 0.0,
    anchor_reg: float = 0.0,
    pred_reg: float = 0.0,
    kl_reg: float = 0.0,
    entropy_reg: float = 0.0,
    attn_anchor_reg: float = 0.0,
    attn_anchor_tokens: str = "prompt",
    attn_anchor_ref: str = "step",
    text_token_count: Optional[int] = None,
    source_noise_sink: Optional[dict] = None,
    sigmas_full: Optional[torch.Tensor] = None,
    source_reg: float = 0.0,
    source_reg_every_step: float = 0.0,
    source_trace: Optional[list] = None,
    step_callback: Optional[Callable[[int, dict], None]] = None,
    desc: str = "Steering",
    step_offset: int = 0,
) -> Callable:
    """Build the per-step optimization denoising loop (LTX pipeline ``loop=``).

    Thin wrapper: constructs an :class:`LTXSteeringAdapter` from the pipeline-provided
    state and delegates the actual optimization to the model-agnostic
    :func:`will_it_render.inference.steering_core.run_steering_optimization`. Signature/behavior are
    unchanged from before the core was extracted.

    ``n_iters`` — gradient steps per steered denoising step (0 disables steering;
    a step is also skipped when ``steering_model.should_steer`` is False).
    ``optimizer_name`` / ``lr`` — optimizer over ``x_t`` (``"simple"`` = plain GD).

    Regularizers (``magnitude/anchor/pred/kl/entropy/attn_anchor/source``) are documented on
    :func:`will_it_render.inference.steering_core.run_steering_optimization`.

    Video-edit only (``LTXBackend.run_with_steering_from_video``): ``step_offset`` = steps
    skipped at the head of the schedule (``sigmas`` is the truncated tail), so the steering
    model / callbacks see absolute step numbers. ``source_noise_sink`` + ``sigmas_full``
    enable the source-video anchor: ``source_reg`` (loss term at steered steps) and
    ``source_reg_every_step`` (no-grad blend after every step; per-step gaps appended to
    ``source_trace``). See :meth:`LTXSteeringAdapter.source_reference`.
    """

    def steering_loop(
        sigmas: torch.Tensor,
        video_state,
        audio_state,
        stepper,
        transformer,
        denoiser,
    ):
        adapter = LTXSteeringAdapter(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper,
            transformer=transformer,
            denoiser=denoiser,
            video_seq_len=video_seq_len,
            capture_blocks=capture_blocks,
            opt_denoiser=opt_denoiser,
            step_offset=step_offset,
            source_noise_sink=source_noise_sink,
            sigmas_full=sigmas_full,
            source_reg_every_step=source_reg_every_step,
            source_trace=source_trace,
        )
        run_steering_optimization(
            adapter,
            steering_model,
            target_words_indices=target_words_indices,
            n_iters=n_iters,
            optimizer_name=optimizer_name,
            lr=lr,
            magnitude_reg=magnitude_reg,
            anchor_reg=anchor_reg,
            pred_reg=pred_reg,
            kl_reg=kl_reg,
            entropy_reg=entropy_reg,
            attn_anchor_reg=attn_anchor_reg,
            attn_anchor_tokens=attn_anchor_tokens,
            attn_anchor_ref=attn_anchor_ref,
            text_token_count=text_token_count,
            source_reg=source_reg,
            step_callback=step_callback,
            desc=desc,
        )
        return adapter.video_state, adapter.audio_state

    return steering_loop
