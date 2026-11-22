"""Probe extraction hooks for LTX-2 text→video cross-attention.

Hooks into the `attention_function` callable on each `attn2` (text cross-attention)
module in all 48 transformer blocks.  Q and K are captured as the model already
computes them — post-projection, post-RMSNorm, post-RoPE — so no extra forward
passes are performed.  The only additional work is materializing `softmax(Q @ K^T)`
since fused backends (XFormers / Flash) do not expose attention weights.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from typing import Callable

import torch
from tqdm import tqdm

from will_it_render.common.probe_types import BlockCapture, CaptureType


class CapturingAttentionFunction:
    """Wraps an AttentionCallable to capture per-head attention weights and text keys.

    Appends one BlockCapture to ``probe_ctx.captures`` each time it is called.
    Using a reference to the context (not the list) so that replacing
    ``probe_ctx.captures`` between steps is safe.

    When ``probe_ctx.video_seq_len`` is set, only the first ``video_seq_len`` query
    tokens (the video portion of a joint video+audio sequence) are captured.
    """

    def __init__(self, original_fn, probe_ctx: "LTXVideoCAProbeContext") -> None:
        self._fn = original_fn
        self._ctx = probe_ctx

    def _store(self, t: torch.Tensor) -> torch.Tensor:
        """Detach for extraction (default); keep the graph when steering (detach=False)."""
        return t if not self._ctx.detach else t.detach()

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Extraction (detach=True) uses the latch to ignore the CFG uncond pass.
        # Steering (detach=False) must recompute the capture on every call so that
        # gradient-checkpoint recompute during backward can rebuild the attention
        # graph — so the latch is bypassed there (its opt forward is cond-only anyway).
        if not self._ctx.active or (self._ctx.detach and self._ctx._cond_pass_complete):
            return self._fn(q, k, v, heads, mask)

        # q: [B, Lq, H*D]  k: [B, Lk, H*D]  — already post-projection, norm, RoPE
        B, Lq, HD = q.shape
        Lk = k.shape[1]
        D = HD // heads

        seq_len = self._ctx.video_seq_len
        q_cap = q[:, :seq_len, :] if seq_len is not None else q  # video tokens only
        Lq_cap = q_cap.shape[1]

        q4 = q_cap.float().view(B, Lq_cap, heads, D).transpose(1, 2)  # [B, H, Lq_cap, D]
        k4 = k.float().view(B, Lk, heads, D).transpose(1, 2)           # [B, H, Lk, D]
        attn_weights = torch.softmax(
            q4 @ k4.transpose(-2, -1) * (D**-0.5), dim=-1
        )  # [B, H, Lq_cap, Lk]

        k_bhd = k.float().view(B, Lk, heads, D)  # [B, Lk, H, D]

        # Append at most one capture per hooked block. Under steering, gradient-
        # checkpoint recompute re-invokes this hook during backward; the softmax above
        # must run (to rebuild the graph) but we must not append duplicates — the cap
        # keeps ctx.captures at exactly the hooked-block count.
        if len(self._ctx.captures) < self._ctx._n_blocks:
            self._ctx.captures.append(
                BlockCapture(
                    attn_map=self._store(attn_weights),
                    text_key=self._store(k_bhd),
                    capture_type=CaptureType.VIDEO_CA,
                )
            )
        # After all blocks have fired once, the cond pass is complete. Latch off
        # to prevent the uncond (and any further) BatchSplitAdapter chunk from
        # appending duplicate captures (extraction only — see bypass above).
        if self._ctx.detach and len(self._ctx.captures) == self._ctx._n_blocks:
            self._ctx._cond_pass_complete = True
        return self._fn(q, k, v, heads, mask)


class LTXVideoCAProbeContext:
    """Context manager that hooks into ``attn2.attention_function`` on transformer blocks.

    Each denoising step produces one BlockCapture per hooked block, accessible via
    ``self.captures`` (in ascending block index — the natural firing order).  Clear
    between steps with ``self.captures = []``.

    ``capture_blocks`` selects which transformer blocks to hook (by 0-based index);
    ``None`` hooks all of them.  Hooking only a subset both shrinks the stored
    feature and skips the per-block ``softmax(Q·Kᵀ)`` materialization for the rest.
    ``n_blocks_total`` always reflects the model's full block count so downstream
    can restore original indices.
    """

    def __init__(
        self,
        velocity_model: torch.nn.Module,
        video_seq_len: int | None = None,
        capture_blocks: list[int] | None = None,
        detach: bool = True,
    ) -> None:
        # detach=True (default) → extraction: captures are detached, no autograd graph.
        # detach=False → steering: captures stay connected to the latent so a loss on
        # the attention maps can be backpropagated to x_t.
        self.detach: bool = detach
        self._blocks = velocity_model.transformer_blocks
        self.n_blocks_total: int = len(self._blocks)
        if capture_blocks is None:
            self._hook_indices = list(range(self.n_blocks_total))
        else:
            self._hook_indices = sorted(set(int(b) for b in capture_blocks))
            bad = [b for b in self._hook_indices if not (0 <= b < self.n_blocks_total)]
            if bad:
                raise ValueError(
                    f"capture_blocks {bad} out of range for model with "
                    f"{self.n_blocks_total} transformer blocks"
                )
        self._originals: dict[int, object] = {}
        self.captures: list[BlockCapture] = []
        self.active: bool = True
        self.video_seq_len: int | None = video_seq_len
        self._n_blocks: int = len(self._hook_indices)
        self._cond_pass_complete: bool = False

    def __enter__(self) -> "LTXVideoCAProbeContext":
        self.captures = []
        for i in self._hook_indices:
            attn2 = self._blocks[i].attn2
            self._originals[i] = attn2.attention_function
            attn2.attention_function = CapturingAttentionFunction(attn2.attention_function, self)
        return self

    def __exit__(self, *_) -> None:
        for i, orig in self._originals.items():
            self._blocks[i].attn2.attention_function = orig
        self._originals = {}


def make_ltx_extraction_loop(
    step_probe_fn: Callable[[int, list[BlockCapture]], None] | None,
    desc: str = "Generating",
    capture_every: int = 1,
    first_step_to_capture: int = 0,
    capture_steps: list[int] | None = None,
    capture_blocks: list[int] | None = None,
    block_meta: dict | None = None,
    video_seq_len: int | None = None,
) -> Callable:
    """Return a denoising loop with optional per-step cross-attention capture.

    The returned callable has the same signature as ``euler_denoising_loop`` and
    can be passed as the ``loop=`` argument to ``DiffusionStage.__call__``.

    When ``step_probe_fn`` is provided, hooks are installed and
    ``step_probe_fn(step_idx, captures)`` is called after each denoiser forward
    pass.  When ``None``, runs as a plain generation loop with no overhead.

    Step gating:
      * ``capture_steps`` — explicit list of 0-based step indices; when given,
        exactly those steps are captured (overrides the stride schedule).
      * otherwise ``first_step_to_capture`` + ``capture_every`` define a regular
        stride and the final step is always captured.

    Block selection:
      * ``capture_blocks`` — 0-based transformer-block indices to hook; ``None``
        hooks all. ``block_meta`` (if given) is populated with ``n_blocks_total``
        so the caller can record original block indices.
    """
    capture_step_set = set(capture_steps) if capture_steps is not None else None

    def extraction_loop(
        sigmas: torch.Tensor,
        video_state,
        audio_state,
        stepper,
        transformer,
        denoiser,
    ):
        from ltx_pipelines.utils.helpers import post_process_latent

        velocity_model = transformer.velocity_model
        use_probe = step_probe_fn is not None
        if use_probe:
            ctx = LTXVideoCAProbeContext(
                velocity_model, video_seq_len=video_seq_len, capture_blocks=capture_blocks
            )
            if block_meta is not None:
                block_meta["n_blocks_total"] = ctx.n_blocks_total
        else:
            ctx = nullcontext()

        with ctx as probe_ctx:
            num_steps = len(sigmas) - 1
            for step_idx, _ in enumerate(tqdm(sigmas[:-1], desc=desc, unit="step")):
                if use_probe:
                    if capture_step_set is not None:
                        probe_ctx.active = step_idx in capture_step_set
                    else:
                        is_last = step_idx == num_steps - 1
                        after_first = step_idx >= first_step_to_capture
                        on_schedule = (step_idx - first_step_to_capture) % capture_every == 0
                        probe_ctx.active = is_last or (after_first and on_schedule)
                    probe_ctx.captures = []
                    probe_ctx._cond_pass_complete = False

                video_result, audio_result = denoiser(
                    transformer, video_state, audio_state, sigmas, step_idx
                )

                if use_probe and probe_ctx.active:
                    step_probe_fn(step_idx, list(probe_ctx.captures))

                if video_state is not None and video_result is not None:
                    dv = post_process_latent(
                        video_result.denoised,
                        video_state.denoise_mask,
                        video_state.clean_latent,
                    )
                    video_state = replace(
                        video_state,
                        latent=stepper.step(video_state.latent, dv, sigmas, step_idx),
                    )

                if audio_state is not None and audio_result is not None:
                    da = post_process_latent(
                        audio_result.denoised,
                        audio_state.denoise_mask,
                        audio_state.clean_latent,
                    )
                    audio_state = replace(
                        audio_state,
                        latent=stepper.step(audio_state.latent, da, sigmas, step_idx),
                    )

        return video_state, audio_state

    return extraction_loop
