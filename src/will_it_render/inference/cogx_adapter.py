"""CogVideoX steering adapter.

Exposes the CogVideoX joint-attention DiT generation step to the shared steering core
(:func:`will_it_render.inference.steering_core.run_steering_optimization`), mirroring
:class:`steering.wan_steering_loop.WanSteeringAdapter`.

Decomposition (same contract as WAN):
  * ``grad_forward`` runs the **cond pass only**, with grad, under
    :class:`data_extraction.cogvideox_probe.CogVideoXCrossAttnProbeContext` (``detach=False``
    so each captured video→text softmax carries an autograd graph the steering loss reads).
  * ``denoise_step`` / ``estimate_x0`` run the **full CFG** (cond + uncond) under ``no_grad``.
  * Each block's feed-forward (``ff``) is gradient-checkpointed during the grad pass; the
    joint attention ``attn1`` is NOT wrapped (its captured softmax must stay resident — the
    CogVideoX analog of the WAN ``cross_attn`` carve-out).

CogVideoX-5B ships a ``CogVideoXDDIMScheduler``; latents are ``[B, F, C, H, W]`` and
``scale_model_input`` is identity for DDIM (kept in the path for correctness/symmetry).
"""

import os
from typing import Iterable, Optional

import torch

from will_it_render.inference.steering_core import SteeringBackendAdapter


def install_cogx_block_grad_checkpointing(transformer, capture_blocks=None) -> callable:
    """Gradient-checkpoint the steering ``grad_forward`` (recompute in backward).

    The full 42-block CogVideoX-5B forward with grad at Lq≈17776 is the bulk of peak VRAM.
    Two carve-outs:
      * ``ff`` is checkpointed on **every** block (its gelu activation is the single largest).
      * ``attn1`` (the joint self-attention) is checkpointed on **non-captured** blocks only —
        those blocks store nothing, so recomputing their attention in backward is free and
        exact. On **captured** blocks ``attn1`` is left resident: its softmax is stored in
        ``ctx.captures`` and read by the probe loss AFTER the forward, which a checkpointed
        region cannot safely export.
    Under ``torch.no_grad`` (CFG denoise / x0 passes) checkpoint runs the function directly,
    so this is a no-op there. Numerically identical to the un-checkpointed forward (no
    dropout). Returns a ``restore()`` that unwraps every block.
    """
    import torch.utils.checkpoint as _cp

    cap = set(int(b) for b in capture_blocks) if capture_blocks is not None else None
    originals = []

    def _wrap(orig_forward):
        def wrapped(*args, **kwargs):
            if torch.is_grad_enabled():
                return _cp.checkpoint(orig_forward, *args, use_reentrant=False, **kwargs)
            return orig_forward(*args, **kwargs)
        return wrapped

    for i, blk in enumerate(transformer.transformer_blocks):
        names = ["ff"]
        if cap is not None and i not in cap:
            names.append("attn1")  # not captured on this block → safe to recompute
        for name in names:
            sub = getattr(blk, name, None)
            if sub is None:
                continue
            originals.append((sub, sub.forward))
            sub.forward = _wrap(sub.forward)

    def restore():
        for sub, fn in originals:
            sub.forward = fn

    return restore


class CogVideoXSteeringAdapter(SteeringBackendAdapter):
    def __init__(
        self,
        transformer,
        scheduler,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        latents: torch.Tensor,            # [B, F, C, H, W]
        timesteps: torch.Tensor,
        image_rotary_emb,
        guidance_scale: float,
        use_dynamic_cfg: bool,
        capture_blocks: Optional[list],
        dtype: torch.dtype,
        steer_pool: Optional[tuple] = None,
    ) -> None:
        self.transformer = transformer
        self.steer_pool = steer_pool
        self.scheduler = scheduler
        self.prompt_embeds = prompt_embeds
        self.negative_prompt_embeds = negative_prompt_embeds
        self.latents = latents
        self.timesteps = timesteps
        self.image_rotary_emb = image_rotary_emb
        self.guidance_scale = guidance_scale
        self.use_dynamic_cfg = use_dynamic_cfg
        self.capture_blocks = capture_blocks
        self.dtype = dtype
        self._cm = None
        self.probe_ctx = None
        self._restore_ckpt = None

    # -- context: install grad-checkpointing + the detach=False capture hook --
    def __enter__(self) -> "CogVideoXSteeringAdapter":
        from will_it_render.backends.hooks.cogvideox_hooks import CogVideoXCrossAttnProbeContext

        if os.environ.get("STEER_MEM_SNAPSHOT"):
            torch.cuda.memory._record_memory_history(
                context="all", stacks="python", max_entries=200000)

        self._restore_ckpt = install_cogx_block_grad_checkpointing(
            self.transformer, capture_blocks=self.capture_blocks,
        )
        self._cm = CogVideoXCrossAttnProbeContext(
            self.transformer, capture_blocks=self.capture_blocks, detach=False,
            steer_pool=self.steer_pool,
        )
        self.probe_ctx = self._cm.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self._cm is not None:
            self._cm.__exit__(*exc)
            self._cm = None
        if self._restore_ckpt is not None:
            self._restore_ckpt()
            self._restore_ckpt = None

    def end_optimization(self) -> None:
        # Release the last grad_forward's capture buffer. After an early-exit break
        # (A&E stop_iteration / es_threshold) no backward consumed the graph, so these
        # checkpoint outputs (live grad_fn) would otherwise keep the FULL un-backwarded
        # forward graph (~49 GB) alive through the solver advance
        # into the next steered step — the difference between fitting and OOMing a
        # 94 GB card. After a normal backward this only frees ~2 GB of output tensors.
        if self.probe_ctx is not None:
            self.probe_ctx.captures = []

    # -- state accessors --
    def steps(self) -> Iterable[int]:
        return range(len(self.timesteps))

    def get_latent(self) -> torch.Tensor:
        return self.latents

    def commit_latent(self, x_t: torch.Tensor) -> None:
        self.latents = x_t

    def _cfg_scale(self, step_idx: int) -> float:
        import math

        n = len(self.timesteps)
        if self.use_dynamic_cfg:
            # Schedule on the TIMESTEP VALUE (999→0), not the loop index — matches
            # CogVideoXPipeline (`num_inference_steps - t.item()`). step_idx collapsed
            # guidance to ~1 for most of the loop; see model_backend._sample.
            t_val = float(self.timesteps[step_idx])
            return 1.0 + self.guidance_scale * (
                (1 - math.cos(math.pi * ((n - t_val) / n) ** 5.0)) / 2.0
            )
        return self.guidance_scale

    def _forward(self, latent_model_input, encoder_hidden_states, t):
        timestep = t.expand(latent_model_input.shape[0])
        return self.transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            image_rotary_emb=self.image_rotary_emb,
            return_dict=False,
        )[0].float()

    # -- cond-only, grad-enabled, captures the video→text softmax --
    def grad_forward(self, x_t: torch.Tensor, step_idx: int) -> tuple[list, torch.Tensor]:
        t = self.timesteps[step_idx]
        if os.environ.get("STEER_VRAM_DEBUG"):
            a = torch.cuda.memory_allocated() / 1e9
            r = torch.cuda.memory_reserved() / 1e9
            print(f"[vram] grad_forward step={step_idx} entry alloc={a:.1f} reserved={r:.1f} "
                  f"peak={torch.cuda.max_memory_allocated() / 1e9:.1f}", flush=True)
            snap = os.environ.get("STEER_MEM_SNAPSHOT")
            if snap and step_idx == 1:
                torch.cuda.memory._dump_snapshot(snap)
                print(f"[vram] snapshot → {snap}", flush=True)
        lmi = self.scheduler.scale_model_input(x_t, t)
        self.probe_ctx.active = True
        self.probe_ctx.captures = []
        noise_cond = self._forward(lmi, self.prompt_embeds, t)
        captures = list(self.probe_ctx.captures)
        self.probe_ctx.active = False
        if os.environ.get("STEER_VRAM_DEBUG"):
            print(f"[vram] grad_forward step={step_idx} exit alloc={torch.cuda.memory_allocated() / 1e9:.1f} "
                  f"peak={torch.cuda.max_memory_allocated() / 1e9:.1f}", flush=True)
        return captures, noise_cond

    # -- predicted clean latent x̂0 (diagnostic / KL reg); no grad, no scheduler change --
    def estimate_x0(self, x_t: torch.Tensor, step_idx: int) -> torch.Tensor:
        t = self.timesteps[step_idx]
        self.probe_ctx.active = False
        with torch.no_grad():
            lmi = self.scheduler.scale_model_input(x_t, t)
            nc = self._forward(lmi, self.prompt_embeds, t)
            nu = self._forward(lmi, self.negative_prompt_embeds, t)
            noise_pred = nu + self._cfg_scale(step_idx) * (nc - nu)
        ac = self.scheduler.alphas_cumprod.to(device=x_t.device, dtype=torch.float32)
        at = ac[t.long()].view(-1, *([1] * (x_t.dim() - 1)))
        sqrt_at, sqrt_1m = at.sqrt(), (1 - at).sqrt()
        pred_type = getattr(self.scheduler.config, "prediction_type", "v_prediction")
        xf = x_t.float()
        if pred_type == "v_prediction":
            x0 = sqrt_at * xf - sqrt_1m * noise_pred
        else:  # epsilon
            x0 = (xf - sqrt_1m * noise_pred) / sqrt_at
        return x0.detach().to(x_t.dtype)

    # -- full CFG advance of the committed latent by one scheduler step --
    def denoise_step(self, step_idx: int) -> None:
        t = self.timesteps[step_idx]
        self.probe_ctx.active = False
        with torch.no_grad():
            lmi = self.scheduler.scale_model_input(self.latents, t)
            nc = self._forward(lmi, self.prompt_embeds, t)
            nu = self._forward(lmi, self.negative_prompt_embeds, t)
            noise_pred = nu + self._cfg_scale(step_idx) * (nc - nu)
            nxt = self.scheduler.step(noise_pred, t, self.latents, return_dict=False)[0]
        self.latents = nxt.to(self.dtype)
