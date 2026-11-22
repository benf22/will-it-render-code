"""WAN adapter for the gradient-guided attention-steering loop.

The steering *algorithm* lives in ``steering/steering_core.py``
(``run_steering_optimization``) and is shared across all backends — LTX and WAN run the
*same* optimization and differ only in this thin adapter. ``WanSteeringAdapter`` plumbs
WAN's per-step forward (cond-only, grad-enabled, with cross-attention capture), its CFG
denoise + scheduler advance, and its list-wrapped latent state into the
``SteeringBackendAdapter`` contract.

Unlike LTX (whose pipeline exposes a ``loop=`` hook), WAN's ``generate()`` loop is
monolithic and under ``torch.no_grad()``, so the adapter drives the sampling itself: it
holds ``latents`` and advances them step by step.
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.cuda.amp as amp

from will_it_render.inference.steering_core import SteeringBackendAdapter


class WanSteeringAdapter(SteeringBackendAdapter):
    """WAN-specific seam over the shared steering core.

    Stateful, bound to a single generation. ``grad_forward`` runs the **cond-only**
    forward with grad (the probe was trained on cond-pass attention; CFG only blends
    cond/uncond *after* the transformer, so this is exact for the loss) and returns the
    captured cross-attention plus the velocity prediction (used by ``pred_reg``).
    ``denoise_step`` runs the full CFG denoise + scheduler advance with no grad.
    """

    #: WAN predicts velocity v, so grad_forward's pred is NOT x̂0 (pred_is_x0 stays False),
    #: but sigma(step_idx) lets the core recover x̂0 = x_t − σ_t·v for pred_reg / MOVE_BAND.
    pred_to_x0_supported = True

    def __init__(
        self,
        *,
        model,
        param_dtype,
        context,
        context_null,
        latents,
        scheduler,
        timesteps,
        seq_len,
        guide_scale,
        seed_g,
        capture_blocks: Optional[list[int]] = None,
        video_seq_len: Optional[int] = None,
    ) -> None:
        self.model = model
        self.param_dtype = param_dtype
        self.context = context
        self.context_null = context_null
        self.latents = latents  # list holding a single [C, F, H, W] latent
        self.scheduler = scheduler
        self.timesteps = timesteps
        self.seq_len = seq_len
        self.guide_scale = guide_scale
        self.seed_g = seed_g
        self.capture_blocks = capture_blocks
        self.video_seq_len = video_seq_len
        self._cm = None
        self.probe_ctx = None
        self._restore_ckpt = None

    def __enter__(self) -> "WanSteeringAdapter":
        from will_it_render.backends.hooks.wan_hooks import (
            WanCrossAttnProbeContext,
            install_block_grad_checkpointing,
        )

        # Freeze params so autograd builds a graph only to x_t (params in the path but
        # their grads are never computed/retained), and keep the attention graph
        # (detach=False) so a loss on the cross-attention backprops to the latent.
        for p in self.model.parameters():
            p.requires_grad_(False)
        # Gradient-checkpoint self_attn + ffn so the full 30-block grad forward at Lq=32760
        # fits (~177 GB un-checkpointed OOMs even a 180 GB B200). No-op under the no_grad CFG
        # denoise; cross_attn (the capture) stays resident. See install_block_grad_checkpointing.
        self._restore_ckpt = install_block_grad_checkpointing(self.model)
        self._cm = WanCrossAttnProbeContext(
            self.model, video_seq_len=self.video_seq_len,
            capture_blocks=self.capture_blocks, detach=False,
        )
        self.probe_ctx = self._cm.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self._cm is not None:
            self._cm.__exit__(*exc)
        if self._restore_ckpt is not None:
            self._restore_ckpt()
            self._restore_ckpt = None

    def steps(self) -> Iterable[int]:
        return range(len(self.timesteps))

    def get_latent(self) -> torch.Tensor:
        return self.latents[0]

    def commit_latent(self, x_t: torch.Tensor) -> None:
        self.latents = [x_t]
        self._steered = True   # consumed by denoise_step (SOLVER_RESET)

    def grad_forward(self, x_t: torch.Tensor, step_idx: int) -> tuple[list, torch.Tensor]:
        t = self.timesteps[step_idx]
        timestep = torch.stack([t])
        self.probe_ctx.active = True
        self.probe_ctx.captures = []
        with amp.autocast(dtype=self.param_dtype):
            pred = self.model([x_t], t=timestep, context=self.context, seq_len=self.seq_len)[0]
        captures = list(self.probe_ctx.captures)
        self.probe_ctx.active = False
        return captures, pred

    def estimate_x0(self, x_t: torch.Tensor, step_idx: int) -> torch.Tensor:
        """Predicted clean latent x̂0 from x_t at this step (diagnostic / decode).

        Full CFG velocity (matches denoise_step), then the flow-matching relation
        ``x̂0 = x_t − σ_t·v`` (FlowUniPC, predict_x0). No grad; no scheduler-state change.
        """
        t = self.timesteps[step_idx]
        timestep = torch.stack([t])
        self.probe_ctx.active = False
        with torch.no_grad(), amp.autocast(dtype=self.param_dtype):
            nc = self.model([x_t], t=timestep, context=self.context, seq_len=self.seq_len)[0]
            nu = self.model([x_t], t=timestep, context=self.context_null, seq_len=self.seq_len)[0]
            v = nu + self.guide_scale * (nc - nu)
        sigma = self.scheduler.sigmas[step_idx].to(device=x_t.device, dtype=x_t.dtype)
        return (x_t - sigma * v).detach()

    def sigma(self, step_idx: int) -> torch.Tensor:
        """σ_t of this step (flow-matching: x̂0 = x_t − σ_t·v); used by the MOVE_BAND x0 space."""
        return self.scheduler.sigmas[step_idx].to(device=self.latents[0].device, dtype=self.latents[0].dtype)

    def uncond_velocity(self, x_t: torch.Tensor, step_idx: int) -> torch.Tensor:
        """Null-context velocity at this step (no grad, no capture) — for a CFG look-ahead."""
        t = self.timesteps[step_idx]
        self.probe_ctx.active = False
        with torch.no_grad(), amp.autocast(dtype=self.param_dtype):
            return self.model([x_t], t=torch.stack([t]), context=self.context_null, seq_len=self.seq_len)[0].detach()

    def flow_advance(self, x_t: torch.Tensor, v: torch.Tensor, step_idx: int) -> torch.Tensor:
        """One Euler flow step to the next timestep's latent, x_{t+1} = x_t + (σ_{t+1} − σ_t)·v.

        Differentiable look-ahead for the PERSIST_REG "sawtooth" term (the real advance is the
        multistep scheduler under CFG; this is its first-order, cond-only stand-in).
        """
        s = self.scheduler.sigmas
        nxt = min(step_idx + 1, len(s) - 1)
        ds = (s[nxt] - s[step_idx]).to(device=x_t.device, dtype=x_t.dtype)
        return x_t + ds * v

    def denoise_step(self, step_idx: int) -> None:
        t = self.timesteps[step_idx]
        timestep = torch.stack([t])
        self.probe_ctx.active = False
        from steering import steering_core as _sc
        if _sc.SOLVER_RESET and getattr(self, "_steered", False):
            # A steered latent was committed for this step: skip the UniPC corrector (it would
            # rebuild the sample from the previous, un-steered latent), and under "full" also
            # restart the multistep warm-up so this advance is a first-order (Euler-like) step.
            self.scheduler.last_sample = None
            if _sc.SOLVER_RESET == "full":
                self.scheduler.lower_order_nums = 0
        self._steered = False
        with torch.no_grad(), amp.autocast(dtype=self.param_dtype):
            nc = self.model(self.latents, t=timestep, context=self.context, seq_len=self.seq_len)[0]
            nu = self.model(self.latents, t=timestep, context=self.context_null, seq_len=self.seq_len)[0]
            noise_pred = nu + self.guide_scale * (nc - nu)
            from will_it_render.inference.steering_core import project_velocity
            noise_pred = project_velocity(noise_pred, step_idx)   # no-op unless steering.velocity_proj is set
            if _sc.FLOW_DUMP_SINK is not None:
                _sc.FLOW_DUMP_SINK("denoise", step_idx, -1, noise_pred)
            x0 = self.scheduler.step(
                noise_pred.unsqueeze(0), t, self.latents[0].unsqueeze(0),
                return_dict=False, generator=self.seed_g,
            )[0]
            self.latents = [x0.squeeze(0)]
