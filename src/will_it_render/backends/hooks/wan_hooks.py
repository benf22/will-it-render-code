"""Probe extraction + steering hooks for WAN (Wan2.1) text→video cross-attention.

WAN's transformer is a stack of ``WanAttentionBlock``s, each with a ``self_attn`` and a
``cross_attn`` (``WanT2VCrossAttention``) submodule. The cross-attention computes ``q``
from the video/latent tokens and ``k, v`` from the T5 text context, then calls the
module-level ``flash_attention`` — a *fused* kernel that never materializes the attention
weight matrix. We deliberately run **without** ``flash_attn`` (simpler env + deterministic
backward for steering), so this module:

  1. ``install_sdpa_flash_attention`` — swaps WAN's module-level ``flash_attention`` for an
     ``F.scaled_dot_product_attention`` implementation. Both ``self_attn`` and non-hooked
     ``cross_attn`` call ``flash_attention`` directly, so this is required for the model to
     run at all when ``flash_attn`` is absent.
  2. ``WanCrossAttnProbeContext`` — wraps each selected block's ``cross_attn.forward`` to
     recompute ``softmax(QKᵀ/√d)`` explicitly and record a ``BlockCapture`` (mirroring
     ``data_extraction/ltx_probe.py``), while producing the real output via the SDPA path.

The capture shape matches the LTX backend exactly (``attn_map`` ``[B, H, Lq, Lk]``,
``text_key`` ``[B, Lk, H, D]``) so the generic aggregator / token-mapper / storage are
reused unchanged.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from will_it_render.common.probe_types import BlockCapture, CaptureType

_MISSING = object()


def sdpa_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """Drop-in replacement for ``wan.modules.attention.flash_attention`` using SDPA.

    Signature mirrors WAN's ``flash_attention``. Inputs are ``[B, L, N, C]`` (batch, seq,
    heads, head-dim); output is ``[B, Lq, Nq, C]`` in ``q``'s dtype. ``k_lens`` (per-batch
    valid key length) becomes a key-padding mask; ``causal`` maps to SDPA's ``is_causal``;
    ``softmax_scale=None`` uses SDPA's default ``1/√C`` (== WAN's default). ``window_size``
    (sliding-window local attention) is not supported and is asserted off — the t2v-1.3B
    config uses ``(-1, -1)`` everywhere.
    """
    assert tuple(window_size) == (-1, -1), (
        f"sdpa_attention does not implement sliding-window attention (got {window_size}). "
        "The WAN t2v-1.3B config uses window_size=(-1,-1)."
    )
    out_dtype = q.dtype
    lk, nk = k.shape[1], k.shape[2]
    nq = q.shape[2]

    if q_scale is not None:
        q = q * q_scale

    # [B, L, N, C] -> [B, N, L, C]
    qt = q.transpose(1, 2)
    kt = k.transpose(1, 2)
    vt = v.transpose(1, 2)

    # Grouped-query attention: expand kv heads to match q heads if needed.
    if nk != nq:
        assert nq % nk == 0, f"num q heads {nq} not divisible by num kv heads {nk}"
        rep = nq // nk
        kt = kt.repeat_interleave(rep, dim=1)
        vt = vt.repeat_interleave(rep, dim=1)

    attn_mask = None
    if k_lens is not None:
        # Keep keys [0, k_lens[b]); mask the padded tail. [B, 1, 1, Lk] boolean.
        idx = torch.arange(lk, device=k.device)
        keep = idx[None, :] < k_lens.to(k.device)[:, None]  # [B, Lk]
        if not bool(keep.all()):
            attn_mask = keep[:, None, None, :]

    out = F.scaled_dot_product_attention(
        qt, kt, vt,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale,
    )
    out = out.transpose(1, 2).contiguous()  # [B, Lq, Nq, C]
    return out.to(out_dtype)


def install_sdpa_flash_attention() -> callable:
    """Swap WAN's module-level ``flash_attention`` for :func:`sdpa_attention`.

    ``WanSelfAttention`` and ``WanT2VCrossAttention`` reference the name
    ``flash_attention`` imported into ``wan.modules.model``; patching that binding (and the
    source module) makes the whole DiT run on SDPA without ``flash_attn`` installed.
    Returns a callable that restores the originals.
    """
    import wan.modules.attention as _attn
    import wan.modules.model as _model

    originals = []
    for mod in (_model, _attn):
        if hasattr(mod, "flash_attention"):
            originals.append((mod, "flash_attention", mod.flash_attention))
            mod.flash_attention = sdpa_attention

    def restore():
        for mod, name, fn in originals:
            setattr(mod, name, fn)

    return restore


def install_block_grad_checkpointing(model) -> callable:
    """Gradient-checkpoint each block's ``self_attn`` and ``ffn`` (recompute in backward).

    Steering's ``grad_forward`` runs the full 30-block WAN forward with grad; retaining every
    block's self-attention and FFN activations at Lq=32760 is ~177 GB — the edge of a 180 GB
    B200, which the Attend-and-Excite baseline tips over. Wrapping ``self_attn`` and ``ffn``
    (the ``gelu`` is the largest activation) in :func:`torch.utils.checkpoint.checkpoint`
    frees those activations after the forward and recomputes them during backward, ~halving
    peak VRAM for ~+30 % wall-clock.

    ``cross_attn`` is deliberately **not** wrapped: the steering loss reads its captured
    softmax directly (see :class:`WanCrossAttnProbeContext`), so that graph must stay
    resident. Under ``torch.no_grad`` (the CFG denoise pass) checkpoint runs the function
    directly, so this is a no-op there. Numerically identical to the un-checkpointed forward
    (WAN self-attn/FFN are deterministic — no dropout), so ``per_step`` steering is unchanged.
    Returns a ``restore()`` that unwraps every block.
    """
    import torch.utils.checkpoint as _cp

    originals = []

    def _wrap(orig_forward):
        def wrapped(*args, **kwargs):
            if torch.is_grad_enabled():
                return _cp.checkpoint(orig_forward, *args, use_reentrant=False, **kwargs)
            return orig_forward(*args, **kwargs)
        return wrapped

    for blk in model.blocks:
        for name in ("self_attn", "ffn"):
            sub = getattr(blk, name, None)
            if sub is None:
                continue
            originals.append((sub, sub.forward))
            sub.forward = _wrap(sub.forward)

    def restore():
        for sub, fn in originals:
            sub.forward = fn

    return restore


class WanCrossAttnProbeContext:
    """Context manager that hooks ``cross_attn.forward`` on WAN transformer blocks.

    Each denoising step produces one ``BlockCapture`` per hooked block (in ascending
    block index — the natural firing order), accessible via ``self.captures``. Clear
    between steps with ``self.captures = []`` and toggle ``self.active`` to gate the
    cond vs. uncond forward (WAN runs them as two separate model calls, so unlike LTX no
    CFG-batch latch is needed — the driver simply deactivates capture for the uncond pass).

    ``capture_blocks`` selects which blocks to hook (0-based); ``None`` hooks all.
    ``video_seq_len`` restricts the captured query tokens to the first N (default: all —
    for t2v the whole sequence is video tokens). ``detach=True`` (extraction) detaches the
    captures; ``detach=False`` (steering) keeps them connected to the latent for backprop.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        video_seq_len: Optional[int] = None,
        capture_blocks: Optional[list[int]] = None,
        detach: bool = True,
    ) -> None:
        self.detach = detach
        self._blocks = model.blocks
        self.n_blocks_total = len(self._blocks)
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
        self.video_seq_len = video_seq_len
        self._n_blocks = len(self._hook_indices)

    def _store(self, t: torch.Tensor) -> torch.Tensor:
        return t if not self.detach else t.detach()

    def _make_forward(self, module):
        ctx = self

        def forward(x, context, context_lens):
            # Mirror WanT2VCrossAttention.forward: q from video tokens, k/v from text.
            b, n, d = x.size(0), module.num_heads, module.head_dim
            q = module.norm_q(module.q(x)).view(b, -1, n, d)        # [B, Lq, n, d]
            k = module.norm_k(module.k(context)).view(b, -1, n, d)  # [B, Lk, n, d]
            v = module.v(context).view(b, -1, n, d)                 # [B, Lk, n, d]

            # Materialize softmax(QKᵀ/√d) for the capture (fused SDPA below never exposes it).
            # Cap the append at the hooked-block count so a checkpoint recompute (steering)
            # can rebuild the graph without duplicating captures.
            if ctx.active and len(ctx.captures) < ctx._n_blocks:
                seq = ctx.video_seq_len
                q_cap = q[:, :seq, :, :] if seq is not None else q
                q4 = q_cap.float().permute(0, 2, 1, 3)  # [B, n, Lq, d]
                k4 = k.float().permute(0, 2, 1, 3)       # [B, n, Lk, d]
                attn = torch.softmax(q4 @ k4.transpose(-2, -1) * (d ** -0.5), dim=-1)
                ctx.captures.append(
                    BlockCapture(
                        attn_map=ctx._store(attn),        # [B, n, Lq, Lk]
                        text_key=ctx._store(k.float()),   # [B, Lk, n, d]
                        capture_type=CaptureType.VIDEO_CA,
                    )
                )

            # Real output via the SDPA path (deterministic; no flash_attn).
            out = sdpa_attention(q, k, v, k_lens=context_lens)
            out = out.flatten(2)
            out = module.o(out)
            return out

        return forward

    def __enter__(self) -> "WanCrossAttnProbeContext":
        self.captures = []
        for i in self._hook_indices:
            ca = self._blocks[i].cross_attn
            # Save any pre-existing instance-level ``forward`` override (normally none —
            # the class method is used), so __exit__ can restore it exactly. Setting an
            # instance attribute shadows the class method; deleting it reverts cleanly.
            self._originals[i] = ca.__dict__.get("forward", _MISSING)
            ca.forward = self._make_forward(ca)
        return self

    def __exit__(self, *_) -> None:
        for i, orig in self._originals.items():
            ca = self._blocks[i].cross_attn
            if orig is _MISSING:
                ca.__dict__.pop("forward", None)
            else:
                ca.forward = orig
        self._originals = {}
