"""CPU validation of the WAN capture primitives (will_it_render/backends/hooks/wan_hooks.py).

Exercises the real ``WanT2VCrossAttention`` module (random weights, no checkpoint, no
GPU) under ``WanCrossAttnProbeContext`` — the capture wrapper uses SDPA, so it runs on
CPU. Verifies:
  * one BlockCapture per hooked block, with the exact shapes the aggregator expects
    (attn_map [B, H, Lq, Lk], text_key [B, Lk, H, D]) and softmax rows summing to 1;
  * the wrapper's output equals the SDPA reference on the module's own q/k/v;
  * detach=False keeps the graph so a loss on the attention backprops to the latent;
  * sdpa_attention matches a manual softmax(QKᵀ/√d)·V, honors k_lens masking, and GQA.

Requires the WAN checkout (set VP_WAN_ROOT) + its deps (diffusers/transformers).
Marked @pytest.mark.gpu and skipped unless VP_WAN_ROOT is set — excluded from CI (R7).
"""
from __future__ import annotations

import os
import sys



import torch

import pytest

pytestmark = pytest.mark.gpu  # needs the external WAN checkout; excluded from CI (R7)

_wan_root = os.environ.get("VP_WAN_ROOT")
if not _wan_root:
    pytest.skip("VP_WAN_ROOT not set (WAN checkout required)", allow_module_level=True)
sys.path.insert(0, _wan_root)
try:
    from will_it_render.common.probe_types import CaptureType
    from will_it_render.backends.hooks.wan_hooks import (
        WanCrossAttnProbeContext,
        install_sdpa_flash_attention,
        sdpa_attention,
    )
except Exception as _e:  # pragma: no cover - needs the WAN checkout
    pytest.skip(f"WAN deps unavailable: {_e}", allow_module_level=True)
from wan.modules.model import WanT2VCrossAttention


class _Block:
    def __init__(self, cross_attn):
        self.cross_attn = cross_attn


class _Model:
    def __init__(self, blocks):
        self.blocks = blocks


def _build_model(n_blocks, dim, heads):
    torch.manual_seed(0)
    blocks = [_Block(WanT2VCrossAttention(dim, heads, (-1, -1), True, 1e-6).eval())
              for _ in range(n_blocks)]
    return _Model(blocks)


def test_capture_shapes_and_softmax():
    B, Lq, Lk, dim, heads = 1, 7, 5, 16, 4
    d = dim // heads
    model = _build_model(n_blocks=3, dim=dim, heads=heads)
    x = torch.randn(B, Lq, dim)
    context = torch.randn(B, Lk, dim)

    with WanCrossAttnProbeContext(model, detach=True) as ctx:
        for blk in model.blocks:
            out = blk.cross_attn(x, context, None)
        caps = ctx.captures

    assert len(caps) == 3
    for c in caps:
        assert c.capture_type == CaptureType.VIDEO_CA
        assert tuple(c.attn_map.shape) == (B, heads, Lq, Lk), c.attn_map.shape
        assert tuple(c.text_key.shape) == (B, Lk, heads, d), c.text_key.shape
        rows = c.attn_map.sum(dim=-1)
        assert torch.allclose(rows, torch.ones_like(rows), atol=1e-5)
    assert tuple(out.shape) == (B, Lq, dim)


def test_wrapper_output_matches_sdpa_reference():
    B, Lq, Lk, dim, heads = 1, 6, 4, 16, 4
    model = _build_model(n_blocks=1, dim=dim, heads=heads)
    ca = model.blocks[0].cross_attn
    x = torch.randn(B, Lq, dim)
    context = torch.randn(B, Lk, dim)

    with WanCrossAttnProbeContext(model, detach=True):
        out = ca(x, context, None)

    # Reference: recompute q/k/v exactly as the module does, run sdpa, project.
    b, n, dd = x.size(0), ca.num_heads, ca.head_dim
    q = ca.norm_q(ca.q(x)).view(b, -1, n, dd)
    k = ca.norm_k(ca.k(context)).view(b, -1, n, dd)
    v = ca.v(context).view(b, -1, n, dd)
    ref = ca.o(sdpa_attention(q, k, v, k_lens=None).flatten(2))
    assert torch.allclose(out, ref, atol=1e-6), (out - ref).abs().max()


def test_detach_false_backprops_to_latent():
    B, Lq, Lk, dim, heads = 1, 5, 4, 16, 4
    model = _build_model(n_blocks=2, dim=dim, heads=heads)
    x = torch.randn(B, Lq, dim, requires_grad=True)
    context = torch.randn(B, Lk, dim)

    with WanCrossAttnProbeContext(model, detach=False) as ctx:
        for blk in model.blocks:
            blk.cross_attn(x, context, None)
        # A loss on the captured attention must reach x (the "latent").
        loss = sum(c.attn_map.pow(2).sum() for c in ctx.captures)
    grad = torch.autograd.grad(loss, x)[0]
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_context_restored_on_exit():
    # Bound methods are freshly created per access, so compare by presence of an
    # instance-level 'forward' override in __dict__ instead of identity.
    model = _build_model(n_blocks=2, dim=16, heads=4)
    assert all("forward" not in blk.cross_attn.__dict__ for blk in model.blocks)
    with WanCrossAttnProbeContext(model):
        assert all("forward" in blk.cross_attn.__dict__ for blk in model.blocks)  # patched
    assert all("forward" not in blk.cross_attn.__dict__ for blk in model.blocks)  # restored


def test_sdpa_matches_manual_softmax():
    B, L, N, C = 1, 5, 4, 8
    torch.manual_seed(1)
    q = torch.randn(B, L, N, C)
    k = torch.randn(B, L, N, C)
    v = torch.randn(B, L, N, C)
    out = sdpa_attention(q, k, v)
    # Manual: [B,N,L,C]
    qt, kt, vt = (t.transpose(1, 2).float() for t in (q, k, v))
    attn = torch.softmax(qt @ kt.transpose(-2, -1) / (C ** 0.5), dim=-1)
    ref = (attn @ vt).transpose(1, 2)
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()


def test_sdpa_klens_masking():
    B, Lq, Lk, N, C = 1, 3, 6, 2, 8
    torch.manual_seed(2)
    q = torch.randn(B, Lq, N, C)
    k = torch.randn(B, Lk, N, C)
    v = torch.randn(B, Lk, N, C)
    k_lens = torch.tensor([4])  # last 2 keys masked
    out = sdpa_attention(q, k, v, k_lens=k_lens)
    # Reference: drop the masked keys entirely.
    ref = sdpa_attention(q, k[:, :4], v[:, :4], k_lens=None)
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()


def test_sdpa_gqa_expands_kv_heads():
    B, Lq, Lk, Nq, Nk, C = 1, 4, 5, 8, 2, 8
    torch.manual_seed(3)
    q = torch.randn(B, Lq, Nq, C)
    k = torch.randn(B, Lk, Nk, C)
    v = torch.randn(B, Lk, Nk, C)
    out = sdpa_attention(q, k, v)  # should expand kv heads 2 -> 8
    assert tuple(out.shape) == (B, Lq, Nq, C)


def _run_all():
    install_sdpa_flash_attention()  # so any fallback path is CPU-safe
    fns = [v for kk, v in sorted(globals().items()) if kk.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} WAN capture tests passed.")


if __name__ == "__main__":
    _run_all()
