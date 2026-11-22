"""Input-space regularizers for the attention-map probes.

Two independent transforms, both aimed at the same failure mode: a probe that keys on
*which* (block, head) channel a signal arrived in, or on a handful of extreme attention
values, rather than on the spatial/temporal pattern.

``percentile_clip`` — winsorize each per-(block, head) attention map to its own
``[lo, hi]`` percentile band. Attention is extremely long-tailed (on LTX step 4 the p99 is
~40x the median), so a few query tokens dominate the map's scale; clipping them removes
that lever without changing the map's shape elsewhere. This is *preprocessing*, not
augmentation: it is deterministic and must be applied identically to train, val, and any
downstream (steering) use of the checkpoint — hence it lives at dataset load time.

``shuffle_heads`` — randomly permute the head axis, independently per sample (and, in
``per_layer`` mode, independently per transformer block). Head ``h`` of block ``l`` carries
no canonical meaning shared across runs, so a probe that learns per-head weights is free to
memorize channel identity; re-drawing the permutation every epoch forces the aggregation and
the stem to treat heads as an exchangeable set. This *is* augmentation: train batches only.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch


def parse_clip_percentiles(val) -> Optional[tuple[float, float]]:
    """Normalize the config value to a ``(lo, hi)`` fraction pair, or ``None`` when off.

    Accepts ``[5, 95]`` (percent) or ``[0.05, 0.95]`` (fraction) — anything above 1 is read
    as a percentage. ``None`` / ``[]`` disables clipping.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        raise ValueError(f"attn_clip_percentiles must be a (lo, hi) pair, got {val!r}")
    vals = [float(v) for v in val]
    if not vals:
        return None
    if len(vals) != 2:
        raise ValueError(f"attn_clip_percentiles must have exactly 2 entries, got {vals}")
    lo, hi = vals
    if max(lo, hi) > 1.0:
        lo, hi = lo / 100.0, hi / 100.0
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError(f"attn_clip_percentiles must satisfy 0 <= lo < hi <= 1, got ({lo}, {hi})")
    return (lo, hi)


def percentile_clip(feat: torch.Tensor, lo: float, hi: float, n_channels: int) -> torch.Tensor:
    """Winsorize ``feat`` per channel to its own ``[lo, hi]`` quantile band.

    ``feat`` is the raw flat feature vector for one sample; it is viewed as
    ``[n_channels, -1]`` (one row per (block, head) attention map) and each row is clamped
    to its own quantiles, so a head whose map is uniformly small is not clipped away by a
    louder head's bounds.

    Uses ``kthvalue`` rather than ``torch.quantile``: quantile sorts and refuses inputs past
    2**24 elements, while kthvalue is a linear-time selection with no size cap (the 48-block
    LTX vector is 5.1M values today but scales with block count). The resulting bounds are
    the nearest order statistics — no interpolation — which is what "cut the tails" means
    anyway.
    """
    shape = feat.shape
    x = feat.reshape(n_channels, -1)
    n = x.shape[1]
    # Clamp k into [1, n] so a short row (or lo=0 / hi=1) degenerates to min/max, not an error.
    k_lo = min(max(int(lo * n), 1), n)
    k_hi = min(max(int(hi * n) + 1, 1), n)
    v_lo = x.kthvalue(k_lo, dim=1, keepdim=True).values
    v_hi = x.kthvalue(k_hi, dim=1, keepdim=True).values
    return torch.max(torch.min(x, v_hi), v_lo).reshape(shape)


def shuffle_heads(
    feats: torch.Tensor,
    n_blocks: int,
    n_heads: int,
    mode: str = "per_layer",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Randomly permute the head axis of a batch of stacked attention maps.

    ``feats`` is ``[B, n_blocks * n_heads, T, h, w]`` in BLOCK-MAJOR order (channel index
    ``l * n_heads + h``) — the layout ``ProbeDataset`` produces for ``avg_mode="stack"``. For
    an already layer-reduced input (``avg_mode="layers"``, or a single per-layer block) pass
    ``n_blocks=1``.

    ``mode``:
      ``per_layer`` — an independent permutation per (sample, block). Head identity is
                      destroyed outright; this is the right default, since head ``h`` of
                      block 0 and head ``h`` of block 40 are unrelated quantities.
      ``shared``    — one permutation per sample, applied to every block. Destroys head
                      identity but preserves the (meaningless) cross-block head alignment;
                      kept as the weaker ablation.
    """
    if mode == "none":
        return feats
    if mode not in ("per_layer", "shared"):
        raise ValueError(f"Unknown head_shuffle mode {mode!r}. Choose none|per_layer|shared.")
    b = feats.shape[0]
    rest = feats.shape[2:]
    if feats.shape[1] != n_blocks * n_heads:
        raise ValueError(
            f"head_shuffle: channel dim {feats.shape[1]} != n_blocks*n_heads "
            f"({n_blocks}*{n_heads})"
        )
    n_rows = b if mode == "shared" else b * n_blocks
    # argsort of uniform noise == a uniform random permutation per row.
    noise = torch.rand(n_rows, n_heads, device=feats.device, generator=generator)
    perm = noise.argsort(dim=1).view(b, -1, n_heads)
    if mode == "shared":
        perm = perm.expand(b, n_blocks, n_heads)
    # Block-major channel index: channel (l, perm[l, h]) -> l * n_heads + perm[l, h].
    offset = torch.arange(n_blocks, device=feats.device).view(1, n_blocks, 1) * n_heads
    chan = (perm + offset).reshape(b, n_blocks * n_heads)
    return feats[torch.arange(b, device=feats.device).unsqueeze(1), chan]


def head_shuffle_supported(avg_mode: str, per_layer: bool) -> bool:
    """Whether the feature layout has an addressable head axis.

    ``avg_mode="all"`` has averaged the heads away, and the NaN-scattered ``"none"`` layout
    mixes captured and un-captured blocks — neither can be permuted meaningfully.
    """
    return per_layer or avg_mode in ("stack", "layers")
