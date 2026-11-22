from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Sequence

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from will_it_render.common.probe_types import BlockCapture


def _resolve_spatial(Lq: int, h_lat: int, w_lat: int) -> tuple[int, int, int]:
    """Return (T, h_p, w_p) such that Lq = T * h_p * w_p.

    The transformer may patchify the VAE latent spatially (e.g. LTX uses patch_size=2),
    so the actual token grid h_p = h_lat // patch_scale may be smaller than h_lat.
    We search patch_scale in {1, 2, 4, 8} for the first exact factorisation.
    """
    for ps in (1, 2, 4, 8):
        h_p, w_p = h_lat // ps, w_lat // ps
        if h_p > 0 and w_p > 0 and Lq % (h_p * w_p) == 0:
            return Lq // (h_p * w_p), h_p, w_p
    raise ValueError(
        f"Cannot factor Lq={Lq} as T*h_p*w_p with latent_hw=({h_lat},{w_lat}). "
        "Check latent_scale_factors and video_frame_height_width in the model config."
    )


def _spatial_downsample(
    attn_agg: torch.Tensor,
    h_lat: int,
    w_lat: int,
    factor: int,
) -> torch.Tensor:
    """2D average-pool the spatial dims of a flattened attention row.

    attn_agg: [H, T*h_p*w_p]  where h_p = h_lat // patch_scale (auto-detected)
    Returns:  [H, T*(h_p//factor)*(w_p//factor)]
    """
    H, Lq = attn_agg.shape
    T, h_p, w_p = _resolve_spatial(Lq, h_lat, w_lat)
    x = attn_agg.view(H, T, h_p, w_p).reshape(H * T, 1, h_p, w_p)
    x = F.avg_pool2d(x, kernel_size=factor, stride=factor)
    return x.reshape(H, T * (h_p // factor) * (w_p // factor))


def probe_shape_from_captures(
    captures: List["BlockCapture"],
    spatial_downsample: int = 1,
    latent_hw: tuple[int, int] | None = None,
    **_,
) -> dict:
    """Return feature shape info by running pooling on a dummy tensor and reading output dims.

    Uses the same _spatial_downsample call as aggregate_captures_to_vector so the
    reported dimensions exactly match what is stored in features.safetensors.
    """
    if not captures:
        return {}
    cap = captures[0]
    n_heads = cap.attn_map.shape[1]
    Lq_raw = cap.attn_map.shape[2]
    dummy = torch.zeros(n_heads, Lq_raw)
    if latent_hw is not None:
        _, h_p, w_p = _resolve_spatial(Lq_raw, latent_hw[0], latent_hw[1])
        h_eff = h_p // spatial_downsample
        w_eff = w_p // spatial_downsample
    else:
        h_eff, w_eff = None, None
    if spatial_downsample > 1 and latent_hw is not None:
        dummy = _spatial_downsample(dummy, latent_hw[0], latent_hw[1], spatial_downsample)
    Lq_eff = dummy.shape[1]  # read from actual pooling output
    n_blocks = len(captures)
    return {
        "n_blocks": n_blocks,
        "n_heads": n_heads,
        "n_blocks_heads": n_blocks * n_heads,
        "Lq": Lq_eff,
        "latent_hw": [h_eff, w_eff] if h_eff is not None else None,
    }


def aggregate_captures_to_vector(
    captures: List["BlockCapture"],
    target_indices: Sequence[int],
    *,
    extract_keys: bool = False,
    spatial_downsample: int = 1,
    latent_hw: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Convert a list of per-block BlockCaptures into a flat feature vector.

    Default (extract_keys=False, spatial_downsample=1):
        attn: [n_blocks, H, Lq] → flatten
    With extract_keys=True:
        attn + keys concatenated
    With spatial_downsample=t and latent_hw=(Hlat, Wlat):
        attn spatial dims pooled by t before flattening (video_ca only)
    """
    attn_blocks: List[torch.Tensor] = []
    key_blocks: List[torch.Tensor] = []

    for cap in captures:
        attn_map = cap.attn_map  # [1, H, Lq, Lk]
        Lk = attn_map.shape[-1]
        valid_tgt = [idx for idx in target_indices if idx < Lk]

        if valid_tgt:
            attn_agg = attn_map[0, :, :, valid_tgt].mean(dim=-1)  # [H, Lq]
        else:
            H = attn_map.shape[1]
            Lq = attn_map.shape[2]
            attn_agg = torch.zeros(H, Lq)

        if spatial_downsample > 1 and latent_hw is not None:
            attn_agg = _spatial_downsample(attn_agg, latent_hw[0], latent_hw[1], spatial_downsample)

        attn_blocks.append(attn_agg)

        if extract_keys:
            text_key = cap.text_key  # [1, Lk, H, D]
            if valid_tgt:
                key_agg = text_key[0, valid_tgt, :, :].mean(dim=0)  # [H, D]
            else:
                H = text_key.shape[2]
                D = text_key.shape[3]
                key_agg = torch.zeros(H, D)
            key_blocks.append(key_agg)

    parts = [torch.stack(attn_blocks).flatten()]
    if extract_keys:
        parts.append(torch.stack(key_blocks).flatten())
    return torch.cat(parts)


def aggregate_captures_per_word(
    captures: List["BlockCapture"],
    token_index_map: Dict[str, List[int]],
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Return {word: flat_feature_vector} using per-word token indices.

    All kwargs are forwarded to aggregate_captures_to_vector.
    """
    return {
        word: aggregate_captures_to_vector(captures, indices, **kwargs)
        for word, indices in token_index_map.items()
    }
