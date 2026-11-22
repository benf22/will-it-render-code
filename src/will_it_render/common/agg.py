"""Aggregation-flag derivation from a model extraction config.

Hoisted out of ``data_extraction.model_backend`` (``WILL_IT_RENDER_PLAN.md`` §10b R1) so the
steering path can derive ``agg_kwargs`` without importing the full backend module.
``data_extraction.model_backend`` re-exports ``build_agg_kwargs`` (and keeps the historical
private name ``_build_agg_kwargs``) for backwards compatibility.
"""
from __future__ import annotations


def build_agg_kwargs(model_cfg: dict) -> dict:
    """Extract aggregation flags from the model extraction config."""
    frame_hw = model_cfg.get("video_frame_height_width", None)
    sf = model_cfg.get("latent_scale_factors", {})
    if frame_hw and sf:
        latent_hw = (frame_hw[0] // int(sf["height"]), frame_hw[1] // int(sf["width"]))
    else:
        # Fallback: older scalar key (used by Ovi config)
        compression = model_cfg.get("latent_spatial_compression", None)
        latent_hw = (frame_hw[0] // compression, frame_hw[1] // compression) if compression and frame_hw else None
    return dict(
        extract_keys=bool(model_cfg.get("extract_keys", False)),
        spatial_downsample=int(model_cfg.get("attn_spatial_downsample", 1)),
        latent_hw=latent_hw,
    )
