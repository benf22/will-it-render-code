"""Guards the causal-VAE per-frame mapping (training/probe_dataset.py).

``bits_to_latent_frames`` encodes the causal-VAE native→latent frame tiling.
LTX-2, WAN and CogVideoX share a tiling it handles (verified exactly at call time);
other backends compress time differently, so calling it for one of those must fail
loudly instead of silently mis-binning per-frame targets.
"""
from __future__ import annotations

import os
import sys


from will_it_render.training.feature_dataset import bits_to_latent_frames


def test_ltx_mapping_still_works():
    # 97 native → 13 latent, stride 8 (the canonical LTX 512×512×97 config).
    bits = "1" * 97
    out = bits_to_latent_frames(bits, 13, "majority", "ltx")
    assert len(out) == 13 and all(v == 1.0 for v in out)


def test_default_model_type_is_ltx():
    # Back-compat: existing callers that omit model_type get the LTX behavior.
    out = bits_to_latent_frames("1" * 97, 13)
    assert len(out) == 13


def test_supported_models_accepted():
    # ltx/wan/cogvideox share the handled causal-VAE tiling (checked exactly per call).
    for mt in ("ltx", "wan", "cogvideox"):
        out = bits_to_latent_frames("1" * 97, 13, "majority", mt)
        assert len(out) == 13


def test_unsupported_model_raises():
    for mt in ("ovi", "svd"):
        try:
            bits_to_latent_frames("1" * 97, 13, "majority", mt)
        except NotImplementedError as e:
            assert mt in str(e)
        else:
            raise AssertionError(f"expected NotImplementedError for model_type={mt!r}")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} probe-dataset VAE-guard tests passed.")


if __name__ == "__main__":
    _run_all()
