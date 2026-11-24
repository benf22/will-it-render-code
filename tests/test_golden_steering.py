"""Golden end-to-end steering check — the shipped WAN bundle drives a few dataset prompts to
known per-word probabilities, so the repo self-validates WITHOUT the original dev repo.

These are recorded reference values (seed 42, the shipped WAN "Ours" bundle). Seed 42 ships a
precomputed, GPU-independent initial latent, so the only cross-GPU variance is the bf16 DiT
forward — hence a tolerance band rather than bit-equality. The test asserts both:
  * the recipe SUCCEEDS (every target word ends ≥ 0.85), and
  * each final probability is within ``TOL`` of the recorded anchor.

Marked ``@pytest.mark.gpu`` (needs a GPU + the WAN checkout + the bundle), excluded from CI.
Run it with::

    export VP_WAN_ROOT=/path/to/Wan2.1
    export VP_CONFIG_DIR=/path/to/configs        # configs/models/wan.yaml -> your WAN weights
    export WIR_TEST_WAN_BUNDLE="$(python -m scripts.download_artifacts \
        --repo benf22/will-it-render --bundle wan --quiet)"
    pytest tests/test_golden_steering.py -m gpu
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

_BUNDLE = os.environ.get("WIR_TEST_WAN_BUNDLE")
if not _BUNDLE or not os.environ.get("VP_WAN_ROOT"):
    pytest.skip("set WIR_TEST_WAN_BUNDLE (a WAN bundle dir) and VP_WAN_ROOT to run",
                allow_module_level=True)

# Recorded anchors: final (last-steered-step) per-word probability, WAN "Ours" bundle, seed 42.
GOLDEN = {
    3: {"orange": 0.990, "clock": 0.959},
    12: {"knife": 0.995, "tv": 0.992},
}
TOL = 0.05
FLOOR = 0.85


def test_golden_wan_steering(tmp_path):
    import torch

    from will_it_render.inference.steer import build_backend, load_configs, run_steering

    bundle = Path(_BUNDLE)
    raw_cfg, steering, model_cfg, model_type, model_cfg_path, model_extraction_cfg = load_configs(
        str(bundle / "steering.yaml"))
    steering.setdefault("model", {})
    steering["model"]["checkpoints_dir"] = str(bundle / "checkpoints")
    steering["model"]["extraction_config"] = str(bundle / "extraction_config.yaml")
    steering.setdefault("objective", {})["target"] = 1.0
    if (bundle / "seed_latents").is_dir():
        os.environ.setdefault("VP_SEED_LATENTS_DIR", str(bundle / "seed_latents"))

    device = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    logger = logging.getLogger("golden")
    backend = build_backend(model_type, model_cfg, model_extraction_cfg, steering, device, logger)

    csv = Path(__file__).parent / "data" / "golden_wan_prompts.csv"
    result = run_steering(steering, backend, model_extraction_cfg, raw_cfg, model_cfg,
                          model_cfg_path, tmp_path, device, logger, prompts_csv=str(csv))

    seen = {}
    for d in result.get("prompt_dirs", []):
        metrics = json.loads((Path(d) / "steering_metrics.json").read_text())
        pidx = int(Path(d).name.split("prompt_")[1].split("__")[0])
        seen[pidx] = metrics[-1]["per_word_prob_after"]

    for pidx, expected in GOLDEN.items():
        assert pidx in seen, f"no output for prompt_idx {pidx}"
        for word, ref in expected.items():
            got = seen[pidx].get(word)
            assert got is not None, f"prompt {pidx}: no prob for {word!r}"
            assert got >= FLOOR, f"prompt {pidx} {word}: {got:.3f} < floor {FLOOR} (steering failed)"
            assert abs(got - ref) <= TOL, f"prompt {pidx} {word}: {got:.3f} off anchor {ref} by >{TOL}"
