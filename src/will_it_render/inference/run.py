"""``wir-run`` — one prompt + one AttnSteer bundle → a steered video.

The headline entry point. It is NOT a zero-config one-liner: steering needs a
trained AttnSteer, the extraction geometry it was trained on, and a matched seed latent. Those
ship together as a **bundle**:

    bundle/
      steering.yaml            # steering recipe (steps, optimizer, regularizers, model_type)
      checkpoints/             # attnsteer_step_NNN.pt (per-step checkpoints)
      extraction_config.yaml   # capture geometry (from a FULL-block extraction)
      seed_latents/            # a matched, GPU-tagged seed latent (optional; see --seed)

Fetch a bundle from Hugging Face with ``scripts/download_artifacts.py``, or pass a local
bundle directory. The committed demo bundle's seed latent reproduces exactly only on the GPU
type it was tagged for; ``--seed`` for any other value regenerates noise on the fly
(reproducible only within one GPU architecture — CUDA RNG is architecture-dependent).

Example:
    wir-run --bundle ./bundles/wan_dog --prompt "a dog on a beach" --seed 42

Running end to end needs a GPU and the backend engine checkout (set VP_LTX_ROOT /
VP_WAN_ROOT / VP_COGX_ROOT). Determinism is on by default (STEER_DETERMINISTIC=0 to opt out).
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

import torch

from will_it_render.inference.steer import build_backend, load_configs, run_steering


def _resolve_bundle(bundle: Path) -> dict:
    """Validate the bundle layout and return its key paths."""
    steering_yaml = bundle / "steering.yaml"
    checkpoints = bundle / "checkpoints"
    extraction_cfg = bundle / "extraction_config.yaml"
    missing = [p.name for p in (steering_yaml, checkpoints, extraction_cfg) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"bundle {bundle} is missing {missing}. A bundle needs steering.yaml, "
            f"checkpoints/, and extraction_config.yaml (see wir-run --help)."
        )
    return {"steering_yaml": steering_yaml, "checkpoints": checkpoints, "extraction_cfg": extraction_cfg}


def _write_prompt_csv(prompt: str, object_name: str, target_words: str, seed: int, path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt_idx", "prompt", "object_name", "target_words", "seed"])
        w.writeheader()
        w.writerow({"prompt_idx": 0, "prompt": prompt, "object_name": object_name,
                    "target_words": target_words, "seed": seed})


def main() -> None:
    ap = argparse.ArgumentParser(description="Steer a video so a target class renders.")
    ap.add_argument("--bundle", required=True, type=Path, help="AttnSteer bundle directory")
    ap.add_argument("--prompt", required=True, help="text prompt")
    ap.add_argument("--target-words", default=None,
                    help="comma-separated words to steer (default: the whole prompt's nouns, "
                         "or pass e.g. 'dog')")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--suppress", action="store_true",
                    help="suppress the target instead of inducing it (objective target 0.0)")
    ap.add_argument("--output-dir", type=Path, default=Path("./output/run"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("wir-run")

    paths = _resolve_bundle(args.bundle)
    raw_cfg, steering, model_cfg, model_type, model_cfg_path, model_extraction_cfg = load_configs(
        str(paths["steering_yaml"])
    )

    # Point the steering recipe at the bundle's checkpoints + extraction geometry.
    steering.setdefault("model", {})
    steering["model"]["checkpoints_dir"] = str(paths["checkpoints"])
    steering["model"]["extraction_config"] = str(paths["extraction_cfg"])
    # Objective: induce by default, suppress with --suppress.
    steering.setdefault("objective", {})["target"] = 0.0 if args.suppress else 1.0
    # If the bundle ships seed latents, use them (else the backend falls back to its configured dir).
    seed_dir = args.bundle / "seed_latents"
    if seed_dir.is_dir():
        os.environ.setdefault("VP_SEED_LATENTS_DIR", str(seed_dir))

    target_words = args.target_words or args.prompt
    object_name = (args.target_words or "target").split(",")[0].strip().replace(" ", "_")

    device = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # build_backend calls enable_determinism() (on by default; STEER_DETERMINISTIC=0 to opt out).
    backend = build_backend(model_type, model_cfg, model_extraction_cfg, steering, device, logger)

    output_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    with tempfile.TemporaryDirectory() as td:
        prompts_csv = Path(td) / "prompt.csv"
        _write_prompt_csv(args.prompt, object_name, target_words, args.seed, prompts_csv)
        result = run_steering(steering, backend, model_extraction_cfg, raw_cfg, model_cfg,
                              model_cfg_path, output_dir, device, logger, prompts_csv=str(prompts_csv))

    for d in result.get("prompt_dirs", []):
        logger.info("steered video → %s", d)


if __name__ == "__main__":
    main()
