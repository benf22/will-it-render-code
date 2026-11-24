"""Fetch a backbone's AttnSteer artifacts from the Hugging Face Hub and assemble a local bundle.

The Hub stores artifacts by type — ``checkpoints/<backbone>/`` (the per-step probe weights +
``extraction_config.yaml``) and ``seed_latents/<backbone>/`` (the precomputed per-seed latents).
The steering recipe is NOT on the Hub (it is the repo's ``configs/steering/<backbone>.yaml``).
This script fetches both artifact dirs and assembles the local **bundle** that ``wir-run`` consumes
(``steering.yaml`` + ``checkpoints/`` + ``extraction_config.yaml`` + ``seed_latents/``):

    bundle=$(python -m scripts.download_artifacts --backbone wan --quiet)
    wir-run --bundle "$bundle" --prompt "a kite and a baseball bat" \
        --target-words "kite,baseball bat" --seed 1892721232

Needs ``huggingface_hub`` (``pip install huggingface_hub``).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from will_it_render.common import paths

DEFAULT_REPO = "benf22/will-it-render"
BACKBONES = ("wan", "ltx", "cogvideox", "ltx_timeline")


def _link(target: Path, link: Path) -> None:
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch a backbone's AttnSteer artifacts and assemble a bundle.")
    ap.add_argument("--backbone", required=True, choices=BACKBONES, help="which backbone's probe to fetch")
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"HF repo id (default: {DEFAULT_REPO})")
    ap.add_argument("--revision", default=None, help="git revision / tag (default: main)")
    ap.add_argument("--quiet", action="store_true", help="print only the resolved bundle path")
    args = ap.parse_args()
    bb = args.backbone

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is required: pip install huggingface_hub")

    local = Path(snapshot_download(
        repo_id=args.repo, revision=args.revision,
        allow_patterns=[f"checkpoints/{bb}/*", f"seed_latents/{bb}/*"],
        cache_dir=str(paths.artifact_cache_dir()),
    ))
    ckpt_dir = local / "checkpoints" / bb
    seed_dir = local / "seed_latents" / bb
    extraction_cfg = ckpt_dir / "extraction_config.yaml"
    if not ckpt_dir.is_dir():
        sys.exit(f"no checkpoints/{bb}/ in {args.repo} (available: {BACKBONES})")

    # The steering recipe lives in the repo (not on the Hub) — copy it in as the bundle's steering.yaml.
    steering_src = paths.config_dir() / "steering" / f"{bb}.yaml"
    if not steering_src.exists():
        sys.exit(f"steering recipe not found: {steering_src}")

    # Assemble the bundle that wir-run consumes.
    bundle = paths.artifact_cache_dir() / "bundles" / bb
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    _link(ckpt_dir, bundle / "checkpoints")                    # probe_step_*.pt (+ extraction_config.yaml)
    _link(extraction_cfg, bundle / "extraction_config.yaml")
    if seed_dir.is_dir():
        _link(seed_dir, bundle / "seed_latents")
    shutil.copyfile(steering_src, bundle / "steering.yaml")

    if args.quiet:
        print(str(bundle))
    else:
        print(f"bundle ready → {bundle}")
        print(f'run it:  wir-run --bundle "{bundle}" --prompt "a kite and a baseball bat" '
              f'--target-words "kite,baseball bat" --seed 1892721232')


if __name__ == "__main__":
    main()
