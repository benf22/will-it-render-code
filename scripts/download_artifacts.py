"""Fetch an AttnSteer bundle (or extra seed latents) from the Hugging Face Hub.

A bundle is a directory with ``steering.yaml`` + ``checkpoints/`` + ``extraction_config.yaml``
(+ optional ``seed_latents/``) — exactly what ``wir-run --bundle <dir>`` consumes (see the
inference README). This downloads one to the local artifact cache and prints its path.

    python -m scripts.download_artifacts --repo <hf-user>/<repo> --bundle wan_dog
    wir-run --bundle "$(python -m scripts.download_artifacts --repo <hf-user>/<repo> --bundle wan_dog --quiet)" \
        --prompt "a dog on a beach"

Needs ``huggingface_hub`` (``pip install huggingface_hub``). The HF repo id is chosen at
upload time (see VERSION / release notes); there is no default.
"""
from __future__ import annotations

import argparse
import sys

from will_it_render.common import paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Download an AttnSteer bundle from Hugging Face Hub.")
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. your-user/attnsteer-bundles")
    ap.add_argument("--bundle", required=True, help="bundle subdirectory within the repo")
    ap.add_argument("--revision", default=None, help="git revision / tag (default: main)")
    ap.add_argument("--quiet", action="store_true", help="print only the resolved local path")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is required: pip install huggingface_hub")

    local = snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        allow_patterns=[f"{args.bundle}/*"],
        cache_dir=str(paths.artifact_cache_dir()),
    )
    bundle_path = f"{local.rstrip('/')}/{args.bundle}"
    if args.quiet:
        print(bundle_path)
    else:
        print(f"bundle ready → {bundle_path}")
        print(f'run it:  wir-run --bundle "{bundle_path}" --prompt "a dog on a beach"')


if __name__ == "__main__":
    main()
