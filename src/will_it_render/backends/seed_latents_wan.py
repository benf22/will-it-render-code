#!/usr/bin/env python
"""Precompute WAN (Wan2.1) per-seed initial latents on a fixed GPU.

Mirrors ``data_extraction/gen_seed_latents.py`` (the LTX version): PyTorch's CUDA RNG
maps ``(seed, shape)`` to noise in a GPU-architecture-dependent way, so the *same seed*
yields a *different* initial latent — and video — on an A100 vs H100 vs B200. We sample
the initial latent ONCE on a canonical GPU and store it; ``WanBackend._prepare`` then
LOADS it instead of sampling, so every GPU denoises from the identical latent.

Faithfulness: WAN's ``WanT2V.generate`` draws
``torch.randn(z_dim, F', H/8, W/8, dtype=float32, device=cuda, generator=Generator(cuda).manual_seed(seed))``
before the denoising loop — this reproduces exactly that (z_dim=16 for the Wan2.1 VAE;
no DiT/T5/VAE load needed, so it runs in seconds).

Output (one file per seed), matching ``WanBackend._seed_latent_path``::

    <out_root>/wan-<task>/<H>x<W>x<F>_<fps>fps/seed_<NNNNN>__<gpu_tag>.pt
      { "video_noise": float32 CPU tensor [z_dim, F', H/8, W/8], "seed": int, "meta": {...} }

Usage (on ctxh100-01 for the h100 tag):
    cd <repo> && source <your-wan-venv>/bin/activate   # set VP_WAN_ROOT
    CUDA_VISIBLE_DEVICES=<gpu> python -m data_extraction.gen_seed_latents_wan \
        --seeds 42,111,123,456,469,696,35426,873641 --gpu-tag h100 \
        --height 480 --width 832 --frames 81 --fps 16
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import socket
import sys
from pathlib import Path

import torch

_WAN_ROOT = Path.home() / "repos" / "Wan2.1"
if str(_WAN_ROOT) not in sys.path:
    sys.path.insert(0, str(_WAN_ROOT))
from wan.configs import WAN_CONFIGS  # noqa: E402

Z_DIM = 16  # Wan2.1 WanVAE z_dim (verified); WanBackend asserts the loaded shape matches.
DEFAULT_OUT_ROOT = os.environ.get("VP_SEED_LATENTS_DIR", "./seed_latents")
# The 8 canonical seeds of the myBench_1to6 short set.
DEFAULT_SEEDS = [42, 111, 123, 456, 469, 696, 35426, 873641]


def res_tag(h, w, frames, fps):  # MUST match WanBackend._res_tag
    return f"{h}x{w}x{frames}_{fps:g}fps"


def latent_path(out_root, task, h, w, frames, fps, seed, gpu_tag):
    return (Path(out_root) / f"wan-{task}" / res_tag(h, w, frames, fps)
            / f"seed_{int(seed):05d}__{gpu_tag}.pt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--task", default="t2v-1.3B")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--fps", type=float, default=16.0)
    ap.add_argument("--gpu-tag", default="h100")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA not available — cannot capture canonical GPU noise.")
    cfg = WAN_CONFIGS[args.task]
    vst = cfg.vae_stride  # (4, 8, 8)
    shape = (Z_DIM, (args.frames - 1) // vst[0] + 1, args.height // vst[1], args.width // vst[2])

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    device = torch.device("cuda:0")  # caller pins the physical GPU via CUDA_VISIBLE_DEVICES
    torch.cuda.init()
    meta_common = {
        "gpu_name": torch.cuda.get_device_name(0), "gpu_tag": args.gpu_tag,
        "hostname": socket.gethostname(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
        "python_version": platform.python_version(), "task": args.task, "z_dim": Z_DIM,
        "gen_params": {"frames": args.frames, "height": args.height, "width": args.width, "fps": args.fps},
    }
    out_dir = Path(args.out_root) / f"wan-{args.task}" / res_tag(args.height, args.width, args.frames, args.fps)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for s in seeds:
        p = latent_path(args.out_root, args.task, args.height, args.width, args.frames, args.fps, s, args.gpu_tag)
        if p.exists() and not args.force:
            print(f"[skip] seed={s} exists: {p}", flush=True); written.append(s); continue
        # Exactly WAN's generate() draw.
        g = torch.Generator(device=device).manual_seed(int(s))
        noise = torch.randn(*shape, dtype=torch.float32, device=device, generator=g).cpu()
        meta = dict(meta_common)
        meta.update({"seed": int(s), "video_shape": list(noise.shape),
                     "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        torch.save({"video_noise": noise, "seed": int(s), "meta": meta}, p)
        written.append(s)
        print(f"[saved] seed={s} shape={tuple(noise.shape)} -> {p}", flush=True)

    (out_dir / "index.json").write_text(json.dumps(
        {"seeds": sorted(set(written)), "video_shape": list(shape), "provenance": meta_common,
         "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}, indent=2))
    print(f"\nDONE {len(written)} seed(s) in {out_dir}  gpu='{meta_common['gpu_name']}'", flush=True)


if __name__ == "__main__":
    main()
