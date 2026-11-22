#!/usr/bin/env python
"""Precompute CogVideoX per-seed initial latents on a fixed GPU.

Mirrors gen_seed_latents_wan.py / gen_seed_latents.py: PyTorch's CUDA RNG maps
``(seed, shape)`` to noise in a GPU-architecture-dependent way, so the *same seed* yields a
*different* initial latent — and video — on an A100 vs H100 vs B200. We sample the initial
latent ONCE on a canonical GPU and store it; ``CogVideoXBackend._prepare`` then LOADS it
instead of sampling, so every GPU denoises from the identical initial latent.

CogVideoX ``prepare_latents`` draws
``randn([B, F', C, H/8, W/8])`` with a CUDA generator and (init_noise_sigma == 1.0) does no
scaling — so this reproduces exactly that draw (F' = (frames-1)//4 + 1, C = 16). No
DiT/T5/VAE load is needed, so it runs in seconds. We store the batch-less [F', C, H/8, W/8]
tensor in float32; the backend adds the batch dim and casts to the transformer dtype.

NOTE: only the INITIAL latent is made GPU-independent here. The denoising forward is still
GPU-tier dependent, so the FINAL video differs across A100/H100/H200/B200 — record the
generating GPU per prompt (run_extraction writes gen_meta.json) and route the pass-2
feature regen to the SAME tier to reproduce the pass-1 video (and thus its labels) exactly.

Output (one file per seed), matching ``CogVideoXBackend._seed_latent_path``::

    <out_root>/<model_tag>/<H>x<W>x<F>_<fps>fps/seed_<NNNNN>__<gpu_tag>.pt
      { "video_noise": float32 CPU tensor [F', C, H/8, W/8], "seed": int, "meta": {...} }

Usage (on an A100 for the a100 tag):
    cd <repo> && source <your-cogx-venv>/bin/activate   # set VP_COGX_ROOT
    CUDA_VISIBLE_DEVICES=<gpu> python -m data_extraction.gen_seed_latents_cogx \
        --seeds 42,111,123,456,469,696,35426,873641 --gpu-tag a100 \
        --height 480 --width 720 --frames 49 --fps 8
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

LATENT_CHANNELS = 16           # AutoencoderKLCogVideoX latent_channels
VAE_TEMPORAL_STRIDE = 4        # temporal_compression_ratio
VAE_SPATIAL_STRIDE = 8         # spatial_compression_ratio
DEFAULT_OUT_ROOT = os.environ.get("VP_SEED_LATENTS_DIR", "./seed_latents")
DEFAULT_MODEL_TAG = "cogvideox-5b"
DEFAULT_SEEDS = [42, 111, 123, 456, 469, 696, 35426, 873641]


def res_tag(h, w, frames, fps):  # MUST match CogVideoXBackend._res_tag
    return f"{h}x{w}x{frames}_{fps:g}fps"


def latent_shape(h, w, frames):
    return (
        (frames - 1) // VAE_TEMPORAL_STRIDE + 1,  # F'
        LATENT_CHANNELS,                          # C
        h // VAE_SPATIAL_STRIDE,                  # H/8
        w // VAE_SPATIAL_STRIDE,                  # W/8
    )


def latent_path(out_root, model_tag, h, w, frames, fps, seed, gpu_tag):
    return (Path(out_root) / model_tag / res_tag(h, w, frames, fps)
            / f"seed_{int(seed):05d}__{gpu_tag}.pt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--model-tag", default=DEFAULT_MODEL_TAG)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--gpu-tag", default="a100")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA not available — cannot capture canonical GPU noise.")
    shape = latent_shape(args.height, args.width, args.frames)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    device = torch.device("cuda:0")  # caller pins the physical GPU via CUDA_VISIBLE_DEVICES
    torch.cuda.init()
    meta_common = {
        "gpu_name": torch.cuda.get_device_name(0), "gpu_tag": args.gpu_tag,
        "hostname": socket.gethostname(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
        "python_version": platform.python_version(), "model_tag": args.model_tag,
        "latent_channels": LATENT_CHANNELS,
        "gen_params": {"frames": args.frames, "height": args.height, "width": args.width, "fps": args.fps},
    }
    out_dir = Path(args.out_root) / args.model_tag / res_tag(args.height, args.width, args.frames, args.fps)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for s in seeds:
        p = latent_path(args.out_root, args.model_tag, args.height, args.width, args.frames, args.fps, s, args.gpu_tag)
        if p.exists() and not args.force:
            print(f"[skip] seed={s} exists: {p}", flush=True); written.append(s); continue
        # Exactly CogVideoX prepare_latents' draw (init_noise_sigma == 1.0 → no scaling).
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
