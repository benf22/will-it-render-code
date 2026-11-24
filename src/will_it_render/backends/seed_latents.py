#!/usr/bin/env python
"""Precompute LTX-2 per-seed initial latents (video + audio) on a fixed GPU.

Why this exists
---------------
PyTorch's CUDA RNG (Philox) maps ``(seed, shape, dtype)`` to tensor positions in
a way that depends on the GPU architecture's SM layout, so the *same seed* yields
*different* initial noise — and therefore a different video — across different GPU
generations (mean cosine ≈ 0.553 across archs, empirically).

To make seeds canonical and GPU-independent we sample the initial latent ONCE, on
a fixed canonical GPU, and store it. The extraction/generation/
steering pipeline then *loads* these latents instead of sampling
(``data_extraction.model_backend.LTXBackend._build_pipeline_inputs`` →
``_PrecomputedNoiser``), so every GPU starts denoising from the identical latent.

Faithfulness
------------
We mirror exactly what ``ltx_pipelines/utils/blocks.py`` does when building the
denoising stage (lines ~288-302):

  * ``VideoLatentShape.from_pixel_shape`` + ``VideoLatentTools(patch_size=1)``
  * ``AudioLatentShape.from_video_pixel_shape`` + ``AudioLatentTools(patch_size=1)``
  * a SINGLE ``GaussianNoiser`` / ``torch.Generator(cuda).manual_seed(seed)`` shared
    between modalities, drawing **video first, then audio** (the generator state
    carries over from the video draw to the audio draw — this ordering matters).

Conditionings are empty and the clean latent is zeros with an all-ones denoise
mask, so the noised state equals the raw Gaussian noise. No transformer / VAE /
checkpoint load is needed, so this runs in seconds and <1 GB.

Output (one file per seed)
--------------------------
``<out_root>/<model_name>/<H>x<W>x<F>_<fps>fps/seed_<NNNNN>__<gpu_tag>.pt`` holding::

    {"video_noise": bf16 CPU tensor [1, 3328, 128],
     "audio_noise": bf16 CPU tensor [1, La, Da],
     "seed": int, "meta": {... full provenance ...}}

plus an ``index.json`` in the same directory recording the seeds present and the
generation provenance (gpu, host, torch/cuda versions, shapes).

Usage (on your canonical GPU)
-----------------------------
    cd <repo> && source <your-venv>/bin/activate   # set VP_LTX_ROOT/VP_WAN_ROOT/VP_COGX_ROOT
    CUDA_VISIBLE_DEVICES=<free_gpu> python -m data_extraction.gen_seed_latents \
        --seeds 42,101,202,303,404,505,131,242,353,464,575 --gpu-tag ctxa100
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

import torch

# --- make LTX-2 importable (shape/patchify utils only; no model load) ---
_LTX_ROOT = Path.home() / "repos" / "LTX-2"
for _pkg in ("ltx-core", "ltx-pipelines"):
    _p = str(_LTX_ROOT / "packages" / _pkg / "src")
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ltx_core.components.noisers import GaussianNoiser  # noqa: E402
from ltx_core.components.patchifiers import (  # noqa: E402
    AudioPatchifier,
    VideoLatentPatchifier,
)
from ltx_core.tools import AudioLatentTools, VideoLatentTools  # noqa: E402
from ltx_core.types import (  # noqa: E402
    AudioLatentShape,
    VideoLatentShape,
    VideoPixelShape,
)

DTYPE = torch.bfloat16  # LTX-2 stage dtype (ti2vid_one_stage.py)

DEFAULT_OUT_ROOT = os.environ.get("VP_SEED_LATENTS_DIR", "./seed_latents")
DEFAULT_MODEL_NAME = "ltx-2.3-22b-distilled-1.1"
DEFAULT_SEEDS = [42, 101, 202, 303, 404, 505, 131, 242, 353, 464, 575]


def res_tag(height: int, width: int, frames: int, fps: float) -> str:
    """Resolution sub-directory name. MUST match the loader's path construction in
    ``LTXBackend._seed_latent_path`` (data_extraction/model_backend.py)."""
    return f"{height}x{width}x{frames}_{fps:g}fps"


def latent_path(out_root: str | Path, model_name: str, height: int, width: int,
                frames: int, fps: float, seed: int, gpu_tag: str) -> Path:
    return (Path(out_root) / model_name / res_tag(height, width, frames, fps)
            / f"seed_{int(seed):05d}__{gpu_tag}.pt")


def build_tools(frames: int, height: int, width: int, fps: float):
    """Build the exact video+audio LatentTools the LTX stage uses (blocks.py)."""
    pixel = VideoPixelShape(batch=1, frames=frames, height=height, width=width, fps=fps)
    v_shape = VideoLatentShape.from_pixel_shape(pixel)
    video_tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), v_shape, fps)
    a_shape = AudioLatentShape.from_video_pixel_shape(pixel)
    audio_tools = AudioLatentTools(AudioPatchifier(patch_size=1), a_shape)
    return video_tools, audio_tools


def latents_for_seed(seed: int, video_tools, audio_tools, device: torch.device):
    """Reproduce the stage's initial (video, audio) latents for one seed.

    A single shared generator/noiser draws VIDEO then AUDIO — matching the call
    order in ``blocks.py`` (the audio draw continues the RNG stream after video).
    With empty conditionings + zero clean latent + all-ones mask, the noised state
    equals the raw Gaussian noise.
    """
    gen = torch.Generator(device=device).manual_seed(int(seed))
    noiser = GaussianNoiser(generator=gen)

    v_state = noiser(video_tools.create_initial_state(device, DTYPE, None), 1.0)
    a_state = noiser(audio_tools.create_initial_state(device, DTYPE, None), 1.0)
    return v_state.latent.detach().cpu(), a_state.latent.detach().cpu()


def _nvidia_driver() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 else None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)),
                    help="comma-separated seed list (default: the 11 canonical seeds)")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                    help="LTX checkpoint stem; must match the loader (checkpoint filename stem)")
    ap.add_argument("--frames", type=int, default=97)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--gpu-tag", default="ctxa100", help="written into each filename")
    ap.add_argument("--force", action="store_true", help="regenerate even if the file exists")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA not available on this host — cannot capture canonical GPU noise.")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    device = torch.device("cuda:0")  # caller pins the physical GPU via CUDA_VISIBLE_DEVICES
    torch.cuda.init()
    gpu_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    host = socket.gethostname()
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    video_tools, audio_tools = build_tools(args.frames, args.height, args.width, args.fps)
    out_dir = (Path(args.out_root) / args.model_name
               / res_tag(args.height, args.width, args.frames, args.fps))
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_common = {
        "gpu_name": gpu_name,
        "gpu_tag": args.gpu_tag,
        "compute_capability": f"sm_{cap[0]}{cap[1]}",
        "hostname": host,
        "cuda_visible_devices": cvd,
        "machine": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "nvidia_driver": _nvidia_driver(),
        "python_version": platform.python_version(),
        "dtype": "bfloat16",
        "model_name": args.model_name,
        "gen_params": {"frames": args.frames, "height": args.height,
                       "width": args.width, "fps": args.fps},
    }

    written = []
    v_shape = a_shape = None
    for s in seeds:
        path = latent_path(args.out_root, args.model_name, args.height, args.width,
                           args.frames, args.fps, s, args.gpu_tag)
        if path.exists() and not args.force:
            print(f"[skip] seed={s} already exists: {path}", flush=True)
            written.append(s)
            continue
        v_noise, a_noise = latents_for_seed(s, video_tools, audio_tools, device)
        v_shape, a_shape = list(v_noise.shape), list(a_noise.shape)
        meta = dict(meta_common)
        meta.update({
            "seed": int(s),
            "video_shape": v_shape,
            "audio_shape": a_shape,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
        torch.save({"video_noise": v_noise, "audio_noise": a_noise,
                    "seed": int(s), "meta": meta}, path)
        written.append(s)
        print(f"[saved] seed={s}  video={tuple(v_noise.shape)}  audio={tuple(a_noise.shape)}  "
              f"-> {path}", flush=True)

    index = {
        "seeds": sorted(set(written)),
        "video_shape": v_shape,
        "audio_shape": a_shape,
        "provenance": meta_common,
        "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nDONE  {len(written)} seed(s) in {out_dir}  gpu='{gpu_name}'", flush=True)


if __name__ == "__main__":
    main()
