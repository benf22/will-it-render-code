"""DDP entry point for probe feature extraction.

=== STANDALONE / TORCHRUN (single machine) ===

    # 1 GPU
    python data_extraction/run_extraction.py \\
        --config configs/probing.yaml \\
        --prompts_csv example_prompts/dog_prompt.csv \\
        --output_dir /tmp/probe_test

    # 4 GPUs (DDP)
    torchrun --nproc_per_node=4 data_extraction/run_extraction.py \\
        --config configs/probing.yaml \\
        --prompts_csv example_prompts/dog_prompt.csv \\
        --output_dir /tmp/probe_test

Modes:
    --mode generate   Generate videos only, no feature extraction (fast).
    --mode extract    Generate videos and extract per-step probe features (default).

=== FILESYSTEM TASK-QUEUE (multi-VM / SLURM, no server needed) ===

Step 1 — seed the queue (once, from anywhere):
    python infra/seed_queue.py \\
        --csv example_prompts/dog_prompt.csv \\
        --queue-dir /outputs/run1/queue

Step 2a — launch workers on a VM (one process per GPU):
    bash infra/launch_vm.sh 0,1,2,3 \\
        python infra/worker.py \\
            --task-module data_extraction.run_extraction \\
            --queue-dir /outputs/run1/queue \\
            --config configs/probing.yaml \\
            --output_dir /outputs/run1 \\
            --mode extract

Step 2b — OR launch a SLURM array job (one array task per GPU):
    export QUEUE_DIR=/outputs/run1/queue CONFIG=configs/probing.yaml \\
           OUTPUT_DIR=/outputs/run1 TASK_MODULE=data_extraction.run_extraction
    sbatch --array=0-15 infra/launch_slurm.sbatch

Step 3 — monitor progress (optional):
    python infra/monitor.py --queue-dir /outputs/run1/queue

Notes:
  - --output_dir and --queue-dir must be on a shared filesystem (NFS, etc.).
  - VM workers and SLURM workers can share the same queue simultaneously.
  - Tasks are claimed via os.rename() — atomic on POSIX/NFS, no race conditions.
  - Restarting a worker with the same --worker-id recovers its stale claims.
  - Workers exit after --timeout seconds (default 60) of a fully-empty queue.
  - Failed tasks live in <queue-dir>/failed/; inspect with --tail-failed.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import os
import shutil
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add parent directory to sys.path to enable imports

import numpy as np
import scipy.io.wavfile as wavfile_io
import torch
import torch.distributed as dist
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

from will_it_render.backends.model_backend import create_backend
from will_it_render.common.storage import (
    StepwiseFeatureSaver,
    StepwiseWordFeatureSaver,
    make_output_dir,
    resolve_feature_dtype,
)
from will_it_render.common.probe_types import CaptureType

# Prompt-CSV helpers hoisted to ``common.prompts`` (WILL_IT_RENDER_PLAN.md §10b R1);
# re-exported here so existing ``data_extraction.run_extraction`` callers keep working.
from will_it_render.common.prompts import load_prompts, normalize_row, parse_target_words  # noqa: F401


def gpu_tag_and_name(device: int) -> tuple[str, str]:
    """Normalized tier tag + raw name of the GPU that generated this video.

    The initial latent is GPU-independent (loaded from precomputed noise), but the
    denoising forward is GPU-tier dependent, so the FINAL video differs across tiers.
    Recording the tier per prompt lets pass-2 (feature) regen be routed to the SAME tier,
    reproducing the pass-1 video — and thus its labels — bit-for-bit.
    """
    name = torch.cuda.get_device_name(device)
    n = name.lower()
    if "b200" in n:
        tag = "b200"
    elif "h200" in n:
        tag = "h200"
    elif "h100" in n:
        tag = "h100"
    elif "a100" in n:
        tag = "a100"
    elif "rtx pro 6000" in n or ("blackwell" in n and "rtx" in n):
        tag = "rtxpro6000"
    elif "l40" in n:
        tag = "l40s"
    elif "l4" in n:
        tag = "l4"
    else:
        tag = "".join(c if c.isalnum() else "_" for c in n).strip("_")
    return tag, name


def _copy_if_missing(src: str | Path, dst: Path) -> None:
    if not dst.exists():
        shutil.copy(src, dst)


@dataclass
class ExtractionContext:
    backend: Any
    model_extraction_cfg: dict
    experiment_cfg: dict
    hook_type: CaptureType
    per_word_features: bool
    total_steps: int
    output_dir: Path
    frame_rate: float
    sample_rate: int
    device: int
    logger: logging.Logger


def add_worker_args(parser: argparse.ArgumentParser) -> None:
    """Register static CLI args on the shared worker parser (consumed by infra/worker.py)."""
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--mode",
        choices=["extract", "generate"],
        default="extract",
        help="'generate' saves video only; 'extract' also captures probe features (default)",
    )


def setup_backend(args: argparse.Namespace) -> ExtractionContext:
    """Load model and prepare output directory. Called once per worker process."""
    device = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(device)

    with open(args.config) as f:
        raw_cfg = yaml.safe_load(f)

    model_type: str = raw_cfg.get("model_type", "ltx")
    model_cfg_path = Path(args.config).parent / "models" / f"{model_type}.yaml"
    with open(model_cfg_path) as f:
        model_cfg = yaml.safe_load(f)

    engine_cfg = OmegaConf.create(model_cfg.get("engine", {}))
    model_extraction_cfg: dict = model_cfg.get("extraction", {})
    experiment_cfg: dict = raw_cfg.get("extraction", {})

    logger = logging.getLogger("probe_extraction")
    logging.basicConfig(level=logging.INFO)

    hook_type = CaptureType(model_extraction_cfg.get("hook_type", "video_ca"))
    per_word_features: bool = bool(model_extraction_cfg.get("per_word_features", False))

    backend = create_backend(model_type, engine_cfg, model_extraction_cfg, experiment_cfg, device, logger)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Existence guards prevent races when multiple workers share the same output_dir
    _copy_if_missing(args.config, output_dir / Path(args.config).name)
    _copy_if_missing(model_cfg_path, output_dir / model_cfg_path.name)
    args_yaml = output_dir / "args.yaml"
    if not args_yaml.exists():
        args_dict = {
            k: str(Path(v).resolve()) if isinstance(v, str) and Path(v).exists() else v
            for k, v in vars(args).items()
        }
        with open(args_yaml, "w") as _f:
            yaml.dump(args_dict, _f, default_flow_style=False)
    cmd_txt = output_dir / "cmd.txt"
    if not cmd_txt.exists():
        cmd_txt.write_text(" ".join(sys.argv) + "\n")

    return ExtractionContext(
        backend=backend,
        model_extraction_cfg=model_extraction_cfg,
        experiment_cfg=experiment_cfg,
        hook_type=hook_type,
        per_word_features=per_word_features,
        total_steps=int(experiment_cfg.get("sample_steps", 50)),
        output_dir=output_dir,
        frame_rate=float(model_extraction_cfg.get("frame_rate", 25.0)),
        sample_rate=int(model_extraction_cfg.get("sample_rate", 16000)),
        device=device,
        logger=logger,
    )


def process_task(task: dict, ctx: ExtractionContext, args: argparse.Namespace) -> bool:
    """Process one task (one CSV row dict). Returns True on success, False on failure."""
    worker_id = os.environ.get("WORKER_ID", "standalone")
    row = normalize_row(task)
    prompt = row["prompt"]
    object_name = row["object_name"]
    target_words = parse_target_words(row["target_words"], sep=row.get("_target_words_sep", ","))

    default_seed = ctx.experiment_cfg.get("seed", 42)
    seed_str = row.get("seed", "").strip()
    row_seed = int(seed_str) if seed_str else default_seed

    prompt_idx = int(row["prompt_idx"])
    out_dir = make_output_dir(ctx.output_dir, prompt_idx, seed=int(seed_str) if seed_str else None)
    # Per-prompt key for the GPU-noise save/load harness (model_backend): when
    # LTX_SAVE_LATENT/LTX_LOAD_LATENT point to a directory, the noise file is keyed by this
    # so one worker run handles many prompts without overwriting (no-op when the harness is off).
    os.environ["LTX_NOISE_KEY"] = out_dir.name
    # Provenance: which GPU/host produced this clip. Generation is only bit-exact within a
    # GPU architecture, so a later re-run (e.g. feature extraction of an already-generated
    # clip) can be routed to matching hardware, and mismatches are diagnosable after the fact.
    task_info = dict(task)
    task_info["_gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    task_info["_hostname"] = socket.gethostname()
    task_info["_worker_id"] = worker_id
    (out_dir / "prompt_info.json").write_text(json.dumps(task_info, indent=2))
    (out_dir / "run_config.yaml").write_text(yaml.dump({"seed": row_seed}, default_flow_style=False))

    common_kwargs = dict(
        video_negative_prompt=ctx.model_extraction_cfg.get("video_negative_prompt", ""),
        video_h=ctx.model_extraction_cfg.get("video_frame_height_width", [512, 512])[0],
        video_w=ctx.model_extraction_cfg.get("video_frame_height_width", [512, 512])[1],
        steps=ctx.total_steps,
        solver=ctx.model_extraction_cfg.get("solver_name", "unipc"),
        shift=ctx.model_extraction_cfg.get("shift", 5.0),
        video_guidance_scale=ctx.model_extraction_cfg.get("video_guidance_scale", 5.0),
        audio_guidance_scale=ctx.model_extraction_cfg.get("audio_guidance_scale", 4.0),
        slg_layer=ctx.model_extraction_cfg.get("slg_layer", 9),
        seed=row_seed,
    )

    try:
        if args.mode == "generate":
            gen_video, gen_audio = ctx.backend.run_generation_only(
                prompt=prompt, **common_kwargs
            )
        else:
            feat_dtype = resolve_feature_dtype(ctx.experiment_cfg.get("feature_dtype"))
            saver = (StepwiseWordFeatureSaver(out_dir, dtype=feat_dtype) if ctx.per_word_features
                     else StepwiseFeatureSaver(out_dir, dtype=feat_dtype))
            step_bar = tqdm(
                total=ctx.total_steps,
                desc="  saving features",
                unit="step",
                position=1,
                leave=False,
            )

            def make_callback(s, bar):
                def cb(step_idx: int, feature):
                    s.save_step(step_idx, feature)
                    bar.update(1)
                return cb

            gen_video, gen_audio, _, _, token_index_map, probe_shape = ctx.backend.run_with_probes(
                prompt=prompt,
                audio_negative_prompt=ctx.model_extraction_cfg.get("audio_negative_prompt", ""),
                target_words=target_words,
                hook_type=ctx.hook_type,
                step_callback=make_callback(saver, step_bar),
                **common_kwargs,
            )
            step_bar.close()

            feat_dir = saver.finalize(token_index_map) if ctx.per_word_features else saver.finalize()
            print(f"\n  features → {feat_dir}/")

            probe_cfg_path = ctx.output_dir / "probe_config.yaml"
            if not probe_cfg_path.exists() and probe_shape:
                lhw = probe_shape.get("latent_hw")
                Lq = probe_shape["Lq"]
                latent_num_frames = None
                if lhw is not None:
                    h_lat, w_lat = lhw
                    for ps in (1, 2, 4, 8):
                        h_p, w_p = h_lat // ps, w_lat // ps
                        if h_p > 0 and w_p > 0 and Lq % (h_p * w_p) == 0:
                            latent_num_frames = Lq // (h_p * w_p)
                            break
                probe_cfg_data = {
                    **probe_shape,
                    "latent_num_frames": latent_num_frames,
                    "feature_dim": probe_shape["n_blocks_heads"] * Lq,
                }
                with open(probe_cfg_path, "w") as _f:
                    yaml.dump(probe_cfg_data, _f, default_flow_style=False)
                print(f"  probe_config → {probe_cfg_path}")

        media_stem = out_dir.name
        ctx.backend.save_video(
            str(out_dir / f"{media_stem}.mp4"),
            gen_video, gen_audio,
            fps=ctx.frame_rate, sample_rate=ctx.sample_rate, rank=ctx.device,
        )
        # Video-only backends (e.g. WAN) return no audio track — skip the WAV write
        # rather than crashing on a None/empty waveform. Audio backends (LTX/Ovi) are
        # unaffected.
        if gen_audio is not None and getattr(gen_audio, "size", 0):
            audio_int16 = (np.clip(gen_audio, -1.0, 1.0) * 32767).astype(np.int16)
            wavfile_io.write(str(out_dir / f"{media_stem}.wav"), ctx.sample_rate, audio_int16)

        # In generate-only mode no feature saver runs, so write the metadata
        # sidecar the labelers read (words = target objects to detect).
        if args.mode == "generate":
            (out_dir / "feature_metadata.json").write_text(
                json.dumps({"words": target_words}, indent=2)
            )

        # Record the generating GPU tier so pass-2 feature regen can be routed to the
        # same tier (the seed latent fixes the initial noise, but the denoising forward is
        # GPU-tier dependent — same tier ⇒ bit-exact video ⇒ pass-1 quick-labels transfer).
        gpu_tag, gpu_name = gpu_tag_and_name(ctx.device)
        (out_dir / "gen_meta.json").write_text(json.dumps({
            "gpu_tag": gpu_tag,
            "gpu_name": gpu_name,
            "hostname": socket.gethostname(),
            "seed": row_seed,
            "mode": args.mode,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }, indent=2))

        print(f"[{worker_id}] saved → {out_dir}  (gpu={gpu_tag})")
        return True

    except Exception:
        ctx.logger.exception("[%s] failed on prompt %r — skipping", worker_id, prompt)
        return False
    finally:
        torch.cuda.empty_cache()


def run(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    os.environ.setdefault("WORKER_ID", f"rank{local_rank}")
    if world_size > 1:
        dist.init_process_group(backend="nccl")

    ctx = setup_backend(args)
    rows = load_prompts(args.prompts_csv)
    for idx, row in enumerate(rows):
        if "prompt_idx" not in row:
            row["prompt_idx"] = idx
    my_rows = rows[local_rank::world_size]

    n_ok = n_fail = 0
    for row in tqdm(my_rows, desc=f"rank {local_rank}", unit="prompt", position=0):
        if process_task(row, ctx, args):
            n_ok += 1
        else:
            n_fail += 1

    worker_id = os.environ["WORKER_ID"]
    summary = f"[{worker_id}] done — {n_ok} succeeded, {n_fail} failed (total {n_ok + n_fail})"
    print(summary)
    (ctx.output_dir / f"run_summary_{worker_id}.txt").write_text(summary + "\n")

    if world_size > 1:
        dist.destroy_process_group()


def is_task_done(task: dict, args: argparse.Namespace) -> bool:
    """Return True if this task's output already exists (used by infra/worker.py for stale-claim reclaim)."""
    seed_str = task.get("seed", "").strip()
    seed = int(seed_str) if seed_str else None
    out_dir = make_output_dir(Path(args.output_dir), int(task["prompt_idx"]), seed=seed)
    # run_config.yaml is written *before* generation, so it can exist for a crashed
    # task; the .mp4 is written only on success → use it as the completion marker.
    return (out_dir / f"{out_dir.name}.mp4").exists()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_worker_args(parser)
    parser.add_argument("--prompts_csv", required=True)
    run(parser.parse_args())
