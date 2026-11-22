"""Entry point for inference-time attention steering on an EXISTING video (SDEdit-style
video editing) — used for object-removal: VAE-encode a real video, partially re-noise
it, then run the same probe-gradient steering loop as ``steering/steer.py`` but
starting from the noised real latent instead of from scratch on pure noise.

    python -m steering.steer_video_edit --config steering/steering_video_edit.yaml
    python -m steering.steer_video_edit --config steering/steering_video_edit.yaml \
        --prompts_csv example_prompts/remove_dog.csv --output_dir /tmp/steer_edit

Set ``steering.objective.target: 0.0`` in the config to SUPPRESS the target word(s)
(object removal) — the same flip used for from-scratch object-removal steering, see
``steering/CLAUDE.md``. Set it to ``1.0`` to instead reinforce/induce an object already
faintly present.

Reuses ``load_configs`` / ``build_backend`` from ``steering.steer`` verbatim; only the
per-prompt loop and the CSV schema differ (``video_path, prompt, target_words[, seed]``
instead of a from-scratch prompts CSV).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Make the repo root importable when run as a script.

import numpy as np
import scipy.io.wavfile as wavfile_io
import torch
import yaml

from will_it_render.common.agg import build_agg_kwargs
from will_it_render.common.prompts import load_prompts, normalize_row, parse_target_words
from will_it_render.common.storage import make_output_dir
from will_it_render.inference.steer import build_backend, load_configs
from will_it_render.inference.steering_model import build_steering_model


def run_video_edit_steering(steering: dict, backend, model_extraction_cfg: dict, raw_cfg: dict,
                            model_cfg: dict, model_cfg_path: Path, output_dir: Path,
                            device: int, logger: logging.Logger, prompts_csv: str | None = None,
                            wandb_run=None) -> dict:
    """Run video-edit (SDEdit-style) steering for every row into ``output_dir``.

    Mirrors ``steering.steer.run_steering`` (same steering-model construction,
    optimization knobs, and per-prompt output layout) but reads a
    ``video_path, prompt, target_words[, seed]`` CSV and calls
    ``backend.run_with_steering_from_video`` instead of ``run_with_steering``.

    Returns ``{"n_prompts": int, "prompt_dirs": list[Path]}``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "steering.yaml").write_text(yaml.dump(raw_cfg, default_flow_style=False))
    (output_dir / model_cfg_path.name).write_text(yaml.dump(model_cfg, default_flow_style=False))

    agg_kwargs = build_agg_kwargs(model_extraction_cfg)
    steps_to_steer = steering.get("steps_to_steer", None)
    probe_config_path = steering["model"].get("probe_config")
    if probe_config_path:
        with open(probe_config_path) as f:
            probe_config = yaml.safe_load(f)
    else:
        probe_config = {}
    model_block = {**steering["model"], "objective": steering.get("objective", {})}
    steering_model = build_steering_model(
        model_block, torch.device(f"cuda:{device}"), agg_kwargs, probe_config,
        steps_filter=steps_to_steer,
    )
    logger.debug("steering model built: mode=%s checkpoints_dir=%s",
                 steering["model"].get("mode"), steering["model"].get("checkpoints_dir"))

    video_edit = steering.get("video_edit", {})
    prompts_csv = prompts_csv or video_edit["prompts_csv"]
    noise_strength = float(video_edit.get("noise_strength", 0.5))

    opt = steering.get("optimization", {})
    n_iters = int(opt.get("n_iters", 5))
    optimizer_name = str(opt.get("optimizer", "adam"))
    lr = float(opt.get("lr", 0.1))
    magnitude_reg = float(opt.get("magnitude_reg", 0.0))
    anchor_reg = float(opt.get("anchor_reg", 0.0))
    pred_reg = float(opt.get("pred_reg", 0.0))
    kl_reg = float(opt.get("kl_reg", 0.0))
    entropy_reg = float(opt.get("entropy_reg", 0.0))
    source_reg = float(opt.get("source_reg", 0.0))
    source_reg_every_step = float(opt.get("source_reg_every_step", 0.0))

    steps = int(steering.get("sample_steps", 50))
    seed_default = int(steering.get("seed", 42))
    vh, vw = model_extraction_cfg.get("video_frame_height_width", [512, 512])
    vgs = float(steering.get("guidance_scale", model_extraction_cfg.get("video_guidance_scale", 5.0)))
    ags = float(model_extraction_cfg.get("audio_guidance_scale", 4.0))
    vneg = model_extraction_cfg.get("video_negative_prompt", "")
    frame_rate = float(model_extraction_cfg.get("frame_rate", 25.0))
    sample_rate = int(model_extraction_cfg.get("sample_rate", 16000))

    rows = load_prompts(prompts_csv)
    logger.info("loaded %d row(s) from %s; noise_strength=%.3f; output → %s",
                len(rows), prompts_csv, noise_strength, output_dir)
    prompt_dirs: list[Path] = []
    for idx, raw_row in enumerate(rows):
        row = normalize_row(raw_row)
        if "prompt_idx" not in row:
            row["prompt_idx"] = idx
        video_path = row["video_path"]
        prompt = row["prompt"]
        target_words = parse_target_words(row["target_words"], sep=row.get("_target_words_sep", ","))
        seed_str = str(row.get("seed", "")).strip()
        seed = int(seed_str) if seed_str else seed_default

        out_dir = make_output_dir(output_dir, int(row["prompt_idx"]), seed=int(seed_str) if seed_str else None)

        metrics_log: list[dict] = []

        def _cb(step_idx: int, m: dict) -> None:
            metrics_log.append(m)
            pb, pa = m.get("prob_before"), m.get("prob_after")
            logger.debug("    step %3d: prob %.4f → %.4f  (target=%s)",
                         step_idx, pb, pa, steering["objective"].get("target", 1.0))

        if hasattr(steering_model, "set_prompt_context"):
            steering_model.set_prompt_context(prompt_idx=int(row["prompt_idx"]), prompt=prompt)

        print(f"[video-edit-steering] prompt_idx={row['prompt_idx']} seed={seed} "
              f"video={video_path} targets={target_words} noise_strength={noise_strength}: {prompt[:60]}")
        gen_video, gen_audio, source_trace = backend.run_with_steering_from_video(
            video_path=video_path,
            prompt=prompt,
            video_negative_prompt=vneg,
            target_words=target_words,
            steering_model=steering_model,
            noise_strength=noise_strength,
            video_h=vh,
            video_w=vw,
            steps=steps,
            n_iters=n_iters,
            optimizer_name=optimizer_name,
            lr=lr,
            video_guidance_scale=vgs,
            audio_guidance_scale=ags,
            seed=seed,
            magnitude_reg=magnitude_reg,
            anchor_reg=anchor_reg,
            pred_reg=pred_reg,
            kl_reg=kl_reg,
            entropy_reg=entropy_reg,
            source_reg=source_reg,
            source_reg_every_step=source_reg_every_step,
            step_callback=_cb,
        )

        media_stem = out_dir.name
        mp4_path = out_dir / f"{media_stem}.mp4"
        backend.save_video(
            str(mp4_path),
            gen_video, gen_audio,
            fps=frame_rate, sample_rate=sample_rate, rank=device,
        )
        audio_int16 = (np.clip(gen_audio, -1.0, 1.0) * 32767).astype(np.int16)
        wavfile_io.write(str(out_dir / f"{media_stem}.wav"), sample_rate, audio_int16)
        (out_dir / "prompt_info.json").write_text(json.dumps(row, indent=2))
        (out_dir / "steering_metrics.json").write_text(json.dumps(metrics_log, indent=2))
        (out_dir / "source_anchor_trace.json").write_text(json.dumps(source_trace, indent=2))
        print(f"[video-edit-steering] saved → {out_dir}")
        prompt_dirs.append(out_dir)

        if wandb_run is not None and metrics_log:
            last = metrics_log[-1]
            wandb_run.log({
                f"prompt_{row['prompt_idx']}/final_prob":
                    last.get("prob_after", last.get("prob")),
            })

        torch.cuda.empty_cache()

    return {"n_prompts": len(rows), "prompt_dirs": prompt_dirs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="steering/steering_video_edit.yaml")
    parser.add_argument("--prompts_csv", default=None, help="overrides steering.video_edit.prompts_csv")
    parser.add_argument("--output_dir", default=None, help="overrides steering.output_dir")
    args = parser.parse_args()

    raw_cfg, steering, model_cfg, model_type, model_cfg_path, model_extraction_cfg = load_configs(args.config)

    device = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(device)
    verbose = bool(steering.get("verbose", False))
    logger = logging.getLogger("steering_video_edit")
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    logger.debug("config=%s model_type=%s device=cuda:%d", args.config, model_type, device)
    logger.debug("steering knobs: steps_to_steer=%s optimization=%s objective=%s video_edit=%s",
                 steering.get("steps_to_steer"), steering.get("optimization"),
                 steering.get("objective"), steering.get("video_edit"))

    backend = build_backend(model_type, model_cfg, model_extraction_cfg, steering, device, logger)

    output_dir = Path(args.output_dir or steering["output_dir"]) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_video_edit_steering(steering, backend, model_extraction_cfg, raw_cfg, model_cfg,
                            model_cfg_path, output_dir, device, logger, prompts_csv=args.prompts_csv)


if __name__ == "__main__":
    main()
