# Training entry point for video probing.
#
# Run with Hydra — supports dot-notation CLI overrides for any key in configs/probing.yaml:
#
#   python -m training.trainer                                      # use default config
#   python -m training.trainer training.wandb_notes="my note"       # add a W&B run note
#   python -m training.trainer training.lr=5e-4 training.num_epochs=100
#   python -m training.trainer --config-name my_experiment          # alternate yaml in configs/
#
# Multi-GPU: use the atomic filesystem queue — see training/queue_trainer.py:
#
#   python -m training.queue_trainer --mode seed ...
#   bash infra/launch_vm.sh 0,1,2,3 --task-module training.queue_trainer --queue-dir ...
#
# Each worker claims independent (step, block) tasks from the queue — no DDP needed.
#
# Hydra also writes a timestamped .hydra/ folder with the resolved config next to each run.

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import hydra  # only needed by the training CLI (main); the steering path imports
except ModuleNotFoundError:  # build_attnsteer without it (e.g. the WAN gen venv has no hydra).
    hydra = None
try:
    import wandb  # same story: only the training CLI logs to W&B; the WAN gen venv has no wandb
except ModuleNotFoundError:
    wandb = None
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

from will_it_render.training.adversarial import (
    STAT_NAMES,
    AdvConfig,
    AttentionPGD,
    evaluate_adversarial,
    save_adv_examples,
)
from will_it_render.training.augment import head_shuffle_supported, parse_clip_percentiles, shuffle_heads
from will_it_render.training.evaluation import (
    evaluate_class,
    evaluate_perframe,
    evaluate_probe,
    plot_adv_curves,
    plot_adv_diagnostics,
    plot_adv_examples,
    plot_epoch_summary,
    plot_fp_fn,
    plot_per_layer_f1_summary,
    plot_per_layer_loss_by_step,
    plot_pr_curves,
)
from will_it_render.training.mlp import ProbeMLP
from will_it_render.training.summarize_run import select_metric_for, write_run_summary
from will_it_render.training.feature_dataset import BalancedBatchSampler, ProbeDataset


# ``build_attnsteer`` and the probe-architecture classes it instantiates moved to
# ``common.probe_factory`` so the steering path can build a probe without importing this
# module's full training stack (WILL_IT_RENDER_PLAN.md §10b R1). Re-exported here for
# backwards compatibility — ``from training.trainer import build_attnsteer`` keeps working.
from will_it_render.common.attnsteer_factory import build_attnsteer  # noqa: E402,F401


def _dump_epoch_preds(train_ds, train_probs, val_ds, val_probs, path: Path) -> None:
    """Dump per-sample (split, extraction_dir, object_name, gt, prob) for train+val.

    `train_probs`/`val_probs` come from evaluate_probe on shuffle=False loaders, so they
    align positionally with each dataset's `records` list. Consumed by
    training/visualize_training.py to recreate the FP/FN distribution for this epoch.
    """
    rows = []
    for ds, probs, split in ((train_ds, train_probs, "train"), (val_ds, val_probs, "val")):
        for (extraction_dir, _step, label, obj), p in zip(ds.records, probs):
            rows.append({
                "split": split,
                "extraction_dir": extraction_dir,
                "object_name": obj,
                "gt": float(label),
                "prob": float(p),
            })
    pd.DataFrame(rows, columns=["split", "extraction_dir", "object_name", "gt", "prob"]).to_csv(
        str(path), index=False
    )


def _get_available_steps(
    labels_csv: str | Path, root_override: Optional[str | Path] = None
) -> list[int]:
    df = pd.read_csv(str(labels_csv), skip_blank_lines=True).dropna(
        subset=["root_extraction_dir", "relative_extraction_dir"]
    )
    for _, row in df.drop_duplicates(subset=["root_extraction_dir", "relative_extraction_dir"]).iterrows():
        root = root_override if root_override is not None else row["root_extraction_dir"]
        extraction_dir = Path(root) / row["relative_extraction_dir"]
        steps = sorted(int(p.stem.split("_")[1]) for p in extraction_dir.glob("step_*.safetensors"))
        if steps:
            return steps
    raise FileNotFoundError(
        f"No existing extraction directory found in {labels_csv} "
        f"(root_override={root_override})"
    )


def _load_probe_config(
    labels_csv: str | Path, root_override: Optional[str | Path] = None
) -> dict:
    """Read probe_config.yaml from the extraction root (parent of first existing extraction_dir)."""
    df = pd.read_csv(str(labels_csv), skip_blank_lines=True).dropna(
        subset=["root_extraction_dir", "relative_extraction_dir"]
    )
    for _, row in df.drop_duplicates(subset=["root_extraction_dir", "relative_extraction_dir"]).iterrows():
        root = root_override if root_override is not None else row["root_extraction_dir"]
        extraction_dir = Path(root) / row["relative_extraction_dir"]
        probe_cfg_path = extraction_dir.parent / "probe_config.yaml"
        if probe_cfg_path.exists():
            with open(probe_cfg_path) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(
        f"No probe_config.yaml found via {labels_csv} "
        f"(root_override={root_override})"
    )


def build_class_vocab(labels_csv: str | Path) -> dict[str, int]:
    """Build the class-identity vocabulary {object_name: index} from a labels CSV.

    The class target is the ``object_name`` column, which is the canonical class (dog / cat /
    …) written at labeling time from ``prompt_info.json["class_name"]``. The vocab is the sorted
    set of distinct object_names over the LABELLED rows (``label >= 0`` — the same rows
    ProbeDataset keeps), so it is deterministic: every queue worker builds the identical map
    from the same train CSV with no coordination. Classes appearing only in val (absent here)
    map to -1 in ProbeDataset and are masked out of the loss/metrics.
    """
    df = pd.read_csv(str(labels_csv), skip_blank_lines=True).dropna(subset=["object_name", "label"])
    df = df[df["label"] >= 0]
    names = sorted(str(n) for n in df["object_name"].unique())
    if not names:
        raise ValueError(f"build_class_vocab: no labelled object_names in {labels_csv}")
    return {name: i for i, name in enumerate(names)}


def _parse_avg_mode(val) -> str:
    """Normalize the avg_channels config (bool or str) to a channel-reduction mode.

    none   → no averaging         → [n_blocks_heads, T, h, w]
    all    → average all channels → [1, T, h, w]        (legacy avg_channels: true)
    layers → average over blocks  → [n_heads, T, h, w]  (keep heads separate)

    Accepts legacy bools so old configs/checkpoints keep working:
    False/None → none, True → all.
    """
    if val is None or val is False:
        return "none"
    if val is True:
        return "all"
    s = str(val).strip().lower()
    if s in ("none", "false", "0"):
        return "none"
    if s in ("all", "true", "1"):
        return "all"
    if s == "layers":
        return "layers"
    raise ValueError(f"Invalid avg_channels {val!r}; choose none | all | layers (or a bool).")


def run_label_for(step: int, block: Optional[int]) -> str:
    """Label used for filenames, W&B keys, and console output."""
    return f"step_{step:03d}_block_{block:02d}" if block is not None else f"step_{step:03d}"


def step_output_dir(run_dir: Path, step: int, block: Optional[int]) -> Path:
    """Output directory for one (step, block) task inside the run dir."""
    d = Path(run_dir) / f"step_{step:03d}"
    return d / f"block_{block:02d}" if block is not None else d


def _bernoulli_kl(logit_p: torch.Tensor, logit_q: torch.Tensor) -> torch.Tensor:
    """Mean KL( Bernoulli(sigmoid(logit_p)) ‖ Bernoulli(sigmoid(logit_q)) ).

    The TRADES consistency term for a binary head: it asks the adversarial reading to match
    the clean one, rather than asking it to be independently correct. Written with logsigmoid
    so it stays finite at saturated logits.
    """
    log_p1, log_p0 = F.logsigmoid(logit_p), F.logsigmoid(-logit_p)
    log_q1, log_q0 = F.logsigmoid(logit_q), F.logsigmoid(-logit_q)
    p1, p0 = log_p1.exp(), log_p0.exp()
    return (p1 * (log_p1 - log_q1) + p0 * (log_p0 - log_q0)).mean()


def build_training_kwargs(t) -> dict:
    """Build the train_one_step kwargs dict from a training config section.

    ``t`` is ``cfg.training`` (DictConfig or plain dict with .get). Shared by the
    hydra entry point (main) and the queue seeding (training/queue_trainer.py).
    """
    return dict(
        probe_model=str(t.get("probe_model", "conv")),
        avg_channels=t.get("avg_channels", "none"),  # none|all|layers (or legacy bool); _parse_avg_mode normalizes
        train_spatial_downsample=int(t.get("train_spatial_downsample", 1)),
        feature_cache_dir=(str(t.get("feature_cache_dir")) if t.get("feature_cache_dir") is not None else None),
        aug_w_path=(str(t.get("aug_w_path")) if t.get("aug_w_path") is not None else None),
        aug_w_prob=float(t.get("aug_w_prob", 0.5)),
        aug_w_alpha=tuple(t.get("aug_w_alpha", (0.1, 0.4))),
        aug_w_on=str(t.get("aug_w_on", "neg")),
        bottleneck_ch=int(t.get("bottleneck_ch", 128)),
        conv_kernel_size=int(t.get("conv_kernel_size", 3)),
        mlp_hidden=int(t.get("mlp_hidden", 128)),
        mlp_hidden_dims=tuple(int(d) for d in t.get("mlp_hidden_dims", [512, 128])),
        dropout=float(t.get("dropout", 0.3)),
        batch_size=int(t.get("batch_size", 64)),
        num_epochs=int(t.get("num_epochs", 20)),
        lr=float(t.get("lr", 1e-3)),
        weight_decay=float(t.get("weight_decay", 1e-4)),
        num_workers=int(t.get("num_workers", 4)),
        gpu_resident_data=bool(t.get("gpu_resident_data", True)),
        gpu_resident_fp16=bool(t.get("gpu_resident_fp16", False)),
        gpu_resident_stream_fp32=bool(t.get("gpu_resident_stream_fp32", False)),
        stream_resident_gb=(float(t.get("stream_resident_gb")) if t.get("stream_resident_gb") is not None else None),
        stream_reserve_gb=float(t.get("stream_reserve_gb", 28.0)),
        eval_every_n_epochs=int(t.get("eval_every_n_epochs", 5)),
        plot_every_n_epochs=int(t.get("plot_every_n_epochs", 10)),
        best_metric=str(t.get("best_metric", "val_loss")),
        early_stop_patience=int(t.get("early_stop_patience", 40)),
        lr_scheduler=str(t.get("lr_scheduler", "cosine")),
        cosine_restart_period=int(t.get("cosine_restart_period", 50)),
        cosine_restart_mult=int(t.get("cosine_restart_mult", 1)),
        attn_pool_conv_ch=int(t.get("attn_pool_conv_ch", 32)),
        attn_pool_mlp_hidden=int(t.get("attn_pool_mlp_hidden", 64)),
        resnet_base_ch=int(t.get("resnet_base_ch", 16)),
        resnet_n_blocks=int(t.get("resnet_n_blocks", 3)),
        resnet_groups=int(t.get("resnet_groups", 8)),
        resnet_temporal_spatial=bool(t.get("resnet_temporal_spatial", True)),
        resnet_head_dropout=float(t.get("resnet_head_dropout", 0.0)),
        resnet_dropblock_size=int(t.get("resnet_dropblock_size", 0)),
        resnet_norm=str(t.get("resnet_norm", "group")),
        resnet_layer_agg=str(t.get("resnet_layer_agg", "none")),
        resnet_layer_agg_k=int(t.get("resnet_layer_agg_k", 1)),
        resnet_layer_agg_hidden=int(t.get("resnet_layer_agg_hidden", 16)),
        resnet_input_norm=str(t.get("resnet_input_norm", "none")),
        resnet_layer_agg_init_blocks=(
            [int(b) for b in t.get("resnet_layer_agg_init_blocks")]
            if t.get("resnet_layer_agg_init_blocks") is not None else None
        ),
        resnet_layer_agg_init_random=bool(t.get("resnet_layer_agg_init_random", False)),
        layer_subset=(
            [int(b) for b in t.get("layer_subset")]
            if t.get("layer_subset") is not None else None
        ),
        resnet_layer_agg_l1=float(t.get("resnet_layer_agg_l1", 0.0)),
        resnet_layer_agg_entropy=float(t.get("resnet_layer_agg_entropy", 0.0)),
        label_smoothing=float(t.get("label_smoothing", 0.0)),
        obj_loss=str(t.get("obj_loss", "bce")),
        gce_q=float(t.get("gce_q", 0.7)),
        conf_weights_csv=(str(t.get("conf_weights_csv")) if t.get("conf_weights_csv") is not None else None),
        lipschitz_weight=float(t.get("lipschitz_weight", 0.0)),
        entropy_weight=float(t.get("entropy_weight", 0.0)),
        # Input-space regularizers (training/augment.py). Both default to off, so every
        # existing config keeps its exact behaviour.
        attn_clip_percentiles=(
            [float(v) for v in t.get("attn_clip_percentiles")]
            if t.get("attn_clip_percentiles") is not None else None
        ),
        head_shuffle=str(t.get("head_shuffle", "none")),
        resnet_step_cond=bool(t.get("resnet_step_cond", False)),
        resnet_step_embed_dim=int(t.get("resnet_step_embed_dim", 64)),
        multitask_aesthetic=bool(t.get("multitask_aesthetic", False)),
        aesthetic_csv=(str(t.get("aesthetic_csv")) if t.get("aesthetic_csv") is not None else None),
        aesthetic_col=str(t.get("aesthetic_col", "score_aesthetic_quality")),
        aesthetic_loss_weight=float(t.get("aesthetic_loss_weight", 0.5)),
        aesthetic_loss=str(t.get("aesthetic_loss", "bce_soft")),
        perframe_presence_head=bool(t.get("perframe_presence_head", False)),
        perframe_csv=(str(t.get("perframe_csv")) if t.get("perframe_csv") is not None else None),
        perframe_col=str(t.get("perframe_col", "frames_final")),
        perframe_target_mode=str(t.get("perframe_target_mode", "majority")),
        perframe_head=str(t.get("perframe_head", "shared")),
        perframe_tap=str(t.get("perframe_tap", "post_temporal")),
        perframe_loss_weight=float(t.get("perframe_loss_weight", 1.0)),
        perframe_loss_mode=str(t.get("perframe_loss_mode", "combined")),
        perframe_class_loss_weight=float(t.get("perframe_class_loss_weight", 1.0)),
        perframe_init_ckpt_dir=(str(t.get("perframe_init_ckpt_dir"))
                                if t.get("perframe_init_ckpt_dir") is not None else None),
        perframe_freeze_backbone=bool(t.get("perframe_freeze_backbone", False)),
        class_head=bool(t.get("class_head", False)),
        class_head_arch=str(t.get("class_head_arch", "pooled")),
        class_mlp_hidden=int(t.get("class_mlp_hidden", 128)),
        class_loss_weight=float(t.get("class_loss_weight", 1.0)),
        class_loss_balance=str(t.get("class_loss_balance", "none")),
        class_loss_balance_beta=float(t.get("class_loss_balance_beta", 0.999)),
        class_init_ckpt_dir=(str(t.get("class_init_ckpt_dir"))
                             if t.get("class_init_ckpt_dir") is not None else None),
        class_freeze_backbone=bool(t.get("class_freeze_backbone", False)),
        transformer_d_model=int(t.get("transformer_d_model", 256)),
        transformer_nhead=int(t.get("transformer_nhead", 4)),
        transformer_num_layers=int(t.get("transformer_num_layers", 3)),
        transformer_dim_feedforward=int(t.get("transformer_dim_feedforward", 512)),
        transformer_head_attn_nheads=int(t.get("transformer_head_attn_nheads", 4)),
        transformer_layer_attn_nheads=int(t.get("transformer_layer_attn_nheads", 4)),
        transformer_token_hidden=int(t.get("transformer_token_hidden", 0)),
        transformer_token_mode=str(t.get("transformer_token_mode", "linear")),
        transformer_token_conv_ch=int(t.get("transformer_token_conv_ch", 32)),
        transformer_mlp_hidden=int(t.get("transformer_mlp_hidden", 128)),
        transformer_pool=str(t.get("transformer_pool", "query")),
        transformer_timestep_mode=str(t.get("transformer_timestep_mode", "none")),
        transformer_num_timesteps=int(t.get("transformer_num_timesteps", 0)),
        pos_weight=float(t.get("pos_weight")) if t.get("pos_weight") is not None else None,
        balanced_batches=bool(t.get("balanced_batches", False)),
        loss_balance=str(t.get("loss_balance", "none")),
        loss_balance_beta=float(t.get("loss_balance_beta", 1.0)),
        # Adversarial (PGD-on-attention-maps) training; a plain dict so queue tasks
        # round-trip through run_config.json unchanged.
        adv=AdvConfig.from_training_cfg(t).to_dict(),
        source_tag=str(t.get("source_tag", "2026_06_22_dense_balanced")),
    )


def resolve_steps_to_run(t, train_csv: str | Path, train_root_override=None) -> Optional[list[int]]:
    """Resolve steps_indices_to_optimize / steps_values_to_optimize to step values.

    Returns None when neither is set (= train all available steps).
    """
    indices = t.get("steps_indices_to_optimize")
    values = t.get("steps_values_to_optimize")

    if indices is not None and values is not None:
        print("Error: set only one of steps_indices_to_optimize or steps_values_to_optimize")
        sys.exit(1)

    if indices is None and values is None:
        return None  # all steps

    available = _get_available_steps(train_csv, root_override=train_root_override)
    steps_to_run: list[int] = []
    if values is not None:
        for v in values:
            if int(v) not in available:
                print(f"[warning] step value {v} not in extracted steps {available}, skipping")
            else:
                steps_to_run.append(int(v))
    else:
        for i in indices:
            idx = len(available) - 1 if int(i) == -1 else int(i)
            if idx >= len(available):
                print(f"[warning] step index {i} out of range ({len(available)} steps available), skipping")
            else:
                steps_to_run.append(available[idx])
    return steps_to_run


def resolve_block_indices(
    train_labels_csv: str | Path,
    root_override,
    per_layer: bool,
    layer_indices: Optional[list[int]],
) -> list[Optional[int]]:
    """Resolve which transformer-block indices to train.

    Returns [None] when per_layer=False. Validates requested layer_indices against
    capture_blocks (subset-block extractions) and resolves negative indices against
    the original full block count.
    """
    if not per_layer:
        return [None]

    probe_cfg = _load_probe_config(train_labels_csv, root_override=root_override)
    # If extraction stored only a subset of blocks, block indices refer to the full
    # (original) layout via n_blocks_total + capture_blocks; otherwise n_blocks IS total.
    capture_blocks: Optional[list[int]] = probe_cfg.get("capture_blocks")
    n_blocks_total: int = probe_cfg.get("n_blocks_total") or probe_cfg.get("n_blocks", 0)

    if n_blocks_total == 0:
        raise ValueError("per_layer=True requires n_blocks in probe_config.yaml (run extraction with a recent backend)")

    if layer_indices:
        resolved = [i % n_blocks_total for i in layer_indices]
        if capture_blocks is not None:
            missing = [b for b in resolved if b not in capture_blocks]
            if missing:
                raise ValueError(
                    f"layer_indices request blocks {missing} that were NOT extracted "
                    f"(capture_blocks={capture_blocks}). This extraction stored only a "
                    f"subset of transformer blocks; pick layer_indices from capture_blocks."
                )
        return resolved
    if capture_blocks is not None:
        return list(capture_blocks)
    return list(range(n_blocks_total))


def write_task_result(
    step_dir: Path,
    step: int,
    block: Optional[int],
    metrics: dict,
    worker_id: str = "",
) -> None:
    """Persist one (step, block) task's summary to step_dir/results.json.

    Written atomically (tmp + rename) AFTER the checkpoint / eval CSVs /
    predictions CSV, so its existence is the durable "task done" marker used by
    the queue's is_task_done and by aggregate_run.
    """
    if metrics["pr_table"].empty:
        status, row = "skipped", None
    else:
        best = metrics["pr_table"].loc[metrics["pr_table"]["f1"].idxmax()]
        status = "ok"
        row = {
            "step": step,
            "best_threshold": float(best["threshold"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "f1": float(best["f1"]),
            "balanced_accuracy": float(best["balanced_accuracy"]),
            "auroc": float(metrics["auroc"]),
            "average_precision": float(metrics["average_precision"]),
            "baseline_f1": float(metrics["baseline_f1"]),
            "pos_base_rate": float(metrics["pos_base_rate"]),
            "best_epoch": metrics.get("best_epoch"),
            "best_val_loss": metrics.get("best_val_loss"),
            "early_stopped": metrics.get("early_stopped"),
        }
        # Per-frame headline metrics, so training_results.csv carries them alongside the
        # clip-level ones (absent → NaN for runs without the per-frame head).
        pf = metrics.get("perframe")
        if pf:
            for k in ("frame_auroc", "frame_average_precision", "frame_balanced_accuracy",
                      "frame_pos_base_rate", "allframes_auroc", "allframes_pos_base_rate",
                      "anyframe_auroc"):
                if k in pf:
                    row[k] = float(pf[k])
        # Zero-training readout (warm-started runs): the same metrics before any optimizer
        # step, so the report can show what the representation already carried.
        pf0 = metrics.get("perframe_init")
        if pf0:
            for k in ("frame_auroc", "frame_average_precision", "allframes_auroc"):
                if k in pf0:
                    row[f"init_{k}"] = float(pf0[k])
        # Class-identity head headline metrics (absent → NaN for runs without the class head).
        ch = metrics.get("classhead")
        if ch:
            for k in ("class_top1_acc", "class_top5_acc", "class_macro_acc",
                      "class_macro_precision", "class_macro_recall", "class_macro_f1",
                      "n_classes", "n_eval"):
                if k in ch:
                    row[k] = float(ch[k])
        # Robustness under the training-time threat model (adversarial runs only), so
        # training_results.csv carries the headline "is it still exploitable" numbers.
        for key, prefix in (("robust", ""), ("robust_train", "train_")):
            rb = metrics.get(key)
            if rb:
                for k in ("adv_auroc", "adv_balanced_accuracy", "adv_average_precision",
                          "flip_rate", "neg_spike_rate", "mean_score_shift"):
                    if k in rb:
                        row[f"{prefix}{k}"] = float(rb[k])
        if block is not None:
            row["block"] = block
    payload = {
        "step": step,
        "block": block,
        "status": status,
        "row": row,
        "epoch_losses": [[int(e), float(tr), float(vl)] for e, tr, vl in metrics.get("epoch_losses", [])],
        "aux_epoch_losses": [[int(e), float(ta), float(va)]
                             for e, ta, va in metrics.get("aux_epoch_losses", [])],
        "frame_epoch_losses": [[int(e), float(tf), float(vf)]
                               for e, tf, vf in metrics.get("frame_epoch_losses", [])],
        "class_epoch_losses": [[int(e), float(tc), float(vc)]
                               for e, tc, vc in metrics.get("class_epoch_losses", [])],
        # Explicit per-clip and combined loss curves (train, val) — no derivation needed downstream.
        "clip_epoch_losses": [[int(e), float(tc), float(vc)]
                              for e, tc, vc in metrics.get("clip_epoch_losses", [])],
        "combined_epoch_losses": [[int(e), float(tk), float(vk)]
                                  for e, tk, vk in metrics.get("combined_epoch_losses", [])],
        "perframe": metrics.get("perframe"),
        "perframe_init": metrics.get("perframe_init"),
        "classhead": metrics.get("classhead"),
        "robust": metrics.get("robust"),
        "robust_train": metrics.get("robust_train"),
        "best_epoch": metrics.get("best_epoch"),
        "best_val_loss": metrics.get("best_val_loss"),
        "monitor_metric": metrics.get("monitor_metric"),
        "monitor_value": metrics.get("monitor_value"),
        "early_stopped": metrics.get("early_stopped"),
        "epochs_trained": metrics.get("epochs_trained"),
        "worker_id": worker_id,
        "finished_at": time.time(),
    }
    out = Path(step_dir) / "results.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.rename(tmp, out)


def aggregate_run(run_dir: str | Path, wandb_run=None, metric: str = "auroc") -> pd.DataFrame:
    """Rebuild run-level summaries from per-task results.json files on disk.

    Fully disk-driven and idempotent: writes training_results.csv and summary
    plots into run_dir, overwriting any previous aggregate. Called by the
    single-process trainer at the end of a run, by the last queue worker, and by
    `python -m training.queue_trainer --mode aggregate`.

    `metric` selects the best epoch used for `training_results.csv`, the
    prompt-length-segmented table, and `training_summary.png` (see summarize_run).
    """
    run_dir = Path(run_dir)
    result_files = sorted(run_dir.glob("step_*/results.json")) + sorted(run_dir.glob("step_*/block_*/results.json"))
    if not result_files:
        print(f"[aggregate] no results.json found under {run_dir} — nothing to aggregate")
        return pd.DataFrame()

    tasks = [json.loads(p.read_text()) for p in result_files]
    per_layer = any(t["block"] is not None for t in tasks)

    n_skipped = sum(1 for t in tasks if t["status"] != "ok")
    if n_skipped:
        print(f"[aggregate] {n_skipped} task(s) had status=skipped (one-class train set or empty val)")

    # Merge per-task prediction CSVs (written unchanged by train_one_step).
    all_predictions: list[pd.DataFrame] = []
    for t in tasks:
        step, block = t["step"], t["block"]
        preds_path = step_output_dir(run_dir, step, block) / f"predictions_{run_label_for(step, block)}.csv"
        if preds_path.exists():
            preds = pd.read_csv(preds_path)
            preds.insert(0, "denoising_step", step)
            if per_layer:
                preds.insert(1, "block_index", block)
            all_predictions.append(preds)

    if per_layer:
        # Per-step loss-by-block plots from the persisted epoch losses.
        steps = sorted({t["step"] for t in tasks})
        for step in steps:
            blocks_for_step = {
                t["block"]: [tuple(e) for e in t["epoch_losses"]]
                for t in tasks if t["step"] == step and t["block"] is not None and t["epoch_losses"]
            }
            if blocks_for_step:
                loss_plot = run_dir / f"step_{step:03d}" / f"loss_by_block_step_{step:03d}.png"
                plot_per_layer_loss_by_step(step, blocks_for_step, loss_plot)
                if loss_plot.exists() and wandb_run is not None:
                    wandb_run.log({f"step_{step:03d}/loss_by_block": wandb.Image(str(loss_plot))})

    # Single summary authority: best-epoch table, prompt-length-segmented table, and
    # the two-subplot training_summary.png — all derived from the per-epoch dumps.
    results_df = write_run_summary(run_dir, metric=metric, wandb_run=wandb_run)

    end_log: dict = {}
    if all_predictions and wandb_run is not None:
        end_log["predictions"] = wandb.Table(dataframe=pd.concat(all_predictions, ignore_index=True))
    if per_layer and not results_df.empty and {"step", "block", "f1"}.issubset(results_df.columns):
        f1_plot = run_dir / "per_layer_f1_summary.png"
        plot_per_layer_f1_summary(results_df, f1_plot)
        if f1_plot.exists() and wandb_run is not None:
            end_log["per_layer_f1_summary"] = wandb.Image(str(f1_plot))
    if wandb_run is not None and end_log:
        wandb_run.log(end_log)
    print(f"[aggregate] {len(results_df)} task results → {run_dir / 'training_results.csv'}")
    return results_df


def train_one_step(
    step: int,
    train_labels_csv: str | Path,
    val_labels_csv: str | Path,
    output_dir: str | Path,
    device: torch.device = torch.device("cuda"),
    probe_model: str = "conv",
    avg_channels: bool = False,
    train_spatial_downsample: int = 1,
    feature_cache_dir: Optional[str] = None,
    aug_w_path: Optional[str] = None,
    aug_w_prob: float = 0.5,
    aug_w_alpha: tuple[float, float] = (0.1, 0.4),
    aug_w_on: str = "neg",
    bottleneck_ch: int = 128,
    conv_kernel_size: int = 3,
    mlp_hidden: int = 128,
    mlp_hidden_dims: tuple[int, ...] = (512, 128),
    dropout: float = 0.3,
    batch_size: int = 64,
    num_epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    gpu_resident_data: bool = True,
    gpu_resident_fp16: bool = False,
    gpu_resident_stream_fp32: bool = False,
    stream_resident_gb: Optional[float] = None,
    stream_reserve_gb: float = 28.0,
    eval_every_n_epochs: int = 5,
    plot_every_n_epochs: int = 10,
    best_metric: str = "val_loss",
    early_stop_patience: int = 40,
    lr_scheduler: str = "cosine",
    cosine_restart_period: int = 50,
    cosine_restart_mult: int = 1,
    attn_pool_conv_ch: int = 32,
    attn_pool_mlp_hidden: int = 64,
    resnet_base_ch: int = 16,
    resnet_n_blocks: int = 3,
    resnet_groups: int = 8,
    resnet_temporal_spatial: bool = True,
    resnet_head_dropout: float = 0.0,
    resnet_dropblock_size: int = 0,
    resnet_norm: str = "group",
    resnet_layer_agg: str = "none",
    resnet_layer_agg_k: int = 1,
    resnet_layer_agg_hidden: int = 16,
    resnet_input_norm: str = "none",
    resnet_layer_agg_init_blocks: Optional[list[int]] = None,
    resnet_layer_agg_init_random: bool = False,
    layer_subset: Optional[list[int]] = None,
    resnet_layer_agg_l1: float = 0.0,
    resnet_layer_agg_entropy: float = 0.0,
    label_smoothing: float = 0.0,
    obj_loss: str = "bce",
    gce_q: float = 0.7,
    conf_weights_csv: Optional[str] = None,
    lipschitz_weight: float = 0.0,
    entropy_weight: float = 0.0,
    attn_clip_percentiles: Optional[Sequence[float]] = None,
    head_shuffle: str = "none",
    resnet_step_cond: bool = False,
    resnet_step_embed_dim: int = 64,
    multitask_aesthetic: bool = False,
    aesthetic_csv: Optional[str] = None,
    aesthetic_col: str = "score_aesthetic_quality",
    aesthetic_loss_weight: float = 0.5,
    aesthetic_loss: str = "bce_soft",
    perframe_presence_head: bool = False,
    perframe_csv: Optional[str] = None,
    perframe_col: str = "frames_final",
    perframe_target_mode: str = "majority",
    perframe_head: str = "shared",
    perframe_tap: str = "post_temporal",
    perframe_loss_weight: float = 1.0,
    perframe_loss_mode: str = "combined",   # combined | clip_only | frame_only | alternate
    perframe_class_loss_weight: float = 1.0,
    perframe_init_ckpt_dir: Optional[str] = None,
    perframe_freeze_backbone: bool = False,
    class_head: bool = False,
    class_head_arch: str = "pooled",
    class_mlp_hidden: int = 128,
    class_loss_weight: float = 1.0,
    class_loss_balance: str = "none",
    class_loss_balance_beta: float = 0.999,
    class_init_ckpt_dir: Optional[str] = None,
    class_freeze_backbone: bool = False,
    transformer_d_model: int = 256,
    transformer_nhead: int = 4,
    transformer_num_layers: int = 3,
    transformer_dim_feedforward: int = 512,
    transformer_head_attn_nheads: int = 4,
    transformer_layer_attn_nheads: int = 4,
    transformer_token_hidden: int = 0,
    transformer_token_mode: str = "linear",
    transformer_token_conv_ch: int = 32,
    transformer_mlp_hidden: int = 128,
    transformer_pool: str = "query",
    transformer_timestep_mode: str = "none",
    transformer_num_timesteps: int = 0,
    pos_weight: Optional[float] = None,
    balanced_batches: bool = False,
    loss_balance: str = "none",
    loss_balance_beta: float = 1.0,
    adv: Optional[dict | AdvConfig] = None,
    source_tag: str = "2026_06_22_dense_balanced",
    wandb_run=None,
    block_index: Optional[int] = None,
    train_root_override: Optional[str | Path] = None,
    val_root_override: Optional[str | Path] = None,
    preloaded_datasets: Optional[tuple] = None,
    return_datasets: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_cfg = _load_probe_config(train_labels_csv, root_override=train_root_override)
    n_blocks_heads: int = probe_cfg["n_blocks_heads"]
    n_blocks: int = probe_cfg.get("n_blocks", 0)
    n_heads: int = probe_cfg.get("n_heads", 0)
    Lq: int = probe_cfg["Lq"]
    lhw = probe_cfg["latent_hw"]
    h_lat, w_lat = lhw
    # latent_hw stores raw VAE spatial dims; the transformer patchifies spatially (2x for LTX).
    # Find the patch scale so Lq = T * (H/ps) * (W/ps) divides evenly.
    latent_thw: tuple[int, int, int] | None = None
    for ps in (1, 2, 4, 8):
        h_p, w_p = h_lat // ps, w_lat // ps
        if h_p > 0 and w_p > 0 and Lq % (h_p * w_p) == 0:
            latent_thw = (Lq // (h_p * w_p), h_p, w_p)
            break
    if latent_thw is None:
        raise ValueError(f"Cannot factor Lq={Lq} into T*(H/ps)*(W/ps) for latent_hw={lhw}")

    per_layer = block_index is not None
    # label used for filenames, W&B keys, and console output
    run_label = run_label_for(step, block_index)

    # attn_pool / per_layer use raw per-block maps — no channel averaging.
    avg_mode = _parse_avg_mode(avg_channels)
    effective_avg_mode = "none" if (per_layer or probe_model == "attn_pool") else avg_mode
    # Learned layer aggregation: the ResNet probe consumes the captured blocks STACKED
    # (avg_mode="stack") and learns the layer combine. It therefore conflicts with any
    # path that already reduces or slices the block axis — guard loudly rather than
    # silently mis-shaping the input.
    if resnet_layer_agg != "none":
        if probe_model != "resnet":
            raise ValueError(
                f"resnet_layer_agg={resnet_layer_agg!r} requires probe_model='resnet' "
                f"(got {probe_model!r})."
            )
        if per_layer:
            raise ValueError(
                "resnet_layer_agg is incompatible with per_layer=True (a single block is "
                "already sliced); set per_layer=False to let the probe aggregate layers."
            )
        if avg_mode != "none":
            raise ValueError(
                f"resnet_layer_agg requires avg_channels=none (got {avg_mode!r}); the "
                "aggregator needs the un-reduced stacked blocks."
            )
        effective_avg_mode = "stack"
    # The transformer probe consumes the same STACKED blocks as resnet_layer_agg (it learns
    # the layer/head aggregation via attention), so force avg_mode="stack" and guard the same
    # conflicts (a reduced/sliced block axis leaves the aggregator nothing to combine).
    if probe_model == "transformer":
        if per_layer:
            raise ValueError(
                "probe_model='transformer' is incompatible with per_layer=True (a single "
                "block is already sliced); set per_layer=False to let the probe aggregate layers."
            )
        if avg_mode != "none":
            raise ValueError(
                f"probe_model='transformer' requires avg_channels=none (got {avg_mode!r}); "
                "it consumes the un-reduced stacked blocks."
            )
        effective_avg_mode = "stack"
    # Multi-task aesthetic head: a second regression head predicting the clip's aesthetic
    # score alongside the class logit. Implemented for the ResNet probe (the conv_k champion);
    # the target is joined from aesthetic_csv into a 2nd label column by ProbeDataset.
    if multitask_aesthetic:
        if probe_model != "resnet":
            raise ValueError(
                f"multitask_aesthetic requires probe_model='resnet' (got {probe_model!r})."
            )
        if not aesthetic_csv:
            raise ValueError("multitask_aesthetic=True requires aesthetic_csv (per-video score CSV).")
        if aesthetic_loss not in ("bce_soft", "mse"):
            raise ValueError(f"aesthetic_loss must be 'bce_soft' or 'mse' (got {aesthetic_loss!r}).")
    # Per-frame presence head: predicts, for each of the T latent frames, whether the target
    # object is visible in that frame — alongside (not replacing) the clip logit. The targets
    # come from the object labeler's `frames_final` bit-strings, binned onto the latent frames
    # by ProbeDataset into T extra label columns. Implemented for the ResNet probe (the conv_k
    # champion), like the multi-task aesthetic head; the two flags are independent.
    if perframe_presence_head:
        if probe_model != "resnet":
            raise ValueError(
                f"perframe_presence_head requires probe_model='resnet' (got {probe_model!r})."
            )
        if not perframe_csv:
            raise ValueError(
                "perframe_presence_head=True requires perframe_csv (one or more "
                "comma-separated object-labeler CSVs carrying the frames_final column)."
            )
        if perframe_target_mode not in ("majority", "any", "soft"):
            raise ValueError(
                f"perframe_target_mode must be 'majority', 'any' or 'soft' "
                f"(got {perframe_target_mode!r})."
            )
        if perframe_head not in ("shared", "separate"):
            raise ValueError(
                f"perframe_head must be 'shared' or 'separate' (got {perframe_head!r})."
            )
        if perframe_loss_mode not in ("combined", "clip_only", "frame_only", "alternate"):
            raise ValueError(
                f"perframe_loss_mode must be combined|clip_only|frame_only|alternate "
                f"(got {perframe_loss_mode!r})."
            )
    elif perframe_freeze_backbone or perframe_init_ckpt_dir:
        raise ValueError(
            "perframe_freeze_backbone / perframe_init_ckpt_dir are only meaningful with "
            "perframe_presence_head=True."
        )
    # Class-identity head: a from-scratch multi-class head predicting the object CLASS
    # (object_name — dog / cat / …) alongside the class-presence logit. It mirrors the per-frame
    # head — same (optionally frozen) backbone, a parallel 1×1×1 readout — but with n_classes
    # softmax outputs and a cross-entropy loss. The vocab is built once from the train split.
    class_vocab: Optional[dict[str, int]] = None
    if class_head:
        if probe_model != "resnet":
            raise ValueError(f"class_head requires probe_model='resnet' (got {probe_model!r}).")
        class_vocab = build_class_vocab(train_labels_csv)
        (output_dir / "class_vocab.json").write_text(json.dumps(class_vocab, indent=2))
        print(f"  [class_head] {len(class_vocab)} classes from {Path(train_labels_csv).name}")
    elif class_freeze_backbone or class_init_ckpt_dir:
        raise ValueError(
            "class_freeze_backbone / class_init_ckpt_dir are only meaningful with class_head=True."
        )
    # Adversarial (PGD-on-the-attention-maps) training — see training/adversarial.py.
    adv_cfg = AdvConfig.coerce(adv)
    adv_cfg.validate()
    if adv_cfg.enabled and adv_cfg.objective == "perframe_min" and not perframe_presence_head:
        raise ValueError(
            "adv_objective='perframe_min' requires perframe_presence_head=True (there are no "
            "per-frame logits to attack otherwise); use adv_objective='clip'."
        )
    # layer_subset trains on only a subset of the stored transformer blocks (by ORIGINAL
    # index). It slices the layer axis, so it only makes sense for the stacked/mean layer
    # paths — guard against combining it with a single-block slice.
    if layer_subset is not None:
        if effective_avg_mode not in ("stack", "layers"):
            raise ValueError(
                f"layer_subset requires avg_mode in ('stack','layers') (got "
                f"{effective_avg_mode!r}); set resnet_layer_agg (stack) or avg_channels=layers."
            )
        if per_layer:
            raise ValueError("layer_subset is incompatible with per_layer=True (a single block "
                             "is already sliced).")
    # n_blocks is needed by the dataset for both block_index slicing and avg_mode="layers".
    ds_kwargs: dict = dict(n_blocks_heads=n_blocks_heads, latent_thw=latent_thw,
                           avg_mode=effective_avg_mode, n_blocks=n_blocks)
    # Train-time spatial pool: the dataset reshapes the raw feature with the FULL latent_thw,
    # then average-pools (h, w) by this factor. The probe below is therefore built with the
    # POOLED spatial dims (see the latent_thw reassignment after the datasets load).
    if train_spatial_downsample and train_spatial_downsample > 1:
        if latent_thw[1] % train_spatial_downsample or latent_thw[2] % train_spatial_downsample:
            raise ValueError(
                f"train_spatial_downsample={train_spatial_downsample} does not evenly divide "
                f"the latent grid {latent_thw[1]}x{latent_thw[2]}."
            )
        ds_kwargs["train_spatial_downsample"] = int(train_spatial_downsample)
    if feature_cache_dir:
        ds_kwargs["feature_cache_dir"] = feature_cache_dir
    if layer_subset is not None:
        ds_kwargs["layer_subset"] = layer_subset
    # Steering-direction augmentation (off unless training.aug_w_path is set). See
    # ProbeDataset._augment: applies the measured per-(block,head) mass shift to negatives.
    if aug_w_path:
        ds_kwargs["aug_w_path"] = aug_w_path
        ds_kwargs["aug_w_prob"] = float(aug_w_prob)
        ds_kwargs["aug_w_alpha"] = tuple(aug_w_alpha)
        ds_kwargs["aug_w_on"] = str(aug_w_on)
    # Percentile clipping is preprocessing, not augmentation: it goes through the dataset so
    # TRAIN and VAL get the identical transform (and so a checkpoint's inputs are defined by
    # the config that trained it — steering must apply the same band).
    clip_band = parse_clip_percentiles(attn_clip_percentiles)
    if clip_band is not None:
        ds_kwargs["clip_percentiles"] = clip_band
        ds_kwargs["clip_device"] = device if device.type == "cuda" else None
    if multitask_aesthetic:
        ds_kwargs["aux_csv"] = aesthetic_csv
        ds_kwargs["aux_col"] = aesthetic_col
    if perframe_presence_head:
        ds_kwargs["perframe_csv"] = perframe_csv
        ds_kwargs["perframe_col"] = perframe_col
        ds_kwargs["perframe_T"] = latent_thw[0]
        ds_kwargs["perframe_mode"] = perframe_target_mode
    if class_head:
        ds_kwargs["class_head"] = True
        ds_kwargs["class_vocab"] = class_vocab
    # Subset-block extractions: map stored blocks back to original indices on load.
    capture_blocks: Optional[list[int]] = probe_cfg.get("capture_blocks")
    if capture_blocks is not None:
        ds_kwargs["capture_blocks"] = capture_blocks
        ds_kwargs["n_blocks_total"] = probe_cfg.get("n_blocks_total") or n_blocks
    if per_layer:
        ds_kwargs["block_index"] = block_index
    # GPU-resident data: preload features straight into VRAM (no per-batch host→device
    # copy, reused across all epochs). None → keep in CPU RAM and stream per batch
    # (pin_memory + non_blocking overlap the copy with compute).
    #
    # GpuBatchStreamer (hybrid GPU-resident-cache + pinned-stream): the large un-averaged
    # "stack" features (~377 GB fp32 train) don't fit GPU-resident, and naive CPU-streaming
    # re-copies them every epoch (slow). Instead keep TRAIN in PINNED host RAM and feed it via
    # GpuBatchStreamer, which caches as many rows as fit GPU-resident (no per-epoch copy) and
    # streams only the overflow. VAL is small → GPU-resident directly.
    #   gpu_resident_fp16        → streamer in fp16 (compact; caches ~all of a B200) [production]
    #   gpu_resident_stream_fp32 → streamer in fp32 (exact; caches less, streams more) [fp32 ref]
    use_streamer = gpu_resident_fp16 or gpu_resident_stream_fp32
    streamer_half = gpu_resident_fp16
    if use_streamer:
        # BOTH train and val are kept in fp16 PINNED host RAM and fed via GpuBatchStreamer:
        # train caches as many rows as fit GPU-resident (auto-sized to free VRAM), val gets a
        # second streamer built after (so it caches whatever's left, usually streaming all of
        # its small set). The only fixed GPU allocation is a single batch, so the job fits even
        # on a shared/contended GPU. Val is always fp16 (held-out eval data; dtype immaterial to
        # the train-precision question and identical across the fp16 and fp32 runs).
        # Pin the source only when it's small enough to page-lock: fp16 train (~188 GB) pins
        # fine, but fp32 train (~377 GB) exceeds what cudaHostAlloc allows and fails as a
        # "CUDA out of memory". Pinning the big source isn't required anyway — the streamer's
        # small reusable pinned BUFFER provides the fast H2D; an unpinned source only makes the
        # per-batch gather marginally slower. Val is small (fp16), always pin.
        train_device, train_half, train_pin = None, streamer_half, streamer_half
        val_device, val_half, val_pin = None, True, True
    else:
        train_device = val_device = (device if gpu_resident_data else None)
        train_half = val_half = False
        train_pin = val_pin = False
    if preloaded_datasets is not None:
        # Load-once/train-many: the caller built these once for this step and reuses them
        # across configs that differ ONLY in model/optimizer/loss (weight_decay, dropout,
        # label_smoothing, perframe_loss_weight, ...) — the features/labels are identical.
        # train_one_step never mutates the dataset (streamers wrap it read-only), so sharing
        # is safe. Saves the ~15 min preload per config.
        train_ds, val_ds = preloaded_datasets
    else:
        train_ds = ProbeDataset(train_labels_csv, denoising_step=step, root_override=train_root_override, preload=True, device=train_device, half=train_half, pin_memory=train_pin, **ds_kwargs)
        val_ds = ProbeDataset(val_labels_csv, denoising_step=step, root_override=val_root_override, preload=True, device=val_device, half=val_half, pin_memory=val_pin, **ds_kwargs)

    n_pos = int(sum(r[2] >= 0.5 for r in train_ds.records))
    n_neg = int(sum(r[2] < 0.5 for r in train_ds.records))
    n_pos_val = int(sum(r[2] >= 0.5 for r in val_ds.records))
    n_neg_val = int(sum(r[2] < 0.5 for r in val_ds.records))

    # The dataset pooled (h, w) by train_spatial_downsample after reshaping with the FULL
    # latent_thw; every model-facing use below (feat_dim print, build_attnsteer, the saved
    # checkpoint's latent_thw) must therefore see the POOLED spatial dims. ds_kwargs still
    # holds the full tuple, so the already-constructed datasets are unaffected.
    if train_spatial_downsample and train_spatial_downsample > 1:
        _f = int(train_spatial_downsample)
        latent_thw = (latent_thw[0], latent_thw[1] // _f, latent_thw[2] // _f)

    T, h, w = latent_thw
    if per_layer:
        probe_n_blocks_heads = n_heads  # single block: n_heads channels
        probe_n_blocks = 1
    elif effective_avg_mode == "layers":
        probe_n_blocks_heads = n_heads  # blocks averaged away, heads kept as channels
        probe_n_blocks = 1
    elif effective_avg_mode == "all":
        probe_n_blocks_heads = 1
        probe_n_blocks = n_blocks
    elif effective_avg_mode == "stack":
        # Learned layer aggregation: the model receives the stacked blocks and splits
        # n_blocks_heads into [L, H]. L = number of STORED blocks (the captured subset if
        # the extraction stored one), which is what the stacked tensor actually contains.
        # layer_subset narrows L further to the selected blocks.
        if layer_subset is not None:
            probe_n_blocks = len(layer_subset)
            probe_n_blocks_heads = probe_n_blocks * n_heads
        else:
            probe_n_blocks_heads = n_blocks_heads
            probe_n_blocks = len(capture_blocks) if capture_blocks is not None else n_blocks
    else:
        probe_n_blocks_heads = n_blocks_heads
        probe_n_blocks = n_blocks
    # Head-order augmentation: needs an addressable head axis, and the head count has to
    # divide the channel dim of whatever layout the dataset hands back.
    head_shuffle = str(head_shuffle or "none")
    if head_shuffle != "none":
        if not head_shuffle_supported(effective_avg_mode, per_layer):
            raise ValueError(
                f"head_shuffle={head_shuffle!r} needs a layout with a head axis "
                f"(per_layer, avg_channels=layers, or the stacked layer-agg input); "
                f"got avg_mode={effective_avg_mode!r}."
            )
        if n_heads <= 0 or probe_n_blocks * n_heads != probe_n_blocks_heads:
            raise ValueError(
                f"head_shuffle: probe_config n_heads={n_heads} is inconsistent with "
                f"channels={probe_n_blocks_heads} over {probe_n_blocks} blocks."
            )

    if n_pos == 0 or n_neg == 0:
        print(
            f"[warning] [{run_label}] train set has only one class "
            f"(n_pos={n_pos}, n_neg={n_neg}) — skipping."
        )
        return {"pr_table": pd.DataFrame(columns=["threshold", "precision", "recall", "f1"]),
                "probs": [], "labels_binary": [], "epoch_losses": []}

    if len(val_ds) == 0:
        print(
            f"[warning] [{run_label}] val dataset is empty — "
            f"step_{step:03d}.safetensors not found in val CSV paths. Skipping."
        )
        return {"pr_table": pd.DataFrame(columns=["threshold", "precision", "recall", "f1"]),
                "probs": [], "labels_binary": [], "epoch_losses": []}
    
    print(
        f"\n[{run_label}] train={len(train_ds)} ({n_pos}+ {n_pos/(n_pos+n_neg)*100:.0f}%/ {n_neg}- {n_neg/(n_pos+n_neg)*100:.0f}%)  "
        f"val={len(val_ds)} ({n_pos_val}+ {n_pos_val/(n_pos_val+n_neg_val)*100:.0f}%/ {n_neg_val}- {n_neg_val/(n_pos_val+n_neg_val)*100:.0f}%)  "
        f"latent_thw=({T},{h},{w})  feat_dim={probe_n_blocks_heads * T * h * w:,}"
        f"{'  [per_layer]' if per_layer else f'  [avg:{effective_avg_mode}]' if effective_avg_mode != 'none' else ''}"
        f"{f'  [clip:{clip_band[0]:.2f}-{clip_band[1]:.2f}]' if clip_band is not None else ''}"
        f"{f'  [head_shuffle:{head_shuffle}]' if head_shuffle != 'none' else ''}"
    )

    # Source + class balanced loss: per-sample weights that equalize the chosen groups
    # (class / source / source×class) so the loss is a weighted mean. The weight handles class
    # balance, so it REPLACES pos_weight when active. Streamer-path only (production); the weights
    # are threaded through GpuBatchStreamer and applied as a weighted BCE in the train loop.
    sample_weights = None
    if loss_balance != "none":
        from training.balanced_loss import compute_sample_weights, summarize_weights
        srcs = ["dense" if source_tag in str(r[0]) else "vbench" for r in train_ds.records]
        labs = [r[2] for r in train_ds.records]
        sample_weights = compute_sample_weights(labs, srcs, mode=loss_balance, beta=loss_balance_beta)
        print(f"  [{run_label}] loss_balance={loss_balance} beta={loss_balance_beta} → "
              f"{summarize_weights(labs, srcs, sample_weights)}")
        if not use_streamer:
            raise NotImplementedError(
                "loss_balance is implemented for the GpuBatchStreamer path only "
                "(gpu_resident_fp16=true or gpu_resident_stream_fp32=true)."
            )

    # Soft confidence weighting: multiply each sample's loss weight by its detector-agreement
    # confidence (3/3=1.0, 2-of-3=0.67, 1=0.33), keyed by (relative_dir, object_name). Keeps all
    # data (unlike hard-filtering) but trusts the unanimous labels more. Streamer-path only.
    if conf_weights_csv:
        import csv as _csv, os as _os, numpy as _np
        _cmap = {}
        with open(conf_weights_csv) as _f:
            for _r in _csv.DictReader(_f):
                _cmap[(_r["relative_extraction_dir"], _r["object_name"])] = float(_r["weight"])
        _conf = _np.array([_cmap.get((_os.path.basename(r[0]), r[3]), 1.0)
                           for r in train_ds.records], dtype=_np.float32)
        if sample_weights is None:
            sample_weights = _conf
        else:
            sample_weights = _np.asarray(sample_weights, dtype=_np.float32) * _conf
        sample_weights = sample_weights / max(sample_weights.mean(), 1e-8)   # keep mean 1
        print(f"  [{run_label}] confidence weighting from {_os.path.basename(conf_weights_csv)} "
              f"(raw conf mean {_conf.mean():.3f})")
        if not use_streamer:
            raise NotImplementedError("conf_weights_csv needs the streamer path (gpu_resident_fp16=true).")

    train_sampler = BalancedBatchSampler(train_ds, batch_size=batch_size, balanced=balanced_batches)
    # num_workers=0: data is already in RAM/VRAM after preload; workers add fork overhead
    # (and forking a CUDA context is unsafe, so GPU-resident data requires 0 workers).
    # pin_memory only helps the CPU-streaming path; GPU-resident tensors can't be pinned.
    if use_streamer:
        # Hybrid GPU-resident-cache + pinned-stream for the big train set; val is
        # GPU-resident so a plain DataLoader (num_workers=0) reads it straight from VRAM.
        from will_it_render.training.feature_dataset import GpuBatchStreamer, sequential_batches
        # Train streamer first, val streamer after. Default (stream_resident_gb unset) is
        # ADAPTIVE to the GPU: the train cache takes free VRAM minus stream_reserve_gb (batch +
        # activations headroom) minus the val set's size, so val fits resident behind it on
        # any card (B200 caches more train rows, H200 fewer). Explicit stream_resident_gb still
        # pins the train cache (historical configs). Only the streamed fraction (speed)
        # depends on this, never the values.
        val_bytes = val_ds._data.numel() * val_ds._data.element_size()
        streamer = GpuBatchStreamer(train_ds._data, train_ds._labels, device, batch_size=batch_size,
                                    resident_gb=stream_resident_gb,
                                    reserve_gb=stream_reserve_gb + val_bytes / 1e9,
                                    weights=sample_weights)
        val_streamer = GpuBatchStreamer(val_ds._data, val_ds._labels, device, batch_size=batch_size)
        train_loader = streamer.loader(train_sampler, with_weights=sample_weights is not None)
        train_eval_loader = streamer.loader(sequential_batches(len(train_ds), batch_size))
        val_loader = val_streamer.loader(sequential_batches(len(val_ds), batch_size))
    else:
        # pin_memory only helps (and is only valid) for CPU-resident tensors.
        pin = not gpu_resident_data
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=0, pin_memory=pin)
        train_eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    model = build_attnsteer(
        probe_model=probe_model,
        n_blocks_heads=probe_n_blocks_heads,
        latent_thw=latent_thw,
        bottleneck_ch=bottleneck_ch,
        conv_kernel_size=conv_kernel_size,
        mlp_hidden=mlp_hidden,
        mlp_hidden_dims=mlp_hidden_dims,
        dropout=dropout,
        n_blocks=probe_n_blocks,
        n_heads=n_heads,
        attn_pool_conv_ch=attn_pool_conv_ch,
        attn_pool_mlp_hidden=attn_pool_mlp_hidden,
        resnet_base_ch=resnet_base_ch,
        resnet_n_blocks=resnet_n_blocks,
        resnet_groups=resnet_groups,
        resnet_temporal_spatial=resnet_temporal_spatial,
        resnet_head_dropout=resnet_head_dropout,
        resnet_dropblock_size=resnet_dropblock_size,
        resnet_norm=resnet_norm,
        resnet_layer_agg=resnet_layer_agg,
        resnet_layer_agg_k=resnet_layer_agg_k,
        resnet_layer_agg_hidden=resnet_layer_agg_hidden,
        resnet_layer_agg_init_blocks=resnet_layer_agg_init_blocks,
        resnet_layer_agg_init_random=resnet_layer_agg_init_random,
        resnet_step_cond=resnet_step_cond,
        resnet_step_embed_dim=resnet_step_embed_dim,
        resnet_multitask=multitask_aesthetic,
        resnet_perframe=perframe_presence_head,
        resnet_perframe_head=perframe_head,
        resnet_perframe_tap=perframe_tap,
        resnet_input_norm=resnet_input_norm,
        resnet_classhead=class_head,
        resnet_n_classes=(len(class_vocab) if class_head else 0),
        resnet_class_head_arch=class_head_arch,
        resnet_class_mlp_hidden=class_mlp_hidden,
        transformer_d_model=transformer_d_model,
        transformer_nhead=transformer_nhead,
        transformer_num_layers=transformer_num_layers,
        transformer_dim_feedforward=transformer_dim_feedforward,
        transformer_head_attn_nheads=transformer_head_attn_nheads,
        transformer_layer_attn_nheads=transformer_layer_attn_nheads,
        transformer_token_hidden=transformer_token_hidden,
        transformer_token_mode=transformer_token_mode,
        transformer_token_conv_ch=transformer_token_conv_ch,
        transformer_mlp_hidden=transformer_mlp_hidden,
        transformer_pool=transformer_pool,
        transformer_timestep_mode=transformer_timestep_mode,
        transformer_num_timesteps=transformer_num_timesteps,
    ).to(device)

    # Per-frame warm start: reuse a trained clip-level probe's weights for this same step.
    # With perframe_head="shared" the state dict is identical to the clip-only probe's, so a
    # champion checkpoint loads exactly; "separate" adds conv_out_frame, which is seeded from
    # conv_out at construction and therefore left untouched by the (non-strict) load.
    # Warm start (shared by the per-frame head-only fine-tune and the adversarial fine-tune:
    # both take an already-trained probe for this same step and keep training it).
    frozen_modules: list[nn.Module] = []
    init_ckpt_dir = perframe_init_ckpt_dir or class_init_ckpt_dir or adv_cfg.init_ckpt_dir
    if init_ckpt_dir:
        init_key = ("perframe_init_ckpt_dir" if perframe_init_ckpt_dir else
                    "class_init_ckpt_dir" if class_init_ckpt_dir else "adv_init_ckpt_dir")
        init_path = Path(init_ckpt_dir) / run_label / f"probe_{run_label}.pt"
        if not init_path.exists():
            raise FileNotFoundError(
                f"{init_key} is set but {init_path} does not exist. Point it at a "
                f"training run dir containing {run_label}/probe_{run_label}.pt."
            )
        init_ckpt = torch.load(str(init_path), map_location=device, weights_only=False)
        incompatible = model.load_state_dict(init_ckpt["model_state"], strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(
                f"warm-start checkpoint {init_path} has unexpected keys "
                f"{incompatible.unexpected_keys} — architecture mismatch."
            )
        pre_temporal = getattr(model, "perframe_tap", "post_temporal") == "pre_temporal"
        if pre_temporal:
            # The pre-temporal frame branch (frame_block / frame_norm_out / conv_out_frame) does
            # not exist in a clip-only checkpoint — allow it to keep its fresh init. It reads a
            # different feature stream than conv_out, so it is NOT re-seeded from the clip head.
            allowed_missing = {k for k in incompatible.missing_keys
                               if k.startswith(("frame_stem.", "frame_res.", "frame_down.",
                                                "frame_block.", "frame_norm_out.",
                                                "conv_out_frame."))}
        else:
            allowed_missing = {"conv_out_frame.weight", "conv_out_frame.bias"}
        if class_head:
            # A clip-only checkpoint has no class head; let the fresh class head (pooled conv OR
            # the separate branch + FC) keep its random init — the "train from scratch" case.
            allowed_missing = allowed_missing | {
                k for k in incompatible.missing_keys
                if k.startswith(("conv_out_class.", "class_stem.", "class_res.", "class_down.",
                                 "class_block.", "class_norm.", "class_fc."))
            }
        if set(incompatible.missing_keys) - allowed_missing:
            raise ValueError(
                f"warm-start checkpoint {init_path} is missing keys "
                f"{sorted(set(incompatible.missing_keys) - allowed_missing)}."
            )
        if not pre_temporal and "conv_out_frame.weight" in incompatible.missing_keys:
            # A "separate" frame head seeded from conv_out at construction would otherwise
            # keep its copy of the RANDOM init; re-seed it from the weights we just loaded so
            # the warm start actually starts at the clip probe's readout.
            with torch.no_grad():
                model.conv_out_frame.weight.copy_(model.conv_out.weight)
                model.conv_out_frame.bias.copy_(model.conv_out.bias)
        print(f"[train] warm-started from {init_path} "
              f"(best_epoch={init_ckpt.get('best_epoch')}, "
              f"best_val_loss={init_ckpt.get('best_val_loss')})")

    if perframe_freeze_backbone:
        # Head-only fine-tune: everything but the logit head(s) is frozen, so the only thing
        # training can change is HOW the fixed features are read out per frame. Frozen modules
        # are also kept in eval() each epoch (below) so dropout / DropBlock don't inject noise
        # into what are meant to be fixed features.
        head_params = set(id(p) for p in model.frame_head_parameters())
        for p in model.parameters():
            p.requires_grad_(id(p) in head_params)
        frozen_modules = [m for name, m in model.named_children()
                          if name not in ("conv_out", "conv_out_frame", "frame_stem",
                                          "frame_res", "frame_down", "frame_block",
                                          "frame_norm_out")]
        n_train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all_p = sum(p.numel() for p in model.parameters())
        print(f"[train] frozen backbone: {n_train_p}/{n_all_p} trainable params (head only)")

    if class_freeze_backbone:
        # Class-head-only fine-tune: freeze the whole model except the new class head, so the
        # only thing training can change is the class readout over the fixed features. Frozen
        # modules are kept in eval() each epoch (below) so dropout/DropBlock don't perturb them.
        head_params = set(id(p) for p in model.class_head_parameters())
        for p in model.parameters():
            p.requires_grad_(id(p) in head_params)
        _CLASS_HEAD_MODS = {"conv_out_class", "class_stem", "class_res", "class_down",
                            "class_block", "class_norm", "class_fc"}
        frozen_modules = [m for name, m in model.named_children() if name not in _CLASS_HEAD_MODS]
        n_train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all_p = sum(p.numel() for p in model.parameters())
        print(f"[train] frozen backbone (class head only, arch={class_head_arch}): "
              f"{n_train_p}/{n_all_p} trainable params")

    # Adversarial training: a PGD attack on the input attention maps, run per batch to
    # simulate the inference-time steering optimizer's search for score-spiking maps.
    attack: Optional[AttentionPGD] = None
    if adv_cfg.enabled:
        attack = AttentionPGD(adv_cfg, feat_shape=(probe_n_blocks_heads, *latent_thw),
                              n_blocks=probe_n_blocks)
        print(f"  [{run_label}] {adv_cfg.summary()}")
    if adv_cfg.objective == "perframe_min":
        def adv_forward(m, z):
            return m(z, return_frames=True)
    else:
        def adv_forward(m, z):
            return m(z)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=lr, weight_decay=weight_decay)
    # LR schedule (stepped once per epoch). "cosine" anneals once over the whole run;
    # "cosine_warm_restarts" (SGDR) restarts the cosine every cosine_restart_period epochs
    # (period growing by cosine_restart_mult each cycle) — the periodic warm restarts re-raise
    # the LR to escape sharp minima. Pair with a generous early_stop_patience so a transient
    # post-restart val spike doesn't trip early stopping mid-cycle.
    if lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif lr_scheduler == "cosine_warm_restarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=cosine_restart_period, T_mult=cosine_restart_mult
        )
    else:
        raise ValueError(
            f"Invalid lr_scheduler {lr_scheduler!r}; choose 'cosine' | 'cosine_warm_restarts'"
        )
    auto_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    pw = pos_weight if pos_weight is not None else auto_pos_weight
    print(f"  [{run_label}] pos_weight={pw:.3f} (n_pos={n_pos}, n_neg={n_neg}{'  [auto]' if pos_weight is None else '  [manual]'})")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    # Weighted-loss path: per-sample BCE (no pos_weight — the sample weights already balance
    # class as well as source) reduced as a weighted mean. mean(weights)==1 keeps the loss scale
    # comparable to the unweighted run.
    weighted = sample_weights is not None
    criterion_w = nn.BCEWithLogitsLoss(reduction="none")
    if weighted:
        print(f"  [{run_label}] using source/class weighted loss (pos_weight disabled)")
    val_criterion = nn.BCEWithLogitsLoss()
    # Aux (aesthetic) head loss. The target is the continuous score in [0,1]:
    #   bce_soft → BCEWithLogits with the score as a soft target (prediction = sigmoid(logit),
    #              bounded to [0,1]; reuses the class head's loss family — the default).
    #   mse      → MSE on sigmoid(logit) vs the score.
    # reduction="none" so NaN aux targets (clips absent from aesthetic_csv) are masked per-sample.
    aux_criterion = nn.BCEWithLogitsLoss(reduction="none") if aesthetic_loss == "bce_soft" else None
    # Per-frame presence loss. reduction="none" so undecided frames (all-'u' bins) and rows
    # absent from the per-frame CSVs — both NaN — are masked out per (sample, frame) cell.
    frame_criterion = nn.BCEWithLogitsLoss(reduction="none")
    # Column offset of the first per-frame target: 0=class, 1=aesthetic aux (multitask only),
    # then the T per-frame columns. Mirrors ProbeDataset.perframe_offset.
    pf_off = 1 + int(multitask_aesthetic)
    pf_T = latent_thw[0]
    # Column index of the class-identity target (the last packed label column): after the
    # class-presence label, the aesthetic aux, and the T per-frame columns. Mirrors
    # ProbeDataset.class_offset. Cross-entropy over the class logits; targets of -1 (object_name
    # absent from the train-built vocab) are masked out.
    class_off = pf_off + (pf_T if perframe_presence_head else 0)
    # Per-class CE weights for the class head (imbalanced 156-way target). Computed once from the
    # TRAIN split's class counts and normalized to mean 1 (so the loss scale is comparable to the
    # unweighted run and to class_loss_weight). none = plain CE.
    #   inverse       w_c ∝ 1/n_c                 (full inverse-frequency)
    #   sqrt_inverse  w_c ∝ 1/sqrt(n_c)           (gentler)
    #   effective_num w_c ∝ (1-β)/(1-β^{n_c})     (class-balanced loss, Cui et al. 2019)
    class_weight = None
    if class_head and class_loss_balance != "none":
        import numpy as _np
        counts = _np.zeros(len(class_vocab), dtype=_np.float64)
        for r in train_ds.records:
            ci = class_vocab.get(r[3], -1)
            if ci >= 0:
                counts[ci] += 1
        counts = _np.maximum(counts, 1.0)
        if class_loss_balance == "inverse":
            w = counts.sum() / counts
        elif class_loss_balance == "sqrt_inverse":
            w = 1.0 / _np.sqrt(counts)
        elif class_loss_balance == "effective_num":
            b = class_loss_balance_beta
            w = (1.0 - b) / (1.0 - _np.power(b, counts))
        else:
            raise ValueError(
                f"class_loss_balance must be none|inverse|sqrt_inverse|effective_num "
                f"(got {class_loss_balance!r}).")
        w = w * (len(w) / w.sum())  # mean weight → 1
        class_weight = torch.tensor(w, dtype=torch.float32, device=device)
        print(f"  [class_head] class_loss_balance={class_loss_balance} "
              f"(weights: min={w.min():.3f} max={w.max():.3f})")

    def compute_task_loss(feats, labels, wts):
        """Forward + task loss for one batch of features (clean OR adversarial).

        Factored out of the epoch loop so the adversarial batch is scored by exactly the
        same objective as the clean one. Returns ``(loss, aux_value, frame_value)``; the
        two component values are plain floats for the per-epoch component curves.
        """
        clabels = labels[:, :1]
        if lipschitz_weight:
            feats.requires_grad_(True)   # input-gradient penalty needs grad w.r.t. the input maps
        # Single forward requesting whichever extra heads are active; the returns come back in a
        # fixed order (logit, [aux], [frame_logits], [class_logits]) so this unpack stays valid
        # for any combination of heads.
        want_aux, want_frames, want_class = multitask_aesthetic, perframe_presence_head, class_head
        # The extra-head kwargs only exist on ResNetProbe; other probes take model(feats). All
        # three heads are resnet-guarded, so pass the kwargs only when one is actually requested.
        outs = (model(feats, return_aux=want_aux, return_frames=want_frames, return_class=want_class)
                if (want_aux or want_frames or want_class) else model(feats))
        aux_pred = frame_logits = class_logits = None
        if isinstance(outs, tuple):
            rest = list(outs)
            logits = rest.pop(0)
            if want_aux:
                aux_pred = rest.pop(0)
            if want_frames:
                frame_logits = rest.pop(0)
            if want_class:
                class_logits = rest.pop(0)
        else:
            logits = outs
        # Label smoothing pulls the BCE targets off {0,1} → reduces overconfidence/overfit.
        tgt = clabels if label_smoothing <= 0 else clabels * (1 - label_smoothing) + 0.5 * label_smoothing
        if obj_loss == "ranking":
            # Pairwise AUROC surrogate: softplus(-(logit_pos - logit_neg)) over all pos×neg pairs.
            yh = (clabels.view(-1) > 0.5)
            lp = logits.view(-1)[yh]; ln = logits.view(-1)[~yh]
            if lp.numel() > 0 and ln.numel() > 0:
                loss = torch.nn.functional.softplus(-(lp[:, None] - ln[None, :])).mean()
            else:
                loss = criterion(logits, tgt)  # degenerate (all-one-class) batch → fall back
        elif obj_loss == "gce":
            # Generalized Cross-Entropy (noise-robust): (1 - p_t^q)/q on the hard target.
            p = torch.sigmoid(logits)
            yh = (clabels > 0.5).float().view_as(p)
            pt = (p * yh + (1 - p) * (1 - yh)).clamp(min=1e-6)
            per = (1 - pt ** gce_q) / gce_q
            if wts is not None:
                w = wts.view_as(per); loss = (per * w).sum() / w.sum()
            else:
                loss = per.mean()
        elif wts is not None:
            w = wts.view_as(logits)
            loss = (criterion_w(logits, tgt) * w).sum() / w.sum()
        else:
            loss = criterion(logits, tgt)
        clip_bce_value = loss.item()  # raw per-clip loss (BCE/GCE/ranking), before aux/per-frame
        # Loss-level regularizers (added to the total, not to the logged task loss):
        if entropy_weight:
            # Confidence penalty (Pereyra): subtract β·H(p) → encourage higher-entropy (less
            # overconfident) outputs; a differentiable sibling of label smoothing.
            p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
            H = -(p * p.log() + (1 - p) * (1 - p).log())
            loss = loss - entropy_weight * H.mean()
        if lipschitz_weight:
            # Input-gradient (Lipschitz) penalty: λ·mean(‖∂logit/∂feats‖²) → smoother probe,
            # harder for the steering optimizer to walk off-manifold.
            g = torch.autograd.grad(logits.sum(), feats, create_graph=True, retain_graph=True)[0]
            loss = loss + lipschitz_weight * g.flatten(1).pow(2).sum(1).mean()
        aux_value = 0.0
        frame_value = 0.0
        # Aux (aesthetic) regression head: masked over NaN targets (clips absent from
        # aesthetic_csv), weighted by aesthetic_loss_weight, added to the class loss.
        if multitask_aesthetic:
            aux_tgt = labels[:, 1:2]
            aux_mask = ~torch.isnan(aux_tgt)
            if aux_mask.any():
                safe_tgt = torch.nan_to_num(aux_tgt, nan=0.0)
                if aux_criterion is not None:            # bce_soft
                    per = aux_criterion(aux_pred, safe_tgt)
                else:                                    # mse on sigmoid(logit)
                    per = (torch.sigmoid(aux_pred) - safe_tgt) ** 2
                aux_loss = (per * aux_mask).sum() / aux_mask.sum()
                loss = loss + aesthetic_loss_weight * aux_loss
                aux_value = aux_loss.item()              # unweighted aesthetic component
        # Per-frame presence head: masked BCE over the [B, T] block of label columns.
        # Undecided frames and unjoined rows are NaN and drop out of both numerator and
        # denominator. perframe_class_loss_weight scales the clip term (0 = pure per-frame).
        if perframe_presence_head:
            class_term = loss  # per-clip head objective (clip BCE, + aux if multitask)
            ftgt = labels[:, pf_off:pf_off + pf_T]
            fmask = ~torch.isnan(ftgt)
            frame_term = None
            if fmask.any():
                fper = frame_criterion(frame_logits, torch.nan_to_num(ftgt, nan=0.0))
                frame_term = (fper * fmask).sum() / fmask.sum()
                frame_value = frame_term.item()          # unweighted per-frame component
            # Select the OPTIMISED loss by mode. per-clip / per-frame BCE are still logged in
            # every mode (train_clip_loss / train_frame_loss), so the report shows all three.
            if perframe_loss_mode == "clip_only":
                loss = class_term
            elif perframe_loss_mode == "frame_only":
                loss = frame_term if frame_term is not None else class_term * 0.0
            elif perframe_loss_mode == "alternate":         # even epoch → clip, odd → per-frame
                loss = class_term if (epoch % 2 == 0) else (
                    frame_term if frame_term is not None else class_term * 0.0)
            else:                                           # "combined" (default): weighted sum
                loss = perframe_class_loss_weight * class_term
                if frame_term is not None:
                    loss = loss + perframe_loss_weight * frame_term
        # Class-identity head: multi-class cross-entropy on the class logits, added to the
        # objective (weighted by class_loss_weight). Targets of -1 (object_name not in the
        # train-built vocab) are masked out. With class_freeze_backbone the backbone + clip head
        # have no grad, so this term drives ONLY the new class head.
        class_value = 0.0
        if class_head and class_logits is not None:
            ctgt = labels[:, class_off].long()
            cmask = ctgt >= 0
            if cmask.any():
                class_loss = F.cross_entropy(class_logits[cmask], ctgt[cmask], weight=class_weight)
                loss = loss + class_loss_weight * class_loss
                class_value = class_loss.item()
        return (loss, aux_value, frame_value, clip_bce_value, logits, frame_logits,
                class_value, class_logits)

    epoch_losses: list[tuple[int, float, float]] = []
    train_epoch_metrics: list[tuple[int, float, float, float]] = []
    val_epoch_metrics: list[tuple[int, float, float, float]] = []

    if wandb_run is not None:
        wandb_run.define_metric(f"{run_label}/epoch")
        wandb_run.define_metric(f"{run_label}/*", step_metric=f"{run_label}/epoch")

    # Best-epoch tracking. Select the epoch that optimizes `best_metric`; keep a
    # CPU snapshot of its weights so the saved checkpoint + reported results
    # reflect that epoch rather than the (often overfit) final one.
    # val_adv_loss / val_combined select on ROBUSTNESS, not just clean skill. The 2026-08-13
    # run showed why that matters: clean val_loss bottoms out before adversarial training has
    # done its work, so selecting on it saves near-un-hardened weights (step_010: adv AUROC
    # 0.010 at the saved epoch vs 0.709 at the last). val_combined is the mean of the clean and
    # adversarial class BCE — the quantity to minimise when the goal is BOTH losses low.
    # val_frame_* select on the PER-FRAME head — the readout the steering objective
    # uses. Selecting on the clip head (val_combined) is what produced the
    # 2026-08-13 mis-ranking: clip adv AUROC 0.998 with min-frame at 0.000.
    ADV_METRICS = ("val_adv_loss", "val_combined", "val_frame_combined",
                   "val_frame_adv_auroc")
    CLASS_METRICS = ("val_class_loss", "val_class_acc")
    if best_metric not in ("val_loss", "val_auroc", "val_f1", *ADV_METRICS, *CLASS_METRICS):
        raise ValueError(f"Invalid best_metric {best_metric!r}; choose val_loss | val_auroc | "
                         f"val_f1 | val_adv_loss | val_combined | val_class_loss | val_class_acc")
    if best_metric.startswith("val_frame") and not perframe_presence_head:
        raise ValueError(f"best_metric={best_metric!r} requires perframe_presence_head=true.")
    if best_metric in CLASS_METRICS and not class_head:
        raise ValueError(f"best_metric={best_metric!r} requires class_head=true.")
    if best_metric in ADV_METRICS and not (adv_cfg.enabled and adv_cfg.eval_robust):
        raise ValueError(f"best_metric={best_metric!r} requires adv_training=true and "
                         f"adv_eval_robust=true (it is read from the robust val pass).")
    higher_better = best_metric in ("val_auroc", "val_f1", "val_frame_adv_auroc", "val_class_acc")
    best_value = float("-inf") if higher_better else float("inf")
    best_epoch = 0
    best_state: Optional[dict] = None
    best_val_loss = float("nan")
    best_train_loss = float("nan")
    best_lr = float("nan")
    epochs_since_improve = 0
    early_stopped = False
    epochs_trained = 0

    # Per-component loss curves (multitask only): (epoch, train_aux, val_aux), unweighted, so the
    # report can split total into class vs aesthetic. val_loss/epoch_val_loss stay class-only.
    aux_epoch_losses: list[tuple[int, float, float]] = []
    # Same idea for the per-frame head: (epoch, train_frame, val_frame), unweighted.
    frame_epoch_losses: list[tuple[int, float, float]] = []
    # Class-identity head: (epoch, train_class_ce, val_class_ce), unweighted.
    class_epoch_losses: list[tuple[int, float, float]] = []
    # Per-epoch class-head metrics flushed to disk EVERY epoch (live monitoring): train/val CE,
    # train/val top-1, val top-5, lr. Written to class_metrics_{run_label}.csv as it accumulates
    # so a running (or killed) job exposes the class curve without waiting for results.json.
    class_epoch_rows: list[dict] = []
    # Explicit per-clip and COMBINED loss curves so the report never has to derive them (the
    # combine weights may change): (epoch, train, val). per-clip = the class BCE component;
    # combined = per-clip + per-frame as actually weighted in the objective.
    clip_epoch_losses: list[tuple[int, float, float]] = []
    combined_epoch_losses: list[tuple[int, float, float]] = []
    last_perframe_eval: Optional[dict] = None
    perframe_init_eval: Optional[dict] = None
    # Adversarial bookkeeping, all flushed to CSV as it accumulates (see the eval block).
    adv_train_rows: list[dict] = []
    adv_eval_rows: list[dict] = []
    adv_traces: list[tuple[int, list]] = []
    # Robustness is measured on BOTH splits: the attack is only ever *trained* against the
    # train set, so the train→val robust gap says whether robustness generalized or was
    # memorized on the training maps. The train pass is capped (adv_eval_train_max) because
    # attacking ~18k examples every eval epoch would cost more than the training itself.
    adv_train_max_batches = (
        max(1, -(-adv_cfg.eval_train_max // batch_size)) if adv_cfg.eval_train_max else None
    )
    adv_splits: list[tuple[str, object, object, Optional[int]]] = []
    if attack is not None and adv_cfg.eval_robust:
        if adv_cfg.eval_train:
            adv_splits.append(("train", train_eval_loader, train_ds, adv_train_max_batches))
        adv_splits.append(("val", val_loader, val_ds, None))

    def run_robust(epoch_label: int, **kw) -> dict[str, dict]:
        """Attack every configured split; return {split: metrics}. Bulky payloads stay out
        of the returned scalars — the caller persists them."""
        out: dict[str, dict] = {}
        for split, loader, ds, max_b in adv_splits:
            out[split] = evaluate_adversarial(
                model, loader, device, attack, forward=adv_forward, dataset=ds,
                max_batches=max_b,
                perframe=((pf_off, pf_T) if perframe_presence_head else None),
                **(kw if split == "val" else {}),
            )
        return out
    if perframe_presence_head:
        # Per-frame readout BEFORE any optimizer step. Warm-started from a clip-level probe
        # this is the zero-training reference: because the clip head is linear over a mean,
        # the champion checkpoint ALREADY defines per-frame logits, and this measures how much
        # per-frame presence its representation carries for free. Training is then judged
        # against this row, not just against chance.
        perframe_init_eval = evaluate_perframe(model, val_loader, device, pf_T, pf_off)
        perframe_init_eval["per_frame"].to_csv(
            str(output_dir / f"perframe_eval_{run_label}_init.csv"), index=False)
        print(f"  [{run_label}] per-frame at init (no training): "
              f"frame AUROC={perframe_init_eval['frame_auroc']:.3f}  "
              f"all-frames AUROC={perframe_init_eval.get('allframes_auroc', float('nan')):.3f}")
    epoch_bar = tqdm(range(num_epochs), desc=run_label, unit="ep", leave=True)
    for epoch in epoch_bar:
        val_eval = None  # cache this epoch's val evaluate_probe (reused for monitoring)
        model.train()
        for m in frozen_modules:
            m.eval()  # frozen backbone → no dropout/DropBlock noise in the fixed features
        train_loss = 0.0        # COMBINED (per-clip + per-frame, as weighted below)
        train_clip_loss = 0.0   # per-clip BCE component only (unweighted)
        train_aux_loss = 0.0
        train_frame_loss = 0.0
        train_class_loss = 0.0  # class-identity CE component (unweighted)
        train_class_correct = 0
        train_class_total = 0
        train_clean_loss = 0.0
        train_adv_loss = 0.0
        n_train = 0
        n_adv = 0
        # Clean-only warmup: attacking a randomly-initialised probe just chases noise, so
        # the attack switches on once the clean objective has shaped a boundary to attack.
        use_adv = attack is not None and epoch >= adv_cfg.warmup_epochs
        for batch in train_loader:
            # weighted path yields (feats, labels, weights); default yields (feats, labels)
            if weighted:
                feats, labels, wts = batch
                wts = wts.to(device, non_blocking=True)
            else:
                feats, labels = batch
                wts = None
            # non_blocking overlaps the copy with compute on the CPU-streaming path
            # (pinned memory); it's a no-op when data is already GPU-resident.
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # Re-drawn every batch, so a given sample is seen under a different head order
            # each epoch — TRAIN ONLY (val/eval keep the natural order, which is also what
            # the probe sees at steering time).
            if head_shuffle != "none":
                feats = shuffle_heads(feats, probe_n_blocks, n_heads, head_shuffle)
            bsz = labels.shape[0]
            optimizer.zero_grad()
            # Column 0 is the class label; extra columns (multitask) are aux targets.
            # compute_task_loss owns the forward + perframe_loss_mode selection; it returns the
            # raw per-clip BCE (clip_bce_value) so all three components stay logged in every mode.
            loss, aux_value, frame_value, clip_bce_value, clean_logits, clean_frames, class_value, clean_class = compute_task_loss(
                feats, labels, wts if weighted else None)
            train_aux_loss += aux_value * bsz
            train_frame_loss += frame_value * bsz
            train_class_loss += class_value * bsz
            if class_head and clean_class is not None:
                ct = labels[:, class_off].long(); cm = ct >= 0
                if cm.any():
                    train_class_correct += int((clean_class[cm].detach().argmax(1) == ct[cm]).sum())
                    train_class_total += int(cm.sum())
            train_clip_loss += clip_bce_value * bsz   # per-clip BCE component (always logged)
            train_clean_loss += loss.item() * bsz   # pre-interpolation, for the clean/adv gap
            # Aggregator-targeted regularization (L1 sparsity / attn entropy) on the layer combine;
            # no-op for non-resnet probes or when both coefficients are 0. Read here, off the
            # CLEAN forward's cached weights, before any adversarial forward overwrites them.
            reg_term = None
            if resnet_layer_agg_l1 or resnet_layer_agg_entropy:
                reg = getattr(model, "layer_agg_reg", None)
                if reg is not None:
                    reg_term = reg(resnet_layer_agg_l1, resnet_layer_agg_entropy)
            # SPLIT BACKWARD. Summing the clean and adversarial losses and backward-ing once
            # keeps the CLEAN graph alive through the whole PGD inner loop — and that loop
            # already holds several full-batch fp32 copies of the input (2.6 GB each at batch
            # 128 × 1536 × 13 × 16 × 16). Backward-ing each term as soon as it is produced is
            # mathematically identical, since gradients accumulate into .grad and
            #   d/dθ[(1-α)·L_clean + α·L_adv] = (1-α)·dL_clean + α·dL_adv,
            # but it roughly halves peak VRAM. The aggregator regularizer rides with the
            # clean term so its cached forward weights are still inside that graph.
            # Snapshot the clean logits before backward frees the graph — alp/trades need them
            # as a detached reference for the consistency term.
            # Reference for the alp/trades consistency term. It MUST be the readout the
            # attack targets: with adv_objective=perframe_min the attack moves
            # min_t frame_logit, so pairing on the clip logit (the original behaviour, kept
            # as adv_pair_on='clip') hardens a head the attack never aims at — measured as
            # clip adv AUROC 0.998 while min-frame adv AUROC stayed 0.000.
            pair_on_frames = (adv_cfg.pair_on == "frame") and (clean_frames is not None)
            clean_ref = ((clean_frames if pair_on_frames else clean_logits).detach()
                         if use_adv else None)
            clean_w = (1.0 - adv_cfg.alpha) if (use_adv and adv_cfg.loss != "trades") else 1.0
            clean_term = clean_w * loss
            if reg_term is not None:
                clean_term = clean_term + reg_term
            clean_term.backward()
            total_value = clean_term.item()
            del clean_term, loss, reg_term, clean_logits, clean_frames
            # Adversarial batch: PGD-perturbed attention maps for the same labels, scored by
            # the same objective. The attack runs with frozen params and takes gradients
            # w.r.t. the input only, so it never touches the p.grad accumulated above.
            if use_adv:
                sel = attack.select(labels[:, :1])
                n_sel = int(sel.sum().item())
                if n_sel:
                    feats_adv = attack.perturb(model, feats[sel], labels[sel, :1],
                                               forward=adv_forward)
                    adv_loss, _, _, _, adv_logits, adv_frames, _, _ = compute_task_loss(
                        feats_adv, labels[sel], wts[sel] if weighted else None)
                    cur = adv_frames if pair_on_frames else adv_logits
                    # How the adversarial term enters the objective:
                    #   bce    — Madry: plain task loss on the perturbed maps.
                    #   alp    — adversarial logit pairing: adds λ·(logit_clean − logit_adv)²,
                    #            tying the two readings together instead of only pushing each
                    #            toward its label. Aimed at the clean/robust trade-off.
                    #   trades — the adversarial BCE is REPLACED by β·KL(p_adv ‖ p_clean), so the
                    #            clean term keeps full weight and the attack only has to be made
                    #            consistent with it, not separately correct.
                    ref = clean_ref[sel]
                    if adv_cfg.loss == "trades":
                        adv_term = adv_cfg.trades_beta * _bernoulli_kl(cur, ref)
                    else:
                        adv_term = adv_cfg.alpha * adv_loss
                        if adv_cfg.loss == "alp":
                            adv_term = adv_term + adv_cfg.trades_beta * (
                                (cur - ref) ** 2).mean()
                    adv_term.backward()
                    train_adv_loss += adv_loss.item() * n_sel
                    total_value += adv_term.item()
                    n_adv += n_sel
                    del feats_adv, adv_loss, adv_term
            optimizer.step()
            train_loss += total_value * bsz   # clean (+ adversarial) total actually optimised
            n_train += bsz
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_aux_loss = 0.0
        val_frame_loss = 0.0
        val_class_loss = 0.0
        val_class_correct = 0
        val_class_top5_correct = 0
        val_class_total = 0
        n_val = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats = feats.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                bsz = labels.shape[0]
                # val_loss tracks the primary (class-presence) task only, so model selection /
                # best_metric stay comparable to single-task runs. Same fixed-order unpack as
                # compute_task_loss.
                want_aux, want_frames, want_class = (
                    multitask_aesthetic, perframe_presence_head, class_head)
                vouts = (model(feats, return_aux=want_aux, return_frames=want_frames,
                               return_class=want_class)
                         if (want_aux or want_frames or want_class) else model(feats))
                vaux = vframe = vclass = None
                if isinstance(vouts, tuple):
                    vrest = list(vouts)
                    vlogit = vrest.pop(0)
                    if want_aux:
                        vaux = vrest.pop(0)
                    if want_frames:
                        vframe = vrest.pop(0)
                    if want_class:
                        vclass = vrest.pop(0)
                else:
                    vlogit = vouts
                val_loss += val_criterion(vlogit, labels[:, :1]).item() * bsz
                if multitask_aesthetic:
                    vtgt = labels[:, 1:2]; vmask = ~torch.isnan(vtgt)
                    if vmask.any():
                        vsafe = torch.nan_to_num(vtgt, nan=0.0)
                        vper = (aux_criterion(vaux, vsafe) if aux_criterion is not None
                                else (torch.sigmoid(vaux) - vsafe) ** 2)
                        val_aux_loss += ((vper * vmask).sum() / vmask.sum()).item() * bsz
                if perframe_presence_head:
                    vftgt = labels[:, pf_off:pf_off + pf_T]
                    vfmask = ~torch.isnan(vftgt)
                    if vfmask.any():
                        vfper = frame_criterion(vframe, torch.nan_to_num(vftgt, nan=0.0))
                        val_frame_loss += ((vfper * vfmask).sum() / vfmask.sum()).item() * bsz
                if class_head and vclass is not None:
                    vctgt = labels[:, class_off].long()
                    vcmask = vctgt >= 0
                    n_valid = int(vcmask.sum())
                    if n_valid:
                        vc = vclass[vcmask]; vt = vctgt[vcmask]
                        val_class_loss += F.cross_entropy(vc, vt).item() * n_valid
                        val_class_correct += int((vc.argmax(dim=1) == vt).sum())
                        k5 = min(5, vc.shape[1])
                        top5 = vc.topk(k5, dim=1).indices
                        val_class_top5_correct += int((top5 == vt[:, None]).any(dim=1).sum())
                        val_class_total += n_valid
                n_val += bsz

        epoch_train_loss = train_loss / n_train if n_train > 0 else float("nan")
        epoch_val_loss = val_loss / n_val if n_val > 0 else float("nan")
        epoch_losses.append((epoch + 1, epoch_train_loss, epoch_val_loss))
        # Class-identity head epoch summaries (NaN when the head is off / no valid targets).
        epoch_class_loss = val_class_loss / val_class_total if val_class_total else float("nan")
        epoch_class_acc = val_class_correct / val_class_total if val_class_total else float("nan")
        epoch_class_top5 = val_class_top5_correct / val_class_total if val_class_total else float("nan")
        epoch_train_class_ce = train_class_loss / n_train if n_train > 0 else float("nan")
        epoch_train_class_acc = train_class_correct / train_class_total if train_class_total else float("nan")
        if class_head:
            class_epoch_losses.append((epoch + 1, epoch_train_class_ce, epoch_class_loss))
        if multitask_aesthetic:
            aux_epoch_losses.append((epoch + 1,
                                     train_aux_loss / n_train if n_train > 0 else float("nan"),
                                     val_aux_loss / n_val if n_val > 0 else float("nan")))
        if perframe_presence_head:
            epoch_frame_train = train_frame_loss / n_train if n_train > 0 else float("nan")
            epoch_frame_val = val_frame_loss / n_val if n_val > 0 else float("nan")
            frame_epoch_losses.append((epoch + 1, epoch_frame_train, epoch_frame_val))
            # per-clip (unweighted class BCE) train + val, and the COMBINED objective train + val.
            epoch_clip_train = train_clip_loss / n_train if n_train > 0 else float("nan")
            clip_epoch_losses.append((epoch + 1, epoch_clip_train, epoch_val_loss))
            combined_val = (perframe_class_loss_weight * epoch_val_loss
                            + perframe_loss_weight * epoch_frame_val)
            combined_epoch_losses.append((epoch + 1, epoch_train_loss, combined_val))
        current_lr = scheduler.get_last_lr()[0]
        epoch_bar.set_postfix(
            epoch=epoch + 1,
            train_loss=f"{epoch_train_loss:.4f}",
            val_loss=f"{epoch_val_loss:.4f}",
            lr=f"{current_lr:.2e}",
        )

        wandb_log = {
            f"{run_label}/train_loss": epoch_train_loss,
            f"{run_label}/val_loss": epoch_val_loss,
            f"{run_label}/lr": current_lr,
            f"{run_label}/epoch": epoch + 1,
            "denoising_step": step,
        }
        if per_layer:
            wandb_log["block_index"] = block_index
        if class_head:
            wandb_log[f"{run_label}/train_class_loss"] = epoch_train_class_ce
            wandb_log[f"{run_label}/val_class_loss"] = epoch_class_loss
            wandb_log[f"{run_label}/val_class_acc"] = epoch_class_acc
            wandb_log[f"{run_label}/val_class_top5"] = epoch_class_top5
            wandb_log[f"{run_label}/train_class_acc"] = epoch_train_class_acc
            # Flush the class curve to disk EVERY epoch (live monitoring; no wait for results.json).
            class_epoch_rows.append({
                "epoch": epoch + 1,
                "train_class_ce": epoch_train_class_ce,
                "train_class_acc": epoch_train_class_acc,
                "val_class_ce": epoch_class_loss,
                "val_class_top1": epoch_class_acc,
                "val_class_top5": epoch_class_top5,
                "lr": current_lr,
            })
            pd.DataFrame(class_epoch_rows).to_csv(
                str(output_dir / f"class_metrics_{run_label}.csv"), index=False)
        if attack is not None and n_train > 0:
            # One row per epoch, flushed to disk every epoch so a killed run keeps its
            # history: the clean/adversarial loss gap is the primary convergence signal.
            adv_train_rows.append({
                "epoch": epoch + 1,
                "train_clean_loss": train_clean_loss / n_train,
                "train_adv_loss": (train_adv_loss / n_adv) if n_adv else float("nan"),
                "train_total_loss": epoch_train_loss,
                "val_clean_loss": epoch_val_loss,
                "frac_attacked": n_adv / n_train,
                "attack_active": bool(use_adv),
            })
            pd.DataFrame(adv_train_rows).to_csv(
                str(output_dir / f"adv_train_{run_label}.csv"), index=False)
            wandb_log[f"{run_label}/train_clean_loss"] = train_clean_loss / n_train
            if n_adv:
                wandb_log[f"{run_label}/train_adv_loss"] = train_adv_loss / n_adv
                wandb_log[f"{run_label}/adv_frac_attacked"] = n_adv / n_train

        if epoch == 0 or (epoch + 1) % eval_every_n_epochs == 0:
            train_mid = evaluate_probe(model, train_eval_loader, device, step)
            val_mid = evaluate_probe(model, val_loader, device, step)
            val_eval = val_mid  # reuse for best-metric monitoring below
            train_best = train_mid["pr_table"].loc[train_mid["pr_table"]["f1"].idxmax()]
            val_best = val_mid["pr_table"].loc[val_mid["pr_table"]["f1"].idxmax()]
            val_mid["pr_table"].to_csv(str(output_dir / f"eval_{run_label}_epoch_{epoch + 1:03d}.csv"), index=False)
            _dump_epoch_preds(
                train_ds, train_mid["probs"], val_ds, val_mid["probs"],
                output_dir / f"epoch_preds_{run_label}_epoch_{epoch + 1:03d}.csv",
            )
            train_epoch_metrics.append((epoch + 1, float(train_best["precision"]), float(train_best["recall"]), float(train_best["f1"])))
            val_epoch_metrics.append((epoch + 1, float(val_best["precision"]), float(val_best["recall"]), float(val_best["f1"])))
            wandb_log.update({
                f"{run_label}/train_precision": float(train_best["precision"]),
                f"{run_label}/train_recall": float(train_best["recall"]),
                f"{run_label}/train_f1": float(train_best["f1"]),
                f"{run_label}/train_auroc": float(train_mid["auroc"]),
                f"{run_label}/val_precision": float(val_best["precision"]),
                f"{run_label}/val_recall": float(val_best["recall"]),
                f"{run_label}/val_f1": float(val_best["f1"]),
                f"{run_label}/val_balanced_accuracy": float(val_best["balanced_accuracy"]),
                f"{run_label}/val_auroc": float(val_mid["auroc"]),
                f"{run_label}/val_average_precision": float(val_mid["average_precision"]),
                f"{run_label}/val_baseline_f1": float(val_mid["baseline_f1"]),
                f"{run_label}/pos_base_rate": float(val_mid["pos_base_rate"]),
                f"{run_label}/best_threshold": float(val_best["threshold"]),
            })
            if perframe_presence_head:
                pf_eval = evaluate_perframe(model, val_loader, device, pf_T, pf_off)
                pf_eval["per_frame"].to_csv(
                    str(output_dir / f"perframe_eval_{run_label}_epoch_{epoch + 1:03d}.csv"),
                    index=False,
                )
                last_perframe_eval = pf_eval
                wandb_log.update({
                    f"{run_label}/val_frame_auroc": float(pf_eval["frame_auroc"]),
                    f"{run_label}/val_frame_average_precision": float(pf_eval["frame_average_precision"]),
                    f"{run_label}/val_frame_balanced_accuracy": float(pf_eval["frame_balanced_accuracy"]),
                    f"{run_label}/val_allframes_auroc": float(pf_eval.get("allframes_auroc", float("nan"))),
                    f"{run_label}/val_anyframe_auroc": float(pf_eval.get("anyframe_auroc", float("nan"))),
                })
            if class_head:
                # Per-class breakdown (support + top-1) at this eval epoch, plus macro/top-1/top-5.
                ce_eval = evaluate_class(model, val_loader, device, class_off,
                                         len(class_vocab), class_vocab=class_vocab)
                ce_eval["per_class"].to_csv(
                    str(output_dir / f"class_eval_{run_label}_epoch_{epoch + 1:03d}.csv"), index=False)
                wandb_log.update({
                    f"{run_label}/val_class_macro_acc": float(ce_eval["class_macro_acc"]),
                    f"{run_label}/val_class_top1_eval": float(ce_eval["class_top1_acc"]),
                    f"{run_label}/val_class_top5_eval": float(ce_eval["class_top5_acc"]),
                })
            if attack is not None and adv_cfg.eval_robust:
                # How exploitable is the probe right now? adv_auroc under the same threat
                # model, plus the flip / negative-spike rates that mirror the inference
                # failure (score spikes with no object). Everything is written to disk each
                # eval epoch: the scalar row, the per-example clean/adv scores + map stats,
                # and the PGD trace — so the post-hoc plots never need a re-run.
                rb_by_split = run_robust(epoch + 1)
                for split, rb in rb_by_split.items():
                    rb_scalars = {k: v for k, v in rb.items()
                                  if k not in ("per_example", "trace", "examples")}
                    adv_eval_rows.append({"epoch": epoch + 1, "split": split, **rb_scalars})
                    wandb_log.update({f"{run_label}/robust_{split}_{k}": float(v)
                                      for k, v in rb_scalars.items()})
                pd.DataFrame(adv_eval_rows).to_csv(
                    str(output_dir / f"adv_metrics_{run_label}.csv"), index=False)
                rb_val = rb_by_split.get("val", {})
                if "per_example" in rb_val:
                    rb_val["per_example"].to_csv(
                        str(output_dir / f"adv_preds_{run_label}_epoch_{epoch + 1:03d}.csv"),
                        index=False)
                if rb_val.get("trace"):
                    adv_traces.append((epoch + 1, rb_val["trace"]))
                    pd.DataFrame([{"epoch": e, "pgd_iter": i, "p_flipped": p}
                                  for e, tr in adv_traces for i, p in enumerate(tr)]).to_csv(
                        str(output_dir / f"adv_pgd_trace_{run_label}.csv"), index=False)

        if epoch == 0 or (epoch + 1) % plot_every_n_epochs == 0:
            train_plot = evaluate_probe(model, train_eval_loader, device, step)
            val_plot = evaluate_probe(model, val_loader, device, step)
            val_plot_best = val_plot["pr_table"].loc[val_plot["pr_table"]["f1"].idxmax()]
            ep_suffix = f"_epoch_{epoch + 1:03d}"
            fp_fn_path = output_dir / f"fp_fn_{run_label}{ep_suffix}.png"
            plot_fp_fn(
                train_plot["probs"], train_plot["labels_binary"],
                val_plot["probs"], val_plot["labels_binary"],
                step=step,
                threshold=float(val_plot_best["threshold"]),
                output_path=fp_fn_path,
            )
            if wandb_run is not None:
                plot_log = {f"{run_label}/epoch": epoch + 1}
                if fp_fn_path.exists():
                    plot_log[f"{run_label}/fp_fn_plot"] = wandb.Image(str(fp_fn_path))
                wandb_run.log(plot_log)

        # --- best-epoch selection + early stopping ---
        epochs_trained = epoch + 1
        if best_metric == "val_loss":
            monitor_value = epoch_val_loss
        elif best_metric == "val_class_loss":
            monitor_value = epoch_class_loss
        elif best_metric == "val_class_acc":
            monitor_value = epoch_class_acc
        elif best_metric in ADV_METRICS:
            # Attack the val split EVERY epoch so selection has per-epoch granularity (the
            # reporting pass only runs on eval epochs). adv_eval_val_max caps it, since this
            # costs (1 + adv_pgd_steps) forwards over whatever it covers.
            sel_max = (max(1, -(-adv_cfg.eval_val_max // batch_size))
                       if adv_cfg.eval_val_max else None)
            rb_sel = evaluate_adversarial(
                model, val_loader, device, attack, forward=adv_forward, max_batches=sel_max,
                perframe=((pf_off, pf_T) if perframe_presence_head else None))
            if not rb_sel:
                monitor_value = float("nan")
            elif best_metric == "val_adv_loss":
                monitor_value = float(rb_sel["adv_bce"])
            elif best_metric == "val_frame_combined":
                monitor_value = 0.5 * (float(rb_sel["frame_clean_bce"])
                                       + float(rb_sel["frame_adv_bce"]))
            elif best_metric == "val_frame_adv_auroc":
                monitor_value = float(rb_sel["frame_adv_auroc"])
            else:  # val_combined — both losses low is the actual objective
                monitor_value = 0.5 * (float(rb_sel["clean_bce"]) + float(rb_sel["adv_bce"]))
            wandb_log[f"{run_label}/sel_clean_bce"] = float(rb_sel.get("clean_bce", float("nan")))
            wandb_log[f"{run_label}/sel_adv_bce"] = float(rb_sel.get("adv_bce", float("nan")))
        else:
            if val_eval is None:  # not an eval epoch — run a val pass just for monitoring
                val_eval = evaluate_probe(model, val_loader, device, step)
            if best_metric == "val_auroc":
                monitor_value = float(val_eval["auroc"])
            else:  # val_f1
                monitor_value = float(val_eval["pr_table"]["f1"].max())

        is_nan = monitor_value != monitor_value
        improved = (not is_nan) and (
            monitor_value > best_value if higher_better else monitor_value < best_value
        )
        if improved:
            best_value = monitor_value
            best_epoch = epoch + 1
            best_val_loss = epoch_val_loss
            best_train_loss = epoch_train_loss
            best_lr = current_lr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        elif not is_nan:
            epochs_since_improve += 1
        wandb_log[f"{run_label}/{best_metric}_monitor"] = monitor_value
        wandb_log[f"{run_label}/{best_metric}_best"] = best_value

        if wandb_run is not None:
            wandb_run.log(wandb_log)

        if early_stop_patience > 0 and epochs_since_improve >= early_stop_patience:
            early_stopped = True
            print(f"  [{run_label}] early stop at epoch {epoch + 1}: no {best_metric} "
                  f"improvement for {early_stop_patience} epoch(s) (best={best_value:.4f} @ epoch {best_epoch})")
            break

    # Fall back to final weights if no epoch ever improved (e.g. all-NaN monitor).
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epochs_trained
        best_val_loss = epoch_val_loss

    plot_epoch_summary(
        epoch_losses, step,
        output_dir / f"epoch_summary_{run_label}.png",
    )
    plot_pr_curves(
        train_epoch_metrics, val_epoch_metrics, step,
        output_dir / f"pr_curves_{run_label}.png",
    )

    # Shared architecture/metadata; model_state filled per checkpoint below.
    ckpt_meta = {
        "probe_model": probe_model,
        "avg_mode": effective_avg_mode,
        "avg_channels": effective_avg_mode == "all",  # legacy key for external loaders
        "step": step,
        "block_index": block_index,
        "n_blocks_heads": probe_n_blocks_heads,
        "latent_thw": latent_thw,
        "bottleneck_ch": bottleneck_ch,
        "conv_kernel_size": conv_kernel_size,
        "mlp_hidden": mlp_hidden,
        "mlp_hidden_dims": mlp_hidden_dims,
        "n_blocks": probe_n_blocks,
        "n_heads": n_heads,
        "attn_pool_conv_ch": attn_pool_conv_ch,
        "attn_pool_mlp_hidden": attn_pool_mlp_hidden,
        "resnet_base_ch": resnet_base_ch,
        "resnet_n_blocks": resnet_n_blocks,
        "resnet_groups": resnet_groups,
        "resnet_temporal_spatial": resnet_temporal_spatial,
        "resnet_head_dropout": resnet_head_dropout,
        "resnet_dropblock_size": resnet_dropblock_size,
        "resnet_norm": resnet_norm,
        "resnet_layer_agg": resnet_layer_agg,
        "resnet_layer_agg_k": resnet_layer_agg_k,
        "resnet_layer_agg_hidden": resnet_layer_agg_hidden,
        "resnet_input_norm": resnet_input_norm,
        "layer_subset": layer_subset,
        "resnet_step_cond": resnet_step_cond,
        "resnet_step_embed_dim": resnet_step_embed_dim,
        "resnet_multitask": multitask_aesthetic,
        "aesthetic_col": aesthetic_col if multitask_aesthetic else None,
        "resnet_perframe": perframe_presence_head,
        "resnet_perframe_head": perframe_head,
        "resnet_perframe_tap": perframe_tap,
        "resnet_classhead": class_head,
        "resnet_n_classes": (len(class_vocab) if class_head else 0),
        "resnet_class_head_arch": class_head_arch if class_head else None,
        "resnet_class_mlp_hidden": class_mlp_hidden if class_head else None,
        "class_vocab": class_vocab if class_head else None,
        "class_loss_balance": class_loss_balance if class_head else None,
        "perframe_frames": pf_T if perframe_presence_head else None,
        "perframe_target_mode": perframe_target_mode if perframe_presence_head else None,
        "perframe_col": perframe_col if perframe_presence_head else None,
        "transformer_d_model": transformer_d_model,
        "transformer_nhead": transformer_nhead,
        "transformer_num_layers": transformer_num_layers,
        "transformer_dim_feedforward": transformer_dim_feedforward,
        "transformer_head_attn_nheads": transformer_head_attn_nheads,
        "transformer_layer_attn_nheads": transformer_layer_attn_nheads,
        "transformer_token_hidden": transformer_token_hidden,
        "transformer_token_mode": transformer_token_mode,
        "transformer_token_conv_ch": transformer_token_conv_ch,
        "transformer_mlp_hidden": transformer_mlp_hidden,
        "transformer_pool": transformer_pool,
        "transformer_timestep_mode": transformer_timestep_mode,
        "transformer_num_timesteps": transformer_num_timesteps,
        "adv_training": adv_cfg.enabled,
        "adv": adv_cfg.to_dict() if adv_cfg.enabled else None,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "best_value": best_value,
        "best_val_loss": best_val_loss,
        "epochs_trained": epochs_trained,
        "early_stopped": early_stopped,
    }
    # Final-epoch checkpoint (for comparison/resume). CPU state for portability.
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    last_ckpt_path = output_dir / f"probe_{run_label}_last.pt"
    torch.save({**ckpt_meta, "model_state": last_state, "saved_epoch": epochs_trained},
               str(last_ckpt_path))
    # Best-epoch checkpoint (the canonical one).
    ckpt_path = output_dir / f"probe_{run_label}.pt"
    torch.save({**ckpt_meta, "model_state": best_state, "saved_epoch": best_epoch},
               str(ckpt_path))

    # Restore best-epoch weights so the reported results below reflect that epoch.
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    train_final = evaluate_probe(model, train_eval_loader, device, step)
    metrics = evaluate_probe(model, val_loader, device, step, dataset=val_ds)
    metrics["epoch_losses"] = epoch_losses
    metrics["aux_epoch_losses"] = aux_epoch_losses
    metrics["frame_epoch_losses"] = frame_epoch_losses
    metrics["class_epoch_losses"] = class_epoch_losses
    metrics["clip_epoch_losses"] = clip_epoch_losses
    metrics["combined_epoch_losses"] = combined_epoch_losses
    if class_head:
        # Re-evaluate the class head at the restored best-epoch weights (per-epoch curves above
        # are mid-training snapshots). Writes per-class support/accuracy and reports top-1/top-5.
        cls_final = evaluate_class(model, val_loader, device, class_off, len(class_vocab),
                                   class_vocab=class_vocab)
        cls_final["per_class"].to_csv(
            str(output_dir / f"class_eval_{run_label}.csv"), index=False)
        # Per-sample (true, pred) dump → feeds the confusion matrix + per-class analysis offline.
        _inv = {i: n for n, i in class_vocab.items()}
        _yt, _yp = cls_final.get("y_true"), cls_final.get("y_pred")
        if _yt is not None and len(_yt):
            pd.DataFrame({
                "true_idx": _yt, "pred_idx": _yp,
                "true_name": [_inv.get(int(i), "") for i in _yt],
                "pred_name": [_inv.get(int(i), "") for i in _yp],
            }).to_csv(str(output_dir / f"class_preds_{run_label}.csv"), index=False)
        metrics["classhead"] = {k: v for k, v in cls_final.items()
                                if k not in ("per_class", "y_true", "y_pred")}
        print(f"  → class head  top1={cls_final['class_top1_acc']:.3f}  "
              f"top5={cls_final['class_top5_acc']:.3f}  "
              f"macro-acc={cls_final['class_macro_acc']:.3f}  "
              f"macro-F1={cls_final.get('class_macro_f1', float('nan')):.3f}  "
              f"[{len(class_vocab)} classes, n={cls_final['n_eval']}]")
    if perframe_presence_head:
        # Re-evaluate at the restored best-epoch weights so the reported per-frame numbers
        # match the saved checkpoint (the per-epoch CSVs above are mid-training snapshots).
        pf_final = evaluate_perframe(model, val_loader, device, pf_T, pf_off)
        pf_final["per_frame"].to_csv(
            str(output_dir / f"perframe_eval_{run_label}.csv"), index=False)
        metrics["perframe"] = {k: v for k, v in pf_final.items() if k != "per_frame"}
        if perframe_init_eval is not None:
            metrics["perframe_init"] = {k: v for k, v in perframe_init_eval.items()
                                        if k != "per_frame"}
        print(f"  → per-frame  AUROC={pf_final['frame_auroc']:.3f}  "
              f"AP={pf_final['frame_average_precision']:.3f}  "
              f"BalAcc={pf_final['frame_balanced_accuracy']:.3f}  "
              f"[pos={pf_final['frame_pos_base_rate']:.0%}, "
              f"valid cells={pf_final['valid_cell_frac']:.1%}]")
        print(f"  → all-frames (min_t) AUROC={pf_final.get('allframes_auroc', float('nan')):.3f} "
              f"[pos={pf_final.get('allframes_pos_base_rate', float('nan')):.0%}]  |  "
              f"any-frame (max_t) AUROC={pf_final.get('anyframe_auroc', float('nan')):.3f}")
    if attack is not None and adv_cfg.eval_robust:
        # Robustness of the SAVED (best-epoch) weights, under the same threat model it was
        # trained against. adv_auroc is the headline; flip_rate / neg_spike_rate say how
        # often a small legal attention edit still buys a wrong answer. This pass also
        # keeps the adversarial MAPS themselves for the saved examples file.
        final_by_split = run_robust(
            best_epoch,
            n_examples_kept=adv_cfg.save_examples,
            save_maps=adv_cfg.save_maps,
            example_seed=adv_cfg.example_seed,
        )
        rb_final = final_by_split.get("val", {})
        for split, rb in final_by_split.items():
            scalars = {k: v for k, v in rb.items()
                       if k not in ("per_example", "trace", "examples")}
            metrics["robust" if split == "val" else f"robust_{split}"] = scalars
            if "per_example" in rb:
                suffix = "" if split == "val" else f"_{split}"
                rb["per_example"].to_csv(
                    str(output_dir / f"adv_preds{suffix}_{run_label}.csv"), index=False)
            if split != "val" and scalars:
                print(f"  → robust[{split}]  AUROC={scalars['adv_auroc']:.3f} "
                      f"(clean {scalars['clean_auroc']:.3f})  "
                      f"flip={scalars['flip_rate']:.1%}  "
                      f"neg-spike={scalars['neg_spike_rate']:.1%}  [n={scalars['n_examples']}]")
        if rb_final:
            per_ex = rb_final["per_example"]
            adv_eval_df = pd.DataFrame(adv_eval_rows)
            plot_adv_curves(pd.DataFrame(adv_train_rows), adv_eval_df, per_ex, adv_traces,
                            step, output_dir / f"adv_curves_{run_label}.png")
            plot_adv_diagnostics(per_ex, adv_eval_df, step,
                                 output_dir / f"adv_diagnostics_{run_label}.png",
                                 stat_names=STAT_NAMES)
            if "examples" in rb_final:
                save_adv_examples(output_dir / f"adv_examples_{run_label}.pt",
                                  rb_final["examples"], adv_cfg, step, best_epoch)
                plot_adv_examples(rb_final["examples"], step,
                                  output_dir / f"adv_examples_{run_label}.png")
            print(f"  → robust[val]  AUROC={rb_final['adv_auroc']:.3f} (clean {rb_final['clean_auroc']:.3f})  "
                  f"BalAcc={rb_final['adv_balanced_accuracy']:.3f}  "
                  f"flip={rb_final['flip_rate']:.1%}  neg-spike={rb_final['neg_spike_rate']:.1%}  "
                  f"|Δscore|={rb_final['mean_score_shift']:.3f}")
            print(f"  → adv-vs-real drift (KS): " + "  ".join(
                f"{k}={rb_final[f'ks_{k}']:.3f}" for k in STAT_NAMES))
    metrics["best_epoch"] = best_epoch
    metrics["best_val_loss"] = best_val_loss
    metrics["monitor_metric"] = best_metric
    metrics["monitor_value"] = best_value
    metrics["early_stopped"] = early_stopped
    metrics["epochs_trained"] = epochs_trained
    print(f"  [{run_label}] best epoch {best_epoch}/{epochs_trained} "
          f"({best_metric}={best_value:.4f}, val_loss={best_val_loss:.4f}"
          f"{', early-stopped' if early_stopped else ''})")
    best = metrics["pr_table"].loc[metrics["pr_table"]["f1"].idxmax()]
    print(
        f"  → val  prec={best['precision']:.3f}  rec={best['recall']:.3f}  "
        f"f1={best['f1']:.3f}  (thr={best['threshold']})  |  "
        f"AUROC={metrics['auroc']:.3f}  AP={metrics['average_precision']:.3f}  "
        f"BalAcc={best['balanced_accuracy']:.3f}  "
        f"[pos={metrics['pos_base_rate']:.0%}, all-pos F1={metrics['baseline_f1']:.3f}]"
    )
    plot_fp_fn(
        train_final["probs"], train_final["labels_binary"],
        metrics["probs"], metrics["labels_binary"],
        step=step,
        threshold=float(best["threshold"]),
        output_path=output_dir / f"fp_fn_{run_label}.png",
    )
    if "predictions" in metrics:
        metrics["predictions"].to_csv(str(output_dir / f"predictions_{run_label}.csv"), index=False)

    if wandb_run is not None:
        log = {f"{run_label}/val_{k}": float(best[k]) for k in ("precision", "recall", "f1", "balanced_accuracy")}
        log[f"{run_label}/best_threshold"] = float(best["threshold"])
        log[f"{run_label}/val_auroc"] = float(metrics["auroc"])
        log[f"{run_label}/val_average_precision"] = float(metrics["average_precision"])
        log[f"{run_label}/val_baseline_f1"] = float(metrics["baseline_f1"])
        log[f"{run_label}/pos_base_rate"] = float(metrics["pos_base_rate"])
        log["denoising_step"] = step
        if per_layer:
            log["block_index"] = block_index
        for path_key, path_val in [
            ("fp_fn_plot", output_dir / f"fp_fn_{run_label}.png"),
            ("epoch_summary", output_dir / f"epoch_summary_{run_label}.png"),
            ("pr_curves", output_dir / f"pr_curves_{run_label}.png"),
            ("adv_curves", output_dir / f"adv_curves_{run_label}.png"),
            ("adv_diagnostics", output_dir / f"adv_diagnostics_{run_label}.png"),
            ("adv_examples", output_dir / f"adv_examples_{run_label}.png"),
        ]:
            if path_val.exists():
                log[f"{run_label}/{path_key}"] = wandb.Image(str(path_val))
        wandb_run.log(log)

    if return_datasets:
        # Hand the preloaded datasets back so a load-once/train-many driver can pass them
        # to the next config via preloaded_datasets=. Not serialized by write_task_result.
        metrics["_datasets"] = (train_ds, val_ds)
    return metrics


def train_all_steps(
    train_labels_csv: str | Path,
    val_labels_csv: str | Path,
    output_dir: str | Path,
    device: Optional[torch.device] = None,
    wandb_project: str = "video_probing",
    wandb_name: Optional[str] = None,
    wandb_notes: Optional[str] = None,
    disable_wandb: bool = False,
    cfg_yaml: Optional[str] = None,
    extra_cfg_paths: Optional[list[Path]] = None,
    steps_to_run: Optional[list[int]] = None,
    per_layer: bool = False,
    layer_indices: Optional[list[int]] = None,
    train_root_override: Optional[str | Path] = None,
    val_root_override: Optional[str | Path] = None,
    **kwargs,
) -> None:
    """Train one probe per (step, block) work item in a single process.

    Multi-GPU distribution goes through the filesystem queue instead — see
    training/queue_trainer.py.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run = wandb.init(
        project=wandb_project,
        name=wandb_name,
        notes=wandb_notes,
        config={
            "train_labels_csv": str(train_labels_csv),
            "val_labels_csv": str(val_labels_csv),
            "output_dir": str(output_dir),
            **{k: v for k, v in kwargs.items()},
        },
    ) if not disable_wandb else None

    run_dir = Path(output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    details_dir = run_dir / "run_details"
    details_dir.mkdir()
    (details_dir / "cmd.txt").write_text(" ".join(sys.argv) + "\n")
    if cfg_yaml is not None:
        (details_dir / "resolved_config.yaml").write_text(cfg_yaml)
    for p in extra_cfg_paths or []:
        p = Path(p)
        if p.exists():
            shutil.copy(p, details_dir / p.name)

    all_steps = steps_to_run if steps_to_run is not None else _get_available_steps(train_labels_csv, root_override=train_root_override)
    resolved_block_indices = resolve_block_indices(train_labels_csv, train_root_override, per_layer, layer_indices)
    if per_layer:
        print(f"Training {len(all_steps)} denoising steps × {len(resolved_block_indices)} blocks {resolved_block_indices} (per_layer=True)")
    else:
        print(f"Training {len(all_steps)} denoising steps: {all_steps}")

    work_items = [(s, b) for s in all_steps for b in resolved_block_indices]

    for step, block_idx in tqdm(work_items, desc="tasks", unit="model"):
        step_dir = step_output_dir(run_dir, step, block_idx)
        step_dir.mkdir(parents=True, exist_ok=True)
        metrics = train_one_step(
            step=step,
            train_labels_csv=train_labels_csv,
            val_labels_csv=val_labels_csv,
            output_dir=step_dir,
            device=device,
            wandb_run=run,
            block_index=block_idx,
            train_root_override=train_root_override,
            val_root_override=val_root_override,
            **kwargs,
        )
        write_task_result(step_dir, step, block_idx, metrics)

    aggregate_run(run_dir, wandb_run=run, metric=select_metric_for(kwargs.get("best_metric")))
    if run is not None:
        run.finish()
    print(f"Done. Results saved to {run_dir}")


def main(cfg: DictConfig) -> None:
    t = cfg.training
    train_csv = t.train_csv
    val_csv = t.val_csv
    output_dir = t.output_dir

    kwargs = build_training_kwargs(t)

    train_root_override = t.get("train_root_override", None)
    val_root_override = t.get("val_root_override", None)
    steps_to_run = resolve_steps_to_run(t, train_csv, train_root_override)

    original_cwd = Path(hydra.utils.get_original_cwd())
    model_type = str(cfg.get("model_type", "ltx"))
    extra_cfg_paths = [
        original_cwd / "configs" / "probing.yaml",
        original_cwd / "configs" / "models" / f"{model_type}.yaml",
    ]

    per_layer = bool(t.get("per_layer", False))
    layer_indices_raw = t.get("layer_indices", None)
    layer_indices = list(layer_indices_raw) if layer_indices_raw is not None else None

    train_all_steps(
        train_labels_csv=train_csv,
        val_labels_csv=val_csv,
        output_dir=output_dir,
        steps_to_run=steps_to_run,
        wandb_project=t.get("wandb_project", "video_probing"),
        wandb_name=t.get("wandb_name", None),
        wandb_notes=t.get("wandb_notes", None),
        disable_wandb=bool(t.get("disable_wandb", False)),
        cfg_yaml=OmegaConf.to_yaml(cfg),
        extra_cfg_paths=extra_cfg_paths,
        per_layer=per_layer,
        layer_indices=layer_indices,
        train_root_override=train_root_override,
        val_root_override=val_root_override,
        **kwargs,
    )


if __name__ == "__main__":
    if hydra is None:
        raise ModuleNotFoundError(
            "hydra is required for the training CLI (python -m training.trainer). "
            "Install hydra-core, or use the steering path which does not need it.")
    # Apply the hydra CLI decorator only when actually running as the trainer CLI, so
    # importing build_attnsteer (e.g. from the steering package) never requires hydra.
    hydra.main(config_path="../configs", config_name="probing", version_base=None)(main)()
