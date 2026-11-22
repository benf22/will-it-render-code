"""Single end-of-run summariser for a probe training run.

Re-derives the run-level summary from the per-epoch prediction dumps
(`epoch_preds_{run_label}_epoch_{NNN}.csv`) that `trainer.py` writes at every eval
epoch. It runs automatically at the end of training (called by `aggregate_run` in
`trainer.py`) **and** can be re-run later over any past run dir via the CLI.

For each (denoising step, block) it:

  1. Picks the *best epoch* on the validation split (by `metric`, default AUROC)
     instead of the last epoch, and recomputes all metrics there.
  2. Writes the main per-step table        → `training_results.csv`.
  3. Writes a table segmented by the number of concepts in the prompt (= distinct
     `object_name` per `extraction_dir`): rows = step, columns = (metric × prompt
     length) for balanced_accuracy / auroc / average_precision
                                            → `training_results_by_nobj.csv`.
  4. Re-draws `training_summary.png` as two subplots:
       a. per step, one line per metric (balanced_accuracy, auroc, average_precision)
       b. per step, one line per prompt length, coloured by metric.

CLI (post-hoc over a past run):
    python -m training.summarize_run <run_dir> [--metric auroc]

`<run_dir>` is a timestamped run directory containing `step_*/` subdirs (optionally
`step_*/block_*/`). The per-epoch dumps must be present — runs that predate them
(no `epoch_preds_*.csv`) cannot be summarised this way.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Balance-aware, threshold-free metric helpers (shared with the trainer so the
# numbers match what training logs).
from will_it_render.training.evaluation import auroc, average_precision, baseline_f1

# Metrics reported in the segmented table / subplot (b), in display order.
SEG_METRICS = ["balanced_accuracy", "auroc", "average_precision"]
# Metrics that can drive best-epoch selection. All but val_loss are higher-better.
SELECT_METRICS = ["auroc", "average_precision", "f1", "balanced_accuracy", "val_loss"]


def select_metric_for(best_metric: str | None) -> str:
    """Map the trainer's `best_metric` config to a dump-computable selection metric.

    `val_loss` is computed from the dumped probabilities; `val_auroc`/`val_f1` map to
    their threshold-free / best-F1 equivalents. Anything else falls back to AUROC.
    """
    return {
        "val_loss": "val_loss",
        "val_auroc": "auroc",
        "val_f1": "f1",
    }.get(str(best_metric), "auroc")


# ---------------------------------------------------------------------------
# Run-dir discovery
# ---------------------------------------------------------------------------


def _run_label(step: int, block: int | None) -> str:
    return f"step_{step:03d}_block_{block:02d}" if block is not None else f"step_{step:03d}"


def _epoch_preds_for(dir_: Path, label: str) -> dict[int, Path]:
    """{epoch: path} for every epoch_preds dump in `dir_` for this run label."""
    out: dict[int, Path] = {}
    for p in dir_.glob(f"epoch_preds_{label}_epoch_*.csv"):
        try:
            out[int(p.stem.split("_epoch_")[-1])] = p
        except ValueError:
            pass
    return out


def discover_items(run_dir: Path) -> list[tuple[int, int | None, dict[int, Path]]]:
    """List (step, block, {epoch: preds_path}) for every trained item with dumps."""
    items: list[tuple[int, int | None, dict[int, Path]]] = []
    for sd in sorted(run_dir.glob("step_*")):
        if not sd.is_dir():
            continue
        try:
            step = int(sd.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        block_dirs = sorted(sd.glob("block_*"))
        if block_dirs:
            for bd in block_dirs:
                try:
                    block = int(bd.name.split("_")[1])
                except (IndexError, ValueError):
                    continue
                preds = _epoch_preds_for(bd, _run_label(step, block))
                if preds:
                    items.append((step, block, preds))
        else:
            preds = _epoch_preds_for(sd, _run_label(step, None))
            if preds:
                items.append((step, None, preds))
    return items


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _add_n_objects(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each row with n_objects = distinct object_name sharing its extraction_dir."""
    df = df.copy()
    df["n_objects"] = df.groupby("extraction_dir")["object_name"].transform("nunique").astype(int)
    return df


def _best_threshold(probs: np.ndarray, gtb: np.ndarray) -> float:
    """Threshold in [0.05, 0.95] that maximises F1 (matches the trainer's sweep)."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        preds = (probs >= t).astype(int)
        tp = int(((preds == 1) & (gtb == 1)).sum())
        fp = int(((preds == 1) & (gtb == 0)).sum())
        fn = int(((preds == 0) & (gtb == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def _threshold_metrics(probs: np.ndarray, gtb: np.ndarray, thr: float) -> dict:
    """Precision / recall / f1 / balanced_accuracy at a fixed threshold."""
    preds = (probs >= thr).astype(int)
    tp = int(((preds == 1) & (gtb == 1)).sum())
    fp = int(((preds == 1) & (gtb == 0)).sum())
    fn = int(((preds == 0) & (gtb == 1)).sum())
    tn = int(((preds == 0) & (gtb == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0  # TPR
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "balanced_accuracy": (rec + tnr) / 2}


def _bce(probs: np.ndarray, gtb: np.ndarray) -> float:
    """Binary cross-entropy from probabilities (matches the trainer's unweighted val loss)."""
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    return float(-(gtb * np.log(p) + (1 - gtb) * np.log(1 - p)).mean())


def full_metrics(df_val: pd.DataFrame, thr: float | None = None) -> dict:
    """All reported metrics on a val slice. Threshold defaults to the best-F1 sweep."""
    probs = df_val["prob"].to_numpy(dtype=float)
    gtb = (df_val["gt"].to_numpy(dtype=float) >= 0.5).astype(int)
    if len(gtb) == 0:
        return {}
    if thr is None:
        thr = _best_threshold(probs, gtb)
    pos_rate = float(gtb.mean())
    out = {
        "best_threshold": thr,
        "auroc": auroc(gtb, probs),
        "average_precision": average_precision(gtb, probs),
        "baseline_f1": baseline_f1(pos_rate),
        "pos_base_rate": pos_rate,
        "n_val": len(gtb),
    }
    out.update(_threshold_metrics(probs, gtb, thr))
    return out


def epoch_score(df_val: pd.DataFrame, metric: str) -> tuple[float, bool]:
    """(value, higher_better) for one epoch's val predictions under `metric`."""
    probs = df_val["prob"].to_numpy(dtype=float)
    gtb = (df_val["gt"].to_numpy(dtype=float) >= 0.5).astype(int)
    if len(gtb) == 0:
        return float("nan"), True
    if metric == "auroc":
        return auroc(gtb, probs), True
    if metric == "average_precision":
        return average_precision(gtb, probs), True
    if metric == "val_loss":
        return _bce(probs, gtb), False
    # f1 / balanced_accuracy at the best-F1 threshold
    return _threshold_metrics(probs, gtb, _best_threshold(probs, gtb))[metric], True


def segment_metrics(df_val: pd.DataFrame, thr: float) -> dict[int, dict]:
    """Per-n_objects {n_obj: {balanced_accuracy, auroc, average_precision}}.

    AUROC / AP are threshold-free; balanced_accuracy uses the run-level `thr` so
    every prompt length is read at the same operating point.
    """
    out: dict[int, dict] = {}
    for n_obj, slice_df in df_val.groupby("n_objects"):
        probs = slice_df["prob"].to_numpy(dtype=float)
        gtb = (slice_df["gt"].to_numpy(dtype=float) >= 0.5).astype(int)
        out[int(n_obj)] = {
            "balanced_accuracy": _threshold_metrics(probs, gtb, thr)["balanced_accuracy"],
            "auroc": auroc(gtb, probs),
            "average_precision": average_precision(gtb, probs),
        }
    return out


# ---------------------------------------------------------------------------
# Per-item summary
# ---------------------------------------------------------------------------


def summarize_item(step: int, block: int | None, preds: dict[int, Path], metric: str) -> dict:
    """Best-epoch main row + per-n_objects segment metrics for one (step, block)."""
    best_epoch, best_score, best_df, best_hb = None, None, None, True
    for epoch in sorted(preds):
        df = pd.read_csv(preds[epoch])
        score, hb = epoch_score(df[df["split"] == "val"], metric)
        if score != score:  # NaN
            continue
        if best_score is None or (score > best_score if hb else score < best_score):
            best_epoch, best_score, best_df, best_hb = epoch, score, df, hb

    if best_df is None:  # no usable epoch — fall back to the last available dump
        best_epoch = max(preds)
        best_df = pd.read_csv(preds[best_epoch])
        best_score = float("nan")

    # n_objects counts every probed concept in the clip (incl. masked rows); metrics
    # then exclude masked (-1) labels.
    df_val_all = _add_n_objects(best_df[best_df["split"] == "val"])
    df_val = df_val_all[df_val_all["gt"] >= 0]
    fm = full_metrics(df_val)
    row = {"step": step}
    if block is not None:
        row["block"] = block
    row.update({
        "best_epoch": best_epoch,
        f"select_{metric}": best_score,
        "best_threshold": fm["best_threshold"],
        "precision": fm["precision"], "recall": fm["recall"], "f1": fm["f1"],
        "balanced_accuracy": fm["balanced_accuracy"],
        "auroc": fm["auroc"], "average_precision": fm["average_precision"],
        "baseline_f1": fm["baseline_f1"], "pos_base_rate": fm["pos_base_rate"],
        "n_val": fm["n_val"],
    })
    return {"row": row, "block": block, "step": step,
            "segments": segment_metrics(df_val, fm["best_threshold"])}


# ---------------------------------------------------------------------------
# Tables + plot
# ---------------------------------------------------------------------------


def build_segment_table(summaries: list[dict], per_layer: bool) -> pd.DataFrame:
    """rows = step (or step/block); columns = MultiIndex (metric, n_objects)."""
    nobjs = sorted({n for s in summaries for n in s["segments"]})
    cols = pd.MultiIndex.from_product([SEG_METRICS, nobjs], names=["metric", "n_objects"])
    index_rows, data_rows = [], []
    for s in summaries:
        index_rows.append((s["step"], s["block"]) if per_layer else s["step"])
        data_rows.append([s["segments"].get(n, {}).get(m, np.nan)
                          for m in SEG_METRICS for n in nobjs])
    index = (pd.MultiIndex.from_tuples(index_rows, names=["step", "block"])
             if per_layer else pd.Index(index_rows, name="step"))
    return pd.DataFrame(data_rows, index=index, columns=cols)


def plot_summary(main_df: pd.DataFrame, seg_df: pd.DataFrame, per_layer: bool, output_path: Path) -> None:
    """Two stacked subplots — overall metrics, then metrics split by prompt length."""
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    if per_layer:
        labels = [_run_label(st, bl) for st, bl in zip(main_df["step"], main_df["block"])]
    else:
        labels = [str(s) for s in main_df["step"]]
    x = np.arange(len(main_df))
    metric_colors = {"balanced_accuracy": "seagreen", "auroc": "steelblue",
                     "average_precision": "tomato"}

    # (a) one line per metric, over steps.
    for m, c in metric_colors.items():
        ax_top.plot(x, main_df[m], marker="o", color=c, label=m)
    ax_top.set_ylabel("Score")
    ax_top.set_ylim(0, 1)
    ax_top.set_title("Best-epoch metrics across denoising steps")
    ax_top.legend()
    ax_top.grid(True, alpha=0.3)

    # (b) per prompt length, coloured by metric (line style cycles per length).
    nobjs = sorted({n for (_, n) in seg_df.columns})
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
    for m, c in metric_colors.items():
        for i, n in enumerate(nobjs):
            ax_bot.plot(x, seg_df[(m, n)].to_numpy(), color=c,
                        linestyle=linestyles[i % len(linestyles)],
                        marker="o", markersize=3, linewidth=1.3)
    ax_bot.set_ylabel("Score")
    ax_bot.set_ylim(0, 1)
    ax_bot.set_xlabel("step / block" if per_layer else "Denoising step")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(labels, rotation=45 if per_layer else 0,
                           ha="right" if per_layer else "center")
    ax_bot.set_title("Metrics by prompt length (colour = metric, line style = prompt length)")
    ax_bot.grid(True, alpha=0.3)
    from matplotlib.lines import Line2D
    color_handles = [Line2D([], [], color=c, label=m) for m, c in metric_colors.items()]
    style_handles = [Line2D([], [], color="gray", linestyle=linestyles[i % len(linestyles)],
                            label=f"{n} obj") for i, n in enumerate(nobjs)]
    leg1 = ax_bot.legend(handles=color_handles, loc="upper left", fontsize=8, title="metric")
    ax_bot.add_artist(leg1)
    ax_bot.legend(handles=style_handles, loc="upper right", fontsize=8, title="prompt length")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


_RESULTS_JSON_EXTRAS = (
    "adv_auroc", "adv_average_precision", "adv_balanced_accuracy",
    "flip_rate", "neg_spike_rate", "mean_score_shift",
    "train_adv_auroc", "train_adv_balanced_accuracy",
    "train_flip_rate", "train_neg_spike_rate",
)


def _merge_results_json_extras(run_dir: Path, main_df: pd.DataFrame,
                               per_layer: bool) -> pd.DataFrame:
    """Add results.json-only columns (the robust/adversarial block) onto the summary table.

    No-op for runs without them, so non-adversarial runs keep their exact column set.
    """
    import json

    rows = []
    for f in sorted(run_dir.glob("step_*/results.json")):
        try:
            row = (json.loads(f.read_text()) or {}).get("row") or {}
        except (OSError, ValueError):
            continue
        extra = {k: row[k] for k in _RESULTS_JSON_EXTRAS if k in row}
        if not extra:
            continue
        key = {"step": row.get("step")}
        if per_layer:
            key["block"] = row.get("block")
        rows.append({**key, **extra})
    if not rows or "step" not in main_df.columns:
        return main_df
    on = ["step", "block"] if (per_layer and "block" in main_df.columns) else ["step"]
    return main_df.merge(pd.DataFrame(rows).drop_duplicates(subset=on), on=on, how="left")


def write_run_summary(run_dir: str | Path, metric: str = "auroc", wandb_run=None) -> pd.DataFrame:
    """Write training_results.csv + training_results_by_nobj.csv + training_summary.png.

    Re-derived from the per-epoch dumps, selecting the best epoch per item by `metric`.
    Returns the main best-epoch DataFrame (empty if no dumps were found — callers that
    need a hard error, like the CLI, should check `.empty`). Logs the tables + figure
    to `wandb_run` when provided.
    """
    run_dir = Path(run_dir)
    items = discover_items(run_dir)
    if not items:
        print(f"[summary] no epoch_preds_*.csv dumps under {run_dir} — skipping summary")
        return pd.DataFrame()
    per_layer = any(b is not None for _, b, _ in items)
    print(f"[summary] {len(items)} item(s){' (per-layer)' if per_layer else ''}; "
          f"best epoch by val {metric}")

    summaries = [summarize_item(step, block, preds, metric) for step, block, preds in items]
    summaries.sort(key=lambda s: (s["step"], -1 if s["block"] is None else s["block"]))

    main_df = pd.DataFrame([s["row"] for s in summaries])
    # The summary is re-derived from the per-epoch prediction dumps, which only carry clean
    # scores. Metrics that exist solely in results.json — the adversarial/robust block — would
    # otherwise be silently dropped from training_results.csv even though write_task_result
    # recorded them. Join them back on (step, block).
    main_df = _merge_results_json_extras(run_dir, main_df, per_layer)
    main_df.to_csv(str(run_dir / "training_results.csv"), index=False)

    seg_df = build_segment_table(summaries, per_layer)
    seg_df.to_csv(str(run_dir / "training_results_by_nobj.csv"))

    summary_png = run_dir / "training_summary.png"
    plot_summary(main_df, seg_df, per_layer, summary_png)
    print(f"[summary] → training_results.csv, training_results_by_nobj.csv, training_summary.png")

    if wandb_run is not None:
        import wandb
        # seg_df has a MultiIndex on columns (metric, n_objects); reset_index keeps those
        # as tuples, which wandb.Table rejects ("columns expects list of strings or ints").
        # Flatten tuple columns to "metric__nobj" strings before logging.
        seg_flat = seg_df.reset_index()
        seg_flat.columns = [
            "__".join(str(x) for x in c if str(x) != "") if isinstance(c, tuple) else str(c)
            for c in seg_flat.columns
        ]
        log = {"training_results": wandb.Table(dataframe=main_df),
               "training_results_by_nobj": wandb.Table(dataframe=seg_flat)}
        if summary_png.exists():
            log["training_summary"] = wandb.Image(str(summary_png))
        wandb_run.log(log)
    return main_df


def main() -> None:
    ap = argparse.ArgumentParser(description="Best-epoch summary of a probe training run")
    ap.add_argument("run_dir", help="Timestamped run dir containing step_*/ subdirs")
    ap.add_argument("--metric", default="auroc", choices=SELECT_METRICS,
                    help="val metric used to pick the best epoch (default: auroc)")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        sys.exit(f"run_dir not found: {run_dir}")
    df = write_run_summary(run_dir, args.metric)
    if df.empty:
        sys.exit(
            "Nothing summarised: this run has no epoch_preds_*.csv dumps (it predates the "
            "per-epoch prediction dumps). Re-run training with the current trainer to emit them."
        )


if __name__ == "__main__":
    main()
