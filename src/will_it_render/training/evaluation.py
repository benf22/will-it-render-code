from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Balance-aware, threshold-free metrics
#
# F1 / precision / recall are measured on the positive class only and are blind
# to class imbalance — a trivial "predict-all-positive" classifier scores
# F1 = 2p/(1+p) for positive base rate p (e.g. 0.88 at p=0.79). These helpers add
# threshold-free separation metrics (AUROC, average precision) and the trivial
# baseline so the reported numbers can't be mistaken for skill.
# ---------------------------------------------------------------------------


def _rankdata_avg(a: np.ndarray) -> np.ndarray:
    """Average-rank of each element (ties share the mean of their ranks)."""
    order = np.argsort(a, kind="mergesort")
    a_sorted = a[order]
    ranks = np.empty(len(a), dtype=np.float64)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a_sorted[j + 1] == a_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0  # mean of 1-based ranks i+1..j+1
        i = j + 1
    return ranks


def auroc(labels_binary: np.ndarray, probs: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (tie-aware)."""
    labels_binary = np.asarray(labels_binary)
    n_pos = int((labels_binary == 1).sum())
    n_neg = int((labels_binary == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _rankdata_avg(np.asarray(probs, dtype=np.float64))
    return float((r[labels_binary == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(labels_binary: np.ndarray, probs: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve)."""
    labels_binary = np.asarray(labels_binary)
    n_pos = int((labels_binary == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-np.asarray(probs, dtype=np.float64), kind="mergesort")
    y = labels_binary[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def baseline_f1(pos_base_rate: float) -> float:
    """F1 of the trivial predict-all-positive classifier at the given base rate."""
    return float(2 * pos_base_rate / (1 + pos_base_rate)) if pos_base_rate > 0 else 0.0


def evaluate_perframe(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_frames: int,
    offset: int = 1,
) -> dict:
    """Evaluate the per-latent-frame presence head.

    ``offset`` is the label column where the ``n_frames`` per-frame targets start
    (``ProbeDataset.perframe_offset``: 1 normally, 2 alongside the aesthetic aux). Cells whose
    target is NaN — an undecided (all-``'u'``) frame bin, or a row absent from the per-frame
    CSVs — are excluded from every metric, matching how the loss masks them.

    Returns a dict of scalars plus a ``per_frame`` DataFrame (one row per latent-frame index):

    ``frame_*``     pooled over all valid (sample, frame) cells — how well the head reads
                    presence in an individual frame.
    ``allframes_*`` score ``min_t logit_t`` vs GT "present in every decided frame" — the
                    quantity a min-over-frames steering objective optimizes, and the
                    apples-to-apples target against a TempoControl all-frames baseline.
    ``anyframe_*``  score ``max_t logit_t`` vs GT "present in at least one frame" — a sanity
                    check that should track the clip-level head.
    """
    model.eval()
    frame_logits: List[np.ndarray] = []
    frame_targets: List[np.ndarray] = []
    with torch.no_grad():
        for feats, labels in loader:
            feats = feats.to(device, non_blocking=True)
            out = model(feats, return_frames=True)
            fl = out[-1]  # (logit, frames) or (logit, aux, frames)
            frame_logits.append(fl.float().cpu().numpy())
            frame_targets.append(labels[:, offset:offset + n_frames].float().cpu().numpy())

    logits = np.concatenate(frame_logits, axis=0)      # [N, T]
    targets = np.concatenate(frame_targets, axis=0)    # [N, T], NaN = undecided
    valid = ~np.isnan(targets)

    out: dict = {
        "n_samples": int(logits.shape[0]),
        "n_frames": int(n_frames),
        "valid_cell_frac": float(valid.mean()) if valid.size else float("nan"),
    }

    # --- pooled per-frame metrics over all valid cells ---
    flat_y = (targets[valid] >= 0.5).astype(int)
    flat_s = logits[valid]
    out["frame_pos_base_rate"] = float(flat_y.mean()) if flat_y.size else float("nan")
    out["frame_auroc"] = auroc(flat_y, flat_s)
    out["frame_average_precision"] = average_precision(flat_y, flat_s)
    out["frame_baseline_f1"] = baseline_f1(out["frame_pos_base_rate"])
    pred = (flat_s >= 0.0).astype(int)  # logit >= 0 ⇔ prob >= 0.5
    tp = int(((pred == 1) & (flat_y == 1)).sum()); fp = int(((pred == 1) & (flat_y == 0)).sum())
    fn = int(((pred == 0) & (flat_y == 1)).sum()); tn = int(((pred == 0) & (flat_y == 0)).sum())
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    out["frame_balanced_accuracy"] = (tpr + tnr) / 2
    out["frame_f1"] = (2 * prec * tpr / (prec + tpr)) if (prec + tpr) else 0.0

    # --- per-frame-index breakdown (does the head degrade at the clip boundaries?) ---
    rows = []
    for t in range(n_frames):
        m = valid[:, t]
        y = (targets[m, t] >= 0.5).astype(int)
        rows.append({
            "frame": t,
            "n_valid": int(m.sum()),
            "pos_base_rate": float(y.mean()) if y.size else float("nan"),
            "auroc": auroc(y, logits[m, t]),
            "average_precision": average_precision(y, logits[m, t]),
        })
    out["per_frame"] = pd.DataFrame(rows)

    # --- aggregate temporal targets: min-over-frames / max-over-frames ---
    # Only rows with at least one decided frame take part; the GT quantifies over the
    # DECIDED frames only, mirroring the frac-of-frames convention (undecided frames are
    # excluded rather than counted as absent).
    has_any = valid.any(axis=1)
    if has_any.any():
        masked_hi = np.where(valid, targets, np.nan)
        with np.errstate(invalid="ignore"):
            all_y = (np.nanmin(masked_hi[has_any], axis=1) >= 0.5).astype(int)
            any_y = (np.nanmax(masked_hi[has_any], axis=1) >= 0.5).astype(int)
        # Score with the same reduction the objective would use, over decided frames only.
        masked_logits = np.where(valid, logits, np.nan)
        all_s = np.nanmin(masked_logits[has_any], axis=1)
        any_s = np.nanmax(masked_logits[has_any], axis=1)
        out["allframes_pos_base_rate"] = float(all_y.mean())
        out["allframes_auroc"] = auroc(all_y, all_s)
        out["allframes_average_precision"] = average_precision(all_y, all_s)
        out["anyframe_pos_base_rate"] = float(any_y.mean())
        out["anyframe_auroc"] = auroc(any_y, any_s)
        out["anyframe_average_precision"] = average_precision(any_y, any_s)

    # --- WITHIN-CLIP temporal correlation ---
    # Per (video, object): correlation between the predicted per-frame presence prob and the GT
    # per-frame timeline, over that sample's DECIDED frames. Unlike frame_auroc (cross-sample at a
    # fixed frame) this measures whether the head reproduces the temporal SHAPE within a clip — the
    # quantity per-frame steering (and TempoControl's Pearson objective) actually care about. Only
    # samples whose GT timeline VARIES over decided frames are scored (a constant timeline has no
    # shape to track); the fraction scored is reported so a low n is visible.
    probs = 1.0 / (1.0 + np.exp(-logits))
    pears, spear = [], []
    from scipy.stats import spearmanr  # local import; scipy already a dep of the eval stack
    for i in range(logits.shape[0]):
        m = valid[i]
        if int(m.sum()) < 3:
            continue
        y = (targets[i, m] >= 0.5).astype(float)
        p = probs[i, m]
        if y.std() < 1e-8 or p.std() < 1e-8:      # constant timeline or constant prediction
            continue
        pears.append(float(np.corrcoef(p, y)[0, 1]))
        sr = spearmanr(p, y).correlation
        if np.isfinite(sr):
            spear.append(float(sr))
    out["frame_temporal_pearson"] = float(np.mean(pears)) if pears else float("nan")
    out["frame_temporal_spearman"] = float(np.mean(spear)) if spear else float("nan")
    out["frame_temporal_corr_n"] = len(pears)      # clips with a non-constant timeline
    out["frame_temporal_corr_frac"] = (len(pears) / logits.shape[0]) if logits.shape[0] else float("nan")
    return out


def evaluate_class(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    offset: int,
    n_classes: int,
    class_vocab: Optional[dict] = None,
) -> dict:
    """Evaluate the class-identity head (predicts which object class the map belongs to).

    ``offset`` is the label column holding the integer class index
    (``ProbeDataset.class_offset``). Rows whose target is -1 (object_name absent from the
    train-built vocab) are excluded from every metric, matching how the loss masks them.

    Returns top-1 / top-5 / macro (mean per-class) accuracy plus a ``per_class`` DataFrame
    (support + top-1 accuracy per class). ``class_vocab`` (name→index), when given, labels the
    per-class rows with the class name.
    """
    model.eval()
    logits_all: List[np.ndarray] = []
    tgt_all: List[np.ndarray] = []
    with torch.no_grad():
        for feats, labels in loader:
            feats = feats.to(device, non_blocking=True)
            out = model(feats, return_class=True)
            cl = out[-1]  # (logit, class_logits) — class logits are last
            logits_all.append(cl.float().cpu().numpy())
            tgt_all.append(labels[:, offset].cpu().numpy())

    logits = np.concatenate(logits_all, axis=0)              # [N, n_classes]
    targets = np.concatenate(tgt_all, axis=0).astype(int)    # [N], -1 = unknown class
    valid = targets >= 0
    logits, targets = logits[valid], targets[valid]
    n = int(logits.shape[0])

    out: dict = {"n_classes": int(n_classes), "n_eval": n}
    if n == 0:
        out.update(class_top1_acc=float("nan"), class_top5_acc=float("nan"),
                   class_macro_acc=float("nan"), per_class=pd.DataFrame())
        return out

    pred1 = logits.argmax(axis=1)
    out["class_top1_acc"] = float((pred1 == targets).mean())
    k = min(5, n_classes)
    topk = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    out["class_top5_acc"] = float((topk == targets[:, None]).any(axis=1).mean())

    # Per-class precision / recall / F1 (+ macro), computed with numpy (no sklearn dependency —
    # the training venv has none). 0 for classes never predicted / with no support.
    tp = np.array([int(((pred1 == c) & (targets == c)).sum()) for c in range(n_classes)],
                  dtype=float)
    pred_c = np.bincount(pred1, minlength=n_classes).astype(float)   # TP+FP per predicted class
    sup = np.bincount(targets, minlength=n_classes).astype(float)    # TP+FN per true class = support
    prec = np.divide(tp, pred_c, out=np.zeros_like(tp), where=pred_c > 0)
    rec = np.divide(tp, sup, out=np.zeros_like(tp), where=sup > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    inv = {i: name for name, i in class_vocab.items()} if class_vocab else {}
    rows, per_class_acc = [], []
    for c in range(n_classes):
        m = targets == c
        support = int(m.sum())
        acc = float((pred1[m] == c).mean()) if support else float("nan")
        if support:
            per_class_acc.append(acc)
        rows.append({"class_index": c, "class_name": inv.get(c, ""),
                     "support": support, "top1_acc": acc,
                     "precision": float(prec[c]), "recall": float(rec[c]), "f1": float(f1[c])})
    out["class_macro_acc"] = float(np.mean(per_class_acc)) if per_class_acc else float("nan")
    # Macro P/R/F1 over classes with support (support-0 classes excluded, matching macro_acc).
    seen = sup > 0
    out["class_macro_precision"] = float(prec[seen].mean()) if seen.any() else float("nan")
    out["class_macro_recall"] = float(rec[seen].mean()) if seen.any() else float("nan")
    out["class_macro_f1"] = float(f1[seen].mean()) if seen.any() else float("nan")
    out["per_class"] = pd.DataFrame(rows)
    # Per-sample true/pred (valid rows only) for the confusion matrix + per-sample dump.
    out["y_true"] = targets
    out["y_pred"] = pred1
    return out


def evaluate_probe(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    step: int,
    thresholds: Optional[List[float]] = None,
    dataset=None,
) -> dict:
    """Evaluate the probe on val_loader.

    If `dataset` is provided (a ProbeDataset with a `records` list), a per-sample
    predictions DataFrame is returned under the key ``"predictions"``.  The loader
    must use ``shuffle=False`` so that batch order matches ``dataset.records``.
    Binary predictions use the median of `thresholds` as the cutoff.
    """
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    model.eval()
    all_probs: List[float] = []
    all_labels: List[float] = []

    with torch.no_grad():
        for feats, labels in val_loader:
            feats = feats.to(device, non_blocking=True)
            logits = model(feats).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().tolist()
            all_probs.extend(probs if isinstance(probs, list) else [probs])
            # Column 0 is the class label; a multi-task loader yields [B, 1+K] labels
            # (class + aux targets), so slice it out. Single-task [B, 1] → [B] unchanged.
            labs = labels[..., 0].cpu().tolist()
            all_labels.extend(labs if isinstance(labs, list) else [labs])

    probs_np = np.array(all_probs)
    labels_binary = (np.array(all_labels) >= 0.5).astype(int)

    n_total = len(labels_binary)
    agg_records = []
    preds_by_thr: dict[float, np.ndarray] = {}
    for thr in thresholds:
        preds = (probs_np >= thr).astype(int)
        preds_by_thr[thr] = preds
        tp = int(((preds == 1) & (labels_binary == 1)).sum())
        fp = int(((preds == 1) & (labels_binary == 0)).sum())
        fn = int(((preds == 0) & (labels_binary == 1)).sum())
        tn = int(((preds == 0) & (labels_binary == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        balanced_accuracy = (recall + tnr) / 2  # recall is TPR; robust to imbalance
        accuracy = (tp + tn) / n_total if n_total > 0 else 0.0
        agg_records.append({
            "threshold": thr, "precision": precision, "recall": recall, "f1": f1,
            "balanced_accuracy": balanced_accuracy, "accuracy": accuracy,
        })

    pr_table = pd.DataFrame(agg_records)
    n_pos = int((labels_binary == 1).sum())
    pos_base_rate = n_pos / n_total if n_total > 0 else 0.0
    result = {
        "pr_table": pr_table,
        "probs": probs_np,
        "labels_binary": labels_binary,
        "auroc": auroc(labels_binary, probs_np),
        "average_precision": average_precision(labels_binary, probs_np),
        "pos_base_rate": pos_base_rate,
        "baseline_f1": baseline_f1(pos_base_rate),
    }

    if dataset is not None:
        sample_rows = []
        for i, (prob, gt_raw) in enumerate(zip(all_probs, all_labels)):
            feat_path = dataset.records[i][0]
            info_path = Path(feat_path).parent / "prompt_info.json"
            if info_path.exists():
                info = json.loads(info_path.read_text())
                _sample_name = f"{info.get('class_name', '')}__seed_{info.get('seed', '')}"
            else:
                _sample_name = Path(feat_path).parent.name
            row = {
                "sample_name": _sample_name,
                "gt": gt_raw,
                "prob_pred": prob,
            }
            for thr in thresholds:
                row[f"binary_pred_{thr}"] = int(preds_by_thr[thr][i])
            sample_rows.append(row)
        result["predictions"] = pd.DataFrame(sample_rows)

    return result


def plot_epoch_summary(
    epoch_losses: list[tuple[int, float, float]],
    step: int,
    output_path: str | Path,
) -> None:
    """Single-subplot loss curves for one denoising step's training run.

    epoch_losses: list of (epoch, train_loss, val_loss) — one entry per epoch
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    if epoch_losses:
        epochs_l = [e for e, *_ in epoch_losses]
        train_losses = [tl for _, tl, _ in epoch_losses]
        val_losses = [vl for _, _, vl in epoch_losses]
        ax.plot(epochs_l, train_losses, label="Train loss", color="steelblue")
        ax.plot(epochs_l, val_losses, label="Val loss", color="tomato")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Step {step} — train / val loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()


def plot_pr_curves(
    train_epoch_metrics: list[tuple[int, float, float, float]],
    val_epoch_metrics: list[tuple[int, float, float, float]],
    step: int,
    output_path: str | Path,
) -> None:
    """Two-subplot P/R/F1 curves (train top, val bottom) saved at end of training.

    Each metrics list contains (epoch, precision, recall, f1) entries.
    """
    fig, (ax_train, ax_val) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    def _plot_metrics(ax, metrics, title):
        if metrics:
            epochs_m = [e for e, *_ in metrics]
            precisions = [p for _, p, *_ in metrics]
            recalls = [r for _, _, r, _ in metrics]
            f1s = [f for *_, f in metrics]
            ax.plot(epochs_m, precisions, marker="o", label="Precision", color="steelblue")
            ax.plot(epochs_m, recalls, marker="s", label="Recall", color="tomato")
            ax.plot(epochs_m, f1s, marker="^", label="F1", color="seagreen")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    _plot_metrics(ax_train, train_epoch_metrics, f"Step {step} — train P/R/F1")
    _plot_metrics(ax_val, val_epoch_metrics, f"Step {step} — val P/R/F1")
    ax_val.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()


def plot_per_layer_f1_summary(
    results_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """F1 vs denoising step with one line per transformer block.

    results_df must have columns: step, block, f1.
    """
    blocks = sorted(results_df["block"].unique())
    steps = sorted(results_df["step"].unique())
    cmap = plt.get_cmap("tab20" if len(blocks) <= 20 else "hsv")
    colors = [cmap(i / max(len(blocks) - 1, 1)) for i in range(len(blocks))]

    fig, ax = plt.subplots(figsize=(12, 5))
    for color, block in zip(colors, blocks):
        row = results_df[results_df["block"] == block].sort_values("step")
        ax.plot(row["step"], row["f1"], marker="o", markersize=3, linewidth=1.2,
                color=color, label=f"block {block}")

    ax.set_xlabel("Denoising step")
    ax.set_ylabel("F1")
    ax.set_title("F1 per layer across denoising steps")
    ax.set_ylim(0, 1)
    ax.set_xticks(steps)
    ax.grid(True, alpha=0.3)
    # compact legend outside the plot when there are many blocks
    ncol = max(1, len(blocks) // 10)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7,
              ncol=ncol, borderaxespad=0)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_layer_loss_by_step(
    step: int,
    block_losses: dict,  # block_idx → list of (epoch, train_loss, val_loss)
    output_path: str | Path,
) -> None:
    """Train (dotted) and val (solid) loss curves per block for a single denoising step.

    block_losses: {block_idx: [(epoch, train_loss, val_loss), ...]}
    """
    blocks = sorted(block_losses)
    cmap = plt.get_cmap("tab20" if len(blocks) <= 20 else "hsv")
    colors = [cmap(i / max(len(blocks) - 1, 1)) for i in range(len(blocks))]

    fig, ax = plt.subplots(figsize=(12, 5))
    for color, block in zip(colors, blocks):
        losses = block_losses[block]
        if not losses:
            continue
        epochs = [e for e, _, _ in losses]
        train_l = [tl for _, tl, _ in losses]
        val_l = [vl for _, _, vl in losses]
        ax.plot(epochs, train_l, linestyle="--", linewidth=1, color=color, alpha=0.8)
        ax.plot(epochs, val_l, linestyle="-", linewidth=1, color=color, alpha=0.8,
                label=f"block {block}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Step {step} — train (dotted) / val (solid) loss by block")
    ax.grid(True, alpha=0.3)
    ncol = max(1, len(blocks) // 10)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7,
              ncol=ncol, borderaxespad=0)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_fp_fn(
    train_probs: np.ndarray,
    train_labels_binary: np.ndarray,
    val_probs: np.ndarray,
    val_labels_binary: np.ndarray,
    step: int,
    threshold: float = 0.5,
    output_path: Optional[str | Path] = None,
    n_bins: int = 50,
) -> None:
    """Two-subplot raw-histogram plot (left=train, right=val) of pos/neg predicted-prob
    distributions, with the bars on the wrong side of the threshold (FN = positives below,
    FP = negatives above) emphasised. Uses raw counts — no KDE / smoothing — so the true,
    possibly jagged distribution and any class imbalance are visible.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    width = bins[1] - bins[0]

    def _draw_hist(ax, probs, labels_binary, thr, title):
        pos_probs = probs[labels_binary == 1]
        neg_probs = probs[labels_binary == 0]
        pos_c, _ = np.histogram(pos_probs, bins=bins)
        neg_c, _ = np.histogram(neg_probs, bins=bins)

        # full distributions (light), then emphasise the misclassified side (solid + hatch)
        ax.bar(centers, pos_c, width=width, color="steelblue", alpha=0.45, label="Positive", align="center")
        ax.bar(centers, neg_c, width=width, color="tomato", alpha=0.45, label="Negative", align="center")
        fn_mask = centers < thr   # positives predicted negative
        fp_mask = centers >= thr  # negatives predicted positive
        ax.bar(centers[fn_mask], pos_c[fn_mask], width=width, color="steelblue", alpha=0.9,
               hatch="//", edgecolor="white", linewidth=0, label="FN", align="center")
        ax.bar(centers[fp_mask], neg_c[fp_mask], width=width, color="tomato", alpha=0.9,
               hatch="//", edgecolor="white", linewidth=0, label="FP", align="center")
        ax.axvline(thr, linestyle="--", color="gray", linewidth=1.5)

        preds = (probs >= thr).astype(int)
        tp = int(((preds == 1) & (labels_binary == 1)).sum())
        fp_c = int(((preds == 1) & (labels_binary == 0)).sum())
        fn_c = int(((preds == 0) & (labels_binary == 1)).sum())
        tn = int(((preds == 0) & (labels_binary == 0)).sum())
        prec = tp / (tp + fp_c) if (tp + fp_c) else 0.0
        rec = tp / (tp + fn_c) if (tp + fn_c) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        tnr = tn / (tn + fp_c) if (tn + fp_c) else 0.0
        bal_acc = (rec + tnr) / 2
        base = pos_probs.size / labels_binary.size if labels_binary.size else 0.0
        auc = auroc(labels_binary, probs)
        ax.set_title(
            f"{title} | Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}\n"
            f"AUROC={auc:.3f}  BalAcc={bal_acc:.3f}  pos={base:.0%}  "
            f"all-pos F1={baseline_f1(base):.3f}",
            fontsize=10,
        )
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 1)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)

    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(14, 5))
    _draw_hist(ax_train, train_probs, train_labels_binary, threshold, f"Step {step} — Train")
    _draw_hist(ax_val, val_probs, val_labels_binary, threshold, f"Step {step} — Val")
    plt.tight_layout()
    if output_path:
        plt.savefig(str(output_path), dpi=150)
        plt.close()
    else:
        plt.show()
