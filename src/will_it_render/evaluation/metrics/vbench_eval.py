from __future__ import annotations

import argparse
import importlib
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

# ── VBench on sys.path (mirrors common/visual_classifier.py) ───────────────
_VBENCH_ROOT = Path(__file__).parents[2] / "VBench"
if str(_VBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_VBENCH_ROOT))

from vbench.utils import init_submodules, save_json

# ── Dimension lists ───────────────────────────────────────────────────────────
GROUP_A = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "human_action",
    "temporal_style",
    "overall_consistency",
]
GROUP_B = ["object_class", "multiple_objects", "color", "scene", "appearance_style"]
ALL_DIMS = GROUP_A + GROUP_B  # spatial_relationship intentionally excluded


def _resolve_video(video_dir: Path) -> Path | None:
    if not video_dir.is_dir():
        warnings.warn(f"Directory not found, skipping: {video_dir}")
        return None
    mp4s = list(video_dir.glob("*.mp4"))
    if not mp4s:
        warnings.warn(f"No .mp4 found in {video_dir}, skipping")
        return None
    if len(mp4s) > 1:
        warnings.warn(f"Multiple .mp4 files in {video_dir}, using first: {mp4s[0].name}")
    return mp4s[0]


def build_full_info(
    root_dir: Path,
    dimensions: list[str],
) -> tuple[list[dict[str, Any]], dict[str, dict]]:
    """
    Scan root_dir for video subdirs.
    Returns (entries_for_vbench, video_meta) where video_meta maps
    video_path_str → {n_objects, object_names, prompt_words}.
    """
    wants_object_class = "object_class" in dimensions
    wants_multiple     = "multiple_objects" in dimensions
    non_object_dims    = [d for d in dimensions if d not in ("object_class", "multiple_objects")]

    entries: list[dict[str, Any]] = []
    video_meta: dict[str, dict] = {}

    for video_dir in sorted(root_dir.iterdir()):
        if not video_dir.is_dir():
            continue
        metadata_path = video_dir / "feature_metadata.json"
        if not metadata_path.exists():
            continue

        video_path = _resolve_video(video_dir)
        if video_path is None:
            continue

        with metadata_path.open() as f:
            object_names = json.load(f).get("words", [])

        prompt_en = video_path.stem.replace("_", " ")
        video_path_str = str(video_path)

        video_meta[video_path_str] = {
            "n_objects":    len(object_names),
            "object_names": object_names,
            "prompt_words": len(prompt_en.split()),
        }

        if non_object_dims:
            entries.append({
                "prompt_en":  prompt_en,
                "dimension":  non_object_dims,
                "video_list": [video_path_str],
            })

        if wants_object_class:
            for obj in object_names:
                entries.append({
                    "prompt_en":      prompt_en,
                    "dimension":      ["object_class"],
                    "video_list":     [video_path_str],
                    "auxiliary_info": {"object_class": {"object": obj}},
                })

        if wants_multiple:
            if len(object_names) < 2:
                warnings.warn(
                    f"multiple_objects requires at least 2 objects but found {len(object_names)} "
                    f"for {video_dir.name} — skipping this video for multiple_objects."
                )
            else:
                entries.append({
                    "prompt_en":      prompt_en,
                    "dimension":      ["multiple_objects"],
                    "video_list":     [video_path_str],
                    "auxiliary_info": {"multiple_objects": {"object": " and ".join(object_names)}},
                })

    return entries, video_meta


def run_evaluation(
    full_info_path: Path,
    dimensions: list[str],
    device: str,
) -> tuple[dict[str, float], list[dict]]:
    """
    Returns (aggregate_scores, per_video_records).
    per_video_records is a list of {video_path, dimension, score}.
    """
    import torch
    torch_device = torch.device(device)
    submodules = init_submodules(dimensions, local=True)
    scores: dict[str, float] = {}
    per_video: list[dict] = []

    for dim in dimensions:
        module = importlib.import_module(f"vbench.{dim}")
        compute_fn = getattr(module, f"compute_{dim}")
        all_results, video_results = compute_fn(str(full_info_path), torch_device, submodules[dim])

        for vr in video_results:
            per_video.append({
                "video_path": vr["video_path"],
                "dimension":  dim,
                "score":      float(vr["video_results"]),
            })

        if isinstance(all_results, dict):
            val = list(all_results.values())[0]
            scores[dim] = float(val[0]) if isinstance(val, (list, tuple)) else float(val)
        else:
            scores[dim] = float(all_results)

    return scores, per_video


# ── Segmentation helpers ──────────────────────────────────────────────────────

def _pivot_with_avg(df: pd.DataFrame, group_col: str, dimensions: list[str]) -> pd.DataFrame:
    """Pivot to (segment_value × dimension) with an 'Average' row appended."""
    present_dims = [d for d in dimensions if d in df["dimension"].unique()]
    table = (
        df.groupby([group_col, "dimension"])["score"]
        .mean()
        .unstack("dimension")
        .reindex(columns=present_dims)
    )
    avg = df.groupby("dimension")["score"].mean().reindex(present_dims)
    avg_row = pd.DataFrame([avg], index=["Average"])
    return pd.concat([table, avg_row])


def _print_table(table: pd.DataFrame, title: str) -> None:
    width = max(60, len(title) + 4)
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print("=" * width)
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))
    print()


def build_segment_tables(
    per_video: list[dict],
    video_meta: dict[str, dict],
    dimensions: list[str],
    output_dir: Path,
    name: str,
) -> None:
    df = pd.DataFrame(per_video)
    if df.empty:
        print("No per-video records — skipping segmentation tables.")
        return

    # Attach metadata columns
    df["n_objects"]    = df["video_path"].map(lambda p: video_meta.get(p, {}).get("n_objects"))
    df["prompt_words"] = df["video_path"].map(lambda p: video_meta.get(p, {}).get("prompt_words"))

    # ── Wide per-video scores (one row per video, dimensions as columns) ──────
    present_dims = [d for d in dimensions if d in df["dimension"].unique()]
    wide = (
        df.pivot_table(index="video_path", columns="dimension", values="score", aggfunc="mean")
        .reindex(columns=present_dims)
        .reset_index()
    )
    meta_cols = df[["video_path", "n_objects", "prompt_words"]].drop_duplicates("video_path")
    wide = wide.merge(meta_cols, on="video_path", how="left")
    per_video_path = output_dir / f"{name}_per_video_scores.csv"
    wide.to_csv(per_video_path, index=False)
    print(f"Saved per-video scores (wide) → {per_video_path}")

    # ── Save video_meta JSON for serve_results.py ─────────────────────────────
    meta_path = output_dir / f"{name}_video_meta.json"
    with meta_path.open("w") as f:
        json.dump(video_meta, f)
    print(f"Saved video metadata → {meta_path}")

    # ── Helper: attach group_by / group_value prefix cols to a pivot table ────
    def _to_labeled(t: pd.DataFrame, group_by_name: str) -> pd.DataFrame:
        labeled = t.copy()
        labeled.insert(0, "group_value", labeled.index.astype(str))
        labeled.insert(0, "group_by", group_by_name)
        labeled.index = range(len(labeled))
        return labeled

    # ── Table 1: by number of objects ─────────────────────────────────────────
    def _n_bucket(n: Any) -> str:
        if pd.isna(n):
            return "unknown"
        return str(int(n))

    df1 = df.copy()
    df1["n_objects_bucket"] = df1["n_objects"].map(_n_bucket)
    t1 = _pivot_with_avg(df1, "n_objects_bucket", dimensions)
    _print_table(t1, "Results by number of objects in clip")

    # ── Table 2: by object name (each video contributes to all its objects) ───
    obj_rows = []
    for _, row in df.iterrows():
        objects = video_meta.get(row["video_path"], {}).get("object_names") or ["unknown"]
        for obj in objects:
            r = row.to_dict()
            r["object_name"] = obj
            obj_rows.append(r)
    obj_df = pd.DataFrame(obj_rows)
    t2 = _pivot_with_avg(obj_df, "object_name", dimensions)
    _print_table(t2, "Results by object name")

    # ── Table 3: by prompt length (word count) ────────────────────────────────
    df3 = df.copy()
    df3["prompt_words"] = (
        df3["prompt_words"].fillna(-1).astype(int)
        .map(lambda w: "unknown" if w == -1 else str(w))
    )
    t3 = _pivot_with_avg(df3, "prompt_words", dimensions)
    _print_table(t3, "Results by prompt length (word count)")

    # ── Combined segmentation CSV (all 3 tables, stacked) ────────────────────
    combined = pd.concat(
        [_to_labeled(t1, "n_objects"), _to_labeled(t2, "object_name"), _to_labeled(t3, "prompt_words")],
        ignore_index=True,
    )
    seg_path = output_dir / f"{name}_segmented.csv"
    combined.to_csv(seg_path, index=False)
    print(f"Saved segmented results → {seg_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VBench evaluation over an extraction root directory."
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        nargs="?",
        help="Root extraction directory. Each subdir must contain feature_metadata.json and an .mp4 file. "
             "Not required when --resegment is used.",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=ALL_DIMS,
        choices=ALL_DIMS + ["spatial_relationship"],
        metavar="DIM",
        help=f"Dimensions to evaluate (default: all {len(ALL_DIMS)}). spatial_relationship is not supported.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./vbench_results"), help="Directory for output files.")
    parser.add_argument("--name", default="eval", help="Prefix for output files (default: eval).")
    parser.add_argument("--device", default="cuda", help="Torch device (default: cuda).")
    parser.add_argument(
        "--resegment",
        action="store_true",
        help="Skip evaluation; reload existing per-video scores and recompute segmentation tables only.",
    )
    args = parser.parse_args()

    if "spatial_relationship" in args.dimensions:
        parser.error("spatial_relationship is not supported.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Fast path: recompute segmentation from saved scores ──────────────────
    if args.resegment:
        wide_path = args.output_dir / f"{args.name}_per_video_scores.csv"
        meta_path = args.output_dir / f"{args.name}_video_meta.json"
        for p in (wide_path, meta_path):
            if not p.exists():
                parser.error(f"--resegment requires {p} (run without --resegment first).")

        wide_df = pd.read_csv(wide_path)
        with meta_path.open() as f:
            video_meta = json.load(f)

        dim_cols = [c for c in wide_df.columns if c not in ("video_path", "n_objects", "prompt_words")]
        long_df = wide_df.melt(
            id_vars=["video_path"],
            value_vars=dim_cols,
            var_name="dimension",
            value_name="score",
        ).dropna(subset=["score"])
        per_video = long_df.to_dict("records")

        print(f"Re-segmenting {len(wide_df)} videos × {len(dim_cols)} dimension(s) …")
        build_segment_tables(per_video, video_meta, dim_cols, args.output_dir, args.name)
        return

    # ── Full evaluation path ──────────────────────────────────────────────────
    if args.root_dir is None:
        parser.error("root_dir is required unless --resegment is used.")

    print(f"Scanning {args.root_dir} ...")
    entries, video_meta = build_full_info(args.root_dir, args.dimensions)
    if not entries:
        print("No valid video entries found. Exiting.")
        return

    full_info_path = args.output_dir / f"{args.name}_full_info.json"
    save_json(entries, str(full_info_path))
    print(f"Saved full_info ({len(entries)} entries) → {full_info_path}")

    print(f"Running evaluation on {len(args.dimensions)} dimension(s): {args.dimensions}")
    scores, per_video = run_evaluation(full_info_path, args.dimensions, args.device)

    # Aggregate results JSON
    results_path = args.output_dir / f"{args.name}_eval_results.json"
    with results_path.open("w") as f:
        json.dump(scores, f, indent=2)
    print(f"Saved results → {results_path}\n")

    # Aggregate results table
    col_w = max(len(d) for d in scores) + 2
    print(f"{'dimension':<{col_w}}  score")
    print("-" * (col_w + 8))
    for dim, score in scores.items():
        print(f"{dim:<{col_w}}  {score:.4f}")

    # Segmented tables + per-video CSV
    build_segment_tables(per_video, video_meta, args.dimensions, args.output_dir, args.name)


if __name__ == "__main__":
    main()
