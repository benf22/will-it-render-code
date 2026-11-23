"""Evaluate object presence in generated videos — the paper's det-all / det-ratio metrics.

Standalone entry point: point it at your own generated clips (any model) and it reports how
often the requested objects actually render. Two modes:

    # (A) you already have per-(clip, object) detector labels -> just aggregate (no heavy deps)
    python -m will_it_render.evaluation.run --labels labels.csv

    # (B) you have a videos directory -> run the object detectors, then aggregate
    #     (needs the [detectors] extra; one subdir per clip with a feature_metadata.json,
    #      or pass --prompts to supply target_words)
    python -m will_it_render.evaluation.run --videos out/clips --prompts data/WIR_testset/wir_testset.csv

Metrics (from will_it_render.evaluation.metrics.metrics_utils):
  * det-all   = all_objects_present_rate   — fraction of clips with EVERY requested object present
  * det-ratio = obj_level_detection_rate   — fraction of (clip, object) pairs present
  * clip-level detection rate              — per-clip mean, clips weighted equally
A per-k breakdown is printed when the labels carry object counts.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from will_it_render.evaluation.metrics import metrics_utils as M


def _per_k(df: pd.DataFrame) -> pd.DataFrame | None:
    """det-all / det-ratio split by object count k (#objects per clip), if derivable."""
    if "k" not in df.columns:
        if "object_name" in df.columns and M.CLIP_KEY in df.columns:
            k = df.groupby(M.CLIP_KEY)["object_name"].transform("nunique")
            df = df.assign(k=k)
        else:
            return None
    rows = []
    for k, g in df.groupby("k"):
        rows.append({"k": int(k),
                     "det_all": M.all_objects_present_rate(g),
                     "det_ratio": M.obj_level_detection_rate(g),
                     "n_clips": g[M.CLIP_KEY].nunique()})
    return pd.DataFrame(rows).sort_values("k")


def _load_labels(args) -> pd.DataFrame:
    if args.labels:
        return pd.read_csv(args.labels)
    if not args.videos:
        sys.exit("provide --labels <csv> or --videos <dir> (see --help)")
    # videos -> labels via the object detectors (heavy; needs the [detectors] extra)
    from will_it_render.datasets.video_object_labeler import run_labeling  # noqa
    sys.exit(
        "video labeling from --videos is run via will_it_render.datasets.video_object_labeler "
        "(torchrun, [detectors] extra); produce labels.csv there, then re-run with --labels. "
        "See src/will_it_render/evaluation/README.md."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="det-all / det-ratio object-presence evaluation.")
    ap.add_argument("--labels", help="per-(clip,object) labels CSV (extraction_dir, object_name, label)")
    ap.add_argument("--videos", help="directory of generated clips (runs detectors; needs [detectors])")
    ap.add_argument("--prompts", help="prompts CSV with target_words (used with --videos)")
    args = ap.parse_args()

    df = _load_labels(args)
    s = M.summarize(df)
    print(f"clips: {df[M.CLIP_KEY].nunique()}  |  (clip,object) rows: {len(df)}")
    print(f"det-all   (all objects present): {M.all_objects_present_rate(df):.4f}")
    print(f"det-ratio (object-level)       : {s[M.OBJ_LEVEL_DETECTION_RATE]:.4f}")
    print(f"clip-level detection rate      : {s[M.CLIP_LEVEL_DETECTION_RATE]:.4f}")
    pk = _per_k(df)
    if pk is not None:
        print("\nper-k:")
        print(pk.to_string(index=False))


if __name__ == "__main__":
    main()
