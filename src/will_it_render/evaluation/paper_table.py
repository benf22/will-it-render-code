"""Reproduce the paper's object-presence table (det-all / det-ratio, per-k + average)
from the shipped annotated-clip labels — **no GPU, no model, no detectors**.

    python -m will_it_render.evaluation.paper_table

Reads the detector-label CSVs under ``data/paper_eval/`` (one per backbone × method; see that
directory's README) and prints the paper's main object-presence table. This is the "run the evaluation over the
annotated clips and get the paper's numbers" reproduction path.

**Rule C** (the paper's tie-handling): the per-(clip, object) ``label`` is a GRiT+GDINO+OWL
majority vote with ``-1`` = tie/uncertain. Rule C **drops the entire clip** if *any* of its
objects is a tie, then:
  * det-all   = clip counts iff EVERY (remaining) object is present (label 1); averaged per-k.
  * det-ratio = fraction of the clip's objects present; averaged per-k.
The reported average weights each object-count ``k`` equally (mean of the per-k rates), matching
the paper. (This clip-level tie drop differs from ``metrics_utils``' row-level default; see the
evaluation README.)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

# backbone -> display label; method -> (file stem, display label). Order = paper table order.
BACKBONES = [("wan", "WAN"), ("ltx", "LTX-2"), ("cogvideox", "CogVideoX")]
METHODS = [("unsteered", "Unsteered"), ("ae", "A&E"), ("ours", "Ours")]

# Paper headline numbers (det-all %, det-ratio %) — used by tests and the ✓/✗ column.
PAPER = {
    ("wan", "unsteered"): (27.1, 73.4), ("wan", "ae"): (39.2, 79.8), ("wan", "ours"): (59.9, 89.3),
    ("ltx", "unsteered"): (38.3, 78.9), ("ltx", "ae"): (37.8, 78.3), ("ltx", "ours"): (75.5, 93.9),
    ("cogvideox", "unsteered"): (30.5, 73.1), ("cogvideox", "ae"): (56.1, 87.3),
    ("cogvideox", "ours"): (65.9, 91.2),
}


HF_REPO = "benf22/will-it-render"   # the annotated-clip labels are hosted here, under paper_eval/


def data_dir(local: str | os.PathLike | None = None, repo: str = HF_REPO,
             download: bool = True) -> Path:
    """Locate the ``paper_eval`` labels. Resolution order:

    1. explicit ``local`` (``--data-dir``) or ``VP_PAPER_EVAL_DIR`` — use that directory;
    2. a repo-local ``data/paper_eval/`` (present in a source checkout) — offline;
    3. download ``paper_eval/*`` from the Hugging Face dataset ``repo`` (cached) — the default
       once the labels live on HF rather than in the git repo.
    """
    env = local or os.environ.get("VP_PAPER_EVAL_DIR")
    if env:
        return Path(env).expanduser()
    for parent in Path(__file__).resolve().parents:
        cand = parent / "data" / "paper_eval"
        if cand.is_dir() and any(cand.glob("*.csv")):
            return cand
    if not download:
        raise RuntimeError("paper_eval labels not found locally; set VP_PAPER_EVAL_DIR or allow download.")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError("paper_eval labels not found locally and huggingface_hub is not "
                           "installed (pip install huggingface_hub) to fetch them from HF.")
    from will_it_render.common import paths
    # Same HF repo as the AttnSteer bundles (see scripts/download_artifacts.py), labels under paper_eval/.
    local_root = snapshot_download(repo_id=repo, allow_patterns=["paper_eval/*"],
                                   cache_dir=str(paths.artifact_cache_dir()))
    return Path(local_root) / "paper_eval"


def rule_c_per_k(df: pd.DataFrame) -> dict:
    """Rule-C det-all/det-ratio per object-count k, and the k-averaged means.

    Returns ``{k: (det_all, det_ratio, n_clips), ..., "avg": (det_all, det_ratio, n_k)}``.
    """
    da: dict[int, list] = {}
    dr: dict[int, list] = {}
    for _, clip in df.groupby("extraction_dir"):
        labs = list(clip["label"])
        if not labs or any(x == -1 for x in labs):   # rule C: drop any clip with a tie object
            continue
        k = int(clip["n_objects"].iloc[0])
        da.setdefault(k, []).append(1.0 if all(x == 1 for x in labs) else 0.0)
        dr.setdefault(k, []).append(sum(1 for x in labs if x == 1) / len(labs))
    out: dict = {}
    ks = sorted(da)
    for k in ks:
        out[k] = (sum(da[k]) / len(da[k]), sum(dr[k]) / len(dr[k]), len(da[k]))
    if ks:
        out["avg"] = (sum(out[k][0] for k in ks) / len(ks),
                      sum(out[k][1] for k in ks) / len(ks), len(ks))
    return out


def evaluate(root: Path | None = None, **kw) -> dict:
    """{(backbone, method): rule_c_per_k(...)} over the label files (local or fetched from HF)."""
    root = Path(root) if root else data_dir(**kw)
    results = {}
    for bb, _ in BACKBONES:
        for m, _ in METHODS:
            f = root / f"{bb}_{m}.csv"
            if f.exists():
                results[(bb, m)] = rule_c_per_k(pd.read_csv(f))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce the paper's det-all/det-ratio table from labels.")
    ap.add_argument("--data-dir", default=None,
                    help="local paper_eval/ dir (else fetched from HF; VP_PAPER_EVAL_DIR also works)")
    ap.add_argument("--repo", default=HF_REPO, help="HF dataset repo hosting paper_eval/ labels")
    args = ap.parse_args()
    res = evaluate(local=args.data_dir, repo=args.repo)

    ks = [2, 3, 4, 5, 6, 7]
    hdr = f"{'backbone':10} {'method':10} {'det-all':>8} {'det-ratio':>9}  " + \
          " ".join(f"k{k:>5}" for k in ks) + "   paper(da/dr)"
    print(hdr)
    print("-" * len(hdr))
    for bb, bb_disp in BACKBONES:
        for m, m_disp in METHODS:
            r = res.get((bb, m))
            if not r:
                continue
            da, dr, _ = r["avg"]
            perk = " ".join(f"{r.get(k, (float('nan'),))[0]*100:6.1f}" for k in ks)
            pa, pr = PAPER[(bb, m)]
            ok = "✓" if abs(da*100 - pa) < 0.2 and abs(dr*100 - pr) < 0.2 else "✗"
            print(f"{bb_disp:10} {m_disp:10} {da*100:8.1f} {dr*100:9.1f}  {perk}   {pa}/{pr} {ok}")


if __name__ == "__main__":
    main()
