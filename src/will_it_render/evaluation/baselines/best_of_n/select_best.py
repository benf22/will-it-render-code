"""Best-of-N baseline selection + curve.

Given the detection labels of unsteered clips (N random seeds per prompt) and the
seed->rep map from ``make_seeds.py``, pick the best clip per prompt and report the
best-of-n detection curve.

Per-clip metrics (one clip == one (prompt_idx, seed), i.e. one ``extraction_dir``):
  det_ratio   = fraction of the prompt's VALID objects (label != -1) that are present
  all_present = 1 iff every valid object is present (det-all)
Rows with label == -1 (detector tie) are dropped before aggregation.

Selection (per prompt, over a given set of reps): pick the clip maximizing
(all_present, det_ratio); ties broken by the LOWEST rep (prefix-stable, deterministic).

Curves (for n = 1..max_n), per-k and overall, with binomial SE on the mean:
  prefix    = best-of-n using reps 0..n-1 (what this run actually achieves)
  expected  = closed-form E[best-of-n] over uniformly random n-subsets of the AVAILABLE
              reps per prompt (unbiased for the "pick n random seeds" protocol)

Outputs (in ``--out-dir``):
  clip_metrics.csv          per-clip det_ratio / all_present / rep
  selection_n{N}.csv        the winning clip per prompt at the full n
  bestofn_labels_n{N}.csv    label rows of the winning clips only (det-style, for downstream)
  curve.csv                 n, scope (overall|k=..), kind (prefix|expected), metric, mean, se, n_prompts
  curve.png                 det-all & det-ratio vs n (overall + per-k), prefix vs expected
If ``--link-dir`` is given, symlink each winning clip dir into it (for photo/VBench scoring).
"""
from __future__ import annotations

import argparse
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# per-clip aggregation
# ---------------------------------------------------------------------------

def clip_metrics(labels: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    """One row per (prompt_idx, seed, extraction_dir) with det_ratio, all_present, rep."""
    lab = labels.copy()
    lab["label"] = pd.to_numeric(lab["label"], errors="coerce")
    lab = lab.dropna(subset=["label"])
    # dedupe repeated label rows (a clip may have been scored twice across waves)
    lab = lab.drop_duplicates(subset=["extraction_dir", "object_name"], keep="last")
    lab = lab[lab["label"] != -1]  # drop detector ties

    g = lab.groupby(["prompt_idx", "seed", "extraction_dir"])
    m = g["label"].agg(n_valid="count", n_present=lambda s: int((s == 1).sum())).reset_index()
    m = m[m["n_valid"] > 0]
    m["det_ratio"] = m["n_present"] / m["n_valid"]
    m["all_present"] = (m["n_present"] == m["n_valid"]).astype(int)

    m["prompt_idx"] = m["prompt_idx"].astype(int)
    m["seed"] = m["seed"].astype(int)
    rep_of = seeds.astype({"prompt_idx": int, "seed": int}).set_index(["prompt_idx", "seed"])
    m = m.join(rep_of[["rep", "k"]], on=["prompt_idx", "seed"])
    if m["rep"].isna().any():
        missing = m[m["rep"].isna()][["prompt_idx", "seed"]].head(5).to_dict("records")
        raise SystemExit(f"{int(m['rep'].isna().sum())} labelled clips have no (prompt,seed) in seeds_all, e.g. {missing}")
    m["rep"] = m["rep"].astype(int)
    m["k"] = m["k"].astype(int)
    return m.sort_values(["prompt_idx", "rep"]).reset_index(drop=True)


def pick_best(df: pd.DataFrame) -> pd.DataFrame:
    """One winner row per prompt: max (all_present, det_ratio), tie -> lowest rep."""
    ordered = df.sort_values(["all_present", "det_ratio", "rep"], ascending=[False, False, True])
    return ordered.drop_duplicates(subset="prompt_idx", keep="first").reset_index(drop=True)


# ---------------------------------------------------------------------------
# curves
# ---------------------------------------------------------------------------

def _se(mean: float, n: int) -> float:
    return float(np.sqrt(max(mean * (1 - mean), 0.0) / n)) if n else float("nan")


def prefix_curve(m: pd.DataFrame, max_n: int, metric: str) -> list[tuple[int, float, float, int]]:
    """best-of-n over reps 0..n-1, per n."""
    out = []
    for n in range(1, max_n + 1):
        best = pick_best(m[m["rep"] < n])[metric]
        out.append((n, float(best.mean()), _se(float(best.mean()), len(best)), len(best)))
    return out


def expected_curve(m: pd.DataFrame, max_n: int, metric: str) -> list[tuple[int, float, float, int]]:
    """E[best-of-n] over uniformly random n-subsets of each prompt's available reps.

    For a prompt with R clips whose metric values sorted ascending are v_1..v_R, the
    expected max of a random n-subset is  sum_i v_i * [C(i-1,n-1) / C(R,n)]  (i = rank,
    1-based). Averaged over prompts. Exact; no sampling.
    """
    out = []
    for n in range(1, max_n + 1):
        vals, R_ok = [], 0
        for _, d in m.groupby("prompt_idx"):
            v = np.sort(d[metric].to_numpy())
            R = len(v)
            if R < n:
                continue
            R_ok += 1
            denom = comb(R, n)
            w = np.array([comb(i, n - 1) for i in range(R)]) / denom  # i = rank-1 (0-based)
            vals.append(float((w * v).sum()))
        mean = float(np.mean(vals)) if vals else float("nan")
        # SE across prompts of the per-prompt expectation (treats prompts as the sampling unit)
        se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float("nan")
        out.append((n, mean, se, R_ok))
    return out


def build_curves(m: pd.DataFrame, max_n: int) -> pd.DataFrame:
    rows = []
    scopes = [("overall", m)] + [(f"k={k}", g) for k, g in m.groupby("k")]
    for scope, sub in scopes:
        for metric in ("all_present", "det_ratio"):
            for kind, fn in (("prefix", prefix_curve), ("expected", expected_curve)):
                for n, mean, se, npr in fn(sub, max_n, metric):
                    rows.append(dict(n=n, scope=scope, kind=kind, metric=metric,
                                     mean=mean, se=se, n_prompts=npr))
    return pd.DataFrame(rows)


def plot_curves(curve: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric in zip(axes, ("all_present", "det_ratio")):
        ov = curve[(curve.scope == "overall") & (curve.metric == metric)]
        for kind, style in (("prefix", "-o"), ("expected", "--")):
            d = ov[ov.kind == kind].sort_values("n")
            ax.errorbar(d.n, d["mean"], yerr=d.se, fmt=style, capsize=3,
                        label=f"overall ({kind})", lw=2)
        for k in sorted(curve[curve.scope.str.startswith("k=")].scope.unique()):
            d = curve[(curve.scope == k) & (curve.metric == metric) & (curve.kind == "expected")].sort_values("n")
            ax.plot(d.n, d["mean"], alpha=0.4, label=f"{k} (exp)")
        ax.set_xlabel("N (seeds per prompt)")
        ax.set_ylabel(metric)
        ax.set_title(f"best-of-N: {metric}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="+", required=True, help="one or more sweep label CSVs")
    ap.add_argument("--seeds-csv", required=True, help="seeds_all.csv from make_seeds")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=None, help="n to emit selection for (default = max available rep+1)")
    ap.add_argument("--link-dir", default=None, help="symlink winning clip dirs here for photo/VBench scoring")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    labels = pd.concat([pd.read_csv(f) for f in args.labels], ignore_index=True)
    seeds = pd.read_csv(args.seeds_csv)

    m = clip_metrics(labels, seeds)
    m.to_csv(out / "clip_metrics.csv", index=False)
    max_rep = int(m["rep"].max())
    max_n = max_rep + 1
    n_sel = args.n or max_n
    print(f"{m['extraction_dir'].nunique()} clips over {m['prompt_idx'].nunique()} prompts; "
          f"reps 0..{max_rep}; selecting at n={n_sel}")

    # per-prompt reps present (coverage report)
    cov = m.groupby("prompt_idx")["rep"].nunique()
    print(f"reps-per-prompt: min={cov.min()} median={int(cov.median())} max={cov.max()}; "
          f"{int((cov < n_sel).sum())} prompts have < {n_sel} clips")

    sel = pick_best(m[m["rep"] < n_sel])
    sel.to_csv(out / f"selection_n{n_sel}.csv", index=False)

    # label rows of the winners only (det-style downstream)
    labels2 = labels.copy()
    win_dirs = set(sel["extraction_dir"])
    labels2[labels2["extraction_dir"].isin(win_dirs)].to_csv(out / f"bestofn_labels_n{n_sel}.csv", index=False)

    curve = build_curves(m, max_n)
    curve.to_csv(out / "curve.csv", index=False)
    plot_curves(curve, out / "curve.png")

    ov = curve[curve.scope == "overall"]
    print("\noverall best-of-n (prefix):")
    for metric in ("all_present", "det_ratio"):
        d = ov[(ov.metric == metric) & (ov.kind == "prefix")].sort_values("n")
        print(f"  {metric}: " + "  ".join(f"n{r.n}={r['mean']:.3f}±{r.se:.3f}" for _, r in d.iterrows()))

    if args.link_dir:
        link = Path(args.link_dir)
        link.mkdir(parents=True, exist_ok=True)
        n_linked = 0
        for d in sel["extraction_dir"]:
            src = Path(d)
            dst = link / src.name
            if not dst.exists():
                dst.symlink_to(src)
                n_linked += 1
        print(f"linked {n_linked} winning clip dirs -> {link}")


if __name__ == "__main__":
    main()
