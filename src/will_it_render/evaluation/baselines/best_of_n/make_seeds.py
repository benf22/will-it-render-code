"""Best-of-N baseline: derive the per-(prompt, rep) seed stream from ONE master seed.

For every prompt in the given CSVs we draw ``max_n`` distinct seeds from a numpy
Generator seeded with ``master_seed``. The layout is rep-major and prefix-stable:
rep r of prompt p is the same seed whatever ``max_n`` is, so best-of-n is valid for any
completed n and a run can be topped up later.

Outputs (in ``--out-dir``):
  seeds_all.csv            prompt_idx, k, rep, seed            (the full stream, the record)
  prompts_r<lo>to<hi>_k<k>.csv   per-(rep-block, k) generation CSVs (prompt cols + ``seed`` + ``rep``),
                           one row per (prompt, rep) in the block; consumed by steering.sweep
  seeds_unique.txt         every distinct seed (for gen_seed_latents*)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_stream(prompt_idxs: list[int], master_seed: int, max_n: int,
                 seed_lo: int = 1000, seed_hi: int = 2**31 - 1) -> dict[int, list[int]]:
    """prompt_idx -> list of max_n distinct seeds, globally unique across prompts."""
    rng = np.random.default_rng(master_seed)
    used: set[int] = set()
    out: dict[int, list[int]] = {}
    for p in sorted(prompt_idxs):
        seeds: list[int] = []
        while len(seeds) < max_n:
            s = int(rng.integers(seed_lo, seed_hi))
            if s in used:
                continue
            used.add(s)
            seeds.append(s)
        out[p] = seeds
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts-csv", nargs="+", required=True,
                    help="per-k prompt CSVs (prompt_idx,prompt,target_words,object_name); filename must end _k<k>.csv")
    ap.add_argument("--master-seed", type=int, required=True)
    ap.add_argument("--max-n", type=int, default=8)
    ap.add_argument("--blocks", nargs="*", default=None,
                    help="rep blocks as lo-hi (inclusive), e.g. 0-3 4-7; default = one block 0-(max_n-1)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    for f in args.prompts_csv:
        df = pd.read_csv(f)
        df["k"] = int(Path(f).stem.rsplit("_k", 1)[1])
        frames.append(df)
    prompts = pd.concat(frames, ignore_index=True)
    if prompts.prompt_idx.duplicated().any():
        dup = prompts.prompt_idx[prompts.prompt_idx.duplicated()].unique()[:10]
        raise SystemExit(f"prompt_idx not unique across CSVs, e.g. {list(dup)}")

    stream = build_stream(prompts.prompt_idx.tolist(), args.master_seed, args.max_n)
    k_of = prompts.set_index("prompt_idx").k.to_dict()
    rows = [dict(prompt_idx=p, k=int(k_of[p]), rep=r, seed=s)
            for p, ss in stream.items() for r, s in enumerate(ss)]
    allseeds = pd.DataFrame(rows)
    allseeds.to_csv(out / "seeds_all.csv", index=False)
    (out / "seeds_unique.txt").write_text(
        "\n".join(str(s) for s in sorted(allseeds.seed.unique())) + "\n")

    prompt_cols = [c for c in prompts.columns if c != "k"]
    blocks = args.blocks or [f"0-{args.max_n - 1}"]
    for b in blocks:
        lo, hi = (int(x) for x in b.split("-"))
        sub = allseeds[(allseeds.rep >= lo) & (allseeds.rep <= hi)]
        m = sub.merge(prompts, on=["prompt_idx", "k"])
        m = m.sort_values(["k", "rep", "prompt_idx"])
        for k, g in m.groupby("k"):
            g[prompt_cols + ["seed", "rep"]].to_csv(out / f"prompts_r{lo}to{hi}_k{k}.csv", index=False)
        print(f"block {b}: {len(m)} generations")
    print(f"{len(prompts)} prompts x {args.max_n} reps = {len(allseeds)} seeds "
          f"({allseeds.seed.nunique()} unique) -> {out}")


if __name__ == "__main__":
    main()
