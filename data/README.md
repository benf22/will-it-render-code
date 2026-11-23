# Datasets

Evaluation data for [Will It Render?](https://benf22.github.io/will-it-render/) — object-presence
and temporal-control benchmarks for text-to-video generation.

## WIR test set (the headline benchmark)

We introduce a **473-prompt multi-object benchmark** that extends VBench's `multiple_objects`
dimension from **two to seven** objects. Each prompt uses VBench's template ("a X and a Y and a
Z …") and draws *k* objects at random from the same 76-class list, under a **pairwise-uniqueness
constraint** — no two objects that share one prompt ever co-occur in another. Because combinations
are sampled independently of natural co-occurrence, prompts are frequently unusual (e.g., *"a pizza
and a surfboard and a kite"*), which makes object presence hard and prevents generators from
exploiting familiar scene statistics.

All three backbones (LTX-2, Wan 2.1, CogVideoX-5B) and all arms (base model, Best-of-N, A&E, Ours)
are evaluated on the **same 473 prompts**, each with a fixed per-prompt seed shared across arms.

| Object count *k* | 2 | 3 | 4 | 5 | 6 | 7 | **total** |
|---|---|---|---|---|---|---|---|
| # prompts | 80 | 80 | 80 | 80 | 80 | 73 | **473** |

→ **[`WIR_testset/`](WIR_testset/)** — the 473 prompts + the generation seeds.

## Layout

| Path | What |
|------|------|
| [`WIR_testset/wir_testset.csv`](WIR_testset/) | the 473 prompts (with object count `k`, sorted by `k`) |
| [`WIR_testset/seeds.csv`](WIR_testset/) | per-prompt seeds: one main seed + two best-of-N seeds |
| [`other_benchmarks/vbench_single_object.csv`](other_benchmarks/) | VBench single-object class prompts (single-object table) |
| [`other_benchmarks/tempocontrol/`](other_benchmarks/) | per-frame temporal-control test sets (one / two / multi-object) |

These are **prompts and seeds only** — the generated videos and detector labels are produced by
the pipeline (`will_it_render.datasets` + `will_it_render.evaluation`), not shipped here.
