# Datasets

Evaluation data for [Will It Render?](https://benf22.github.io/will-it-render/) — object-presence
and temporal-control benchmarks for text-to-video generation. The headline set is the **473-prompt
[WIR test set](WIR_testset/)** (see its README for how it was built).

## Layout

| Path | What |
|------|------|
| [`WIR_testset/wir_testset.csv`](WIR_testset/) | the 473 prompts (with object count `k`, sorted by `k`) |
| [`WIR_testset/seeds.csv`](WIR_testset/) | per-prompt seeds: one main seed + two best-of-N seeds |
| [`other_benchmarks/vbench_single_object.csv`](other_benchmarks/) | VBench single-object class prompts (single-object table) |
| [`other_benchmarks/tempocontrol/`](other_benchmarks/) | per-frame temporal-control test sets (one / two / multi-object) |

These are **prompts and seeds only** — the generated videos and detector labels are produced by
the pipeline (`will_it_render.datasets` + `will_it_render.evaluation`), not shipped here.
