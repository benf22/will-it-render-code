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

## Training data (on Hugging Face)

The per-backbone **AttnSteer training sets** — the prompt list with seeds, the train/val split,
and the detector annotation of each clip — are hosted on HF (too large/data-heavy for git):
[`benf22/will-it-render` → `datasets/training/`](https://huggingface.co/benf22/will-it-render/tree/main/datasets/training)
(`wan.csv`, `ltx.csv`, `cogvideox.csv`).

Columns: `prompt_idx, prompt, seed, object_name, n_objects, split, label` (`label` ∈ {1 present,
0 absent, −1 tie}; WAN uses the colour-corrected `rgbfix` labels that the shipped probe was
trained on). Point `training.train_csv`/`val_csv` at your split to retrain — see
[`training/`](../src/will_it_render/training/README.md). The paper's **object-presence eval
labels** live alongside them under `paper_eval/` (see
[`evaluation/`](../src/will_it_render/evaluation/README.md)).
