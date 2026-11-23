# WIR test set — the headline benchmark

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

## `wir_testset.csv` — the prompts

Sorted by object count `k`, then `prompt_idx`.

| Column | Meaning |
|--------|---------|
| `k` | number of objects in the prompt (2–7) |
| `prompt_idx` | stable id (joins to `seeds.csv`) |
| `prompt` | the text prompt ("a X and a Y and …") |
| `object_name` | underscore-joined object list — the label key |
| `target_words` | comma-separated words to steer / detect |

## `seeds.csv` — the generation seeds

Generation is seeded per prompt (not a single global seed). All arms (base / A&E / Ours) use
`seed_main`; **Best-of-N = best-of-3** over `{seed_main, seed_bon2, seed_bon3}` and keeps the clip
the detector scores best. Seeds come from one master seed (`20261026`) via a prefix-stable
per-(prompt, rep) stream, so rep *r* of a prompt is fixed regardless of N.

| Column | Meaning |
|--------|---------|
| `prompt_idx`, `k`, `prompt` | join keys / context |
| `seed_main` | primary generation seed (rep 0), shared across arms |
| `seed_bon2`, `seed_bon3` | the two additional Best-of-N seeds (reps 1, 2) |

## Use

```bash
# steer the full test set with the WAN "Ours" recipe (seed per prompt from seeds.csv)
python -m will_it_render.inference.steer \
    --config configs/steering/wan.yaml \
    --prompts_csv data/WIR_testset/wir_testset.csv --output_dir out/wir_wan
```
