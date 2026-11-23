# WIR test set

The 473-prompt multi-object object-presence benchmark (k = 2…7 objects per prompt). See
[`../README.md`](../README.md) for how it was constructed (VBench `multiple_objects` extended
2→7 under a pairwise-uniqueness constraint).

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
