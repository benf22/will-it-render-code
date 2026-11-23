# d473 — the object-presence evaluation benchmark

The prompt set behind the paper's main (per-clip object-induction) table. Built from VBench's
object-class vocabulary: multi-object prompts of the form "a {obj} and a {obj} and …", stratified
by object count **k = 2…7**.

## Files

| File | Rows | What |
|------|------|------|
| `d473_multi_object.csv` | **473** | the headline benchmark — 80 prompts each at k=2,3,4,5,6, 73 at k=7 |
| `vbench_single_object.csv` | 79 | VBench single-object class prompts (the single-object table) |

## Columns

`prompt` · `object_name` (underscore-joined object list, used as the label key) ·
`target_words` (comma-separated words to steer / detect) · `seed` (generation seed, 42).

## Use

```bash
# steer every prompt with the WAN "Ours" recipe
wir-run --bundle <wan-bundle> --prompt "$(…)"            # one prompt, or
python -m will_it_render.inference.steer \
    --config configs/steering/wan.yaml \
    --prompts_csv data/d473/d473_multi_object.csv --output_dir out/d473_wan
```

Evaluate the generated clips with `will_it_render.evaluation.metrics` (det-all = all objects
present, det-ratio = per-object detection rate). Each prompt is scored by the external object
detectors against its `target_words`.

> These are prompts only — the generated videos and detector labels are produced by the pipeline,
> not shipped here.
