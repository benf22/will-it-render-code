# evaluation

Measure **object presence** (and quality) in generated videos — the paper's headline metrics.
Works on clips from **any** model, so you can evaluate your own generations, not just ours.

**Standalone (N2):** imports only `common` and `backends`.

## Entry point

```bash
# (A) you already have per-(clip, object) detector labels -> aggregate instantly (no heavy deps)
python -m will_it_render.evaluation.run --labels labels.csv

# (B) you have a directory of clips -> detect objects, then aggregate ([detectors] extra)
python -m will_it_render.datasets.video_object_labeler --root out/clips --out labels.csv   # step 1: detect
python -m will_it_render.evaluation.run --labels labels.csv                                 # step 2: metrics
```

Prints **det-all** / **det-ratio** / clip-level detection rate, plus a per-k breakdown.

```bash
# (C) reproduce the paper's §1 table from the shipped annotated clips (no GPU, no model)
python -m will_it_render.evaluation.paper_table    # det-all / det-ratio per backbone × method
```

`paper_table` aggregates the labels under [`data/paper_eval/`](../../../data/paper_eval/) under
**rule C** (drop a clip if any object is a detector tie `-1`; average per-k) and reproduces the
published numbers exactly — see `tests/test_paper_table.py`. (Rule C's clip-level tie drop
differs from the row-level default below; `data/paper_eval/README.md` explains why.)

## Metrics (`metrics/metrics_utils.py` — single source of truth)

| Name | Function | Meaning |
|------|----------|---------|
| **det-all** | `all_objects_present_rate` | fraction of clips where **every** requested object is present |
| **det-ratio** | `obj_level_detection_rate` | fraction of (clip, object) pairs present (objects weighted equally) |
| clip-level | `clip_level_detection_rate` | per-clip mean (clips weighted equally) |

Labels are per-`(clip, object)` rows with `label` ∈ {1 present, 0 absent, −1 uncertain};
`−1` is ignored. `extraction_dir` keys one clip.

## Files

| File | What it does | Entry point |
|------|--------------|-------------|
| `run.py` | aggregate labels → det-all / det-ratio (+ per-k) | `python -m will_it_render.evaluation.run --labels labels.csv` |
| `metrics/metrics_utils.py` | the detection-metric definitions | library (`summarize(df)`) |
| `metrics/temporal_accuracy.py` | temporal-control accuracy (TempoControl protocol) | library / per-frame eval |
| `metrics/vbench_eval.py` | VBench quality dims (optional `[vbench]` extra) | `python -m will_it_render.evaluation.metrics.vbench_eval …` |
| `baselines/` | runnable comparison methods (A&E, TempoControl, best-of-N) — see `configs/baselines/` | via `will_it_render.inference.steer` |

> Runtime timing of steering (wall-clock/clip) is reported with `STEER_DETERMINISTIC=0`
> (~24% faster than the deterministic path).
