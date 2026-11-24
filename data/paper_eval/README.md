# paper_eval — annotated clips for the headline table

Per-`(clip, object)` **detector labels** for the paper's §1 four-method comparison, so you can
reproduce the object-presence table **without a GPU, a model, or the detectors**.

**The label CSVs are hosted on Hugging Face** (`benf22/will-it-render`, under `paper_eval/`) rather
than committed here — they are data artifacts, like the AttnSteer bundles. The evaluation driver
fetches and caches them automatically:

```bash
python -m will_it_render.evaluation.paper_table          # fetches paper_eval/ from HF, prints the table
```

To run offline (or from a local copy), point it at a directory of the CSVs:

```bash
python -m will_it_render.evaluation.paper_table --data-dir /path/to/paper_eval
#   or: export VP_PAPER_EVAL_DIR=/path/to/paper_eval
```

## Files (on HF, one per backbone × method)

`<backbone>_<method>.csv` for `backbone ∈ {wan, ltx, cogvideox}` and
`method ∈ {unsteered, ae, ours}` — the three methods whose numbers are a direct aggregate of one
labels file. (**Best-of-N** is a *selection protocol* over N seeds, not a single-file aggregate —
see `will_it_render.evaluation.baselines.best_of_n`.)

Columns: `prompt_idx, prompt, extraction_dir, object_name, n_objects, label`.
- `extraction_dir` — one clip (video) per distinct value; the clip key for aggregation (relative paths).
- `n_objects` — the clip's object count `k` (for the per-k breakdown).
- `label` — GRiT+GDINO+OWL majority vote: `1` present, `0` absent, `-1` tie/uncertain.

## How the numbers are computed (rule C)

The paper's **rule C** drops the **entire clip** when *any* of its objects is a tie (`-1`), then
reports det-all / det-ratio averaged over `k` with each `k` weighted equally. This clip-level tie
drop is what `paper_table.py` implements; note it differs from `metrics_utils`' row-level default
(which drops only the `-1` row). `tests/test_paper_table.py` asserts every cell matches the
published value (skips if the labels aren't fetchable).

Labels generated with the `rgbfix` + rule-C protocol (GRiT+GDINO+OWL @0.15, PIL-negation fix).
Source experiment: `2026_11_02_STEER_EVAL_baseline_4method_3bb`.
