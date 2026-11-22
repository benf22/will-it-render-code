# evaluation

Generation-quality metrics for steered video + baseline comparisons. (The *AttnSteer-model*
quality metrics — AUROC/AP/F1 — live in `training/evaluation.py`, not here; plan §10c N3.)

**Standalone (N2):** imports only `common` and `backends`.

## `metrics/` — the four shipped metrics (N3)

- **det-all** — `all_objects_present_rate` (every target object present in the clip).
- **det-ratio** — `obj_level_detection_rate` (+ `clip_level_detection_rate`).
- **VBench** — `vbench_eval` (optional `.[vbench]` extra; never import-time-required).
- **runtime** — `runtime_bench` (NEW thin single-node timing collector; timing runs use
  `STEER_DETERMINISTIC=0`, which is ~24% faster than the deterministic path).

`temporal_accuracy` is kept here iff the TempoControl comparison numbers ship (baselines are
in scope → likely kept). `photorealism_clip` / `semantic_drift` only if realism/OOD numbers
appear on the project page.

## `baselines/` — runnable comparisons

Clean reimplementations driven through the same steering loop as AttnSteer, each with **one
sanitized config** so the comparison is reproducible (plan §10c N5):
- `attend_excite/` — Attend-and-Excite (`loss.py` + `selfcheck.py`).
- `tempocontrol/` — TempoControl (`loss.py` + `selfcheck.py`); the upstream fork / `asis_shim`
  is **not** ported — the README points to upstream for the authors' exact code.
- `best_of_n/` — seed-search protocol.

> Scaffold — metrics + baselines ported in Phase 2b; `runtime_bench` is new work.
