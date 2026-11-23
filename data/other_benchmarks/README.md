# Other benchmarks

Auxiliary evaluation sets beyond the main [WIR test set](../WIR_testset/).

## `vbench_single_object.csv`

VBench single-object class prompts ("a {class}") — the **single-object** results table. Columns:
`prompt`, `object_name`, `target_words`.

## `tempocontrol/` — per-frame temporal control

Test sets for the **temporal-control** table (control *when* each object is on screen), following
the TempoControl protocol.

| File | Protocol | Rows |
|------|----------|------|
| `one_object.csv` | one object appears on a schedule | 80 |
| `two_objects.csv` | two objects, independent schedules (static / dynamic) | 82 |
| `multiobjects.csv` | multi-object, all-present schedule | 82 |

### Columns

| Column | Used by | Meaning |
|--------|---------|---------|
| `prompt` | LTX-2 & CogVideoX | the prompt (the long "A video of an empty scene. Suddenly…" phrasing) |
| `WAN_prompt` | **WAN only** | WAN needs a shorter phrasing (and runs 21 frames), so it uses this adapted prompt instead of `prompt`; everything else (schedule, targets) is identical. Present in `one_object`/`two_objects` only (WAN was not run on the multi-object set). |
| `target_words` | all | comma-separated words to steer / detect |
| `timelines` | **Ours** (per-frame) | per-object per-frame on/off bit-string, e.g. `bird:0001111111111` — the schedule our steerer follows |
| `control_signal` / `control_signal1` / `control_signal2` | **TempoControl baseline** | the same schedule in TempoControl's native per-object format |

So `timelines` and `control_signal` are the **same schedule in two formats** — one read by our
method, one by the baseline. (The research CSVs also carried construction metadata — `Object 1/2`,
`temp_object`, `static_object`, `original_prompt` — dropped here as they're not read at run time.)

Used to evaluate both the per-frame "Ours" (`configs/steering/ltx_timeline.yaml`) and the
TempoControl baseline (`configs/baselines/tempocontrol_*.yaml`).
