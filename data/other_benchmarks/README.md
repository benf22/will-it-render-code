# Other benchmarks

Auxiliary evaluation sets beyond the main [WIR test set](../WIR_testset/).

## `vbench_single_object.csv`

VBench single-object class prompts ("a {class}") — the **single-object** results table. Columns:
`prompt`, `object_name`, `target_words`.

## `tempocontrol/` — per-frame temporal control

Test sets for the **temporal-control** table (control *when* each object is on screen), following
the TempoControl protocol. Each row carries a `timelines` column: per object, a per-frame on/off
bit-string (e.g. `bird:1111100000;cat:0000011111`).

| File | Protocol | Rows |
|------|----------|------|
| `one_object.csv` | one object appears on a schedule | 80 |
| `two_objects.csv` | two objects, independent schedules (static / dynamic) | 82 |
| `multiobjects.csv` | multi-object, all-present schedule | 82 |

Used to evaluate both the per-frame "Ours" (`configs/steering/ltx_timeline.yaml`) and the
TempoControl baseline (`configs/baselines/tempocontrol_*.yaml`).
