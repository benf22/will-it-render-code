# Baseline method configs (paper comparison columns)

The baselines run through the **same** steering loop as AttnSteer (`steer.py` / `wir-run`),
selected by `steering.model.mode`. A&E and TempoControl are **parameter-free** (no trained probe
/ bundle) — they only need the engine + prompts.

## Attend-and-Excite — `mode: attend_excite` (per-clip induction; main-table A&E column)

| Config | Backbone | scale_factor (lr) |
|--------|----------|-------------------|
| `attend_excite_ltx.yaml` | LTX-2 | 20 |
| `attend_excite_wan.yaml` | WAN | 500 |
| `attend_excite_cogvideox.yaml` | CogVideoX-5B | 500 |

## TempoControl — `mode: tempo_control` (per-frame temporal; temporal-table baseline)

| Config | Backbone | lr (one-object) |
|--------|----------|-----------------|
| `tempocontrol_ltx.yaml` | LTX-2 | 5e-4 |
| `tempocontrol_cogvideox.yaml` | CogVideoX-5B | 5e-4 |

Two-object / action protocols use lr 1e-3 (override `steering.optimization.lr`). The WAN
TempoControl arm was exploratory and is not a formal paper arm.

## Best-of-N — a protocol, not a steering config (main-table Best-of-N column)

Best-of-N generates N **unsteered** clips at different seeds and keeps the one the detector
scores best. The paper uses **best-of-3** (fresh, unbiased seeds). It is not a `steer.py` recipe —
run the two scripts under `src/will_it_render/evaluation/baselines/best_of_n/`:

    make_seeds.py    # deterministic seed->rep map (rep r of prompt p is fixed across N)
    select_best.py   # pick the best clip per prompt over reps 0..N-1; emit the det curve

Generate the N clips with the unsteered pipeline (`wir-run … --n-iters 0` per seed), then select.
