# inference

Inference-time attention steering: at chosen denoising steps, an AttnSteer model scores the live
cross-attention and its loss gradient is backpropagated into the latent so the target object
renders. **This is use #1 (run inference).**

**Standalone:** imports only `common` and `backends`.

## Entry points

```bash
# headline: one prompt + one AttnSteer bundle -> a steered video
# --target-words = the word(s) from the prompt to induce (comma-separated; omit = all nouns)
wir-run --bundle <bundle-dir> --prompt "a kite and a baseball bat" \
    --target-words "kite,baseball bat" --seed 1892721232

# config/CSV driven (whole test set); pick the recipe from configs/steering/
python -m will_it_render.inference.steer \
    --config configs/steering/wan.yaml \
    --prompts_csv data/WIR_testset/wir_testset.csv --output_dir out/wir_wan
```

## Files

| File | What it does | Entry point |
|------|--------------|-------------|
| `run.py` | `wir-run` — load a bundle (checkpoints + `extraction_config.yaml` + seed latent) + one prompt → steered video | `wir-run --bundle … --prompt …` |
| `steer.py` | config/CSV-driven steering over many prompts (`load_configs`/`build_backend`/`run_steering`) | `python -m will_it_render.inference.steer --config …` |
| `steer_video_edit.py` | steer an EXISTING video (SDEdit-style, e.g. suppress an object) | `python -m will_it_render.inference.steer_video_edit --config …` |
| `steering_core.py` | model-agnostic optimization loop (`run_steering_optimization`, `SteeringBackendAdapter`, regularizers, the `gated0.95_mutual[_zero]` grad-aggregation rule) | library |
| `steering_model.py` | loads AttnSteer checkpoints, turns captures → steering loss (`build_steering_model`) | library |
| `ltx_adapter.py` / `wan_adapter.py` / `cogx_adapter.py` | per-backend seams onto `steering_core` | library |

## The bundle contract

`wir-run` is not zero-config — it consumes an **artifact bundle**:

```
bundle/  steering.yaml  checkpoints/attnsteer_step_NNN.pt  extraction_config.yaml  [seed_latents/]
```

Assemble one with `python -m scripts.download_artifacts --backbone <wan|ltx|cogvideox|ltx_timeline>`,
or build it from a training run. On the Hub the artifacts are stored by type — `checkpoints/<backbone>/`
(probe weights + `extraction_config.yaml`) and `seed_latents/<backbone>/` — and the steering recipe
is the repo's `configs/steering/<backbone>.yaml`; `download_artifacts` fetches the two artifact dirs
and copies the recipe in to form the bundle above. A bundled seed latent reproduces exactly only on
its tagged GPU type (CUDA RNG is architecture-dependent); any other seed samples noise on the fly.

## Gradient-aggregation rule

The shipped recipe uses `grad_aggregate: gated0.95_mutual` (LTX) / `gated0.95_mutual_zero`
(WAN/CogVideoX) — gate p≥0.95 + mutual-agree fold over survivors (worst-object / zero fallback).
Defined in `steering_core.py`'s `GRAD_AGGREGATE_REGISTRY`.

## Temporal control (control *when* an object is on screen)

The same steering loop can enforce a **per-frame** presence schedule — make an object appear only
during part of the clip (the TempoControl protocol), not just "present somewhere". This uses a
**distinct, per-frame AttnSteer bundle** (a per-frame probe) and the
recipe in `configs/steering/ltx_timeline.yaml`. **LTX-2 only.**

Each prompt row carries a `timelines` column: one `word:schedule` entry per object, entries joined
by `;`, where the schedule is a bit-string (one `0`/`1` per frame, resampled to the probe's latent
frames) — `0` = absent, `1` = present. Example: `cat:0000001111111;bird:1111111111111` (the bird
throughout, the cat only in the second half). A target word with no entry defaults to all-ones
(= the ordinary per-clip "make it render" objective).

```bash
# steer the shipped per-frame test set (one/two/multi-object schedules live in data/)
python -m will_it_render.inference.steer \
    --config configs/steering/ltx_timeline.yaml \
    --prompts_csv data/other_benchmarks/tempocontrol/two_objects.csv \
    --output_dir out/tempo_ltx
```

Score the result with the temporal-control metric (fraction of frames whose detected presence
matches the requested schedule):

```bash
python -m will_it_render.datasets.video_object_labeler --root out/tempo_ltx --out tempo_labels.csv --per-frame
python -c "from will_it_render.evaluation.metrics.temporal_accuracy import *"   # temporal-acc helpers
```

(`wir-run` is for the single-objective per-clip case and does not take `timelines` — use
`inference.steer` for temporal control.)

## Reproducibility: seeds, GPUs, and determinism

There are **two** independent sources of run-to-run variation, handled separately.

**1. The initial latent (generation).** A seed maps to a noise tensor through CUDA's RNG, which
is **GPU-architecture dependent** — the *same* seed produces a *different* initial latent across
different GPU generations, so "seed 42" alone does **not** reproduce a video across machines. We
fix this by **precomputing** the noise once and *loading* it instead of sampling: the per-seed
latent files (`seed_latents/`, see `backends/seed_latents_*.py`) make a given seed yield the same
starting point on any GPU. A bundle ships the latent for its demo seed; for any other seed the
backend samples on the fly (reproducible only *within* one GPU type — see `--seed`). Note this
solves reproducibility of the **starting latent only** — it says nothing about the steering
optimization below.

**2. The steering optimization (inference-time).** Steering backpropagates the AttnSteer loss into
the latent. Some CUDA ops (notably Flash-Attention's backward) are **non-deterministic**, so even
with an identical starting latent the steered result can differ **across GPU types and between
runs on the same GPU**. To reproduce a steered video *exactly* you therefore need **both**: the
same GPU type **and** deterministic mode.

**Deterministic mode** is built in and **on by default** (`enable_determinism()` sets
`torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`). It makes steering bit-reproducible within a GPU type but costs
**~+24% wall-clock**, so turn it off when you don't need exact reproduction (e.g. runtime timing or
bulk generation):

```bash
# exact reproduction (default): just run on the same GPU type
wir-run --bundle <dir> --prompt "a dog on a beach" --seed 42

# faster, non-reproducible run-to-run:
STEER_DETERMINISTIC=0 wir-run --bundle <dir> --prompt "a dog on a beach" --seed 42
```

(`STEER_DETERMINISTIC` accepts `0` / `off` / `false` to disable; anything else — or unset — keeps
it on.) Rule of thumb: **leave determinism on only when you need bit-exact reproduction**; disable
it for speed otherwise.
