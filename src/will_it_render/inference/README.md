# inference

Inference-time attention steering: at chosen denoising steps, an AttnSteer model scores the live
cross-attention and its loss gradient is backpropagated into the latent so the target object
renders. **This is use #1 (run inference).**

**Standalone (N2):** imports only `common` and `backends`.

## Entry points

```bash
# headline: one prompt + one AttnSteer bundle -> a steered video
wir-run --bundle <bundle-dir> --prompt "a dog on a beach" --seed 42

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

## The bundle contract (R2)

`wir-run` is not zero-config — it consumes an **artifact bundle**:

```
bundle/  steering.yaml  checkpoints/attnsteer_step_NNN.pt  extraction_config.yaml  [seed_latents/]
```

Fetch one with `scripts/download_artifacts.py`, or build it from a training run. The committed
demo seed latent reproduces exactly only on its tagged GPU type (CUDA RNG is architecture-
dependent); `--seed N` regenerates noise on the fly for any other seed.

## Gradient-aggregation rule

The shipped recipe uses `grad_aggregate: gated0.95_mutual` (LTX) / `gated0.95_mutual_zero`
(WAN/CogVideoX) — gate p≥0.95 + mutual-agree fold over survivors (worst-object / zero fallback).
Defined in `steering_core.py`'s `GRAD_AGGREGATE_REGISTRY`.

## Determinism

On by default (bit-reproducible within a GPU type). `STEER_DETERMINISTIC=0` opts out (~24%
faster; used for runtime timing).
