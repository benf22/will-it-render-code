# inference

Inference-time attention steering: at chosen denoising steps, an AttnSteer model scores the
live cross-attention and its loss gradient is backpropagated into the current-step latent to
make the target class more (or less) likely to render.

**Standalone (N2):** imports only `common` and `backends`.

- `steering_core.py` — model-agnostic optimization loop (`run_steering_optimization`,
  `SteeringBackendAdapter` ABC, regularizers, grad-aggregation rules).
- `steering_model.py` — loads AttnSteer checkpoints, turns captures → steering loss.
- `{ltx,wan,cogx}_adapter.py` — per-backend seams.
- `steer.py` — config/CSV-driven entry point.
- `run.py` — **the headline entry point**: one prompt + one AttnSteer bundle + `--seed` →
  steered video.

## The bundle contract (plan §10b R2)

`run.py` is not a zero-config one-liner. It consumes an **artifact bundle** (from Hugging Face
Hub or a local path):

```
bundle/
  attnsteer_step_NNN.pt     # per-step AttnSteer checkpoints
  extraction_config.yaml    # capture layout the checkpoints were trained on (FULL-block extraction)
  seed_latent.*             # a matched, GPU-tagged seed latent
```

`run.py` hardcodes sane determinism defaults (and must not trip the determinism guard when
`STEER_DETERMINISTIC` is unset). The committed demo bundle reproduces exactly only on the GPU
type its seed latent was tagged for; `--seed N` for any other seed regenerates noise on the
fly (reproducible only on the same GPU type — CUDA RNG is architecture-dependent).

The grad-aggregation rule glossary (ported from the source `steering/RULE_NAMES.md`) lives
here.

> Scaffold — loop + `run.py` ported in Phases 2b / 4.
