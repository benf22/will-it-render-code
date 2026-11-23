# will-it-render

**AttnSteer** — train light **attention-steering models** (linear / MLP / conv) on a video
diffusion transformer's cross-attention maps to predict whether a target class (e.g. a
barking dog) will appear in the generated video, then use those models to **steer**
generation at inference time so the class *does* render.

Backends: **LTX-2**, **WAN (Wan2.1)**, **CogVideoX**.

> **Status: migrated, GPU-validated, pre-release.** The full module tree (common / backends /
> inference / datasets / training / evaluation + baselines), configs, packaging, docs, tests,
> and CI are in place; the whole package imports cleanly and the CPU suite is green.
> **GPU-verified on an L40S:** WAN runs end-to-end — `wir-run` produces a steered video (both
> `steer.py` and the bundle entry point, determinism on and off), and the 7 WAN capture-hook
> tests pass against the real engine. The LTX env-var path resolution and the CogVideoX
> attention hook + backend are validated; a *complete* steered LTX/CogVideoX clip needs a
> larger card (both are B200/H200-class — the 48 GB L40S OOMs on their steering grad). Probe
> weights / bundles are not yet uploaded. See `WILL_IT_RENDER_PLAN.md` in the source repo.

## Why "will it render"

An AttnSteer model reads one denoising step's cross-attention toward the target word and
predicts the probability the object will be present in the final video — *will it render?*
At inference we backpropagate that model's loss into the current-step latent to make the
answer yes (induce the object) or no (suppress it).

## Install

```bash
pip install -e .                 # core (AttnSteer + feature I/O + configs)
pip install -e ".[all]"          # + detectors, audio, baselines, vbench (heavy)
```

The DiT engines are **not** pip dependencies — point the package at your local checkouts:

```bash
export VP_LTX_ROOT=/path/to/LTX-2
export VP_WAN_ROOT=/path/to/Wan2.1
export VP_COGX_ROOT=/path/to/CogVideoX     # or rely on pip `diffusers` for CogVideoX
```

Pinned upstream commits are documented per backend in `src/will_it_render/backends/README.md`.

## Quick start (planned)

```bash
# one prompt + one AttnSteer bundle (checkpoint + extraction config + seed latent) → steered video
wir-run --backend wan --bundle <hf-bundle-or-path> --prompt "a dog on a beach" --seed 42
```

See `src/will_it_render/inference/README.md` for the bundle contract and determinism notes.

## Layout

| Path | Purpose |
|------|---------|
| `src/will_it_render/common/` | dependency-sink: AttnSteer factory, feature aggregation, prompt/config helpers, env-var path resolver |
| `src/will_it_render/backends/` | DiT backend adapters + attention-capture hooks (LTX / WAN / CogVideoX) + seed latents |
| `src/will_it_render/inference/` | steering optimization loop + `run.py` entry point |
| `src/will_it_render/datasets/` | prompt generation, feature extraction, labeling |
| `src/will_it_render/training/` | AttnSteer architectures + trainer + AUROC/AP/F1 evaluation |
| `src/will_it_render/evaluation/` | det-all / det-ratio / VBench / runtime metrics + baselines |
| `configs/` · `scripts/` · `examples/` · `tests/` | sanitized configs, thin CLIs, demo data, test suite |

Each `src/will_it_render/*` subpackage is **standalone** (imports only `common/` and
`backends/`, never a sibling) and carries its own README.

## License

MIT (see `LICENSE`). Third-party components under other licenses (Apache-2.0, AGPL, model
community licenses) are listed in `THIRD_PARTY.md`.
