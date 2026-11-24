# Third-party code and licenses

The project is MIT-licensed (see `LICENSE`), but it incorporates or depends on third-party
components under other licenses. This file records their provenance and notices.

## Vendored / adapted code (ships in this repo)

| Component | Where | Upstream | License | Notes |
|-----------|-------|----------|---------|-------|
| CogVideoX attention hook | `backends/hooks/cogvideox_hooks.py` | 🤗 diffusers | **Apache-2.0** | Body copied ~verbatim from the diffusers `CogVideoXAttnProcessor2_0`; keep the attribution header + reproduce the Apache-2.0 NOTICE. MIT alone is insufficient. |
| Attend-and-Excite baseline | `evaluation/baselines/attend_excite/loss.py` | Attend-and-Excite (paper) | per-paper | Clean reimplementation; keep per-paper attribution header. |
| TempoControl baseline | `evaluation/baselines/tempocontrol/loss.py` | TempoControl (arXiv:2510.02226) | per-paper | Clean reimplementation (loss only; upstream fork NOT vendored). Keep attribution header. |

## Runtime / optional dependencies (not vendored)

| Dependency | Used by | License | Notes |
|------------|---------|---------|-------|
| YOLOv10 / ultralytics | detectors extra (labeling) | **AGPL-3.0** | AGPL is copyleft — flagged for users; isolated behind the `detectors` extra. |
| VBench | vbench extra (evaluation) | **Apache-2.0** | Video quality metrics. |
| DiT engines: LTX-2 / Wan2.1 / CogVideoX | backends (external checkouts) | respective upstream | Resolved via `VP_*_ROOT`; not redistributed here. |
