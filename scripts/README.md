# scripts

Thin single-node CLI wrappers.

- `download_artifacts.py` — fetch a backbone's AttnSteer artifacts (checkpoints + extraction
  config + seed latents) from the Hugging Face Hub and assemble the bundle that `wir-run`
  consumes: `python -m scripts.download_artifacts --backbone <wan|ltx|cogvideox|ltx_timeline>`.
