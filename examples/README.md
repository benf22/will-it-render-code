# examples

- `example_prompts/` — small prompt CSVs for the extraction / steering demos.
  `wir_testset_sample.csv` is a handful of WIR-test-set prompts at their real per-prompt seeds,
  ready to feed to `wir-run` / `will_it_render.inference.steer`.

Trained probe checkpoints and precomputed seed latents are **not** committed here — fetch them
from the Hugging Face Hub with `python -m scripts.download_artifacts --backbone <bb>` (see the
top-level README quick-start). The CPU tests use synthetic tensors, so they need no external data.
