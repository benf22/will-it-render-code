# examples

- `example_prompts/` — small prompt CSVs for extraction / steering demos.
- seed-42 latents (3, one per backbone, ~8 MB each) — committed for an out-of-box exact-repro
  demo (force-added past `.gitignore`'s `*.safetensors` rule). Every other seed/latent is
  regenerated on the fly. Each committed latent is GPU-tagged and reproduces exactly only on
  that GPU type (CUDA RNG is architecture-dependent — see the inference README).
- a tiny committed extracted-feature sample so the CPU tests run without any external models.

> Scaffold — populated alongside Phases 4 / 7.
