# training

Trains AttnSteer models (one per denoising step) on the extracted cross-attention features
and evaluates their predictive quality.

**Standalone (N2):** imports only `common` and `backends`.

- architectures — conv / mlp / resnet / attn_pool / maxpool / transformer (the
  `build_attnsteer` factory itself lives in `common`).
- `trainer.py` — single-node training (`train_all_steps` / `train_one_step`).
- `feature_dataset.py` (was `probe_dataset.py`) — lazy safetensors streaming.
- `evaluation.py` — **the AttnSteer-quality metrics module**: AUROC, average precision,
  precision/recall/F1, balanced accuracy, and per-frame (all-frames / any-frame) AUROC. These
  are the "how good is the model" metrics (plan §10c N3) — distinct from the generation-quality
  metrics under `evaluation/`.
- `adversarial.py`, `augment.py` — robustness / augmentation during training.

Config key: `model_arch` (was `probe_model`). Checkpoints carry every arg the factory needs;
class names and state-dict keys are unchanged from the source so existing trained checkpoints
load as-is (plan §10c N1).

> Scaffold — ported in Phase 2b.
