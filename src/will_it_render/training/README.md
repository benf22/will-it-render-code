# training

Train AttnSteer models (one per denoising step) on extracted cross-attention features, and
evaluate their predictive quality. **This is use #2 (train your own AttnSteer)** — paired with
`datasets/` (which extracts the features first).

**Standalone (N2):** imports only `common` and `backends`.

## Pipeline

```bash
# 1. extract features (datasets/) -> out/feats/<clip>/step_NNN.safetensors + extraction_config.yaml
# 2. label the clips (datasets/) -> labels.csv
# 3. train one AttnSteer per step:
python -m will_it_render.training.trainer --config configs/attnsteer.yaml \
    training.train_csv=train.csv training.val_csv=val.csv training.output_dir=out/attnsteer
```

The config key `model_arch` picks the architecture (`conv | mlp | resnet | attn_pool | maxpool
| transformer`); the trained `attnsteer_step_NNN.pt` + `extraction_config.yaml` + a seed latent
form the inference bundle.

## Files

| File | What it does | Entry point |
|------|--------------|-------------|
| `trainer.py` | `train_all_steps` / `train_one_step` + the Hydra CLI | `python -m will_it_render.training.trainer …` |
| `feature_dataset.py` | lazy safetensors streaming (`FeatureDataset`); causal-VAE per-frame target mapping | library |
| `conv.py` / `mlp.py` / `resnet.py` / `attn_pool.py` / `maxpool.py` / `transformer.py` | the AttnSteer architectures | library (via `common.attnsteer_factory.build_attnsteer`) |
| `evaluation.py` | **AttnSteer-quality metrics**: AUROC, average precision, P/R/F1, balanced acc, per-frame (all-/any-frame) AUROC | library (`evaluate_probe`) |
| `adversarial.py` / `augment.py` | PGD adversarial training on attention maps; augmentation | library |
| `summarize_run.py` | best-epoch summary + plots over a finished run | library |

> These are the *model-quality* metrics (does the AttnSteer predict presence well). The
> *generation-quality* metrics (det-all / det-ratio on videos) live in `evaluation/`.
