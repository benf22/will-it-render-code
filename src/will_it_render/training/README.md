# training

Train AttnSteer models (one per denoising step) on extracted cross-attention features, and
evaluate their predictive quality. **This is use #2 (train your own AttnSteer)** — paired with
`datasets/` (which extracts the features first).

**Standalone:** imports only `common` and `backends`.

## Pipeline

```bash
# 1. extract features (datasets/) -> out/feats/<clip>/step_NNN.safetensors + extraction_config.yaml
# 2. label the clips (datasets/) -> labels.csv
# 3. train one AttnSteer per step:
python -m will_it_render.training.trainer --config configs/attnsteer.yaml \
    training.train_csv=train.csv training.val_csv=val.csv training.output_dir=out/attnsteer
```

### Producing `train_csv` / `val_csv`

After extraction, label the clips with `video_object_labeler` to get `labels.csv`, then
split it into `train.csv` / `val.csv` with `datasets/splitter.py`:

```bash
# label every clip under the extraction root -> out/feats/labels.csv
torchrun --nproc_per_node=1 -m will_it_render.datasets.video_object_labeler \
    --root out/feats --out out/feats/labels.csv

# split the labels CSV 85/15 into train.csv / val.csv
python -c "import pandas as pd; from will_it_render.datasets.splitter import train_val_split; \
rows = pd.read_csv('out/feats/labels.csv').to_dict('records'); \
tr, va = train_val_split(rows); \
pd.DataFrame(tr).to_csv('train.csv', index=False); pd.DataFrame(va).to_csv('val.csv', index=False)"
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
| `augment.py` | input-space augmentation of attention maps | library |
| `summarize_run.py` | best-epoch summary + plots over a finished run | library |

> These are the *model-quality* metrics (does the AttnSteer predict presence well). The
> *generation-quality* metrics (det-all / det-ratio on videos) live in `evaluation/`.
