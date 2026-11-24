# Canonical steering configs (paper "Ours")

Each backend's shipped recipe is the paper's headline steering configuration. Each is a flat, standalone
steering recipe; the matching trained AttnSteer weights ship as an **HF bundle** (checkpoints +
`extraction_config.yaml` + seed latent) that `wir-run --bundle …` / `scripts/download_artifacts.py`
supply — the `checkpoints_dir` / `extraction_config` paths here are placeholders `wir-run` overrides.

## Per-clip object induction (main table: det-all / det-ratio)

| Config | Formal version | Paper det-all / det-ratio | GPU |
|--------|----------------|---------------------------|-----|
| `ltx.yaml` | LTX_ours_mutualFix | 0.755 / 0.939 | ~84 GB |
| `wan.yaml` | WAN_ours_mutualFix_rgbfixProbe | 0.599 / 0.893 | ~45 GB |
| `cogvideox.yaml` | CogVideoX_ours_mutualFix | 0.659 / 0.912 | ~477 s/clip; determinism ON |

All three use the mutual-gated rule (`gated0.95_mutual[_zero]` + `worst_first` + `anchor` fold).

## Per-frame temporal control (temporal table: temporal-acc / mean7)

| Config | Formal version | Paper temporal-acc (1/2/multi-obj) |
|--------|----------------|-------------------------------------|
| `ltx_timeline.yaml` | LTX_perframe_v3 (bce_huber_m2) | 0.681 / 0.841 / 0.913 |

LTX-only in the paper (WAN per-frame probe trained but no formal steering; CogVideoX none). Uses a
distinct per-frame probe bundle and a per-prompt `timelines` schedule column.
