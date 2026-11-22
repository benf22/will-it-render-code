# configs

Sanitized configuration (plan §10b R4 / §10c N4). All personal paths become env-vars or
placeholders; the ~20 experiment variants in the source collapse to **one clean default per
backend**.

- `attnsteer.yaml` (was `probing.yaml`) — main experiment config; config key `model_arch`
  (was `probe_model`).
- `models/{ltx,wan,cogvideox}.yaml` — per-backend extraction defaults + engine paths (env-var
  resolved).
- one **canonical steering config per backend** pinned to the paper's headline recipe (the WAN
  one is pinned last — its recipe is still being finalized).

> Scaffold — populated in Phase 5.
