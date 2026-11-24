# configs

One config per backend. Point the paths (env-vars / placeholders) at your own engine checkouts and model weights.

- `attnsteer.yaml` (was `probing.yaml`) — main experiment config; config key `model_arch`
  (was `probe_model`).
- `models/{ltx,wan,cogvideox}.yaml` — per-backend extraction defaults + engine paths (env-var
  resolved).
- one **canonical steering config per backend** pinned to the paper's headline recipe.
