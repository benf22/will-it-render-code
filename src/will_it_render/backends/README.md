# backends

DiT backend adapters + attention-capture hooks for the three engines. Shared by `datasets`
(extraction) and `inference` (steering); not an entry point itself.

## Engine resolution

Engines are **external checkouts** located via env vars (we ship only the hooks). See the main
README's install table for clone commands + versions.

| Engine | Env var | Pinned commit | torch |
|--------|---------|---------------|-------|
| LTX-2 | `VP_LTX_ROOT` | `1799988` (github.com/Lightricks/LTX-2) | 2.9.1 cu128 |
| Wan2.1 | `VP_WAN_ROOT` | `7c81b2f` (github.com/benf22/Wan2.1, SDPA fork) | 2.9.1 cu128 |
| CogVideoX | `VP_COGX_ROOT` (optional) | pip `diffusers==0.39` | 2.11 cu128 |

Hooks patch private upstream internals by name, so each does an import-time attribute self-check
that fails loudly if the pinned API moved.

## Files

| File | What it does |
|------|--------------|
| `model_backend.py` | `ModelBackend` ABC + `create_backend()` factory (LTX / WAN / CogVideoX); loads checkpoints, path-injects the engine via `common.paths` |
| `hooks/ltx_hooks.py` | LTX cross-attention capture context |
| `hooks/wan_hooks.py` | WAN capture + SDPA shim for its fused attention (deterministic backward) |
| `hooks/cogvideox_hooks.py` | CogVideoX joint-attention capture (adapted from diffusers, **Apache-2.0** — see THIRD_PARTY.md) |
| `seed_latents.py` / `seed_latents_wan.py` / `seed_latents_cogx.py` | precompute replayable per-seed noise latents (GPU-tagged) |

```bash
# precompute a seed latent (so a seed reproduces the same clip on that GPU type)
VP_LTX_ROOT=/path/to/LTX-2 python -m will_it_render.backends.seed_latents --seeds 42 --out-root ./seed_latents
```
