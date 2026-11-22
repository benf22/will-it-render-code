# backends

DiT backend adapters and attention-capture hooks for the three supported engines.

- `model_backend.py` — `ModelBackend` ABC + `create_backend()` factory (LTX / WAN / CogVideoX).
- `hooks/` — attention-capture hooks that monkeypatch each engine's attention to materialize
  `softmax(QKᵀ)` for capture and backprop:
  - `ltx_hooks.py` (was `ltx_probe.py`)
  - `wan_hooks.py` (was `wan_probe.py`; installs an SDPA shim for deterministic backward)
  - `cogvideox_hooks.py` (was `cogvideox_probe.py`; body adapted from diffusers, **Apache-2.0**
    — see `THIRD_PARTY.md`).
- `seed_latents.py` — precompute + on-the-fly generation of replayable seed noise.

## Engine resolution (plan §10b R5)

Engines are **external checkouts**, located via env vars — not pip packages, not `sys.path`
injection:

```bash
export VP_LTX_ROOT=/path/to/LTX-2
export VP_WAN_ROOT=/path/to/Wan2.1
export VP_COGX_ROOT=/path/to/CogVideoX
```

Hooks patch private upstream internals by name, so each adds an **import-time
version/attribute self-check** that fails loudly if the pinned upstream API has moved.

### Pinned upstream commits

| Engine | Env var | Pinned commit | Notes |
|--------|---------|---------------|-------|
| LTX-2 | `VP_LTX_ROOT` | TODO | |
| Wan2.1 | `VP_WAN_ROOT` | TODO | user fork; SDPA attention |
| CogVideoX | `VP_COGX_ROOT` | TODO | or pip `diffusers` |

> Scaffold — hooks + resolver ported in Phase 3.
