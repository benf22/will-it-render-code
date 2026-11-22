"""Resolve external DiT-engine checkouts and artifact locations from env vars.

Replaces the source repo's three incompatible mechanisms (LTX ``parents[3]`` directory
layout, WAN ``engine.wan_root``-or-``~/repos/Wan2.1``, CogVideoX pip-diffusers) with one
explicit contract (plan §10b R5). The engines are NOT pip packages — they are local
checkouts that each backend imports after their root is placed on ``sys.path``.

    export VP_LTX_ROOT=/path/to/LTX-2
    export VP_WAN_ROOT=/path/to/Wan2.1
    export VP_COGX_ROOT=/path/to/CogVideoX     # optional if CogVideoX comes from pip diffusers

Resolution order for an engine root: an explicit ``override`` (e.g. a config's
``engine.wan_root``) → the env var → a clear, actionable error.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ENGINES = {
    "ltx": ("VP_LTX_ROOT", "LTX-2"),
    "wan": ("VP_WAN_ROOT", "Wan2.1"),
    "cogx": ("VP_COGX_ROOT", "CogVideoX"),
}


def engine_root(which: str, override: str | os.PathLike | None = None) -> Path:
    """Locate a DiT engine checkout. ``which`` ∈ {'ltx','wan','cogx'}."""
    if which not in _ENGINES:
        raise KeyError(f"unknown engine {which!r}; choose one of {sorted(_ENGINES)}")
    env_var, name = _ENGINES[which]
    if override:
        root = Path(override).expanduser()
        if not root.exists():
            raise FileNotFoundError(
                f"{name} root {root} (from config override) does not exist."
            )
        return root
    val = os.environ.get(env_var)
    if not val:
        raise RuntimeError(
            f"Cannot locate the {name} engine. Set {env_var} to your {name} checkout, e.g.\n"
            f"    export {env_var}=/path/to/{name}"
        )
    root = Path(val).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"{env_var}={root} does not exist.")
    return root


def add_to_syspath(root: str | os.PathLike) -> None:
    """Prepend ``root`` to ``sys.path`` once (how the engine packages are imported)."""
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


def ltx_root(override=None) -> Path:
    return engine_root("ltx", override)


def wan_root(override=None) -> Path:
    return engine_root("wan", override)


def cogx_root(override=None) -> Path:
    return engine_root("cogx", override)


def config_dir() -> Path:
    """Locate the ``configs/`` directory (holding ``models/<backend>.yaml``).

    ``VP_CONFIG_DIR`` → that path; else walk up from this package looking for a directory
    that contains ``configs/models`` (works from a source checkout). Raises if not found —
    set ``VP_CONFIG_DIR`` for an installed package whose configs live elsewhere.
    """
    env = os.environ.get("VP_CONFIG_DIR")
    if env:
        return Path(env).expanduser()
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "configs"
        if (candidate / "models").is_dir():
            return candidate
    raise RuntimeError(
        "Could not locate the configs/ directory. Set VP_CONFIG_DIR to the directory "
        "containing models/<backend>.yaml."
    )


def artifact_cache_dir() -> Path:
    """Where fetched AttnSteer bundles / probe weights / cached features live.

    ``VP_ARTIFACT_CACHE`` → that path; else ``~/.cache/will-it-render``. Created on demand.
    """
    base = os.environ.get("VP_ARTIFACT_CACHE")
    path = Path(base).expanduser() if base else Path.home() / ".cache" / "will-it-render"
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_latents_dir() -> Path:
    """Default output root for precomputed seed latents.

    ``VP_SEED_LATENTS_DIR`` → that path; else ``<artifact_cache_dir>/seed_latents``.
    """
    base = os.environ.get("VP_SEED_LATENTS_DIR")
    path = Path(base).expanduser() if base else artifact_cache_dir() / "seed_latents"
    path.mkdir(parents=True, exist_ok=True)
    return path
