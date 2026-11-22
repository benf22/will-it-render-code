"""Unit tests for the env-var engine/artifact resolver (common.paths)."""
from __future__ import annotations

import pytest

from will_it_render.common import paths


def test_engine_root_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VP_WAN_ROOT", str(tmp_path))
    assert paths.wan_root() == tmp_path


def test_engine_root_override_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("VP_WAN_ROOT", "/does/not/matter")
    assert paths.wan_root(override=str(tmp_path)) == tmp_path


def test_engine_root_unset_raises_actionable(monkeypatch):
    monkeypatch.delenv("VP_LTX_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="VP_LTX_ROOT"):
        paths.ltx_root()


def test_engine_root_nonexistent_raises(monkeypatch):
    monkeypatch.setenv("VP_COGX_ROOT", "/no/such/dir/xyz")
    with pytest.raises(FileNotFoundError):
        paths.cogx_root()


def test_unknown_engine():
    with pytest.raises(KeyError):
        paths.engine_root("svd")


def test_seed_latents_dir_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VP_SEED_LATENTS_DIR", str(tmp_path / "sl"))
    assert paths.seed_latents_dir() == tmp_path / "sl"
    assert (tmp_path / "sl").is_dir()
