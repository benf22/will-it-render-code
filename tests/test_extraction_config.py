"""Unit tests for the extraction-config resolver (N1 rename: prefer the new name, accept
the legacy one)."""
from __future__ import annotations

from will_it_render.common.extraction_config import (
    CONFIG_NAME,
    LEGACY_NAME,
    find_extraction_config,
)


def test_prefers_new_name(tmp_path):
    (tmp_path / CONFIG_NAME).write_text("a: 1\n")
    (tmp_path / LEGACY_NAME).write_text("a: 2\n")
    assert find_extraction_config(tmp_path) == tmp_path / CONFIG_NAME


def test_accepts_legacy_name(tmp_path):
    (tmp_path / LEGACY_NAME).write_text("a: 1\n")
    assert find_extraction_config(tmp_path) == tmp_path / LEGACY_NAME


def test_missing_returns_none(tmp_path):
    assert find_extraction_config(tmp_path) is None


def test_new_name_has_no_probe_word():
    assert "probe" not in CONFIG_NAME
