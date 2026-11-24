"""Locate the per-extraction geometry config written at extraction time.

Renamed ``probe_config.yaml`` → ``extraction_config.yaml`` (no "probe" on
public surfaces). Freshly written extractions / HF bundles use the new name; the resolver
still accepts the legacy name so older extraction roots keep working.
"""
from __future__ import annotations

from pathlib import Path

CONFIG_NAME = "extraction_config.yaml"
LEGACY_NAME = "probe_config.yaml"


def find_extraction_config(directory: str | Path) -> Path | None:
    """Return the extraction-config path in ``directory`` (new name preferred, legacy
    accepted), or ``None`` if neither exists."""
    d = Path(directory)
    new = d / CONFIG_NAME
    if new.exists():
        return new
    legacy = d / LEGACY_NAME
    if legacy.exists():
        return legacy
    return None
