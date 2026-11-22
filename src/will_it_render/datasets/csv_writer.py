from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd


def n_objects_in_clip(subdir: str | Path) -> int | None:
    """Number of target objects for a clip, read from its metadata files.

    Prefers ``prompt_info.json`` ``target_words`` (comma-separated), then falls
    back to ``feature_metadata.json`` ``words`` (a list). Returns ``None`` when
    neither source exists — callers must record it as unknown, never guess it
    from the prompt text (downstream segmentation drops unknown n_objects)."""
    subdir = Path(subdir)
    info = subdir / "prompt_info.json"
    if info.exists():
        raw = json.loads(info.read_text()).get("target_words")
        if raw:
            words = [w for w in str(raw).split(",") if w.strip()]
            if words:
                return len(words)
    meta = subdir / "feature_metadata.json"
    if meta.exists():
        words = json.loads(meta.read_text()).get("words") or []
        if words:
            return len(words)
    return None


def write_labels_csv(rows: List[dict], path: str | Path) -> None:
    df = pd.DataFrame(rows)
    df["label"] = df["label"].astype("float32")
    df.to_csv(str(path), index=False)
