"""Prompt-CSV parsing helpers, dependency-free.

Hoisted out of ``data_extraction.run_extraction`` (``WILL_IT_RENDER_PLAN.md`` §10b R1) so
the steering entry points can parse prompt CSVs without importing the whole DDP extraction
module (which pulls the model backends). ``data_extraction.run_extraction`` re-exports these
names for backwards compatibility.
"""
from __future__ import annotations

import csv


def load_prompts(csv_path: str) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def parse_target_words(raw: str, sep: str = ",") -> list[str]:
    return [w.strip() for w in raw.split(sep) if w.strip()]


def normalize_row(row: dict) -> dict:
    """Normalize alternative CSV column names to the canonical format."""
    if "object_name" not in row and "class_name" in row:
        row["object_name"] = row["class_name"].replace(" ", "_")
    if "target_words" not in row and "concepts" in row:
        row["target_words"] = row["concepts"]
        row["_target_words_sep"] = "|"
    return row
