from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ObjectEntry:
    object_name: str
    target_words: List[str]


def read_objects_csv(path: str | Path) -> List[ObjectEntry]:
    """Read a CSV with columns: object_name, target_words (comma-separated)."""
    entries: List[ObjectEntry] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            words = [w.strip() for w in row["target_words"].split(",") if w.strip()]
            entries.append(ObjectEntry(object_name=row["object_name"].strip(), target_words=words))
    return entries
