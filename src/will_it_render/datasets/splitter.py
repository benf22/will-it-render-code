from __future__ import annotations

import random
from typing import List, TypeVar

T = TypeVar("T")


def train_val_split(
    rows: List[T],
    val_fraction: float = 0.15,
    seed: int = 42,
    stratify_key=None,  # reserved for future use
) -> tuple[List[T], List[T]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_fraction)
    return shuffled[n_val:], shuffled[:n_val]
