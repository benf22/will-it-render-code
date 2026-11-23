"""Smoke + golden test for the evaluation entry point's aggregation path (CPU, no detectors)."""
from __future__ import annotations

import pandas as pd

from will_it_render.evaluation import run as R
from will_it_render.evaluation.metrics import metrics_utils as M


def _labels():
    # 2 clips: clip A (2 objects, both present), clip B (2 objects, one present).
    return pd.DataFrame([
        {"extraction_dir": "A", "object_name": "dog", "label": 1},
        {"extraction_dir": "A", "object_name": "cat", "label": 1},
        {"extraction_dir": "B", "object_name": "dog", "label": 1},
        {"extraction_dir": "B", "object_name": "cat", "label": 0},
    ])


def test_summarize_golden():
    df = _labels()
    assert M.all_objects_present_rate(df) == 0.5        # only clip A has all present
    assert M.obj_level_detection_rate(df) == 0.75       # 3 of 4 (clip,object) present


def test_per_k_breakdown():
    pk = R._per_k(_labels())
    assert pk is not None
    row = pk[pk["k"] == 2].iloc[0]
    assert row["n_clips"] == 2 and row["det_all"] == 0.5
