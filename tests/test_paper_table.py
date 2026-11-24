"""The shipped annotated-clip labels reproduce the paper's §1 det-all / det-ratio table.

CPU-only, no model: aggregates ``data/paper_eval/*.csv`` under rule C (see paper_table.py) and
asserts every (backbone, method) cell matches the published number. This is the self-contained
"run the evaluation over the annotated clips → get the paper's results" check — it needs neither
a GPU nor the original repo.
"""
from __future__ import annotations

import pytest

from will_it_render.evaluation.paper_table import PAPER, evaluate


def test_paper_table_reproduces_published_numbers():
    try:
        res = evaluate()
    except Exception as e:  # labels not present locally and not fetchable (offline / not uploaded)
        pytest.skip(f"paper_eval labels unavailable (local or HF): {e}")
    if not res:  # resolved a location but it has no label files yet (e.g. HF upload pending)
        pytest.skip("paper_eval labels not found at the resolved location")
    missing = [cell for cell in PAPER if cell not in res]
    assert not missing, f"missing shipped labels for {missing}"
    for cell, (pa, pr) in PAPER.items():
        da, dr, _ = res[cell]["avg"]
        assert abs(da * 100 - pa) < 0.15, f"{cell} det-all {da*100:.2f} != paper {pa}"
        assert abs(dr * 100 - pr) < 0.15, f"{cell} det-ratio {dr*100:.2f} != paper {pr}"


def test_rule_c_drops_tie_clips():
    import pandas as pd
    from will_it_render.evaluation.paper_table import rule_c_per_k
    # one clean k=2 clip (both present) + one tie clip (dropped entirely by rule C)
    df = pd.DataFrame({
        "extraction_dir": ["clipA", "clipA", "clipB", "clipB"],
        "object_name": ["x", "y", "x", "y"],
        "n_objects": [2, 2, 2, 2],
        "label": [1, 1, 1, -1],   # clipB has a tie -> dropped
    })
    r = rule_c_per_k(df)
    assert r[2] == (1.0, 1.0, 1)   # only clipA counts
