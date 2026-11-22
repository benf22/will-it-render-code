"""Canonical object-detection metrics derived from a labels CSV.

Single source of truth for the detection metrics computed from per-``(clip, object)``
rows whose binary ``label`` is 1 (present), 0 (absent), or -1 (uncertain). Import the
name constants + functions here instead of re-deriving the logic, so every tool
(``labels_to_serve``, ``serve_results``, report scripts) agrees on **names** and
**aggregation**.

``label`` semantics (from ``labeling.video_object_labeler``): majority vote of the
per-classifier (grit/gdino/owl) decisions; ``-1`` = tie/empty = uncertain.

Aggregation vocabulary:
  * **clip-level**   — average over clips (each video weighted equally).
  * **object-level** — pooled over all ``(clip, object)`` rows (each object instance
    weighted equally; == clip-level weighted by the clip's object count).

``-1`` handling: ``clip_*`` and ``obj_level_detection_rate`` use ``_valid`` rows only
(label present and ``!= -1``), so **-1 is ignored entirely** — not in the numerator and
not in the denominator. ``all_objects_present`` resolves a clip as: a definite ``0`` on
**any** object → ``0`` (clip fails, even if other objects are ``-1``); otherwise all
objects ``1`` → ``1``; otherwise (no ``0`` but ≥1 ``-1``) the clip is **uncertain and
excluded** (``None`` — dropped from the rate's numerator and denominator), since we can't
tell whether every object is present.
"""
from __future__ import annotations

import pandas as pd

# ── Canonical metric names (snake_case; serve_results renders `_` as space) ──
CLIP_LEVEL_DETECTION_RATE = "clip_level_detection_rate"
OBJ_LEVEL_DETECTION_RATE = "obj_level_detection_rate"
ALL_OBJECTS_PRESENT = "all_objects_present"

#: Group-level detection-metric vocabulary.
DETECTION_METRICS = (CLIP_LEVEL_DETECTION_RATE, OBJ_LEVEL_DETECTION_RATE, ALL_OBJECTS_PRESENT)

#: Old metric name -> canonical, so already-generated trios / CSVs still render.
LEGACY_DETECTION_ALIASES = {"obj_detection_rate": CLIP_LEVEL_DETECTION_RATE}

LABEL_COL = "label"        # per-(clip, object) binary decision: 1 / 0 / -1
CLIP_KEY = "extraction_dir"  # one clip (video) per distinct value


def _valid(labels: pd.Series) -> pd.Series:
    """Rows usable for a detection rate: label present and not uncertain (-1)."""
    return labels[labels.notna() & (labels != -1)]


# ── Per-clip metrics (operate on one clip's object rows) ───────────────────
def clip_detection_rate(clip: pd.DataFrame) -> float | None:
    """Fraction of this clip's target objects detected (``label==1``) over valid rows.

    ``None`` if the clip has no valid (non-uncertain) object rows.
    """
    valid = _valid(clip[LABEL_COL])
    return float((valid == 1).mean()) if len(valid) else None


def all_objects_present(clip: pd.DataFrame) -> float | None:
    """Per-clip 'every target object present?'.

    * a definite absence (``label==0``) on any object -> ``0.0`` (fails the clip,
      even if other objects are uncertain ``-1``);
    * else every object ``label==1`` -> ``1.0``;
    * else (no ``0`` but at least one ``-1``) -> ``None``: the clip is uncertain
      and is excluded from :func:`all_objects_present_rate`.
    """
    labels = clip[LABEL_COL]
    if bool((labels == 0).any()):
        return 0.0
    if bool((labels == 1).all()):
        return 1.0
    return None


# ── Group / dataset metrics (rows pooled across clips) ─────────────────────
def obj_level_detection_rate(df: pd.DataFrame) -> float | None:
    """Object-weighted: fraction of all valid ``(clip, object)`` rows with ``label==1``."""
    valid = _valid(df[LABEL_COL])
    return float((valid == 1).mean()) if len(valid) else None


def clip_level_detection_rate(df: pd.DataFrame) -> float | None:
    """Clip-weighted: mean over clips of :func:`clip_detection_rate` (no-valid clips skipped)."""
    rates = [r for _, c in df.groupby(CLIP_KEY) if (r := clip_detection_rate(c)) is not None]
    return float(sum(rates) / len(rates)) if rates else None


def all_objects_present_rate(df: pd.DataFrame) -> float | None:
    """Mean over clips of :func:`all_objects_present` (fraction of clips fully satisfied).

    Uncertain clips (``all_objects_present`` -> ``None``: no absence but ≥1 ``-1``) are
    skipped, so they sit in neither the numerator nor the denominator.
    """
    vals = [v for _, c in df.groupby(CLIP_KEY) if (v := all_objects_present(c)) is not None]
    return float(sum(vals) / len(vals)) if vals else None


def all_objects_present_rate_legacy(df: pd.DataFrame) -> float | None:
    """DEPRECATED pre-2026-06-22 definition: a ``-1`` (uncertain) object **fails** the
    clip (clip scores 1.0 only if every object is ``label==1``, else 0.0). Kept only so
    reports can show the old number beside the corrected one; use
    :func:`all_objects_present_rate` for everything else."""
    vals = [1.0 if bool((c[LABEL_COL] == 1).all()) else 0.0 for _, c in df.groupby(CLIP_KEY)]
    return float(sum(vals) / len(vals)) if vals else None


def summarize(df: pd.DataFrame) -> dict[str, float | None]:
    """All three group-level detection metrics keyed by canonical name."""
    return {
        CLIP_LEVEL_DETECTION_RATE: clip_level_detection_rate(df),
        OBJ_LEVEL_DETECTION_RATE: obj_level_detection_rate(df),
        ALL_OBJECTS_PRESENT: all_objects_present_rate(df),
    }
