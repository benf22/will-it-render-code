"""Temporal-accuracy metric for temporally-controlled video generation.

The evaluation metric from *TempoControl: Temporal Attention Guidance for Text-to-Video
Models* (Schiber et al., CVPR 2026 — arXiv:2510.02226), reimplemented so it scores videos
from **any** generator (LTX-2, Wan, Ovi…). It only touches the final RGB video plus a
per-frame 0/1 control signal, so it is model/backend-agnostic by construction.

Two detection metrics are reported side by side, **scored on the identical sampled frames**:

1. **YOLOv10x** (`jameslahm/yolov10x`) — the detector the paper uses for this metric (it
   replaced VBench's GRiT, unreliable on small/partial objects; the single-object
   benchmark's 80 prompts *are* YOLO's 80 COCO classes).
2. **Our detectors** — GRiT / Grounding-DINO / OWLv2 (`labeling/visual_classifier.py`),
   plus their per-frame majority vote, on the same frames.

The metric
----------
Sample ``T`` evenly-spaced frames from the video (`np.linspace`, as the paper). For a set
of ``(object, control)`` pairs, run each detector on those frames. Frame ``t`` is a
**success** for a detector when *every* object matches its requirement:

    control[t] == 1 → object must be DETECTED (present)
    control[t] == 0 → object must be ABSENT (not detected)

* ``temporal_accuracy`` = mean over videos of (successful frames / T).
* per object: ``present_rate`` and ``absent_rate``.

This one rule reproduces all three benchmarks: one-object (1 control), two-object (temp
control + static all-ones), multiple-objects (all objects all-ones).

Control signal ↔ LTX frames
---------------------------
Controls are given at the model's **latent-frame** resolution (13 for LTX-2 @512×512×97;
21 for Wan). LTX's temporal VAE is causal with stride 8: latent 0 decodes native frame 0,
latent ``t`` decodes native ``[8(t-1)+1 … 8t]``. Each sampled frame's native index is
therefore mapped back to its latent frame to read the control value — so the metric is
correct even though the first latent frame covers a single native frame. The stride is
derived from the video's actual frame count, so any (latent-count, native-count) pair works.

Usage
-----
    python -m evaluation.temporal_accuracy --selfcheck        # aggregation logic, no models

    # LTX-adapted benchmark CSV (baselines/tempocontrol/data_ltx/*.csv) + flat videos:
    python -m evaluation.temporal_accuracy --benchmark one-object \
        --videos-path <dir> --csv baselines/tempocontrol/data_ltx/one_object.csv \
        --detectors yolov10 grit gdino owl --num-frames 20 --out results/ta.json

    # our per-prompt output dirs → a manifest (video_path, objects, controls):
    python -m evaluation.temporal_accuracy --manifest <manifest.csv> --detectors yolov10 --out ta.json
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence


# ── control-signal helpers ─────────────────────────────────────────────────────

def parse_control_signal(raw: "str | Sequence[int]") -> list[int]:
    """Whitespace/comma-separated 0/1 string (or int sequence) → ``list[int]``."""
    if isinstance(raw, str):
        return [int(x) for x in raw.replace(",", " ").split()]
    return [int(x) for x in raw]


def latent_value_at_native(latent: Sequence[int], native_idx: int, native_total: int) -> int:
    """Control value at native frame ``native_idx``, from a latent-resolution signal.

    Causal-VAE grouping: native 0 → latent 0; native ``i>0`` → latent ``(i-1)//stride + 1``
    with ``stride = (native_total-1)//(len(latent)-1)``. Robust to off-by-one native counts
    (clamps the latent index).
    """
    n_lat = len(latent)
    if n_lat == 1:
        return int(latent[0])
    stride = max(1, (native_total - 1) // (n_lat - 1))
    lat_idx = 0 if native_idx == 0 else min((native_idx - 1) // stride + 1, n_lat - 1)
    return int(latent[lat_idx])


# ── frame sampling ─────────────────────────────────────────────────────────────

def sample_frame_indices(native_total: int, num_frames: int) -> list[int]:
    """``num_frames`` evenly-spaced native-frame indices (paper's ``np.linspace``)."""
    import numpy as np
    n = min(num_frames, native_total)
    return [int(i) for i in np.linspace(0, native_total - 1, num=n, dtype=int)]


def read_frames(video_path: str, indices: Sequence[int]) -> list:
    """Return the requested native frames as PIL RGB images (via OpenCV), robust to decoders.

    Sequential read (no seeking) — the frame layout TempoControl's sampler relies on.
    Returns ``[]`` if the video can't be opened / has no frames.
    """
    import cv2
    from PIL import Image

    cap = None
    for _ in range(3):
        cap = cv2.VideoCapture(video_path)
        if cap is not None and cap.isOpened():
            break
        if cap is not None:
            cap.release()
    if cap is None or not cap.isOpened():
        print(f"[temporal_accuracy] could not open {video_path}")
        return []
    try:
        want = set(int(i) for i in indices)
        order = sorted(want)
        out_by_idx: dict[int, "Image.Image"] = {}
        idx = 0
        max_idx = max(want) if want else -1
        while idx <= max_idx:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if idx in want and frame.size > 0:
                rgb = frame[:, :, ::-1]  # BGR → RGB
                out_by_idx[idx] = Image.fromarray(rgb.copy())
            idx += 1
        return [out_by_idx[i] for i in order if i in out_by_idx]
    finally:
        cap.release()


def video_frame_count(video_path: str) -> int:
    import cv2
    cap = cv2.VideoCapture(video_path)
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    finally:
        cap.release()


# ── core scoring (pure, detector-independent) ──────────────────────────────────

@dataclass
class VideoResult:
    video_path: Optional[str]
    video_accuracy: float
    n_frames: int
    per_object: dict = field(default_factory=dict)
    # Per-frame detection + control per object, e.g. {"cat": {"detected": [0,0,1,...],
    # "control": [0,0,1,...]}} — saved so downstream (qualitative panels, debugging) never
    # has to re-run the detectors.
    per_frame: dict = field(default_factory=dict)


def temporal_metrics(detected: Sequence[int], control: Sequence[int]) -> dict:
    """Object-level temporal-control metrics from per-frame detected vs binary control.

    Classification family (per-frame confusion over the decided frames):
      TP = detected & present, FP = detected & absent, FN = present & !detected, TN = absent & !detected
      - temporal_precision   = TP/(TP+FP)   (of the frames it appeared in, how many were requested) — None if it never appears
      - temporal_recall      = TP/(TP+FN)   (= presence_accuracy)                                     — None if no present frames
      - temporal_specificity = TN/(TN+FP)   (= absent_accuracy)                                        — None if no absent frames
      - temporal_f1          = harmonic mean of precision & recall

    IoU family (segment = the contiguous [first, last] temporal bounding box):
      - t_iou = |target_seg ∩ pred_seg| / |target_seg ∪ pred_seg|   (0 if nothing detected; None if no present frames)
      - t_l2  = sqrt((start_pred−start_tgt)² + (end_pred−end_tgt)²) in frames  (None if nothing detected / no present)
    ``t_iou`` feeds dataset-level ``recall_at_tiou`` (fraction of objects with t_iou > threshold).
    """
    det = [int(bool(x)) for x in detected]
    ctl = [int(x) for x in control]
    T = len(ctl)
    P = [t for t in range(T) if ctl[t] == 1]        # requested-present frames
    A = [t for t in range(T) if ctl[t] == 0]        # requested-absent frames
    D = [t for t in range(T) if det[t] == 1]        # detected frames
    tp = sum(1 for t in P if det[t]); fn = len(P) - tp
    fp = sum(1 for t in A if det[t]); tn = len(A) - fp
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / len(P) if P else None
    specificity = tn / len(A) if A else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision not in (None, 0) and recall not in (None, 0) else
          (0.0 if (precision is not None and recall is not None) else None))

    # segment overlap (contiguous bounding boxes)
    t_iou = t_l2 = None
    if P:
        ts, te = min(P), max(P)                     # target segment [start,end] inclusive
        if D:
            ps, pe = min(D), max(D)                 # predicted segment
            inter = max(0, min(te, pe) - max(ts, ps) + 1)
            union = (te - ts + 1) + (pe - ps + 1) - inter
            t_iou = inter / union if union > 0 else 0.0
            t_l2 = float(((ps - ts) ** 2 + (pe - te) ** 2) ** 0.5)
        else:
            t_iou = 0.0                             # requested but never generated
    # control_exist: is there ANY temporal variation in the object's detected presence?
    # 1 if the detected sequence contains both a detected and a not-detected frame; 0 if it
    # is constant (object always present OR always absent). Independent of the control signal
    # — measures only whether steering produced any change in presence over time (a
    # necessary condition for temporal control; the trivial all-on / all-off outputs score 0).
    control_exist = int(len(set(det)) > 1) if det else 0
    return {"temporal_precision": precision, "temporal_recall": recall,
            "temporal_specificity": specificity, "temporal_f1": f1,
            "t_iou": t_iou, "t_l2": t_l2, "control_exist": control_exist}


def score_from_presence(
    presence: dict[str, Sequence[bool]],
    controls: dict[str, Sequence[int]],
) -> VideoResult:
    """Score per-object detected/absent booleans against per-object per-frame controls.

    ``presence[obj][t]`` = object detected in frame t. ``controls[obj][t]`` = 0/1 required.
    A frame is a success iff every object matches (present when 1, absent when 0). Pure —
    unit-testable, detector-agnostic.
    """
    objects = list(controls)
    T = len(next(iter(controls.values())))
    for obj in objects:
        if len(presence[obj]) != T or len(controls[obj]) != T:
            raise ValueError(f"length mismatch for {obj!r}: "
                             f"presence {len(presence[obj])}, control {len(controls[obj])}, T {T}")

    per_object = {
        obj: {"present_success": 0, "present_frames": 0, "absent_success": 0, "absent_frames": 0}
        for obj in objects
    }
    frame_successes = 0
    for t in range(T):
        frame_ok = True
        for obj in objects:
            want = controls[obj][t] == 1
            is_present = bool(presence[obj][t])
            s = per_object[obj]
            if want:
                s["present_frames"] += 1
                s["present_success"] += int(is_present)
                frame_ok = frame_ok and is_present
            else:
                s["absent_frames"] += 1
                s["absent_success"] += int(not is_present)
                frame_ok = frame_ok and (not is_present)
        frame_successes += int(frame_ok)

    acc = frame_successes / T if T > 0 else 0.0
    per_frame = {obj: {"detected": [int(bool(presence[obj][t])) for t in range(T)],
                       "control": [int(controls[obj][t]) for t in range(T)]}
                 for obj in objects}
    # object-level temporal-control metrics (precision/recall/specificity/f1 + t-IoU/t-L2)
    for obj in objects:
        per_object[obj]["temporal"] = temporal_metrics(presence[obj], controls[obj])
    return VideoResult(None, acc, T, per_object, per_frame)


def aggregate(results: list[VideoResult]) -> dict:
    """Reduce per-video results to a summary dict (paper-style keys)."""
    import numpy as np
    scored = [r for r in results if r.n_frames > 0]
    mean_acc = float(np.mean([r.video_accuracy for r in scored])) if scored else 0.0

    totals: dict[str, dict] = {}
    for r in scored:
        for obj, s in r.per_object.items():
            agg = totals.setdefault(obj, {"present_success": 0, "present_frames": 0,
                                          "absent_success": 0, "absent_frames": 0})
            for k in agg:
                agg[k] += s[k]

    def rate(n, d):
        return n / d if d > 0 else 0.0

    gp_s = sum(s["present_success"] for s in totals.values())
    gp_f = sum(s["present_frames"] for s in totals.values())
    ga_s = sum(s["absent_success"] for s in totals.values())
    ga_f = sum(s["absent_frames"] for s in totals.values())

    # Per-video rows (pooled across the video's objects) — the sample the downstream
    # std / confidence-interval graphs are computed over. Per-clip present/absent rate is
    # NaN when the video has no present/absent frames (kept out of those CIs).
    TM_KEYS = ["temporal_precision", "temporal_recall", "temporal_specificity",
               "temporal_f1", "t_iou", "t_l2", "control_exist"]
    obj_temporal = []   # one temporal-metrics dict per (video, object) — the object-level sample
    per_video = []
    for r in scored:
        ps = sum(s["present_success"] for s in r.per_object.values())
        pf = sum(s["present_frames"] for s in r.per_object.values())
        as_ = sum(s["absent_success"] for s in r.per_object.values())
        af = sum(s["absent_frames"] for s in r.per_object.values())
        tms = [s["temporal"] for s in r.per_object.values() if "temporal" in s]
        obj_temporal += tms
        # per-video temporal = mean over the video's objects (ignoring None per key)
        vid_tm = {k: (float(np.mean([m[k] for m in tms if m.get(k) is not None]))
                      if any(m.get(k) is not None for m in tms) else None) for k in TM_KEYS}
        per_video.append({
            "video_path": r.video_path,
            "temporal_accuracy": r.video_accuracy,
            "present_rate": (ps / pf) if pf > 0 else None,
            "absent_rate": (as_ / af) if af > 0 else None,
            "temporal": vid_tm,
            "per_frame": r.per_frame,   # {obj: {detected:[...], control:[...]}} — no recompute
        })

    def _mean(key):
        vals = [m[key] for m in obj_temporal if m.get(key) is not None]
        return float(np.mean(vals)) if vals else None
    tiou_vals = [m["t_iou"] for m in obj_temporal if m.get("t_iou") is not None]
    temporal_summary = {k: _mean(k) for k in TM_KEYS}
    temporal_summary["recall_at_tiou_0.5"] = (
        float(np.mean([v > 0.5 for v in tiou_vals])) if tiou_vals else None)
    temporal_summary["n_objects"] = len(obj_temporal)

    return {
        "temporal_accuracy": mean_acc,
        "n_videos": len(scored),
        "n_videos_missing": len(results) - len(scored),
        "present_rate": rate(gp_s, gp_f),
        "absent_rate": rate(ga_s, ga_f),
        "temporal_metrics": temporal_summary,   # object-level precision/recall/spec/f1/t-IoU/t-L2 + Recall@t-IoU
        "per_object": {obj: {"present_rate": rate(s["present_success"], s["present_frames"]),
                             "absent_rate": rate(s["absent_success"], s["absent_frames"]),
                             **{k: v for k, v in s.items() if k != "temporal"}}
                       for obj, s in sorted(totals.items())},
        "per_video": per_video,
    }


# ── jobs ───────────────────────────────────────────────────────────────────────

@dataclass
class VideoJob:
    video_path: str
    controls: dict[str, list[int]]   # object → latent-resolution 0/1 signal


def jobs_from_tempocontrol_csv(csv_file, videos_path, benchmark, video_index=0) -> list[VideoJob]:
    """Jobs from an (LTX-adapted) TempoControl CSV + flat ``{key}-{i}.mp4`` layout.

    * ``one-object``      → ``temp_object`` with ``control_signal1``; key = ``prompt``.
    * ``two-object``      → ``temp_object`` w/ ``control_signal1`` + ``static_object`` w/
      ``control_signal2``; key = ``original_prompt`` (falls back to ``prompt``).
    * ``multiple-objects``→ every ``Object N`` / ``objects`` column, all-ones control.
    """
    import pandas as pd
    df = pd.read_csv(csv_file)
    jobs = []
    for _, row in df.iterrows():
        if benchmark == "one-object":
            key = str(row["prompt"])
            controls = {str(row["temp_object"]).strip().lower(): parse_control_signal(row["control_signal1"])}
        elif benchmark == "two-object":
            key = str(row.get("original_prompt", row["prompt"]))
            controls = {
                str(row["temp_object"]).strip().lower(): parse_control_signal(row["control_signal1"]),
                str(row["static_object"]).strip().lower(): parse_control_signal(row["control_signal2"]),
            }
        elif benchmark == "multiple-objects":
            key = str(row.get("original_prompt", row["prompt"]))
            obj_cols = [c for c in df.columns if c.lower().startswith("object")]
            objs = [str(row[c]).strip().lower() for c in obj_cols if str(row.get(c, "")).strip()]
            T = len(parse_control_signal(row["control_signal1"]))
            controls = {o: [1] * T for o in objs}
        else:
            raise ValueError(f"unknown benchmark {benchmark!r}")
        jobs.append(VideoJob(os.path.join(videos_path, f"{key}-{video_index}.mp4"), controls))
    return jobs


def jobs_from_manifest(manifest_csv) -> list[VideoJob]:
    """Model-agnostic input. Columns: ``video_path``, ``objects`` (``;``-sep), ``controls``
    (``;``-sep, one whitespace 0/1 signal per object)."""
    import pandas as pd
    df = pd.read_csv(manifest_csv)
    jobs = []
    for _, row in df.iterrows():
        objs = [o.strip().lower() for o in str(row["objects"]).split(";") if o.strip()]
        ctrls = [parse_control_signal(c) for c in str(row["controls"]).split(";")]
        if len(objs) != len(ctrls):
            raise ValueError(f"{row['video_path']}: {len(objs)} objects vs {len(ctrls)} controls")
        jobs.append(VideoJob(str(row["video_path"]), dict(zip(objs, ctrls))))
    return jobs


# ── detector runner ────────────────────────────────────────────────────────────

DETECTOR_NAMES = ["yolov10", "grit", "gdino", "owl"]
OURS = ["grit", "gdino", "owl"]


def _build_detector(name: str, device: str):
    from will_it_render.common.visual_classifier import (
        GRiTClassifier, GroundingDINOClassifier, OWLv2Classifier, YoloV10Classifier)
    return {
        "yolov10": YoloV10Classifier, "grit": GRiTClassifier,
        "gdino": GroundingDINOClassifier, "owl": OWLv2Classifier,
    }[name](device=device)


def run(jobs: list[VideoJob], detectors: list[str], device: str, num_frames: int) -> dict:
    """Score every job with every detector on the SAME sampled frames; build the table.

    Detectors run one at a time (loaded, used, dropped) to bound VRAM; frame sampling is
    deterministic (`np.linspace`) so every detector sees identical frames. Per-frame
    presence bit-strings are cached so the majority-of-ours row costs no extra inference.
    """
    # Precompute per-job sampled indices + per-frame controls (needs the real frame count).
    plan = []
    for job in jobs:
        n = video_frame_count(job.video_path)
        if n <= 0:
            plan.append((job, None, None))
            continue
        idxs = sample_frame_indices(n, num_frames)
        controls = {obj: [latent_value_at_native(sig, i, n) for i in idxs]
                    for obj, sig in job.controls.items()}
        plan.append((job, idxs, controls))

    # presence[detector][job_i][obj] = list[bool]
    presence: dict[str, list[Optional[dict]]] = {}
    for det_name in detectors:
        det = _build_detector(det_name, device)
        rows: list[Optional[dict]] = []
        for i, (job, idxs, controls) in enumerate(plan):
            if idxs is None:
                rows.append(None)
                print(f"[temporal_accuracy] {det_name} ({i+1}/{len(plan)}) missing: {job.video_path}")
                continue
            frames = read_frames(job.video_path, idxs)
            if len(frames) < len(idxs):  # decoder returned fewer; trim controls to match
                controls = {o: c[:len(frames)] for o, c in controls.items()}
            per_obj = {}
            in_vocab = {}
            for obj in job.controls:
                bits = det.score_pil_frames(frames, obj) if det.in_vocabulary(obj) else None
                per_obj[obj] = bits
                in_vocab[obj] = bits is not None
            rows.append({"controls": controls, "presence": per_obj, "in_vocab": in_vocab})
            print(f"[temporal_accuracy] {det_name} ({i+1}/{len(plan)}) {os.path.basename(job.video_path)}")
        presence[det_name] = rows
        del det

    # Score each detector; add a majority-of-ours row when all three are present.
    per_detector: dict[str, dict] = {}
    detectors_out = list(detectors)
    if all(d in detectors for d in OURS):
        detectors_out.append("majority_ours")

    for det_name in detectors_out:
        results: list[VideoResult] = []
        for i, (job, idxs, controls) in enumerate(plan):
            if idxs is None:
                results.append(VideoResult(job.video_path, 0.0, 0))
                continue
            row_controls, pres = _presence_for(det_name, presence, i, job)
            if pres is None:
                results.append(VideoResult(job.video_path, 0.0, 0))
                continue
            r = score_from_presence(pres, row_controls)
            r.video_path = job.video_path
            results.append(r)
        per_detector[det_name] = aggregate(results)

    return {"num_frames": num_frames, "detectors": detectors_out, "per_detector": per_detector}


def _presence_for(det_name, presence, i, job):
    """Resolve the (controls, presence) for a detector row, handling out-of-vocab and the
    synthetic ``majority_ours`` (per-frame majority across in-vocab GRiT/GDINO/OWL)."""
    if det_name != "majority_ours":
        row = presence[det_name][i]
        if row is None:
            return None, None
        # out-of-vocab objects: treat as "never detected" (all-absent) — that object's
        # present frames simply fail, matching the labeler's empty-bitstring convention.
        controls = row["controls"]
        T = len(next(iter(controls.values())))
        pres = {obj: (bits if bits is not None else [False] * T)
                for obj, bits in row["presence"].items()}
        return controls, pres

    # majority across ours
    rows = [presence[d][i] for d in OURS]
    if any(r is None for r in rows):
        return None, None
    controls = rows[0]["controls"]
    T = len(next(iter(controls.values())))
    pres = {}
    for obj in job.controls:
        votes = [r["presence"][obj] for r in rows if r["presence"][obj] is not None]
        if not votes:
            pres[obj] = [False] * T
        else:
            pres[obj] = [sum(v[t] for v in votes) * 2 >= len(votes) for t in range(T)]
    return controls, pres


def print_table(summary: dict) -> None:
    print(f"\ntemporal accuracy  ({summary['num_frames']} frames/video)")
    print(f"{'detector':<14} {'temporal_acc':>12} {'present':>9} {'absent':>9} {'videos':>7}")
    print("-" * 55)
    for det in summary["detectors"]:
        s = summary["per_detector"][det]
        print(f"{det:<14} {s['temporal_accuracy']:>12.4f} {s['present_rate']:>9.4f} "
              f"{s['absent_rate']:>9.4f} {s['n_videos']:>7d}")


# ── self-check ──────────────────────────────────────────────────────────────────

def _selfcheck() -> None:
    # one object, present in second half.
    pres = {"dog": [False, False, True, True]}
    ctrl = {"dog": [0, 0, 1, 1]}
    assert score_from_presence(pres, ctrl).video_accuracy == 1.0
    pres = {"dog": [True, False, True, True]}
    assert score_from_presence(pres, ctrl).video_accuracy == 0.75

    # two objects: static always present, temp second half.
    pres = {"cat": [False, False, True, True], "bird": [True, True, True, True]}
    ctrl = {"cat": [0, 0, 1, 1], "bird": [1, 1, 1, 1]}
    assert score_from_presence(pres, ctrl).video_accuracy == 1.0
    pres["bird"] = [False, True, True, True]  # static missing frame 0
    assert score_from_presence(pres, ctrl).video_accuracy == 0.75

    # latent → native mapping: LTX 13 latent over 97 native, stride 8.
    lat = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert latent_value_at_native(lat, 0, 97) == 0     # native 0 → latent 0
    assert latent_value_at_native(lat, 20, 97) == 1    # native 20 → latent 3 (=1)
    assert latent_value_at_native(lat, 16, 97) == 0    # native 16 → latent 2 (=0)

    assert sample_frame_indices(97, 5) == [0, 24, 48, 72, 96]
    assert aggregate([score_from_presence(pres, ctrl)])["n_videos"] == 1
    print("temporal_accuracy self-check passed.")


# ── CLI ──────────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--benchmark", choices=["one-object", "two-object", "multiple-objects"])
    p.add_argument("--videos-path")
    p.add_argument("--csv", dest="csv_file")
    p.add_argument("--video-index", type=int, default=0)
    p.add_argument("--manifest")
    p.add_argument("--detectors", nargs="+", default=["yolov10"], choices=DETECTOR_NAMES,
                   help="Detectors to score (default: yolov10). Include grit gdino owl for our metric.")
    p.add_argument("--num-frames", type=int, default=20, help="Frames sampled per video (paper: ~20).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out")
    args = p.parse_args(argv)

    if args.selfcheck:
        _selfcheck()
        return

    if args.manifest:
        jobs = jobs_from_manifest(args.manifest)
    elif args.benchmark and args.videos_path:
        default_csv = {"one-object": "one_object.csv", "two-object": "two_objects.csv",
                       "multiple-objects": "two_objects.csv"}[args.benchmark]
        csv_file = args.csv_file or os.path.join(
            os.path.dirname(__file__), "..", "baselines", "tempocontrol", "data_ltx", default_csv)
        jobs = jobs_from_tempocontrol_csv(csv_file, args.videos_path, args.benchmark, args.video_index)
    else:
        p.error("provide --manifest, or --benchmark with --videos-path (see --help).")

    summary = run(jobs, args.detectors, args.device, args.num_frames)
    print_table(summary)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
