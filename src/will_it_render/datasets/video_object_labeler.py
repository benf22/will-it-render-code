"""Label videos using object-detection classifiers (GRiT, Grounding DINO, OWLv2).

Majority-vote across classifiers. Each (video, object) row carries both clip-level
columns (``score_<clf>`` = detected-frame fraction, ``label_<clf>`` = per-detector
binary vote, ``label`` = majority vote) and compact **per-frame** columns:
``frames_<clf>`` is a bit-string of per-frame 0/1 detections (one char per native
frame; empty if the word is out of that detector's vocabulary), and ``frames_final``
is the per-frame majority vote across in-vocab detectors ('1'/'0'/'u' for tie).

The per-frame columns cover **all native frames** of the clip. The legacy columns
(``score_<clf>`` / ``label_<clf>`` / ``label``) are reproduced byte-for-byte with the
old 16-frame logic: detectors score every native frame once, and the legacy columns
aggregate only the 16 middle-sampled frames the old pipeline used (no extra inference).

Two execution modes:

Mode: torchrun (default)
    torchrun --nproc_per_node=4 -m labeling.video_object_labeler \\
        --root /data/extractions \\
        --out /data/labels.csv \\
        [--classifiers grit gdino owl] \\
        [--frame-threshold 0.15]

Mode: queue (fault-tolerant, distributed)
    # Step 1 — seed once. --out accepts a file or a directory (labels.csv is
    # appended to a directory); if omitted it defaults to <root>/labels.csv.
    python -m labeling.video_object_labeler --mode seed \\
        --root /data/extractions \\
        --queue-dir /data/queue \\
        [--out /data/labels.csv] \\
        [--classifiers grit gdino owl] \\
        [--frame-threshold 0.15] \\
        [--label my_run]

    # Step 2a — launch workers on a VM (one per GPU)
    bash infra/launch_vm.sh 0,1,2,3 \\
        --task-module labeling.video_object_labeler \\
        --queue-dir /data/queue \\
        [--device cuda]

    # Step 2b — OR launch a SLURM array job. SLURM logs are written to
    # <out-dir>/slurm_logs/ (the directory of --out recorded at seed time).
    bash infra/launch_slurm.sh \\
        --partition gpu_a100 \\
        --task-module labeling.video_object_labeler \\
        --queue-dir /data/queue \\
        [--n-workers 8] \\
        [--n-gpus 1] \\
        [--mem 0] \\
        [--device cuda]
"""
from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

from labeling.csv_writer import write_labels_csv
from labeling.visual_classifier import (
    GRiTClassifier,
    GroundingDINOClassifier,
    OWLv2Classifier,
    VisualClassifier,
    VLMClassifier,
    legacy_frame_indices,
    set_frame_colour,
)


def _resolve_out(out: str | None, root: Path) -> Path:
    """Resolve --out to a CSV file path.

    Accepts either a file or a directory. If a directory is given (an existing
    dir, a trailing-slash path, or a suffix-less path) the default filename
    ``labels.csv`` is appended. If omitted entirely, defaults to
    ``<root>/labels.csv``.
    """
    if out is None:
        return root / "labels.csv"
    p = Path(out)
    if p.is_dir() or out.endswith(os.sep) or p.suffix == "":
        return p / "labels.csv"
    return p


def _combine_labels(votes: list[int]) -> float:
    """Majority vote; tie or empty → -1 (uncertain)."""
    positive = sum(votes)
    negative = len(votes) - positive
    if positive > negative:
        return 1.0
    if negative > positive:
        return 0.0
    return -1.0


def _frames_bitstring(frames: list[int]) -> str:
    """Per-frame 0/1 detections → a compact string e.g. [0,1,1,0] → '0110'."""
    return "".join(str(int(f)) for f in frames)


def _combine_frame_votes(frame_lists: list[list[int]]) -> str:
    """Per-frame majority vote across detectors → string of '1'/'0'/'u' (tie/uncertain).

    All lists are assumed equal length (same video, same num_frames). Empty input
    (no in-vocabulary detector) → empty string.
    """
    if not frame_lists:
        return ""
    out = []
    for votes in zip(*frame_lists):
        positive = sum(votes)
        negative = len(votes) - positive
        out.append("1" if positive > negative else "0" if negative > positive else "u")
    return "".join(out)


def _score_word(
    word: str,
    mp4_path: Path,
    classifiers: List[VisualClassifier],
    frame_threshold: float,
    quick_frames: int | None = None,
) -> tuple[dict, str, float]:
    """Score one target word across all classifiers.

    Returns ``(clf_decisions, frames_final, label)``. ``clf_decisions`` holds the
    ``score_<clf>`` (detected-frame fraction), ``label_<clf>`` (binary vote) and
    ``frames_<clf>`` (per-frame 0/1 bit-string; empty if the word is out of that
    detector's vocabulary) columns. ``frames_final`` is the per-frame majority
    vote across in-vocabulary detectors; ``label`` is the clip-level vote.

    Full mode (``quick_frames=None``): the per-frame columns cover **all native
    frames**, and the legacy ``score_<clf>`` / ``label_<clf>`` / ``label`` columns
    are reproduced byte-for-byte with the old 16-frame logic by aggregating only
    the middle-sampled subset the old pipeline scored — same frames, no extra
    inference.

    Quick mode (``quick_frames=N``): detectors decode and score only N frames
    evenly middle-sampled from the clip (the old hard-coded VBench behaviour, e.g.
    N=16). All columns then describe those N frames — ``score_<clf>`` is the
    detected fraction over the N scored frames (no 16-frame sub-selection), and
    the bit-strings are length N.
    """
    votes: list[int] = []
    clf_decisions: dict = {}
    in_vocab_frames: list[list[int]] = []
    for clf in classifiers:
        if clf.in_vocabulary(word):
            frames = clf.score_frames(mp4_path, word, quick_frames=quick_frames)
            if quick_frames is None:
                # Legacy score/label: aggregate the same 16 middle-sampled frames
                # the old pipeline used (load_video(num_frames=16)).
                legacy = [frames[i] for i in legacy_frame_indices(len(frames))] if frames else []
            else:
                # Quick mode already scored only the sampled frames — use them all.
                legacy = frames
            raw_score = sum(legacy) / len(legacy) if legacy else 0.0
            decision = int(raw_score >= frame_threshold)
            votes.append(decision)
            clf_decisions[f"score_{clf.name}"] = raw_score
            clf_decisions[f"label_{clf.name}"] = float(decision)
            clf_decisions[f"frames_{clf.name}"] = _frames_bitstring(frames)
            in_vocab_frames.append(frames)
        else:
            clf_decisions[f"score_{clf.name}"] = float("nan")
            clf_decisions[f"label_{clf.name}"] = float("nan")
            clf_decisions[f"frames_{clf.name}"] = ""
    return clf_decisions, _combine_frame_votes(in_vocab_frames), _combine_labels(votes)


def _build_rows_for_dir(
    subdir: Path,
    extraction_root: Path,
    classifiers: List[VisualClassifier],
    frame_threshold: float,
    quick_frames: int | None = None,
) -> List[dict] | None:
    """Label one extraction dir. Returns rows (possibly empty if no words), or
    None if the dir lacks an MP4 / feature_metadata.json."""
    mp4_files = list(subdir.glob("*.mp4"))
    metadata_path = subdir / "feature_metadata.json"
    if not mp4_files or not metadata_path.exists():
        return None

    with open(metadata_path) as f:
        words = json.load(f).get("words", [])

    mp4_path = mp4_files[0]
    rows: List[dict] = []
    for word in words:
        clf_decisions, frames_final, label = _score_word(
            word, mp4_path, classifiers, frame_threshold, quick_frames=quick_frames
        )
        rows.append({
            "root_extraction_dir": str(extraction_root),
            "relative_extraction_dir": str(subdir.relative_to(extraction_root)),
            "object_name": word,
            "n_objects": len(words),
            **clf_decisions,
            "frames_final": frames_final,
            "label": label,
        })
    return rows


def run_labeling(
    extraction_root: str | Path,
    classifiers: List[VisualClassifier],
    output_csv: str | Path,
    frame_threshold: float = 0.15,
    rank: int = 0,
    world_size: int = 1,
    quick_frames: int | None = None,
) -> None:
    subdirs = sorted(p for p in Path(extraction_root).iterdir() if p.is_dir())
    shard = subdirs[rank::world_size]
    rows: List[dict] = []

    desc = f"Labeling videos (rank {rank}/{world_size})" if world_size > 1 else "Labeling videos"
    root_path = Path(extraction_root)
    for subdir in tqdm(shard, desc=desc, position=rank):
        dir_rows = _build_rows_for_dir(subdir, root_path, classifiers, frame_threshold, quick_frames=quick_frames)
        if dir_rows:
            rows.extend(dir_rows)

    write_labels_csv(rows, output_csv)
    print(f"[rank {rank}] Labeled {len(rows)} (video, class) pairs → {output_csv}")


# ---------------------------------------------------------------------------
# Filesystem-queue task module interface (used by infra/worker.py)
# ---------------------------------------------------------------------------

def add_worker_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda", help="Device for classifiers (cpu or cuda)")


def add_frame_colour_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fix-pil-negation", action="store_true",
                        help="Cast frames to uint8 before ToPILImage so GDINO/OWLv2/VLM see true "
                             "colours (default off = legacy: they score photographic negatives)")
    parser.add_argument("--source-negated", action="store_true",
                        help="The mp4s are stored as negatives (CogX clips before 2026-10-09); "
                             "invert on load for every classifier")


def setup_backend(args: argparse.Namespace) -> dict:
    cfg_path = Path(args.queue_dir) / "run_config.json"
    cfg = json.loads(cfg_path.read_text())

    set_frame_colour(cfg.get("fix_pil_negation", False), cfg.get("source_negated", False))
    classifiers: List[VisualClassifier] = []
    if "grit" in cfg["classifiers"]:
        classifiers.append(GRiTClassifier(device=args.device))
    if "gdino" in cfg["classifiers"]:
        classifiers.append(GroundingDINOClassifier(model_id=cfg["gdino_model"], device=args.device))
    if "owl" in cfg["classifiers"]:
        classifiers.append(OWLv2Classifier(model_id=cfg["owl_model"], device=args.device))
    if "vlm" in cfg["classifiers"]:
        classifiers.append(VLMClassifier(model_id=cfg.get("vlm_model", "Qwen/Qwen2.5-VL-32B-Instruct"),
                                         device=args.device))

    return {
        "classifiers": classifiers,
        "output_csv": Path(cfg["output_csv"]),
        "frame_threshold": cfg["frame_threshold"],
        "quick_frames": cfg.get("quick_frames"),
    }


# Some parallel filesystems (Lustre/NFS) intermittently or permanently reject
# fcntl.flock with ENOLCK ("No locks available"). When that happens we fall back
# to lock-free per-worker shard files (each worker only ever appends to its own),
# which `finalize_shards()` folds back into the main CSV. Once a process has seen
# a fatal lock error it stops trying to lock (per-process latch).
_LOCK_ERRNOS = {errno.ENOLCK, errno.EACCES, errno.EAGAIN, errno.EDEADLK, errno.ENOSYS}
_LOCK_BROKEN = False


def _shard_path(output_csv: Path) -> Path:
    wid = os.environ.get("WORKER_ID") or f"{socket.gethostname()}_{os.getpid()}"
    wid = wid.replace("/", "_").replace(":", "_")
    shard_dir = output_csv.parent / f"{output_csv.stem}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    return shard_dir / f"{output_csv.stem}.{wid}.csv"


def _append_shard(df: pd.DataFrame, output_csv: Path) -> None:
    """Lock-free append to a per-worker shard (safe: single writer per file)."""
    shard = _shard_path(output_csv)
    header = not (shard.exists() and shard.stat().st_size > 0)
    df.to_csv(shard, mode="a", index=False, header=header)


def _append_rows_locked(rows: List[dict], output_csv: Path, retries: int = 5) -> None:
    """Append rows to the shared CSV under an exclusive lock, with retry + a
    lock-free per-worker shard fallback when the filesystem can't lock."""
    global _LOCK_BROKEN
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if not _LOCK_BROKEN:
        for attempt in range(retries):
            try:
                with open(output_csv, "a", newline="") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    try:
                        df.to_csv(f, index=False, header=f.tell() == 0)
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)
                return
            except OSError as e:
                if e.errno in _LOCK_ERRNOS:
                    time.sleep(0.3 * (attempt + 1))
                    continue
                raise
        # Locking is unusable on this node/FS — latch off and use shards.
        _LOCK_BROKEN = True
    _append_shard(df, output_csv)


def finalize_shards(output_csv: str | Path) -> int:
    """Fold any per-worker shard CSVs into the main CSV and remove them.
    Returns the number of shard rows merged. Safe to call repeatedly."""
    output_csv = Path(output_csv)
    shard_dir = output_csv.parent / f"{output_csv.stem}_shards"
    if not shard_dir.exists():
        return 0
    shards = sorted(shard_dir.glob(f"{output_csv.stem}.*.csv"))
    if not shards:
        return 0
    parts = [pd.read_csv(s) for s in shards]
    merged = pd.concat(parts, ignore_index=True)
    header = not (output_csv.exists() and output_csv.stat().st_size > 0)
    merged.to_csv(output_csv, mode="a", index=False, header=header)
    for s in shards:
        s.unlink()
    return len(merged)


def process_task(task: dict, ctx: dict, args: argparse.Namespace) -> bool:
    subdir = Path(task["extraction_dir"])
    extraction_root = subdir.parent
    rows = _build_rows_for_dir(
        subdir, extraction_root, ctx["classifiers"], ctx["frame_threshold"],
        quick_frames=ctx.get("quick_frames"),
    )

    if rows is None:  # missing MP4 / feature_metadata.json
        return False

    if rows:
        _append_rows_locked(rows, ctx["output_csv"])
    return True


def _run_seed(args: argparse.Namespace) -> None:
    _repo_root = str(Path(__file__).parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from infra.fs_queue import FilesystemQueue, register_queue

    root = Path(args.root)
    subdirs = sorted(
        p for p in root.iterdir()
        if p.is_dir() and list(p.glob("*.mp4")) and (p / "feature_metadata.json").exists()
    )

    if not subdirs:
        print(f"No valid extraction dirs found in {root}", file=sys.stderr)
        return

    queue_dir = Path(args.queue_dir)
    if queue_dir.exists():
        print(f"ERROR: Queue directory already exists: {queue_dir}\n"
              f"Delete it or choose a different --queue-dir.", file=sys.stderr)
        sys.exit(1)

    cfg = {
        "output_csv": str(Path(args.out).resolve()),
        "frame_threshold": args.frame_threshold,
        "classifiers": args.classifiers,
        "gdino_model": args.gdino_model,
        "vlm_model": getattr(args, "vlm_model", "Qwen/Qwen2.5-VL-32B-Instruct"),
        "owl_model": args.owl_model,
        "quick_frames": args.quick_frames,
        "fix_pil_negation": args.fix_pil_negation,
        "source_negated": args.source_negated,
    }
    queue_dir.mkdir(parents=True, exist_ok=True)
    (queue_dir / "run_config.json").write_text(json.dumps(cfg, indent=2))

    label = args.label or root.name
    run_meta = {
        "label": label,
        "created_at": time.time(),
        "n_tasks": len(subdirs),
        "output_dir": str(root.resolve()),
    }
    (queue_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

    queue = FilesystemQueue(queue_dir)
    for subdir in subdirs:
        queue.push({"extraction_dir": str(subdir)})

    register_queue(queue_dir, label=label, output_dir=str(root), n_tasks=len(subdirs))

    c = queue.counts()
    print(f"Pushed {len(subdirs)} task(s) to {queue_dir}")
    print(f"Queue: pending={c['pending']}  claimed={c['claimed']}  done={c['done']}  failed={c['failed']}")
    print(f"Run config: {queue_dir / 'run_config.json'}")
    print(f"Label: {label!r}  |  Run monitor: python infra/monitor.py")


def main():
    parser = argparse.ArgumentParser(description="Label videos using object detection classifiers")
    parser.add_argument(
        "--mode",
        choices=["torchrun", "seed"],
        default="torchrun",
        help="torchrun: multi-GPU DDP sharding (default); seed: populate a filesystem queue",
    )
    parser.add_argument("--root", required=True, help="Experiment root directory")
    parser.add_argument("--out", default=None,
                        help="Output labels CSV file, or a directory (labels.csv is appended). "
                             "Default: <root>/labels.csv")
    parser.add_argument(
        "--frame-threshold",
        type=float,
        default=0.15,
        help="Per-classifier frame detection rate threshold for voting (default: 0.15)",
    )
    parser.add_argument(
        "--quick-frames",
        type=int,
        default=None,
        help="Quick mode: score only this many frames, evenly middle-sampled from each "
             "clip (the old hard-coded VBench behaviour, e.g. 16), instead of every native "
             "frame. Much faster; all CSV columns then describe these N frames.",
    )
    parser.add_argument("--device", default="cuda", help="Device (cpu or cuda)")
    parser.add_argument(
        "--classifiers",
        nargs="+",
        choices=["grit", "gdino", "owl", "vlm"],
        default=["grit", "gdino", "owl"],
        help="Which classifiers to use (default: grit gdino owl)",
    )
    parser.add_argument(
        "--gdino-model",
        default="IDEA-Research/grounding-dino-base",
        help="Grounding DINO HuggingFace model ID",
    )
    parser.add_argument(
        "--vlm-model",
        default="Qwen/Qwen2.5-VL-32B-Instruct",
        help="VLM judge checkpoint (Qwen2.5-VL 7B/32B/72B, InternVL, ...)",
    )
    parser.add_argument(
        "--owl-model",
        default="google/owlv2-base-patch16-ensemble",
        help="OWL-v2 HuggingFace model ID",
    )
    add_frame_colour_args(parser)
    # seed-mode only
    parser.add_argument("--queue-dir", default=None, help="[seed] Queue root directory")
    parser.add_argument("--label", default=None, help="[seed] Human-readable run label for the monitor")
    args = parser.parse_args()

    args.out = str(_resolve_out(args.out, Path(args.root)))

    if args.mode == "seed":
        if not args.queue_dir:
            parser.error("--queue-dir is required for --mode seed")
        _run_seed(args)
        return

    # --- torchrun mode (unchanged) ---
    rank       = int(os.environ.get("RANK",       0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        import torch
        import torch.distributed as dist
        gpu_id = local_rank % torch.cuda.device_count()
        torch.cuda.set_device(gpu_id)
        dist.init_process_group(backend="gloo")
        args.device = f"cuda:{gpu_id}"

    out = Path(args.out)
    rank_out = out.with_stem(f"{out.stem}_rank{rank}") if world_size > 1 else out

    set_frame_colour(args.fix_pil_negation, args.source_negated)
    classifiers: List[VisualClassifier] = []
    if "grit" in args.classifiers:
        classifiers.append(GRiTClassifier(device=args.device))
    if "gdino" in args.classifiers:
        classifiers.append(GroundingDINOClassifier(model_id=args.gdino_model, device=args.device))
    if "owl" in args.classifiers:
        classifiers.append(OWLv2Classifier(model_id=args.owl_model, device=args.device))
    if "vlm" in args.classifiers:
        classifiers.append(VLMClassifier(model_id=args.vlm_model, device=args.device))

    run_labeling(args.root, classifiers, rank_out, frame_threshold=args.frame_threshold,
                 rank=rank, world_size=world_size, quick_frames=args.quick_frames)

    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()
        if rank == 0:
            parts = [out.with_stem(f"{out.stem}_rank{r}") for r in range(world_size)]
            pd.concat([pd.read_csv(p) for p in parts]).to_csv(out, index=False)
            for p in parts:
                p.unlink()
            print(f"Merged {world_size} shards → {out}")
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
