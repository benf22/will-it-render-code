"""Score generated WAVs with PANNs Cnn14 and write a binary labels CSV.

Usage:
    python -m labeling.audio_labeler \\
        --root /data/extractions \\
        --out /data/labels.csv \\
        [--threshold 0.5] \\
        [--device cuda]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from labeling.audio_classifier import AudioSetClassifier
from labeling.csv_writer import write_labels_csv


def run_labeling(
    extraction_root: str | Path,
    classifier: AudioSetClassifier,
    output_csv: str | Path,
    threshold: float = 0.5,
) -> None:
    extraction_root = Path(extraction_root)
    rows: List[dict] = []

    for subdir in sorted(extraction_root.iterdir()):
        if not subdir.is_dir():
            continue

        wav_files = list(subdir.glob("*.wav"))
        if not wav_files:
            continue
        wav_path = wav_files[0]

        info_path = subdir / "prompt_info.json"
        # NOTE: audio_labeler only supports the new single-object convention
        # (prompt_info.json with a `class_name`). Old-convention / per-word dirs
        # have no prompt_info.json and carry multiple target words in
        # feature_metadata.json. To support those in the future, read
        # `feature_metadata.json["words"]` (see labeling/video_object_labeler.py)
        # and emit one row per (dir, word) instead of one row per dir.
        assert info_path.exists(), (
            f"{subdir} has no prompt_info.json — audio_labeler only supports the "
            f"new single-object convention. See note above for per-word support."
        )
        object_name = json.loads(info_path.read_text())["class_name"]

        score = classifier.score(str(wav_path), object_name)

        rows.append({
            "extraction_dir": str(subdir),
            "object_name": object_name,
            "label": int(score >= threshold),
        })

    write_labels_csv(rows, output_csv)
    print(f"Labeled {len(rows)} videos → {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Extraction output root dir")
    parser.add_argument("--out", required=True, help="Output labels CSV path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold for binary label")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    classifier = AudioSetClassifier(device=args.device)
    run_labeling(args.root, classifier, args.out, threshold=args.threshold)


if __name__ == "__main__":
    main()
