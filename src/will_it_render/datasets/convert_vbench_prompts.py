"""Convert VBench prompts to will-it-render CSV format.

Usage (VBench dimension — resolves both .txt and metadata JSON):
    python -m dataset_creation.convert_vbench_prompts \\
        --dimension object_class \\
        --output object_class_prompts.csv \\
        [--vbench_dir ~/repos/VBench]

Usage (plain .txt file, no metadata):
    python -m dataset_creation.convert_vbench_prompts \\
        --prompts_txt /path/to/prompts.txt \\
        --object_name my_class \\
        --target_words "word1,word2" \\
        --output my_prompts.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Iterator


_DIMENSION_FIELD: dict[str, str] = {
    "object_class": "object_en",
    "multiple_objects": "object_en",
    "color": "color_en",
    "appearance_style": "style_en",
}


def _slugify(text: str) -> str:
    """Convert a phrase to a snake_case slug."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return re.sub(r"\s+", "_", text)


def _extract_row(entry: dict, dimension: str | None = None) -> tuple[str, str, str]:
    """Return (prompt, object_name, target_words) from a metadata entry.

    Handles all VBench metadata schemas:
      - object_en  (object_class, multiple_objects)
      - color_en   (color)
      - style_en   (appearance_style)
      - object_a_en + object_b_en  (spatial_relationship)
    """
    prompt = entry["prompt_en"]

    # spatial relationship: two objects
    if "object_a_en" in entry and "object_b_en" in entry:
        a, b = entry["object_a_en"], entry["object_b_en"]
        object_name = f"{_slugify(a)}_and_{_slugify(b)}"
        target_words = f"{a},{b}"
        return prompt, object_name, target_words

    # look up the expected field for this dimension, fall back to first non-prompt key
    field = _DIMENSION_FIELD.get(dimension or "", None)
    if field is None:
        semantic_keys = [k for k in entry if k != "prompt_en"]
        if not semantic_keys:
            raise ValueError(f"No semantic key found in entry: {entry}")
        field = semantic_keys[0]

    value = entry[field]
    object_name = _slugify(value)
    # split on " and " so "cat and dog" yields two targets: "cat", "dog"
    target_words = ",".join(part.strip() for part in value.split(" and ") if part.strip())
    return prompt, object_name, target_words


def _rows_from_metadata(metadata_path: Path, dimension: str | None = None) -> Iterator[tuple[str, str, str]]:
    with open(metadata_path) as f:
        entries = json.load(f)
    for entry in entries:
        yield _extract_row(entry, dimension)


def _rows_from_txt(
    txt_path: Path,
    object_name: str,
    target_words: str,
) -> Iterator[tuple[str, str, str]]:
    with open(txt_path) as f:
        for line in f:
            prompt = line.strip()
            if prompt:
                yield prompt, object_name, target_words


def _write_csv(rows: Iterator[tuple[str, str, str]], output: Path) -> int:
    count = 0
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "object_name", "target_words"])
        for prompt, object_name, target_words in rows:
            writer.writerow([prompt, object_name, target_words])
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert VBench prompts to will-it-render CSV format."
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--dimension",
        help="VBench dimension name (e.g. object_class). Looks up both the .txt "
             "and the paired metadata JSON under --vbench_dir.",
    )
    src.add_argument(
        "--prompts_txt",
        type=Path,
        help="Path to a VBench-style .txt file (one prompt per line). "
             "Requires --object_name and --target_words.",
    )

    parser.add_argument(
        "--vbench_dir",
        type=Path,
        default=Path.home() / "repos" / "VBench",
        help="Root of the VBench repo (default: ~/repos/VBench).",
    )
    parser.add_argument(
        "--object_name",
        help="Slug for object_name column (required with --prompts_txt when no "
             "metadata JSON is available).",
    )
    parser.add_argument(
        "--target_words",
        help="Comma-separated target words (required with --prompts_txt when no "
             "metadata JSON is available).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path.",
    )

    args = parser.parse_args()

    if args.dimension:
        vbench = args.vbench_dir
        metadata_path = vbench / "prompts" / "metadata" / f"{args.dimension}.json"
        txt_path = vbench / "prompts" / "prompts_per_dimension" / f"{args.dimension}.txt"

        if metadata_path.exists():
            rows = _rows_from_metadata(metadata_path, args.dimension)
        elif txt_path.exists():
            if not args.object_name or not args.target_words:
                parser.error(
                    f"No metadata JSON for dimension '{args.dimension}'. "
                    "Provide --object_name and --target_words to fall back to "
                    "the .txt file."
                )
            rows = _rows_from_txt(txt_path, args.object_name, args.target_words)
        else:
            sys.exit(
                f"Neither {metadata_path} nor {txt_path} found. "
                "Check --vbench_dir and --dimension."
            )
    else:
        if not args.prompts_txt.exists():
            sys.exit(f"File not found: {args.prompts_txt}")
        if not args.object_name or not args.target_words:
            parser.error("--prompts_txt requires --object_name and --target_words.")
        rows = _rows_from_txt(args.prompts_txt, args.object_name, args.target_words)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = _write_csv(rows, args.output)
    print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
