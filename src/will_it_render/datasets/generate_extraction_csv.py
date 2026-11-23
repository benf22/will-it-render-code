"""CLI to validate / normalize a prompts CSV for `run_extraction.py`.

    python -m will_it_render.datasets.generate_extraction_csv \
        --manual_prompts_csv my_prompts.csv \
        --output validated_prompts.csv

The input CSV needs `prompt`, `object_name`, `target_words` columns (see data/WIR_testset/ for the
paper benchmark format). The LLM prompt-enrichment path (Llama-3) from the research repo is not
part of this release — author prompt CSVs directly (or reuse data/WIR_testset/).
"""
from __future__ import annotations

import argparse
import csv
import sys

_REQUIRED_COLS = {"prompt", "object_name", "target_words"}


def _validate_manual_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = _REQUIRED_COLS - set(reader.fieldnames or [])
        if missing:
            print(f"ERROR: manual CSV missing columns: {missing}", file=sys.stderr)
            sys.exit(1)
        for row in reader:
            rows.append({k: row[k] for k in _REQUIRED_COLS})
    return rows


def _write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "object_name", "target_words"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual_prompts_csv", required=True, help="prompts CSV to validate")
    parser.add_argument("--output", required=True, help="Output extraction CSV path")
    args = parser.parse_args()
    _write_csv(_validate_manual_csv(args.manual_prompts_csv), args.output)


if __name__ == "__main__":
    main()
