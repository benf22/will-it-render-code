"""CLI to generate the extraction CSV consumed by run_extraction.py.

Usage (enriched — generates prompts via Llama-3-8B):
    python -m dataset_creation.generate_extraction_csv \\
        --objects_csv objects.csv \\
        --output enriched_prompts.csv \\
        [--n_prompts 10] \\
        [--model meta-llama/Meta-Llama-3-8B-Instruct] \\
        [--device cuda]

Usage (manual prompts, skip enrichment):
    python -m dataset_creation.generate_extraction_csv \\
        --manual_prompts_csv my_prompts.csv \\
        --output validated_prompts.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from will_it_render.datasets.csv_reader import read_objects_csv


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
    parser.add_argument("--objects_csv", help="CSV with object_name, target_words columns")
    parser.add_argument("--manual_prompts_csv", help="Pre-written prompts CSV (skips enrichment)")
    parser.add_argument("--output", required=True, help="Output extraction CSV path")
    parser.add_argument("--n_prompts", type=int, default=10)
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.manual_prompts_csv:
        rows = _validate_manual_csv(args.manual_prompts_csv)
        _write_csv(rows, args.output)
        return

    if not args.objects_csv:
        parser.error("Either --objects_csv or --manual_prompts_csv is required")

    entries = read_objects_csv(args.objects_csv)
    from will_it_render.datasets.enricher import build_enricher, enrich_entry
    pipe = build_enricher(args.model, args.device)

    rows = []
    for entry in entries:
        prompts = enrich_entry(entry, pipe, n=args.n_prompts)
        for prompt in prompts:
            rows.append({
                "prompt": prompt,
                "object_name": entry.object_name,
                "target_words": ",".join(entry.target_words),
            })

    _write_csv(rows, args.output)


if __name__ == "__main__":
    main()
