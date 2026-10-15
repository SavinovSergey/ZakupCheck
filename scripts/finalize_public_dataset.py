#!/usr/bin/env python3
"""Finalize a human-reviewed public dataset candidate for Hugging Face."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from privacy.finalize import FinalizationError, finalize_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "data/public/zakupcheck-benchmark-v1-candidate",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--version", required=True, help="Semantic version, e.g. 1.0.0")
    parser.add_argument(
        "--reviewed-by",
        required=True,
        help="Person or team that completed the record-by-record privacy review",
    )
    parser.add_argument(
        "--review-date",
        default=date.today().isoformat(),
        help="Review date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--license-file",
        type=Path,
        required=True,
        help="Reviewed data-license text to copy as DATA_LICENSE.md",
    )
    parser.add_argument(
        "--hf-license",
        default="other",
        help="Hugging Face dataset-card license identifier (default: other)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (
        ROOT / "data/public/releases" / f"zakupcheck-benchmark-{args.version}"
    )
    try:
        destination = finalize_dataset(
            input_dir=args.input_dir,
            output_dir=output_dir,
            version=args.version,
            reviewed_by=args.reviewed_by,
            review_date=args.review_date,
            license_file=args.license_file,
            hf_license=args.hf_license,
        )
    except FinalizationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Finalized publication-ready dataset at {destination}")
    print("Upload the contents of this directory, not the candidate directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
