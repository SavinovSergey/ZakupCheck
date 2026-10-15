#!/usr/bin/env python3
"""Download and validate the public ZakupCheck benchmark from Hugging Face."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_access.public_dataset import (  # noqa: E402
    DEFAULT_HF_DATASET_ID,
    DEFAULT_HF_REVISION,
    PublicDatasetError,
    download_public_dataset,
)


def _safe_revision(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_HF_DATASET_ID)
    parser.add_argument("--revision", default=DEFAULT_HF_REVISION)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use the local Hugging Face cache without network access",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir or (
        ROOT
        / "data/hf/zakupcheck-44fz-retrieval-benchmark"
        / _safe_revision(args.revision)
    )
    try:
        summary = download_public_dataset(
            output_dir,
            repo_id=args.repo_id,
            revision=args.revision,
            force_download=args.force_download,
            local_files_only=args.local_files_only,
        )
    except PublicDatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Downloaded {summary.repo_id}@{args.revision} to {summary.root}")
    print(
        f"Validated release {summary.version}: "
        f"dev={summary.dev_count}, eval={summary.eval_count}, "
        f"eval_labels_included={summary.eval_labels_included}"
    )
    print(f"Use labeled cases with --cases-dir {summary.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
