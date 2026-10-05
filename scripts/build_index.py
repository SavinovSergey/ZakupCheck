#!/usr/bin/env python3
"""Загрузить structural chunks и напечатать размер BM25-индекса (в памяти, без файла индекса)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from index.bm25 import Bm25Index, load_chunks  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Проверить BM25-индекс по chunks jsonl")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/law/2026-08-04/chunks_structural_m2000.jsonl"),
    )
    args = parser.parse_args(argv)
    if not args.chunks.is_file():
        print(f"нет файла чанков: {args.chunks}", file=sys.stderr)
        return 1
    chunks = load_chunks(args.chunks)
    index = Bm25Index(chunks)
    print(f"chunks={len(index.chunks)} tokens={index.token_count} file={args.chunks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
