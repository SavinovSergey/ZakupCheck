#!/usr/bin/env python3
"""BM25 recall@k на verified-кейсах для нескольких max_chunk (DESIGN §7.1, режим D)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from index.bm25 import Bm25Index, load_chunks  # noqa: E402
from index.eval_tables import (  # noqa: E402
    chunks_path,
    collect_rows,
    evaluate_index,
    load_verified,
    markdown_table,
)

DEFAULT_MAX_CHUNKS = (300, 500, 700, 1000, 1500, 2000, 2500)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BM25 recall@k по verified EvalCase")
    parser.add_argument("--edition-dir", type=Path, default=Path("data/law/2026-08-04"))
    parser.add_argument("--cases-dir", type=Path, default=Path("evals/cases"))
    parser.add_argument(
        "--max-chunk",
        type=int,
        action="append",
        dest="max_chunks",
        help="Ограничить прогон этими размерами (можно несколько). По умолчанию 300…2500.",
    )
    parser.add_argument("--query", choices=["raw", "blind", "both"], default="both")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evals/reports/bm25_structural_by_max_chunk.md"),
    )
    args = parser.parse_args(argv)
    sizes = tuple(args.max_chunks) if args.max_chunks else DEFAULT_MAX_CHUNKS
    variants = ("raw", "blind") if args.query == "both" else (args.query,)

    cases = load_verified(args.cases_dir)
    if not cases:
        print(f"нет verified-кейсов в {args.cases_dir}", file=sys.stderr)
        return 1

    summary_rows = []
    topic_rows = []

    for max_chunk in sizes:
        path = chunks_path(args.edition_dir, max_chunk)
        if not path.is_file():
            print(
                f"нет {path}; соберите: python scripts/build_law_corpus.py "
                f"--edition-dir {args.edition_dir} --max-chunk {max_chunk}",
                file=sys.stderr,
            )
            return 1
        index = Bm25Index(load_chunks(path))
        print(f"max_chunk={max_chunk} chunks={len(index.chunks)} tokens={index.token_count}", file=sys.stderr)
        for variant in variants:
            grouped, mean_ms = evaluate_index(index, cases, variant)
            summary, topics = collect_rows(grouped, max_chunk, variant, mean_ms)
            for row in summary:
                print(
                    f"  {variant} {row[2]} n={row[4][10].n} "
                    f"R@5 art={row[4][5].recall_article:.3f} "
                    f"R@10 art={row[4][10].recall_article:.3f} ms/query={row[5]:.1f}",
                    file=sys.stderr,
                )
            summary_rows.extend(summary)
            topic_rows.extend(topics)

    report = "\n".join(
        [
            "# BM25 structural chunks",
            "",
            "Запрос D: `complaint_argument_raw` / `complaint_argument_blind`.",
            "Hit: `gold_norms` ∩ объединение `norm_ids` top-k чанков.",
            "exact — буквальный id; art — совпадение `art.N`. all — доля кейсов, где найдено всё gold.",
            "R@5 и R@10 считаются по одному и тому же ранжированию: top-5 — это первые пять чанков из top-10.",
            "ms/query — среднее время одного search() на CPU после прогрева: "
            "разбор запроса и отбор top-10. Построение индекса не входит.",
            "",
            "## dev / eval",
            "",
            markdown_table(summary_rows),
            "",
            "## по темам",
            "",
            markdown_table(topic_rows),
            "",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {args.report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
