#!/usr/bin/env python3
"""Пересечение top-k dense и BM25 по уже посчитанным эмбеддингам чанков."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from index.bm25 import Bm25Index, load_chunks  # noqa: E402
from index.dense import DenseIndex, SentenceTransformerEncoder  # noqa: E402
from index.eval_tables import chunks_path, load_verified, query_text  # noqa: E402
from index.overlap import KS, OverlapScore, mean_overlap, overlap_at_k  # noqa: E402
from run_dense_eval import DEFAULT_MODEL, PROFILES, profile_for  # noqa: E402


def default_models() -> list[str]:
    return [DEFAULT_MODEL, *PROFILES.keys()]


def markdown_table(rows: list[tuple[str, int, str, str, int, int, OverlapScore]]) -> str:
    header = (
        "| model | max_chunk | query | split | n | k | jaccard "
        "| dense R art | bm25 R art | union R art | only dense | only bm25 | both |"
    )
    sep = "|" + "|".join(["---"] * 13) + "|"
    lines = [header, sep]
    for model, max_chunk, variant, split, n, k, score in rows:
        lines.append(
            f"| {model} | {max_chunk} | {variant} | {split} | {n} | {k} | {score.jaccard:.3f} "
            f"| {score.dense_article:.3f} | {score.bm25_article:.3f} | {score.union_article:.3f} "
            f"| {score.only_dense_article:.3f} | {score.only_bm25_article:.3f} | {score.both_article:.3f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Jaccard и Recall объединения dense с BM25")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--edition-dir", type=Path, default=Path("data/law/2026-08-04"))
    parser.add_argument("--cases-dir", type=Path, default=Path("evals/cases"))
    parser.add_argument("--max-chunk", type=int, action="append", dest="max_chunks")
    parser.add_argument("--query", choices=["raw", "blind", "both"], default="both")
    parser.add_argument("--report", type=Path, default=Path("evals/reports/overlap_dense_bm25.md"))
    args = parser.parse_args(argv)
    models = args.models if args.models else default_models()
    variants = ("raw", "blind") if args.query == "both" else (args.query,)

    cases = load_verified(args.cases_dir)
    if not cases:
        print(f"нет verified-кейсов в {args.cases_dir}", file=sys.stderr)
        return 1

    rows: list[tuple[str, int, str, str, int, int, OverlapScore]] = []
    missing: list[str] = []

    for model in models:
        profile = profile_for(model)
        sizes = tuple(args.max_chunks) if args.max_chunks else profile.max_chunks
        print(f"loading {model}", file=sys.stderr)
        encoder = SentenceTransformerEncoder(
            model,
            query_prefix=profile.query_prefix,
            document_prefix=profile.document_prefix,
            max_tokens=profile.max_tokens,
        )
        for max_chunk in sizes:
            path = chunks_path(args.edition_dir, max_chunk)
            if not path.is_file():
                print(f"нет {path}", file=sys.stderr)
                return 1
            chunks = load_chunks(path)
            dense = DenseIndex.load_cached(
                chunks,
                encoder,
                cache_dir=args.edition_dir / "dense",
                model_name=model,
                max_chunk=max_chunk,
            )
            if dense is None:
                note = f"{model} m{max_chunk}"
                missing.append(note)
                print(f"нет кэша эмбеддингов для {note}, пропускаю", file=sys.stderr)
                continue
            bm25 = Bm25Index(chunks)
            print(f"max_chunk={max_chunk} chunks={len(chunks)}", file=sys.stderr)
            for variant in variants:
                grouped: dict[str, dict[int, list[OverlapScore]]] = defaultdict(lambda: {k: [] for k in KS})
                for case in cases:
                    query = query_text(case, variant)
                    dense_hits = dense.search(query, k=max(KS))
                    bm25_hits = bm25.search(query, k=max(KS))
                    for k in KS:
                        grouped[case.split][k].append(
                            overlap_at_k(case.gold_norms, dense_hits[:k], bm25_hits[:k])
                        )
                for split, by_k in sorted(grouped.items()):
                    for k, scores in by_k.items():
                        mean = mean_overlap(scores)
                        rows.append((model, max_chunk, variant, split, len(scores), k, mean))
                        print(
                            f"  {variant} {split} k={k} jaccard={mean.jaccard:.3f} "
                            f"union R art={mean.union_article:.3f} "
                            f"only dense={mean.only_dense_article:.3f} "
                            f"only bm25={mean.only_bm25_article:.3f}",
                            file=sys.stderr,
                        )

    report = "\n".join(
        [
            "# Пересечение dense и BM25",
            "",
            "Запрос D: `complaint_argument_raw` / `complaint_argument_blind`.",
            "Jaccard — пересечение множеств id чанков в top-k.",
            "R art — доля золотых статей в этих чанках. union — статья есть хотя бы в одном списке.",
            "only dense и only bm25 — доля золотых статей, которые нашла только эта модель.",
            "only dense + only bm25 + both = union R art.",
            "Эмбеддинги чанков взяты из кэша прогона. Запросы кодируются заново.",
            "",
            markdown_table(rows),
            "",
        ]
    )
    if missing:
        report += "Нет кэша, строки не посчитаны: " + ", ".join(missing) + ".\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {args.report}", file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
