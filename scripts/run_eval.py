#!/usr/bin/env python3
"""BM25 recall@k на verified-кейсах для нескольких max_chunk (DESIGN §7.1, режим D)."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from index.bm25 import Bm25Index, load_chunks  # noqa: E402
from index.recall import AggregateRecall, CaseScore, aggregate, score_case  # noqa: E402
from schemas.eval_case import EvalCase  # noqa: E402

DEFAULT_MAX_CHUNKS = (300, 500, 700, 1000, 1500, 2000, 2500)
KS = (5, 10)
QUERY_FIELDS = {"raw": "complaint_argument_raw", "blind": "complaint_argument_blind"}


def chunks_path(edition_dir: Path, max_chunk: int) -> Path:
    return edition_dir / f"chunks_structural_m{max_chunk}.jsonl"


def load_verified(cases_dir: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for path in sorted(cases_dir.glob("*.json")):
        if path.name == "schema_example.json":
            continue
        case = EvalCase.model_validate_json(path.read_text(encoding="utf-8"))
        if case.annotation_status == "verified":
            cases.append(case)
    return cases


def query_text(case: EvalCase, variant: str) -> str:
    text = getattr(case, QUERY_FIELDS[variant])
    if not text:
        raise ValueError(f"{case.case_id}: empty {variant} query")
    return text


def score_at_k(case: EvalCase, ranked_ids: list[set[str]], k: int) -> CaseScore:
    return score_case(case.gold_norms, ranked_ids[k])


def run_size(
    index: Bm25Index,
    cases: list[EvalCase],
    variant: str,
) -> dict[tuple[str, str], dict[int, list[CaseScore]]]:
    """(split, topic) -> k -> scores. topic '*' is the split total."""
    grouped: dict[tuple[str, str], dict[int, list[CaseScore]]] = defaultdict(
        lambda: {k: [] for k in KS}
    )
    for case in cases:
        ranked = index.search(query_text(case, variant), k=max(KS))
        ids_by_k: dict[int, set[str]] = {}
        for k in KS:
            ids: set[str] = set()
            for chunk in ranked[:k]:
                ids.update(chunk.norm_ids)
            ids_by_k[k] = ids
        for k in KS:
            scored = score_at_k(case, ids_by_k, k)
            grouped[(case.split, "*")][k].append(scored)
            grouped[(case.split, case.topic)][k].append(scored)
    return grouped


def markdown_table(
    rows: list[tuple[int, str, str, str, dict[int, AggregateRecall]]],
) -> str:
    header = (
        "| max_chunk | query | split | topic | n "
        "| R@5 exact | all@5 exact | R@10 exact | all@10 exact "
        "| R@5 art | all@5 art | R@10 art | all@10 art |"
    )
    sep = "|" + "|".join(["---"] * 13) + "|"
    lines = [header, sep]
    for max_chunk, variant, split, topic, by_k in rows:
        m5, m10 = by_k[5], by_k[10]
        lines.append(
            f"| {max_chunk} | {variant} | {split} | {topic} | {m5.n} "
            f"| {m5.recall_exact:.3f} | {m5.all_exact:.3f} "
            f"| {m10.recall_exact:.3f} | {m10.all_exact:.3f} "
            f"| {m5.recall_article:.3f} | {m5.all_article:.3f} "
            f"| {m10.recall_article:.3f} | {m10.all_article:.3f} |"
        )
    return "\n".join(lines)


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

    summary_rows: list[tuple[int, str, str, str, dict[int, AggregateRecall]]] = []
    topic_rows: list[tuple[int, str, str, str, dict[int, AggregateRecall]]] = []

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
            grouped = run_size(index, cases, variant)
            for (split, topic), by_k_scores in sorted(grouped.items()):
                by_k = {k: aggregate(scores) for k, scores in by_k_scores.items()}
                row = (max_chunk, variant, split, topic, by_k)
                if topic == "*":
                    summary_rows.append((max_chunk, variant, split, "all", by_k))
                else:
                    topic_rows.append(row)

    report = "\n".join(
        [
            "# BM25 structural chunks",
            "",
            "Запрос D: `complaint_argument_raw` / `complaint_argument_blind`.",
            "Hit: `gold_norms` ∩ объединение `norm_ids` top-k чанков.",
            "exact — буквальный id; art — совпадение `art.N`. all — доля кейсов, где найдено всё gold.",
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
