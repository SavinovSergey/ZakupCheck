"""Shared recall tables for BM25 and dense eval scripts."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from index.recall import AggregateRecall, CaseScore, aggregate, score_case
from schemas.corpus import Chunk
from schemas.eval_case import EvalCase

KS = (5, 10)
QUERY_FIELDS = {"raw": "complaint_argument_raw", "blind": "complaint_argument_blind"}

# max_chunk, variant, split, topic, metrics, mean search milliseconds per query
Row = tuple[int, str, str, str, dict[int, AggregateRecall], float]


class Searcher(Protocol):
    def search(self, query: str, k: int) -> list[Chunk]: ...


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


def evaluate_index(
    index: Searcher,
    cases: Sequence[EvalCase],
    variant: str,
) -> tuple[
    dict[tuple[str, str], dict[int, list[CaseScore]]],
    dict[tuple[str, str], float],
]:
    """(split, topic) -> k -> scores, and mean search ms for that group.

    topic '*' is the split total. One untimed search warms the encoder
    before the clock starts. The clock covers search() only: query encoding
    and top-k lookup, not index construction.
    """
    grouped: dict[tuple[str, str], dict[int, list[CaseScore]]] = defaultdict(lambda: {k: [] for k in KS})
    elapsed: dict[tuple[str, str], list[float]] = defaultdict(list)
    if cases:
        index.search(query_text(cases[0], variant), k=max(KS))
    for case in cases:
        started = time.perf_counter()
        ranked = index.search(query_text(case, variant), k=max(KS))
        ms = (time.perf_counter() - started) * 1000
        elapsed[(case.split, "*")].append(ms)
        elapsed[(case.split, case.topic)].append(ms)
        for k in KS:
            ids: set[str] = set()
            for chunk in ranked[:k]:
                ids.update(chunk.norm_ids)
            scored = score_case(case.gold_norms, ids)
            grouped[(case.split, "*")][k].append(scored)
            grouped[(case.split, case.topic)][k].append(scored)
    mean_ms = {key: sum(values) / len(values) for key, values in elapsed.items()}
    return grouped, mean_ms


def collect_rows(
    grouped: dict[tuple[str, str], dict[int, list[CaseScore]]],
    max_chunk: int,
    variant: str,
    mean_ms: dict[tuple[str, str], float],
) -> tuple[list[Row], list[Row]]:
    summary: list[Row] = []
    topics: list[Row] = []
    for (split, topic), by_k_scores in sorted(grouped.items()):
        by_k = {k: aggregate(scores) for k, scores in by_k_scores.items()}
        ms = mean_ms.get((split, topic), 0.0)
        if topic == "*":
            summary.append((max_chunk, variant, split, "all", by_k, ms))
        else:
            topics.append((max_chunk, variant, split, topic, by_k, ms))
    return summary, topics


def markdown_table(rows: Sequence[Row]) -> str:
    header = (
        "| max_chunk | query | split | topic | n "
        "| R@5 exact | all@5 exact | R@10 exact | all@10 exact "
        "| R@5 art | all@5 art | R@10 art | all@10 art | ms/query |"
    )
    sep = "|" + "|".join(["---"] * 14) + "|"
    lines = [header, sep]
    for max_chunk, variant, split, topic, by_k, ms in rows:
        m5, m10 = by_k[5], by_k[10]
        lines.append(
            f"| {max_chunk} | {variant} | {split} | {topic} | {m5.n} "
            f"| {m5.recall_exact:.3f} | {m5.all_exact:.3f} "
            f"| {m10.recall_exact:.3f} | {m10.all_exact:.3f} "
            f"| {m5.recall_article:.3f} | {m5.all_article:.3f} "
            f"| {m10.recall_article:.3f} | {m10.all_article:.3f} | {ms:.1f} |"
        )
    return "\n".join(lines)
