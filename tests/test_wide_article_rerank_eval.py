from __future__ import annotations

import sys
from pathlib import Path

from index.recall import AggregateRecall
from schemas.corpus import Chunk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_wide_article_rerank_eval import (
    SweepResult,
    all_chunks_for_articles,
    chunks_by_article,
    pareto_frontier,
    select_winner,
    unique_article_prefix,
)


def _chunk(chunk_id: str, article: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        edition_id="2026-08-04",
        chunking="point",
        text=chunk_id,
        char_start=0,
        char_end=len(chunk_id),
        norm_ids=[f"44FZ:2026-08-04:{article}:ch.1:p.1"],
        max_chunk=700,
    )


def _result(article: float, chunks: float, exact: float = 0.0) -> SweepResult:
    metric = AggregateRecall(1, exact, exact, article, article)
    return SweepResult({}, metric, metric, chunks / 2, chunks, [])


def test_unique_article_prefix_looks_past_repeated_chunks() -> None:
    ranking = [_chunk("a1", "art.1"), _chunk("a2", "art.1"), _chunk("b", "art.2")]
    assert unique_article_prefix(ranking, 2) == ["art.1", "art.2"]


def test_all_chunks_for_articles_deduplicates_and_sorts() -> None:
    a = _chunk("z", "art.1")
    b = _chunk("a", "art.2")
    mapping = chunks_by_article([a, b])
    mapping["art.1"].append(b)
    assert [chunk.chunk_id for chunk in all_chunks_for_articles(["art.1", "art.2"], mapping)] == [
        "a",
        "z",
    ]


def test_pareto_frontier_removes_more_expensive_equal_quality_point() -> None:
    cheap = _result(0.6, 20)
    expensive = _result(0.6, 30)
    better = _result(0.7, 40)
    assert pareto_frontier([expensive, better, cheap]) == [cheap, better]


def test_winner_uses_exact_inside_article_tolerance_then_cost() -> None:
    lower_article_better_exact = _result(0.695, 30, exact=0.4)
    higher_article = _result(0.7, 20, exact=0.3)
    assert select_winner([higher_article, lower_article_better_exact]) is lower_article_better_exact

    cheap = _result(0.7, 10, exact=0.3)
    expensive = _result(0.7, 20, exact=0.3)
    assert select_winner([expensive, cheap]) is cheap
