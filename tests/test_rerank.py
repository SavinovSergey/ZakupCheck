from __future__ import annotations

from collections.abc import Sequence

import pytest

from index.rerank import (
    article_candidate_pool,
    best_chunk_per_article,
    rerank_chunks,
    source_quota_articles,
)
from schemas.corpus import Chunk


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


class FakeScorer:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def score(self, query: str, chunks: Sequence[Chunk]) -> list[float]:
        return [self.scores[chunk.chunk_id] for chunk in chunks]


def test_article_candidate_pool_interleaves_sources_and_caps_articles() -> None:
    dense = [_chunk("d33a", "art.33"), _chunk("d33b", "art.33"), _chunk("d42", "art.42")]
    bm25 = [_chunk("b33", "art.33"), _chunk("b42", "art.42")]
    pool = article_candidate_pool([dense, bm25], ["art.42", "art.33"], per_article=2)
    assert [chunk.chunk_id for chunk in pool] == ["b42", "d42", "d33a", "b33"]


def test_rerank_chunks_uses_score_then_chunk_id() -> None:
    chunks = [_chunk("b", "art.33"), _chunk("a", "art.42"), _chunk("c", "art.48")]
    ranked = rerank_chunks("query", chunks, FakeScorer({"a": 1.0, "b": 1.0, "c": 0.0}))
    assert [chunk.chunk_id for chunk in ranked] == ["a", "b", "c"]


def test_rerank_chunks_rejects_wrong_score_count() -> None:
    class BrokenScorer:
        def score(self, query: str, chunks: Sequence[Chunk]) -> list[float]:
            return []

    with pytest.raises(ValueError, match="expected 1 scores"):
        rerank_chunks("query", [_chunk("a", "art.33")], BrokenScorer())


def test_source_quota_articles_preserves_dense_then_bm25_order() -> None:
    assert source_quota_articles(
        ["art.1", "art.2", "art.3"],
        ["art.2", "art.4", "art.5"],
        dense_quota=2,
        bm25_quota=3,
    ) == ["art.1", "art.2", "art.4", "art.5"]


def test_best_chunk_per_article_uses_max_score_and_deterministic_ties() -> None:
    chunks = [
        _chunk("a-low", "art.1"),
        _chunk("a-best", "art.1"),
        _chunk("b", "art.2"),
    ]
    ranked = best_chunk_per_article(
        chunks,
        {"a-low": 0.1, "a-best": 0.9, "b": 0.9},
        ["art.1", "art.2"],
        limit=2,
    )
    assert [chunk.chunk_id for chunk in ranked] == ["a-best", "b"]
