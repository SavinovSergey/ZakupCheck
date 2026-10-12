from __future__ import annotations

from index.rerank_comparison import (
    article_evidence_ranking,
    article_ranking,
    ranked_ids,
    spearman_rank,
    top_overlap,
)
from schemas.corpus import Chunk


def _chunk(chunk_id: str, *norm_ids: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        edition_id="2026-08-04",
        chunking="point",
        text=chunk_id,
        char_start=0,
        char_end=len(chunk_id),
        norm_ids=list(norm_ids),
    )


def test_rank_and_overlap_are_deterministic() -> None:
    assert ranked_ids({"b": 1.0, "a": 1.0, "c": 2.0}) == ["c", "a", "b"]
    assert top_overlap(["a", "b"], ["b", "c"], k=2) == 0.5


def test_spearman_rank() -> None:
    assert spearman_rank(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert spearman_rank(["a", "b", "c"], ["c", "b", "a"]) == -1.0


def test_article_ranking_uses_best_chunk() -> None:
    chunks = {
        "a1": _chunk("a1", "44FZ:art.1:p.1"),
        "a2": _chunk("a2", "44FZ:art.1:p.2"),
        "b": _chunk("b", "44FZ:art.2"),
    }
    articles, evidence = article_ranking(chunks, {"a1": 0.1, "a2": 0.9, "b": 0.8})
    assert articles == ["art.1", "art.2"]
    assert evidence["art.1"] == "a2"
    assert [chunk.chunk_id for chunk in article_evidence_ranking(chunks, {"a1": 0.1, "a2": 0.9, "b": 0.8})] == ["a2", "b"]
