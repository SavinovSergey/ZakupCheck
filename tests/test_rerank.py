from __future__ import annotations

from collections.abc import Sequence

import pytest

from index.rerank import (
    article_candidate_map,
    article_candidate_pool,
    article_candidate_pool_from_map,
    article_neighbors,
    best_chunk_per_article,
    best_chunks_in_article_order,
    build_evidence_pack,
    build_adaptive_evidence_pack,
    chunk_structure_keys,
    diverse_chunk_selection,
    rank_articles_by_best_chunk,
    rerank_chunks,
    source_quota_articles,
    two_pass_article_rerank,
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


def test_precomputed_article_candidates_reproduce_direct_pool() -> None:
    dense = [_chunk("d1", "art.1"), _chunk("d2", "art.2"), _chunk("d1b", "art.1")]
    bm25 = [_chunk("b2", "art.2"), _chunk("b1", "art.1")]
    candidates = article_candidate_map(
        [dense, bm25], ["art.1", "art.2"], per_article=2
    )
    cached = article_candidate_pool_from_map(
        candidates, ["art.2", "art.1"], per_article=2
    )
    direct = article_candidate_pool(
        [dense, bm25], ["art.2", "art.1"], per_article=2
    )
    assert [chunk.chunk_id for chunk in cached] == [chunk.chunk_id for chunk in direct]


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


def test_rank_articles_by_best_chunk_exposes_winning_evidence() -> None:
    chunks = [_chunk("a", "art.1"), _chunk("b", "art.2")]
    articles, evidence = rank_articles_by_best_chunk(
        chunks,
        {"a": 0.2, "b": 0.7},
        ["art.1", "art.2"],
    )
    assert articles == ["art.2", "art.1"]
    assert evidence["art.2"].chunk_id == "b"


def test_best_chunks_in_article_order_keeps_external_article_ranking() -> None:
    chunks = [_chunk("a", "art.1"), _chunk("b", "art.2")]
    ranked = best_chunks_in_article_order(
        chunks,
        {"a": 0.9, "b": 0.1},
        ["art.2", "art.1"],
        limit=2,
    )
    assert [chunk.chunk_id for chunk in ranked] == ["b", "a"]


def test_two_pass_article_rerank_counts_unique_scored_chunks() -> None:
    dense = [
        _chunk("a1", "art.1"),
        _chunk("b1", "art.2"),
        _chunk("a2", "art.1"),
        _chunk("b2", "art.2"),
    ]
    bm25 = [
        _chunk("c1", "art.3"),
        _chunk("c2", "art.3"),
    ]
    scores = {"a1": 0.9, "a2": 0.8, "b1": 0.7, "b2": 0.6, "c1": 0.1, "c2": 1.0}
    shortlist, final_pool, output, scored = two_pass_article_rerank(
        [dense, bm25],
        ["art.1", "art.2", "art.3"],
        scores,
        first_pass_per_article=1,
        shortlist_articles=2,
        final_per_article=2,
        limit=2,
    )
    assert shortlist == ["art.1", "art.2"]
    assert [chunk.chunk_id for chunk in final_pool] == ["a1", "a2", "b1", "b2"]
    assert [chunk.chunk_id for chunk in output] == ["a1", "b1"]
    assert [chunk.chunk_id for chunk in scored] == ["a1", "a2", "b1", "b2", "c1"]


def test_two_pass_article_rerank_rejects_shortlist_below_output_limit() -> None:
    with pytest.raises(ValueError, match="at least limit"):
        two_pass_article_rerank(
            [[]],
            [],
            {},
            first_pass_per_article=1,
            shortlist_articles=9,
            final_per_article=2,
            limit=10,
        )


def test_diverse_chunk_selection_avoids_a_near_duplicate() -> None:
    duplicate_a = _chunk("a", "art.1").model_copy(update={"text": "alpha beta gamma"})
    duplicate_b = _chunk("b", "art.1").model_copy(update={"text": "alpha beta gamma"})
    different = _chunk("c", "art.1").model_copy(
        update={
            "text": "delta epsilon",
            "norm_ids": ["44FZ:2026-08-04:art.1:ch.1:p.2"],
        }
    )
    selected = diverse_chunk_selection(
        [duplicate_a, duplicate_b, different],
        {"a": 1.0, "b": 0.9, "c": 0.8},
        limit=2,
        diversity_weight=0.5,
    )
    assert [chunk.chunk_id for chunk in selected] == ["a", "c"]


def test_build_evidence_pack_guarantees_primaries_then_adds_best_extra() -> None:
    a1 = _chunk("a1", "art.1")
    a2 = _chunk("a2", "art.1")
    b1 = _chunk("b1", "art.2")
    b2 = _chunk("b2", "art.2")
    pack = build_evidence_pack(
        ["art.1", "art.2"],
        {"art.1": [a1, a2], "art.2": [b1, b2]},
        {"a1": 0.5, "a2": 0.9, "b1": 0.4, "b2": 0.8},
        budget=3,
        per_article_cap=2,
        primary_articles=2,
    )
    assert [chunk.chunk_id for chunk in pack] == ["a1", "b1", "a2"]


def test_article_neighbors_follow_structural_offsets() -> None:
    middle = _chunk("middle", "art.1").model_copy(update={"char_start": 20, "char_end": 29})
    before = _chunk("before", "art.1").model_copy(update={"char_start": 10, "char_end": 19})
    after = _chunk("after", "art.1").model_copy(update={"char_start": 30, "char_end": 39})
    foreign = _chunk("foreign", "art.2").model_copy(update={"char_start": 21, "char_end": 22})
    assert [chunk.chunk_id for chunk in article_neighbors(
        [after, foreign, middle, before], middle
    )] == ["before", "after"]


def test_chunk_structure_keys_extract_part_and_point() -> None:
    chunk = _chunk("a", "art.1").model_copy(
        update={"norm_ids": ["44FZ:2026-08-04:art.1:ch.2:p.3"]}
    )
    assert chunk_structure_keys(chunk) == {("art.1", "2", "3")}


def test_adaptive_pack_respects_ratio_token_budget_and_hierarchy_bonus() -> None:
    a1 = _chunk("a1", "art.1")
    a2 = _chunk("a2", "art.1").model_copy(
        update={"norm_ids": ["44FZ:2026-08-04:art.1:ch.2:p.1"]}
    )
    a3 = _chunk("a3", "art.1")
    b1 = _chunk("b1", "art.2")
    pack = build_adaptive_evidence_pack(
        ["art.1", "art.2"],
        {"art.1": [a1, a3, a2], "art.2": [b1]},
        {"a1": 1.0, "a2": 0.55, "a3": 0.6, "b1": 0.8},
        {"a1": 10, "a2": 10, "a3": 10, "b1": 10},
        max_chunks=3,
        token_budget=30,
        per_article_cap=2,
        min_relative_score=0.5,
        hierarchy_bonus=0.1,
    )
    assert [chunk.chunk_id for chunk in pack] == ["a1", "b1", "a2"]
