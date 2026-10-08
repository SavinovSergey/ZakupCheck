"""Overlap of dense and BM25 hit sets. No model download."""

from index.overlap import jaccard, mean_overlap, overlap_at_k
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
        max_chunk=1000,
        article_title_prefix=False,
    )


ART_33 = "44FZ:2026-08-04:art.33:ch.1:p.1"
ART_42 = "44FZ:2026-08-04:art.42:ch.1:p.1"


def test_jaccard_of_identical_and_disjoint_sets() -> None:
    assert jaccard({"a"}, {"a"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard({"a", "b"}, {"b", "c"}) == 1 / 3
    assert jaccard(set(), set()) == 1.0


def test_overlap_splits_gold_articles_between_the_two_lists() -> None:
    score = overlap_at_k(
        [ART_33, ART_42],
        [_chunk("d", ART_33)],
        [_chunk("b", ART_42)],
    )
    assert score.jaccard == 0.0
    assert score.dense_article == 0.5
    assert score.bm25_article == 0.5
    assert score.union_article == 1.0
    assert score.only_dense_article == 0.5
    assert score.only_bm25_article == 0.5
    assert score.both_article == 0.0


def test_same_chunk_counts_as_found_by_both() -> None:
    shared = [_chunk("c", ART_33)]
    score = overlap_at_k([ART_33], shared, shared)
    assert score.jaccard == 1.0
    assert score.union_article == 1.0
    assert score.both_article == 1.0
    assert score.only_dense_article == 0.0
    assert score.only_bm25_article == 0.0


def test_mean_overlap_averages_cases() -> None:
    shared = overlap_at_k([ART_33], [_chunk("c", ART_33)], [_chunk("c", ART_33)])
    split = overlap_at_k([ART_33], [_chunk("d", ART_33)], [_chunk("b", ART_42)])
    mean = mean_overlap([shared, split])
    assert mean.jaccard == 0.5
    assert mean.union_article == 1.0
    assert mean.only_dense_article == 0.5
    assert mean.both_article == 0.5
