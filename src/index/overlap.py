"""Overlap of a dense top-k with BM25 on the same chunks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from index.recall import article_key
from schemas.corpus import Chunk

KS = (5, 10)


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _norm_ids(chunks: Sequence[Chunk]) -> set[str]:
    found: set[str] = set()
    for chunk in chunks:
        found.update(chunk.norm_ids)
    return found


@dataclass(frozen=True)
class OverlapScore:
    """One case at one k. Article rates are shares of that case's gold articles.

    only_dense + only_bm25 + both equals union_article.
    """

    jaccard: float
    dense_article: float
    bm25_article: float
    union_article: float
    only_dense_article: float
    only_bm25_article: float
    both_article: float


def overlap_at_k(gold_norms: Sequence[str], dense: Sequence[Chunk], bm25: Sequence[Chunk]) -> OverlapScore:
    """Jaccard is over chunk ids. Article rates use the norm ids inside those chunks."""
    gold_articles = {article_key(norm_id) for norm_id in gold_norms}
    if not gold_articles:
        raise ValueError("gold_norms must not be empty")
    dense_norms = _norm_ids(dense)
    bm25_norms = _norm_ids(bm25)
    dense_articles = {article_key(norm_id) for norm_id in dense_norms}
    bm25_articles = {article_key(norm_id) for norm_id in bm25_norms}
    found_dense = gold_articles & dense_articles
    found_bm25 = gold_articles & bm25_articles
    n = len(gold_articles)
    return OverlapScore(
        jaccard=jaccard({chunk.chunk_id for chunk in dense}, {chunk.chunk_id for chunk in bm25}),
        dense_article=len(found_dense) / n,
        bm25_article=len(found_bm25) / n,
        union_article=len(found_dense | found_bm25) / n,
        only_dense_article=len(found_dense - found_bm25) / n,
        only_bm25_article=len(found_bm25 - found_dense) / n,
        both_article=len(found_dense & found_bm25) / n,
    )


def mean_overlap(scores: Sequence[OverlapScore]) -> OverlapScore:
    n = len(scores)
    if n == 0:
        return OverlapScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return OverlapScore(
        jaccard=sum(item.jaccard for item in scores) / n,
        dense_article=sum(item.dense_article for item in scores) / n,
        bm25_article=sum(item.bm25_article for item in scores) / n,
        union_article=sum(item.union_article for item in scores) / n,
        only_dense_article=sum(item.only_dense_article for item in scores) / n,
        only_bm25_article=sum(item.only_bm25_article for item in scores) / n,
        both_article=sum(item.both_article for item in scores) / n,
    )
