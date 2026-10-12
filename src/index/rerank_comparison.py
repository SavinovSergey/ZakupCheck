"""Pure ranking-comparison helpers for reranker runtime benchmarks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from index.recall import article_key
from schemas.corpus import Chunk


def ranked_ids(scores: Mapping[str, float]) -> list[str]:
    """Rank ids by descending score with deterministic id ties."""
    return sorted(scores, key=lambda item_id: (-scores[item_id], item_id))


def top_overlap(left: Sequence[str], right: Sequence[str], *, k: int) -> float:
    """Intersection divided by k, with a smaller common universe handled safely."""
    if k < 0:
        raise ValueError("k must be non-negative")
    denominator = min(k, len(left), len(right))
    if denominator == 0:
        return 1.0
    return len(set(left[:k]) & set(right[:k])) / denominator


def spearman_rank(left: Sequence[str], right: Sequence[str]) -> float:
    """Spearman correlation for two complete, tie-broken rankings."""
    if set(left) != set(right):
        raise ValueError("rankings must contain the same ids")
    n = len(left)
    if n < 2:
        return 1.0
    right_rank = {item_id: rank for rank, item_id in enumerate(right)}
    squared = sum((rank - right_rank[item_id]) ** 2 for rank, item_id in enumerate(left))
    return 1.0 - 6.0 * squared / (n * (n * n - 1))


def article_ranking(
    chunks: Mapping[str, Chunk],
    scores: Mapping[str, float],
) -> tuple[list[str], dict[str, str]]:
    """Rank articles by their best chunk and return that evidence chunk per article."""
    best: dict[str, tuple[float, str]] = {}
    for chunk_id, score in scores.items():
        chunk = chunks[chunk_id]
        for article in {article_key(norm_id) for norm_id in chunk.norm_ids}:
            candidate = (float(score), chunk_id)
            current = best.get(article)
            if current is None or candidate[0] > current[0] or (
                candidate[0] == current[0] and candidate[1] < current[1]
            ):
                best[article] = candidate
    ordered = sorted(best, key=lambda article: (-best[article][0], best[article][1], article))
    return ordered, {article: best[article][1] for article in ordered}


def article_evidence_ranking(
    chunks: Mapping[str, Chunk],
    scores: Mapping[str, float],
    *,
    limit: int = 10,
) -> list[Chunk]:
    """One best-scoring chunk for each of the top articles."""
    articles, evidence = article_ranking(chunks, scores)
    return [chunks[evidence[article]] for article in articles[:limit]]
