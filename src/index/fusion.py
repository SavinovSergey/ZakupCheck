"""Fusing two rankings and restricting a ranking to a few articles."""

from __future__ import annotations

import re
from collections.abc import Sequence

from index.recall import article_key
from schemas.corpus import Chunk

# Standard RRF constant from Cormack, Clarke, Buettcher (2009).
RRF_K = 60
_POINTER_RE = re.compile(r"в соответствии со стать", re.IGNORECASE)


def rrf(rankings: Sequence[Sequence[Chunk]], *, k: int = RRF_K, limit: int = 10) -> list[Chunk]:
    """Reciprocal rank fusion. Earlier rank in a list scores higher. Ties break by chunk id."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if limit < 0:
        raise ValueError("limit must be non-negative")
    scores: dict[str, float] = {}
    chosen: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chosen.setdefault(chunk.chunk_id, chunk)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [chosen[chunk_id] for chunk_id in ordered[:limit]]


def weighted_rrf(
    rankings: Sequence[Sequence[Chunk]],
    *,
    weights: Sequence[float],
    k: int = RRF_K,
    pool: int = 50,
    limit: int = 10,
) -> list[Chunk]:
    """Weighted reciprocal-rank fusion over fixed-size input pools.

    Ties are resolved by the best source rank and then by ``chunk_id`` so the
    result is stable across runs.
    """
    if len(rankings) != len(weights):
        raise ValueError("rankings and weights must have the same length")
    if k < 0 or pool < 0 or limit < 0:
        raise ValueError("k, pool, and limit must be non-negative")
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative")
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    chosen: dict[str, Chunk] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, chunk in enumerate(ranking[:pool], start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + weight / (k + rank)
            best_rank[chunk.chunk_id] = min(best_rank.get(chunk.chunk_id, rank), rank)
            chosen.setdefault(chunk.chunk_id, chunk)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], best_rank[chunk_id], chunk_id))
    return [chosen[chunk_id] for chunk_id in ordered[:limit]]


def interleave(
    rankings: Sequence[Sequence[Chunk]],
    *,
    pool: int = 50,
    limit: int = 10,
) -> list[Chunk]:
    """Round-robin unique chunks, preserving the order of source rankings."""
    if pool < 0 or limit < 0:
        raise ValueError("pool and limit must be non-negative")
    picked: list[Chunk] = []
    seen: set[str] = set()
    for rank in range(pool):
        for ranking in rankings:
            if rank >= len(ranking):
                continue
            chunk = ranking[rank]
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            picked.append(chunk)
            if len(picked) == limit:
                return picked
    return picked


def article_diversify(ranked: Sequence[Chunk], *, pool: int = 50, limit: int = 10) -> list[Chunk]:
    """Keep the highest-ranked representative of each article."""
    if pool < 0 or limit < 0:
        raise ValueError("pool and limit must be non-negative")
    picked: list[Chunk] = []
    seen_articles: set[str] = set()
    for chunk in ranked[:pool]:
        articles = {article_key(norm_id) for norm_id in chunk.norm_ids}
        if articles & seen_articles:
            continue
        picked.append(chunk)
        seen_articles.update(articles)
        if len(picked) == limit:
            break
    return picked


def top_articles(ranked: Sequence[Chunk], *, article_pool: int, n_articles: int) -> list[str]:
    """Articles in the order of their first chunk inside the pool."""
    if article_pool < 0 or n_articles < 0:
        raise ValueError("article_pool and n_articles must be non-negative")
    best_rank: dict[str, int] = {}
    for rank, chunk in enumerate(ranked[:article_pool], start=1):
        for art in {article_key(norm_id) for norm_id in chunk.norm_ids}:
            best_rank.setdefault(art, rank)
    return [art for art, _ in sorted(best_rank.items(), key=lambda item: (item[1], item[0]))[:n_articles]]


def article_then_points(
    ranked: Sequence[Chunk],
    *,
    article_pool: int,
    n_articles: int,
    limit: int,
) -> list[Chunk]:
    """Keep the top articles by first appearance, then their chunks in the original order."""
    if limit < 0:
        raise ValueError("limit must be non-negative")
    chosen = set(top_articles(ranked, article_pool=article_pool, n_articles=n_articles))
    picked: list[Chunk] = []
    for chunk in ranked:
        arts = {article_key(norm_id) for norm_id in chunk.norm_ids}
        if arts & chosen:
            picked.append(chunk)
            if len(picked) == limit:
                break
    return picked


def is_pointer_text(text: str, *, max_chars: int = 240) -> bool:
    """A one-line cross-reference such as 'в соответствии со статьей 33'."""
    folded = " ".join(text.split())
    return len(folded) <= max_chars and _POINTER_RE.search(folded) is not None


def gold_id_status(gold_id: str, corpus_ids: set[str]) -> str:
    """exact, descendant (corpus has a finer id), ancestor (corpus has a coarser id), or absent."""
    if gold_id in corpus_ids:
        return "exact"
    prefix = gold_id + ":"
    if any(norm_id.startswith(prefix) for norm_id in corpus_ids):
        return "descendant"
    if any(gold_id.startswith(norm_id + ":") for norm_id in corpus_ids):
        return "ancestor"
    return "absent"
