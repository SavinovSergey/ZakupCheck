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


def project_article_ranking(ranked: Sequence[Chunk], *, pool: int = 50) -> list[str]:
    """Collapse a chunk ranking into first-seen articles.

    The returned ranks are dense: repeated chunks from one article consume only
    one position.  Articles inside a multi-article chunk are ordered by their
    canonical id so the projection is deterministic.
    """
    if pool < 0:
        raise ValueError("pool must be non-negative")
    articles: list[str] = []
    seen: set[str] = set()
    for chunk in ranked[:pool]:
        for article in sorted({article_key(norm_id) for norm_id in chunk.norm_ids}):
            if article in seen:
                continue
            seen.add(article)
            articles.append(article)
    return articles


def weighted_article_rrf(
    rankings: Sequence[Sequence[Chunk]],
    *,
    weights: Sequence[float],
    k: int = RRF_K,
    pool: int = 50,
    limit: int = 10,
) -> list[str]:
    """Weighted RRF after projecting incompatible chunkings to articles.

    Each source contributes at most once per article, so a corpus with smaller
    chunks cannot dominate merely by returning more chunks from the same
    article.  Ties break by the best source rank and then canonical article id.
    """
    if len(rankings) != len(weights):
        raise ValueError("rankings and weights must have the same length")
    if k < 0 or pool < 0 or limit < 0:
        raise ValueError("k, pool, and limit must be non-negative")
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative")
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, article in enumerate(project_article_ranking(ranking, pool=pool), start=1):
            scores[article] = scores.get(article, 0.0) + weight / (k + rank)
            best_rank[article] = min(best_rank.get(article, rank), rank)
    return sorted(scores, key=lambda article: (-scores[article], best_rank[article], article))[:limit]


def fuse_article_rankings(
    rankings: Sequence[Sequence[str]],
    *,
    candidates: Sequence[str],
    weights: Sequence[float],
    k: int = RRF_K,
    limit: int = 10,
) -> list[str]:
    """Weighted RRF for already projected article rankings.

    Source positions are preserved even when only a subset of articles is
    eligible.  This matters when a candidate union contains different prefix
    lengths from dense and BM25.
    """
    if len(rankings) != len(weights):
        raise ValueError("rankings and weights must have the same length")
    if k < 0 or limit < 0:
        raise ValueError("k and limit must be non-negative")
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative")
    eligible = set(candidates)
    scores = {article: 0.0 for article in dict.fromkeys(candidates)}
    best_rank = {article: len(eligible) + 1 for article in eligible}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, article in enumerate(ranking, start=1):
            if article not in eligible:
                continue
            scores[article] += weight / (k + rank)
            best_rank[article] = min(best_rank[article], rank)
    return sorted(scores, key=lambda article: (-scores[article], best_rank[article], article))[
        :limit
    ]


def protected_article_shortlist(
    dense_articles: Sequence[str],
    bm25_articles: Sequence[str],
    fallback_articles: Sequence[str],
    *,
    candidates: Sequence[str],
    dense_keep: int,
    bm25_keep: int,
    limit: int,
) -> list[str]:
    """Protect source prefixes, then fill the shortlist from another ranking."""
    if min(dense_keep, bm25_keep, limit) < 0:
        raise ValueError("limits must be non-negative")
    if dense_keep + bm25_keep > limit:
        raise ValueError("protected quotas must fit inside limit")
    eligible = set(candidates)
    output: list[str] = []
    seen: set[str] = set()

    def append_from(ranking: Sequence[str], count: int | None) -> None:
        if count == 0:
            return
        added = 0
        for article in ranking:
            if article not in eligible or article in seen:
                continue
            output.append(article)
            seen.add(article)
            added += 1
            if len(output) == limit or (count is not None and added == count):
                return

    append_from(dense_articles, dense_keep)
    append_from(bm25_articles, bm25_keep)
    append_from(fallback_articles, None)
    return output


def rank_articles_with_retrieval_bonus(
    ce_scores: dict[str, float],
    dense_articles: Sequence[str],
    bm25_articles: Sequence[str],
    *,
    candidates: Sequence[str],
    alpha: float,
    k: int = 10,
    limit: int = 10,
) -> list[str]:
    """Add a normalized dense/BM25 reciprocal-rank prior to raw CE scores."""
    if alpha < 0 or k < 0 or limit < 0:
        raise ValueError("alpha, k, and limit must be non-negative")
    eligible = set(candidates)
    prior = {article: 0.0 for article in dict.fromkeys(candidates)}
    best_rank = {article: len(eligible) + 1 for article in eligible}
    for ranking in (dense_articles, bm25_articles):
        for rank, article in enumerate(ranking, start=1):
            if article not in eligible:
                continue
            prior[article] += 1.0 / (k + rank)
            best_rank[article] = min(best_rank[article], rank)
    scale = max(prior.values(), default=0.0)
    if scale:
        prior = {article: value / scale for article, value in prior.items()}
    combined = {
        article: ce_scores.get(article, float("-inf")) + alpha * prior[article]
        for article in prior
    }
    return sorted(
        combined,
        key=lambda article: (-combined[article], best_rank[article], article),
    )[:limit]


def dense_quota_article_fusion(
    dense: Sequence[Chunk],
    bm25: Sequence[Chunk],
    *,
    dense_keep: int,
    weights: Sequence[float] = (1.0, 1.0),
    k: int = RRF_K,
    pool: int = 50,
    limit: int = 10,
) -> list[str]:
    """Preserve a dense article prefix, then fill remaining slots by RRF.

    This is a conservative alternative to replacing the full dense ranking:
    BM25 can add or promote candidates only outside the protected prefix.
    """
    if dense_keep < 0 or dense_keep > limit:
        raise ValueError("dense_keep must be between zero and limit")
    dense_articles = project_article_ranking(dense, pool=pool)
    fused = weighted_article_rrf(
        [dense, bm25],
        weights=weights,
        k=k,
        pool=pool,
        limit=pool * 2,
    )
    picked = dense_articles[:dense_keep]
    seen = set(picked)
    for article in fused:
        if article in seen:
            continue
        picked.append(article)
        seen.add(article)
        if len(picked) == limit:
            break
    return picked


def select_article_evidence(
    evidence_ranking: Sequence[Chunk],
    article_ranking: Sequence[str],
    *,
    n_articles: int,
    limit: int = 10,
) -> list[Chunk]:
    """Return evidence chunks for fused articles from one canonical corpus.

    The best chunk of every selected article is emitted first in fused article
    order.  Remaining slots are filled in the evidence ranking's original
    order.  This preserves article coverage while allowing extra parts or
    points when fewer than ``limit`` articles are selected.
    """
    if n_articles < 0 or limit < 0:
        raise ValueError("n_articles and limit must be non-negative")
    selected = list(dict.fromkeys(article_ranking[:n_articles]))
    selected_set = set(selected)
    best_by_article: dict[str, Chunk] = {}
    for chunk in evidence_ranking:
        for article in {article_key(norm_id) for norm_id in chunk.norm_ids} & selected_set:
            best_by_article.setdefault(article, chunk)

    picked: list[Chunk] = []
    seen_chunks: set[str] = set()
    for article in selected:
        chunk = best_by_article.get(article)
        if chunk is None or chunk.chunk_id in seen_chunks:
            continue
        picked.append(chunk)
        seen_chunks.add(chunk.chunk_id)
        if len(picked) == limit:
            return picked

    for chunk in evidence_ranking:
        if chunk.chunk_id in seen_chunks:
            continue
        articles = {article_key(norm_id) for norm_id in chunk.norm_ids}
        if not articles & selected_set:
            continue
        picked.append(chunk)
        seen_chunks.add(chunk.chunk_id)
        if len(picked) == limit:
            break
    return picked


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
