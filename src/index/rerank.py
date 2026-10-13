"""Candidate construction and cross-encoder reranking helpers."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from index.recall import article_key
from schemas.corpus import Chunk

_TOKEN_RE = re.compile(r"[\wё]+", re.IGNORECASE)


class PairScorer(Protocol):
    def score(self, query: str, chunks: Sequence[Chunk]) -> list[float]: ...


def article_candidate_pool(
    rankings: Sequence[Sequence[Chunk]],
    articles: Sequence[str],
    *,
    per_article: int,
) -> list[Chunk]:
    """Interleave source rankings and keep up to N candidates per article."""
    by_article = article_candidate_map(rankings, articles, per_article=per_article)
    return article_candidate_pool_from_map(by_article, articles, per_article=per_article)


def article_candidate_map(
    rankings: Sequence[Sequence[Chunk]],
    articles: Sequence[str],
    *,
    per_article: int,
) -> dict[str, list[Chunk]]:
    """Interleave sources once and retain an ordered candidate list per article."""
    if per_article < 0:
        raise ValueError("per_article must be non-negative")
    selected = list(dict.fromkeys(articles))
    selected_set = set(selected)
    by_article: dict[str, list[Chunk]] = {article: [] for article in selected}
    seen_by_article: dict[str, set[str]] = {article: set() for article in selected}
    max_len = max((len(ranking) for ranking in rankings), default=0)
    for rank in range(max_len):
        for ranking in rankings:
            if rank >= len(ranking):
                continue
            chunk = ranking[rank]
            chunk_articles = {article_key(norm_id) for norm_id in chunk.norm_ids} & selected_set
            for article in sorted(chunk_articles):
                if len(by_article[article]) >= per_article:
                    continue
                if chunk.chunk_id in seen_by_article[article]:
                    continue
                by_article[article].append(chunk)
                seen_by_article[article].add(chunk.chunk_id)
        if all(len(chunks) >= per_article for chunks in by_article.values()):
            break

    return by_article


def article_candidate_pool_from_map(
    candidates: dict[str, list[Chunk]],
    articles: Sequence[str],
    *,
    per_article: int,
) -> list[Chunk]:
    """Materialize a deduplicated pool from precomputed per-article candidates."""
    if per_article < 0:
        raise ValueError("per_article must be non-negative")

    pool: list[Chunk] = []
    seen_chunks: set[str] = set()
    for article in dict.fromkeys(articles):
        for chunk in candidates.get(article, [])[:per_article]:
            if chunk.chunk_id in seen_chunks:
                continue
            pool.append(chunk)
            seen_chunks.add(chunk.chunk_id)
    return pool


def rerank_chunks(
    query: str,
    chunks: Sequence[Chunk],
    scorer: PairScorer,
) -> list[Chunk]:
    """Sort chunks by cross-encoder score with deterministic chunk-id ties."""
    scores = scorer.score(query, chunks)
    if len(scores) != len(chunks):
        raise ValueError(f"expected {len(chunks)} scores, got {len(scores)}")
    order = sorted(range(len(chunks)), key=lambda index: (-scores[index], chunks[index].chunk_id))
    return [chunks[index] for index in order]


def source_quota_articles(
    dense_articles: Sequence[str],
    bm25_articles: Sequence[str],
    *,
    dense_quota: int,
    bm25_quota: int,
) -> list[str]:
    """Union two article prefixes with deterministic source-order ties."""
    if dense_quota < 0 or bm25_quota < 0:
        raise ValueError("source quotas must be non-negative")
    return list(
        dict.fromkeys(
            list(dense_articles[:dense_quota]) + list(bm25_articles[:bm25_quota])
        )
    )


def best_chunk_per_article(
    chunks: Sequence[Chunk],
    scores: dict[str, float],
    articles: Sequence[str],
    *,
    limit: int = 10,
) -> list[Chunk]:
    """Rank articles by their best scored chunk and return that evidence.

    A chunk shared by multiple articles may represent each of them while scoring,
    but is emitted only once. Score ties use article id and then chunk id.
    """
    if limit < 0:
        raise ValueError("limit must be non-negative")
    ranked_articles, best = rank_articles_by_best_chunk(chunks, scores, articles)
    output: list[Chunk] = []
    seen: set[str] = set()
    for article in ranked_articles:
        chunk = best[article]
        if chunk.chunk_id in seen:
            continue
        output.append(chunk)
        seen.add(chunk.chunk_id)
        if len(output) == limit:
            break
    return output


def rank_articles_by_best_chunk(
    chunks: Sequence[Chunk],
    scores: dict[str, float],
    articles: Sequence[str],
) -> tuple[list[str], dict[str, Chunk]]:
    """Rank articles globally by their best cross-encoder chunk.

    The returned mapping makes the winning evidence explicit.  Unlike
    :func:`best_chunk_per_article`, the article ranking is not shortened when
    one multi-article chunk represents more than one article.
    """
    selected = set(articles)
    best: dict[str, tuple[float, Chunk]] = {}
    for chunk in chunks:
        if chunk.chunk_id not in scores:
            raise ValueError(f"missing score for {chunk.chunk_id}")
        score = scores[chunk.chunk_id]
        for article in {article_key(norm_id) for norm_id in chunk.norm_ids} & selected:
            previous = best.get(article)
            if previous is None or (-score, chunk.chunk_id) < (-previous[0], previous[1].chunk_id):
                best[article] = (score, chunk)
    ranked = sorted(best, key=lambda article: (-best[article][0], article, best[article][1].chunk_id))
    return ranked, {article: best[article][1] for article in ranked}


def best_chunks_in_article_order(
    chunks: Sequence[Chunk],
    scores: dict[str, float],
    articles: Sequence[str],
    *,
    limit: int = 10,
) -> list[Chunk]:
    """Choose CE evidence per article without changing the supplied article order."""
    if limit < 0:
        raise ValueError("limit must be non-negative")
    _, best = rank_articles_by_best_chunk(chunks, scores, articles)
    output: list[Chunk] = []
    seen: set[str] = set()
    for article in articles:
        chunk = best.get(article)
        if chunk is None or chunk.chunk_id in seen:
            continue
        output.append(chunk)
        seen.add(chunk.chunk_id)
        if len(output) == limit:
            break
    return output


def two_pass_article_rerank(
    rankings: Sequence[Sequence[Chunk]],
    articles: Sequence[str],
    scores: dict[str, float],
    *,
    first_pass_per_article: int,
    shortlist_articles: int,
    final_per_article: int,
    limit: int = 10,
) -> tuple[list[str], list[Chunk], list[Chunk], list[Chunk]]:
    """Apply a broad cheap pass, then spend more chunk budget on a shortlist.

    Returns ``(shortlist, final_pool, output, scored_union)``.  ``scored_union``
    contains every unique query/chunk pair that an online implementation would
    have to score across both passes.
    """
    if first_pass_per_article <= 0 or final_per_article <= 0:
        raise ValueError("per-article limits must be positive")
    if shortlist_articles < limit:
        raise ValueError("shortlist_articles must be at least limit")
    first_pool = article_candidate_pool(
        rankings,
        articles,
        per_article=first_pass_per_article,
    )
    ranked_articles, _ = rank_articles_by_best_chunk(first_pool, scores, articles)
    shortlist = ranked_articles[:shortlist_articles]
    final_pool = article_candidate_pool(
        rankings,
        shortlist,
        per_article=final_per_article,
    )
    output = best_chunk_per_article(final_pool, scores, shortlist, limit=limit)
    scored = {chunk.chunk_id: chunk for chunk in first_pool}
    scored.update({chunk.chunk_id: chunk for chunk in final_pool})
    scored_union = [scored[chunk_id] for chunk_id in sorted(scored)]
    return shortlist, final_pool, output, scored_union


def diverse_chunk_selection(
    chunks: Sequence[Chunk],
    relevance_scores: dict[str, float],
    *,
    limit: int,
    diversity_weight: float,
) -> list[Chunk]:
    """Greedily balance retrieval relevance and non-redundant legal evidence."""
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if not 0.0 <= diversity_weight <= 1.0:
        raise ValueError("diversity_weight must be between zero and one")
    unique = {chunk.chunk_id: chunk for chunk in chunks}
    missing = [chunk_id for chunk_id in unique if chunk_id not in relevance_scores]
    if missing:
        raise ValueError(f"missing relevance score for {missing[0]}")
    values = [relevance_scores[chunk_id] for chunk_id in unique]
    low = min(values, default=0.0)
    high = max(values, default=0.0)

    def normalized(chunk_id: str) -> float:
        if high == low:
            return 1.0
        return (relevance_scores[chunk_id] - low) / (high - low)

    token_sets = {
        chunk_id: set(_TOKEN_RE.findall(chunk.text.lower())) for chunk_id, chunk in unique.items()
    }
    norm_sets = {chunk_id: set(chunk.norm_ids) for chunk_id, chunk in unique.items()}

    def similarity(left: str, right: str) -> float:
        token_union = token_sets[left] | token_sets[right]
        token_overlap = (
            len(token_sets[left] & token_sets[right]) / len(token_union)
            if token_union
            else 0.0
        )
        norm_union = norm_sets[left] | norm_sets[right]
        norm_overlap = (
            len(norm_sets[left] & norm_sets[right]) / len(norm_union)
            if norm_union
            else 0.0
        )
        return max(token_overlap, norm_overlap)

    selected: list[str] = []
    remaining = set(unique)
    while remaining and len(selected) < limit:
        def key(chunk_id: str) -> tuple[float, str]:
            redundancy = max((similarity(chunk_id, other) for other in selected), default=0.0)
            utility = (
                (1.0 - diversity_weight) * normalized(chunk_id)
                + diversity_weight * (1.0 - redundancy)
            )
            return (-utility, chunk_id)

        chosen = min(remaining, key=key)
        selected.append(chosen)
        remaining.remove(chosen)
    return [unique[chunk_id] for chunk_id in selected]


def build_evidence_pack(
    article_order: Sequence[str],
    ranked_by_article: dict[str, Sequence[Chunk]],
    chunk_scores: dict[str, float],
    *,
    budget: int,
    per_article_cap: int,
    primary_articles: int,
) -> list[Chunk]:
    """Guarantee primary article evidence, then fill globally with secondary chunks."""
    if min(budget, per_article_cap, primary_articles) < 0:
        raise ValueError("evidence limits must be non-negative")
    if primary_articles > budget:
        raise ValueError("primary_articles cannot exceed budget")
    ordered_articles = list(dict.fromkeys(article_order))
    output: list[Chunk] = []
    seen: set[str] = set()
    counts: dict[str, int] = {}
    article_rank = {article: rank for rank, article in enumerate(ordered_articles)}

    def add(article: str, chunk: Chunk) -> None:
        if chunk.chunk_id in seen or counts.get(article, 0) >= per_article_cap:
            return
        output.append(chunk)
        seen.add(chunk.chunk_id)
        counts[article] = counts.get(article, 0) + 1

    for article in ordered_articles[:primary_articles]:
        candidates = ranked_by_article.get(article, ())
        if candidates:
            add(article, candidates[0])

    extras: list[tuple[str, int, Chunk]] = []
    for article in ordered_articles:
        for within_rank, chunk in enumerate(ranked_by_article.get(article, ()), start=1):
            if chunk.chunk_id not in chunk_scores:
                raise ValueError(f"missing evidence score for {chunk.chunk_id}")
            extras.append((article, within_rank, chunk))
    extras.sort(
        key=lambda item: (
            -chunk_scores[item[2].chunk_id],
            article_rank[item[0]],
            item[1],
            item[2].chunk_id,
        )
    )
    for article, _, chunk in extras:
        if len(output) == budget:
            break
        add(article, chunk)
    return output


def article_neighbors(
    chunks: Sequence[Chunk],
    anchor: Chunk,
    *,
    radius: int = 1,
) -> list[Chunk]:
    """Return structural neighbours of ``anchor`` inside its article.

    Corpus order is reconstructed from character offsets, so the helper is
    independent of retrieval rank.  The closest previous/next chunks are
    emitted first with deterministic ties.
    """
    if radius < 0:
        raise ValueError("radius must be non-negative")
    articles = {article_key(norm_id) for norm_id in anchor.norm_ids}
    ordered = sorted(
        (
            chunk
            for chunk in chunks
            if {article_key(norm_id) for norm_id in chunk.norm_ids} & articles
        ),
        key=lambda chunk: (chunk.char_start, chunk.char_end, chunk.chunk_id),
    )
    positions = [index for index, chunk in enumerate(ordered) if chunk.chunk_id == anchor.chunk_id]
    if not positions:
        raise ValueError(f"anchor {anchor.chunk_id} is not present in chunks")
    position = positions[0]
    output: list[Chunk] = []
    for distance in range(1, radius + 1):
        for index in (position - distance, position + distance):
            if 0 <= index < len(ordered):
                output.append(ordered[index])
    return output


def chunk_structure_keys(chunk: Chunk) -> set[tuple[str, str | None, str | None]]:
    """Return article/part/point paths represented by a chunk."""
    keys: set[tuple[str, str | None, str | None]] = set()
    for norm_id in chunk.norm_ids:
        article = article_key(norm_id)
        suffix = norm_id.split(":art.", 1)[-1].split(":", 1)
        tail = suffix[1] if len(suffix) == 2 else ""
        pieces = tail.split(":") if tail else []
        part = next((piece[3:] for piece in pieces if piece.startswith("ch.")), None)
        point = next((piece[2:] for piece in pieces if piece.startswith("p.")), None)
        keys.add((article, part, point))
    return keys


def build_adaptive_evidence_pack(
    article_order: Sequence[str],
    ranked_by_article: dict[str, Sequence[Chunk]],
    chunk_scores: dict[str, float],
    token_costs: dict[str, int],
    *,
    max_chunks: int,
    token_budget: int,
    per_article_cap: int = 2,
    min_relative_score: float = 0.0,
    hierarchy_bonus: float = 0.0,
) -> list[Chunk]:
    """Build a token-bounded evidence pack with optional structural novelty.

    One primary chunk is guaranteed for every available article.  Extra chunks
    must clear a score ratio relative to that article's primary; among eligible
    extras, a bonus can favour a previously unseen part/point path.
    """
    if min(max_chunks, token_budget, per_article_cap) < 0:
        raise ValueError("evidence limits must be non-negative")
    if not 0.0 <= min_relative_score <= 1.0:
        raise ValueError("min_relative_score must be between zero and one")
    if hierarchy_bonus < 0.0:
        raise ValueError("hierarchy_bonus must be non-negative")
    ordered_articles = list(dict.fromkeys(article_order))
    output: list[Chunk] = []
    seen: set[str] = set()
    counts: dict[str, int] = {}
    structures: dict[str, set[tuple[str, str | None, str | None]]] = {}
    used_tokens = 0

    def add(article: str, chunk: Chunk, *, force: bool = False) -> bool:
        nonlocal used_tokens
        if chunk.chunk_id in seen or counts.get(article, 0) >= per_article_cap:
            return False
        if chunk.chunk_id not in token_costs:
            raise ValueError(f"missing token cost for {chunk.chunk_id}")
        cost = token_costs[chunk.chunk_id]
        if not force and output and used_tokens + cost > token_budget:
            return False
        output.append(chunk)
        seen.add(chunk.chunk_id)
        counts[article] = counts.get(article, 0) + 1
        structures.setdefault(article, set()).update(chunk_structure_keys(chunk))
        used_tokens += cost
        return True

    primary_scores: dict[str, float] = {}
    for article in ordered_articles:
        candidates = ranked_by_article.get(article, ())
        if not candidates or len(output) >= max_chunks:
            continue
        primary = candidates[0]
        if primary.chunk_id not in chunk_scores:
            raise ValueError(f"missing evidence score for {primary.chunk_id}")
        primary_scores[article] = chunk_scores[primary.chunk_id]
        add(article, primary, force=True)

    article_rank = {article: rank for rank, article in enumerate(ordered_articles)}
    while len(output) < max_chunks:
        eligible: list[tuple[float, int, int, str, str, Chunk]] = []
        for article in ordered_articles:
            if counts.get(article, 0) >= per_article_cap or article not in primary_scores:
                continue
            candidates = ranked_by_article.get(article, ())
            for within_rank, chunk in enumerate(candidates[1:], start=2):
                if chunk.chunk_id in seen:
                    continue
                if chunk.chunk_id not in chunk_scores or chunk.chunk_id not in token_costs:
                    raise ValueError(f"missing evidence metadata for {chunk.chunk_id}")
                primary_score = max(primary_scores[article], 1e-9)
                if chunk_scores[chunk.chunk_id] < primary_score * min_relative_score:
                    continue
                if used_tokens + token_costs[chunk.chunk_id] > token_budget:
                    continue
                novelty = not chunk_structure_keys(chunk).issubset(structures.get(article, set()))
                utility = chunk_scores[chunk.chunk_id] + hierarchy_bonus * float(novelty)
                eligible.append(
                    (-utility, article_rank[article], within_rank, chunk.chunk_id, article, chunk)
                )
        if not eligible:
            break
        _, _, _, _, article, chunk = min(eligible)
        add(article, chunk)
    return output


class SentenceTransformerCrossEncoder:
    """CPU cross-encoder wrapper loaded lazily from sentence-transformers."""

    def __init__(
        self,
        model_name: str,
        *,
        max_length: int = 512,
        batch_size: int = 8,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, max_length=max_length, device="cpu")

    def score(self, query: str, chunks: Sequence[Chunk]) -> list[float]:
        if not chunks:
            return []
        pairs = [(query, chunk.text) for chunk in chunks]
        values = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(values, dtype=np.float32).reshape(-1).tolist()
