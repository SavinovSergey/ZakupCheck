"""Candidate construction and cross-encoder reranking helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from index.recall import article_key
from schemas.corpus import Chunk


class PairScorer(Protocol):
    def score(self, query: str, chunks: Sequence[Chunk]) -> list[float]: ...


def article_candidate_pool(
    rankings: Sequence[Sequence[Chunk]],
    articles: Sequence[str],
    *,
    per_article: int,
) -> list[Chunk]:
    """Interleave source rankings and keep up to N candidates per article."""
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

    pool: list[Chunk] = []
    seen_chunks: set[str] = set()
    for article in selected:
        for chunk in by_article[article]:
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
    output: list[Chunk] = []
    seen: set[str] = set()
    for article in ranked:
        chunk = best[article][1]
        if chunk.chunk_id in seen:
            continue
        output.append(chunk)
        seen.add(chunk.chunk_id)
        if len(output) == limit:
            break
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
