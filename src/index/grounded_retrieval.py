"""Frozen article-first retrieval for one complaint argument.

The selection policy matches the dev-locked m700/BM25-m1500 experiment. This
module deliberately does not load eval labels or cached per-case CE scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from index.bm25 import Bm25Index, load_chunks
from index.dense import DenseIndex, SentenceTransformerEncoder
from index.fusion import fuse_article_rankings, project_article_ranking
from index.rerank import (
    PairScorer,
    SentenceTransformerCrossEncoder,
    article_candidate_map,
    article_candidate_pool_from_map,
    build_adaptive_evidence_pack,
    rank_articles_by_best_chunk,
    source_quota_articles,
)
from schemas.corpus import Chunk

DENSE_MODEL = "Roflmax/bge-m3-russian-legal"
CROSS_ENCODER_MODEL = "qilowoq/bge-reranker-v2-m3-en-ru"
DENSE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DENSE_MAX_CHUNK = 700
BM25_MAX_CHUNK = 1500
SOURCE_ARTICLES = 50
ARTICLE_QUOTA = 10
FINAL_ARTICLES = 10
FIRST_PASS_CHUNKS = 2
FINAL_CHUNKS_PER_ARTICLE = 8
DEFAULT_EVIDENCE_MAX_CHUNKS = 20


class Searcher(Protocol):
    chunks: list[Chunk]

    def search(self, query: str, k: int) -> list[Chunk]: ...


@dataclass(frozen=True)
class RetrievalEvidence:
    article_ids: list[str]
    chunks: list[Chunk]
    scored_chunk_ids: list[str]


class FrozenArticleRetriever:
    """Use the benchmark's locked ranking and evidence-selection parameters."""

    def __init__(
        self,
        dense: Searcher,
        bm25_articles: Searcher,
        bm25_evidence: Searcher,
        scorer: PairScorer,
        *,
        chunks700: list[Chunk],
        evidence_max_chunks: int = DEFAULT_EVIDENCE_MAX_CHUNKS,
    ) -> None:
        if evidence_max_chunks <= 0:
            raise ValueError("evidence_max_chunks must be positive")
        self.dense = dense
        self.bm25_articles = bm25_articles
        self.bm25_evidence = bm25_evidence
        self.scorer = scorer
        self.chunks700 = chunks700
        self.evidence_max_chunks = evidence_max_chunks

    @classmethod
    def from_local(
        cls,
        edition_dir: Path,
        *,
        cross_encoder_path: Path,
        batch_size: int = 8,
        evidence_max_chunks: int = DEFAULT_EVIDENCE_MAX_CHUNKS,
    ) -> "FrozenArticleRetriever":
        chunks700 = load_chunks(edition_dir / "chunks_structural_m700.jsonl")
        encoder = SentenceTransformerEncoder(
            DENSE_MODEL,
            query_prefix=DENSE_QUERY_PREFIX,
            document_prefix="",
            max_tokens=512,
        )
        dense = DenseIndex.load_cached(
            chunks700,
            encoder,
            cache_dir=edition_dir / "dense",
            model_name=DENSE_MODEL,
            max_chunk=DENSE_MAX_CHUNK,
        )
        if dense is None:
            raise FileNotFoundError("compatible dense m700 document cache is missing")
        if not cross_encoder_path.is_dir():
            raise FileNotFoundError(f"local cross-encoder is missing: {cross_encoder_path}")
        scorer = SentenceTransformerCrossEncoder(
            str(cross_encoder_path), max_length=512, batch_size=batch_size
        )
        return cls(
            dense,
            Bm25Index(load_chunks(edition_dir / "chunks_structural_m1500.jsonl")),
            Bm25Index(chunks700),
            scorer,
            chunks700=chunks700,
            evidence_max_chunks=evidence_max_chunks,
        )

    def retrieve(self, query: str) -> RetrievalEvidence:
        dense_chunks = self.dense.search(query, len(self.dense.chunks))
        bm25_article_chunks = self.bm25_articles.search(
            query, len(self.bm25_articles.chunks)
        )
        bm25_evidence_chunks = self.bm25_evidence.search(
            query, len(self.bm25_evidence.chunks)
        )
        dense_articles = project_article_ranking(dense_chunks, pool=len(dense_chunks))[:SOURCE_ARTICLES]
        bm25_articles = project_article_ranking(
            bm25_article_chunks, pool=len(bm25_article_chunks)
        )[:SOURCE_ARTICLES]
        candidates = source_quota_articles(
            dense_articles, bm25_articles,
            dense_quota=ARTICLE_QUOTA, bm25_quota=ARTICLE_QUOTA,
        )
        first_map = article_candidate_map(
            [dense_chunks, bm25_evidence_chunks], candidates,
            per_article=FIRST_PASS_CHUNKS,
        )
        first_pool = article_candidate_pool_from_map(
            first_map, candidates, per_article=FIRST_PASS_CHUNKS
        )
        first_scores = self._score(query, first_pool)
        ce_articles, _ = rank_articles_by_best_chunk(first_pool, first_scores, candidates)
        shortlist = fuse_article_rankings(
            [dense_articles, bm25_articles, ce_articles],
            candidates=candidates, weights=[1, 1, 2], k=30, limit=FINAL_ARTICLES,
        )
        final_map = article_candidate_map(
            [dense_chunks, bm25_evidence_chunks], shortlist,
            per_article=FINAL_CHUNKS_PER_ARTICLE,
        )
        final_pool = article_candidate_pool_from_map(
            final_map, shortlist, per_article=FINAL_CHUNKS_PER_ARTICLE
        )
        scores = dict(first_scores)
        unscored = [chunk for chunk in final_pool if chunk.chunk_id not in scores]
        scores.update(self._score(query, unscored))
        ranked_by_article = {
            article: sorted(
                chunks,
                key=lambda chunk: (-scores[chunk.chunk_id], chunk.chunk_id),
            )
            for article, chunks in final_map.items()
        }
        token_costs = {
            chunk.chunk_id: max(1, math.ceil(len(chunk.text) / 3))
            for chunk in final_pool
        }
        evidence = build_adaptive_evidence_pack(
            shortlist,
            ranked_by_article,
            scores,
            token_costs,
            max_chunks=self.evidence_max_chunks,
            token_budget=3000,
            per_article_cap=2,
            min_relative_score=0.25,
            hierarchy_bonus=0.05,
        )
        return RetrievalEvidence(shortlist, evidence, sorted(scores))

    def _score(self, query: str, chunks: list[Chunk]) -> dict[str, float]:
        values = self.scorer.score(query, chunks) if chunks else []
        if len(values) != len(chunks):
            raise ValueError("cross-encoder returned the wrong number of scores")
        return {chunk.chunk_id: float(score) for chunk, score in zip(chunks, values, strict=True)}
