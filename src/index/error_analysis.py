"""Pure helpers for case-level retrieval error analysis."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from index.fusion import gold_id_status
from index.recall import AggregateRecall, article_key, score_case
from schemas.corpus import Chunk
from schemas.eval_case import EvalCase

RANK_BUCKETS = ("top-10", "11-20", "21-50", "51-100", ">100")


def rank_bucket(rank: int | None) -> str:
    if rank is not None and rank <= 10:
        return "top-10"
    if rank is not None and rank <= 20:
        return "11-20"
    if rank is not None and rank <= 50:
        return "21-50"
    if rank is not None and rank <= 100:
        return "51-100"
    return ">100"


def best_rank(gold_id: str, ranked: Sequence[Chunk], *, level: str) -> int | None:
    """One-based best rank of a gold id at exact or article granularity."""
    if level not in {"exact", "article"}:
        raise ValueError("level must be exact or article")
    gold_article = article_key(gold_id)
    for rank, chunk in enumerate(ranked, start=1):
        if level == "exact" and gold_id in chunk.norm_ids:
            return rank
        if level == "article" and any(article_key(norm_id) == gold_article for norm_id in chunk.norm_ids):
            return rank
    return None


def complementarity(gold_id: str, dense: Sequence[Chunk], bm25: Sequence[Chunk], *, k: int = 10) -> str:
    gold_article = article_key(gold_id)

    def found(ranked: Sequence[Chunk]) -> bool:
        return any(
            article_key(norm_id) == gold_article
            for chunk in ranked[:k]
            for norm_id in chunk.norm_ids
        )

    dense_hit, bm25_hit = found(dense), found(bm25)
    if dense_hit and bm25_hit:
        return "both"
    if dense_hit:
        return "only-dense"
    if bm25_hit:
        return "only-bm25"
    return "neither"


def repeated_article_slots(ranked: Sequence[Chunk], gold_norms: Sequence[str], *, k: int = 10) -> int:
    """Extra top-k slots occupied by repeated non-gold articles."""
    gold_articles = {article_key(norm_id) for norm_id in gold_norms}
    counts: Counter[str] = Counter()
    for chunk in ranked[:k]:
        for article in {article_key(norm_id) for norm_id in chunk.norm_ids} - gold_articles:
            counts[article] += 1
    return sum(max(0, count - 1) for count in counts.values())


def corpus_gold_statuses(cases: Sequence[EvalCase], chunks: Sequence[Chunk]) -> dict[str, str]:
    corpus_ids = {norm_id for chunk in chunks for norm_id in chunk.norm_ids}
    return {
        gold_id: gold_id_status(gold_id, corpus_ids)
        for case in cases
        for gold_id in case.gold_norms
    }


def aggregate_rankings(cases: Sequence[EvalCase], rankings: Sequence[Sequence[Chunk]], *, k: int = 10) -> AggregateRecall:
    if len(cases) != len(rankings):
        raise ValueError("cases and rankings must have the same length")
    exact: list[float] = []
    article: list[float] = []
    all_exact: list[bool] = []
    all_article: list[bool] = []
    for case, ranked in zip(cases, rankings, strict=True):
        norm_ids = {norm_id for chunk in ranked[:k] for norm_id in chunk.norm_ids}
        score = score_case(case.gold_norms, norm_ids)
        exact.append(score.exact.recall)
        article.append(score.article.recall)
        all_exact.append(score.exact.all_hit)
        all_article.append(score.article.all_hit)
    n = len(cases)
    if not n:
        return AggregateRecall(0, 0.0, 0.0, 0.0, 0.0)
    return AggregateRecall(
        n=n,
        recall_exact=sum(exact) / n,
        all_exact=sum(all_exact) / n,
        recall_article=sum(article) / n,
        all_article=sum(all_article) / n,
    )


def per_case_article_recall(cases: Sequence[EvalCase], rankings: Sequence[Sequence[Chunk]], *, k: int = 10) -> list[float]:
    values: list[float] = []
    for case, ranked in zip(cases, rankings, strict=True):
        norm_ids = {norm_id for chunk in ranked[:k] for norm_id in chunk.norm_ids}
        values.append(score_case(case.gold_norms, norm_ids).article.recall)
    return values


@dataclass(frozen=True)
class BootstrapInterval:
    mean_delta: float
    low: float
    high: float


def paired_group_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    groups: Sequence[str],
    *,
    samples: int = 10_000,
    seed: int = 20261008,
) -> BootstrapInterval:
    """Percentile CI for candidate-baseline, resampling case groups."""
    if not (len(baseline) == len(candidate) == len(groups)):
        raise ValueError("baseline, candidate, and groups must have the same length")
    if not baseline or samples <= 0:
        raise ValueError("non-empty values and positive samples are required")
    by_group: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        by_group.setdefault(group, []).append(index)
    names = sorted(by_group)
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(samples):
        chosen = [rng.choice(names) for _ in names]
        indices = [index for name in chosen for index in by_group[name]]
        deltas.append(sum(candidate[i] - baseline[i] for i in indices) / len(indices))
    deltas.sort()
    low_index = int(0.025 * (samples - 1))
    high_index = int(0.975 * (samples - 1))
    mean_delta = sum(c - b for b, c in zip(baseline, candidate, strict=True)) / len(baseline)
    return BootstrapInterval(mean_delta, deltas[low_index], deltas[high_index])
