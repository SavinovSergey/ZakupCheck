"""Recall of gold norm ids against the norm_ids of retrieved chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass

_ARTICLE_RE = re.compile(r":(art\.\d+(?:\.\d+)?)(?:$|:)")


def article_key(norm_id: str) -> str:
    """`44FZ:…:art.33:ch.1:p.1` → `art.33`. No expansion of an article into its points."""
    match = _ARTICLE_RE.search(norm_id)
    if match is None:
        raise ValueError(f"norm_id has no article: {norm_id}")
    return match.group(1)


@dataclass(frozen=True)
class LevelScore:
    """One case at one granularity. recall is |found| / |gold|; all_hit means every gold item was found."""

    recall: float
    all_hit: bool


@dataclass(frozen=True)
class CaseScore:
    exact: LevelScore
    article: LevelScore


def score_case(gold_norms: list[str], retrieved_norm_ids: set[str]) -> CaseScore:
    if not gold_norms:
        raise ValueError("gold_norms must not be empty")
    found_exact = [norm_id for norm_id in gold_norms if norm_id in retrieved_norm_ids]
    gold_articles = {article_key(norm_id) for norm_id in gold_norms}
    retrieved_articles = {article_key(norm_id) for norm_id in retrieved_norm_ids}
    found_articles = gold_articles & retrieved_articles
    return CaseScore(
        exact=LevelScore(recall=len(found_exact) / len(gold_norms), all_hit=len(found_exact) == len(gold_norms)),
        article=LevelScore(
            recall=len(found_articles) / len(gold_articles),
            all_hit=found_articles == gold_articles,
        ),
    )


@dataclass(frozen=True)
class AggregateRecall:
    n: int
    recall_exact: float
    all_exact: float
    recall_article: float
    all_article: float


def aggregate(scores: list[CaseScore]) -> AggregateRecall:
    n = len(scores)
    if n == 0:
        return AggregateRecall(0, 0.0, 0.0, 0.0, 0.0)
    return AggregateRecall(
        n=n,
        recall_exact=sum(item.exact.recall for item in scores) / n,
        all_exact=sum(item.exact.all_hit for item in scores) / n,
        recall_article=sum(item.article.recall for item in scores) / n,
        all_article=sum(item.article.all_hit for item in scores) / n,
    )
