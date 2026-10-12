from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from index.recall import AggregateRecall

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_hybrid_retrieval_eval import (  # noqa: E402
    HybridResult,
    HybridSpec,
    _metrics,
    select_winner,
)


def _result(
    *,
    bm25_max_chunk: int,
    article: float,
    exact: float,
    all_article: float,
    latency: float,
) -> HybridResult:
    spec = HybridSpec(bm25_max_chunk, 30, 1.0, 1.0, 10)
    metrics = AggregateRecall(1, exact, exact, article, all_article)
    return HybridResult(spec, metrics, metrics, latency, [])


def test_select_winner_uses_exact_inside_article_tolerance() -> None:
    higher_article = _result(
        bm25_max_chunk=2500,
        article=0.600,
        exact=0.10,
        all_article=0.60,
        latency=1.0,
    )
    better_exact = _result(
        bm25_max_chunk=1500,
        article=0.595,
        exact=0.20,
        all_article=0.50,
        latency=2.0,
    )

    assert select_winner([higher_article, better_exact]).spec.bm25_max_chunk == 1500


def test_select_winner_prefers_article_outside_tolerance() -> None:
    higher_article = _result(
        bm25_max_chunk=2500,
        article=0.610,
        exact=0.10,
        all_article=0.50,
        latency=2.0,
    )
    better_exact = _result(
        bm25_max_chunk=1500,
        article=0.590,
        exact=0.20,
        all_article=0.60,
        latency=1.0,
    )

    assert select_winner([higher_article, better_exact]).spec.bm25_max_chunk == 2500


def test_metrics_respects_cutoffs_above_ten() -> None:
    case = SimpleNamespace(gold_norms=["44FZ:art.99"])
    ranking = [
        SimpleNamespace(norm_ids=[f"44FZ:art.{index}"])
        for index in range(1, 11)
    ] + [SimpleNamespace(norm_ids=["44FZ:art.99"])]

    assert _metrics([case], [ranking], 10).recall_article == 0.0
    assert _metrics([case], [ranking], 11).recall_article == 1.0
