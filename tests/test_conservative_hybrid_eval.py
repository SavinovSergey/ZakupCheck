from __future__ import annotations

import sys
from pathlib import Path

from index.recall import AggregateRecall

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_conservative_hybrid_eval import (  # noqa: E402
    ConservativeResult,
    ConservativeSpec,
    select_winner,
)


def _result(article: float, exact: float, all_article: float, latency: float) -> ConservativeResult:
    spec = ConservativeSpec(1500, 8, 1.0, 1.0, 10, 1.0, 1.0)
    metrics = AggregateRecall(1, exact, exact, article, all_article)
    return ConservativeResult(spec, metrics, metrics, latency, [])


def test_select_winner_uses_exact_inside_article_tolerance() -> None:
    higher_article = _result(0.600, 0.10, 0.60, 1.0)
    better_exact = _result(0.595, 0.20, 0.50, 2.0)
    assert select_winner([higher_article, better_exact]) == better_exact


def test_select_winner_prefers_article_outside_tolerance() -> None:
    higher_article = _result(0.610, 0.10, 0.50, 2.0)
    better_exact = _result(0.590, 0.20, 0.60, 1.0)
    assert select_winner([higher_article, better_exact]) == higher_article


def test_select_winner_ignores_sub_millisecond_noise_before_stable_spec_order() -> None:
    metrics = AggregateRecall(1, 0.2, 0.2, 0.6, 0.6)
    keep_eight = ConservativeResult(
        ConservativeSpec(1500, 8, 1.0, 1.0, 10, 1.0, 2.0),
        metrics,
        metrics,
        294.11,
        [],
    )
    keep_seven = ConservativeResult(
        ConservativeSpec(1500, 7, 1.0, 1.0, 10, 1.0, 2.0),
        metrics,
        metrics,
        294.41,
        [],
    )
    assert select_winner([keep_eight, keep_seven]) == keep_seven
