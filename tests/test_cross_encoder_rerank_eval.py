from __future__ import annotations

import sys
from pathlib import Path

from index.recall import AggregateRecall

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_cross_encoder_rerank_eval import (  # noqa: E402
    CrossEncoderResult,
    CrossEncoderSpec,
    _load_score_cache,
    legacy_score_id_map,
    select_winner,
)


def _result(article: float, exact: float, latency: float) -> CrossEncoderResult:
    metrics = AggregateRecall(1, exact, exact, article, article)
    return CrossEncoderResult(CrossEncoderSpec(4, 10), metrics, metrics, latency, [])


def test_select_winner_uses_exact_inside_article_tolerance() -> None:
    higher_article = _result(0.600, 0.10, 1.0)
    higher_exact = _result(0.595, 0.20, 2.0)
    assert select_winner([higher_article, higher_exact]) == higher_exact


def test_select_winner_prefers_article_outside_tolerance() -> None:
    higher_article = _result(0.610, 0.10, 2.0)
    higher_exact = _result(0.590, 0.20, 1.0)
    assert select_winner([higher_article, higher_exact]) == higher_article


def test_score_cache_is_reset_when_corpus_fingerprint_changes(tmp_path: Path) -> None:
    path = tmp_path / "scores.json"
    path.write_text(
        '{"version":1,"model":"BAAI/bge-reranker-v2-m3",'
        '"corpus_fingerprint":"old","splits":{"dev":{"case":{"chunk":1.0}}},'
        '"latency_seconds":{"dev":1.0}}',
        encoding="utf-8",
    )
    cache = _load_score_cache(path, "new")
    assert cache["corpus_fingerprint"] == "new"
    assert cache["splits"] == {}
    assert cache["latency_seconds"] == {}


def test_pre_fingerprint_score_cache_is_migrated(tmp_path: Path) -> None:
    path = tmp_path / "scores.json"
    path.write_text(
        '{"version":1,"model":"BAAI/bge-reranker-v2-m3",'
        '"splits":{"dev":{"case":{"old-1":1.0,"removed":2.0}}},'
        '"latency_seconds":{"dev":3.0}}',
        encoding="utf-8",
    )
    cache = _load_score_cache(path, "new", id_migration={"old-1": "new-2"})
    assert cache["corpus_fingerprint"] == "new"
    assert cache["splits"]["dev"]["case"] == {"new-2": 1.0}
    assert cache["latency_seconds"] == {"dev": 3.0}
