from __future__ import annotations

import json
import sys
from pathlib import Path

from index.recall import AggregateRecall

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_dense_diversification_eval import (  # noqa: E402
    CandidateResult,
    DenseConfig,
    discover_cached_configs,
    select_winner,
)


def _candidate(model: str, article: float, exact: float, all_article: float, latency: float) -> CandidateResult:
    baseline = AggregateRecall(1, 0.0, 0.0, 0.0, 0.0)
    diversified = AggregateRecall(1, exact, exact, article, all_article)
    return CandidateResult(
        DenseConfig(model, 700, 0, 1),
        baseline,
        diversified,
        latency,
        [],
        [],
    )


def test_discover_cached_configs_requires_matching_vector_and_chunks(tmp_path: Path) -> None:
    edition = tmp_path / "edition"
    dense = edition / "dense"
    dense.mkdir(parents=True)
    (edition / "chunks_structural_m700.jsonl").write_text("", encoding="utf-8")
    meta = dense / "owner__model_m700.meta.json"
    meta.write_text(
        json.dumps({"model": "owner/model", "truncated": 2, "rows": 10}),
        encoding="utf-8",
    )
    (dense / "owner__model_m700.npy").write_bytes(b"vectors")

    assert discover_cached_configs(edition) == [DenseConfig("owner/model", 700, 2, 10)]


def test_select_winner_uses_exact_inside_article_tolerance() -> None:
    higher_article = _candidate("higher-article", 0.600, 0.10, 0.60, 1.0)
    better_exact = _candidate("better-exact", 0.595, 0.20, 0.50, 2.0)

    assert select_winner([higher_article, better_exact]).config.model == "better-exact"


def test_select_winner_prefers_article_outside_tolerance() -> None:
    higher_article = _candidate("higher-article", 0.610, 0.10, 0.50, 2.0)
    better_exact = _candidate("better-exact", 0.590, 0.20, 0.60, 1.0)

    assert select_winner([higher_article, better_exact]).config.model == "higher-article"
