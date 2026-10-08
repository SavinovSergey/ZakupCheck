from __future__ import annotations

import sys
from pathlib import Path

from index.recall import AggregateRecall

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_bm25_diversification_eval import (  # noqa: E402
    Bm25Result,
    discover_chunk_sizes,
    select_winner,
)


def _result(max_chunk: int, article: float, exact: float, all_article: float, latency: float) -> Bm25Result:
    metrics = AggregateRecall(1, exact, exact, article, all_article)
    return Bm25Result(max_chunk, metrics, metrics, latency, [], [])


def test_discover_chunk_sizes(tmp_path: Path) -> None:
    for name in ("chunks_structural_m700.jsonl", "chunks_structural_m300.jsonl", "other.jsonl"):
        (tmp_path / name).write_text("", encoding="utf-8")

    assert discover_chunk_sizes(tmp_path) == [300, 700]


def test_select_winner_uses_exact_for_close_article_scores() -> None:
    higher_article = _result(700, 0.600, 0.10, 0.60, 1.0)
    higher_exact = _result(1500, 0.595, 0.20, 0.50, 2.0)

    assert select_winner([higher_article, higher_exact]).max_chunk == 1500
