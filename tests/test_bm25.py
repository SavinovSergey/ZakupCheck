"""BM25 search and recall on a handful of fake chunks."""

from __future__ import annotations

from index.bm25 import Bm25Index
from index.recall import score_case
from schemas.corpus import Chunk


def _chunk(chunk_id: str, text: str, norm_ids: list[str]) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        edition_id="2026-08-04",
        chunking="point",
        text=text,
        char_start=0,
        char_end=len(text),
        norm_ids=norm_ids,
        max_chunk=2000,
        article_title_prefix=False,
    )


def test_search_returns_chunk_with_matching_norm() -> None:
    index = Bm25Index(
        [
            _chunk("a", "товарный знак или эквивалент в описании объекта", ["44FZ:2026-08-04:art.33:ch.1:p.1"]),
            _chunk("b", "требования к участникам закупки лицензия опыт", ["44FZ:2026-08-04:art.31:ch.1"]),
            _chunk("c", "условия контракта срок оплаты", ["44FZ:2026-08-04:art.34:ch.1"]),
        ]
    )
    hits = index.search("товарный знак эквивалент", k=1)
    assert hits[0].norm_ids == ["44FZ:2026-08-04:art.33:ch.1:p.1"]


def test_article_hit_is_not_an_exact_hit() -> None:
    score = score_case(
        ["44FZ:2026-08-04:art.33:ch.1:p.1"],
        {"44FZ:2026-08-04:art.33"},
    )
    assert score.exact.recall == 0.0
    assert score.exact.all_hit is False
    assert score.article.recall == 1.0
    assert score.article.all_hit is True
