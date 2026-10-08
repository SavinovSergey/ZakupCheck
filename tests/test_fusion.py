"""RRF and article-then-point fusion. No model download."""

from index.fusion import (
    article_diversify,
    article_then_points,
    gold_id_status,
    interleave,
    is_pointer_text,
    rrf,
    top_articles,
    weighted_rrf,
)
from schemas.corpus import Chunk


def _chunk(chunk_id: str, *norm_ids: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        edition_id="2026-08-04",
        chunking="point",
        text=chunk_id,
        char_start=0,
        char_end=len(chunk_id),
        norm_ids=list(norm_ids),
        max_chunk=1000,
        article_title_prefix=False,
    )


ART_33 = "44FZ:2026-08-04:art.33:ch.1:p.1"
ART_42 = "44FZ:2026-08-04:art.42:ch.1:p.1"
ART_48 = "44FZ:2026-08-04:art.48:ch.1:p.1"


def test_rrf_promotes_a_chunk_that_both_lists_rank_high() -> None:
    shared = _chunk("shared", ART_33)
    only_dense = _chunk("dense", ART_42)
    only_bm25 = _chunk("bm25", ART_48)
    fused = rrf([[shared, only_dense], [shared, only_bm25]], limit=3)
    assert [chunk.chunk_id for chunk in fused] == ["shared", "bm25", "dense"]


def test_rrf_limit_cuts_the_fused_list() -> None:
    fused = rrf(
        [[_chunk("a", ART_33), _chunk("b", ART_42)], [_chunk("c", ART_48)]],
        limit=1,
    )
    assert [chunk.chunk_id for chunk in fused] == ["a"]


def test_article_then_points_drops_other_articles() -> None:
    ranked = [
        _chunk("gold-article", ART_33),
        _chunk("other-1", ART_48),
        _chunk("other-2", ART_42),
        _chunk("gold-point", ART_33),
    ]
    picked = article_then_points(ranked, article_pool=3, n_articles=1, limit=10)
    assert [chunk.chunk_id for chunk in picked] == ["gold-article", "gold-point"]
    assert top_articles(ranked, article_pool=3, n_articles=1) == ["art.33"]


def test_pointer_text_and_gold_id_status() -> None:
    pointer = "1) описание объекта закупки в соответствии со статьей 33 настоящего Федерального закона;"
    assert is_pointer_text(pointer)
    assert not is_pointer_text("Заказчик описывает объект закупки подробно, " * 20)
    corpus = {"44FZ:2026-08-04:art.33:ch.1:p.1", "44FZ:2026-08-04:art.8:ch.1"}
    assert gold_id_status("44FZ:2026-08-04:art.33:ch.1:p.1", corpus) == "exact"
    assert gold_id_status("44FZ:2026-08-04:art.8", corpus) == "descendant"
    assert gold_id_status("44FZ:2026-08-04:art.33:ch.1:p.1:sub.2", corpus) == "ancestor"
    assert gold_id_status("44FZ:2026-08-04:art.99", corpus) == "absent"


def test_weighted_rrf_can_prefer_one_source() -> None:
    dense = [_chunk("dense", ART_33), _chunk("shared", ART_42)]
    bm25 = [_chunk("bm25", ART_48), _chunk("shared", ART_42)]
    fused = weighted_rrf([dense, bm25], weights=[2.0, 1.0], k=10, pool=2, limit=3)
    assert [chunk.chunk_id for chunk in fused] == ["shared", "dense", "bm25"]


def test_interleave_is_unique_and_respects_source_order() -> None:
    shared = _chunk("shared", ART_42)
    dense = [_chunk("dense", ART_33), shared]
    bm25 = [_chunk("bm25", ART_48), shared]
    merged = interleave([dense, bm25], pool=2, limit=4)
    assert [chunk.chunk_id for chunk in merged] == ["dense", "bm25", "shared"]


def test_article_diversify_keeps_one_representative_per_article() -> None:
    ranked = [
        _chunk("33-a", ART_33),
        _chunk("33-b", ART_33),
        _chunk("42", ART_42),
        _chunk("48", ART_48),
    ]
    diversified = article_diversify(ranked, pool=4, limit=3)
    assert [chunk.chunk_id for chunk in diversified] == ["33-a", "42", "48"]
