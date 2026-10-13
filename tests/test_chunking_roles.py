import pytest

from index.chunking_roles import (
    covered_fraction,
    outside_selected_fraction,
    overlap_weighted_article_ranking,
    pack_article_evidence,
)
from schemas.corpus import Chunk


def _chunk(chunk_id: str, start: int, end: int, *norm_ids: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        edition_id="ed",
        chunking="window",
        text="x" * (end - start),
        char_start=start,
        char_end=end,
        norm_ids=list(norm_ids),
        max_chunk=100,
    )


A = "44FZ:ed:art.1"
B = "44FZ:ed:art.2"


def test_overlap_weighting_demotes_a_boundary_sliver() -> None:
    spans = {"art.1": (0, 100), "art.2": (100, 200)}
    boundary = _chunk("boundary", 0, 105, A, B)
    mostly_b = _chunk("mostly-b", 100, 180, B)
    assert overlap_weighted_article_ranking([boundary, mostly_b], spans, pool=2, rank_k=0) == [
        "art.1",
        "art.2",
    ]


def test_evidence_pack_deduplicates_multi_article_representative() -> None:
    spans = {"art.1": (0, 100), "art.2": (100, 200)}
    shared = _chunk("shared", 80, 120, A, B)
    packed = pack_article_evidence([shared], ["art.1", "art.2"], spans, char_budget=40)
    assert [chunk.chunk_id for chunk in packed] == ["shared"]


def test_span_coverage_uses_union_and_noise_uses_source_spans() -> None:
    spans = {"art.1": (0, 100), "art.2": (100, 200)}
    chunks = [_chunk("a", 0, 60, A), _chunk("b", 40, 120, A, B)]
    assert covered_fraction(chunks, spans["art.1"]) == 1.0
    assert outside_selected_fraction(chunks, ["art.1"], spans) == pytest.approx(20 / 140)
