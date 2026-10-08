"""Error-analysis helpers on synthetic chunks; no model downloads."""

from index.error_analysis import (
    best_rank,
    complementarity,
    paired_group_bootstrap,
    rank_bucket,
    repeated_article_slots,
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
        max_chunk=700,
        article_title_prefix=False,
    )


ART_33 = "44FZ:2026-08-04:art.33:ch.1:p.1"
ART_33_OTHER = "44FZ:2026-08-04:art.33:ch.2:p.1"
ART_42 = "44FZ:2026-08-04:art.42:ch.1:p.1"


def test_rank_bucket_boundaries() -> None:
    assert [rank_bucket(rank) for rank in (1, 10, 11, 20, 21, 50, 51, 100, 101, None)] == [
        "top-10", "top-10", "11-20", "11-20", "21-50", "21-50", "51-100", "51-100", ">100", ">100"
    ]


def test_best_rank_separates_article_and_exact() -> None:
    ranked = [_chunk("wrong-point", ART_33_OTHER), _chunk("exact", ART_33)]
    assert best_rank(ART_33, ranked, level="article") == 1
    assert best_rank(ART_33, ranked, level="exact") == 2


def test_complementarity_has_four_exclusive_states() -> None:
    assert complementarity(ART_33, [_chunk("d", ART_33)], [_chunk("b", ART_42)]) == "only-dense"
    assert complementarity(ART_33, [_chunk("d", ART_42)], [_chunk("b", ART_33)]) == "only-bm25"
    assert complementarity(ART_33, [_chunk("d", ART_33)], [_chunk("b", ART_33)]) == "both"
    assert complementarity(ART_33, [_chunk("d", ART_42)], [_chunk("b", ART_42)]) == "neither"


def test_repeated_article_slots_counts_only_extra_non_gold_chunks() -> None:
    ranked = [_chunk("a", ART_42), _chunk("b", ART_42), _chunk("c", ART_42), _chunk("gold", ART_33)]
    assert repeated_article_slots(ranked, [ART_33]) == 2


def test_group_bootstrap_is_deterministic_and_paired() -> None:
    interval = paired_group_bootstrap([0.0, 1.0, 0.0], [1.0, 1.0, 0.0], ["a", "a", "b"], samples=200, seed=7)
    again = paired_group_bootstrap([0.0, 1.0, 0.0], [1.0, 1.0, 0.0], ["a", "a", "b"], samples=200, seed=7)
    assert interval == again
    assert interval.mean_delta == 1 / 3
