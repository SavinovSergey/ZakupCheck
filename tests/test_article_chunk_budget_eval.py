from __future__ import annotations

import sys
from pathlib import Path

import pytest

from schemas.corpus import Chunk
from schemas.eval_case import EvalCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_article_chunk_budget_eval import pipeline_diagnostics


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
    )


def test_pipeline_diagnostics_separates_candidate_shortlist_pool_and_final() -> None:
    art1 = "44FZ:2026-08-04:art.1:ch.1:p.1"
    art2 = "44FZ:2026-08-04:art.2:ch.1:p.1"
    case = EvalCase.model_construct(gold_norms=[art1, art2])
    diagnostics = pipeline_diagnostics(
        [case],
        [["art.1", "art.2"]],
        [["art.1"]],
        [[_chunk("pool", art1)]],
        [[_chunk("final", art1)]],
    )
    assert diagnostics["candidate_article_recall"] == 1.0
    assert diagnostics["shortlist_article_recall"] == 0.5
    assert diagnostics["pool_exact_recall"] == 0.5
    assert diagnostics["final_article_recall"] == 0.5
    assert diagnostics["candidate_to_shortlist_article_loss"] == 0.5
    assert diagnostics["pool_to_final_exact_loss"] == 0.0


def test_pipeline_diagnostics_measures_exact_loss_inside_final_top10() -> None:
    gold = "44FZ:2026-08-04:art.1:ch.1:p.2"
    other = "44FZ:2026-08-04:art.1:ch.1:p.1"
    case = EvalCase.model_construct(gold_norms=[gold])
    diagnostics = pipeline_diagnostics(
        [case],
        [["art.1"]],
        [["art.1"]],
        [[_chunk("gold", gold), _chunk("other", other)]],
        [[_chunk("other", other)]],
    )
    assert diagnostics["pool_exact_recall"] == 1.0
    assert diagnostics["final_exact_recall"] == 0.0
    assert diagnostics["pool_to_final_exact_loss"] == 1.0
    assert diagnostics["final_article_recall"] == 1.0


def test_pipeline_diagnostics_rejects_misaligned_cases() -> None:
    case = EvalCase.model_construct(gold_norms=["44FZ:2026-08-04:art.1"])
    with pytest.raises(ValueError):
        pipeline_diagnostics([case], [], [], [], [])
