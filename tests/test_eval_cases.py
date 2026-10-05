from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from schemas.corpus import Edition, NormUnit
from schemas.eval_case import EvalCase
from validate.eval_cases import (
    build_manifest,
    contains_direct_citation,
    manifest_json,
    validate_case,
    validate_group_split,
    validate_split_policy,
)


NORM_ID = "44FZ:2026-08-04:art.33:ch.1:p.1"


def span(path: str, quote: str = "source quote", start: int = 0) -> dict:
    return {
        "path": path,
        "char_start": start,
        "char_end": start + len(quote),
        "quote": quote,
    }


def case_data(path: str = "data/raw/fas/id/text/decision.txt") -> dict:
    return {
        "case_id": "case-1",
        "edition_id": "2026-08-04",
        "topic": "object_description",
        "split": "dev",
        "outcome": "upheld",
        "complaint_argument_raw": "source quote",
        "complaint_argument_blind": "source quote",
        "complaint_argument_summary": "editor summary",
        "complaint_argument_source": span(path),
        "gold_norms": [NORM_ID],
        "gold_evidence": [{"norm_id": NORM_ID, "decision_span": span(path)}],
        "outcome_evidence": span(path),
        "relies_on_bylaw": False,
        "annotation_status": "verified",
        "exclusion_reasons": [],
        "case_group_id": "group-1",
        "duplicate_of": None,
        "doc_fragment": "fragment",
        "doc_paths": ["data/raw/fas/id/", "data/raw/notices/notice-id/"],
        "notice_date": "2026-08-05",
        "fas_decision_id": "id",
    }


def edition() -> Edition:
    return Edition(
        edition_id="2026-08-04",
        effective_from="2026-08-04",
        content_sha256="a" * 64,
    )


def norm() -> NormUnit:
    return NormUnit(
        norm_id=NORM_ID,
        edition_id="2026-08-04",
        article="33",
        part="1",
        point="1",
        level="point",
        parent_norm_id="44FZ:2026-08-04:art.33:ch.1",
        text="norm",
        char_start=0,
        char_end=4,
        order=1,
    )


@pytest.mark.parametrize(
    "missing_field",
    [
        "complaint_argument_raw",
        "complaint_argument_blind",
        "complaint_argument_source",
        "gold_evidence",
        "outcome_evidence",
    ],
)
def test_verified_requires_queries_and_evidence(missing_field: str) -> None:
    data = case_data()
    data[missing_field] = [] if missing_field == "gold_evidence" else None
    with pytest.raises(ValidationError):
        EvalCase.model_validate(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [("outcome", "rejected"), ("relies_on_bylaw", True), ("duplicate_of", "case-0")],
)
def test_ineligible_case_cannot_be_verified(field: str, value: object) -> None:
    data = case_data()
    data[field] = value
    with pytest.raises(ValidationError):
        EvalCase.model_validate(data)


def test_draft_is_not_added_to_benchmark_manifest(tmp_path: Path) -> None:
    verified = EvalCase.model_validate(case_data())
    draft_data = case_data()
    draft_data.update(
        case_id="case-2",
        annotation_status="draft",
        exclusion_reasons=["ambiguous_norm_mapping"],
    )
    draft = EvalCase.model_validate(draft_data)
    norms_path = tmp_path / "norms.jsonl"
    norms_path.write_text("{}\n", encoding="utf-8")
    manifest = build_manifest([verified, draft], edition(), norms_path)
    assert [item["case_id"] for item in manifest["dev"]] == ["case-1"]
    assert manifest["non_benchmark"][0]["case_id"] == "case-2"


def test_group_cannot_cross_dev_and_eval() -> None:
    first = EvalCase.model_validate(case_data())
    second_data = case_data()
    second_data.update(case_id="case-2", split="eval")
    second = EvalCase.model_validate(second_data)
    assert validate_group_split([first, second]) == [
        "case group appears in multiple splits: group-1"
    ]


def test_split_policy_keeps_singleton_topics_in_dev() -> None:
    singleton_data = case_data()
    singleton_data["split"] = "eval"
    singleton = EvalCase.model_validate(singleton_data)
    assert "singleton topic must remain in dev: object_description" in validate_split_policy(
        [singleton]
    )


def test_gold_norm_and_source_span_are_validated(tmp_path: Path) -> None:
    source = tmp_path / "data/raw/fas/id/text/decision.txt"
    source.parent.mkdir(parents=True)
    (tmp_path / "data/raw/notices/notice-id").mkdir(parents=True)
    source.write_text("source quote", encoding="utf-8")
    case = EvalCase.model_validate(case_data())
    assert validate_case(case, root=tmp_path, edition=edition(), norm_units={NORM_ID: norm()}) == []

    errors = validate_case(case, root=tmp_path, edition=edition(), norm_units={})
    assert any("gold norm does not exist" in error for error in errors)
    source.write_text("changed text", encoding="utf-8")
    errors = validate_case(case, root=tmp_path, edition=edition(), norm_units={NORM_ID: norm()})
    assert any("quote does not match source" in error for error in errors)


@pytest.mark.parametrize(
    "query",
    ["нарушена статья 33", "не соблюдена ч. 1", "требования пункта 5"],
)
def test_blind_query_rejects_numbered_citations(query: str) -> None:
    assert contains_direct_citation(query)


def test_blind_query_accepts_argument_without_citations() -> None:
    assert not contains_direct_citation("Заказчик установил избыточное требование к упаковке")


def test_manifest_is_reproducible(tmp_path: Path) -> None:
    norms_path = tmp_path / "norms.jsonl"
    norms_path.write_text('{"stable": true}\n', encoding="utf-8")
    first = EvalCase.model_validate(case_data())
    second_data = deepcopy(case_data())
    second_data.update(case_id="case-2", case_group_id="group-2", split="eval")
    second = EvalCase.model_validate(second_data)

    forward = manifest_json(build_manifest([first, second], edition(), norms_path))
    reverse = manifest_json(build_manifest([second, first], edition(), norms_path))
    assert forward == reverse
