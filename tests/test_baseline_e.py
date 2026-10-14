from __future__ import annotations

import json
from pathlib import Path

import pytest

from index.baseline_e import (
    build_mapping,
    leave_one_group_out_predictions,
    lookup,
    overlay_mapping,
    score_predictions,
)
from schemas.eval_case import VERIFIED_ISSUE_TYPES


N1 = "44FZ:2026-08-04:art.33:ch.1:p.1"
N2 = "44FZ:2026-08-04:art.42:ch.2:p.1"
N3 = "44FZ:2026-08-04:art.8"


def row(case: str, group: str, issue_type: str, norms: list[str]) -> dict[str, object]:
    return {
        "case_id": case,
        "case_group_id": group,
        "issue_type": issue_type,
        "gold_norms": norms,
        "topic": "object_description",
    }


def test_mapping_uses_distinct_group_support_before_case_count() -> None:
    rows = [
        row("a1", "a", "restrictive_object_description", [N1]),
        row("a2", "a", "restrictive_object_description", [N1]),
        row("b", "b", "restrictive_object_description", [N2]),
        row("c", "c", "restrictive_object_description", [N2]),
    ]
    built = build_mapping(rows)
    assert built.mapping["restrictive_object_description"] == (N2, N1)
    assert built.group_support["restrictive_object_description"][N2] == 2
    assert built.case_support["restrictive_object_description"][N1] == 2


def test_mapping_ties_are_resolved_by_norm_id() -> None:
    rows = [row("a", "a", "defective_object_description", [N2, N1])]
    assert build_mapping(rows).mapping["defective_object_description"] == (N1, N2)


def test_service_and_unknown_types_have_no_fallback() -> None:
    built = build_mapping([row("a", "a", "unclear", [N1])])
    assert built.mapping == {}
    assert lookup({"unclear": [N1]}, "unclear") == ()
    assert lookup({}, "new_type") == ()


def test_leave_one_group_out_removes_the_complete_group() -> None:
    rows = [
        row("a1", "a", "defective_object_description", [N1]),
        row("a2", "a", "defective_object_description", [N1]),
        row("b", "b", "defective_object_description", [N2]),
    ]
    predictions = leave_one_group_out_predictions(rows)
    assert predictions == [(N2,), (N2,), (N1,)]


def test_overlay_is_explicit_deduplicated_and_capped() -> None:
    result = overlay_mapping(
        {"defective_object_description": [N1]},
        {"unrelated_items_in_one_lot": [N2, N2, N3]},
        max_norms=2,
    )
    assert result["defective_object_description"] == (N1,)
    assert result["unrelated_items_in_one_lot"] == (N2, N3)
    with pytest.raises(ValueError):
        overlay_mapping({}, {"unclear": [N1]})


def test_scoring_preserves_exact_and_article_semantics() -> None:
    rows = [row("a", "a", "defective_object_description", [N1, N2])]
    metrics, scores = score_predictions(rows, [(N1,)])
    assert metrics.recall_exact == 0.5
    assert metrics.recall_article == 0.5
    assert not scores[0].exact.all_hit
    assert not scores[0].article.all_hit


def test_lookup_applies_top_k() -> None:
    assert lookup(
        {"defective_object_description": [N1, N2]},
        "defective_object_description",
        k=1,
    ) == (N1,)
    with pytest.raises(ValueError):
        lookup({}, "defective_object_description", k=0)


def test_frozen_manifest_and_zero_shot_cover_the_taxonomy() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "evals/retrieval_manifest.json").read_text(encoding="utf-8"))
    zero_shot = json.loads(
        (root / "evals/baselines/baseline_e_zero_shot.json").read_text(encoding="utf-8")
    )
    assert len(manifest["dev"]) == 52
    assert len(manifest["eval"]) == 22
    dev_types = {item["issue_type"] for item in manifest["dev"]}
    eval_types = {item["issue_type"] for item in manifest["eval"]}
    overlay_types = set(zero_shot["overrides"])
    assert overlay_types == eval_types - dev_types
    assert dev_types | overlay_types | {"other"} == VERIFIED_ISSUE_TYPES
