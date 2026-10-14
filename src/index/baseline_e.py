"""Deterministic baseline E: an ``issue_type -> norm_ids`` lookup table."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from index.recall import AggregateRecall, CaseScore, aggregate, score_case


SERVICE_ISSUE_TYPES = frozenset({"other", "unclear"})


@dataclass(frozen=True)
class MappingBuild:
    mapping: dict[str, tuple[str, ...]]
    group_support: dict[str, dict[str, int]]
    case_support: dict[str, dict[str, int]]


def build_mapping(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_norms: int = 10,
) -> MappingBuild:
    """Build a dev-only table with group-aware, deterministic norm ranking.

    A norm is ranked by the number of distinct case groups that support it,
    then by raw case support, then lexicographically by norm id.  Service
    labels deliberately have no statutory fallback.
    """
    if max_norms <= 0:
        raise ValueError("max_norms must be positive")
    groups: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    cases: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        issue_type = str(row["issue_type"])
        if issue_type in SERVICE_ISSUE_TYPES:
            continue
        group_id = str(row["case_group_id"])
        for norm_id in row["gold_norms"]:
            norm = str(norm_id)
            groups[issue_type][norm].add(group_id)
            cases[issue_type][norm] += 1

    mapping: dict[str, tuple[str, ...]] = {}
    group_support: dict[str, dict[str, int]] = {}
    case_support: dict[str, dict[str, int]] = {}
    for issue_type in sorted(groups):
        ranked = sorted(
            groups[issue_type],
            key=lambda norm_id: (
                -len(groups[issue_type][norm_id]),
                -cases[issue_type][norm_id],
                norm_id,
            ),
        )
        mapping[issue_type] = tuple(ranked[:max_norms])
        group_support[issue_type] = {
            norm_id: len(groups[issue_type][norm_id]) for norm_id in ranked
        }
        case_support[issue_type] = {
            norm_id: cases[issue_type][norm_id] for norm_id in ranked
        }
    return MappingBuild(mapping, group_support, case_support)


def overlay_mapping(
    base: Mapping[str, Sequence[str]],
    overrides: Mapping[str, Sequence[str]],
    *,
    max_norms: int = 10,
) -> dict[str, tuple[str, ...]]:
    """Return a copy of ``base`` with explicit zero-shot type overrides."""
    if max_norms <= 0:
        raise ValueError("max_norms must be positive")
    merged = {key: tuple(values[:max_norms]) for key, values in base.items()}
    for issue_type, norm_ids in overrides.items():
        if issue_type in SERVICE_ISSUE_TYPES:
            raise ValueError(f"service issue type cannot have norms: {issue_type}")
        merged[issue_type] = tuple(dict.fromkeys(norm_ids))[:max_norms]
    return merged


def lookup(
    mapping: Mapping[str, Sequence[str]],
    issue_type: str,
    *,
    k: int = 10,
) -> tuple[str, ...]:
    if k <= 0:
        raise ValueError("k must be positive")
    if issue_type in SERVICE_ISSUE_TYPES:
        return ()
    return tuple(mapping.get(issue_type, ()))[:k]


def predict_rows(
    rows: Sequence[Mapping[str, Any]],
    mapping: Mapping[str, Sequence[str]],
    *,
    k: int = 10,
) -> list[tuple[str, ...]]:
    return [lookup(mapping, str(row["issue_type"]), k=k) for row in rows]


def leave_one_group_out_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    k: int = 10,
    max_norms: int = 10,
) -> list[tuple[str, ...]]:
    """Predict every row from a table that excludes its complete case group."""
    predictions: list[tuple[str, ...]] = []
    for row in rows:
        held_out = str(row["case_group_id"])
        training = [item for item in rows if str(item["case_group_id"]) != held_out]
        mapping = build_mapping(training, max_norms=max_norms).mapping
        predictions.append(lookup(mapping, str(row["issue_type"]), k=k))
    return predictions


def score_predictions(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[str]],
) -> tuple[AggregateRecall, list[CaseScore]]:
    if len(rows) != len(predictions):
        raise ValueError("rows and predictions must have the same length")
    scores = [
        score_case(list(row["gold_norms"]), set(predicted))
        for row, predicted in zip(rows, predictions, strict=True)
    ]
    return aggregate(scores), scores
