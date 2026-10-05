"""Validation and freezing helpers for the hand-labelled retrieval benchmark."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from pydantic import ValidationError

from schemas.corpus import Edition, NormUnit
from schemas.eval_case import EvalCase, SourceSpan


DIRECT_CITATION_RE = re.compile(
    r"(?iu)(?:\bстат(?:ья|ьи|ье|ью|ьёй|ей)|\bст\.?|"
    r"\bчаст(?:ь|и|ью)|\bч\.?|\bпункт(?:а|е|ом|ы)?|\bп\.?)\s*\d+(?:\.\d+)*"
)
ALLOWED_GOLD_LEVELS = {"article", "part", "point"}


def contains_direct_citation(text: str) -> bool:
    """Return whether a query exposes a numbered article/part/point citation."""

    return bool(DIRECT_CITATION_RE.search(text))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def discover_case_paths(cases_dir: Path) -> list[Path]:
    return sorted(path for path in cases_dir.glob("*.json") if path.name != "schema_example.json")


def validate_source_span(root: Path, span: SourceSpan, label: str) -> list[str]:
    path = root / span.path
    if not path.is_file():
        return [f"{label}: source does not exist: {span.path}"]
    text = path.read_text(encoding="utf-8")
    if span.char_end > len(text):
        return [f"{label}: char_end {span.char_end} exceeds source length {len(text)}"]
    actual = text[span.char_start : span.char_end]
    if actual != span.quote:
        return [f"{label}: quote does not match source at [{span.char_start}:{span.char_end}]"]
    return []


def derived_exclusion_reasons(case: EvalCase) -> set[str]:
    """Compute blockers that can be derived without legal interpretation."""

    reasons: set[str] = set()
    if case.outcome == "rejected":
        reasons.add("rejected_argument")
    if case.relies_on_bylaw:
        reasons.add("relies_on_bylaw")
    if not case.gold_norms:
        reasons.add("missing_gold_norms")
    if case.duplicate_of:
        reasons.add("duplicate_case")
    return reasons


def validate_case(
    case: EvalCase,
    *,
    root: Path,
    edition: Edition,
    norm_units: dict[str, NormUnit],
) -> list[str]:
    errors: list[str] = []
    prefix = case.case_id

    if case.edition_id != edition.edition_id:
        errors.append(f"{prefix}: edition_id does not match loaded corpus")

    for source_path in case.doc_paths:
        if not (root / source_path).exists():
            errors.append(f"{prefix}: document source does not exist: {source_path}")

    for norm_id in case.gold_norms:
        unit = norm_units.get(norm_id)
        if unit is None:
            errors.append(f"{prefix}: gold norm does not exist: {norm_id}")
        elif unit.level not in ALLOWED_GOLD_LEVELS:
            errors.append(f"{prefix}: unsupported gold norm level {unit.level}: {norm_id}")
        elif unit.edition_id != case.edition_id:
            errors.append(f"{prefix}: gold norm belongs to another edition: {norm_id}")

    spans: list[tuple[str, SourceSpan]] = []
    if case.complaint_argument_source:
        spans.append(("complaint_argument_source", case.complaint_argument_source))
    if case.outcome_evidence:
        spans.append(("outcome_evidence", case.outcome_evidence))
    spans.extend(
        (f"gold_evidence[{evidence.norm_id}]", evidence.decision_span)
        for evidence in case.gold_evidence
    )
    for label, span in spans:
        errors.extend(validate_source_span(root, span, f"{prefix}.{label}"))

    if (
        case.complaint_argument_source
        and case.complaint_argument_raw != case.complaint_argument_source.quote
    ):
        errors.append(f"{prefix}: complaint_argument_raw must equal its exact source quote")
    if case.complaint_argument_blind and contains_direct_citation(case.complaint_argument_blind):
        errors.append(f"{prefix}: complaint_argument_blind contains a direct numbered citation")

    if case.annotation_status == "verified":
        blockers = derived_exclusion_reasons(case)
        if blockers:
            errors.append(f"{prefix}: verified case has eligibility blockers: {sorted(blockers)}")
        try:
            notice_date = date.fromisoformat(case.notice_date or "")
        except ValueError:
            errors.append(f"{prefix}: notice_date must be an ISO date")
        else:
            if edition.effective_from and notice_date < date.fromisoformat(edition.effective_from):
                errors.append(f"{prefix}: corpus edition was not effective on notice_date")
            if edition.effective_to and notice_date > date.fromisoformat(edition.effective_to):
                errors.append(f"{prefix}: corpus edition was no longer effective on notice_date")

        if not case.doc_paths:
            errors.append(f"{prefix}: verified case must list its local source directories")
        decision_spans = [case.outcome_evidence, *(item.decision_span for item in case.gold_evidence)]
        decision_prefix = f"data/raw/fas/{case.fas_decision_id}/text/"
        if not case.fas_decision_id or any(
            span is None or not span.path.startswith(decision_prefix) for span in decision_spans
        ):
            errors.append(f"{prefix}: verified evidence must point to extracted full-decision text")
    else:
        missing = derived_exclusion_reasons(case) - set(case.exclusion_reasons)
        if missing:
            errors.append(f"{prefix}: missing derived exclusion reasons: {sorted(missing)}")

    return errors


def validate_group_split(cases: Iterable[EvalCase]) -> list[str]:
    splits_by_group: dict[str, set[str]] = {}
    for case in cases:
        if case.annotation_status == "verified":
            splits_by_group.setdefault(case.case_group_id, set()).add(case.split)
    return [
        f"case group appears in multiple splits: {group_id}"
        for group_id, splits in sorted(splits_by_group.items())
        if len(splits) > 1
    ]


def validate_split_policy(cases: Iterable[EvalCase]) -> list[str]:
    """Check the frozen 70/30 grouped split and per-topic eval coverage."""

    verified = [case for case in cases if case.annotation_status == "verified"]
    errors: list[str] = []
    expected_eval = round(len(verified) * 0.30)
    actual_eval = sum(case.split == "eval" for case in verified)
    if actual_eval != expected_eval:
        errors.append(f"expected {expected_eval} eval cases for a 70/30 split, found {actual_eval}")

    by_topic: dict[str, list[EvalCase]] = {}
    for case in verified:
        by_topic.setdefault(case.topic, []).append(case)
    for topic, topic_cases in sorted(by_topic.items()):
        eval_count = sum(case.split == "eval" for case in topic_cases)
        if len(topic_cases) >= 2 and eval_count == 0:
            errors.append(f"topic with multiple cases has no eval representative: {topic}")
        if len(topic_cases) == 1 and eval_count:
            errors.append(f"singleton topic must remain in dev: {topic}")
    return errors


def load_and_validate(root: Path) -> tuple[list[EvalCase], Edition, Path, list[str]]:
    cases_dir = root / "evals/cases"
    errors: list[str] = []
    cases: list[EvalCase] = []
    for path in discover_case_paths(cases_dir):
        try:
            cases.append(EvalCase.model_validate_json(path.read_text(encoding="utf-8")))
        except ValidationError as exc:
            errors.append(f"{path.relative_to(root)}: {exc}")

    edition_ids = {case.edition_id for case in cases}
    if len(edition_ids) != 1:
        errors.append(f"expected one corpus edition, found: {sorted(edition_ids)}")
        edition_id = next(iter(edition_ids), "2026-08-04")
    else:
        edition_id = next(iter(edition_ids))

    edition_path = root / "data/law" / edition_id / "edition.json"
    norms_path = root / "data/law" / edition_id / "norm_units.jsonl"
    edition = Edition.model_validate_json(edition_path.read_text(encoding="utf-8"))
    units = {
        unit.norm_id: unit
        for unit in (NormUnit.model_validate(row) for row in _load_jsonl(norms_path))
    }
    for case in cases:
        errors.extend(validate_case(case, root=root, edition=edition, norm_units=units))
    errors.extend(validate_group_split(cases))
    errors.extend(validate_split_policy(cases))
    return cases, edition, norms_path, errors


def build_manifest(cases: Iterable[EvalCase], edition: Edition, norms_path: Path) -> dict[str, Any]:
    ordered = sorted(cases, key=lambda case: case.case_id)
    verified = [case for case in ordered if case.annotation_status == "verified"]
    excluded = [case for case in ordered if case.annotation_status != "verified"]
    counts = Counter(case.annotation_status for case in ordered)
    return {
        "manifest_version": 1,
        "edition_id": edition.edition_id,
        "law_content_sha256": edition.content_sha256,
        "norm_units_sha256": sha256_file(norms_path),
        "query_variants": ["raw", "blind"],
        "counts": {
            "all": len(ordered),
            "verified": counts["verified"],
            "draft": counts["draft"],
            "excluded": counts["excluded"],
            "dev": sum(case.split == "dev" for case in verified),
            "eval": sum(case.split == "eval" for case in verified),
        },
        "dev": [_manifest_case(case) for case in verified if case.split == "dev"],
        "eval": [_manifest_case(case) for case in verified if case.split == "eval"],
        "non_benchmark": [
            {
                "case_id": case.case_id,
                "annotation_status": case.annotation_status,
                "case_group_id": case.case_group_id,
                "duplicate_of": case.duplicate_of,
                "exclusion_reasons": sorted(case.exclusion_reasons),
            }
            for case in excluded
        ],
    }


def _manifest_case(case: EvalCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "case_group_id": case.case_group_id,
        "topic": case.topic,
        "gold_norms": case.gold_norms,
    }


def manifest_json(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
