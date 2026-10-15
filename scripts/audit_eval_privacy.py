#!/usr/bin/env python3
"""Audit publishable EvalCase text fields for automatically detectable PII."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from index.eval_tables import load_verified  # noqa: E402
from privacy.anonymize import (  # noqa: E402
    Detection,
    ManualRedaction,
    fingerprint,
    load_manual_redactions,
    scan_text,
)
from schemas.eval_case import EvalCase  # noqa: E402


def publishable_text_fields(case: EvalCase) -> dict[str, str]:
    """Return text that can be copied into the public benchmark record."""

    fields = {
        "complaint_argument_raw": case.complaint_argument_raw or "",
        "complaint_argument_blind": case.complaint_argument_blind or "",
        "complaint_argument_summary": case.complaint_argument_summary,
        "doc_fragment": case.doc_fragment,
    }
    if case.outcome_evidence:
        fields["outcome_evidence.quote"] = case.outcome_evidence.quote
    for index, evidence in enumerate(case.gold_evidence):
        fields[f"gold_evidence[{index}].decision_quote"] = evidence.decision_span.quote
    return fields


def _detection_record(
    case: EvalCase,
    field: str,
    detection: Detection,
    *,
    include_values: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "case_id": case.case_id,
        "split": case.split,
        "field": field,
        "kind": detection.kind,
        "rule": detection.rule,
        "start": detection.start,
        "end": detection.end,
        "fingerprint": fingerprint(detection.value),
    }
    if include_values:
        record["value"] = detection.value
    return record


def build_audit(
    cases: Iterable[EvalCase],
    *,
    global_redactions: list[ManualRedaction],
    redactions_by_case: dict[str, list[ManualRedaction]],
    include_values: bool = False,
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    kinds: Counter[str] = Counter()
    rules: Counter[str] = Counter()
    field_count = 0
    ordered = sorted(cases, key=lambda case: case.case_id)
    for case in ordered:
        manual = [*global_redactions, *redactions_by_case.get(case.case_id, [])]
        for field, text in publishable_text_fields(case).items():
            field_count += 1
            for detection in scan_text(text, manual_redactions=manual):
                kinds[detection.kind] += 1
                rules[detection.rule] += 1
                findings.append(
                    _detection_record(
                        case,
                        field,
                        detection,
                        include_values=include_values,
                    )
                )
    return {
        "audit_version": 1,
        "scope": "verified public benchmark text fields",
        "manual_review_required": True,
        "warning": (
            "Automatic detection is not proof of anonymization. Review every exported "
            "record and add missed entities to the manual redactions file."
        ),
        "summary": {
            "cases": len(ordered),
            "text_fields": field_count,
            "detections": len(findings),
            "by_kind": dict(sorted(kinds.items())),
            "by_rule": dict(sorted(rules.items())),
        },
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-dir", type=Path, default=ROOT / "evals/cases")
    parser.add_argument("--redactions", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/analysis/eval_privacy_audit.json",
    )
    parser.add_argument(
        "--include-values",
        action="store_true",
        help="Include raw detected values in the local report (never publish it)",
    )
    args = parser.parse_args()

    global_redactions, by_case = load_manual_redactions(args.redactions)
    cases = load_verified(args.cases_dir)
    audit = build_audit(
        cases,
        global_redactions=global_redactions,
        redactions_by_case=by_case,
        include_values=args.include_values,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = audit["summary"]
    print(
        f"Audited {summary['cases']} verified cases: "
        f"{summary['detections']} detection(s). Wrote {args.output}"
    )
    print("Manual review remains required before publication.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
