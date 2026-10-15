#!/usr/bin/env python3
"""Create an anonymized public-benchmark candidate without modifying source cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_eval_privacy import publishable_text_fields  # noqa: E402
from index.eval_tables import load_verified  # noqa: E402
from privacy.anonymize import (  # noqa: E402
    ManualRedaction,
    anonymize_fields,
    fingerprint,
    load_manual_redactions,
    scan_text,
)
from schemas.eval_case import EvalCase  # noqa: E402


def _public_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(f"zakupcheck-public-v1:{value}".encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:12]}"


def _anonymized_record(
    case: EvalCase,
    *,
    manual_redactions: list[ManualRedaction],
    include_labels: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    result = anonymize_fields(
        publishable_text_fields(case),
        manual_redactions=manual_redactions,
    )
    fields = result.fields
    record: dict[str, Any] = {
        "schema_version": 1,
        "case_id": _public_id("case", case.case_id),
        "case_group_id": _public_id("group", case.case_group_id),
        "edition_id": case.edition_id,
        "split": case.split,
        "topic": case.topic,
        "complaint_argument_raw": fields["complaint_argument_raw"],
        "complaint_argument_blind": fields["complaint_argument_blind"],
        "complaint_argument_summary": fields["complaint_argument_summary"],
        "doc_fragment": fields["doc_fragment"],
        "source": {
            "procurement_id": case.procurement_id,
            "procurement_url": case.procurement_url,
            "notice_date": case.notice_date,
            "fas_decision_id": case.fas_decision_id,
            "fas_decision_url": case.fas_decision_url,
        },
    }
    if include_labels:
        record.update(
            {
                "outcome": case.outcome,
                "issue_type": case.issue_type,
                "gold_norms": case.gold_norms,
                "outcome_evidence_quote": fields.get("outcome_evidence.quote"),
                "gold_evidence": [
                    {
                        "norm_id": evidence.norm_id,
                        "decision_quote": fields[
                            f"gold_evidence[{index}].decision_quote"
                        ],
                    }
                    for index, evidence in enumerate(case.gold_evidence)
                ],
            }
        )

    audit_rows: list[dict[str, Any]] = []
    kind_counts: Counter[str] = Counter()
    for field, detections in result.detections.items():
        for detection in detections:
            kind_counts[detection.kind] += 1
            audit_rows.append(
                {
                    "case_id": record["case_id"],
                    "source_case_fingerprint": fingerprint(case.case_id),
                    "split": case.split,
                    "field": field,
                    "kind": detection.kind,
                    "rule": detection.rule,
                    "fingerprint": fingerprint(detection.value),
                    "replacement": result.replacements[fingerprint(detection.value)],
                }
            )

    # Known patterns must not survive the replacement pass. This catches bugs
    # in overlap handling, but does not replace human review for unknown names.
    residual = {
        field: scan_text(value, manual_redactions=manual_redactions)
        for field, value in fields.items()
    }
    remaining = {field: items for field, items in residual.items() if items}
    if remaining:
        raise RuntimeError(
            f"known sensitive patterns survived redaction in {case.case_id}: "
            f"{sorted(remaining)}"
        )
    return record, audit_rows, dict(kind_counts)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-dir", type=Path, default=ROOT / "evals/cases")
    parser.add_argument("--redactions", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/public/zakupcheck-benchmark-v1-candidate",
    )
    parser.add_argument(
        "--include-eval-labels",
        action="store_true",
        help="Explicitly disclose eval gold labels in the export",
    )
    args = parser.parse_args()

    global_redactions, by_case = load_manual_redactions(args.redactions)
    cases = sorted(load_verified(args.cases_dir), key=lambda case: case.case_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_split: dict[str, list[dict[str, Any]]] = {"dev": [], "eval": []}
    audit_rows: list[dict[str, Any]] = []
    by_kind: Counter[str] = Counter()
    for case in cases:
        include_labels = case.split == "dev" or args.include_eval_labels
        manual = [*global_redactions, *by_case.get(case.case_id, [])]
        record, case_audit, case_counts = _anonymized_record(
            case,
            manual_redactions=manual,
            include_labels=include_labels,
        )
        by_split[case.split].append(record)
        audit_rows.extend(case_audit)
        by_kind.update(case_counts)

    _write_jsonl(args.output_dir / "dev.jsonl", by_split["dev"])
    _write_jsonl(args.output_dir / "eval.jsonl", by_split["eval"])
    audit = {
        "audit_version": 1,
        "manual_review_required": True,
        "contains_raw_values": False,
        "summary": {
            "detections": len(audit_rows),
            "by_kind": dict(sorted(by_kind.items())),
        },
        "redactions": audit_rows,
    }
    (args.output_dir / "privacy_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "dataset": "zakupcheck-benchmark",
        "version": "1-candidate",
        "schema_version": 1,
        "manual_review_required": True,
        "publication_ready": False,
        "source_cases_modified": False,
        "eval_labels_included": args.include_eval_labels,
        "counts": {split: len(records) for split, records in by_split.items()},
        "notes": [
            "Source paths and character offsets are intentionally excluded.",
            "Automatic redaction must be followed by a record-by-record human review.",
            "Set publication_ready only in a separate reviewed release process.",
        ],
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote public candidate: {len(by_split['dev'])} dev / "
        f"{len(by_split['eval'])} eval cases to {args.output_dir}"
    )
    print("The candidate is NOT publication-ready until manual review is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
