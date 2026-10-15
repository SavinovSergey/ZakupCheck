from __future__ import annotations

import json
from pathlib import Path

import pytest

from privacy.finalize import FinalizationError, finalize_dataset, validate_candidate


def _record(*, split: str, labels: bool) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": 1,
        "case_id": f"case_{split}",
        "case_group_id": f"group_{split}",
        "edition_id": "2026-08-04",
        "split": split,
        "topic": "object_description",
        "complaint_argument_raw": "Заказчик ограничил конкуренцию.",
        "complaint_argument_blind": "Заказчик ограничил конкуренцию.",
        "complaint_argument_summary": "Ограничение конкуренции.",
        "doc_fragment": "Описание объекта закупки.",
        "source": {"procurement_id": "123", "procurement_url": None},
    }
    if labels:
        record.update(
            {
                "outcome": "upheld",
                "outcome_evidence_quote": "Жалоба признана обоснованной.",
                "issue_type": "restrictive_object_description",
                "gold_norms": ["44FZ:2026-08-04:art.33"],
                "gold_evidence": [],
            }
        )
    return record


def _write_candidate(root: Path, *, eval_labels: bool = False) -> Path:
    root.mkdir()
    dev = _record(split="dev", labels=True)
    eval_record = _record(split="eval", labels=eval_labels)
    (root / "dev.jsonl").write_text(json.dumps(dev) + "\n", encoding="utf-8")
    (root / "eval.jsonl").write_text(
        json.dumps(eval_record) + "\n", encoding="utf-8"
    )
    (root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "publication_ready": False,
                "source_cases_modified": False,
                "eval_labels_included": eval_labels,
                "counts": {"dev": 1, "eval": 1},
                "version": "1-candidate",
            }
        ),
        encoding="utf-8",
    )
    (root / "privacy_audit.json").write_text(
        json.dumps({"contains_raw_values": False}), encoding="utf-8"
    )
    return root


def test_finalize_creates_release_without_modifying_candidate(tmp_path: Path) -> None:
    candidate = _write_candidate(tmp_path / "candidate")
    original_manifest = (candidate / "dataset_manifest.json").read_bytes()
    license_file = tmp_path / "license.md"
    license_file.write_text("Dataset license chosen by its publisher.\n", encoding="utf-8")
    output = tmp_path / "release"

    finalize_dataset(
        input_dir=candidate,
        output_dir=output,
        version="1.0.0",
        reviewed_by="review-team",
        review_date="2026-10-14",
        license_file=license_file,
        hf_license="other",
    )

    manifest = json.loads((output / "dataset_manifest.json").read_text())
    assert manifest["publication_ready"] is True
    assert manifest["privacy_review"]["status"] == "completed"
    assert manifest["counts"] == {"dev": 1, "eval": 1}
    assert (output / "README.md").is_file()
    assert (output / "DATA_LICENSE.md").is_file()
    assert (output / "SHA256SUMS").is_file()
    assert (candidate / "dataset_manifest.json").read_bytes() == original_manifest

    dev_record = json.loads((output / "dev.jsonl").read_text())
    eval_record = json.loads((output / "eval.jsonl").read_text())
    assert set(eval_record) == set(dev_record)
    assert all(
        eval_record[field] is None
        for field in (
            "gold_evidence",
            "gold_norms",
            "issue_type",
            "outcome",
            "outcome_evidence_quote",
        )
    )


def test_candidate_with_eval_labels_is_rejected(tmp_path: Path) -> None:
    candidate = _write_candidate(tmp_path / "candidate", eval_labels=True)

    with pytest.raises(FinalizationError, match="must not include eval labels"):
        validate_candidate(candidate)


def test_candidate_with_private_path_is_rejected(tmp_path: Path) -> None:
    candidate = _write_candidate(tmp_path / "candidate")
    record = json.loads((candidate / "dev.jsonl").read_text())
    record["path"] = "data/raw/fas/private.txt"
    (candidate / "dev.jsonl").write_text(json.dumps(record) + "\n")

    with pytest.raises(FinalizationError, match="private provenance key"):
        validate_candidate(candidate)


def test_finalize_refuses_to_overwrite_release(tmp_path: Path) -> None:
    candidate = _write_candidate(tmp_path / "candidate")
    license_file = tmp_path / "license.md"
    license_file.write_text("License\n", encoding="utf-8")
    output = tmp_path / "release"
    output.mkdir()

    with pytest.raises(FinalizationError, match="refusing to overwrite"):
        finalize_dataset(
            input_dir=candidate,
            output_dir=output,
            version="1.0.0",
            reviewed_by="review-team",
            review_date="2026-10-14",
            license_file=license_file,
            hf_license="other",
        )
