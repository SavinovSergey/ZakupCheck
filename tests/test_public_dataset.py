from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from data_access.public_dataset import (
    PublicDatasetError,
    download_public_dataset,
    load_public_cases,
    validate_public_dataset,
)
from index.eval_tables import load_verified


LABEL_FIELDS = (
    "gold_evidence",
    "gold_norms",
    "issue_type",
    "outcome",
    "outcome_evidence_quote",
)


def _record(split: str) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": 1,
        "case_id": f"case_{split}",
        "case_group_id": f"group_{split}",
        "edition_id": "2026-08-04",
        "split": split,
        "topic": "object_description",
        "complaint_argument_raw": "Raw query",
        "complaint_argument_blind": "Blind query",
        "complaint_argument_summary": "Summary",
        "doc_fragment": "Document fragment",
        "source": {
            "procurement_id": "123",
            "procurement_url": None,
            "notice_date": "2026-08-04",
            "fas_decision_id": "456",
            "fas_decision_url": None,
        },
    }
    if split == "dev":
        record.update(
            {
                "gold_evidence": [
                    {
                        "norm_id": "44FZ:2026-08-04:art.1",
                        "decision_quote": "Quote",
                    }
                ],
                "gold_norms": ["44FZ:2026-08-04:art.1"],
                "issue_type": "other",
                "outcome": "upheld",
                "outcome_evidence_quote": "Quote",
            }
        )
    else:
        record.update({field: None for field in LABEL_FIELDS})
    return record


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_checksums(root: Path, names: tuple[str, ...]) -> None:
    (root / "SHA256SUMS").write_text(
        "".join(f"{_sha256(root / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def _write_release(root: Path) -> Path:
    root.mkdir(parents=True)
    files = {
        "DATA_LICENSE.md": "License\n",
        "README.md": "# Dataset\n",
        "dev.jsonl": json.dumps(_record("dev")) + "\n",
        "eval.jsonl": json.dumps(_record("eval")) + "\n",
        "privacy_audit.json": json.dumps({"contains_raw_values": False}),
    }
    for name, content in files.items():
        (root / name).write_text(content, encoding="utf-8")
    manifest = {
        "version": "1.0.0",
        "publication_ready": True,
        "eval_labels_included": False,
        "counts": {"dev": 1, "eval": 1},
    }
    (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    checksummed = (*files, "dataset_manifest.json")
    _write_checksums(root, checksummed)
    return root


def test_public_release_loads_labeled_dev_and_blind_eval(tmp_path: Path) -> None:
    root = _write_release(tmp_path / "release")

    summary = validate_public_dataset(root)
    cases = load_public_cases(root)

    assert summary.dev_count == 1
    assert summary.eval_count == 1
    assert cases[0].labels_available is True
    assert cases[1].labels_available is False
    assert cases[1].gold_norms is None


def test_load_verified_accepts_public_release_and_skips_blind_eval(tmp_path: Path) -> None:
    root = _write_release(tmp_path / "release")

    cases = load_verified(root)

    assert [case.case_id for case in cases] == ["case_dev"]
    assert cases[0].gold_norms == ["44FZ:2026-08-04:art.1"]


def test_public_release_rejects_checksum_mismatch(tmp_path: Path) -> None:
    root = _write_release(tmp_path / "release")
    (root / "dev.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(PublicDatasetError, match="checksum mismatch"):
        validate_public_dataset(root)


def test_public_release_rejects_non_null_eval_label(tmp_path: Path) -> None:
    root = _write_release(tmp_path / "release")
    record = json.loads((root / "eval.jsonl").read_text(encoding="utf-8"))
    record["gold_norms"] = ["44FZ:2026-08-04:art.1"]
    (root / "eval.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    _write_checksums(
        root,
        (
            "DATA_LICENSE.md",
            "README.md",
            "dev.jsonl",
            "eval.jsonl",
            "privacy_audit.json",
            "dataset_manifest.json",
        ),
    )

    with pytest.raises(PublicDatasetError, match="hidden labels are not null"):
        validate_public_dataset(root)


def test_download_uses_hub_snapshot_and_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_release(tmp_path / "source")
    destination = tmp_path / "destination"
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        destination.mkdir(exist_ok=True)
        for path in source.iterdir():
            (destination / path.name).write_bytes(path.read_bytes())
        return str(destination)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    summary = download_public_dataset(
        destination,
        repo_id="owner/dataset",
        revision="v1.0.0",
    )

    assert summary.repo_id == "owner/dataset"
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["revision"] == "v1.0.0"
