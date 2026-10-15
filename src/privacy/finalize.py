"""Validation and release assembly for a reviewed public benchmark candidate."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from privacy.anonymize import scan_text


SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")
HF_LICENSE_RE = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
FORBIDDEN_KEYS = frozenset(
    {
        "char_end",
        "char_start",
        "doc_locator",
        "doc_paths",
        "path",
    }
)
PRIVATE_PATH_MARKERS = (
    "data/raw/",
    "evals/cases/",
    "/home/",
    "\\Users\\",
)
EVAL_LABEL_FIELDS = (
    "gold_evidence",
    "gold_norms",
    "issue_type",
    "outcome",
    "outcome_evidence_quote",
)
EVAL_LABEL_KEYS = frozenset(EVAL_LABEL_FIELDS)
REQUIRED_TEXT_FIELDS = (
    "complaint_argument_raw",
    "complaint_argument_blind",
    "complaint_argument_summary",
    "doc_fragment",
)


class FinalizationError(ValueError):
    """Raised when a candidate cannot safely be marked publication-ready."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FinalizationError(f"expected an object in {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FinalizationError(f"cannot read {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FinalizationError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise FinalizationError(f"{path}:{line_number}: expected an object")
        records.append(record)
    return records


def _walk(value: Any, location: str = "record") -> Iterable[tuple[str, str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            yield child_location, key, child
            yield from _walk(child, child_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{location}[{index}]")


def _validate_record(record: dict[str, Any], *, split: str, index: int) -> list[str]:
    prefix = f"{split}[{index}]"
    errors: list[str] = []
    if record.get("schema_version") != 1:
        errors.append(f"{prefix}: schema_version must be 1")
    if record.get("split") != split:
        errors.append(f"{prefix}: split field must be {split!r}")
    for key in ("case_id", "case_group_id", "edition_id", "topic"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            errors.append(f"{prefix}: missing non-empty {key}")
    for field in REQUIRED_TEXT_FIELDS:
        text = record.get(field)
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{prefix}: missing non-empty {field}")
        elif detections := scan_text(text):
            kinds = sorted({item.kind for item in detections})
            errors.append(f"{prefix}.{field}: known PII patterns remain: {kinds}")

    if split == "dev":
        for key in ("gold_norms", "gold_evidence", "issue_type", "outcome"):
            if key not in record:
                errors.append(f"{prefix}: dev record is missing label {key}")
    else:
        leaked = sorted(EVAL_LABEL_KEYS.intersection(record))
        if leaked:
            errors.append(f"{prefix}: eval labels are present: {leaked}")

    for location, key, value in _walk(record, prefix):
        if key in FORBIDDEN_KEYS:
            errors.append(f"{location}: forbidden private provenance key")
        if isinstance(value, str) and any(marker in value for marker in PRIVATE_PATH_MARKERS):
            errors.append(f"{location}: contains a private/local path marker")
    return errors


def validate_candidate(input_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Validate the candidate and return records plus its manifest."""

    required = {
        "dev.jsonl",
        "eval.jsonl",
        "dataset_manifest.json",
        "privacy_audit.json",
    }
    missing = sorted(name for name in required if not (input_dir / name).is_file())
    if missing:
        raise FinalizationError(f"candidate is missing files: {missing}")

    manifest = _read_json(input_dir / "dataset_manifest.json")
    audit = _read_json(input_dir / "privacy_audit.json")
    errors: list[str] = []
    if manifest.get("publication_ready") is not False:
        errors.append("candidate manifest must have publication_ready=false")
    if manifest.get("source_cases_modified") is not False:
        errors.append("candidate must state source_cases_modified=false")
    if manifest.get("eval_labels_included") is not False:
        errors.append("candidate must not include eval labels")
    if audit.get("contains_raw_values") is not False:
        errors.append("privacy audit must state contains_raw_values=false")

    records = {
        split: _read_jsonl(input_dir / f"{split}.jsonl")
        for split in ("dev", "eval")
    }
    expected_counts = manifest.get("counts")
    actual_counts = {split: len(items) for split, items in records.items()}
    if expected_counts != actual_counts:
        errors.append(
            f"candidate counts do not match manifest: {actual_counts} != {expected_counts}"
        )

    ids: list[str] = []
    for split, items in records.items():
        for index, record in enumerate(items):
            errors.extend(_validate_record(record, split=split, index=index))
            case_id = record.get("case_id")
            if isinstance(case_id, str):
                ids.append(case_id)
    duplicates = sorted(case_id for case_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate public case ids: {duplicates}")
    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:30])
        suffix = "" if len(errors) <= 30 else f"\n- ... and {len(errors) - 30} more"
        raise FinalizationError(f"candidate validation failed:\n{preview}{suffix}")
    return records, manifest


def _dataset_card(*, version: str, hf_license: str, counts: dict[str, int]) -> str:
    return f"""---
language:
- ru
license: {hf_license}
task_categories:
- text-retrieval
pretty_name: ZakupCheck Retrieval Benchmark
configs:
- config_name: default
  data_files:
  - split: dev
    path: dev.jsonl
  - split: eval
    path: eval.jsonl
---

# ZakupCheck Retrieval Benchmark

Версия `{version}` содержит обезличенные запросы по доводам жалоб в сфере
государственных закупок и предназначена для оценки поиска норм 44-ФЗ.

## Состав

- `dev.jsonl` — {counts['dev']} кейсов с gold-разметкой;
- `eval.jsonl` — {counts['eval']} кейсов без публичных gold-меток;
- `dataset_manifest.json` — версия, privacy review и контрольные суммы;
- `privacy_audit.json` — аудит замен без исходных значений;
- `DATA_LICENSE.md` — условия использования данных.

Сырые жалобы, решения ФАС, извещения и локальные пути в набор не входят.
Идентификаторы внутренних кейсов псевдонимизированы. ФИО и структурированные
персональные данные заменены типизированными маркерами.

## Ограничения

Набор мал и не репрезентирует все виды закупок и нарушений. Разметка является
исследовательской и не представляет юридическое заключение. Публичный `eval`
не содержит эталонных норм; они должны храниться отдельно от датасета.
Поля разметки в `eval` присутствуют только для совместимости схемы и
всегда имеют значение `null`.
"""


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _public_eval_records(
    records: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Add null label placeholders so Hub splits expose one compatible schema."""

    dev_fields = {key for record in records["dev"] for key in record}
    placeholder_fields = [
        field for field in EVAL_LABEL_FIELDS if field in dev_fields
    ]
    return [
        {**record, **{field: None for field in placeholder_fields}}
        for record in records["eval"]
    ]


def finalize_dataset(
    *,
    input_dir: Path,
    output_dir: Path,
    version: str,
    reviewed_by: str,
    review_date: str,
    license_file: Path,
    hf_license: str,
) -> Path:
    """Validate and atomically assemble one immutable release directory."""

    if not SEMVER_RE.fullmatch(version):
        raise FinalizationError("version must be semantic, for example 1.0.0")
    if not reviewed_by.strip():
        raise FinalizationError("reviewed_by must not be empty")
    try:
        date.fromisoformat(review_date)
    except ValueError as exc:
        raise FinalizationError("review_date must use YYYY-MM-DD") from exc
    if not HF_LICENSE_RE.fullmatch(hf_license):
        raise FinalizationError("hf_license contains unsupported characters")
    if not license_file.is_file() or not license_file.read_text(encoding="utf-8").strip():
        raise FinalizationError("license_file must be an existing non-empty text file")
    if output_dir.exists():
        raise FinalizationError(f"refusing to overwrite existing output: {output_dir}")

    records, candidate_manifest = validate_candidate(input_dir)
    counts = {split: len(items) for split, items in records.items()}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary) / output_dir.name
        staging.mkdir()
        for name in ("dev.jsonl", "privacy_audit.json"):
            shutil.copyfile(input_dir / name, staging / name)
        _write_jsonl(staging / "eval.jsonl", _public_eval_records(records))
        shutil.copyfile(license_file, staging / "DATA_LICENSE.md")
        (staging / "README.md").write_text(
            _dataset_card(version=version, hf_license=hf_license, counts=counts),
            encoding="utf-8",
        )

        payload_names = (
            "DATA_LICENSE.md",
            "README.md",
            "dev.jsonl",
            "eval.jsonl",
            "privacy_audit.json",
        )
        file_metadata = {
            name: {
                "sha256": sha256_file(staging / name),
                "bytes": (staging / name).stat().st_size,
            }
            for name in payload_names
        }
        release_manifest = {
            "dataset": "zakupcheck-benchmark",
            "version": version,
            "schema_version": 1,
            "publication_ready": True,
            "privacy_review": {
                "status": "completed",
                "reviewed_by": reviewed_by,
                "review_date": review_date,
            },
            "eval_labels_included": False,
            "counts": counts,
            "hf_license": hf_license,
            "source_candidate_manifest_sha256": sha256_file(
                input_dir / "dataset_manifest.json"
            ),
            "source_candidate_version": candidate_manifest.get("version"),
            "files": file_metadata,
        }
        (staging / "dataset_manifest.json").write_text(
            json.dumps(release_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        checksum_names = (*payload_names, "dataset_manifest.json")
        (staging / "SHA256SUMS").write_text(
            "".join(
                f"{sha256_file(staging / name)}  {name}\n" for name in checksum_names
            ),
            encoding="utf-8",
        )
        os.replace(staging, output_dir)
    return output_dir
