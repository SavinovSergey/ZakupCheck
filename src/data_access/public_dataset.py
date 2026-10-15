"""Download and validate the public ZakupCheck benchmark from Hugging Face."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError


DEFAULT_HF_DATASET_ID = "SergeySavinov/zakupcheck-44fz-retrieval-benchmark"
DEFAULT_HF_REVISION = "v1.0.0"
PUBLIC_FILES = (
    "DATA_LICENSE.md",
    "README.md",
    "SHA256SUMS",
    "dataset_manifest.json",
    "dev.jsonl",
    "eval.jsonl",
    "privacy_audit.json",
)
LABEL_FIELDS = (
    "gold_evidence",
    "gold_norms",
    "issue_type",
    "outcome",
    "outcome_evidence_quote",
)


class PublicDatasetError(ValueError):
    """Raised when a public benchmark snapshot is absent or inconsistent."""


class PublicSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    procurement_id: str | None = None
    procurement_url: str | None = None
    notice_date: str | None = None
    fas_decision_id: str | None = None
    fas_decision_url: str | None = None


class PublicGoldEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    norm_id: str
    decision_quote: str


class PublicRetrievalCase(BaseModel):
    """Publishable case schema shared by labeled dev and blind eval."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    case_id: str
    case_group_id: str
    edition_id: str
    split: Literal["dev", "eval"]
    topic: str
    complaint_argument_raw: str
    complaint_argument_blind: str
    complaint_argument_summary: str
    doc_fragment: str
    source: PublicSource
    gold_evidence: list[PublicGoldEvidence] | None = None
    gold_norms: list[str] | None = None
    issue_type: str | None = None
    outcome: Literal["upheld", "rejected"] | None = None
    outcome_evidence_quote: str | None = None

    @property
    def labels_available(self) -> bool:
        """Whether the case can be used for local retrieval metrics."""

        return bool(self.gold_norms)


@dataclass(frozen=True)
class PublicDatasetSummary:
    root: Path
    repo_id: str
    version: str
    dev_count: int
    eval_count: int
    eval_labels_included: bool


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicDatasetError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PublicDatasetError(f"expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PublicDatasetError(f"cannot read {path}: {exc}") from exc

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PublicDatasetError(
                f"{path}:{line_number}: invalid JSON: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise PublicDatasetError(f"{path}:{line_number}: expected a JSON object")
        records.append(value)
    return records


def verify_checksums(root: Path) -> None:
    """Verify every file listed by the release SHA256SUMS manifest."""

    checksum_path = root / "SHA256SUMS"
    if not checksum_path.is_file():
        raise PublicDatasetError(f"missing checksum manifest: {checksum_path}")
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PublicDatasetError(f"cannot read {checksum_path}: {exc}") from exc

    if not lines:
        raise PublicDatasetError("SHA256SUMS is empty")
    for line_number, line in enumerate(lines, start=1):
        try:
            expected, filename = line.split(maxsplit=1)
        except ValueError as exc:
            raise PublicDatasetError(
                f"SHA256SUMS:{line_number}: expected '<sha256>  <filename>'"
            ) from exc
        filename = filename.lstrip("*").strip()
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise PublicDatasetError(f"unsafe path in SHA256SUMS: {filename}")
        path = root / relative
        if not path.is_file():
            raise PublicDatasetError(f"checksummed file is missing: {filename}")
        actual = _sha256_file(path)
        if actual != expected.lower():
            raise PublicDatasetError(
                f"checksum mismatch for {filename}: {actual} != {expected.lower()}"
            )


def load_public_cases(
    dataset_dir: Path,
    *,
    split: Literal["dev", "eval"] | None = None,
) -> list[PublicRetrievalCase]:
    """Load public JSONL records from a downloaded or assembled release."""

    splits = (split,) if split else ("dev", "eval")
    cases: list[PublicRetrievalCase] = []
    ids: set[str] = set()
    for current_split in splits:
        path = dataset_dir / f"{current_split}.jsonl"
        if not path.is_file():
            raise PublicDatasetError(f"missing public split: {path}")
        for line_number, record in enumerate(_read_jsonl(path), start=1):
            try:
                case = PublicRetrievalCase.model_validate(record)
            except ValidationError as exc:
                raise PublicDatasetError(f"{path}:{line_number}: {exc}") from exc
            if case.split != current_split:
                raise PublicDatasetError(
                    f"{path}:{line_number}: split={case.split!r}, expected {current_split!r}"
                )
            if case.case_id in ids:
                raise PublicDatasetError(f"duplicate public case_id: {case.case_id}")
            ids.add(case.case_id)
            cases.append(case)
    return cases


def validate_public_dataset(
    dataset_dir: Path,
    *,
    repo_id: str = DEFAULT_HF_DATASET_ID,
) -> PublicDatasetSummary:
    """Validate checksums, manifest counts, schemas, and hidden eval labels."""

    missing = [name for name in PUBLIC_FILES if not (dataset_dir / name).is_file()]
    if missing:
        raise PublicDatasetError(f"public dataset is missing files: {missing}")
    verify_checksums(dataset_dir)

    manifest = _read_json_object(dataset_dir / "dataset_manifest.json")
    if manifest.get("publication_ready") is not True:
        raise PublicDatasetError("dataset manifest is not publication-ready")
    cases = load_public_cases(dataset_dir)
    by_split = {
        split: [case for case in cases if case.split == split]
        for split in ("dev", "eval")
    }
    actual_counts = {split: len(items) for split, items in by_split.items()}
    if manifest.get("counts") != actual_counts:
        raise PublicDatasetError(
            f"manifest counts do not match data: {manifest.get('counts')} != {actual_counts}"
        )
    if any(not case.labels_available for case in by_split["dev"]):
        raise PublicDatasetError("dev contains a case without gold_norms")

    eval_labels_included = manifest.get("eval_labels_included") is True
    if not eval_labels_included:
        for record_number, record in enumerate(
            _read_jsonl(dataset_dir / "eval.jsonl"), start=1
        ):
            leaked = [field for field in LABEL_FIELDS if record.get(field) is not None]
            if leaked:
                raise PublicDatasetError(
                    f"eval.jsonl:{record_number}: hidden labels are not null: {leaked}"
                )

    raw_records = [
        *_read_jsonl(dataset_dir / "dev.jsonl"),
        *_read_jsonl(dataset_dir / "eval.jsonl"),
    ]
    schemas = {tuple(sorted(record)) for record in raw_records}
    if len(schemas) != 1:
        raise PublicDatasetError("dev and eval records do not have one compatible schema")

    return PublicDatasetSummary(
        root=dataset_dir,
        repo_id=repo_id,
        version=str(manifest.get("version", "unknown")),
        dev_count=actual_counts["dev"],
        eval_count=actual_counts["eval"],
        eval_labels_included=eval_labels_included,
    )


def download_public_dataset(
    destination: Path,
    *,
    repo_id: str = DEFAULT_HF_DATASET_ID,
    revision: str = DEFAULT_HF_REVISION,
    force_download: bool = False,
    local_files_only: bool = False,
) -> PublicDatasetSummary:
    """Download one immutable Hub snapshot and validate it locally."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise PublicDatasetError(
            "Hugging Face support is not installed; run: pip install -e '.[data]'"
        ) from exc

    destination.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=destination,
            allow_patterns=list(PUBLIC_FILES),
            force_download=force_download,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        raise PublicDatasetError(
            f"failed to download {repo_id}@{revision}: {exc}"
        ) from exc
    return validate_public_dataset(destination, repo_id=repo_id)
