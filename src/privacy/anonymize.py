"""Deterministic redaction primitives for public eval exports.

The rules intentionally favor precision for unlabelled prose. They cover
structured identifiers and common Russian person-name forms, while a separate
manual-redaction file handles entities found during human review. This module
does not claim that regex matching alone is sufficient for publication.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ALLOWED_KINDS = frozenset(
    {
        "ADDRESS",
        "BANK_ACCOUNT",
        "BIRTH_DATE",
        "EMAIL",
        "IDENTIFIER",
        "PASSPORT",
        "PERSON",
        "PHONE",
        "SNILS",
        "TAX_ID",
    }
)


@dataclass(frozen=True)
class Detection:
    """A sensitive span found in one text value."""

    kind: str
    start: int
    end: int
    value: str
    rule: str


@dataclass(frozen=True)
class ManualRedaction:
    """An exact value approved for replacement in all or selected cases."""

    text: str
    kind: str
    ignore_case: bool = False


@dataclass(frozen=True)
class AnonymizationResult:
    fields: dict[str, str]
    detections: dict[str, list[Detection]]
    replacements: dict[str, str]


@dataclass(frozen=True)
class _Rule:
    name: str
    kind: str
    regex: re.Pattern[str]


_NAME_WORD = r"(?:[А-ЯЁ][а-яё-]{1,}|[А-ЯЁ]{2,}(?:-[А-ЯЁ]{2,})?)"
_PATRONYMIC = (
    r"(?:[А-ЯЁ][а-яё-]*(?:(?:(?:ов|ев)?ич)(?:а|у|ем|е)?|"
    r"(?:(?:ов|ев|ич)н)(?:а|ы|е|у|ой|ою))|"
    r"[А-ЯЁ-]*(?:(?:(?:ОВ|ЕВ)?ИЧ)(?:А|У|ЕМ|Е)?|"
    r"(?:(?:ОВ|ЕВ|ИЧ)Н)(?:А|Ы|Е|У|ОЙ|ОЮ)))"
)


def _rule(name: str, kind: str, pattern: str, flags: int = 0) -> _Rule:
    return _Rule(name=name, kind=kind, regex=re.compile(pattern, flags))


_RULES: tuple[_Rule, ...] = (
    _rule(
        "email",
        "EMAIL",
        r"(?<![\w.+-])(?P<value>[\w.+-]+@[\w-]+(?:\.[\w-]+)+)(?![\w-])",
        re.UNICODE,
    ),
    _rule(
        "snils",
        "SNILS",
        r"(?<!\d)(?P<value>\d{3}[- ]\d{3}[- ]\d{3}[ -]\d{2})(?!\d)",
    ),
    _rule(
        "phone_ru",
        "PHONE",
        r"(?<!\d)(?P<value>(?:\+?7|8)(?:[\s()\-]*\d){10})(?!\d)",
    ),
    _rule(
        "passport_label",
        "PASSPORT",
        r"\b(?:паспорт|серия\s+и\s+номер\s+паспорта)\s*[:№-]?\s*"
        r"(?P<value>\d{2}\s*\d{2}\s*\d{6})\b",
        re.IGNORECASE | re.UNICODE,
    ),
    _rule(
        "tax_id_label",
        "TAX_ID",
        r"\b(?:ИНН|КПП|ОГРН(?:ИП)?)\s*[:№-]?\s*(?P<value>\d{9,15})\b",
        re.IGNORECASE | re.UNICODE,
    ),
    _rule(
        "bank_account_label",
        "BANK_ACCOUNT",
        r"\b(?:р/?с|к/?с|расч[её]тн(?:ый|ого)\s+сч[её]т|"
        r"корреспондентск(?:ий|ого)\s+сч[её]т|БИК)\s*[:№-]?\s*"
        r"(?P<value>\d{9,20})\b",
        re.IGNORECASE | re.UNICODE,
    ),
    _rule(
        "birth_date_label",
        "BIRTH_DATE",
        r"\b(?:дата\s+рождения|родил(?:ся|ась))\s*[:,-]?\s*"
        r"(?P<value>\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b",
        re.IGNORECASE | re.UNICODE,
    ),
    _rule(
        "address_label",
        "ADDRESS",
        r"\b(?:адрес\s+(?:регистрации|места\s+жительства|проживания)|"
        r"место\s+жительства)\s*[:—-]\s*"
        r"(?P<value>[^\n;]{5,180})",
        re.IGNORECASE | re.UNICODE,
    ),
    _rule(
        "person_surname_initials",
        "PERSON",
        rf"(?<![А-ЯЁа-яё])(?P<value>{_NAME_WORD}\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.)",
        re.UNICODE,
    ),
    _rule(
        "person_initials_surname",
        "PERSON",
        rf"(?<![А-ЯЁа-яё])(?P<value>[А-ЯЁ]\.\s*[А-ЯЁ]\.\s*{_NAME_WORD})",
        re.UNICODE,
    ),
    _rule(
        "person_full_patronymic",
        "PERSON",
        rf"(?<![А-ЯЁа-яё])(?P<value>{_NAME_WORD}\s+{_NAME_WORD}\s+{_PATRONYMIC})"
        rf"(?![А-ЯЁа-яё])",
        re.UNICODE,
    ),
    _rule(
        "person_label",
        "PERSON",
        rf"\b(?i:Ф\.?\s*И\.?\s*О\.?|"
        rf"индивидуальн(?:ый|ого)\s+предпринимател(?:ь|я)|"
        rf"представител(?:ь|я)|заявител(?:ь|я))\s*[:—-]?\s*"
        rf"(?P<value>{_NAME_WORD}(?:\s+{_NAME_WORD}){{1,2}})",
        re.UNICODE,
    ),
)


def fingerprint(value: str) -> str:
    """Return a non-reversible short identifier suitable for audit reports."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _manual_rules(redactions: Iterable[ManualRedaction]) -> list[_Rule]:
    rules: list[_Rule] = []
    for index, item in enumerate(redactions):
        if not item.text.strip():
            raise ValueError("manual redaction text must not be empty")
        if item.kind not in ALLOWED_KINDS:
            raise ValueError(f"unsupported manual redaction kind: {item.kind}")
        flags = re.IGNORECASE | re.UNICODE if item.ignore_case else re.UNICODE
        rules.append(
            _rule(
                f"manual_{index}",
                item.kind,
                rf"(?P<value>{re.escape(item.text)})",
                flags,
            )
        )
    return rules


def _select_non_overlapping(candidates: Sequence[Detection]) -> list[Detection]:
    # At the same start, retain the widest match. Manual rules are evaluated
    # first by scan_text and therefore win exact ties.
    ordered = sorted(candidates, key=lambda item: (item.start, -(item.end - item.start)))
    selected: list[Detection] = []
    cursor = -1
    for item in ordered:
        if item.start < cursor:
            continue
        selected.append(item)
        cursor = item.end
    return selected


def scan_text(
    text: str,
    *,
    manual_redactions: Iterable[ManualRedaction] = (),
) -> list[Detection]:
    """Find automatically detectable and manually supplied sensitive spans."""

    candidates: list[Detection] = []
    rules = [*_manual_rules(manual_redactions), *_RULES]
    for rule in rules:
        for match in rule.regex.finditer(text):
            start, end = match.span("value")
            candidates.append(
                Detection(
                    kind=rule.kind,
                    start=start,
                    end=end,
                    value=match.group("value"),
                    rule=rule.name,
                )
            )
    return _select_non_overlapping(candidates)


def anonymize_fields(
    fields: Mapping[str, str],
    *,
    manual_redactions: Iterable[ManualRedaction] = (),
) -> AnonymizationResult:
    """Redact a case's fields with stable per-case placeholders."""

    manual = tuple(manual_redactions)
    entity_numbers: dict[tuple[str, str], int] = {}
    next_number: dict[str, int] = {}
    rendered: dict[str, str] = {}
    all_detections: dict[str, list[Detection]] = {}
    replacement_map: dict[str, str] = {}

    for field, text in fields.items():
        detections = scan_text(text, manual_redactions=manual)
        all_detections[field] = detections
        chunks: list[str] = []
        cursor = 0
        for detection in detections:
            chunks.append(text[cursor : detection.start])
            normalized = detection.value.casefold().strip()
            entity_key = (detection.kind, normalized)
            if entity_key not in entity_numbers:
                number = next_number.get(detection.kind, 0) + 1
                next_number[detection.kind] = number
                entity_numbers[entity_key] = number
            placeholder = f"<{detection.kind}_{entity_numbers[entity_key]}>"
            chunks.append(placeholder)
            replacement_map[fingerprint(detection.value)] = placeholder
            cursor = detection.end
        chunks.append(text[cursor:])
        rendered[field] = "".join(chunks)

    return AnonymizationResult(
        fields=rendered,
        detections=all_detections,
        replacements=replacement_map,
    )


def _parse_redaction(item: object, *, location: str) -> ManualRedaction:
    if not isinstance(item, dict):
        raise ValueError(f"{location} must be an object")
    text = item.get("text")
    kind = item.get("kind")
    ignore_case = item.get("ignore_case", False)
    if not isinstance(text, str) or not isinstance(kind, str):
        raise ValueError(f"{location} requires string text and kind")
    if not isinstance(ignore_case, bool):
        raise ValueError(f"{location}.ignore_case must be boolean")
    if kind not in ALLOWED_KINDS:
        raise ValueError(f"unsupported manual redaction kind: {kind}")
    return ManualRedaction(text=text, kind=kind, ignore_case=ignore_case)


def load_manual_redactions(
    path: Path | None,
) -> tuple[list[ManualRedaction], dict[str, list[ManualRedaction]]]:
    """Load global and per-case exact replacements from JSON."""

    if path is None:
        return [], {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manual redactions root must be an object")
    unknown = set(payload) - {"global", "cases"}
    if unknown:
        raise ValueError(f"unknown manual redaction keys: {sorted(unknown)}")

    global_items = payload.get("global", [])
    case_items = payload.get("cases", {})
    if not isinstance(global_items, list) or not isinstance(case_items, dict):
        raise ValueError("global must be a list and cases must be an object")

    global_redactions = [
        _parse_redaction(item, location=f"global[{index}]")
        for index, item in enumerate(global_items)
    ]
    by_case: dict[str, list[ManualRedaction]] = {}
    for case_id, items in case_items.items():
        if not isinstance(case_id, str) or not isinstance(items, list):
            raise ValueError("each cases entry must map a string id to a list")
        by_case[case_id] = [
            _parse_redaction(item, location=f"cases.{case_id}[{index}]")
            for index, item in enumerate(items)
        ]
    return global_redactions, by_case
