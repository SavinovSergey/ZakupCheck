from __future__ import annotations

import json

import pytest

from privacy.anonymize import (
    ManualRedaction,
    anonymize_fields,
    load_manual_redactions,
    scan_text,
)


def test_structured_pii_and_names_are_redacted_deterministically() -> None:
    fields = {
        "query": (
            "Представитель Иванов Иван Иванович, телефон +7 (999) 123-45-67, "
            "email test@example.org. Иванов И. И. поддержал довод."
        ),
        "evidence": "Повторный адрес: test@example.org.",
    }

    result = anonymize_fields(fields)

    assert "Иванов Иван Иванович" not in result.fields["query"]
    assert "+7 (999) 123-45-67" not in result.fields["query"]
    assert "test@example.org" not in result.fields["query"]
    assert result.fields["query"].count("<EMAIL_1>") == 1
    assert result.fields["evidence"].count("<EMAIL_1>") == 1
    assert "<PHONE_1>" in result.fields["query"]
    assert "<PERSON_1>" in result.fields["query"]
    assert "<PERSON_2>" in result.fields["query"]


def test_manual_redaction_handles_entity_missed_by_automatic_rules() -> None:
    manual = [ManualRedaction(text="сергей савинов", kind="PERSON", ignore_case=True)]

    detections = scan_text("Автор: Сергей Савинов", manual_redactions=manual)
    result = anonymize_fields(
        {"text": "Автор: Сергей Савинов"},
        manual_redactions=manual,
    )

    assert [(item.kind, item.rule) for item in detections] == [("PERSON", "manual_0")]
    assert result.fields["text"] == "Автор: <PERSON_1>"


@pytest.mark.parametrize(
    "full_name",
    [
        "Иванов Иван Иванович",
        "Иванова Ивана Ивановича",
        "Иванову Ивану Ивановичу",
        "Ивановым Иваном Ивановичем",
        "Иванове Иване Ивановиче",
        "Петрова Анна Сергеевна",
        "Петровой Анны Сергеевны",
        "Петрову Анну Сергеевну",
        "Петровой Анной Сергеевной",
        "Петровой Анной Сергеевною",
    ],
)
def test_person_full_name_is_redacted_in_russian_cases(full_name: str) -> None:
    result = anonymize_fields({"text": f"Жалобу ИП {full_name} признать обоснованной."})

    assert result.fields["text"] == "Жалобу ИП <PERSON_1> признать обоснованной."


def test_tax_and_passport_values_are_redacted_but_labels_remain() -> None:
    text = "ИНН 123456789012; паспорт 45 01 123456; СНИЛС 123-456-789 01"

    result = anonymize_fields({"text": text})

    assert result.fields["text"] == (
        "ИНН <TAX_ID_1>; паспорт <PASSPORT_1>; СНИЛС <SNILS_1>"
    )


def test_manual_redaction_config_is_validated(tmp_path) -> None:
    path = tmp_path / "redactions.json"
    path.write_text(
        json.dumps(
            {
                "global": [{"text": "Secret", "kind": "PERSON"}],
                "cases": {
                    "case-1": [
                        {"text": "hidden@example.org", "kind": "EMAIL"}
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    global_items, case_items = load_manual_redactions(path)

    assert global_items == [ManualRedaction(text="Secret", kind="PERSON")]
    assert case_items["case-1"] == [
        ManualRedaction(text="hidden@example.org", kind="EMAIL")
    ]


def test_manual_redaction_rejects_unknown_kind(tmp_path) -> None:
    path = tmp_path / "redactions.json"
    path.write_text(
        json.dumps({"global": [{"text": "Secret", "kind": "UNKNOWN"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported manual redaction kind"):
        load_manual_redactions(path)
