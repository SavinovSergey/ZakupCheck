"""Тесты скачивания/HTML→blocks без сети (DESIGN §4.2)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from ingest.law.download import (
    choose_redaction,
    decode_html,
    edition_id_for,
    html_to_blocks,
    normalize_text,
    parse_redactions,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "law"


def test_normalize_text_collapses_space_and_nbsp() -> None:
    assert normalize_text("a\u00a0 \tb\r\n\n\n\nc") == "a b\n\nc"


def test_decode_html_fallback() -> None:
    raw = "Статья".encode("windows-1251")
    assert "Статья" in decode_html(raw, "windows-1251")


def test_parse_redactions_and_choose() -> None:
    html = (FIXTURES / "mini_docbody.html").read_text(encoding="utf-8")
    redactions = parse_redactions(html)
    assert [r.rdk for r in redactions] == [156, 157]
    selected = choose_redaction(redactions, None)
    assert selected.rdk == 157
    assert selected.amending_date == date(2026, 8, 4)
    assert selected.amending_law == "330-ФЗ"
    assert edition_id_for(selected) == "2026-08-04"
    assert choose_redaction(redactions, 156).rdk == 156
    with pytest.raises(RuntimeError, match="rdk=999"):
        choose_redaction(redactions, 999)


def test_html_to_blocks_kinds_w9_and_offsets() -> None:
    html = (FIXTURES / "mini_source.html").read_text(encoding="utf-8")
    full_text, blocks = html_to_blocks(html)

    kinds = [b.kind for b in blocks]
    assert kinds[0] == "chapter_title"
    assert kinds[1] == "article_title"
    assert "Статья 24.1." in next(b.text for b in blocks if b.kind == "article_title" and "24" in b.text)

    art_241 = next(b for b in blocks if b.kind == "article_title" and "24.1" in b.text)
    assert art_241.article_hint == 24
    assert art_241.article_title_hint == "Особенности участия"

    for b in blocks:
        assert full_text[b.char_start : b.char_end] == b.text

    # склейка блоков — через \n\n
    assert "\n\n".join(b.text for b in blocks) == full_text
    assert any(b.text.startswith("1)") for b in blocks if b.kind == "body")
    assert any("утратила силу" in b.text for b in blocks)
