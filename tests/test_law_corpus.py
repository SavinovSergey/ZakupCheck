"""Тесты парсинга NormUnit и структурного чанкинга (DESIGN §4.2)."""

from __future__ import annotations

import json
from pathlib import Path

from ingest.law.corpus import (
    build_norm_units,
    build_structural_chunks,
    build_corpus,
    make_norm_id,
    parse_articles,
    sliding_windows,
)
from ingest.law.download import html_to_blocks

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "law"


def _blocks_from_pairs(pairs: list[tuple[str, str]]) -> list[dict]:
    texts = [t for _, t in pairs]
    full = "\n\n".join(texts)
    blocks: list[dict] = []
    cursor = 0
    for i, (kind, text) in enumerate(pairs):
        start = cursor
        end = start + len(text)
        assert full[start:end] == text
        blocks.append(
            {
                "order": i,
                "block_id": None,
                "kind": kind,
                "text": text,
                "article_hint": None,
                "article_title_hint": None,
                "char_start": start,
                "char_end": end,
            }
        )
        cursor = end + (2 if i < len(pairs) - 1 else 0)
    return blocks


SAMPLE_PAIRS: list[tuple[str, str]] = [
    ("chapter_title", "Глава 1. Общие положения"),
    ("article_title", "Статья 33. Описание объекта закупки"),
    (
        "body",
        "1. Описание объекта закупки должно носить объективный характер:",
    ),
    ("body", "1) в описании указываются функциональные характеристики;"),
    ("body", "а) детали подпункта а;"),
    ("body", "б) детали подпункта б;"),
    ("body", "2) использование показателей допустимо;"),
    ("body", "2. Не допускается указание товарных знаков."),
]


def test_make_norm_id() -> None:
    assert make_norm_id("2026-08-04", "33") == "44FZ:2026-08-04:art.33"
    assert make_norm_id("2026-08-04", "33", "1") == "44FZ:2026-08-04:art.33:ch.1"
    assert (
        make_norm_id("2026-08-04", "24.1", "1", "2")
        == "44FZ:2026-08-04:art.24.1:ch.1:p.2"
    )


def test_parse_articles_parts_points_subpoints_variant_b() -> None:
    articles = parse_articles(_blocks_from_pairs(SAMPLE_PAIRS))
    assert len(articles) == 1
    art = articles[0]
    assert art.number == "33"
    assert art.title == "Описание объекта закупки"
    assert [p.number for p in art.parts] == ["1", "2"]

    part1 = art.parts[0]
    assert [pt.number for pt in part1.points] == ["1", "2"]
    # подпункты а)/б) влиты в пункт 1 (вариант B)
    assert "а) детали подпункта а" in part1.points[0].text
    assert "б) детали подпункта б" in part1.points[0].text
    assert art.parts[1].points == []


def test_build_norm_units_ids_and_levels() -> None:
    articles = parse_articles(_blocks_from_pairs(SAMPLE_PAIRS))
    units = build_norm_units("2026-08-04", articles)
    by_id = {u.norm_id: u for u in units}
    assert "44FZ:2026-08-04:art.33" in by_id
    assert by_id["44FZ:2026-08-04:art.33"].level == "article"
    assert by_id["44FZ:2026-08-04:art.33:ch.1"].level == "part"
    assert by_id["44FZ:2026-08-04:art.33:ch.1:p.1"].level == "point"
    assert by_id["44FZ:2026-08-04:art.33:ch.1"].is_retrieval_unit is True
    # подпункт не отдельная норма
    assert not any(":p.1:a" in u.norm_id or u.norm_id.endswith(":a") for u in units)


def test_chunk_short_part_keeps_part_level() -> None:
    articles = parse_articles(_blocks_from_pairs(SAMPLE_PAIRS))
    chunks = build_structural_chunks("2026-08-04", articles, max_chunk=2000)
    assert all(c.chunking == "part" for c in chunks)
    part_chunk = next(c for c in chunks if c.norm_ids[0].endswith(":ch.1"))
    assert part_chunk.chunking == "part"
    assert "44FZ:2026-08-04:art.33:ch.1" in part_chunk.norm_ids
    assert "44FZ:2026-08-04:art.33:ch.1:p.1" in part_chunk.norm_ids


def test_chunk_long_part_splits_to_points() -> None:
    long_point = "1) " + ("слово " * 400)
    pairs: list[tuple[str, str]] = [
        ("article_title", "Статья 99. Тест"),
        ("body", "1. Длинная часть:"),
        ("body", long_point.strip()),
        ("body", "2) короткий пункт."),
    ]
    articles = parse_articles(_blocks_from_pairs(pairs))
    # часть длиннее max_chunk → по пунктам
    chunks = build_structural_chunks("ed", articles, max_chunk=200)
    kinds = {c.chunking for c in chunks}
    assert "point" in kinds or "window" in kinds
    assert all(c.chunking != "part" for c in chunks)
    short = next(c for c in chunks if "короткий пункт" in c.text)
    assert short.chunking == "point"


def test_chunk_long_point_uses_windows() -> None:
    huge = "1) " + ("абзацтекст " * 300)
    pairs = [
        ("article_title", "Статья 98. Тест"),
        ("body", "1. Часть:"),
        ("body", huge),
    ]
    articles = parse_articles(_blocks_from_pairs(pairs))
    chunks = build_structural_chunks("ed", articles, max_chunk=100, window_overlap=20)
    assert all(c.chunking == "window" for c in chunks)
    assert len(chunks) >= 2
    assert all(c.norm_ids == ["44FZ:ed:art.98:ch.1:p.1"] for c in chunks)


def test_sliding_windows_overlap_and_bounds() -> None:
    text = " ".join(f"w{i}" for i in range(50))
    windows = sliding_windows(text, char_start=100, max_chunk=40, overlap=10)
    assert len(windows) >= 2
    for start, end, chunk in windows:
        assert start >= 100
        assert end == start + len(chunk)
        assert chunk in text


def test_build_corpus_from_html_fixture(tmp_path: Path) -> None:
    html = (FIXTURES / "mini_source.html").read_text(encoding="utf-8")
    full_text, blocks = html_to_blocks(html)
    edition_id = "2026-08-04"
    edition_dir = tmp_path / edition_id
    edition_dir.mkdir()
    (edition_dir / "edition.json").write_text(
        json.dumps(
            {
                "law_id": "44FZ",
                "edition_id": edition_id,
                "content_sha256": "x",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (edition_dir / "full_text.txt").write_text(full_text + "\n", encoding="utf-8")
    with (edition_dir / "blocks.jsonl").open("w", encoding="utf-8") as fh:
        for b in blocks:
            fh.write(json.dumps(b.__dict__, ensure_ascii=False) + "\n")

    stats = build_corpus(edition_dir, max_chunk=2000)
    assert stats["articles"] == 2
    assert (edition_dir / "norm_units.jsonl").exists()
    assert (edition_dir / "chunks_structural_m2000.jsonl").exists()
    assert (edition_dir / "chunks_structural.jsonl").exists()
    assert stats["chunking"]["part"] >= 1

    units = [
        json.loads(line)
        for line in (edition_dir / "norm_units.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(u["norm_id"].endswith("art.24.1") for u in units)
    # подпункты не отдельные units
    assert not any(u["level"] not in {"article", "part", "point"} for u in units)
