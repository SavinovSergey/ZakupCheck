#!/usr/bin/env python3
"""Audit article status and retrieval coverage in a built law corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

AMENDING_LAW_RE = re.compile(r"от\s+(\d{2}\.\d{2}\.\d{4})\s+№\s*([\d.]+-ФЗ)")


def repeal_basis(status_text: str) -> str:
    matches = AMENDING_LAW_RE.findall(status_text)
    if not matches:
        raise ValueError(f"repeal note has no amending law: {status_text}")
    date, law = matches[-1]
    return f"{date} № {law}"


def audit(edition_dir: Path, sizes: list[int]) -> dict[str, Any]:
    edition = json.loads((edition_dir / "edition.json").read_text(encoding="utf-8"))
    full_text = (edition_dir / "full_text.txt").read_text(encoding="utf-8").removesuffix("\n")
    digest = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
    if digest != edition["content_sha256"]:
        raise ValueError("full_text hash differs from edition metadata")

    audits = {
        size: json.loads(
            (edition_dir / f"corpus_article_audit_m{size}.json").read_text(encoding="utf-8")
        )
        for size in sizes
    }
    reference = audits[sizes[0]]
    for size, item in audits.items():
        if item["active_articles"] != reference["active_articles"]:
            raise ValueError(f"m{size}: active article list differs")
        if item["repealed_articles"] != reference["repealed_articles"]:
            raise ValueError(f"m{size}: repealed article list differs")
        if item["empty_articles"]:
            raise ValueError(f"m{size}: empty articles remain: {item['empty_articles']}")
        if set(item["retrieval_articles"]) != set(reference["active_articles"]):
            raise ValueError(f"m{size}: retrieval coverage differs from active articles")

    repealed = []
    for item in reference["repealed_articles"]:
        note = item["status_text"]
        if note not in full_text:
            raise ValueError(f"article {item['article']}: status note absent from official text")
        repealed.append({**item, "basis": repeal_basis(note)})
    stale_dense_caches = 0
    for meta_path in (edition_dir / "dense").glob("*_m*.meta.json"):
        match = re.search(r"_m(\d+)\.meta\.json$", meta_path.name)
        if match is None or int(match.group(1)) not in sizes:
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stats = json.loads(
            (edition_dir / f"corpus_stats_m{match.group(1)}.json").read_text(encoding="utf-8")
        )
        stale_dense_caches += int(meta.get("rows") != stats["chunks"])
    return {
        "edition_id": edition["edition_id"],
        "source_url": edition["source_url"],
        "ips_rdk": edition.get("ips_rdk"),
        "content_sha256": digest,
        "sizes": sizes,
        "article_records": reference["article_records"],
        "active_articles": reference["active_articles"],
        "repealed_articles": repealed,
        "empty_articles": reference["empty_articles"],
        "active_article_only": reference["active_article_only"],
        "coverage_by_size": {
            str(size): len(audits[size]["retrieval_articles"]) for size in sizes
        },
        "stale_dense_caches": stale_dense_caches,
    }


def write_report(state: dict[str, Any], path: Path) -> None:
    repealed = state["repealed_articles"]
    by_basis: dict[str, list[str]] = {}
    for item in repealed:
        by_basis.setdefault(item["basis"], []).append(item["article"])
    lines = [
        "# Аудит покрытия статей 44-ФЗ",
        "",
        f"Редакция: `{state['edition_id']}`, ИПС rdk `{state['ips_rdk']}`. "
        f"Источник: `{state['source_url']}`.",
        "",
        "Проверка выполнена по сохранённому HTML официальной ИПС. SHA-256 повторно "
        f"нормализованного текста совпал с `edition.json`: `{state['content_sha256']}`.",
        "",
        "## Результат",
        "",
        f"Всего article-записей: {state['article_records']}; действующих: "
        f"{len(state['active_articles'])}; полностью утративших силу: {len(repealed)}; "
        f"пустых после исправления: {len(state['empty_articles'])}.",
        f"Во всех вариантах m{min(state['sizes'])}–m{max(state['sizes'])} retrieval-покрытие: "
        f"{len(state['active_articles'])} из {len(state['active_articles'])} действующих статей.",
        "",
        "Статьи без нумерованных частей, для которых теперь создаётся article-level чанк: "
        + ", ".join(state["active_article_only"])
        + ".",
        "",
        "## Полностью утратившие силу статьи",
        "",
    ]
    for basis, articles in sorted(by_basis.items()):
        lines.append(f"- {basis}: статьи {', '.join(articles)}.")
    lines.extend(
        [
            "",
            "Эти статьи сохраняются в `norm_units.jsonl` как article-level записи с "
            "`is_repealed=true`, но не включаются в retrieval-чанки.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit law corpus article coverage")
    parser.add_argument("--edition-dir", type=Path, default=Path("data/law/2026-08-04"))
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[300, 500, 700, 1000, 1500, 2000, 2500],
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evals/reports/law_corpus_audit.md"),
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("data/analysis/law_corpus_audit.json"),
    )
    args = parser.parse_args()
    state = audit(args.edition_dir, args.sizes)
    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.state.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(state, args.report)
    main_path = args.state.parent / "retrieval_error_state.json"
    if main_path.is_file():
        main = json.loads(main_path.read_text(encoding="utf-8"))
        main["law_corpus_audit"] = {**state, "report": str(args.report)}
        main_path.write_text(json.dumps(main, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        from run_retrieval_error_analysis import _report as write_aggregate_report

        write_aggregate_report(main, Path("evals/reports/retrieval_error_analysis.md"))
    print(f"wrote {args.report}")
    print(f"wrote {args.state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
