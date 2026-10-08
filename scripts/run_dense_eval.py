#!/usr/bin/env python3
"""Dense recall@k на verified-кейсах (DESIGN §7.1, режим D)."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from index.bm25 import Bm25Index, load_chunks  # noqa: E402
from index.dense import (  # noqa: E402
    DOCUMENT_PREFIX,
    MAX_TOKENS,
    QUERY_PREFIX,
    DenseIndex,
    SentenceTransformerEncoder,
)
from index.eval_tables import (  # noqa: E402
    chunks_path,
    collect_rows,
    evaluate_index,
    load_verified,
    markdown_table,
)

DEFAULT_MODEL = "sergeyzh/rubert-mini-uncased"
DEFAULT_MAX_CHUNKS = (300, 500, 700, 1000, 1500)
LONG_MAX_CHUNKS = (300, 500, 700, 1000, 1500, 2000, 2500)
# Share of chunks cut by the window. Above this, 1000/1500 rows are not comparable.
NOTABLE_TRUNCATION = 0.05


@dataclass(frozen=True)
class ModelProfile:
    query_prefix: str
    document_prefix: str
    max_tokens: int
    max_chunks: tuple[int, ...]


PROFILES: dict[str, ModelProfile] = {
    "Roflmax/bge-m3-russian-legal": ModelProfile(
        query_prefix="Represent this sentence for searching relevant passages: ",
        document_prefix="",
        max_tokens=512,
        max_chunks=(300, 500, 700, 1000, 1500),
    ),
    "deepvk/USER-bge-m3": ModelProfile(
        query_prefix="",
        document_prefix="",
        max_tokens=8192,
        max_chunks=LONG_MAX_CHUNKS,
    ),
    "deepvk/USER2-base": ModelProfile(
        query_prefix=QUERY_PREFIX,
        document_prefix=DOCUMENT_PREFIX,
        max_tokens=8192,
        max_chunks=LONG_MAX_CHUNKS,
    ),
    "Roflmax/e5-large-legal-ru": ModelProfile(
        query_prefix="query: ",
        document_prefix="passage: ",
        max_tokens=512,
        max_chunks=DEFAULT_MAX_CHUNKS,
    ),
    "alekseevpavel04/multilingual-e5-small-ru-law": ModelProfile(
        query_prefix="query: ",
        document_prefix="passage: ",
        max_tokens=512,
        max_chunks=DEFAULT_MAX_CHUNKS,
    ),
}


def profile_for(model: str) -> ModelProfile:
    return PROFILES.get(
        model,
        ModelProfile(QUERY_PREFIX, DOCUMENT_PREFIX, MAX_TOKENS, DEFAULT_MAX_CHUNKS),
    )


def default_report(model: str) -> Path:
    if model == DEFAULT_MODEL:
        return Path("evals/reports/dense_mini_uncased_by_max_chunk.md")
    slug = model.split("/")[-1]
    return Path(f"evals/reports/dense_{slug}_by_max_chunk.md")


def comparison_table(
    rows: list[tuple[int, str, str, int, float, float, float, float, float, float]],
    invalid_sizes: set[int],
) -> str:
    header = (
        "| max_chunk | query | split | n "
        "| dense R@5 art | dense R@10 art | bm25 R@5 art | bm25 R@10 art "
        "| dense ms/query | bm25 ms/query | сравнение |"
    )
    sep = "|" + "|".join(["---"] * 11) + "|"
    lines = [header, sep]
    for (
        max_chunk,
        variant,
        split,
        n,
        dense_r5,
        dense_r10,
        bm25_r5,
        bm25_r10,
        dense_ms,
        bm25_ms,
    ) in rows:
        mark = "невалидно" if max_chunk in invalid_sizes else "ок"
        lines.append(
            f"| {max_chunk} | {variant} | {split} | {n} "
            f"| {dense_r5:.3f} | {dense_r10:.3f} | {bm25_r5:.3f} | {bm25_r10:.3f} "
            f"| {dense_ms:.1f} | {bm25_ms:.1f} | {mark} |"
        )
    return "\n".join(lines)


def truncation_line(max_chunk: int, truncated: int, total: int) -> tuple[str, bool]:
    share = truncated / total if total else 0.0
    invalid = max_chunk in (1000, 1500) and share >= NOTABLE_TRUNCATION
    text = f"{max_chunk} — {truncated} из {total}"
    if invalid:
        text += " (невалидно для сравнения)"
    return text, invalid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dense recall@k по verified EvalCase")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--edition-dir", type=Path, default=Path("data/law/2026-08-04"))
    parser.add_argument("--cases-dir", type=Path, default=Path("evals/cases"))
    parser.add_argument(
        "--max-chunk",
        type=int,
        action="append",
        dest="max_chunks",
        help="Ограничить прогон этими размерами (можно несколько). Иначе длины из профиля модели.",
    )
    parser.add_argument("--query", choices=["raw", "blind", "both"], default="both")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    profile = profile_for(args.model)
    sizes = tuple(args.max_chunks) if args.max_chunks else profile.max_chunks
    variants = ("raw", "blind") if args.query == "both" else (args.query,)
    report_path = args.report if args.report is not None else default_report(args.model)

    cases = load_verified(args.cases_dir)
    if not cases:
        print(f"нет verified-кейсов в {args.cases_dir}", file=sys.stderr)
        return 1

    summary_rows = []
    topic_rows = []
    compared: list[tuple[int, str, str, int, float, float, float, float, float, float]] = []
    truncated_notes: list[str] = []
    invalid_sizes: set[int] = set()

    print(f"loading {args.model}", file=sys.stderr)
    encoder = SentenceTransformerEncoder(
        args.model,
        query_prefix=profile.query_prefix,
        document_prefix=profile.document_prefix,
        max_tokens=profile.max_tokens,
    )

    for max_chunk in sizes:
        path = chunks_path(args.edition_dir, max_chunk)
        if not path.is_file():
            print(
                f"нет {path}; соберите: python scripts/build_law_corpus.py "
                f"--edition-dir {args.edition_dir} --max-chunk {max_chunk}",
                file=sys.stderr,
            )
            return 1
        chunks = load_chunks(path)
        dense = DenseIndex.build(
            chunks,
            encoder,
            cache_dir=args.edition_dir / "dense",
            model_name=args.model,
            max_chunk=max_chunk,
        )
        bm25 = Bm25Index(chunks)
        note, invalid = truncation_line(max_chunk, dense.truncated, len(chunks))
        truncated_notes.append(note)
        if invalid:
            invalid_sizes.add(max_chunk)
        print(
            f"max_chunk={max_chunk} chunks={len(chunks)} truncated={dense.truncated}",
            file=sys.stderr,
        )
        for variant in variants:
            dense_grouped, dense_ms = evaluate_index(dense, cases, variant)
            bm25_grouped, bm25_ms = evaluate_index(bm25, cases, variant)
            dense_summary, dense_topics = collect_rows(dense_grouped, max_chunk, variant, dense_ms)
            bm25_summary, _ = collect_rows(bm25_grouped, max_chunk, variant, bm25_ms)
            summary_rows.extend(dense_summary)
            topic_rows.extend(dense_topics)
            bm25_by_key = {(row[2], row[3]): (row[4], row[5]) for row in bm25_summary}
            for row in dense_summary:
                metrics = row[4]
                other, other_ms = bm25_by_key[(row[2], row[3])]
                print(
                    f"  {variant} {row[2]} n={metrics[10].n} "
                    f"R@5 art={metrics[5].recall_article:.3f} "
                    f"R@10 art={metrics[10].recall_article:.3f} ms/query={row[5]:.1f}",
                    file=sys.stderr,
                )
                compared.append(
                    (
                        max_chunk,
                        variant,
                        row[2],
                        metrics[10].n,
                        metrics[5].recall_article,
                        metrics[10].recall_article,
                        other[5].recall_article,
                        other[10].recall_article,
                        row[5],
                        other_ms,
                    )
                )

    report = "\n".join(
        [
            "# Dense structural chunks",
            "",
            f"Модель: `{args.model}`.",
            f"Окно: {profile.max_tokens} токенов.",
            f"Префикс запроса: `{profile.query_prefix}`.",
            f"Префикс документа: `{profile.document_prefix}`.",
            "Запрос D: `complaint_argument_raw` / `complaint_argument_blind`.",
            "Hit: `gold_norms` ∩ объединение `norm_ids` top-k чанков.",
            "exact — буквальный id; art — совпадение `art.N`. all — доля кейсов, где найдено всё gold.",
            "R@5 и R@10 считаются по одному и тому же ранжированию: top-5 — это первые пять чанков из top-10.",
            "ms/query — среднее время одного search() на CPU после прогрева: "
            "кодирование запроса и отбор top-10. Построение индекса не входит.",
            "",
            f"Чанков длиннее {profile.max_tokens} токенов (с префиксом документа): "
            + ", ".join(truncated_notes)
            + ".",
            "Строка невалидна для сравнения, если на 1000 или 1500 обрезано не меньше 5% чанков.",
            "",
            "## dev / eval",
            "",
            markdown_table(summary_rows),
            "",
            "## по темам",
            "",
            markdown_table(topic_rows),
            "",
            "## сверка с BM25, R@5 и R@10 art",
            "",
            comparison_table(compared, invalid_sizes),
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
