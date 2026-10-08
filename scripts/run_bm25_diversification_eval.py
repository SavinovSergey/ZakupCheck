#!/usr/bin/env python3
"""Evaluate article diversification for every available BM25 chunk size.

The default ``dev`` phase compares all structural corpora using the blind
query and locks one max_chunk.  The explicit ``eval`` phase evaluates only the
locked baseline and its diversified ranking.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from index.bm25 import Bm25Index, load_chunks  # noqa: E402
from index.error_analysis import (  # noqa: E402
    aggregate_rankings,
    paired_group_bootstrap,
    per_case_article_recall,
)
from index.eval_tables import chunks_path, load_verified, query_text  # noqa: E402
from index.fusion import article_diversify  # noqa: E402
from index.recall import AggregateRecall  # noqa: E402
from schemas.corpus import Chunk  # noqa: E402
from schemas.eval_case import EvalCase  # noqa: E402

POOL = 50
LIMIT = 10
SEED = 20261008
STATE_VERSION = 1
_CHUNKS_RE = re.compile(r"chunks_structural_m(\d+)\.jsonl$")


@dataclass(frozen=True)
class Bm25Result:
    max_chunk: int
    baseline: AggregateRecall
    diversified: AggregateRecall
    latency_ms: float
    baseline_rankings: list[list[Chunk]]
    diversified_rankings: list[list[Chunk]]


def discover_chunk_sizes(edition_dir: Path) -> list[int]:
    sizes = []
    for path in edition_dir.glob("chunks_structural_m*.jsonl"):
        match = _CHUNKS_RE.fullmatch(path.name)
        if match is not None:
            sizes.append(int(match.group(1)))
    return sorted(set(sizes))


def _rank_cases(index: Bm25Index, cases: list[EvalCase]) -> tuple[list[list[Chunk]], float]:
    if cases:
        index.search(query_text(cases[0], "blind"), k=LIMIT)
    rankings: list[list[Chunk]] = []
    elapsed = 0.0
    for case in cases:
        started = time.perf_counter()
        rankings.append(index.search(query_text(case, "blind"), k=POOL))
        elapsed += time.perf_counter() - started
    return rankings, elapsed * 1000 / len(cases)


def _evaluate(max_chunk: int, cases: list[EvalCase], edition_dir: Path) -> Bm25Result:
    chunks = load_chunks(chunks_path(edition_dir, max_chunk))
    rankings, latency_ms = _rank_cases(Bm25Index(chunks), cases)
    diversified = [article_diversify(ranked, pool=POOL, limit=LIMIT) for ranked in rankings]
    return Bm25Result(
        max_chunk=max_chunk,
        baseline=aggregate_rankings(cases, rankings),
        diversified=aggregate_rankings(cases, diversified),
        latency_ms=latency_ms,
        baseline_rankings=rankings,
        diversified_rankings=diversified,
    )


def select_winner(results: list[Bm25Result]) -> Bm25Result:
    if not results:
        raise ValueError("at least one result is required")
    best_article = max(item.diversified.recall_article for item in results)
    tied = [item for item in results if best_article - item.diversified.recall_article < 0.01]
    best_exact = max(item.diversified.recall_exact for item in tied)
    tied = [item for item in tied if item.diversified.recall_exact == best_exact]
    best_all = max(item.diversified.all_article for item in tied)
    tied = [item for item in tied if item.diversified.all_article == best_all]
    return min(tied, key=lambda item: (item.latency_ms, item.max_chunk))


def _metrics(metrics: AggregateRecall) -> dict[str, float | int]:
    return asdict(metrics)


def _serializable(item: Bm25Result) -> dict[str, Any]:
    return {
        "max_chunk": item.max_chunk,
        "baseline": _metrics(item.baseline),
        "diversified": _metrics(item.diversified),
        "article_delta": item.diversified.recall_article - item.baseline.recall_article,
        "exact_delta": item.diversified.recall_exact - item.baseline.recall_exact,
        "latency_ms": item.latency_ms,
    }


def _outcomes(cases: list[EvalCase], result: Bm25Result) -> dict[str, int]:
    before = per_case_article_recall(cases, result.baseline_rankings)
    after = per_case_article_recall(cases, result.diversified_rankings)
    return {
        "gains": sum(right > left for left, right in zip(before, after, strict=True)),
        "losses": sum(right < left for left, right in zip(before, after, strict=True)),
        "ties": sum(right == left for left, right in zip(before, after, strict=True)),
    }


def _write_report(state: dict[str, Any], path: Path) -> None:
    dev = state["dev"]
    winner = dev["winner"]
    interval = dev["bootstrap"]
    lines = [
        "# Диверсификация статей для BM25 по всем max_chunk",
        "",
        "Основной сценарий: `complaint_argument_blind`; пул — 50 чанков; результат — 10 чанков, не более одного представителя статьи.",
        "Победитель выбирается на dev по article R@10. При разнице менее 0.01 используются exact R@10, all-hit article@10 и latency.",
        "",
        "## Победитель dev",
        "",
        f"BM25, `max_chunk={winner['max_chunk']}`: article R@10 "
        f"{winner['baseline']['recall_article']:.3f}→{winner['diversified']['recall_article']:.3f}; "
        f"exact R@10 {winner['baseline']['recall_exact']:.3f}→{winner['diversified']['recall_exact']:.3f}.",
        f"Парная разница article R@10: {interval['mean_delta']:+.3f} "
        f"(95% group-bootstrap [{interval['low']:+.3f}; {interval['high']:+.3f}]).",
        "",
        "## Все размеры, dev / blind",
        "",
        "| max_chunk | baseline art | diverse art | Δ art | baseline exact | diverse exact | all-hit art | ms/query* |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(dev["candidates"], key=lambda item: item["max_chunk"]):
        lines.append(
            f"| {row['max_chunk']} | {row['baseline']['recall_article']:.3f} "
            f"| {row['diversified']['recall_article']:.3f} | {row['article_delta']:+.3f} "
            f"| {row['baseline']['recall_exact']:.3f} | {row['diversified']['recall_exact']:.3f} "
            f"| {row['diversified']['all_article']:.3f} | {row['latency_ms']:.1f} |"
        )
    lines.extend(
        [
            "",
            "\\* latency включает разбор запроса и ранжирование до top-50 на CPU; построение индекса не входит.",
        ]
    )
    if "eval" in state:
        evaluated = state["eval"]
        eval_interval = evaluated["bootstrap"]
        lines.extend(
            [
                "",
                "## Однократная проверка eval",
                "",
                f"Проверен только зафиксированный на dev размер: BM25, `max_chunk={winner['max_chunk']}`.",
                "",
                "| вариант | exact R@10 | all exact@10 | article R@10 | all article@10 |",
                "|---|---:|---:|---:|---:|",
                f"| baseline | {evaluated['baseline']['recall_exact']:.3f} | {evaluated['baseline']['all_exact']:.3f} "
                f"| {evaluated['baseline']['recall_article']:.3f} | {evaluated['baseline']['all_article']:.3f} |",
                f"| article-diverse | {evaluated['diversified']['recall_exact']:.3f} | {evaluated['diversified']['all_exact']:.3f} "
                f"| {evaluated['diversified']['recall_article']:.3f} | {evaluated['diversified']['all_article']:.3f} |",
                "",
                f"Разница article R@10: {eval_interval['mean_delta']:+.3f} "
                f"(95% group-bootstrap [{eval_interval['low']:+.3f}; {eval_interval['high']:+.3f}]); "
                f"выигрышей {evaluated['outcomes']['gains']}, проигрышей {evaluated['outcomes']['losses']}, "
                f"без изменений {evaluated['outcomes']['ties']}.",
            ]
        )
    if state.get("eval_sensitivity"):
        lines.extend(
            [
                "",
                "## Post-hoc sensitivity на eval",
                "",
                "Эти размеры проверены после фиксации основного победителя и не использовались для выбора конфигурации.",
                "",
                "| max_chunk | вариант | exact R@10 | all exact@10 | article R@10 | all article@10 | 95% CI Δ article |",
                "|---:|---|---:|---:|---:|---:|---|",
            ]
        )
        for item in state["eval_sensitivity"]:
            interval = item["bootstrap"]
            lines.extend(
                [
                    f"| {item['max_chunk']} | baseline | {item['baseline']['recall_exact']:.3f} "
                    f"| {item['baseline']['all_exact']:.3f} | {item['baseline']['recall_article']:.3f} "
                    f"| {item['baseline']['all_article']:.3f} | — |",
                    f"| {item['max_chunk']} | article-diverse | {item['diversified']['recall_exact']:.3f} "
                    f"| {item['diversified']['all_exact']:.3f} | {item['diversified']['recall_article']:.3f} "
                    f"| {item['diversified']['all_article']:.3f} "
                    f"| [{interval['low']:+.3f}; {interval['high']:+.3f}] |",
                ]
            )
    lines.extend(
        [
            "",
            "## Интерпретация",
            "",
            "Article-diversification проверяется как этап расширения покрытия статей. Изменение exact R@10 отдельно показывает цену правила «один чанк на статью» и необходимость последующего выбора пунктов внутри статей-кандидатов.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _merge_main_state(state: dict[str, Any], analysis_dir: Path, report: Path) -> None:
    main_path = analysis_dir / "retrieval_error_state.json"
    if not main_path.is_file():
        return
    main = json.loads(main_path.read_text(encoding="utf-8"))
    summary: dict[str, Any] = {
        "report": str(report),
        "dev": {
            "winner": state["dev"]["winner"],
            "bootstrap": state["dev"]["bootstrap"],
            "outcomes": state["dev"]["outcomes"],
            "candidates": state["dev"]["candidates"],
        },
    }
    if "eval" in state:
        summary["eval"] = state["eval"]
    if state.get("eval_sensitivity"):
        summary["eval_sensitivity"] = state["eval_sensitivity"]
    main["bm25_diversification"] = summary
    main_path.write_text(json.dumps(main, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Article-diversification BM25 по всем max_chunk")
    parser.add_argument("--phase", choices=("dev", "eval"), default="dev")
    parser.add_argument("--edition-dir", type=Path, default=Path("data/law/2026-08-04"))
    parser.add_argument("--cases-dir", type=Path, default=Path("evals/cases"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("data/analysis"))
    parser.add_argument(
        "--eval-sensitivity-max-chunk",
        type=int,
        action="append",
        default=[],
        help="Явный post-hoc eval дополнительного max_chunk; можно повторять.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evals/reports/bm25_diversification_by_max_chunk.md"),
    )
    args = parser.parse_args(argv)
    state_path = args.analysis_dir / "bm25_diversification_state.json"
    cases = [case for case in load_verified(args.cases_dir) if case.split == args.phase]
    expected = 52 if args.phase == "dev" else 22
    if len(cases) != expected:
        print(f"expected {expected} {args.phase} cases, got {len(cases)}", file=sys.stderr)
        return 2

    if args.phase == "dev":
        sizes = discover_chunk_sizes(args.edition_dir)
        if not sizes:
            print("no structural chunk corpora", file=sys.stderr)
            return 2
        results = []
        for max_chunk in sizes:
            result = _evaluate(max_chunk, cases, args.edition_dir)
            results.append(result)
            print(
                f"m{max_chunk}: {result.baseline.recall_article:.3f} -> "
                f"{result.diversified.recall_article:.3f}",
                file=sys.stderr,
            )
        winner_result = select_winner(results)
        interval = paired_group_bootstrap(
            per_case_article_recall(cases, winner_result.baseline_rankings),
            per_case_article_recall(cases, winner_result.diversified_rankings),
            [case.case_group_id for case in cases],
            seed=SEED,
        )
        state = {
            "version": STATE_VERSION,
            "seed": SEED,
            "query": "blind",
            "pool": POOL,
            "limit": LIMIT,
            "dev": {
                "n": len(cases),
                "candidates": [_serializable(item) for item in results],
                "winner": _serializable(winner_result),
                "bootstrap": asdict(interval),
                "outcomes": _outcomes(cases, winner_result),
            },
        }
        args.analysis_dir.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"locked winner: BM25 m{winner_result.max_chunk}", file=sys.stderr)
    else:
        if not state_path.is_file():
            print("run the default dev phase first", file=sys.stderr)
            return 2
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("version") != STATE_VERSION:
            print("incompatible state version", file=sys.stderr)
            return 2
        max_chunk = int(state["dev"]["winner"]["max_chunk"])
        print(f"loading locked winner BM25 m{max_chunk}", file=sys.stderr)
        result = _evaluate(max_chunk, cases, args.edition_dir)
        interval = paired_group_bootstrap(
            per_case_article_recall(cases, result.baseline_rankings),
            per_case_article_recall(cases, result.diversified_rankings),
            [case.case_group_id for case in cases],
            seed=SEED,
        )
        state["eval"] = {
            "n": len(cases),
            "baseline": _metrics(result.baseline),
            "diversified": _metrics(result.diversified),
            "bootstrap": asdict(interval),
            "outcomes": _outcomes(cases, result),
            "latency_ms": result.latency_ms,
        }
        if args.eval_sensitivity_max_chunk:
            sensitivity = []
            for extra_chunk in sorted(set(args.eval_sensitivity_max_chunk)):
                if extra_chunk == max_chunk:
                    continue
                path = chunks_path(args.edition_dir, extra_chunk)
                if not path.is_file():
                    print(f"no structural corpus for m{extra_chunk}: {path}", file=sys.stderr)
                    return 2
                print(f"loading post-hoc sensitivity BM25 m{extra_chunk}", file=sys.stderr)
                extra = _evaluate(extra_chunk, cases, args.edition_dir)
                extra_interval = paired_group_bootstrap(
                    per_case_article_recall(cases, extra.baseline_rankings),
                    per_case_article_recall(cases, extra.diversified_rankings),
                    [case.case_group_id for case in cases],
                    seed=SEED,
                )
                sensitivity.append(
                    {
                        "max_chunk": extra_chunk,
                        "baseline": _metrics(extra.baseline),
                        "diversified": _metrics(extra.diversified),
                        "bootstrap": asdict(extra_interval),
                        "outcomes": _outcomes(cases, extra),
                        "latency_ms": extra.latency_ms,
                    }
                )
            state["eval_sensitivity"] = sensitivity
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _write_report(state, args.report)
    _merge_main_state(state, args.analysis_dir, args.report)
    print(f"wrote {args.report}", file=sys.stderr)
    print(f"wrote {state_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
