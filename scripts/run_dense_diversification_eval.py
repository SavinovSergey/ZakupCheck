#!/usr/bin/env python3
"""Compare article diversification across every cached dense configuration.

The default ``dev`` phase screens every cached model/max_chunk pair using the
blind query and locks one winner.  The explicit ``eval`` phase loads only that
winner and compares its baseline with the same ranking after diversification.
Document embeddings are never rebuilt by this script.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from index.bm25 import load_chunks  # noqa: E402
from index.dense import DenseIndex, SentenceTransformerEncoder  # noqa: E402
from index.error_analysis import (  # noqa: E402
    aggregate_rankings,
    paired_group_bootstrap,
    per_case_article_recall,
)
from index.eval_tables import chunks_path, load_verified, query_text  # noqa: E402
from index.fusion import article_diversify  # noqa: E402
from index.recall import AggregateRecall  # noqa: E402
from run_dense_eval import profile_for  # noqa: E402
from schemas.corpus import Chunk  # noqa: E402
from schemas.eval_case import EvalCase  # noqa: E402

POOL = 50
LIMIT = 10
SEED = 20261008
STATE_VERSION = 1
_CACHE_RE = re.compile(r"_m(\d+)\.meta\.json$")


@dataclass(frozen=True)
class DenseConfig:
    model: str
    max_chunk: int
    truncated: int
    rows: int


@dataclass(frozen=True)
class CandidateResult:
    config: DenseConfig
    baseline: AggregateRecall
    diversified: AggregateRecall
    latency_ms: float
    baseline_rankings: list[list[Chunk]]
    diversified_rankings: list[list[Chunk]]


def discover_cached_configs(edition_dir: Path) -> list[DenseConfig]:
    """Return model/chunk configurations with valid document-vector caches."""
    configs: list[DenseConfig] = []
    for meta_path in sorted((edition_dir / "dense").glob("*.meta.json")):
        match = _CACHE_RE.search(meta_path.name)
        if match is None:
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        max_chunk = int(match.group(1))
        npy_path = meta_path.with_name(meta_path.name.removesuffix(".meta.json") + ".npy")
        if not npy_path.is_file() or not chunks_path(edition_dir, max_chunk).is_file():
            continue
        configs.append(
            DenseConfig(
                model=str(meta["model"]),
                max_chunk=max_chunk,
                truncated=int(meta["truncated"]),
                rows=int(meta["rows"]),
            )
        )
    return sorted(configs, key=lambda item: (item.model, item.max_chunk))


def _rank_cases(
    index: DenseIndex,
    cases: list[EvalCase],
) -> tuple[list[list[Chunk]], float]:
    if cases:
        index.search(query_text(cases[0], "blind"), k=LIMIT)
    rankings: list[list[Chunk]] = []
    elapsed = 0.0
    for case in cases:
        started = time.perf_counter()
        rankings.append(index.search(query_text(case, "blind"), k=POOL))
        elapsed += time.perf_counter() - started
    return rankings, elapsed * 1000 / len(cases)


def _evaluate_config(
    config: DenseConfig,
    encoder: SentenceTransformerEncoder,
    cases: list[EvalCase],
    edition_dir: Path,
) -> CandidateResult:
    chunks = load_chunks(chunks_path(edition_dir, config.max_chunk))
    index = DenseIndex.load_cached(
        chunks,
        encoder,
        cache_dir=edition_dir / "dense",
        model_name=config.model,
        max_chunk=config.max_chunk,
    )
    if index is None:
        raise RuntimeError(f"cache mismatch for {config.model} m{config.max_chunk}")
    baseline_rankings, latency_ms = _rank_cases(index, cases)
    diversified_rankings = [
        article_diversify(ranked, pool=POOL, limit=LIMIT) for ranked in baseline_rankings
    ]
    return CandidateResult(
        config=config,
        baseline=aggregate_rankings(cases, baseline_rankings),
        diversified=aggregate_rankings(cases, diversified_rankings),
        latency_ms=latency_ms,
        baseline_rankings=baseline_rankings,
        diversified_rankings=diversified_rankings,
    )


def select_winner(results: list[CandidateResult]) -> CandidateResult:
    """Apply the declared article/exact/all-hit/latency tie-break chain."""
    if not results:
        raise ValueError("at least one candidate is required")
    best_article = max(item.diversified.recall_article for item in results)
    tied = [
        item
        for item in results
        if best_article - item.diversified.recall_article < 0.01
    ]
    best_exact = max(item.diversified.recall_exact for item in tied)
    tied = [item for item in tied if item.diversified.recall_exact == best_exact]
    best_all = max(item.diversified.all_article for item in tied)
    tied = [item for item in tied if item.diversified.all_article == best_all]
    return min(
        tied,
        key=lambda item: (item.latency_ms, item.config.model, item.config.max_chunk),
    )


def _metrics(metrics: AggregateRecall) -> dict[str, float | int]:
    return asdict(metrics)


def _serializable(item: CandidateResult) -> dict[str, Any]:
    return {
        "model": item.config.model,
        "max_chunk": item.config.max_chunk,
        "truncated": item.config.truncated,
        "rows": item.config.rows,
        "baseline": _metrics(item.baseline),
        "diversified": _metrics(item.diversified),
        "article_delta": item.diversified.recall_article - item.baseline.recall_article,
        "exact_delta": item.diversified.recall_exact - item.baseline.recall_exact,
        "latency_ms": item.latency_ms,
    }


def _outcomes(
    cases: list[EvalCase],
    baseline: list[list[Chunk]],
    diversified: list[list[Chunk]],
) -> dict[str, int]:
    before = per_case_article_recall(cases, baseline)
    after = per_case_article_recall(cases, diversified)
    return {
        "gains": sum(right > left for left, right in zip(before, after, strict=True)),
        "losses": sum(right < left for left, right in zip(before, after, strict=True)),
        "ties": sum(right == left for left, right in zip(before, after, strict=True)),
    }


def _write_report(state: dict[str, Any], report_path: Path) -> None:
    dev = state["dev"]
    winner = dev["winner"]
    interval = dev["bootstrap"]
    lines = [
        "# Диверсификация статей по всем dense-конфигурациям",
        "",
        "Основной сценарий: `complaint_argument_blind`; пул — 50 чанков; результат — 10 чанков, не более одного представителя статьи.",
        "Победитель выбирается на dev по article R@10. При разнице менее 0.01 используются exact R@10, all-hit article@10 и latency.",
        "Документные эмбеддинги не пересчитывались: использованы существующие dense-кэши.",
        "",
        "## Победитель dev",
        "",
        f"**{winner['model']}**, `max_chunk={winner['max_chunk']}`: article R@10 "
        f"{winner['diversified']['recall_article']:.3f} после диверсификации против "
        f"{winner['baseline']['recall_article']:.3f} baseline; exact R@10 "
        f"{winner['diversified']['recall_exact']:.3f} против {winner['baseline']['recall_exact']:.3f}.",
        f"Парная разница article R@10: {interval['mean_delta']:+.3f} "
        f"(95% group-bootstrap [{interval['low']:+.3f}; {interval['high']:+.3f}]).",
        "",
        "## Все конфигурации, dev / blind",
        "",
        "| модель | max_chunk | baseline art | diverse art | Δ art | baseline exact | diverse exact | all-hit art | ms/query* |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    candidates = sorted(
        dev["candidates"],
        key=lambda row: (-row["diversified"]["recall_article"], row["model"], row["max_chunk"]),
    )
    for row in candidates:
        lines.append(
            f"| {row['model']} | {row['max_chunk']} | {row['baseline']['recall_article']:.3f} "
            f"| {row['diversified']['recall_article']:.3f} | {row['article_delta']:+.3f} "
            f"| {row['baseline']['recall_exact']:.3f} | {row['diversified']['recall_exact']:.3f} "
            f"| {row['diversified']['all_article']:.3f} | {row['latency_ms']:.1f} |"
        )
    lines.extend(
        [
            "",
            "\\* latency включает кодирование запроса и ранжирование до top-50 на CPU; постобработка диверсификации пренебрежимо мала.",
            "",
            "## Лучший результат каждого эмбеддера",
            "",
            "| модель | max_chunk | baseline art | diverse art | Δ art | diverse exact |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    best_by_model: dict[str, dict[str, Any]] = {}
    for row in candidates:
        best_by_model.setdefault(row["model"], row)
    for model, row in sorted(
        best_by_model.items(),
        key=lambda item: (-item[1]["diversified"]["recall_article"], item[0]),
    ):
        lines.append(
            f"| {model} | {row['max_chunk']} | {row['baseline']['recall_article']:.3f} "
            f"| {row['diversified']['recall_article']:.3f} | {row['article_delta']:+.3f} "
            f"| {row['diversified']['recall_exact']:.3f} |"
        )
    if "eval" in state:
        evaluated = state["eval"]
        eval_interval = evaluated["bootstrap"]
        lines.extend(
            [
                "",
                "## Однократная проверка eval",
                "",
                f"Проверена только зафиксированная на dev конфигурация: **{winner['model']}**, `max_chunk={winner['max_chunk']}`.",
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
    lines.extend(
        [
            "",
            "## Интерпретация",
            "",
            "Диверсификация сравнивает не качество новых эмбеддингов, а устойчивость каждого ранжирования к повторным чанкам одной статьи. "
            "Её следует использовать для отбора статей-кандидатов; падение exact R@10 показывает, что затем нужен второй этап ранжирования частей и пунктов внутри выбранных статей.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _merge_main_state(summary: dict[str, Any], analysis_dir: Path) -> None:
    """Expose the global screen to the aggregate report without case text."""
    main_path = analysis_dir / "retrieval_error_state.json"
    if not main_path.is_file():
        return
    main = json.loads(main_path.read_text(encoding="utf-8"))
    main["cross_model_diversification"] = summary
    main_path.write_text(json.dumps(main, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Article-diversification для всех dense-кэшей")
    parser.add_argument("--phase", choices=("dev", "eval"), default="dev")
    parser.add_argument("--edition-dir", type=Path, default=Path("data/law/2026-08-04"))
    parser.add_argument("--cases-dir", type=Path, default=Path("evals/cases"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("data/analysis"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evals/reports/dense_diversification_by_model.md"),
    )
    args = parser.parse_args(argv)
    state_path = args.analysis_dir / "dense_diversification_state.json"
    cases = [case for case in load_verified(args.cases_dir) if case.split == args.phase]
    expected = 52 if args.phase == "dev" else 22
    if len(cases) != expected:
        print(f"expected {expected} {args.phase} cases, got {len(cases)}", file=sys.stderr)
        return 2

    if args.phase == "dev":
        configs = discover_cached_configs(args.edition_dir)
        if not configs:
            print("no compatible dense caches", file=sys.stderr)
            return 2
        grouped: dict[str, list[DenseConfig]] = defaultdict(list)
        for config in configs:
            grouped[config.model].append(config)
        results: list[CandidateResult] = []
        for model, model_configs in sorted(grouped.items()):
            profile = profile_for(model)
            print(f"loading {model}", file=sys.stderr)
            encoder = SentenceTransformerEncoder(
                model,
                query_prefix=profile.query_prefix,
                document_prefix=profile.document_prefix,
                max_tokens=profile.max_tokens,
            )
            for config in model_configs:
                result = _evaluate_config(config, encoder, cases, args.edition_dir)
                results.append(result)
                print(
                    f"  m{config.max_chunk}: {result.baseline.recall_article:.3f} -> "
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
                "outcomes": _outcomes(
                    cases,
                    winner_result.baseline_rankings,
                    winner_result.diversified_rankings,
                ),
            },
        }
        args.analysis_dir.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            f"locked winner: {winner_result.config.model} m{winner_result.config.max_chunk}",
            file=sys.stderr,
        )
    else:
        if not state_path.is_file():
            print("run the default dev phase first", file=sys.stderr)
            return 2
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("version") != STATE_VERSION:
            print("incompatible state version", file=sys.stderr)
            return 2
        locked = state["dev"]["winner"]
        config = DenseConfig(
            model=locked["model"],
            max_chunk=int(locked["max_chunk"]),
            truncated=int(locked["truncated"]),
            rows=int(locked["rows"]),
        )
        profile = profile_for(config.model)
        print(f"loading locked winner {config.model} m{config.max_chunk}", file=sys.stderr)
        encoder = SentenceTransformerEncoder(
            config.model,
            query_prefix=profile.query_prefix,
            document_prefix=profile.document_prefix,
            max_tokens=profile.max_tokens,
        )
        result = _evaluate_config(config, encoder, cases, args.edition_dir)
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
            "outcomes": _outcomes(cases, result.baseline_rankings, result.diversified_rankings),
            "latency_ms": result.latency_ms,
        }
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _write_report(state, args.report)
    summary = {
        "report": str(args.report),
        "dev": {
            "winner": state["dev"]["winner"],
            "bootstrap": state["dev"]["bootstrap"],
            "outcomes": state["dev"]["outcomes"],
        },
    }
    if "eval" in state:
        summary["eval"] = state["eval"]
    _merge_main_state(summary, args.analysis_dir)
    print(f"wrote {args.report}", file=sys.stderr)
    print(f"wrote {state_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
