#!/usr/bin/env python3
"""Conservative article fusion plus common-corpus chunk reranking.

This second hybrid experiment protects a prefix of the dense article ranking,
lets BM25 fill only the remaining article slots, and reranks final evidence on
the shared m700 corpus with dense+BM25 chunk-level RRF.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from index.bm25 import Bm25Index  # noqa: E402
from index.error_analysis import paired_group_bootstrap, per_case_article_recall  # noqa: E402
from index.fusion import (  # noqa: E402
    dense_quota_article_fusion,
    select_article_evidence,
    weighted_rrf,
)
from index.recall import AggregateRecall  # noqa: E402
from run_hybrid_retrieval_eval import (  # noqa: E402
    BM25_MAX_CHUNKS,
    DENSE_MAX_CHUNK,
    LIMIT,
    MODEL,
    POOL,
    SEED,
    _baseline,
    _load_indexes,
    _metric_dict,
    _metrics,
    _public_baseline,
    _rank_cases,
)
from index.eval_tables import load_verified  # noqa: E402
from schemas.corpus import Chunk  # noqa: E402
from schemas.eval_case import EvalCase  # noqa: E402

ARTICLE_K = 30
ARTICLE_WEIGHTS = ((1.0, 1.0), (2.0, 1.0))
DENSE_KEEPS = (7, 8, 9)
ARTICLE_COUNTS = (8, 10)
CHUNK_K = 60
CHUNK_WEIGHTS = ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0))
STATE_VERSION = 1


@dataclass(frozen=True)
class ConservativeSpec:
    bm25_max_chunk: int
    dense_keep: int
    article_dense_weight: float
    article_bm25_weight: float
    n_articles: int
    chunk_dense_weight: float
    chunk_bm25_weight: float


@dataclass(frozen=True)
class ConservativeResult:
    spec: ConservativeSpec
    at5: AggregateRecall
    at10: AggregateRecall
    latency_ms: float
    rankings: list[list[Chunk]]


def _chunk_fusions(
    dense_full: list[list[Chunk]],
    bm25_m700_full: list[list[Chunk]],
) -> tuple[dict[tuple[float, float], list[list[Chunk]]], dict[tuple[float, float], float]]:
    rankings: dict[tuple[float, float], list[list[Chunk]]] = {}
    latency: dict[tuple[float, float], float] = {}
    for weights in CHUNK_WEIGHTS:
        fused_cases: list[list[Chunk]] = []
        started = time.perf_counter()
        for dense, bm25 in zip(dense_full, bm25_m700_full, strict=True):
            fused_cases.append(
                weighted_rrf(
                    [dense, bm25],
                    weights=weights,
                    k=CHUNK_K,
                    pool=len(dense),
                    limit=len(dense),
                )
            )
        rankings[weights] = fused_cases
        latency[weights] = (time.perf_counter() - started) * 1000 / len(dense_full)
    return rankings, latency


def _apply(
    spec: ConservativeSpec,
    dense_full: list[list[Chunk]],
    bm25_articles: list[list[Chunk]],
    evidence_rankings: list[list[Chunk]],
) -> tuple[list[list[Chunk]], float]:
    results: list[list[Chunk]] = []
    started = time.perf_counter()
    for dense, bm25, evidence in zip(dense_full, bm25_articles, evidence_rankings, strict=True):
        articles = dense_quota_article_fusion(
            dense,
            bm25,
            dense_keep=spec.dense_keep,
            weights=(spec.article_dense_weight, spec.article_bm25_weight),
            k=ARTICLE_K,
            pool=POOL,
            limit=LIMIT,
        )
        results.append(
            select_article_evidence(
                evidence,
                articles,
                n_articles=spec.n_articles,
                limit=LIMIT,
            )
        )
    return results, (time.perf_counter() - started) * 1000 / len(dense_full)


def _evaluate(
    spec: ConservativeSpec,
    cases: list[EvalCase],
    dense_full: list[list[Chunk]],
    bm25_articles: list[list[Chunk]],
    evidence_rankings: list[list[Chunk]],
    *,
    retrieval_latency_ms: float,
    chunk_fusion_latency_ms: float,
) -> ConservativeResult:
    rankings, postprocess_ms = _apply(spec, dense_full, bm25_articles, evidence_rankings)
    return ConservativeResult(
        spec,
        _metrics(cases, rankings, 5),
        _metrics(cases, rankings, 10),
        retrieval_latency_ms + chunk_fusion_latency_ms + postprocess_ms,
        rankings,
    )


def select_winner(results: list[ConservativeResult]) -> ConservativeResult:
    if not results:
        raise ValueError("at least one conservative hybrid result is required")
    best_article = max(item.at10.recall_article for item in results)
    tied = [item for item in results if best_article - item.at10.recall_article < 0.01]
    best_exact = max(item.at10.recall_exact for item in tied)
    tied = [item for item in tied if item.at10.recall_exact == best_exact]
    best_all = max(item.at10.all_article for item in tied)
    tied = [item for item in tied if item.at10.all_article == best_all]
    return min(
        tied,
        key=lambda item: (
            # Sub-millisecond wall-clock noise must not change the locked spec.
            round(item.latency_ms),
            item.spec.bm25_max_chunk,
            item.spec.dense_keep,
            item.spec.n_articles,
            item.spec.article_dense_weight,
            item.spec.chunk_dense_weight,
            item.latency_ms,
        ),
    )


def _serialize(result: ConservativeResult) -> dict[str, Any]:
    return {
        "spec": asdict(result.spec),
        "at5": _metric_dict(result.at5),
        "at10": _metric_dict(result.at10),
        "latency_ms": result.latency_ms,
    }


def _outcomes(
    cases: list[EvalCase],
    baseline: list[list[Chunk]],
    candidate: list[list[Chunk]],
) -> dict[str, int]:
    before = per_case_article_recall(cases, baseline)
    after = per_case_article_recall(cases, candidate)
    return {
        "gains": sum(right > left for left, right in zip(before, after, strict=True)),
        "losses": sum(right < left for left, right in zip(before, after, strict=True)),
        "ties": sum(right == left for left, right in zip(before, after, strict=True)),
    }


def _label(row: dict[str, Any]) -> str:
    spec = row["spec"]
    return (
        f"BM25 m{spec['bm25_max_chunk']}, keep={spec['dense_keep']}, "
        f"article w={int(spec['article_dense_weight'])}:{int(spec['article_bm25_weight'])}, "
        f"articles={spec['n_articles']}, chunk w="
        f"{int(spec['chunk_dense_weight'])}:{int(spec['chunk_bm25_weight'])}"
    )


def _write_report(state: dict[str, Any], path: Path) -> None:
    dev = state["dev"]
    winner = dev["winner"]
    baseline = dev["baseline"]
    interval = dev["bootstrap_vs_dense"]
    lines = [
        "# Conservative hybrid and within-article reranking",
        "",
        "Dense m700 сохраняет верхнюю квоту статей; BM25 m1500/m2500 заполняет только оставшиеся слоты. "
        "Внутри выбранных статей чанки общего m700-корпуса объединяются chunk-level RRF dense+BM25.",
        "Основной запрос — blind; лимит — 10 финальных чанков. Использован dense-кэш текущей версии корпуса.",
        "",
        "## Dev / blind",
        "",
        f"Выбран **{_label(winner)}**: article R@10 {baseline['at10']['recall_article']:.3f}→"
        f"{winner['at10']['recall_article']:.3f}, exact R@10 {baseline['at10']['recall_exact']:.3f}→"
        f"{winner['at10']['recall_exact']:.3f}.",
        f"Разница article R@10 {interval['mean_delta']:+.3f} "
        f"(95% group-bootstrap [{interval['low']:+.3f}; {interval['high']:+.3f}]); "
        f"выигрышей {dev['outcomes_vs_dense']['gains']}, проигрышей "
        f"{dev['outcomes_vs_dense']['losses']}, без изменений {dev['outcomes_vs_dense']['ties']}.",
        "",
        "## Top-15 конфигураций",
        "",
        "| конфигурация | exact R@5 | article R@5 | exact R@10 | article R@10 | all article@10 | ms/query* |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = sorted(
        dev["candidates"],
        key=lambda row: (
            -row["at10"]["recall_article"],
            -row["at10"]["recall_exact"],
            -row["at10"]["all_article"],
            row["latency_ms"],
        ),
    )
    for row in ordered[:15]:
        lines.append(
            f"| {_label(row)} | {row['at5']['recall_exact']:.3f} | "
            f"{row['at5']['recall_article']:.3f} | {row['at10']['recall_exact']:.3f} | "
            f"{row['at10']['recall_article']:.3f} | {row['at10']['all_article']:.3f} | "
            f"{row['latency_ms']:.1f} |"
        )
    lines.extend(
        [
            "",
            "\\* latency включает dense m700, BM25 для статей, BM25 m700 и обе стадии fusion; индексация не входит.",
            "Полная сетка сохранена локально в `data/analysis/conservative_hybrid_state.json`.",
        ]
    )
    if "eval_posthoc" in state:
        evaluated = state["eval_posthoc"]
        eval_interval = evaluated["bootstrap_vs_dense"]
        lines.extend(
            [
                "",
                "## Post-hoc validation на eval / blind",
                "",
                "Этот набор уже использовался при анализе первого гибрида, поэтому результат не является новым untouched test.",
                "",
                "| метод | exact R@5 | article R@5 | exact R@10 | article R@10 | all article@10 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, row in evaluated["methods"].items():
            lines.append(
                f"| {name} | {row['at5']['recall_exact']:.3f} | {row['at5']['recall_article']:.3f} | "
                f"{row['at10']['recall_exact']:.3f} | {row['at10']['recall_article']:.3f} | "
                f"{row['at10']['all_article']:.3f} |"
            )
        lines.extend(
            [
                "",
                f"Разница с dense article-diverse по article R@10: {eval_interval['mean_delta']:+.3f} "
                f"(95% CI [{eval_interval['low']:+.3f}; {eval_interval['high']:+.3f}]); "
                f"выигрышей {evaluated['outcomes_vs_dense']['gains']}, проигрышей "
                f"{evaluated['outcomes_vs_dense']['losses']}, без изменений "
                f"{evaluated['outcomes_vs_dense']['ties']}.",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Conservative hybrid retrieval")
    parser.add_argument("--phase", choices=("dev", "eval-posthoc"), default="dev")
    parser.add_argument("--edition-dir", type=Path, default=Path("data/law/2026-08-04"))
    parser.add_argument("--cases-dir", type=Path, default=Path("evals/cases"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("data/analysis"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evals/reports/hybrid_conservative_rerank.md"),
    )
    args = parser.parse_args(argv)
    split = "dev" if args.phase == "dev" else "eval"
    cases = [case for case in load_verified(args.cases_dir) if case.split == split]
    expected = 52 if split == "dev" else 22
    if len(cases) != expected:
        print(f"expected {expected} {split} cases, got {len(cases)}", file=sys.stderr)
        return 2
    state_path = args.analysis_dir / "conservative_hybrid_state.json"
    if args.phase == "eval-posthoc" and not state_path.is_file():
        print("run the default dev phase first", file=sys.stderr)
        return 2

    dense_chunks, dense_index, bm25_indexes = _load_indexes(args.edition_dir)
    dense_full, dense_ms = _rank_cases(dense_index, cases, k=len(dense_chunks))
    dense_baseline = _baseline(cases, dense_full, latency_ms=dense_ms)
    bm25_articles: dict[int, list[list[Chunk]]] = {}
    bm25_article_ms: dict[int, float] = {}
    for max_chunk, index in bm25_indexes.items():
        bm25_articles[max_chunk], bm25_article_ms[max_chunk] = _rank_cases(index, cases, k=POOL)
    bm25_m700_full, bm25_m700_ms = _rank_cases(
        Bm25Index(dense_chunks),
        cases,
        k=len(dense_chunks),
    )
    chunk_rankings, chunk_fusion_ms = _chunk_fusions(dense_full, bm25_m700_full)

    if args.phase == "dev":
        results: list[ConservativeResult] = []
        for bm25_max_chunk in BM25_MAX_CHUNKS:
            for dense_keep in DENSE_KEEPS:
                for article_weights in ARTICLE_WEIGHTS:
                    for n_articles in ARTICLE_COUNTS:
                        if n_articles < dense_keep:
                            continue
                        for chunk_weights in CHUNK_WEIGHTS:
                            spec = ConservativeSpec(
                                bm25_max_chunk,
                                dense_keep,
                                *article_weights,
                                n_articles,
                                *chunk_weights,
                            )
                            results.append(
                                _evaluate(
                                    spec,
                                    cases,
                                    dense_full,
                                    bm25_articles[bm25_max_chunk],
                                    chunk_rankings[chunk_weights],
                                    retrieval_latency_ms=(
                                        dense_ms + bm25_article_ms[bm25_max_chunk] + bm25_m700_ms
                                    ),
                                    chunk_fusion_latency_ms=chunk_fusion_ms[chunk_weights],
                                )
                            )
        winner = select_winner(results)
        dense_diverse = dense_baseline["rankings"]
        interval = paired_group_bootstrap(
            per_case_article_recall(cases, dense_diverse),
            per_case_article_recall(cases, winner.rankings),
            [case.case_group_id for case in cases],
            seed=SEED,
        )
        state = {
            "version": STATE_VERSION,
            "seed": SEED,
            "query": "blind",
            "model": MODEL,
            "dense_max_chunk": DENSE_MAX_CHUNK,
            "dev": {
                "n": len(cases),
                "baseline": _public_baseline(dense_baseline),
                "candidates": [_serialize(result) for result in results],
                "winner": _serialize(winner),
                "bootstrap_vs_dense": asdict(interval),
                "outcomes_vs_dense": _outcomes(cases, dense_diverse, winner.rankings),
            },
        }
        args.analysis_dir.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"locked dev winner: {_label(_serialize(winner))}", file=sys.stderr)
    else:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("version") != STATE_VERSION:
            print("incompatible state version", file=sys.stderr)
            return 2
        spec = ConservativeSpec(**state["dev"]["winner"]["spec"])
        chunk_weights = (spec.chunk_dense_weight, spec.chunk_bm25_weight)
        winner = _evaluate(
            spec,
            cases,
            dense_full,
            bm25_articles[spec.bm25_max_chunk],
            chunk_rankings[chunk_weights],
            retrieval_latency_ms=(dense_ms + bm25_article_ms[spec.bm25_max_chunk] + bm25_m700_ms),
            chunk_fusion_latency_ms=chunk_fusion_ms[chunk_weights],
        )
        dense_diverse = dense_baseline["rankings"]
        interval = paired_group_bootstrap(
            per_case_article_recall(cases, dense_diverse),
            per_case_article_recall(cases, winner.rankings),
            [case.case_group_id for case in cases],
            seed=SEED,
        )
        state["eval_posthoc"] = {
            "n": len(cases),
            "methods": {
                "dense m700 article-diverse": _public_baseline(dense_baseline),
                "locked conservative hybrid": _serialize(winner),
            },
            "bootstrap_vs_dense": asdict(interval),
            "outcomes_vs_dense": _outcomes(cases, dense_diverse, winner.rankings),
        }
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _write_report(state, args.report)
    print(f"wrote {args.report}", file=sys.stderr)
    print(f"wrote {state_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
