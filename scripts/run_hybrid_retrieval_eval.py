#!/usr/bin/env python3
"""Evaluate article-level fusion across incompatible dense and BM25 chunkings.

The default ``dev`` phase screens a fixed grid on blind queries and locks one
configuration.  The explicit ``eval`` phase evaluates only that winner against
the dense and BM25 article-diverse baselines.  Existing dense document vectors
are loaded from cache; this script never re-encodes the corpus.
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

from index.bm25 import Bm25Index, load_chunks  # noqa: E402
from index.dense import DenseIndex, SentenceTransformerEncoder  # noqa: E402
from index.error_analysis import (  # noqa: E402
    aggregate_rankings,
    paired_group_bootstrap,
    per_case_article_recall,
)
from index.eval_tables import chunks_path, load_verified, query_text  # noqa: E402
from index.fusion import (  # noqa: E402
    article_diversify,
    project_article_ranking,
    select_article_evidence,
    weighted_article_rrf,
)
from index.recall import AggregateRecall, article_key, score_case  # noqa: E402
from run_dense_eval import profile_for  # noqa: E402
from schemas.corpus import Chunk  # noqa: E402
from schemas.eval_case import EvalCase  # noqa: E402

MODEL = "Roflmax/bge-m3-russian-legal"
DENSE_MAX_CHUNK = 700
BM25_MAX_CHUNKS = (1500, 2500)
RRF_KS = (0, 10, 30, 60)
WEIGHTS = ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0))
ARTICLE_COUNTS = (5, 8, 10)
POOL = 50
LIMIT = 10
SEED = 20261008
STATE_VERSION = 1


@dataclass(frozen=True)
class HybridSpec:
    bm25_max_chunk: int
    k: int
    dense_weight: float
    bm25_weight: float
    n_articles: int


@dataclass(frozen=True)
class HybridResult:
    spec: HybridSpec
    at5: AggregateRecall
    at10: AggregateRecall
    latency_ms: float
    rankings: list[list[Chunk]]


def _rank_cases(
    index: Any,
    cases: list[EvalCase],
    *,
    k: int,
) -> tuple[list[list[Chunk]], float]:
    if cases:
        index.search(query_text(cases[0], "blind"), k=min(k, 10))
    rankings: list[list[Chunk]] = []
    elapsed = 0.0
    for case in cases:
        started = time.perf_counter()
        rankings.append(index.search(query_text(case, "blind"), k=k))
        elapsed += time.perf_counter() - started
    return rankings, elapsed * 1000 / len(cases)


def _metrics(cases: list[EvalCase], rankings: list[list[Chunk]], k: int) -> AggregateRecall:
    return aggregate_rankings(cases, rankings, k=k)


def _metric_dict(metrics: AggregateRecall) -> dict[str, float | int]:
    return asdict(metrics)


def _baseline(
    cases: list[EvalCase],
    rankings: list[list[Chunk]],
    *,
    latency_ms: float,
) -> dict[str, Any]:
    diversified = [article_diversify(ranked, pool=POOL, limit=LIMIT) for ranked in rankings]
    return {
        "at5": _metric_dict(_metrics(cases, diversified, 5)),
        "at10": _metric_dict(_metrics(cases, diversified, 10)),
        "latency_ms": latency_ms,
        "rankings": diversified,
    }


def _apply(
    spec: HybridSpec,
    dense_full: list[list[Chunk]],
    bm25: list[list[Chunk]],
) -> tuple[list[list[Chunk]], float]:
    rankings: list[list[Chunk]] = []
    elapsed = 0.0
    for dense_ranked, bm25_ranked in zip(dense_full, bm25, strict=True):
        started = time.perf_counter()
        articles = weighted_article_rrf(
            [dense_ranked, bm25_ranked],
            weights=[spec.dense_weight, spec.bm25_weight],
            k=spec.k,
            pool=POOL,
            limit=LIMIT,
        )
        rankings.append(
            select_article_evidence(
                dense_ranked,
                articles,
                n_articles=spec.n_articles,
                limit=LIMIT,
            )
        )
        elapsed += time.perf_counter() - started
    return rankings, elapsed * 1000 / len(dense_full)


def _evaluate(
    spec: HybridSpec,
    cases: list[EvalCase],
    dense_full: list[list[Chunk]],
    bm25: list[list[Chunk]],
    *,
    retrieval_latency_ms: float,
) -> HybridResult:
    rankings, postprocess_ms = _apply(spec, dense_full, bm25)
    return HybridResult(
        spec=spec,
        at5=_metrics(cases, rankings, 5),
        at10=_metrics(cases, rankings, 10),
        latency_ms=retrieval_latency_ms + postprocess_ms,
        rankings=rankings,
    )


def select_winner(results: list[HybridResult]) -> HybridResult:
    """Apply article R@10, exact R@10, all-hit article@10, latency."""
    if not results:
        raise ValueError("at least one hybrid result is required")
    best_article = max(item.at10.recall_article for item in results)
    tied = [item for item in results if best_article - item.at10.recall_article < 0.01]
    best_exact = max(item.at10.recall_exact for item in tied)
    tied = [item for item in tied if item.at10.recall_exact == best_exact]
    best_all = max(item.at10.all_article for item in tied)
    tied = [item for item in tied if item.at10.all_article == best_all]
    return min(
        tied,
        key=lambda item: (
            item.latency_ms,
            item.spec.bm25_max_chunk,
            item.spec.k,
            item.spec.dense_weight,
            item.spec.bm25_weight,
            item.spec.n_articles,
        ),
    )


def _serialize_result(result: HybridResult) -> dict[str, Any]:
    return {
        "spec": asdict(result.spec),
        "at5": _metric_dict(result.at5),
        "at10": _metric_dict(result.at10),
        "latency_ms": result.latency_ms,
    }


def _outcomes(
    cases: list[EvalCase],
    baseline: list[list[Chunk]],
    hybrid: list[list[Chunk]],
) -> dict[str, int]:
    before = per_case_article_recall(cases, baseline)
    after = per_case_article_recall(cases, hybrid)
    return {
        "gains": sum(right > left for left, right in zip(before, after, strict=True)),
        "losses": sum(right < left for left, right in zip(before, after, strict=True)),
        "ties": sum(right == left for left, right in zip(before, after, strict=True)),
    }


def _write_case_outcomes(
    path: Path,
    cases: list[EvalCase],
    dense: list[list[Chunk]],
    bm25: list[list[Chunk]],
    hybrid: list[list[Chunk]],
    spec: HybridSpec,
) -> None:
    lines: list[str] = []
    for case, dense_ranked, bm25_ranked, hybrid_ranked in zip(
        cases, dense, bm25, hybrid, strict=True
    ):
        dense_ids = {norm_id for chunk in dense_ranked[:LIMIT] for norm_id in chunk.norm_ids}
        hybrid_ids = {norm_id for chunk in hybrid_ranked[:LIMIT] for norm_id in chunk.norm_ids}
        dense_score = score_case(case.gold_norms, dense_ids)
        hybrid_score = score_case(case.gold_norms, hybrid_ids)
        fused_articles = weighted_article_rrf(
            [dense_ranked, bm25_ranked],
            weights=[spec.dense_weight, spec.bm25_weight],
            k=spec.k,
            pool=POOL,
            limit=LIMIT,
        )
        lines.append(
            json.dumps(
                {
                    "case_id": case.case_id,
                    "case_group_id": case.case_group_id,
                    "split": case.split,
                    "topic": case.topic,
                    "gold_articles": sorted({article_key(norm_id) for norm_id in case.gold_norms}),
                    "dense_article_recall": dense_score.article.recall,
                    "hybrid_article_recall": hybrid_score.article.recall,
                    "article_delta": hybrid_score.article.recall - dense_score.article.recall,
                    "dense_articles": project_article_ranking(dense_ranked, pool=POOL)[:LIMIT],
                    "bm25_articles": project_article_ranking(bm25_ranked, pool=POOL)[:LIMIT],
                    "fused_articles": fused_articles[: spec.n_articles],
                    "hybrid_chunks": [chunk.chunk_id for chunk in hybrid_ranked[:LIMIT]],
                },
                ensure_ascii=False,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _label(row: dict[str, Any]) -> str:
    spec = row["spec"]
    return (
        f"BM25 m{spec['bm25_max_chunk']}, k={spec['k']}, "
        f"w={int(spec['dense_weight'])}:{int(spec['bm25_weight'])}, "
        f"articles={spec['n_articles']}"
    )


def _write_report(state: dict[str, Any], path: Path) -> None:
    dev = state["dev"]
    winner = dev["winner"]
    interval = dev["bootstrap_vs_dense"]
    lines = [
        "# Article-level hybrid retrieval",
        "",
        "Dense-компонент: `Roflmax/bge-m3-russian-legal`, structural `max_chunk=700`. "
        "BM25-компоненты: structural `max_chunk=1500` и `2500`. Основной запрос: blind.",
        "Оба top-50 списка проецируются в ранги статей, объединяются weighted RRF, "
        "после чего итоговые evidence-чанки берутся только из dense m700.",
        "Использован dense-кэш, обновлённый для текущей версии корпуса.",
        "",
        "## Протокол выбора",
        "",
        "На dev проверены `k ∈ {0,10,30,60}`, веса dense:BM25 `1:1`, `2:1`, `1:2`, "
        "BM25 m1500/m2500 и 5/8/10 статей при лимите 10 чанков. Победитель выбирается по "
        "article R@10; при разнице менее 0.01 — по exact R@10, all-hit article@10 и latency.",
        "",
        "## Победитель dev / blind",
        "",
        f"**{_label(winner)}**: article R@10 {winner['at10']['recall_article']:.3f}, "
        f"exact R@10 {winner['at10']['recall_exact']:.3f}, "
        f"all-hit article@10 {winner['at10']['all_article']:.3f}.",
        f"Против dense m700 article-diverse: Δ article R@10 {interval['mean_delta']:+.3f} "
        f"(95% group-bootstrap [{interval['low']:+.3f}; {interval['high']:+.3f}]); "
        f"выигрышей {dev['outcomes_vs_dense']['gains']}, проигрышей "
        f"{dev['outcomes_vs_dense']['losses']}, без изменений {dev['outcomes_vs_dense']['ties']}.",
        "",
        "## Одиночные baselines, dev / blind",
        "",
        "| метод | exact R@5 | article R@5 | exact R@10 | article R@10 | all article@10 | ms/query* |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, baseline in dev["baselines"].items():
        lines.append(
            f"| {name} | {baseline['at5']['recall_exact']:.3f} | "
            f"{baseline['at5']['recall_article']:.3f} | {baseline['at10']['recall_exact']:.3f} | "
            f"{baseline['at10']['recall_article']:.3f} | {baseline['at10']['all_article']:.3f} | "
            f"{baseline['latency_ms']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Лучший гибрид для каждого BM25-размера",
            "",
            "| конфигурация | exact R@5 | article R@5 | exact R@10 | article R@10 | all article@10 | ms/query* |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    best_by_bm25: dict[int, dict[str, Any]] = {}
    for row in sorted(
        dev["candidates"],
        key=lambda item: (
            -item["at10"]["recall_article"],
            -item["at10"]["recall_exact"],
            -item["at10"]["all_article"],
            item["latency_ms"],
        ),
    ):
        best_by_bm25.setdefault(int(row["spec"]["bm25_max_chunk"]), row)
    for bm25_chunk in sorted(best_by_bm25):
        row = best_by_bm25[bm25_chunk]
        lines.append(
            f"| {_label(row)} | {row['at5']['recall_exact']:.3f} | "
            f"{row['at5']['recall_article']:.3f} | {row['at10']['recall_exact']:.3f} | "
            f"{row['at10']['recall_article']:.3f} | {row['at10']['all_article']:.3f} | "
            f"{row['latency_ms']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Top-15 конфигураций dev / blind",
            "",
            "| конфигурация | exact R@10 | article R@10 | all article@10 | ms/query* |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        dev["candidates"],
        key=lambda item: (
            -item["at10"]["recall_article"],
            -item["at10"]["recall_exact"],
            -item["at10"]["all_article"],
            item["latency_ms"],
        ),
    )[:15]:
        lines.append(
            f"| {_label(row)} | {row['at10']['recall_exact']:.3f} | "
            f"{row['at10']['recall_article']:.3f} | {row['at10']['all_article']:.3f} | "
            f"{row['latency_ms']:.1f} |"
        )
    lines.extend(
        [
            "",
            "\\* latency включает dense- и BM25-поиск плюс fusion; построение индексов не входит.",
            "Полная сетка из 72 конфигураций сохранена локально в `data/analysis/hybrid_article_fusion_state.json`.",
        ]
    )
    if "eval" in state:
        evaluated = state["eval"]
        eval_interval = evaluated["bootstrap_vs_dense"]
        lines.extend(
            [
                "",
                "## Проверка зафиксированного варианта на eval / blind",
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
                f"Гибрид против dense article-diverse: Δ article R@10 "
                f"{eval_interval['mean_delta']:+.3f} (95% group-bootstrap "
                f"[{eval_interval['low']:+.3f}; {eval_interval['high']:+.3f}]); "
                f"выигрышей {evaluated['outcomes_vs_dense']['gains']}, проигрышей "
                f"{evaluated['outcomes_vs_dense']['losses']}, без изменений "
                f"{evaluated['outcomes_vs_dense']['ties']}.",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _public_baseline(baseline: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in baseline.items() if key != "rankings"}


def _merge_aggregate(
    state: dict[str, Any],
    *,
    analysis_dir: Path,
    report: Path,
    aggregate_report: Path,
) -> None:
    main_path = analysis_dir / "retrieval_error_state.json"
    if not main_path.is_file():
        return
    main = json.loads(main_path.read_text(encoding="utf-8"))
    summary: dict[str, Any] = {
        "report": str(report),
        "dev": {
            "winner": state["dev"]["winner"],
            "bootstrap_vs_dense": state["dev"]["bootstrap_vs_dense"],
            "outcomes_vs_dense": state["dev"]["outcomes_vs_dense"],
            "baselines": state["dev"]["baselines"],
        },
    }
    if "eval" in state:
        summary["eval"] = state["eval"]
    main["article_hybrid"] = summary
    main_path.write_text(json.dumps(main, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Keep the aggregate report reproducible from its existing generator.
    from run_retrieval_error_analysis import _report as write_aggregate_report

    write_aggregate_report(main, aggregate_report)


def _load_indexes(
    edition_dir: Path,
) -> tuple[list[Chunk], DenseIndex, dict[int, Bm25Index]]:
    dense_chunks = load_chunks(chunks_path(edition_dir, DENSE_MAX_CHUNK))
    profile = profile_for(MODEL)
    encoder = SentenceTransformerEncoder(
        MODEL,
        query_prefix=profile.query_prefix,
        document_prefix=profile.document_prefix,
        max_tokens=profile.max_tokens,
    )
    dense_index = DenseIndex.load_cached(
        dense_chunks,
        encoder,
        cache_dir=edition_dir / "dense",
        model_name=MODEL,
        max_chunk=DENSE_MAX_CHUNK,
    )
    if dense_index is None:
        raise RuntimeError(f"missing compatible dense cache for {MODEL} m{DENSE_MAX_CHUNK}")
    bm25_indexes = {
        max_chunk: Bm25Index(load_chunks(chunks_path(edition_dir, max_chunk)))
        for max_chunk in BM25_MAX_CHUNKS
    }
    return dense_chunks, dense_index, bm25_indexes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Article-level dense/BM25 hybrid retrieval")
    parser.add_argument("--phase", choices=("dev", "eval"), default="dev")
    parser.add_argument("--edition-dir", type=Path, default=Path("data/law/2026-08-04"))
    parser.add_argument("--cases-dir", type=Path, default=Path("evals/cases"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("data/analysis"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evals/reports/hybrid_article_fusion.md"),
    )
    parser.add_argument(
        "--aggregate-report",
        type=Path,
        default=Path("evals/reports/retrieval_error_analysis.md"),
    )
    args = parser.parse_args(argv)
    state_path = args.analysis_dir / "hybrid_article_fusion_state.json"
    cases = [case for case in load_verified(args.cases_dir) if case.split == args.phase]
    expected = 52 if args.phase == "dev" else 22
    if len(cases) != expected:
        print(f"expected {expected} {args.phase} cases, got {len(cases)}", file=sys.stderr)
        return 2
    if args.phase == "eval" and not state_path.is_file():
        print("run the default dev phase first", file=sys.stderr)
        return 2

    dense_chunks, dense_index, bm25_indexes = _load_indexes(args.edition_dir)
    dense_full, dense_ms = _rank_cases(dense_index, cases, k=len(dense_chunks))
    dense_baseline = _baseline(cases, dense_full, latency_ms=dense_ms)
    bm25_rankings: dict[int, list[list[Chunk]]] = {}
    bm25_ms: dict[int, float] = {}
    bm25_baselines: dict[int, dict[str, Any]] = {}
    for max_chunk, index in bm25_indexes.items():
        rankings, latency = _rank_cases(index, cases, k=POOL)
        bm25_rankings[max_chunk] = rankings
        bm25_ms[max_chunk] = latency
        bm25_baselines[max_chunk] = _baseline(cases, rankings, latency_ms=latency)

    if args.phase == "dev":
        results: list[HybridResult] = []
        for bm25_max_chunk in BM25_MAX_CHUNKS:
            for k in RRF_KS:
                for dense_weight, bm25_weight in WEIGHTS:
                    for n_articles in ARTICLE_COUNTS:
                        spec = HybridSpec(
                            bm25_max_chunk=bm25_max_chunk,
                            k=k,
                            dense_weight=dense_weight,
                            bm25_weight=bm25_weight,
                            n_articles=n_articles,
                        )
                        results.append(
                            _evaluate(
                                spec,
                                cases,
                                dense_full,
                                bm25_rankings[bm25_max_chunk],
                                retrieval_latency_ms=dense_ms + bm25_ms[bm25_max_chunk],
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
            "pool": POOL,
            "limit": LIMIT,
            "dev": {
                "n": len(cases),
                "candidates": [_serialize_result(result) for result in results],
                "winner": _serialize_result(winner),
                "baselines": {
                    "dense m700 article-diverse": _public_baseline(dense_baseline),
                    **{
                        f"BM25 m{max_chunk} article-diverse": _public_baseline(baseline)
                        for max_chunk, baseline in bm25_baselines.items()
                    },
                },
                "bootstrap_vs_dense": asdict(interval),
                "outcomes_vs_dense": _outcomes(cases, dense_diverse, winner.rankings),
            },
        }
        args.analysis_dir.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _write_case_outcomes(
            args.analysis_dir / "hybrid_article_fusion_outcomes_dev.jsonl",
            cases,
            dense_full,
            bm25_rankings[winner.spec.bm25_max_chunk],
            winner.rankings,
            winner.spec,
        )
        print(f"locked winner: {_label(_serialize_result(winner))}", file=sys.stderr)
    else:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("version") != STATE_VERSION:
            print("incompatible state version", file=sys.stderr)
            return 2
        spec = HybridSpec(**state["dev"]["winner"]["spec"])
        result = _evaluate(
            spec,
            cases,
            dense_full,
            bm25_rankings[spec.bm25_max_chunk],
            retrieval_latency_ms=dense_ms + bm25_ms[spec.bm25_max_chunk],
        )
        dense_diverse = dense_baseline["rankings"]
        interval = paired_group_bootstrap(
            per_case_article_recall(cases, dense_diverse),
            per_case_article_recall(cases, result.rankings),
            [case.case_group_id for case in cases],
            seed=SEED,
        )
        state["eval"] = {
            "n": len(cases),
            "methods": {
                "dense m700 article-diverse": _public_baseline(dense_baseline),
                f"BM25 m{spec.bm25_max_chunk} article-diverse": _public_baseline(
                    bm25_baselines[spec.bm25_max_chunk]
                ),
                "locked article-level hybrid": _serialize_result(result),
            },
            "bootstrap_vs_dense": asdict(interval),
            "outcomes_vs_dense": _outcomes(cases, dense_diverse, result.rankings),
        }
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _write_case_outcomes(
            args.analysis_dir / "hybrid_article_fusion_outcomes_eval.jsonl",
            cases,
            dense_full,
            bm25_rankings[spec.bm25_max_chunk],
            result.rankings,
            spec,
        )

    _write_report(state, args.report)
    _merge_aggregate(
        state,
        analysis_dir=args.analysis_dir,
        report=args.report,
        aggregate_report=args.aggregate_report,
    )
    print(f"wrote {args.report}", file=sys.stderr)
    print(f"wrote {state_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
