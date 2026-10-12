#!/usr/bin/env python3
"""Cross-encoder reranking inside conservatively selected articles."""

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
from index.dense import corpus_fingerprint  # noqa: E402
from index.error_analysis import paired_group_bootstrap, per_case_article_recall  # noqa: E402
from index.eval_tables import load_verified, query_text  # noqa: E402
from index.fusion import (  # noqa: E402
    dense_quota_article_fusion,
    select_article_evidence,
)
from index.rerank import (  # noqa: E402
    SentenceTransformerCrossEncoder,
    article_candidate_pool,
)
from index.recall import AggregateRecall  # noqa: E402
from run_conservative_hybrid_eval import ConservativeSpec  # noqa: E402
from run_hybrid_retrieval_eval import (  # noqa: E402
    LIMIT,
    POOL,
    SEED,
    _baseline,
    _load_indexes,
    _metric_dict,
    _metrics,
    _public_baseline,
    _rank_cases,
)
from schemas.corpus import Chunk  # noqa: E402
from schemas.eval_case import EvalCase  # noqa: E402

MODEL = "BAAI/bge-reranker-v2-m3"
LOCAL_MODEL = ROOT / "data/analysis/models/bge-reranker-v2-m3-pytorch"
PER_ARTICLE = (2, 4, 6)
ARTICLE_COUNTS = (6, 8, 10)
MAX_PER_ARTICLE = max(PER_ARTICLE)
STATE_VERSION = 1


@dataclass(frozen=True)
class CrossEncoderSpec:
    per_article: int
    n_articles: int


@dataclass(frozen=True)
class CrossEncoderResult:
    spec: CrossEncoderSpec
    at5: AggregateRecall
    at10: AggregateRecall
    latency_ms: float
    rankings: list[list[Chunk]]


def select_winner(results: list[CrossEncoderResult]) -> CrossEncoderResult:
    if not results:
        raise ValueError("at least one cross-encoder result is required")
    best_article = max(item.at10.recall_article for item in results)
    tied = [item for item in results if best_article - item.at10.recall_article < 0.01]
    best_exact = max(item.at10.recall_exact for item in tied)
    tied = [item for item in tied if item.at10.recall_exact == best_exact]
    best_all = max(item.at10.all_article for item in tied)
    tied = [item for item in tied if item.at10.all_article == best_all]
    return min(tied, key=lambda item: (item.latency_ms, item.spec.per_article, item.spec.n_articles))


def _serialize(result: CrossEncoderResult) -> dict[str, Any]:
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


def _empty_score_cache(fingerprint: str) -> dict[str, Any]:
    return {
        "version": 1,
        "model": MODEL,
        "corpus_fingerprint": fingerprint,
        "splits": {},
        "latency_seconds": {},
    }


def _load_score_cache(
    path: Path,
    fingerprint: str,
    *,
    id_migration: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        return _empty_score_cache(fingerprint)
    cache = json.loads(path.read_text(encoding="utf-8"))
    if cache.get("version") != 1 or cache.get("model") != MODEL:
        raise ValueError("incompatible cross-encoder score cache")
    cached_fingerprint = cache.get("corpus_fingerprint")
    if cached_fingerprint is None and id_migration is not None:
        # One-time migration for the pre-fingerprint cache. The mapping is
        # built from the verified legacy view of the m700 corpus.
        for cases in cache.get("splits", {}).values():
            for case_id, scores in list(cases.items()):
                cases[case_id] = {
                    id_migration[chunk_id]: score
                    for chunk_id, score in scores.items()
                    if chunk_id in id_migration
                }
        cache["corpus_fingerprint"] = fingerprint
    elif cached_fingerprint != fingerprint:
        # Chunk ids are sequential and can be reassigned when corpus coverage
        # changes. Never reuse scores by chunk_id across unknown fingerprints.
        return _empty_score_cache(fingerprint)
    return cache


def legacy_score_id_map(chunks: list[Chunk]) -> dict[str, str]:
    """Map pre-coverage-fix sequential ids to current unchanged chunks."""
    unchanged = [chunk for chunk in chunks if chunk.chunking != "article"]
    return {
        f"chk_{chunk.edition_id}_{index:05d}": chunk.chunk_id
        for index, chunk in enumerate(unchanged, start=1)
    }


def _score_pools(
    cases: list[EvalCase],
    pools: list[list[Chunk]],
    *,
    split: str,
    cache_path: Path,
    batch_size: int,
    fingerprint: str,
    id_migration: dict[str, str] | None = None,
) -> tuple[list[dict[str, float]], float]:
    cache = _load_score_cache(cache_path, fingerprint, id_migration=id_migration)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False) + "\n", encoding="utf-8")
    split_cache = cache.setdefault("splits", {}).setdefault(split, {})
    latency_cache = cache.setdefault("latency_seconds", {})
    total_elapsed = float(latency_cache.get(split, 0.0))
    scorer: SentenceTransformerCrossEncoder | None = None
    scored: list[dict[str, float]] = []
    for index, (case, pool) in enumerate(zip(cases, pools, strict=True), start=1):
        case_cache = split_cache.setdefault(case.case_id, {})
        missing = [chunk for chunk in pool if chunk.chunk_id not in case_cache]
        if missing:
            if scorer is None:
                model_source = str(LOCAL_MODEL) if LOCAL_MODEL.is_dir() else MODEL
                scorer = SentenceTransformerCrossEncoder(
                    model_source,
                    max_length=512,
                    batch_size=batch_size,
                )
            started = time.perf_counter()
            values = scorer.score(query_text(case, "blind"), missing)
            total_elapsed += time.perf_counter() - started
            latency_cache[split] = total_elapsed
            case_cache.update(
                {chunk.chunk_id: float(score) for chunk, score in zip(missing, values, strict=True)}
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"cross-encoder {split}: {index}/{len(cases)}", file=sys.stderr)
        scored.append({chunk.chunk_id: float(case_cache[chunk.chunk_id]) for chunk in pool})
    return scored, total_elapsed * 1000 / len(cases)


def _rerank(pool: list[Chunk], scores: dict[str, float]) -> list[Chunk]:
    return sorted(pool, key=lambda chunk: (-scores[chunk.chunk_id], chunk.chunk_id))


def _evaluate(
    spec: CrossEncoderSpec,
    cases: list[EvalCase],
    article_rankings: list[list[str]],
    dense_full: list[list[Chunk]],
    bm25_m700_full: list[list[Chunk]],
    max_scores: list[dict[str, float]],
    *,
    retrieval_latency_ms: float,
    cross_encoder_latency_ms: float,
) -> CrossEncoderResult:
    rankings: list[list[Chunk]] = []
    started = time.perf_counter()
    for articles, dense, bm25, scores in zip(
        article_rankings, dense_full, bm25_m700_full, max_scores, strict=True
    ):
        pool = article_candidate_pool(
            [dense, bm25],
            articles,
            per_article=spec.per_article,
        )
        reranked = _rerank(pool, scores)
        rankings.append(
            select_article_evidence(
                reranked,
                articles,
                n_articles=spec.n_articles,
                limit=LIMIT,
            )
        )
    postprocess_ms = (time.perf_counter() - started) * 1000 / len(cases)
    return CrossEncoderResult(
        spec,
        _metrics(cases, rankings, 5),
        _metrics(cases, rankings, 10),
        retrieval_latency_ms + cross_encoder_latency_ms + postprocess_ms,
        rankings,
    )


def _write_report(state: dict[str, Any], path: Path) -> None:
    dev = state["dev"]
    winner = dev["winner"]
    baseline = dev["baseline"]
    conservative = dev["conservative_baseline"]
    interval = dev["bootstrap_vs_dense"]
    ceiling = dev["candidate_ceiling"]
    lines = [
        "# Cross-encoder reranking",
        "",
        f"Модель: `{MODEL}`; CPU, max_length=512. Статьи выбираются консервативным "
        "dense+BM25 fusion, затем cross-encoder переранжирует m700-чанки внутри этих статей.",
        "",
        "## Candidate ceiling, dev / blind",
        "",
        f"В пуле до {MAX_PER_ARTICLE} чанков на статью достижимы article recall "
        f"{ceiling['recall_article']:.3f} и exact recall {ceiling['recall_exact']:.3f}. Это потолок пула, а не top-10.",
        "",
        "## Dev / blind",
        "",
        f"Выбрано: `{winner['spec']['per_article']}` кандидата на статью, "
        f"`{winner['spec']['n_articles']}` обязательных статей. Dense → conservative hybrid → cross-encoder: "
        f"article R@10 {baseline['at10']['recall_article']:.3f}→"
        f"{conservative['at10']['recall_article']:.3f}→{winner['at10']['recall_article']:.3f}; "
        f"exact R@10 {baseline['at10']['recall_exact']:.3f}→"
        f"{conservative['at10']['recall_exact']:.3f}→{winner['at10']['recall_exact']:.3f}.",
        "Article-выигрыш целиком даёт candidate generation; cross-encoder меняет "
        "только выбор чанка внутри уже выбранной статьи.",
        f"Разница article R@10 {interval['mean_delta']:+.3f} "
        f"(95% group-bootstrap [{interval['low']:+.3f}; {interval['high']:+.3f}]); "
        f"выигрышей {dev['outcomes_vs_dense']['gains']}, проигрышей "
        f"{dev['outcomes_vs_dense']['losses']}, без изменений {dev['outcomes_vs_dense']['ties']}.",
        "",
        "| candidates/article | mandatory articles | exact R@5 | article R@5 | exact R@10 | article R@10 | all article@10 | ms/query* |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        dev["candidates"],
        key=lambda item: (
            -item["at10"]["recall_article"],
            -item["at10"]["recall_exact"],
            item["latency_ms"],
        ),
    ):
        lines.append(
            f"| {row['spec']['per_article']} | {row['spec']['n_articles']} | "
            f"{row['at5']['recall_exact']:.3f} | {row['at5']['recall_article']:.3f} | "
            f"{row['at10']['recall_exact']:.3f} | {row['at10']['recall_article']:.3f} | "
            f"{row['at10']['all_article']:.3f} | {row['latency_ms']:.1f} |"
        )
    lines.extend(
        [
            "",
            "\\* latency включает retrieval, cross-encoder и постобработку; повторный запуск может читать scores из кэша, но в отчёте хранится измеренная первичная latency.",
        ]
    )
    if "eval_posthoc" in state:
        evaluated = state["eval_posthoc"]
        eval_interval = evaluated["bootstrap_vs_dense"]
        lines.extend(
            [
                "",
                "## Post-hoc validation eval / blind",
                "",
                "Eval уже не является untouched test; результат только описательный.",
                "",
                "| метод | exact R@10 | article R@10 | all article@10 |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, row in evaluated["methods"].items():
            lines.append(
                f"| {name} | {row['at10']['recall_exact']:.3f} | "
                f"{row['at10']['recall_article']:.3f} | {row['at10']['all_article']:.3f} |"
            )
        lines.extend(
            [
                "",
                f"Разница article R@10 {eval_interval['mean_delta']:+.3f} "
                f"(95% CI [{eval_interval['low']:+.3f}; {eval_interval['high']:+.3f}]).",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _merge_aggregate(
    state: dict[str, Any],
    conservative: dict[str, Any],
    *,
    analysis_dir: Path,
    report: Path,
    conservative_report: Path,
    aggregate_report: Path,
) -> None:
    main_path = analysis_dir / "retrieval_error_state.json"
    if not main_path.is_file():
        return
    main = json.loads(main_path.read_text(encoding="utf-8"))
    main["conservative_hybrid"] = {**conservative, "report": str(conservative_report)}
    main["cross_encoder"] = {**state, "report": str(report)}
    main_path.write_text(json.dumps(main, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    from run_retrieval_error_analysis import _report as write_aggregate_report

    write_aggregate_report(main, aggregate_report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-encoder reranking")
    parser.add_argument("--phase", choices=("dev", "eval-posthoc"), default="dev")
    parser.add_argument("--edition-dir", type=Path, default=Path("data/law/2026-08-04"))
    parser.add_argument("--cases-dir", type=Path, default=Path("evals/cases"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("data/analysis"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evals/reports/cross_encoder_rerank.md"),
    )
    parser.add_argument(
        "--conservative-report",
        type=Path,
        default=Path("evals/reports/hybrid_conservative_rerank.md"),
    )
    parser.add_argument(
        "--aggregate-report",
        type=Path,
        default=Path("evals/reports/retrieval_error_analysis.md"),
    )
    args = parser.parse_args(argv)
    split = "dev" if args.phase == "dev" else "eval"
    cases = [case for case in load_verified(args.cases_dir) if case.split == split]
    expected = 52 if split == "dev" else 22
    if len(cases) != expected:
        print(f"expected {expected} {split} cases, got {len(cases)}", file=sys.stderr)
        return 2

    conservative_path = args.analysis_dir / "conservative_hybrid_state.json"
    if not conservative_path.is_file():
        print("run conservative hybrid dev first", file=sys.stderr)
        return 2
    conservative = json.loads(conservative_path.read_text(encoding="utf-8"))
    conservative_spec = ConservativeSpec(**conservative["dev"]["winner"]["spec"])
    state_path = args.analysis_dir / "cross_encoder_state.json"
    if args.phase == "eval-posthoc" and not state_path.is_file():
        print("run cross-encoder dev first", file=sys.stderr)
        return 2

    dense_chunks, dense_index, bm25_indexes = _load_indexes(args.edition_dir)
    dense_full, dense_ms = _rank_cases(dense_index, cases, k=len(dense_chunks))
    dense_baseline = _baseline(cases, dense_full, latency_ms=dense_ms)
    bm25_articles, bm25_article_ms = _rank_cases(
        bm25_indexes[conservative_spec.bm25_max_chunk],
        cases,
        k=POOL,
    )
    bm25_m700_full, bm25_m700_ms = _rank_cases(
        Bm25Index(dense_chunks),
        cases,
        k=len(dense_chunks),
    )
    article_rankings = [
        dense_quota_article_fusion(
            dense,
            bm25,
            dense_keep=conservative_spec.dense_keep,
            weights=(
                conservative_spec.article_dense_weight,
                conservative_spec.article_bm25_weight,
            ),
            k=30,
            pool=POOL,
            limit=LIMIT,
        )
        for dense, bm25 in zip(dense_full, bm25_articles, strict=True)
    ]
    max_pools = [
        article_candidate_pool(
            [dense, bm25],
            articles,
            per_article=MAX_PER_ARTICLE,
        )
        for articles, dense, bm25 in zip(
            article_rankings, dense_full, bm25_m700_full, strict=True
        )
    ]
    score_cache = args.analysis_dir / "cross_encoder_scores.json"
    scores, cross_encoder_ms = _score_pools(
        cases,
        max_pools,
        split=split,
        cache_path=score_cache,
        batch_size=args.batch_size,
        fingerprint=corpus_fingerprint(dense_chunks),
        id_migration=legacy_score_id_map(dense_chunks),
    )
    candidate_ceiling = _metrics(cases, max_pools, max(len(pool) for pool in max_pools))
    retrieval_ms = dense_ms + bm25_article_ms + bm25_m700_ms

    if args.phase == "dev":
        results = [
            _evaluate(
                CrossEncoderSpec(per_article, n_articles),
                cases,
                article_rankings,
                dense_full,
                bm25_m700_full,
                scores,
                retrieval_latency_ms=retrieval_ms,
                cross_encoder_latency_ms=cross_encoder_ms,
            )
            for per_article in PER_ARTICLE
            for n_articles in ARTICLE_COUNTS
        ]
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
            "model": MODEL,
            "query": "blind",
            "seed": SEED,
            "conservative_article_spec": asdict(conservative_spec),
            "dev": {
                "n": len(cases),
                "baseline": _public_baseline(dense_baseline),
                "conservative_baseline": conservative["dev"]["winner"],
                "candidate_ceiling": _metric_dict(candidate_ceiling),
                "cross_encoder_latency_ms": cross_encoder_ms,
                "candidates": [_serialize(result) for result in results],
                "winner": _serialize(winner),
                "bootstrap_vs_dense": asdict(interval),
                "outcomes_vs_dense": _outcomes(cases, dense_diverse, winner.rankings),
            },
        }
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"locked dev winner: {winner.spec}", file=sys.stderr)
    else:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("version") != STATE_VERSION or state.get("model") != MODEL:
            print("incompatible cross-encoder state", file=sys.stderr)
            return 2
        spec = CrossEncoderSpec(**state["dev"]["winner"]["spec"])
        winner = _evaluate(
            spec,
            cases,
            article_rankings,
            dense_full,
            bm25_m700_full,
            scores,
            retrieval_latency_ms=retrieval_ms,
            cross_encoder_latency_ms=cross_encoder_ms,
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
            "candidate_ceiling": _metric_dict(candidate_ceiling),
            "cross_encoder_latency_ms": cross_encoder_ms,
            "methods": {
                "dense m700 article-diverse": _public_baseline(dense_baseline),
                "locked conservative hybrid": conservative["eval_posthoc"]["methods"][
                    "locked conservative hybrid"
                ],
                "locked cross-encoder rerank": _serialize(winner),
            },
            "bootstrap_vs_dense": asdict(interval),
            "outcomes_vs_dense": _outcomes(cases, dense_diverse, winner.rankings),
        }
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _write_report(state, args.report)
    _merge_aggregate(
        state,
        conservative,
        analysis_dir=args.analysis_dir,
        report=args.report,
        conservative_report=args.conservative_report,
        aggregate_report=args.aggregate_report,
    )
    print(f"wrote {args.report}", file=sys.stderr)
    print(f"wrote {state_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
