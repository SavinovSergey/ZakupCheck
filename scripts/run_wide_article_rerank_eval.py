#!/usr/bin/env python3
"""Wide article-first cross-encoder experiment.

The default dev phase retrieves 50 unique articles from dense m700 and BM25
m1500, scores every m700 chunk belonging to their union once, and then reuses
those scores for two sweeps:

* dense/BM25 article source quotas;
* cross-encoded chunks per selected article at the locked quota.

The explicit eval phase evaluates only the dev-locked configuration.  Scores
are checkpointed after small blocks, so a long CPU run is safely resumable.
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
from index.dense import corpus_fingerprint  # noqa: E402
from index.error_analysis import paired_group_bootstrap, per_case_article_recall  # noqa: E402
from index.eval_tables import load_verified, query_text  # noqa: E402
from index.fusion import article_diversify, project_article_ranking  # noqa: E402
from index.rerank import (  # noqa: E402
    SentenceTransformerCrossEncoder,
    article_candidate_pool,
    best_chunk_per_article,
    source_quota_articles,
)
from index.recall import AggregateRecall, article_key  # noqa: E402
from run_hybrid_retrieval_eval import (  # noqa: E402
    _load_indexes,
    _metric_dict,
    _metrics,
    _rank_cases,
)
from schemas.corpus import Chunk  # noqa: E402
from schemas.eval_case import EvalCase  # noqa: E402

MODEL = "qilowoq/bge-reranker-v2-m3-en-ru"
LOCAL_MODEL = ROOT / "data/analysis/models/bge-reranker-v2-m3-en-ru-pytorch"
DENSE_MAX_CHUNK = 700
BM25_MAX_CHUNK = 1500
SOURCE_ARTICLES = 50
LIMIT = 10
QUOTAS = (0, 5, 10, 15, 20, 30, 40, 50)
CHUNKS_PER_ARTICLE: tuple[int | None, ...] = (1, 2, 4, 6, 8, 12, 16, 24, None)
QUOTA_SCREEN_CHUNKS = 8
SEED = 20261008
STATE_VERSION = 1
CACHE_VERSION = 1


def reranker_cache_path(analysis_dir: Path, precision: str) -> Path:
    """Keep FP32 and legacy BF16 scores physically separate."""
    if precision == "fp32":
        return analysis_dir / "wide_article_rerank_scores_fp32.json"
    return analysis_dir / "wide_article_rerank_scores.json"


@dataclass(frozen=True)
class QuotaSpec:
    dense_articles: int
    bm25_articles: int


@dataclass(frozen=True)
class SweepResult:
    spec: dict[str, Any]
    at5: AggregateRecall
    at10: AggregateRecall
    mean_articles: float
    mean_chunks: float
    rankings: list[list[Chunk]]


def unique_article_prefix(ranking: list[Chunk], limit: int) -> list[str]:
    """Return ``limit`` unique articles, looking through the full ranking."""
    return project_article_ranking(ranking, pool=len(ranking))[:limit]


def chunks_by_article(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    mapped: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        for article in sorted({article_key(norm_id) for norm_id in chunk.norm_ids}):
            mapped.setdefault(article, []).append(chunk)
    return mapped


def all_chunks_for_articles(
    articles: list[str],
    mapping: dict[str, list[Chunk]],
) -> list[Chunk]:
    """Deterministic unique m700 chunks for an article set."""
    chosen: dict[str, Chunk] = {}
    for article in articles:
        for chunk in mapping.get(article, []):
            chosen.setdefault(chunk.chunk_id, chunk)
    return [chosen[chunk_id] for chunk_id in sorted(chosen)]


def pareto_frontier(rows: list[SweepResult]) -> list[SweepResult]:
    """Keep points not dominated on article recall and mean chunk cost."""
    frontier: list[SweepResult] = []
    for row in rows:
        dominated = any(
            other.at10.recall_article >= row.at10.recall_article
            and other.mean_chunks <= row.mean_chunks
            and (
                other.at10.recall_article > row.at10.recall_article
                or other.mean_chunks < row.mean_chunks
            )
            for other in rows
        )
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda item: (item.mean_chunks, -item.at10.recall_article))


def select_winner(rows: list[SweepResult]) -> SweepResult:
    """Article R@10, then exact/all-hit, then reranking cost."""
    if not rows:
        raise ValueError("at least one result is required")
    best_article = max(row.at10.recall_article for row in rows)
    tied = [row for row in rows if best_article - row.at10.recall_article < 0.01]
    best_exact = max(row.at10.recall_exact for row in tied)
    tied = [row for row in tied if row.at10.recall_exact == best_exact]
    best_all = max(row.at10.all_article for row in tied)
    tied = [row for row in tied if row.at10.all_article == best_all]
    return min(
        tied,
        key=lambda row: (
            row.mean_chunks,
            row.mean_articles,
            tuple(sorted(row.spec.items())),
        ),
    )


def _empty_cache(fingerprint: str, precision: str) -> dict[str, Any]:
    return {
        "version": CACHE_VERSION,
        "model": MODEL,
        "corpus_fingerprint": fingerprint,
        "max_length": 512,
        "precision": precision,
        "splits": {},
        "latency_seconds": {},
    }


def _load_cache(path: Path, fingerprint: str, precision: str) -> dict[str, Any]:
    if not path.is_file():
        return _empty_cache(fingerprint, precision)
    cache = json.loads(path.read_text(encoding="utf-8"))
    expected = (CACHE_VERSION, MODEL, fingerprint, 512, precision)
    actual = (
        cache.get("version"),
        cache.get("model"),
        cache.get("corpus_fingerprint"),
        cache.get("max_length"),
        cache.get("precision"),
    )
    if actual != expected:
        raise ValueError("incompatible wide reranker score cache")
    return cache


def _save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _score_pools(
    cases: list[EvalCase],
    pools: list[list[Chunk]],
    *,
    split: str,
    cache_path: Path,
    fingerprint: str,
    batch_size: int,
    checkpoint_pairs: int,
    precision: str,
) -> tuple[list[dict[str, float]], float]:
    cache = _load_cache(cache_path, fingerprint, precision)
    split_cache = cache.setdefault("splits", {}).setdefault(split, {})
    elapsed_by_split = cache.setdefault("latency_seconds", {})
    elapsed = float(elapsed_by_split.get(split, 0.0))
    _save_json(cache_path, cache)

    scorer: SentenceTransformerCrossEncoder | None = None
    outputs: list[dict[str, float]] = []
    total_pairs = sum(len(pool) for pool in pools)
    completed = 0
    for case_index, (case, pool) in enumerate(zip(cases, pools, strict=True), start=1):
        case_cache = split_cache.setdefault(case.case_id, {})
        missing = [chunk for chunk in pool if chunk.chunk_id not in case_cache]
        completed += len(pool) - len(missing)
        for start in range(0, len(missing), checkpoint_pairs):
            block = missing[start : start + checkpoint_pairs]
            if scorer is None:
                if not LOCAL_MODEL.is_dir():
                    raise FileNotFoundError(
                        f"missing local model {LOCAL_MODEL}; download {MODEL} before offline run"
                    )
                scorer = SentenceTransformerCrossEncoder(
                    str(LOCAL_MODEL),
                    max_length=512,
                    batch_size=batch_size,
                )
            started = time.perf_counter()
            if precision == "bf16":
                import torch

                with torch.autocast("cpu", dtype=torch.bfloat16):
                    values = scorer.score(query_text(case, "blind"), block)
            else:
                values = scorer.score(query_text(case, "blind"), block)
            elapsed += time.perf_counter() - started
            case_cache.update(
                {chunk.chunk_id: float(score) for chunk, score in zip(block, values, strict=True)}
            )
            completed += len(block)
            elapsed_by_split[split] = elapsed
            _save_json(cache_path, cache)
            rate = completed / elapsed if elapsed else 0.0
            remaining = (total_pairs - completed) / rate if rate else 0.0
            print(
                f"cross-encoder {split}: case {case_index}/{len(cases)}, "
                f"pairs {completed}/{total_pairs}, {rate:.2f} pairs/s, "
                f"ETA {remaining / 60:.1f} min",
                file=sys.stderr,
                flush=True,
            )
        outputs.append({chunk.chunk_id: float(case_cache[chunk.chunk_id]) for chunk in pool})
    return outputs, elapsed * 1000 / len(cases)


def _serialize(row: SweepResult) -> dict[str, Any]:
    return {
        "spec": row.spec,
        "at5": _metric_dict(row.at5),
        "at10": _metric_dict(row.at10),
        "mean_articles": row.mean_articles,
        "mean_chunks": row.mean_chunks,
    }


def _evaluate_selected(
    cases: list[EvalCase],
    selected_articles: list[list[str]],
    pools: list[list[Chunk]],
    scores: list[dict[str, float]],
    *,
    spec: dict[str, Any],
) -> SweepResult:
    rankings = [
        best_chunk_per_article(pool, score, articles, limit=LIMIT)
        for articles, pool, score in zip(selected_articles, pools, scores, strict=True)
    ]
    return SweepResult(
        spec=spec,
        at5=_metrics(cases, rankings, 5),
        at10=_metrics(cases, rankings, 10),
        mean_articles=sum(map(len, selected_articles)) / len(cases),
        mean_chunks=sum(map(len, pools)) / len(cases),
        rankings=rankings,
    )


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


def _label(spec: dict[str, Any]) -> str:
    chunks = spec.get("chunks_per_article")
    chunks_label = "all" if chunks is None else str(chunks)
    return (
        f"dense={spec['dense_articles']}, BM25={spec['bm25_articles']}, "
        f"chunks/article={chunks_label}"
    )


def _write_report(state: dict[str, Any], path: Path) -> None:
    dev = state["dev"]
    quota_winner = dev["quota_winner"]
    winner = dev["winner"]
    interval = dev["bootstrap_vs_dense"]
    lines = [
        "# Wide article-first cross-encoder reranking",
        "",
        *( [
            "> **Статус:** исторический BF16-sweep; текущий two-pass selector и FP32-контроль "
            "описаны в `article_shortlist_fusion.md`.",
            "",
        ] if state["precision"] == "bf16" else []),
        f"Dense: `Roflmax/bge-m3-russian-legal`, m{DENSE_MAX_CHUNK}; "
        f"BM25: m{BM25_MAX_CHUNK}; cross-encoder: `{MODEL}` "
        f"(bilingual PyTorch {state['precision'].upper()}, CPU).",
        "Каждый источник возвращает до 50 уникальных статей. Квоты сравниваются на одинаковом "
        f"shortlist до {QUOTA_SCREEN_CHUNKS} m700-чанков на статью; после выбора квот для "
        "этих статей досчитываются все m700-чанки.",
        "Итоговый top-10 содержит лучший cross-encoder-чанк каждой статьи.",
        "",
        "## Зафиксированная конфигурация",
        "",
        f"Квоты по полному пулу: **dense {quota_winner['spec']['dense_articles']} + "
        f"BM25 {quota_winner['spec']['bm25_articles']}**. После sweep по стоимости: "
        f"**{_label(winner['spec'])}**.",
        f"Dev/blind: article R@10 {winner['at10']['recall_article']:.3f}, "
        f"exact R@10 {winner['at10']['recall_exact']:.3f}, "
        f"all-hit article@10 {winner['at10']['all_article']:.3f}.",
        f"Против dense m700 article-diverse: Δ article R@10 {interval['mean_delta']:+.3f} "
        f"(95% group-bootstrap [{interval['low']:+.3f}; {interval['high']:+.3f}]); "
        f"выигрышей {dev['outcomes_vs_dense']['gains']}, проигрышей "
        f"{dev['outcomes_vs_dense']['losses']}, без изменений {dev['outcomes_vs_dense']['ties']}.",
        "",
        f"## Парето-фронт квот статей, до {QUOTA_SCREEN_CHUNKS} чанков на статью",
        "",
        "| dense | BM25 | mean unique articles | mean reranked chunks | exact R@10 | article R@10 | all article@10 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dev["quota_pareto"]:
        lines.append(
            f"| {row['spec']['dense_articles']} | {row['spec']['bm25_articles']} | "
            f"{row['mean_articles']:.1f} | {row['mean_chunks']:.1f} | "
            f"{row['at10']['recall_exact']:.3f} | {row['at10']['recall_article']:.3f} | "
            f"{row['at10']['all_article']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Число cross-encoded чанков на статью при зафиксированных квотах",
            "",
            "| chunks/article | mean reranked chunks | exact R@5 | article R@5 | exact R@10 | article R@10 | all article@10 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in dev["chunk_sweep"]:
        value = row["spec"]["chunks_per_article"]
        label = "all" if value is None else str(value)
        lines.append(
            f"| {label} | {row['mean_chunks']:.1f} | {row['at5']['recall_exact']:.3f} | "
            f"{row['at5']['recall_article']:.3f} | {row['at10']['recall_exact']:.3f} | "
            f"{row['at10']['recall_article']:.3f} | {row['at10']['all_article']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Полный 50+50 candidate-пул до {QUOTA_SCREEN_CHUNKS} чанков/статью: в среднем "
            f"{dev['screen_pool']['mean_articles']:.1f} статьи и "
            f"{dev['screen_pool']['mean_chunks']:.1f} чанка на запрос; всего "
            f"{dev['screen_pool']['total_pairs']} пар. Для зафиксированных квот all-chunks-пул: "
            f"{dev['locked_full_pool']['mean_chunks']:.1f} чанка на запрос и "
            f"{dev['locked_full_pool']['total_pairs']} пар. Измеренное суммарное время cross-encoder: "
            f"{dev['cross_encoder_latency_ms'] / 1000:.1f} s/query.",
            "Полная сетка и инкрементальные scores хранятся локально в `data/analysis/`.",
        ]
    )
    if "eval" in state:
        evaluated = state["eval"]
        eval_interval = evaluated["bootstrap_vs_dense"]
        lines.extend(
            [
                "",
                "## Однократная проверка eval / blind",
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
                f"Locked reranker против dense: Δ article R@10 {eval_interval['mean_delta']:+.3f} "
                f"(95% CI [{eval_interval['low']:+.3f}; {eval_interval['high']:+.3f}]).",
                f"Измеренная cross-encoder latency зафиксированного варианта: "
                f"{evaluated['cross_encoder_latency_ms'] / 1000:.1f} s/query на CPU "
                "(retrieval и построение индексов не включены).",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    main["wide_article_rerank"] = {**state, "report": str(report)}
    main_path.write_text(json.dumps(main, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    from run_retrieval_error_analysis import _report as write_aggregate_report

    write_aggregate_report(main, aggregate_report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wide article-first cross-encoder experiment")
    parser.add_argument("--phase", choices=("dev", "eval"), default="dev")
    parser.add_argument("--edition-dir", type=Path, default=Path("data/law/2026-08-04"))
    parser.add_argument("--cases-dir", type=Path, default=Path("evals/cases"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("data/analysis"))
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--checkpoint-pairs", type=int, default=240)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--report", type=Path, default=Path("evals/reports/wide_article_rerank.md")
    )
    parser.add_argument(
        "--aggregate-report",
        type=Path,
        default=Path("evals/reports/retrieval_error_analysis.md"),
    )
    args = parser.parse_args(argv)
    cases = [case for case in load_verified(args.cases_dir) if case.split == args.phase]
    expected = 52 if args.phase == "dev" else 22
    if len(cases) != expected:
        print(f"expected {expected} {args.phase} cases, got {len(cases)}", file=sys.stderr)
        return 2

    state_path = args.analysis_dir / "wide_article_rerank_state.json"
    if args.phase == "eval" and not state_path.is_file():
        print("run the default dev phase first", file=sys.stderr)
        return 2

    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    dense_chunks, dense_index, bm25_indexes = _load_indexes(args.edition_dir)
    dense_full, dense_ms = _rank_cases(dense_index, cases, k=len(dense_chunks))
    bm25_full, bm25_ms = _rank_cases(
        bm25_indexes[BM25_MAX_CHUNK], cases, k=len(bm25_indexes[BM25_MAX_CHUNK].chunks)
    )
    bm25_m700_full, bm25_m700_ms = _rank_cases(
        Bm25Index(dense_chunks), cases, k=len(dense_chunks)
    )
    dense_articles = [unique_article_prefix(ranking, SOURCE_ARTICLES) for ranking in dense_full]
    bm25_articles = [unique_article_prefix(ranking, SOURCE_ARTICLES) for ranking in bm25_full]
    article_map = chunks_by_article(dense_chunks)

    if args.phase == "dev":
        full_articles = [
            source_quota_articles(dense, bm25, dense_quota=50, bm25_quota=50)
            for dense, bm25 in zip(dense_articles, bm25_articles, strict=True)
        ]
        score_pools = [
            article_candidate_pool(
                [dense, bm25], articles, per_article=QUOTA_SCREEN_CHUNKS
            )
            for dense, bm25, articles in zip(
                dense_full, bm25_m700_full, full_articles, strict=True
            )
        ]
    else:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("version") != STATE_VERSION or state.get("model") != MODEL:
            print("incompatible wide reranker state", file=sys.stderr)
            return 2
        if state.get("precision") != args.precision:
            print(
                f"eval precision must match locked dev precision {state.get('precision')}",
                file=sys.stderr,
            )
            return 2
        locked = state["dev"]["winner"]["spec"]
        selected = [
            source_quota_articles(
                dense,
                bm25,
                dense_quota=int(locked["dense_articles"]),
                bm25_quota=int(locked["bm25_articles"]),
            )
            for dense, bm25 in zip(dense_articles, bm25_articles, strict=True)
        ]
        if locked["chunks_per_article"] is None:
            score_pools = [all_chunks_for_articles(articles, article_map) for articles in selected]
        else:
            score_pools = [
                article_candidate_pool(
                    [dense, bm25], articles, per_article=int(locked["chunks_per_article"])
                )
                for dense, bm25, articles in zip(
                    dense_full, bm25_m700_full, selected, strict=True
                )
            ]

    scores, cross_encoder_ms = _score_pools(
        cases,
        score_pools,
        split=args.phase,
        cache_path=reranker_cache_path(args.analysis_dir, args.precision),
        fingerprint=corpus_fingerprint(dense_chunks),
        batch_size=args.batch_size,
        checkpoint_pairs=args.checkpoint_pairs,
        precision=args.precision,
    )
    dense_baseline_rankings = [article_diversify(row, pool=len(row), limit=LIMIT) for row in dense_full]
    dense_baseline = {
        "at5": _metric_dict(_metrics(cases, dense_baseline_rankings, 5)),
        "at10": _metric_dict(_metrics(cases, dense_baseline_rankings, 10)),
        "latency_ms": dense_ms,
    }

    if args.phase == "dev":
        quota_results: list[SweepResult] = []
        for dense_quota in QUOTAS:
            for bm25_quota in QUOTAS:
                if dense_quota + bm25_quota < LIMIT:
                    continue
                selected = [
                    source_quota_articles(
                        dense,
                        bm25,
                        dense_quota=dense_quota,
                        bm25_quota=bm25_quota,
                    )
                    for dense, bm25 in zip(dense_articles, bm25_articles, strict=True)
                ]
                pools = [
                    article_candidate_pool(
                        [dense, bm25], articles, per_article=QUOTA_SCREEN_CHUNKS
                    )
                    for dense, bm25, articles in zip(
                        dense_full, bm25_m700_full, selected, strict=True
                    )
                ]
                quota_results.append(
                    _evaluate_selected(
                        cases,
                        selected,
                        pools,
                        scores,
                        spec={"dense_articles": dense_quota, "bm25_articles": bm25_quota},
                    )
                )
        quota_winner = select_winner(quota_results)
        quota_spec = QuotaSpec(**quota_winner.spec)
        locked_articles = [
            source_quota_articles(
                dense,
                bm25,
                dense_quota=quota_spec.dense_articles,
                bm25_quota=quota_spec.bm25_articles,
            )
            for dense, bm25 in zip(dense_articles, bm25_articles, strict=True)
        ]
        locked_full_pools = [
            all_chunks_for_articles(articles, article_map) for articles in locked_articles
        ]
        locked_scores, cross_encoder_ms = _score_pools(
            cases,
            locked_full_pools,
            split=args.phase,
            cache_path=reranker_cache_path(args.analysis_dir, args.precision),
            fingerprint=corpus_fingerprint(dense_chunks),
            batch_size=args.batch_size,
            checkpoint_pairs=args.checkpoint_pairs,
            precision=args.precision,
        )
        chunk_results: list[SweepResult] = []
        for per_article in CHUNKS_PER_ARTICLE:
            if per_article is None:
                pools = [all_chunks_for_articles(articles, article_map) for articles in locked_articles]
            else:
                pools = [
                    article_candidate_pool(
                        [dense, bm25], articles, per_article=per_article
                    )
                    for dense, bm25, articles in zip(
                        dense_full, bm25_m700_full, locked_articles, strict=True
                    )
                ]
            chunk_results.append(
                _evaluate_selected(
                    cases,
                    locked_articles,
                    pools,
                    locked_scores,
                    spec={
                        **quota_winner.spec,
                        "chunks_per_article": per_article,
                    },
                )
            )
        winner = select_winner(chunk_results)
        interval = paired_group_bootstrap(
            per_case_article_recall(cases, dense_baseline_rankings),
            per_case_article_recall(cases, winner.rankings),
            [case.case_group_id for case in cases],
            seed=SEED,
        )
        state = {
            "version": STATE_VERSION,
            "model": MODEL,
            "precision": args.precision,
            "query": "blind",
            "seed": SEED,
            "dense_model": "Roflmax/bge-m3-russian-legal",
            "dense_max_chunk": DENSE_MAX_CHUNK,
            "bm25_max_chunk": BM25_MAX_CHUNK,
            "source_articles": SOURCE_ARTICLES,
            "dev": {
                "n": len(cases),
                "baseline": dense_baseline,
                "retrieval_latency_ms": dense_ms + bm25_ms + bm25_m700_ms,
                "cross_encoder_latency_ms": cross_encoder_ms,
                "screen_pool": {
                    "mean_articles": sum(map(len, full_articles)) / len(cases),
                    "mean_chunks": sum(map(len, score_pools)) / len(cases),
                    "total_pairs": sum(map(len, score_pools)),
                },
                "locked_full_pool": {
                    "mean_articles": sum(map(len, locked_articles)) / len(cases),
                    "mean_chunks": sum(map(len, locked_full_pools)) / len(cases),
                    "total_pairs": sum(map(len, locked_full_pools)),
                },
                "quota_candidates": [_serialize(row) for row in quota_results],
                "quota_pareto": [_serialize(row) for row in pareto_frontier(quota_results)],
                "quota_winner": _serialize(quota_winner),
                "chunk_sweep": [_serialize(row) for row in chunk_results],
                "winner": _serialize(winner),
                "bootstrap_vs_dense": asdict(interval),
                "outcomes_vs_dense": _outcomes(cases, dense_baseline_rankings, winner.rankings),
            },
        }
        _save_json(state_path, state)
        print(f"locked dev winner: {_label(winner.spec)}", file=sys.stderr)
    else:
        locked = state["dev"]["winner"]["spec"]
        selected = [
            source_quota_articles(
                dense,
                bm25,
                dense_quota=int(locked["dense_articles"]),
                bm25_quota=int(locked["bm25_articles"]),
            )
            for dense, bm25 in zip(dense_articles, bm25_articles, strict=True)
        ]
        result = _evaluate_selected(cases, selected, score_pools, scores, spec=locked)
        interval = paired_group_bootstrap(
            per_case_article_recall(cases, dense_baseline_rankings),
            per_case_article_recall(cases, result.rankings),
            [case.case_group_id for case in cases],
            seed=SEED,
        )
        state["eval"] = {
            "n": len(cases),
            "cross_encoder_latency_ms": cross_encoder_ms,
            "methods": {
                "dense m700 article-diverse": dense_baseline,
                "locked wide cross-encoder": _serialize(result),
            },
            "bootstrap_vs_dense": asdict(interval),
            "outcomes_vs_dense": _outcomes(cases, dense_baseline_rankings, result.rankings),
        }
        _save_json(state_path, state)

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
