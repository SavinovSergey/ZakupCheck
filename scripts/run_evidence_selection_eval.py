#!/usr/bin/env python3
"""Optimize chunk candidates and evidence allocation inside locked articles."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from index.bm25 import Bm25Index  # noqa: E402
from index.dense import corpus_fingerprint  # noqa: E402
from index.error_analysis import aggregate_rankings, paired_group_bootstrap  # noqa: E402
from index.eval_tables import load_verified  # noqa: E402
from index.fusion import fuse_article_rankings, rank_articles_with_retrieval_bonus  # noqa: E402
from index.recall import AggregateRecall, article_key, score_case  # noqa: E402
from index.rerank import (  # noqa: E402
    article_candidate_map,
    article_candidate_pool_from_map,
    build_evidence_pack,
    diverse_chunk_selection,
    rank_articles_by_best_chunk,
)
from run_article_chunk_budget_eval import (  # noqa: E402
    _article_metrics,
    _pool_metrics,
    _scores_for_cases,
    _selected_articles,
)
from run_article_shortlist_fusion_eval import _shortlist  # noqa: E402
from run_hybrid_retrieval_eval import _load_indexes, _metric_dict, _rank_cases  # noqa: E402
from run_wide_article_rerank_eval import (  # noqa: E402
    MODEL,
    _score_pools,
    reranker_cache_path,
    unique_article_prefix,
)
from schemas.corpus import Chunk  # noqa: E402
from schemas.eval_case import EvalCase  # noqa: E402

DENSE_MAX_CHUNK = 700
BM25_MAX_CHUNK = 1500
SOURCE_ARTICLES = 50
SOURCE_CHUNKS_PER_ARTICLE = 24
SEED = 20261008
STATE_VERSION = 1
CANDIDATE_SIZES = (4, 8, 12, 16)
DIVERSITY_WEIGHTS = (0.25, 0.5)
EVIDENCE_MODES = (
    ("one_per_article", 10, 1, 10),
    ("top10_primary7_cap2", 10, 2, 7),
    ("top10_primary5_cap2", 10, 2, 5),
    ("pack20_cap2", 20, 2, 10),
    ("pack30_cap3", 30, 3, 10),
)


@dataclass(frozen=True)
class EvidenceResult:
    spec: dict[str, Any]
    metrics: AggregateRecall
    pool_metrics: AggregateRecall
    article_metrics: AggregateRecall
    mean_scored_chunks: float
    max_scored_chunks: int
    mean_output_chunks: float
    rankings: list[list[Chunk]]


def _source_chunks(
    ranking: list[Chunk],
    articles: list[str],
    *,
    per_article: int,
) -> tuple[dict[str, list[Chunk]], dict[str, int]]:
    selected = set(articles)
    output = {article: [] for article in articles}
    seen = {article: set() for article in articles}
    ranks: dict[str, int] = {}
    for rank, chunk in enumerate(ranking, start=1):
        ranks.setdefault(chunk.chunk_id, rank)
        for article in sorted({article_key(norm_id) for norm_id in chunk.norm_ids} & selected):
            if len(output[article]) >= per_article or chunk.chunk_id in seen[article]:
                continue
            output[article].append(chunk)
            seen[article].add(chunk.chunk_id)
        if all(len(chunks) >= per_article for chunks in output.values()):
            break
    return output, ranks


def _rrf_candidates(
    dense: dict[str, list[Chunk]],
    bm25: dict[str, list[Chunk]],
    dense_ranks: dict[str, int],
    bm25_ranks: dict[str, int],
    articles: list[str],
    *,
    limit: int,
    diverse: bool,
    diversity_weight: float = 0.0,
) -> dict[str, list[Chunk]]:
    output: dict[str, list[Chunk]] = {}
    for article in articles:
        candidates = {
            chunk.chunk_id: chunk for chunk in dense.get(article, []) + bm25.get(article, [])
        }
        scores = {
            chunk_id: (
                (1.0 / (30 + dense_ranks[chunk_id]) if chunk_id in dense_ranks else 0.0)
                + (1.0 / (30 + bm25_ranks[chunk_id]) if chunk_id in bm25_ranks else 0.0)
            )
            for chunk_id in candidates
        }
        ranked = sorted(candidates.values(), key=lambda chunk: (-scores[chunk.chunk_id], chunk.chunk_id))
        output[article] = (
            diverse_chunk_selection(
                ranked,
                scores,
                limit=limit,
                diversity_weight=diversity_weight,
            )
            if diverse
            else ranked[:limit]
        )
    return output


def _selector_specs() -> list[dict[str, Any]]:
    specs = [
        {"candidate_selector": "interleave", "candidates_per_article": size}
        for size in CANDIDATE_SIZES
    ]
    specs.extend(
        {"candidate_selector": "chunk_rrf", "candidates_per_article": size}
        for size in CANDIDATE_SIZES
    )
    specs.extend(
        {
            "candidate_selector": "diverse_rrf",
            "candidates_per_article": size,
            "diversity_weight": weight,
        }
        for size in CANDIDATE_SIZES
        for weight in DIVERSITY_WEIGHTS
    )
    return specs


def _ranker_specs() -> list[dict[str, Any]]:
    return [
        {"evidence_ranker": "ce"},
        {
            "evidence_ranker": "chunk_rrf",
            "dense_weight": 1,
            "bm25_weight": 1,
            "ce_weight": 1,
            "k": 30,
        },
        {
            "evidence_ranker": "chunk_rrf",
            "dense_weight": 1,
            "bm25_weight": 1,
            "ce_weight": 2,
            "k": 30,
        },
        *(
            {"evidence_ranker": "ce_retrieval_bonus", "alpha": alpha, "k": 10}
            for alpha in (0.1, 0.25, 0.5)
        ),
    ]


def _rank_evidence(
    spec: dict[str, Any],
    candidate_map: dict[str, list[Chunk]],
    scores: dict[str, float],
    dense: dict[str, list[Chunk]],
    bm25: dict[str, list[Chunk]],
    articles: list[str],
) -> tuple[dict[str, list[Chunk]], dict[str, float]]:
    ranked_by_article: dict[str, list[Chunk]] = {}
    global_scores: dict[str, float] = {}
    for article in articles:
        chunks = candidate_map.get(article, [])
        ce_order = sorted(chunks, key=lambda chunk: (-scores[chunk.chunk_id], chunk.chunk_id))
        ids = [chunk.chunk_id for chunk in chunks]
        dense_ids = [chunk.chunk_id for chunk in dense.get(article, [])]
        bm25_ids = [chunk.chunk_id for chunk in bm25.get(article, [])]
        if spec["evidence_ranker"] == "ce":
            ordered_ids = [chunk.chunk_id for chunk in ce_order]
            local_scores = {chunk_id: scores[chunk_id] for chunk_id in ordered_ids}
        elif spec["evidence_ranker"] == "chunk_rrf":
            ordered_ids = fuse_article_rankings(
                [dense_ids, bm25_ids, [chunk.chunk_id for chunk in ce_order]],
                candidates=ids,
                weights=[spec["dense_weight"], spec["bm25_weight"], spec["ce_weight"]],
                k=spec["k"],
                limit=len(ids),
            )
            local_scores = {
                chunk_id: 1.0 / rank + 1e-3 * scores[chunk_id]
                for rank, chunk_id in enumerate(ordered_ids, start=1)
            }
        else:
            raw = {chunk.chunk_id: scores[chunk.chunk_id] for chunk in chunks}
            ordered_ids = rank_articles_with_retrieval_bonus(
                raw,
                dense_ids,
                bm25_ids,
                candidates=ids,
                alpha=spec["alpha"],
                k=spec["k"],
                limit=len(ids),
            )
            local_scores = {
                chunk_id: 1.0 / rank + 1e-3 * scores[chunk_id]
                for rank, chunk_id in enumerate(ordered_ids, start=1)
            }
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        ranked_by_article[article] = [by_id[chunk_id] for chunk_id in ordered_ids]
        global_scores.update(local_scores)
    return ranked_by_article, global_scores


def _exact_values(cases: list[EvalCase], rankings: list[list[Chunk]], k: int) -> list[float]:
    values = []
    for case, ranking in zip(cases, rankings, strict=True):
        norm_ids = {norm_id for chunk in ranking[:k] for norm_id in chunk.norm_ids}
        values.append(score_case(case.gold_norms, norm_ids).exact.recall)
    return values


def _serialize(row: EvidenceResult) -> dict[str, Any]:
    return {
        "spec": row.spec,
        "metrics": _metric_dict(row.metrics),
        "pool_metrics": _metric_dict(row.pool_metrics),
        "article_metrics": _metric_dict(row.article_metrics),
        "mean_scored_chunks": row.mean_scored_chunks,
        "max_scored_chunks": row.max_scored_chunks,
        "mean_output_chunks": row.mean_output_chunks,
    }


def _select(rows: list[EvidenceResult]) -> EvidenceResult:
    if not rows:
        raise ValueError("at least one evidence result is required")
    return min(
        rows,
        key=lambda row: (
            -row.metrics.recall_article,
            -row.metrics.recall_exact,
            -row.metrics.all_exact,
            row.mean_scored_chunks,
            tuple(sorted(row.spec.items())),
        ),
    )


def _select_exact(rows: list[EvidenceResult]) -> EvidenceResult:
    if not rows:
        raise ValueError("at least one evidence result is required")
    return min(
        rows,
        key=lambda row: (
            -row.metrics.recall_exact,
            -row.metrics.all_exact,
            -row.metrics.recall_article,
            row.mean_scored_chunks,
            tuple(sorted(row.spec.items())),
        ),
    )


def _label(row: dict[str, Any]) -> str:
    spec = row["spec"]
    selector = f"{spec['candidate_selector']}:{spec['candidates_per_article']}"
    if spec["candidate_selector"] == "diverse_rrf":
        selector += f"@{spec['diversity_weight']}"
    ranker = spec["evidence_ranker"]
    if ranker == "chunk_rrf":
        ranker += f" {spec['dense_weight']}:{spec['bm25_weight']}:{spec['ce_weight']}"
    if ranker == "ce_retrieval_bonus":
        ranker += f" alpha={spec['alpha']}"
    return f"{selector}; {ranker}; {spec['mode']}"


def _write_report(state: dict[str, Any], path: Path) -> None:
    top10 = state["best_top10"]
    exact_top10 = state["best_exact_top10"]
    one = state["best_one_per_article"]
    pack20 = state["best_pack20"]
    pack30 = state["best_pack30"]
    interval = state["top10_vs_baseline_exact"]
    exact_top10_interval = state["exact_top10_vs_baseline_exact"]
    pack20_interval = state["pack20_vs_baseline_exact"]
    pack30_interval = state["pack30_vs_baseline_exact"]
    lines = [
        "# Evidence selection inside locked articles",
        "",
        "> **Статус:** историческая широкая сетка на BF16-кэше. Рабочая сетка `4/8` "
        "кандидатов/статью повторно проверена в FP32; см. `m1000_evidence_comparison.md`.",
        "",
        "Article shortlist is frozen to the dev winner: D10+B10, first=2, three-way article RRF "
        "k=30 with weights 1:1:2, top 10 articles. Eval is not used for tuning.",
        "Candidate selectors compare source interleaving, chunk-level RRF, and diversity-aware RRF. "
        "Evidence rankers compare CE, chunk-level dense+BM25+CE RRF, and CE with retrieval bonus.",
        "",
        "## Best 10-chunk result",
        "",
        f"**{_label(top10)}**: article R@10 {top10['metrics']['recall_article']:.3f}, exact R@10 "
        f"{top10['metrics']['recall_exact']:.3f}. Exact pool ceiling "
        f"{top10['pool_metrics']['recall_exact']:.3f}; mean/max scored candidates "
        f"{top10['mean_scored_chunks']:.1f}/{top10['max_scored_chunks']}.",
        f"Against interleave:8 + CE + one/article, exact Δ "
        f"{interval['mean_delta']:+.3f} (95% group-bootstrap "
        f"[{interval['low']:+.3f}; {interval['high']:+.3f}]).",
        "",
        "## One chunk per article and expanded evidence packs",
        "",
        "| output | configuration | output chunks | exact recall | all exact | article recall | pool exact ceiling |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in (
        ("one/article", one),
        ("best top-10", top10),
        ("exact-first top-10", exact_top10),
        ("pack-20", pack20),
        ("pack-30", pack30),
    ):
        lines.append(
            f"| {name} | {_label(row)} | {row['mean_output_chunks']:.1f} | "
            f"{row['metrics']['recall_exact']:.3f} | {row['metrics']['all_exact']:.3f} | "
            f"{row['metrics']['recall_article']:.3f} | "
            f"{row['pool_metrics']['recall_exact']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"If top-10 is optimized for exact evidence instead of article recall, exact R@10 reaches "
            f"{exact_top10['metrics']['recall_exact']:.3f} (Δ "
            f"{exact_top10_interval['mean_delta']:+.3f}, 95% CI "
            f"[{exact_top10_interval['low']:+.3f}; {exact_top10_interval['high']:+.3f}]), "
            f"but article R@10 falls to {exact_top10['metrics']['recall_article']:.3f}.",
            f"Against the 10-chunk baseline, pack-20 exact recall Δ "
            f"{pack20_interval['mean_delta']:+.3f} (95% group-bootstrap "
            f"[{pack20_interval['low']:+.3f}; {pack20_interval['high']:+.3f}]); "
            f"pack-30 Δ {pack30_interval['mean_delta']:+.3f} "
            f"([{pack30_interval['low']:+.3f}; {pack30_interval['high']:+.3f}]).",
            "Chunk-level dense/BM25 bonuses did not beat plain cross-encoder ordering. "
            "Diversity is useful for a broad evidence pack, not for choosing the single best chunk per article.",
            "",
            f"Evaluated {state['count']} configurations on {state['n']} dev/blind cases. "
            "Full results are local in `data/analysis/evidence_selection_state.json`.",
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
    main["evidence_selection"] = {
        key: state[key]
        for key in (
            "version",
            "n",
            "count",
            "article_selector",
            "baseline",
            "best_one_per_article",
            "best_top10",
            "best_exact_top10",
            "best_pack20",
            "best_pack30",
            "top10_vs_baseline_exact",
            "exact_top10_vs_baseline_exact",
            "pack20_vs_baseline_exact",
            "pack30_vs_baseline_exact",
        )
    }
    main["evidence_selection"]["report"] = str(report)
    main_path.write_text(json.dumps(main, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    from run_retrieval_error_analysis import _report as write_aggregate_report

    write_aggregate_report(main, aggregate_report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edition-dir", type=Path, default=Path("data/law/2026-08-04"))
    parser.add_argument("--cases-dir", type=Path, default=Path("evals/cases"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("data/analysis"))
    parser.add_argument(
        "--report", type=Path, default=Path("evals/reports/evidence_selection.md")
    )
    parser.add_argument(
        "--aggregate-report",
        type=Path,
        default=Path("evals/reports/retrieval_error_analysis.md"),
    )
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--checkpoint-pairs", type=int, default=240)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    args = parser.parse_args(argv)

    cases = [case for case in load_verified(args.cases_dir) if case.split == "dev"]
    if len(cases) != 52:
        print(f"expected 52 dev cases, got {len(cases)}", file=sys.stderr)
        return 2
    shortlist_state = json.loads(
        (args.analysis_dir / "article_shortlist_fusion_state.json").read_text(encoding="utf-8")
    )
    locked = shortlist_state["winner"]["spec"]
    dense_chunks, dense_index, bm25_indexes = _load_indexes(args.edition_dir)
    dense_full, _ = _rank_cases(dense_index, cases, k=len(dense_chunks))
    bm25_full, _ = _rank_cases(
        bm25_indexes[BM25_MAX_CHUNK], cases, k=len(bm25_indexes[BM25_MAX_CHUNK].chunks)
    )
    bm25_m700_full, _ = _rank_cases(Bm25Index(dense_chunks), cases, k=len(dense_chunks))
    dense_articles = [unique_article_prefix(row, SOURCE_ARTICLES) for row in dense_full]
    bm25_articles = [unique_article_prefix(row, SOURCE_ARTICLES) for row in bm25_full]
    selected = _selected_articles(
        dense_articles,
        bm25_articles,
        int(locked["dense_articles"]),
        int(locked["bm25_articles"]),
    )
    first_maps = [
        article_candidate_map([dense, bm25], articles, per_article=int(locked["first_pass_chunks"]))
        for dense, bm25, articles in zip(
            dense_full, bm25_m700_full, selected, strict=True
        )
    ]
    first_pools = [
        article_candidate_pool_from_map(mapping, articles, per_article=int(locked["first_pass_chunks"]))
        for mapping, articles in zip(first_maps, selected, strict=True)
    ]
    first_scores = _scores_for_cases(
        reranker_cache_path(args.analysis_dir, args.precision),
        cases,
        first_pools,
        corpus_fingerprint(dense_chunks),
    )
    shortlists = []
    for articles, dense, bm25, pool, score in zip(
        selected, dense_articles, bm25_articles, first_pools, first_scores, strict=True
    ):
        ce_ranking, evidence = rank_articles_by_best_chunk(pool, score, articles)
        ce_scores = {article: score[evidence[article].chunk_id] for article in ce_ranking}
        shortlists.append(
            _shortlist(
                locked,
                candidates=articles,
                dense_articles=dense,
                bm25_articles=bm25,
                ce_ranking=ce_ranking,
                ce_scores=ce_scores,
                limit=int(locked["shortlist_articles"]),
            )
        )

    source_data = []
    selector_maps: dict[str, list[dict[str, list[Chunk]]]] = {
        json.dumps(spec, sort_keys=True): [] for spec in _selector_specs()
    }
    for dense, bm25, articles in zip(dense_full, bm25_m700_full, shortlists, strict=True):
        dense_map, dense_ranks = _source_chunks(
            dense, articles, per_article=SOURCE_CHUNKS_PER_ARTICLE
        )
        bm25_map, bm25_ranks = _source_chunks(
            bm25, articles, per_article=SOURCE_CHUNKS_PER_ARTICLE
        )
        interleave_map = article_candidate_map(
            [dense, bm25], articles, per_article=max(CANDIDATE_SIZES)
        )
        source_data.append((dense_map, bm25_map))
        for spec in _selector_specs():
            if spec["candidate_selector"] == "interleave":
                mapping = {
                    article: chunks[: spec["candidates_per_article"]]
                    for article, chunks in interleave_map.items()
                }
            else:
                mapping = _rrf_candidates(
                    dense_map,
                    bm25_map,
                    dense_ranks,
                    bm25_ranks,
                    articles,
                    limit=spec["candidates_per_article"],
                    diverse=spec["candidate_selector"] == "diverse_rrf",
                    diversity_weight=spec.get("diversity_weight", 0.0),
                )
            selector_maps[json.dumps(spec, sort_keys=True)].append(mapping)

    score_pools = []
    for case_index, articles in enumerate(shortlists):
        by_id: dict[str, Chunk] = {}
        for mappings in selector_maps.values():
            for chunks in mappings[case_index].values():
                by_id.update({chunk.chunk_id: chunk for chunk in chunks})
        score_pools.append([by_id[chunk_id] for chunk_id in sorted(by_id)])
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    scores, _ = _score_pools(
        cases,
        score_pools,
        split="dev",
        cache_path=reranker_cache_path(args.analysis_dir, args.precision),
        fingerprint=corpus_fingerprint(dense_chunks),
        batch_size=args.batch_size,
        checkpoint_pairs=args.checkpoint_pairs,
        precision=args.precision,
    )

    article_metrics = _article_metrics(cases, shortlists)
    results: list[EvidenceResult] = []
    for selector_spec in _selector_specs():
        mappings = selector_maps[json.dumps(selector_spec, sort_keys=True)]
        pools = [
            article_candidate_pool_from_map(
                mapping, articles, per_article=selector_spec["candidates_per_article"]
            )
            for mapping, articles in zip(mappings, shortlists, strict=True)
        ]
        pool_metrics = _pool_metrics(cases, pools)
        for ranker_spec in _ranker_specs():
            evidence = [
                _rank_evidence(ranker_spec, mapping, score, dense, bm25, articles)
                for mapping, score, (dense, bm25), articles in zip(
                    mappings, scores, source_data, shortlists, strict=True
                )
            ]
            for mode, budget, cap, primary in EVIDENCE_MODES:
                rankings = [
                    build_evidence_pack(
                        articles,
                        ranked,
                        global_scores,
                        budget=budget,
                        per_article_cap=cap,
                        primary_articles=primary,
                    )
                    for articles, (ranked, global_scores) in zip(
                        shortlists, evidence, strict=True
                    )
                ]
                results.append(
                    EvidenceResult(
                        spec={**selector_spec, **ranker_spec, "mode": mode, "budget": budget},
                        metrics=aggregate_rankings(cases, rankings, k=budget),
                        pool_metrics=pool_metrics,
                        article_metrics=article_metrics,
                        mean_scored_chunks=sum(map(len, pools)) / len(cases),
                        max_scored_chunks=max(map(len, pools)),
                        mean_output_chunks=sum(map(len, rankings)) / len(cases),
                        rankings=rankings,
                    )
                )

    baseline = next(
        row
        for row in results
        if row.spec["candidate_selector"] == "interleave"
        and row.spec["candidates_per_article"] == 8
        and row.spec["evidence_ranker"] == "ce"
        and row.spec["mode"] == "one_per_article"
    )
    one = _select_exact([row for row in results if row.spec["mode"] == "one_per_article"])
    top10 = _select([row for row in results if row.spec["budget"] == 10])
    exact_top10 = _select_exact([row for row in results if row.spec["budget"] == 10])
    pack20 = _select_exact([row for row in results if row.spec["mode"] == "pack20_cap2"])
    pack30 = _select_exact([row for row in results if row.spec["mode"] == "pack30_cap3"])
    exact_interval = paired_group_bootstrap(
        _exact_values(cases, baseline.rankings, 10),
        _exact_values(cases, top10.rankings, 10),
        [case.case_group_id for case in cases],
        seed=SEED,
    )
    pack20_interval = paired_group_bootstrap(
        _exact_values(cases, baseline.rankings, 10),
        _exact_values(cases, pack20.rankings, 20),
        [case.case_group_id for case in cases],
        seed=SEED,
    )
    exact_top10_interval = paired_group_bootstrap(
        _exact_values(cases, baseline.rankings, 10),
        _exact_values(cases, exact_top10.rankings, 10),
        [case.case_group_id for case in cases],
        seed=SEED,
    )
    pack30_interval = paired_group_bootstrap(
        _exact_values(cases, baseline.rankings, 10),
        _exact_values(cases, pack30.rankings, 30),
        [case.case_group_id for case in cases],
        seed=SEED,
    )
    state = {
        "version": STATE_VERSION,
        "query": "blind",
        "split": "dev",
        "n": len(cases),
        "count": len(results),
        "article_selector": locked,
        "baseline": _serialize(baseline),
        "best_one_per_article": _serialize(one),
        "best_top10": _serialize(top10),
        "best_exact_top10": _serialize(exact_top10),
        "best_pack20": _serialize(pack20),
        "best_pack30": _serialize(pack30),
        "top10_vs_baseline_exact": asdict(exact_interval),
        "exact_top10_vs_baseline_exact": asdict(exact_top10_interval),
        "pack20_vs_baseline_exact": asdict(pack20_interval),
        "pack30_vs_baseline_exact": asdict(pack30_interval),
        "results": [_serialize(row) for row in results],
    }
    state_path = args.analysis_dir / "evidence_selection_state.json"
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
