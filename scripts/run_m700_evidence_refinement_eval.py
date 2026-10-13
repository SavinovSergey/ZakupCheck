#!/usr/bin/env python3
"""Evaluate neighbour, hierarchy, context, and token-budget evidence selection on m700."""

from __future__ import annotations

import argparse
import json
import math
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
from index.recall import AggregateRecall, article_key, score_case  # noqa: E402
from index.rerank import (  # noqa: E402
    article_candidate_map,
    article_candidate_pool_from_map,
    article_neighbors,
    build_adaptive_evidence_pack,
    build_evidence_pack,
    rank_articles_by_best_chunk,
)
from run_article_chunk_budget_eval import (  # noqa: E402
    _scores_for_cases,
    _selected_articles,
)
from run_article_shortlist_fusion_eval import _shortlist  # noqa: E402
from run_hybrid_retrieval_eval import _load_indexes, _metric_dict, _rank_cases  # noqa: E402
from run_wide_article_rerank_eval import (  # noqa: E402
    _score_pools,
    reranker_cache_path,
    unique_article_prefix,
)
from schemas.corpus import Chunk  # noqa: E402
from schemas.eval_case import EvalCase  # noqa: E402

DENSE_MAX_CHUNK = 700
BM25_MAX_CHUNK = 1500
SOURCE_ARTICLES = 50
SEED = 20261008
STATE_VERSION = 1
TOKEN_BUDGETS = (2000, 2500, 3000, 3500, 4000)
SCORE_RATIOS = (0.0, 0.25, 0.5, 0.75)
HIERARCHY_BONUSES = (0.0, 0.05, 0.1)


@dataclass(frozen=True)
class Result:
    name: str
    spec: dict[str, Any]
    metrics: AggregateRecall
    pool_metrics: AggregateRecall
    mean_chunks: float
    mean_estimated_tokens: float
    max_estimated_tokens: int
    rankings: list[list[Chunk]]


def _norm_ids(chunks: list[Chunk]) -> set[str]:
    return {norm_id for chunk in chunks for norm_id in chunk.norm_ids}


def _pool_metrics(cases: list[EvalCase], pools: list[list[Chunk]]) -> AggregateRecall:
    scores = [
        score_case(case.gold_norms, _norm_ids(pool))
        for case, pool in zip(cases, pools, strict=True)
    ]
    from index.recall import aggregate

    return aggregate(scores)


def _token_cost(chunk: Chunk) -> int:
    """Tokenizer-independent Russian-context estimate, approximately chars / 3."""
    return max(1, math.ceil(len(chunk.text) / 3))


def _context_chunk(chunk: Chunk, titles: dict[str, str]) -> Chunk:
    first = chunk.norm_ids[0]
    article = article_key(first)
    article_number = article.removeprefix("art.")
    suffix = first.split(":art.", 1)[-1].split(":")[1:]
    labels = []
    for piece in suffix:
        if piece.startswith("ch."):
            labels.append(f"часть {piece[3:]}")
        elif piece.startswith("p."):
            labels.append(f"пункт {piece[2:]}")
    header = f"Статья {article_number}. {titles.get(article, '')}".strip()
    if labels:
        header += "\nСтруктура: " + ", ".join(labels)
    return chunk.model_copy(
        update={"chunk_id": chunk.chunk_id + "::context", "text": header + "\n" + chunk.text}
    )


def _ranked_map(
    mapping: dict[str, list[Chunk]], scores: dict[str, float]
) -> tuple[dict[str, list[Chunk]], dict[str, float]]:
    ranked = {
        article: sorted(chunks, key=lambda chunk: (-scores[chunk.chunk_id], chunk.chunk_id))
        for article, chunks in mapping.items()
    }
    return ranked, {
        chunk.chunk_id: scores[chunk.chunk_id]
        for chunks in mapping.values()
        for chunk in chunks
    }


def _result(
    name: str,
    spec: dict[str, Any],
    cases: list[EvalCase],
    rankings: list[list[Chunk]],
    pools: list[list[Chunk]],
) -> Result:
    totals = [sum(_token_cost(chunk) for chunk in ranking) for ranking in rankings]
    return Result(
        name=name,
        spec=spec,
        metrics=aggregate_rankings(cases, rankings, k=max(map(len, rankings), default=0)),
        pool_metrics=_pool_metrics(cases, pools),
        mean_chunks=sum(map(len, rankings)) / len(rankings),
        mean_estimated_tokens=sum(totals) / len(totals),
        max_estimated_tokens=max(totals, default=0),
        rankings=rankings,
    )


def _serialize(result: Result) -> dict[str, Any]:
    return {
        "name": result.name,
        "spec": result.spec,
        "metrics": _metric_dict(result.metrics),
        "pool_metrics": _metric_dict(result.pool_metrics),
        "mean_chunks": result.mean_chunks,
        "mean_estimated_tokens": result.mean_estimated_tokens,
        "max_estimated_tokens": result.max_estimated_tokens,
    }


def _exact_values(cases: list[EvalCase], rankings: list[list[Chunk]]) -> list[float]:
    return [
        score_case(case.gold_norms, _norm_ids(ranking)).exact.recall
        for case, ranking in zip(cases, rankings, strict=True)
    ]


def _comparison(
    cases: list[EvalCase], baseline: Result, candidate: Result
) -> dict[str, Any]:
    return asdict(
        paired_group_bootstrap(
            _exact_values(cases, baseline.rankings),
            _exact_values(cases, candidate.rankings),
            [case.case_group_id for case in cases],
            seed=SEED,
        )
    )


def _select(rows: list[Result]) -> Result:
    return min(
        rows,
        key=lambda row: (
            -row.metrics.recall_exact,
            -row.metrics.all_exact,
            -row.metrics.recall_article,
            row.mean_estimated_tokens,
            row.mean_chunks,
            row.name,
        ),
    )


def _oracle_breakdown(
    cases: list[EvalCase],
    articles: list[list[str]],
    corpus: list[Chunk],
    candidate_pools: list[list[Chunk]],
    final: list[list[Chunk]],
) -> dict[str, int]:
    corpus_ids = _norm_ids(corpus)
    counts = {
        "hit": 0,
        "article_not_selected": 0,
        "exact_unavailable_in_m700": 0,
        "candidate_miss": 0,
        "selection_miss": 0,
    }
    for case, selected, pool, ranking in zip(
        cases, articles, candidate_pools, final, strict=True
    ):
        selected_set = set(selected)
        pool_ids = _norm_ids(pool)
        final_ids = _norm_ids(ranking)
        for gold_id in case.gold_norms:
            if gold_id in final_ids:
                counts["hit"] += 1
            elif gold_id not in corpus_ids:
                counts["exact_unavailable_in_m700"] += 1
            elif article_key(gold_id) not in selected_set:
                counts["article_not_selected"] += 1
            elif gold_id not in pool_ids:
                counts["candidate_miss"] += 1
            else:
                counts["selection_miss"] += 1
    return counts


def _write_report(state: dict[str, Any], path: Path) -> None:
    baseline = state["baseline_pack20"]
    neighbor = state["best_neighbor"]
    context_plain = state["context_plain_pack20"]
    context = state["context_pack20"]
    hierarchy = state["best_hierarchy"]
    dynamic = state["best_dynamic"]
    lines = [
        "# m700 evidence refinement",
        "",
        *( [
            "> **Статус:** расширенные гипотезы ниже считались в BF16. Победившая рабочая "
            "сетка пересчитана в FP32 без изменения exact recall; см. `m1000_evidence_comparison.md`.",
            "",
        ] if state["precision"] == "bf16" else []),
        f"Dev/blind only. Cross-encoder precision: {state['precision'].upper()}. "
        f"Article selector: `{json.dumps(state['article_selector_spec'], ensure_ascii=False, sort_keys=True)}`. "
        "Eval is not used.",
        "",
        "## Results",
        "",
        "| method | chunks | estimated tokens | exact recall | all exact | article recall | pool exact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in (baseline, neighbor, context_plain, context, hierarchy, dynamic):
        lines.append(
            f"| {row['name']} | {row['mean_chunks']:.1f} | "
            f"{row['mean_estimated_tokens']:.0f} | {row['metrics']['recall_exact']:.3f} | "
            f"{row['metrics']['all_exact']:.3f} | {row['metrics']['recall_article']:.3f} | "
            f"{row['pool_metrics']['recall_exact']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Neighbour expansion vs pack-20: exact Δ "
            f"{state['neighbor_vs_baseline']['mean_delta']:+.3f} (95% CI "
            f"[{state['neighbor_vs_baseline']['low']:+.3f}; "
            f"{state['neighbor_vs_baseline']['high']:+.3f}]).",
            f"Structured CE input vs the identical four-candidate plain-text pool: exact Δ "
            f"{state['context_vs_plain']['mean_delta']:+.3f} (95% CI "
            f"[{state['context_vs_plain']['low']:+.3f}; "
            f"{state['context_vs_plain']['high']:+.3f}]).",
            f"Best dynamic pack vs pack-20: exact Δ "
            f"{state['dynamic_vs_baseline']['mean_delta']:+.3f} (95% CI "
            f"[{state['dynamic_vs_baseline']['low']:+.3f}; "
            f"{state['dynamic_vs_baseline']['high']:+.3f}]).",
            "",
            f"The selected dynamic rule keeps the same per-case exact recall while reducing "
            f"mean output from {baseline['mean_chunks']:.1f} to {dynamic['mean_chunks']:.1f} chunks "
            f"and estimated context from {baseline['mean_estimated_tokens']:.0f} to "
            f"{dynamic['mean_estimated_tokens']:.0f} tokens. It uses the original eight-candidate "
            "pool, so it does not increase cross-encoder work.",
            "Neighbour-only evidence and structured prefixes are rejected: neither improves exact "
            "recall. Hierarchy bonus changes which chunks are returned but does not improve recall.",
            "",
            "Estimated tokens use ceil(characters / 3), because the generator tokenizer is not fixed yet.",
            "",
            "## Oracle decomposition of missed gold norms",
            "",
            f"`{json.dumps(state['oracle'], ensure_ascii=False)}`",
            "",
            f"Full local grid: `data/analysis/m700_evidence_refinement_state.json` "
            f"({state['count']} configurations).",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _merge_aggregate(
    state: dict[str, Any], analysis_dir: Path, report: Path, aggregate_report: Path
) -> None:
    main_path = analysis_dir / "retrieval_error_state.json"
    if not main_path.is_file():
        return
    main = json.loads(main_path.read_text(encoding="utf-8"))
    main["m700_evidence_refinement"] = {
        key: state[key]
        for key in (
            "n",
            "count",
            "precision",
            "article_selector_spec",
            "baseline_pack20",
            "best_neighbor",
            "context_pack20",
            "context_plain_pack20",
            "best_hierarchy",
            "best_dynamic",
            "neighbor_vs_baseline",
            "context_vs_plain",
            "dynamic_vs_baseline",
            "oracle",
        )
    }
    main["m700_evidence_refinement"]["report"] = str(report)
    main_path.write_text(json.dumps(main, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    from run_retrieval_error_analysis import _report as write_aggregate_report

    write_aggregate_report(main, aggregate_report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edition-dir", type=Path, default=Path("data/law/2026-08-04"))
    parser.add_argument("--cases-dir", type=Path, default=Path("evals/cases"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("data/analysis"))
    parser.add_argument(
        "--report", type=Path, default=Path("evals/reports/m700_evidence_refinement.md")
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
        raise ValueError(f"expected 52 dev cases, got {len(cases)}")
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
        for dense, bm25, articles in zip(dense_full, bm25_m700_full, selected, strict=True)
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
    shortlists: list[list[str]] = []
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

    base_maps = [
        article_candidate_map([dense, bm25], articles, per_article=16)
        for dense, bm25, articles in zip(dense_full, bm25_m700_full, shortlists, strict=True)
    ]
    base_pools = [
        article_candidate_pool_from_map(mapping, articles, per_article=16)
        for mapping, articles in zip(base_maps, shortlists, strict=True)
    ]
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    base_scores, _ = _score_pools(
        cases,
        base_pools,
        split="dev",
        cache_path=reranker_cache_path(args.analysis_dir, args.precision),
        fingerprint=corpus_fingerprint(dense_chunks),
        batch_size=args.batch_size,
        checkpoint_pairs=args.checkpoint_pairs,
        precision=args.precision,
    )
    base_ranked = [_ranked_map(mapping, scores) for mapping, scores in zip(base_maps, base_scores, strict=True)]

    union_maps_by_radius: dict[int, list[dict[str, list[Chunk]]]] = {1: [], 2: []}
    neighbor_maps_by_radius: dict[int, list[dict[str, list[Chunk]]]] = {1: [], 2: []}
    for mapping, (ranked, _), articles in zip(base_maps, base_ranked, shortlists, strict=True):
        for radius in (1, 2):
            union: dict[str, list[Chunk]] = {}
            neighbours: dict[str, list[Chunk]] = {}
            for article in articles:
                primary = ranked[article][0]
                nearby = article_neighbors(dense_chunks, primary, radius=radius)
                nearby = [
                    chunk
                    for chunk in nearby
                    if article in {article_key(norm_id) for norm_id in chunk.norm_ids}
                ]
                neighbours[article] = [primary, *nearby]
                by_id = {chunk.chunk_id: chunk for chunk in [*mapping[article], *nearby]}
                union[article] = list(by_id.values())
            union_maps_by_radius[radius].append(union)
            neighbor_maps_by_radius[radius].append(neighbours)
    union_maps = union_maps_by_radius[2]
    union_pools = [
        article_candidate_pool_from_map(mapping, articles, per_article=1000)
        for mapping, articles in zip(union_maps, shortlists, strict=True)
    ]

    union_scores, _ = _score_pools(
        cases,
        union_pools,
        split="dev",
        cache_path=reranker_cache_path(args.analysis_dir, args.precision),
        fingerprint=corpus_fingerprint(dense_chunks),
        batch_size=args.batch_size,
        checkpoint_pairs=args.checkpoint_pairs,
        precision=args.precision,
    )
    union_ranked = [
        _ranked_map(mapping, scores)
        for mapping, scores in zip(union_maps, union_scores, strict=True)
    ]
    neighbor_ranked_by_radius = {
        radius: [
            _ranked_map(mapping, scores)
            for mapping, scores in zip(mappings, union_scores, strict=True)
        ]
        for radius, mappings in neighbor_maps_by_radius.items()
    }
    union_ranked_by_radius = {
        radius: [
            _ranked_map(mapping, scores)
            for mapping, scores in zip(mappings, union_scores, strict=True)
        ]
        for radius, mappings in union_maps_by_radius.items()
    }

    units = [json.loads(line) for line in (args.edition_dir / "norm_units.jsonl").read_text(encoding="utf-8").splitlines()]
    titles = {
        f"art.{unit['article']}": unit["title"] or ""
        for unit in units
        if unit["level"] == "article"
    }
    base4_maps = [
        {article: chunks[:4] for article, chunks in mapping.items()} for mapping in base_maps
    ]
    base4_pools = [
        article_candidate_pool_from_map(mapping, articles, per_article=4)
        for mapping, articles in zip(base4_maps, shortlists, strict=True)
    ]
    base4_ranked = [
        _ranked_map(mapping, scores)
        for mapping, scores in zip(base4_maps, base_scores, strict=True)
    ]
    contextual_by_id = {chunk.chunk_id: _context_chunk(chunk, titles) for chunk in dense_chunks}
    context_pools = [
        [contextual_by_id[chunk.chunk_id] for chunk in pool] for pool in base4_pools
    ]
    context_scores_raw, _ = _score_pools(
        cases,
        context_pools,
        split="dev",
        cache_path=args.analysis_dir / f"m700_structured_context_scores_{args.precision}.json",
        fingerprint=corpus_fingerprint(list(contextual_by_id.values())),
        batch_size=args.batch_size,
        checkpoint_pairs=args.checkpoint_pairs,
        precision=args.precision,
    )
    context_scores = [
        {chunk_id.removesuffix("::context"): value for chunk_id, value in scores.items()}
        for scores in context_scores_raw
    ]
    context_ranked = [
        _ranked_map(mapping, scores)
        for mapping, scores in zip(base4_maps, context_scores, strict=True)
    ]

    def packs(
        ranked_data: list[tuple[dict[str, list[Chunk]], dict[str, float]]],
        *,
        budget: int,
        cap: int,
    ) -> list[list[Chunk]]:
        return [
            build_evidence_pack(
                articles,
                ranked,
                scores,
                budget=budget,
                per_article_cap=cap,
                primary_articles=10,
            )
            for articles, (ranked, scores) in zip(shortlists, ranked_data, strict=True)
        ]

    base8_maps = [
        {article: chunks[:8] for article, chunks in mapping.items()} for mapping in base_maps
    ]
    base8_pools = [
        article_candidate_pool_from_map(mapping, articles, per_article=8)
        for mapping, articles in zip(base8_maps, shortlists, strict=True)
    ]
    base8_ranked = [
        _ranked_map(mapping, scores)
        for mapping, scores in zip(base8_maps, base_scores, strict=True)
    ]
    baseline_rankings = packs(base8_ranked, budget=20, cap=2)
    baseline = _result(
        "fixed pack-20", {"family": "baseline"}, cases, baseline_rankings, base8_pools
    )
    neighbour_rows = []
    for radius in (1, 2):
        for family, data in (
            ("neighbors_only", neighbor_ranked_by_radius[radius]),
            ("base_plus_neighbors", union_ranked_by_radius[radius]),
        ):
            rankings = packs(data, budget=20, cap=2)
            neighbour_rows.append(
                _result(
                    f"{family} radius {radius}",
                    {"family": family, "radius": radius},
                    cases,
                    rankings,
                    union_pools,
                )
            )
    best_neighbor = _select(neighbour_rows)

    context_rankings = packs(context_ranked, budget=20, cap=2)
    context_plain_rankings = packs(base4_ranked, budget=20, cap=2)
    context_plain = _result(
        "plain CE pack-20 (4 candidates/article)",
        {"family": "context_plain", "candidates_per_article": 4},
        cases,
        context_plain_rankings,
        base4_pools,
    )
    context = _result(
        "structured CE pack-20 (4 candidates/article)",
        {"family": "structured_context", "candidates_per_article": 4},
        cases,
        context_rankings,
        base4_pools,
    )

    token_costs = {chunk.chunk_id: _token_cost(chunk) for chunk in dense_chunks}
    hierarchy_rows: list[Result] = []
    dynamic_rows: list[Result] = []
    for candidate_family, ranked_data, candidate_pools in (
        ("base8", base8_ranked, base8_pools),
        ("union", union_ranked, union_pools),
    ):
        for bonus in HIERARCHY_BONUSES:
            rankings = [
                build_adaptive_evidence_pack(
                    articles,
                    ranked,
                    scores,
                    token_costs,
                    max_chunks=20,
                    token_budget=10**9,
                    per_article_cap=2,
                    hierarchy_bonus=bonus,
                )
                for articles, (ranked, scores) in zip(shortlists, ranked_data, strict=True)
            ]
            hierarchy_rows.append(
                _result(
                    f"hierarchy {candidate_family} bonus {bonus:g}",
                    {
                        "family": "hierarchy",
                        "candidate_family": candidate_family,
                        "bonus": bonus,
                    },
                    cases,
                    rankings,
                    candidate_pools,
                )
            )
            for token_budget in TOKEN_BUDGETS:
                for ratio in SCORE_RATIOS:
                    dynamic_rankings = [
                        build_adaptive_evidence_pack(
                            articles,
                            ranked,
                            scores,
                            token_costs,
                            max_chunks=20,
                            token_budget=token_budget,
                            per_article_cap=2,
                            min_relative_score=ratio,
                            hierarchy_bonus=bonus,
                        )
                        for articles, (ranked, scores) in zip(
                            shortlists, ranked_data, strict=True
                        )
                    ]
                    dynamic_rows.append(
                        _result(
                            f"dynamic {candidate_family} t{token_budget} r{ratio:g} h{bonus:g}",
                            {
                                "family": "dynamic",
                                "candidate_family": candidate_family,
                                "token_budget": token_budget,
                                "min_relative_score": ratio,
                                "hierarchy_bonus": bonus,
                            },
                            cases,
                            dynamic_rankings,
                            candidate_pools,
                        )
                    )
    best_hierarchy = _select(hierarchy_rows)
    best_dynamic = _select(dynamic_rows)
    oracle = _oracle_breakdown(
        cases, shortlists, dense_chunks, union_pools, best_dynamic.rankings
    )
    state = {
        "version": STATE_VERSION,
        "split": "dev",
        "query": "blind",
        "precision": args.precision,
        "article_selector_spec": locked,
        "n": len(cases),
        "count": 1 + len(neighbour_rows) + 2 + len(hierarchy_rows) + len(dynamic_rows),
        "baseline_pack20": _serialize(baseline),
        "best_neighbor": _serialize(best_neighbor),
        "context_plain_pack20": _serialize(context_plain),
        "context_pack20": _serialize(context),
        "best_hierarchy": _serialize(best_hierarchy),
        "best_dynamic": _serialize(best_dynamic),
        "neighbor_vs_baseline": _comparison(cases, baseline, best_neighbor),
        "context_vs_plain": _comparison(cases, context_plain, context),
        "dynamic_vs_baseline": _comparison(cases, baseline, best_dynamic),
        "oracle": oracle,
        "results": [
            _serialize(row)
            for row in [
                baseline,
                *neighbour_rows,
                context_plain,
                context,
                *hierarchy_rows,
                *dynamic_rows,
            ]
        ],
    }
    state_path = args.analysis_dir / "m700_evidence_refinement_state.json"
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(state, args.report)
    _merge_aggregate(state, args.analysis_dir, args.report, args.aggregate_report)
    print(f"wrote {args.report}", file=sys.stderr)
    print(f"wrote {state_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
