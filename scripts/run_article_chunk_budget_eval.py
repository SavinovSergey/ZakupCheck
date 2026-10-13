#!/usr/bin/env python3
"""Evaluate joint article/chunk budgets and a two-pass article reranker.

The experiment is intentionally dev-only.  It reuses the cross-encoder scores
created by ``run_wide_article_rerank_eval.py`` and never loads or runs the
cross-encoder model.
"""

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
from index.error_analysis import (  # noqa: E402
    aggregate_rankings,
    paired_group_bootstrap,
    per_case_article_recall,
)
from index.eval_tables import load_verified  # noqa: E402
from index.recall import AggregateRecall, aggregate, article_key, score_case  # noqa: E402
from index.rerank import (  # noqa: E402
    article_candidate_map,
    article_candidate_pool,
    article_candidate_pool_from_map,
    best_chunk_per_article,
    rank_articles_by_best_chunk,
    source_quota_articles,
)
from run_hybrid_retrieval_eval import _load_indexes, _metric_dict, _rank_cases  # noqa: E402
from run_wide_article_rerank_eval import (  # noqa: E402
    MODEL,
    reranker_cache_path,
    unique_article_prefix,
)
from schemas.corpus import Chunk  # noqa: E402
from schemas.eval_case import EvalCase  # noqa: E402

DENSE_MAX_CHUNK = 700
BM25_MAX_CHUNK = 1500
SOURCE_ARTICLES = 50
LIMIT = 10
QUOTAS = (0, 5, 10, 15, 20, 30, 40, 50)
CHUNKS_PER_ARTICLE = (1, 2, 4, 6, 8)
FIRST_PASS_CHUNKS = (1, 2)
SHORTLIST_ARTICLES = (10, 15, 20)
FINAL_CHUNKS = (4, 6, 8)
BUDGETS = (50, 100, 150, 200)
MIN_ADAPTIVE_ARTICLE_REDUCTION = 5.0
STATE_VERSION = 2
SEED = 20261008


@dataclass(frozen=True)
class ExperimentResult:
    family: str
    spec: dict[str, Any]
    at10: AggregateRecall
    diagnostics: dict[str, float]
    mean_source_articles: float
    mean_shortlist_articles: float
    mean_scored_chunks: float
    max_scored_chunks: int
    rankings: list[list[Chunk]]


def _article_metrics(cases: list[EvalCase], articles: list[list[str]]) -> AggregateRecall:
    scores = []
    for case, selected in zip(cases, articles, strict=True):
        retrieved = {
            norm_id
            for article in selected
            for norm_id in case.gold_norms
            if article_key(norm_id) == article
        }
        scores.append(score_case(case.gold_norms, retrieved))
    return aggregate(scores)


def _pool_metrics(cases: list[EvalCase], pools: list[list[Chunk]]) -> AggregateRecall:
    scores = []
    for case, pool in zip(cases, pools, strict=True):
        norm_ids = {norm_id for chunk in pool for norm_id in chunk.norm_ids}
        scores.append(score_case(case.gold_norms, norm_ids))
    return aggregate(scores)


def pipeline_diagnostics(
    cases: list[EvalCase],
    source_articles: list[list[str]],
    shortlist_articles: list[list[str]],
    final_pools: list[list[Chunk]],
    final_rankings: list[list[Chunk]],
) -> dict[str, float]:
    """Decompose candidate, shortlist, evidence-pool, and final recall."""
    source = _article_metrics(cases, source_articles)
    shortlist = _article_metrics(cases, shortlist_articles)
    pool = _pool_metrics(cases, final_pools)
    final = aggregate_rankings(cases, final_rankings, k=LIMIT)
    return {
        "candidate_article_recall": source.recall_article,
        "candidate_all_article": source.all_article,
        "shortlist_article_recall": shortlist.recall_article,
        "shortlist_all_article": shortlist.all_article,
        "pool_article_recall": pool.recall_article,
        "pool_exact_recall": pool.recall_exact,
        "pool_all_exact": pool.all_exact,
        "final_article_recall": final.recall_article,
        "final_all_article": final.all_article,
        "final_exact_recall": final.recall_exact,
        "candidate_to_shortlist_article_loss": source.recall_article - shortlist.recall_article,
        "shortlist_to_final_article_loss": shortlist.recall_article - final.recall_article,
        "pool_to_final_exact_loss": pool.recall_exact - final.recall_exact,
    }


def _result(
    *,
    family: str,
    spec: dict[str, Any],
    cases: list[EvalCase],
    source_articles: list[list[str]],
    shortlist_articles: list[list[str]],
    final_pools: list[list[Chunk]],
    scored_pools: list[list[Chunk]],
    rankings: list[list[Chunk]],
) -> ExperimentResult:
    diagnostics = pipeline_diagnostics(
        cases,
        source_articles,
        shortlist_articles,
        final_pools,
        rankings,
    )
    at10 = aggregate_rankings(cases, rankings, k=LIMIT)
    return ExperimentResult(
        family=family,
        spec=spec,
        at10=at10,
        diagnostics=diagnostics,
        mean_source_articles=sum(map(len, source_articles)) / len(cases),
        mean_shortlist_articles=sum(map(len, shortlist_articles)) / len(cases),
        mean_scored_chunks=sum(map(len, scored_pools)) / len(cases),
        max_scored_chunks=max(map(len, scored_pools), default=0),
        rankings=rankings,
    )


def _quality_key(row: ExperimentResult) -> tuple[Any, ...]:
    return (
        -row.at10.recall_article,
        -row.at10.recall_exact,
        -row.at10.all_article,
        row.mean_scored_chunks,
        row.max_scored_chunks,
        tuple(sorted(row.spec.items())),
    )


def select_winner(rows: list[ExperimentResult]) -> ExperimentResult:
    """Use the established <0.01 article tie band, then exact/all-hit/cost."""
    if not rows:
        raise ValueError("at least one result is required")
    best_article = max(row.at10.recall_article for row in rows)
    tied = [row for row in rows if best_article - row.at10.recall_article < 0.01]
    best_exact = max(row.at10.recall_exact for row in tied)
    tied = [row for row in tied if row.at10.recall_exact == best_exact]
    best_all = max(row.at10.all_article for row in tied)
    tied = [row for row in tied if row.at10.all_article == best_all]
    return min(tied, key=lambda row: _quality_key(row)[3:])


def pareto_frontier(rows: list[ExperimentResult]) -> list[ExperimentResult]:
    frontier = []
    for row in rows:
        dominated = any(
            other.at10.recall_article >= row.at10.recall_article
            and other.at10.recall_exact >= row.at10.recall_exact
            and other.mean_scored_chunks <= row.mean_scored_chunks
            and (
                other.at10.recall_article > row.at10.recall_article
                or other.at10.recall_exact > row.at10.recall_exact
                or other.mean_scored_chunks < row.mean_scored_chunks
            )
            for other in rows
        )
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda row: (row.mean_scored_chunks, _quality_key(row)))


def _serialize(row: ExperimentResult) -> dict[str, Any]:
    return {
        "family": row.family,
        "spec": row.spec,
        "at10": _metric_dict(row.at10),
        "diagnostics": row.diagnostics,
        "mean_source_articles": row.mean_source_articles,
        "mean_shortlist_articles": row.mean_shortlist_articles,
        "mean_scored_chunks": row.mean_scored_chunks,
        "max_scored_chunks": row.max_scored_chunks,
    }


def _scores_for_cases(
    cache_path: Path,
    cases: list[EvalCase],
    pools: list[list[Chunk]],
    fingerprint: str,
) -> list[dict[str, float]]:
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache.get("model") != MODEL or cache.get("corpus_fingerprint") != fingerprint:
        raise ValueError("incompatible wide cross-encoder cache")
    split = cache.get("splits", {}).get("dev", {})
    output = []
    for case, pool in zip(cases, pools, strict=True):
        case_scores = split.get(case.case_id, {})
        missing = [chunk.chunk_id for chunk in pool if chunk.chunk_id not in case_scores]
        if missing:
            raise ValueError(
                f"cross-encoder cache misses {len(missing)} chunks for {case.case_id}; "
                "run run_wide_article_rerank_eval.py first"
            )
        output.append({chunk.chunk_id: float(case_scores[chunk.chunk_id]) for chunk in pool})
    return output


def _selected_articles(
    dense_articles: list[list[str]],
    bm25_articles: list[list[str]],
    dense_quota: int,
    bm25_quota: int,
) -> list[list[str]]:
    return [
        source_quota_articles(dense, bm25, dense_quota=dense_quota, bm25_quota=bm25_quota)
        for dense, bm25 in zip(dense_articles, bm25_articles, strict=True)
    ]


def evaluate_surface(
    cases: list[EvalCase],
    candidate_maps: list[dict[str, list[Chunk]]],
    dense_articles: list[list[str]],
    bm25_articles: list[list[str]],
    scores: list[dict[str, float]],
) -> list[ExperimentResult]:
    rows = []
    for dense_quota in QUOTAS:
        for bm25_quota in QUOTAS:
            if dense_quota + bm25_quota < LIMIT:
                continue
            selected = _selected_articles(
                dense_articles, bm25_articles, dense_quota, bm25_quota
            )
            for chunks_per_article in CHUNKS_PER_ARTICLE:
                pools = [
                    article_candidate_pool_from_map(
                        candidates, articles, per_article=chunks_per_article
                    )
                    for candidates, articles in zip(candidate_maps, selected, strict=True)
                ]
                rankings = [
                    best_chunk_per_article(pool, score, articles, limit=LIMIT)
                    for pool, score, articles in zip(pools, scores, selected, strict=True)
                ]
                rows.append(
                    _result(
                        family="surface",
                        spec={
                            "dense_articles": dense_quota,
                            "bm25_articles": bm25_quota,
                            "chunks_per_article": chunks_per_article,
                        },
                        cases=cases,
                        source_articles=selected,
                        shortlist_articles=selected,
                        final_pools=pools,
                        scored_pools=pools,
                        rankings=rankings,
                    )
                )
    return rows


def evaluate_two_pass(
    cases: list[EvalCase],
    candidate_maps: list[dict[str, list[Chunk]]],
    dense_articles: list[list[str]],
    bm25_articles: list[list[str]],
    scores: list[dict[str, float]],
) -> list[ExperimentResult]:
    rows = []
    for dense_quota in QUOTAS:
        for bm25_quota in QUOTAS:
            if dense_quota + bm25_quota < LIMIT:
                continue
            selected = _selected_articles(
                dense_articles, bm25_articles, dense_quota, bm25_quota
            )
            for first_pass in FIRST_PASS_CHUNKS:
                for shortlist_size in SHORTLIST_ARTICLES:
                    for final_chunks in FINAL_CHUNKS:
                        shortlists: list[list[str]] = []
                        final_pools: list[list[Chunk]] = []
                        rankings: list[list[Chunk]] = []
                        scored_pools: list[list[Chunk]] = []
                        for candidates, articles, score in zip(
                            candidate_maps, selected, scores, strict=True
                        ):
                            first_pool = article_candidate_pool_from_map(
                                candidates, articles, per_article=first_pass
                            )
                            article_ranking, _ = rank_articles_by_best_chunk(
                                first_pool, score, articles
                            )
                            shortlist = article_ranking[:shortlist_size]
                            final_pool = article_candidate_pool_from_map(
                                candidates, shortlist, per_article=final_chunks
                            )
                            ranking = best_chunk_per_article(
                                final_pool, score, shortlist, limit=LIMIT
                            )
                            scored_by_id = {chunk.chunk_id: chunk for chunk in first_pool}
                            scored_by_id.update(
                                {chunk.chunk_id: chunk for chunk in final_pool}
                            )
                            scored = [scored_by_id[key] for key in sorted(scored_by_id)]
                            shortlists.append(shortlist)
                            final_pools.append(final_pool)
                            rankings.append(ranking)
                            scored_pools.append(scored)
                        rows.append(
                            _result(
                                family="two_pass",
                                spec={
                                    "dense_articles": dense_quota,
                                    "bm25_articles": bm25_quota,
                                    "first_pass_chunks": first_pass,
                                    "shortlist_articles": shortlist_size,
                                    "final_chunks_per_article": final_chunks,
                                },
                                cases=cases,
                                source_articles=selected,
                                shortlist_articles=shortlists,
                                final_pools=final_pools,
                                scored_pools=scored_pools,
                                rankings=rankings,
                            )
                        )
    return rows


def _budget_winners(rows: list[ExperimentResult]) -> dict[str, dict[str, Any] | None]:
    output: dict[str, dict[str, Any] | None] = {}
    for budget in BUDGETS:
        eligible = [row for row in rows if row.max_scored_chunks <= budget]
        output[str(budget)] = _serialize(select_winner(eligible)) if eligible else None
    return output


def _spec_label(row: dict[str, Any]) -> str:
    spec = row["spec"]
    if row["family"] == "surface":
        return (
            f"D{spec['dense_articles']}+B{spec['bm25_articles']}, "
            f"{spec['chunks_per_article']} chunks/article"
        )
    return (
        f"D{spec['dense_articles']}+B{spec['bm25_articles']}, "
        f"first={spec['first_pass_chunks']}, shortlist={spec['shortlist_articles']}, "
        f"final={spec['final_chunks_per_article']}"
    )


def _report(state: dict[str, Any], path: Path) -> None:
    lines = [
        "# Article/chunk budget and two-pass reranking",
        "",
        "> **Статус:** историческая budget-сетка на BF16-кэше. Победившая архитектура "
        "`D10+B10 → first=2 → shortlist=10` повторно проверена в FP32 без изменения recall.",
        "",
        f"Dense: `Roflmax/bge-m3-russian-legal`, m{DENSE_MAX_CHUNK}; "
        f"BM25 article source: m{BM25_MAX_CHUNK}; evidence chunks: m700; "
        f"cross-encoder: `{MODEL}`.",
        "Only dev/blind is used. Cross-encoder logits are reused from the existing cache; "
        "no model inference was performed.",
        f"A two-pass result is considered genuinely adaptive only when the shortlist removes "
        f"at least {MIN_ADAPTIVE_ARTICLE_REDUCTION:.0f} articles per query on average.",
        "For every configuration the diagnostic path is identical: candidate article recall "
        "after dense/BM25 quotas; article recall after the optional first-pass shortlist; exact "
        "recall reachable anywhere in the final chunk pool; and article/exact recall in the "
        "returned top-10. Reported losses are differences between adjacent stages.",
        "",
        "## Winners",
        "",
    ]
    for title, key in (("Joint surface", "surface_winner"), ("Two-pass", "two_pass_winner")):
        row = state[key]
        d = row["diagnostics"]
        lines.extend(
            [
                f"### {title}",
                "",
                f"**{_spec_label(row)}**; mean/max scored chunks "
                f"{row['mean_scored_chunks']:.1f}/{row['max_scored_chunks']}.",
                f"Candidate article recall {d['candidate_article_recall']:.3f} → "
                f"shortlist {d['shortlist_article_recall']:.3f} → "
                f"final article R@10 {d['final_article_recall']:.3f}.",
                f"Exact pool ceiling {d['pool_exact_recall']:.3f} → "
                f"final exact R@10 {d['final_exact_recall']:.3f}; "
                f"all-hit article@10 {d['final_all_article']:.3f}.",
                "",
            ]
        )
    comparison = state["two_pass_vs_surface"]
    lines.extend(
        [
            f"Two-pass minus joint-surface article R@10: "
            f"{comparison['interval']['mean_delta']:+.3f} (95% group-bootstrap "
            f"[{comparison['interval']['low']:+.3f}; "
            f"{comparison['interval']['high']:+.3f}]); gains "
            f"{comparison['outcomes']['gains']}, losses "
            f"{comparison['outcomes']['losses']}, ties "
            f"{comparison['outcomes']['ties']}.",
            "",
        ]
    )
    lines.extend(
        [
            "## Best configuration under a hard per-query pair budget",
            "",
            "| family | max pairs | configuration | mean pairs | exact R@10 | article R@10 | candidate R | shortlist R | pool exact ceiling |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for family, winners in state["budget_winners"].items():
        for budget, row in winners.items():
            if row is None:
                continue
            d = row["diagnostics"]
            lines.append(
                f"| {family} | {budget} | {_spec_label(row)} | "
                f"{row['mean_scored_chunks']:.1f} | {row['at10']['recall_exact']:.3f} | "
                f"{row['at10']['recall_article']:.3f} | "
                f"{d['candidate_article_recall']:.3f} | "
                f"{d['shortlist_article_recall']:.3f} | {d['pool_exact_recall']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Pareto frontier",
            "",
            "The frontier is non-dominated jointly on article R@10, exact R@10, and mean "
            "number of scored chunks.",
            "",
            "| family | configuration | mean/max pairs | exact R@10 | article R@10 | candidate→shortlist→final article | pool→final exact |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in state["pareto"]:
        d = row["diagnostics"]
        lines.append(
            f"| {row['family']} | {_spec_label(row)} | "
            f"{row['mean_scored_chunks']:.1f}/{row['max_scored_chunks']} | "
            f"{row['at10']['recall_exact']:.3f} | {row['at10']['recall_article']:.3f} | "
            f"{d['candidate_article_recall']:.3f}→{d['shortlist_article_recall']:.3f}"
            f"→{d['final_article_recall']:.3f} | "
            f"{d['pool_exact_recall']:.3f}→{d['final_exact_recall']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Evaluated {state['counts']['surface']} surface and "
            f"{state['counts']['two_pass']} two-pass configurations "
            f"({state['counts']['genuinely_adaptive']} genuinely adaptive) over "
            f"{state['n']} dev cases. Full machine-readable results are stored locally in "
            "`data/analysis/article_chunk_budget_state.json`.",
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
    main["article_chunk_budget"] = {
        "version": state["version"],
        "query": state["query"],
        "split": state["split"],
        "n": state["n"],
        "counts": state["counts"],
        "surface_winner": state["surface_winner"],
        "two_pass_winner": state["two_pass_winner"],
        "two_pass_vs_surface": state["two_pass_vs_surface"],
        "budget_winners": state["budget_winners"],
        "report": str(report),
    }
    main_path.write_text(json.dumps(main, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    from run_retrieval_error_analysis import _report as write_aggregate_report

    write_aggregate_report(main, aggregate_report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edition-dir", type=Path, default=Path("data/law/2026-08-04"))
    parser.add_argument("--cases-dir", type=Path, default=Path("evals/cases"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("data/analysis"))
    parser.add_argument(
        "--report", type=Path, default=Path("evals/reports/article_chunk_budget.md")
    )
    parser.add_argument(
        "--aggregate-report",
        type=Path,
        default=Path("evals/reports/retrieval_error_analysis.md"),
    )
    args = parser.parse_args(argv)

    cases = [case for case in load_verified(args.cases_dir) if case.split == "dev"]
    if len(cases) != 52:
        print(f"expected 52 dev cases, got {len(cases)}", file=sys.stderr)
        return 2

    dense_chunks, dense_index, bm25_indexes = _load_indexes(args.edition_dir)
    dense_full, _ = _rank_cases(dense_index, cases, k=len(dense_chunks))
    bm25_full, _ = _rank_cases(
        bm25_indexes[BM25_MAX_CHUNK],
        cases,
        k=len(bm25_indexes[BM25_MAX_CHUNK].chunks),
    )
    bm25_m700_full, _ = _rank_cases(Bm25Index(dense_chunks), cases, k=len(dense_chunks))
    dense_articles = [unique_article_prefix(row, SOURCE_ARTICLES) for row in dense_full]
    bm25_articles = [unique_article_prefix(row, SOURCE_ARTICLES) for row in bm25_full]
    full_articles = _selected_articles(dense_articles, bm25_articles, 50, 50)
    full_pools = [
        article_candidate_pool([dense, bm25], articles, per_article=8)
        for dense, bm25, articles in zip(
            dense_full, bm25_m700_full, full_articles, strict=True
        )
    ]
    scores = _scores_for_cases(
        reranker_cache_path(args.analysis_dir, "fp32"),
        cases,
        full_pools,
        corpus_fingerprint(dense_chunks),
    )
    candidate_maps = [
        article_candidate_map([dense, bm25], articles, per_article=8)
        for dense, bm25, articles in zip(
            dense_full, bm25_m700_full, full_articles, strict=True
        )
    ]

    surface = evaluate_surface(
        cases, candidate_maps, dense_articles, bm25_articles, scores
    )
    two_pass = evaluate_two_pass(
        cases, candidate_maps, dense_articles, bm25_articles, scores
    )
    adaptive = [
        row
        for row in two_pass
        if row.mean_source_articles - row.mean_shortlist_articles
        >= MIN_ADAPTIVE_ARTICLE_REDUCTION
    ]
    all_rows = surface + adaptive
    surface_winner = select_winner(surface)
    adaptive_winner = select_winner(adaptive)
    before = per_case_article_recall(cases, surface_winner.rankings)
    after = per_case_article_recall(cases, adaptive_winner.rankings)
    interval = paired_group_bootstrap(
        before,
        after,
        [case.case_group_id for case in cases],
        seed=SEED,
    )
    state = {
        "version": STATE_VERSION,
        "query": "blind",
        "split": "dev",
        "n": len(cases),
        "model": MODEL,
        "dense_model": "Roflmax/bge-m3-russian-legal",
        "dense_max_chunk": DENSE_MAX_CHUNK,
        "bm25_max_chunk": BM25_MAX_CHUNK,
        "counts": {
            "surface": len(surface),
            "two_pass": len(two_pass),
            "genuinely_adaptive": len(adaptive),
        },
        "surface_winner": _serialize(surface_winner),
        "two_pass_winner": _serialize(adaptive_winner),
        "two_pass_vs_surface": {
            "interval": asdict(interval),
            "outcomes": {
                "gains": sum(right > left for left, right in zip(before, after, strict=True)),
                "losses": sum(right < left for left, right in zip(before, after, strict=True)),
                "ties": sum(right == left for left, right in zip(before, after, strict=True)),
            },
        },
        "budget_winners": {
            "surface": _budget_winners(surface),
            "two_pass": _budget_winners(adaptive),
        },
        "pareto": [_serialize(row) for row in pareto_frontier(all_rows)],
        "surface": [_serialize(row) for row in surface],
        "two_pass": [_serialize(row) for row in two_pass],
    }
    state_path = args.analysis_dir / "article_chunk_budget_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _report(state, args.report)
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
