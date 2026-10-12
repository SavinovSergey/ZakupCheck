#!/usr/bin/env python3
"""Compare retrieval-aware article shortlist strategies on cached CE scores.

This is a dev/blind-only experiment.  It reuses the full 50+50, eight-chunk
cross-encoder cache and never performs cross-encoder inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
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
from index.fusion import (  # noqa: E402
    article_diversify,
    fuse_article_rankings,
    protected_article_shortlist,
    rank_articles_with_retrieval_bonus,
)
from index.rerank import (  # noqa: E402
    article_candidate_map,
    article_candidate_pool,
    article_candidate_pool_from_map,
    best_chunk_per_article,
    best_chunks_in_article_order,
    rank_articles_by_best_chunk,
)
from run_article_chunk_budget_eval import (  # noqa: E402
    ExperimentResult,
    _result,
    _scores_for_cases,
    _selected_articles,
    _serialize,
    select_winner,
)
from run_hybrid_retrieval_eval import _load_indexes, _metric_dict, _rank_cases  # noqa: E402
from run_wide_article_rerank_eval import (  # noqa: E402
    MODEL,
    _score_pools,
    unique_article_prefix,
)
from schemas.corpus import Chunk  # noqa: E402
from schemas.eval_case import EvalCase  # noqa: E402

DENSE_MAX_CHUNK = 700
BM25_MAX_CHUNK = 1500
SOURCE_ARTICLES = 50
LIMIT = 10
FINAL_CHUNKS = 8
SEED = 20261008
STATE_VERSION = 1

CANDIDATE_QUOTAS = ((5, 10), (10, 10), (10, 20), (20, 30), (0, 50), (50, 0), (50, 50))
FIRST_PASS_CHUNKS = (1, 2)
SHORTLIST_ARTICLES = (10, 15, 20)
RRF_KS = (10, 30, 60)
THREE_WAY_WEIGHTS = ((1, 1, 1), (1, 1, 2), (2, 1, 1), (1, 2, 1))
RETRIEVAL_WEIGHTS = ((1, 1), (2, 1), (1, 2))
PROTECTED_QUOTAS = ((2, 2), (3, 3), (5, 2), (2, 5))
BONUS_ALPHAS = (0.1, 0.25, 0.5, 1.0, 2.0)
BUDGETS = (50, 100, 150, 200)


def _strategy_specs(shortlist_size: int) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [{"strategy": "ce_only"}]
    specs.extend(
        {
            "strategy": "three_way_rrf",
            "k": k,
            "dense_weight": weights[0],
            "bm25_weight": weights[1],
            "ce_weight": weights[2],
        }
        for k in RRF_KS
        for weights in THREE_WAY_WEIGHTS
    )
    specs.extend(
        {
            "strategy": "protected_quotas",
            "dense_keep": dense_keep,
            "bm25_keep": bm25_keep,
        }
        for dense_keep, bm25_keep in PROTECTED_QUOTAS
        if dense_keep + bm25_keep <= shortlist_size
    )
    specs.extend(
        {"strategy": "ce_retrieval_bonus", "alpha": alpha, "k": 10}
        for alpha in BONUS_ALPHAS
    )
    specs.extend(
        {
            "strategy": "retrieval_rrf",
            "k": k,
            "dense_weight": weights[0],
            "bm25_weight": weights[1],
        }
        for k in RRF_KS
        for weights in RETRIEVAL_WEIGHTS
    )
    return specs


def _ce_article_scores(
    pool: list[Chunk],
    scores: dict[str, float],
    articles: list[str],
) -> tuple[list[str], dict[str, float]]:
    ranking, evidence = rank_articles_by_best_chunk(pool, scores, articles)
    return ranking, {
        article: scores[evidence[article].chunk_id]
        for article in ranking
    }


def _shortlist(
    spec: dict[str, Any],
    *,
    candidates: list[str],
    dense_articles: list[str],
    bm25_articles: list[str],
    ce_ranking: list[str],
    ce_scores: dict[str, float],
    limit: int,
) -> list[str]:
    strategy = spec["strategy"]
    if strategy == "ce_only":
        return ce_ranking[:limit]
    if strategy == "three_way_rrf":
        return fuse_article_rankings(
            [dense_articles, bm25_articles, ce_ranking],
            candidates=candidates,
            weights=[spec["dense_weight"], spec["bm25_weight"], spec["ce_weight"]],
            k=spec["k"],
            limit=limit,
        )
    if strategy == "protected_quotas":
        return protected_article_shortlist(
            dense_articles,
            bm25_articles,
            ce_ranking,
            candidates=candidates,
            dense_keep=spec["dense_keep"],
            bm25_keep=spec["bm25_keep"],
            limit=limit,
        )
    if strategy == "ce_retrieval_bonus":
        return rank_articles_with_retrieval_bonus(
            ce_scores,
            dense_articles,
            bm25_articles,
            candidates=candidates,
            alpha=spec["alpha"],
            k=spec["k"],
            limit=limit,
        )
    if strategy == "retrieval_rrf":
        return fuse_article_rankings(
            [dense_articles, bm25_articles],
            candidates=candidates,
            weights=[spec["dense_weight"], spec["bm25_weight"]],
            k=spec["k"],
            limit=limit,
        )
    raise ValueError(f"unknown strategy {strategy}")


def evaluate(
    cases: list[EvalCase],
    candidate_maps: list[dict[str, list[Chunk]]],
    dense_articles: list[list[str]],
    bm25_articles: list[list[str]],
    scores: list[dict[str, float]],
) -> list[ExperimentResult]:
    rows: list[ExperimentResult] = []
    for dense_quota, bm25_quota in CANDIDATE_QUOTAS:
        selected = _selected_articles(
            dense_articles, bm25_articles, dense_quota, bm25_quota
        )
        for first_pass in FIRST_PASS_CHUNKS:
            first_pools = [
                article_candidate_pool_from_map(candidates, articles, per_article=first_pass)
                for candidates, articles in zip(candidate_maps, selected, strict=True)
            ]
            ce_data = [
                _ce_article_scores(pool, score, articles)
                for pool, score, articles in zip(first_pools, scores, selected, strict=True)
            ]
            for shortlist_size in SHORTLIST_ARTICLES:
                for strategy in _strategy_specs(shortlist_size):
                    if strategy["strategy"] == "retrieval_rrf" and first_pass != 1:
                        continue
                    shortlists: list[list[str]] = []
                    final_pools: list[list[Chunk]] = []
                    rankings: list[list[Chunk]] = []
                    scored_pools: list[list[Chunk]] = []
                    for candidates, articles, dense, bm25, first_pool, ce, score in zip(
                        candidate_maps,
                        selected,
                        dense_articles,
                        bm25_articles,
                        first_pools,
                        ce_data,
                        scores,
                        strict=True,
                    ):
                        ce_ranking, ce_scores = ce
                        shortlist = _shortlist(
                            strategy,
                            candidates=articles,
                            dense_articles=dense,
                            bm25_articles=bm25,
                            ce_ranking=ce_ranking,
                            ce_scores=ce_scores,
                            limit=shortlist_size,
                        )
                        final_pool = article_candidate_pool_from_map(
                            candidates, shortlist, per_article=FINAL_CHUNKS
                        )
                        ranking = best_chunks_in_article_order(
                            final_pool, score, shortlist, limit=LIMIT
                        )
                        if strategy["strategy"] == "retrieval_rrf":
                            scored = final_pool
                        else:
                            by_id = {chunk.chunk_id: chunk for chunk in first_pool}
                            by_id.update({chunk.chunk_id: chunk for chunk in final_pool})
                            scored = [by_id[chunk_id] for chunk_id in sorted(by_id)]
                        shortlists.append(shortlist)
                        final_pools.append(final_pool)
                        rankings.append(ranking)
                        scored_pools.append(scored)
                    rows.append(
                        _result(
                            family=strategy["strategy"],
                            spec={
                                "dense_articles": dense_quota,
                                "bm25_articles": bm25_quota,
                                "first_pass_chunks": first_pass,
                                "shortlist_articles": shortlist_size,
                                "final_chunks_per_article": FINAL_CHUNKS,
                                **strategy,
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
    output = {}
    for budget in BUDGETS:
        eligible = [row for row in rows if row.max_scored_chunks <= budget]
        output[str(budget)] = _serialize(select_winner(eligible)) if eligible else None
    return output


def _comparison(
    cases: list[EvalCase],
    baseline: ExperimentResult,
    candidate: ExperimentResult,
) -> dict[str, Any]:
    before = per_case_article_recall(cases, baseline.rankings)
    after = per_case_article_recall(cases, candidate.rankings)
    interval = paired_group_bootstrap(
        before,
        after,
        [case.case_group_id for case in cases],
        seed=SEED,
    )
    return {
        "interval": asdict(interval),
        "outcomes": {
            "gains": sum(right > left for left, right in zip(before, after, strict=True)),
            "losses": sum(right < left for left, right in zip(before, after, strict=True)),
            "ties": sum(right == left for left, right in zip(before, after, strict=True)),
        },
    }


def _label(row: dict[str, Any]) -> str:
    spec = row["spec"]
    base = (
        f"D{spec['dense_articles']}+B{spec['bm25_articles']}; "
        f"first={spec['first_pass_chunks']}; shortlist={spec['shortlist_articles']}"
    )
    strategy = row["family"]
    if strategy == "three_way_rrf":
        return (
            f"{base}; k={spec['k']}; weights="
            f"{spec['dense_weight']}:{spec['bm25_weight']}:{spec['ce_weight']}"
        )
    if strategy == "protected_quotas":
        return f"{base}; protect={spec['dense_keep']}D+{spec['bm25_keep']}B"
    if strategy == "ce_retrieval_bonus":
        return f"{base}; alpha={spec['alpha']}"
    if strategy == "retrieval_rrf":
        return (
            f"D{spec['dense_articles']}+B{spec['bm25_articles']}; "
            f"shortlist={spec['shortlist_articles']}; k={spec['k']}; weights="
            f"{spec['dense_weight']}:{spec['bm25_weight']}"
        )
    return base


def _write_report(state: dict[str, Any], path: Path) -> None:
    lines = [
        "# Retrieval-aware article shortlist fusion",
        "",
        f"Dense: `Roflmax/bge-m3-russian-legal`, m{DENSE_MAX_CHUNK}; BM25 article source: "
        f"m{BM25_MAX_CHUNK}; cross-encoder: `{MODEL}`. Parameters were selected on dev/blind; "
        "eval is a single locked comparison when present.",
        "Every row is decomposed into source candidate recall, shortlist recall, exact pool ceiling, "
        "and final top-10 recall. CE cost counts unique query/chunk pairs across both passes.",
        "",
        "## Best result per shortlist strategy",
        "",
        "| strategy | configuration | mean/max CE pairs | candidate→shortlist→final article | pool→final exact | all article@10 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for strategy, row in state["winners_by_strategy"].items():
        d = row["diagnostics"]
        lines.append(
            f"| {strategy} | {_label(row)} | {row['mean_scored_chunks']:.1f}/"
            f"{row['max_scored_chunks']} | {d['candidate_article_recall']:.3f}→"
            f"{d['shortlist_article_recall']:.3f}→{d['final_article_recall']:.3f} | "
            f"{d['pool_exact_recall']:.3f}→{d['final_exact_recall']:.3f} | "
            f"{d['final_all_article']:.3f} |"
        )
    comparison = state["winner_vs_ce_only"]
    surface_comparison = state["winner_vs_joint_surface"]
    lines.extend(
        [
            "",
            "## Overall winner",
            "",
            f"**{state['winner']['family']}: {_label(state['winner'])}**.",
            f"Article R@10 {state['winner']['at10']['recall_article']:.3f}, exact R@10 "
            f"{state['winner']['at10']['recall_exact']:.3f}, all-hit article@10 "
            f"{state['winner']['at10']['all_article']:.3f}.",
            f"Against the best CE-only shortlist: Δ article R@10 "
            f"{comparison['interval']['mean_delta']:+.3f} (95% group-bootstrap "
            f"[{comparison['interval']['low']:+.3f}; {comparison['interval']['high']:+.3f}]); "
            f"gains {comparison['outcomes']['gains']}, losses {comparison['outcomes']['losses']}, "
            f"ties {comparison['outcomes']['ties']}.",
            f"Against the previous D5+B10 × 8 joint-surface winner: Δ article R@10 "
            f"{surface_comparison['interval']['mean_delta']:+.3f} (95% group-bootstrap "
            f"[{surface_comparison['interval']['low']:+.3f}; "
            f"{surface_comparison['interval']['high']:+.3f}]). Exact R@10 changes from "
            f"{state['joint_surface_baseline']['at10']['recall_exact']:.3f} to "
            f"{state['winner']['at10']['recall_exact']:.3f}.",
            "",
            "## Best result under a hard per-query CE budget",
            "",
            "| max pairs | strategy | configuration | mean pairs | exact R@10 | article R@10 |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for budget, row in state["budget_winners"].items():
        if row is None:
            continue
        lines.append(
            f"| {budget} | {row['family']} | {_label(row)} | "
            f"{row['mean_scored_chunks']:.1f} | {row['at10']['recall_exact']:.3f} | "
            f"{row['at10']['recall_article']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Evaluated {state['count']} configurations over {state['n']} dev cases. "
            "Full machine-readable results are local in "
            "`data/analysis/article_shortlist_fusion_state.json`.",
        ]
    )
    if "eval" in state:
        evaluated = state["eval"]
        dense = evaluated["dense_baseline"]
        winner = evaluated["winner"]
        interval = evaluated["bootstrap_vs_dense"]
        lines.extend(
            [
                "",
                "## One-shot eval / blind",
                "",
                f"Dense m700 article-diverse: article R@10 "
                f"{dense['recall_article']:.3f}, exact R@10 {dense['recall_exact']:.3f}, "
                f"all-hit article@10 {dense['all_article']:.3f}.",
                f"Locked three-way RRF: article R@10 "
                f"{winner['at10']['recall_article']:.3f}, exact R@10 "
                f"{winner['at10']['recall_exact']:.3f}, all-hit article@10 "
                f"{winner['at10']['all_article']:.3f}.",
                f"Δ article R@10 {interval['mean_delta']:+.3f} (95% group-bootstrap "
                f"[{interval['low']:+.3f}; {interval['high']:+.3f}]); gains "
                f"{evaluated['outcomes']['gains']}, losses {evaluated['outcomes']['losses']}, "
                f"ties {evaluated['outcomes']['ties']}. Mean/max required CE pairs "
                f"{winner['mean_scored_chunks']:.1f}/{winner['max_scored_chunks']} per query.",
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
    main["article_shortlist_fusion"] = {
        key: state[key]
        for key in (
            "version",
            "n",
            "count",
            "winner",
            "winners_by_strategy",
            "winner_vs_ce_only",
            "joint_surface_baseline",
            "winner_vs_joint_surface",
            "budget_winners",
            *(("eval",) if "eval" in state else ()),
        )
    }
    main["article_shortlist_fusion"]["report"] = str(report)
    main_path.write_text(json.dumps(main, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    from run_retrieval_error_analysis import _report as write_aggregate_report

    write_aggregate_report(main, aggregate_report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dev", "eval"), default="dev")
    parser.add_argument("--edition-dir", type=Path, default=Path("data/law/2026-08-04"))
    parser.add_argument("--cases-dir", type=Path, default=Path("evals/cases"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("data/analysis"))
    parser.add_argument(
        "--report", type=Path, default=Path("evals/reports/article_shortlist_fusion.md")
    )
    parser.add_argument(
        "--aggregate-report",
        type=Path,
        default=Path("evals/reports/retrieval_error_analysis.md"),
    )
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--checkpoint-pairs", type=int, default=240)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    args = parser.parse_args(argv)

    cases = [case for case in load_verified(args.cases_dir) if case.split == args.phase]
    expected = 52 if args.phase == "dev" else 22
    if len(cases) != expected:
        print(f"expected {expected} {args.phase} cases, got {len(cases)}", file=sys.stderr)
        return 2
    state_path = args.analysis_dir / "article_shortlist_fusion_state.json"
    if args.phase == "eval" and not state_path.is_file():
        print("run the default dev phase first", file=sys.stderr)
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
        article_candidate_pool([dense, bm25], articles, per_article=FINAL_CHUNKS)
        for dense, bm25, articles in zip(
            dense_full, bm25_m700_full, full_articles, strict=True
        )
    ]
    candidate_maps = [
        article_candidate_map([dense, bm25], articles, per_article=FINAL_CHUNKS)
        for dense, bm25, articles in zip(
            dense_full, bm25_m700_full, full_articles, strict=True
        )
    ]

    if args.phase == "eval":
        state = json.loads(state_path.read_text(encoding="utf-8"))
        locked = state["winner"]["spec"]
        selected = _selected_articles(
            dense_articles,
            bm25_articles,
            int(locked["dense_articles"]),
            int(locked["bm25_articles"]),
        )
        first_pools = [
            article_candidate_pool_from_map(
                candidates,
                articles,
                per_article=int(locked["first_pass_chunks"]),
            )
            for candidates, articles in zip(candidate_maps, selected, strict=True)
        ]
        import torch

        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        first_scores, _ = _score_pools(
            cases,
            first_pools,
            split="eval",
            cache_path=args.analysis_dir / "wide_article_rerank_scores.json",
            fingerprint=corpus_fingerprint(dense_chunks),
            batch_size=args.batch_size,
            checkpoint_pairs=args.checkpoint_pairs,
            precision=args.precision,
        )
        ce_data = [
            _ce_article_scores(pool, score, articles)
            for pool, score, articles in zip(first_pools, first_scores, selected, strict=True)
        ]
        shortlists = [
            _shortlist(
                locked,
                candidates=articles,
                dense_articles=dense,
                bm25_articles=bm25,
                ce_ranking=ce[0],
                ce_scores=ce[1],
                limit=int(locked["shortlist_articles"]),
            )
            for articles, dense, bm25, ce in zip(
                selected, dense_articles, bm25_articles, ce_data, strict=True
            )
        ]
        final_pools = [
            article_candidate_pool_from_map(
                candidates,
                shortlist,
                per_article=int(locked["final_chunks_per_article"]),
            )
            for candidates, shortlist in zip(candidate_maps, shortlists, strict=True)
        ]
        final_scores, _ = _score_pools(
            cases,
            final_pools,
            split="eval",
            cache_path=args.analysis_dir / "wide_article_rerank_scores.json",
            fingerprint=corpus_fingerprint(dense_chunks),
            batch_size=args.batch_size,
            checkpoint_pairs=args.checkpoint_pairs,
            precision=args.precision,
        )
        rankings = [
            best_chunks_in_article_order(pool, score, shortlist, limit=LIMIT)
            for pool, score, shortlist in zip(
                final_pools, final_scores, shortlists, strict=True
            )
        ]
        scored_pools = []
        for first_pool, final_pool in zip(first_pools, final_pools, strict=True):
            by_id = {chunk.chunk_id: chunk for chunk in first_pool}
            by_id.update({chunk.chunk_id: chunk for chunk in final_pool})
            scored_pools.append([by_id[chunk_id] for chunk_id in sorted(by_id)])
        result = _result(
            family=state["winner"]["family"],
            spec=locked,
            cases=cases,
            source_articles=selected,
            shortlist_articles=shortlists,
            final_pools=final_pools,
            scored_pools=scored_pools,
            rankings=rankings,
        )
        dense_rankings = [
            article_diversify(row, pool=len(row), limit=LIMIT) for row in dense_full
        ]
        dense_metrics = aggregate_rankings(cases, dense_rankings, k=LIMIT)
        before = per_case_article_recall(cases, dense_rankings)
        after = per_case_article_recall(cases, rankings)
        interval = paired_group_bootstrap(
            before,
            after,
            [case.case_group_id for case in cases],
            seed=SEED,
        )
        state["eval"] = {
            "n": len(cases),
            "dense_baseline": _metric_dict(dense_metrics),
            "winner": _serialize(result),
            "bootstrap_vs_dense": asdict(interval),
            "outcomes": {
                "gains": sum(right > left for left, right in zip(before, after, strict=True)),
                "losses": sum(right < left for left, right in zip(before, after, strict=True)),
                "ties": sum(right == left for left, right in zip(before, after, strict=True)),
            },
        }
        state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
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

    scores = _scores_for_cases(
        args.analysis_dir / "wide_article_rerank_scores.json",
        cases,
        full_pools,
        corpus_fingerprint(dense_chunks),
    )
    rows = evaluate(cases, candidate_maps, dense_articles, bm25_articles, scores)
    by_strategy = {
        strategy: select_winner([row for row in rows if row.family == strategy])
        for strategy in sorted({row.family for row in rows})
    }
    winner = select_winner(rows)
    ce_only = by_strategy["ce_only"]
    surface_articles = _selected_articles(dense_articles, bm25_articles, 5, 10)
    surface_pools = [
        article_candidate_pool_from_map(candidates, articles, per_article=8)
        for candidates, articles in zip(candidate_maps, surface_articles, strict=True)
    ]
    surface_rankings = [
        best_chunk_per_article(pool, score, articles, limit=LIMIT)
        for pool, score, articles in zip(
            surface_pools, scores, surface_articles, strict=True
        )
    ]
    joint_surface = _result(
        family="joint_surface",
        spec={"dense_articles": 5, "bm25_articles": 10, "chunks_per_article": 8},
        cases=cases,
        source_articles=surface_articles,
        shortlist_articles=surface_articles,
        final_pools=surface_pools,
        scored_pools=surface_pools,
        rankings=surface_rankings,
    )
    state = {
        "version": STATE_VERSION,
        "query": "blind",
        "split": "dev",
        "n": len(cases),
        "count": len(rows),
        "model": MODEL,
        "dense_model": "Roflmax/bge-m3-russian-legal",
        "dense_max_chunk": DENSE_MAX_CHUNK,
        "bm25_max_chunk": BM25_MAX_CHUNK,
        "winner": _serialize(winner),
        "winners_by_strategy": {
            strategy: _serialize(row) for strategy, row in by_strategy.items()
        },
        "winner_vs_ce_only": {
            **_comparison(cases, ce_only, winner),
        },
        "joint_surface_baseline": _serialize(joint_surface),
        "winner_vs_joint_surface": _comparison(cases, joint_surface, winner),
        "budget_winners": _budget_winners(rows),
        "results": [_serialize(row) for row in rows],
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
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
