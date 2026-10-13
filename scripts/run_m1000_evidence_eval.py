#!/usr/bin/env python3
"""Compare m700 and m1000 evidence selection under a frozen article shortlist."""

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

from index.bm25 import Bm25Index, load_chunks  # noqa: E402
from index.dense import DenseIndex, corpus_fingerprint  # noqa: E402
from index.error_analysis import paired_group_bootstrap  # noqa: E402
from index.eval_tables import chunks_path, load_verified  # noqa: E402
from index.recall import article_key  # noqa: E402
from index.rerank import (  # noqa: E402
    article_candidate_map,
    article_candidate_pool_from_map,
    build_adaptive_evidence_pack,
    build_evidence_pack,
    rank_articles_by_best_chunk,
)
from run_article_chunk_budget_eval import _scores_for_cases, _selected_articles  # noqa: E402
from run_article_shortlist_fusion_eval import _shortlist  # noqa: E402
from run_hybrid_retrieval_eval import MODEL, _load_indexes, _rank_cases  # noqa: E402
from run_m700_evidence_refinement_eval import (  # noqa: E402
    Result,
    _exact_values,
    _oracle_breakdown,
    _result,
    _select,
    _serialize,
    _token_cost,
)
from run_wide_article_rerank_eval import (  # noqa: E402
    MODEL as RERANKER_MODEL,
    _score_pools,
    reranker_cache_path,
    unique_article_prefix,
)
from schemas.corpus import Chunk  # noqa: E402

M700 = 700
M1000 = 1000
BM25_ARTICLE_CHUNK = 1500
SOURCE_ARTICLES = 50
SEED = 20261008
STATE_VERSION = 1
CANDIDATE_SIZES = (4, 8)
TOKEN_BUDGETS = (2500, 3000, 3500, 4000)
SCORE_RATIOS = (0.0, 0.25, 0.5)
HIERARCHY_BONUSES = (0.0, 0.05, 0.1)
LOCKED_DYNAMIC = {
    "token_budget": 3000,
    "min_relative_score": 0.25,
    "hierarchy_bonus": 0.05,
}


def _seed_m700_fp32_cache(
    source_path: Path,
    target_path: Path,
    fingerprint: str,
) -> int:
    """Seed truncated-model FP32 cache from the validated equivalent BAAI FP32 run."""
    if target_path.is_file() or not source_path.is_file():
        return 0
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("corpus_fingerprint") != fingerprint:
        raise ValueError("incompatible original FP32 score cache")
    splits = source.get("splits", {})
    seeded = sum(len(scores) for split in splits.values() for scores in split.values())
    target = {
        "version": 1,
        "model": RERANKER_MODEL,
        "corpus_fingerprint": fingerprint,
        "max_length": 512,
        "precision": "fp32",
        "splits": splits,
        "latency_seconds": {},
        "seed_source": source.get("model"),
        "seed_pairs": seeded,
    }
    target_path.write_text(json.dumps(target, ensure_ascii=False) + "\n", encoding="utf-8")
    return seeded


def _ranked_maps(
    maps: list[dict[str, list[Chunk]]], scores: list[dict[str, float]]
) -> list[tuple[dict[str, list[Chunk]], dict[str, float]]]:
    output = []
    for mapping, case_scores in zip(maps, scores, strict=True):
        ranked = {
            article: sorted(
                chunks, key=lambda chunk: (-case_scores[chunk.chunk_id], chunk.chunk_id)
            )
            for article, chunks in mapping.items()
        }
        output.append(
            (
                ranked,
                {
                    chunk.chunk_id: case_scores[chunk.chunk_id]
                    for chunks in mapping.values()
                    for chunk in chunks
                },
            )
        )
    return output


def _prefix_maps(
    maps: list[dict[str, list[Chunk]]], size: int
) -> list[dict[str, list[Chunk]]]:
    return [
        {article: chunks[:size] for article, chunks in mapping.items()} for mapping in maps
    ]


def _pools(
    maps: list[dict[str, list[Chunk]]], articles: list[list[str]], size: int
) -> list[list[Chunk]]:
    return [
        article_candidate_pool_from_map(mapping, selected, per_article=size)
        for mapping, selected in zip(maps, articles, strict=True)
    ]


def _fixed_pack(
    articles: list[list[str]],
    ranked: list[tuple[dict[str, list[Chunk]], dict[str, float]]],
    *,
    budget: int,
    cap: int,
) -> list[list[Chunk]]:
    return [
        build_evidence_pack(
            selected,
            by_article,
            scores,
            budget=budget,
            per_article_cap=cap,
            primary_articles=10,
        )
        for selected, (by_article, scores) in zip(articles, ranked, strict=True)
    ]


def _dynamic_pack(
    articles: list[list[str]],
    ranked: list[tuple[dict[str, list[Chunk]], dict[str, float]]],
    token_costs: dict[str, int],
    *,
    token_budget: int,
    min_relative_score: float,
    hierarchy_bonus: float,
) -> list[list[Chunk]]:
    return [
        build_adaptive_evidence_pack(
            selected,
            by_article,
            scores,
            token_costs,
            max_chunks=20,
            token_budget=token_budget,
            per_article_cap=2,
            min_relative_score=min_relative_score,
            hierarchy_bonus=hierarchy_bonus,
        )
        for selected, (by_article, scores) in zip(articles, ranked, strict=True)
    ]


def _comparison(cases: list[Any], left: Result, right: Result) -> dict[str, Any]:
    return asdict(
        paired_group_bootstrap(
            _exact_values(cases, left.rankings),
            _exact_values(cases, right.rankings),
            [case.case_group_id for case in cases],
            seed=SEED,
        )
    )


def _cache_pairs_per_second(path: Path, split: str) -> float:
    cache = json.loads(path.read_text(encoding="utf-8"))
    pairs = sum(len(scores) for scores in cache["splits"][split].values())
    elapsed = float(cache["latency_seconds"][split])
    return pairs / elapsed


def _write_report(state: dict[str, Any], path: Path) -> None:
    m700_fixed = state["m700_fixed20"]
    m700 = state["m700_locked_dynamic"]
    m700_best = state["m700_best20"]
    fixed = state["m1000_fixed20"]
    locked = state["m1000_locked_dynamic"]
    winner = state["m1000_best20"]
    pack30 = state["m1000_pack30"]
    fixed_interval = state["fixed20_m1000_vs_m700"]
    locked_interval = state["locked_m1000_vs_m700"]
    winner_interval = state["best_m1000_vs_best_m700"]
    lines = [
        "# m1000 evidence comparison",
        "",
        "Dev/blind only, cross-encoder FP32. The article shortlist is frozen to the previously locked "
        "m700/BM25 article selector; only evidence chunking and within-article candidates change.",
        "",
        "## Controlled comparison",
        "",
        "| method | candidates/article | output chunks | estimated tokens | exact recall | all exact | article recall | pool exact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in (m700_fixed, m700, m700_best, fixed, locked, winner, pack30):
        lines.append(
            f"| {row['name']} | {row['spec'].get('candidates_per_article', 8)} | "
            f"{row['mean_chunks']:.1f} | {row['mean_estimated_tokens']:.0f} | "
            f"{row['metrics']['recall_exact']:.3f} | {row['metrics']['all_exact']:.3f} | "
            f"{row['metrics']['recall_article']:.3f} | {row['pool_metrics']['recall_exact']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Fixed-20 m1000 versus fixed-20 m700: exact Δ "
            f"{fixed_interval['mean_delta']:+.3f} (95% group-bootstrap "
            f"[{fixed_interval['low']:+.3f}; {fixed_interval['high']:+.3f}]).",
            f"Locked dynamic m1000 versus locked dynamic m700: exact Δ "
            f"{locked_interval['mean_delta']:+.3f} (95% group-bootstrap "
            f"[{locked_interval['low']:+.3f}; {locked_interval['high']:+.3f}]).",
            f"Exploratory best m1000 versus exploratory best m700: exact Δ "
            f"{winner_interval['mean_delta']:+.3f} (95% group-bootstrap "
            f"[{winner_interval['low']:+.3f}; {winner_interval['high']:+.3f}]).",
            "",
            "The same dev sensitivity grid is rerun for m700 and m1000: 4/8 candidates per article, "
            "fixed pack-20, fixed pack-30, and dynamic score/token thresholds. Selection remains dev-only.",
            f"Both corpora were scored with `{state['reranker_model']}` in FP32. "
            f"No score was imported from a BF16 cache. Reused original-model FP32 scores: "
            f"{state['m700_seeded_original_fp32_pairs']}.",
            f"Measured CE inference: m700 {state['m700_cross_encoder_ms_per_query'] / 1000:.1f} s/query "
            f"({state['m700_pairs_per_second']:.2f} pairs/s), m1000 "
            f"{state['m1000_cross_encoder_ms_per_query'] / 1000:.1f} s/query "
            f"({state['m1000_pairs_per_second']:.2f} pairs/s). m1000 is "
            f"{state['m1000_latency_ratio']:.2f}× slower per query for these pools.",
            "",
            "## Oracle decomposition for locked m1000",
            "",
            f"`{json.dumps(state['oracle'], ensure_ascii=False)}`",
            "",
            f"Full local grid: `data/analysis/m1000_evidence_state.json` "
            f"({state['m700_count']} m700 and {state['count']} m1000 configurations).",
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
    main["m1000_evidence"] = {
        key: state[key]
        for key in (
            "n",
            "count",
            "m700_count",
            "precision",
            "m700_seeded_original_fp32_pairs",
            "m700_locked_dynamic",
            "m700_fixed20",
            "m700_best20",
            "m700_pack30",
            "m1000_fixed20",
            "m1000_locked_dynamic",
            "m1000_best20",
            "m1000_pack30",
            "locked_m1000_vs_m700",
            "fixed20_m1000_vs_m700",
            "best_m1000_vs_m700",
            "best_m1000_vs_best_m700",
            "m700_cross_encoder_ms_per_query",
            "m1000_cross_encoder_ms_per_query",
            "m700_pairs_per_second",
            "m1000_pairs_per_second",
            "m1000_latency_ratio",
            "reranker_model",
            "oracle",
        )
    }
    main["m1000_evidence"]["report"] = str(report)
    main_path.write_text(json.dumps(main, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    from run_retrieval_error_analysis import _report as write_aggregate_report

    write_aggregate_report(main, aggregate_report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edition-dir", type=Path, default=Path("data/law/2026-08-04"))
    parser.add_argument("--cases-dir", type=Path, default=Path("evals/cases"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("data/analysis"))
    parser.add_argument(
        "--report", type=Path, default=Path("evals/reports/m1000_evidence_comparison.md")
    )
    parser.add_argument(
        "--aggregate-report",
        type=Path,
        default=Path("evals/reports/retrieval_error_analysis.md"),
    )
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--checkpoint-pairs", type=int, default=240)
    parser.add_argument("--precision", choices=("fp32",), default="fp32")
    args = parser.parse_args(argv)

    cases = [case for case in load_verified(args.cases_dir) if case.split == "dev"]
    if len(cases) != 52:
        raise ValueError(f"expected 52 dev cases, got {len(cases)}")
    locked_article = json.loads(
        (args.analysis_dir / "article_shortlist_fusion_state.json").read_text(encoding="utf-8")
    )["winner"]["spec"]

    m700_chunks, dense700, bm25_indexes = _load_indexes(args.edition_dir)
    dense700_full, _ = _rank_cases(dense700, cases, k=len(m700_chunks))
    bm25_article_full, _ = _rank_cases(
        bm25_indexes[BM25_ARTICLE_CHUNK],
        cases,
        k=len(bm25_indexes[BM25_ARTICLE_CHUNK].chunks),
    )
    bm25_700_full, _ = _rank_cases(Bm25Index(m700_chunks), cases, k=len(m700_chunks))
    dense_articles = [unique_article_prefix(row, SOURCE_ARTICLES) for row in dense700_full]
    bm25_articles = [unique_article_prefix(row, SOURCE_ARTICLES) for row in bm25_article_full]
    selected = _selected_articles(
        dense_articles,
        bm25_articles,
        int(locked_article["dense_articles"]),
        int(locked_article["bm25_articles"]),
    )
    first_maps = [
        article_candidate_map(
            [dense, bm25], articles, per_article=int(locked_article["first_pass_chunks"])
        )
        for dense, bm25, articles in zip(
            dense700_full, bm25_700_full, selected, strict=True
        )
    ]
    first_pools = _pools(first_maps, selected, int(locked_article["first_pass_chunks"]))
    first_scores = _scores_for_cases(
        reranker_cache_path(args.analysis_dir, "fp32"),
        cases,
        first_pools,
        corpus_fingerprint(m700_chunks),
    )
    shortlists: list[list[str]] = []
    for articles, dense, bm25, pool, score in zip(
        selected, dense_articles, bm25_articles, first_pools, first_scores, strict=True
    ):
        ce_ranking, evidence = rank_articles_by_best_chunk(pool, score, articles)
        ce_scores = {article: score[evidence[article].chunk_id] for article in ce_ranking}
        shortlists.append(
            _shortlist(
                locked_article,
                candidates=articles,
                dense_articles=dense,
                bm25_articles=bm25,
                ce_ranking=ce_ranking,
                ce_scores=ce_scores,
                limit=int(locked_article["shortlist_articles"]),
            )
        )

    m700_maps = [
        article_candidate_map([dense, bm25], articles, per_article=8)
        for dense, bm25, articles in zip(
            dense700_full, bm25_700_full, shortlists, strict=True
        )
    ]
    m700_pools = _pools(m700_maps, shortlists, 8)
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    m700_fingerprint = corpus_fingerprint(m700_chunks)
    m700_fp32_cache = reranker_cache_path(args.analysis_dir, "fp32")
    seeded_pairs = 0
    m700_scores, _ = _score_pools(
        cases,
        m700_pools,
        split="dev",
        cache_path=m700_fp32_cache,
        fingerprint=m700_fingerprint,
        batch_size=args.batch_size,
        checkpoint_pairs=args.checkpoint_pairs,
        precision="fp32",
    )
    m700_costs = {chunk.chunk_id: _token_cost(chunk) for chunk in m700_chunks}
    m700_rows: list[Result] = []
    m700_controlled: dict[str, Result] = {}
    for size in CANDIDATE_SIZES:
        maps = _prefix_maps(m700_maps, size)
        pools = _pools(maps, shortlists, size)
        ranked = _ranked_maps(maps, m700_scores)
        fixed = _result(
            f"m700 fixed20 c{size}",
            {"max_chunk": 700, "family": "fixed20", "candidates_per_article": size},
            cases,
            _fixed_pack(shortlists, ranked, budget=20, cap=2),
            pools,
        )
        locked_result = _result(
            f"m700 locked dynamic c{size}",
            {
                "max_chunk": 700,
                "family": "locked_dynamic",
                "candidates_per_article": size,
                **LOCKED_DYNAMIC,
            },
            cases,
            _dynamic_pack(shortlists, ranked, m700_costs, **LOCKED_DYNAMIC),
            pools,
        )
        pack30_result = _result(
            f"m700 pack30 c{size}",
            {"max_chunk": 700, "family": "pack30", "candidates_per_article": size},
            cases,
            _fixed_pack(shortlists, ranked, budget=30, cap=3),
            pools,
        )
        m700_rows.extend((fixed, locked_result, pack30_result))
        if size == 8:
            m700_controlled = {"fixed20": fixed, "locked": locked_result}
        for token_budget in TOKEN_BUDGETS:
            for ratio in SCORE_RATIOS:
                for bonus in HIERARCHY_BONUSES:
                    m700_rows.append(
                        _result(
                            f"m700 dynamic c{size} t{token_budget} r{ratio:g} h{bonus:g}",
                            {
                                "max_chunk": 700,
                                "family": "dynamic",
                                "candidates_per_article": size,
                                "token_budget": token_budget,
                                "min_relative_score": ratio,
                                "hierarchy_bonus": bonus,
                            },
                            cases,
                            _dynamic_pack(
                                shortlists,
                                ranked,
                                m700_costs,
                                token_budget=token_budget,
                                min_relative_score=ratio,
                                hierarchy_bonus=bonus,
                            ),
                            pools,
                        )
                    )
    m700_fixed_result = m700_controlled["fixed20"]
    m700_result = m700_controlled["locked"]
    m700_best20 = _select([row for row in m700_rows if row.spec["family"] != "pack30"])
    m700_pack30 = _select([row for row in m700_rows if row.spec["family"] == "pack30"])

    m1000_chunks = load_chunks(chunks_path(args.edition_dir, M1000))
    dense1000 = DenseIndex.load_cached(
        m1000_chunks,
        dense700._encoder,
        cache_dir=args.edition_dir / "dense",
        model_name=MODEL,
        max_chunk=M1000,
    )
    if dense1000 is None:
        raise RuntimeError(f"missing compatible dense cache for {MODEL} m1000")
    dense1000_full, _ = _rank_cases(dense1000, cases, k=len(m1000_chunks))
    bm25_1000_full, _ = _rank_cases(Bm25Index(m1000_chunks), cases, k=len(m1000_chunks))
    max_candidates = max(CANDIDATE_SIZES)
    m1000_maps16 = [
        article_candidate_map([dense, bm25], articles, per_article=max_candidates)
        for dense, bm25, articles in zip(
            dense1000_full, bm25_1000_full, shortlists, strict=True
        )
    ]
    m1000_pools16 = _pools(m1000_maps16, shortlists, max_candidates)

    m1000_cache = args.analysis_dir / "m1000_evidence_scores_fp32.json"
    m1000_scores, _ = _score_pools(
        cases,
        m1000_pools16,
        split="dev",
        cache_path=m1000_cache,
        fingerprint=corpus_fingerprint(m1000_chunks),
        batch_size=args.batch_size,
        checkpoint_pairs=args.checkpoint_pairs,
        precision="fp32",
    )
    m1000_costs = {chunk.chunk_id: _token_cost(chunk) for chunk in m1000_chunks}
    rows: list[Result] = []
    controlled: dict[str, Result] = {}
    for size in CANDIDATE_SIZES:
        maps = _prefix_maps(m1000_maps16, size)
        pools = _pools(maps, shortlists, size)
        ranked = _ranked_maps(maps, m1000_scores)
        fixed20_rankings = _fixed_pack(shortlists, ranked, budget=20, cap=2)
        fixed20 = _result(
            f"m1000 fixed20 c{size}",
            {"max_chunk": 1000, "family": "fixed20", "candidates_per_article": size},
            cases,
            fixed20_rankings,
            pools,
        )
        rows.append(fixed20)
        locked_rankings = _dynamic_pack(
            shortlists, ranked, m1000_costs, **LOCKED_DYNAMIC
        )
        locked = _result(
            f"m1000 locked dynamic c{size}",
            {
                "max_chunk": 1000,
                "family": "locked_dynamic",
                "candidates_per_article": size,
                **LOCKED_DYNAMIC,
            },
            cases,
            locked_rankings,
            pools,
        )
        rows.append(locked)
        pack30_rankings = _fixed_pack(shortlists, ranked, budget=30, cap=3)
        rows.append(
            _result(
                f"m1000 pack30 c{size}",
                {"max_chunk": 1000, "family": "pack30", "candidates_per_article": size},
                cases,
                pack30_rankings,
                pools,
            )
        )
        if size == 8:
            controlled["fixed20"] = fixed20
            controlled["locked"] = locked
        for token_budget in TOKEN_BUDGETS:
            for ratio in SCORE_RATIOS:
                for bonus in HIERARCHY_BONUSES:
                    rankings = _dynamic_pack(
                        shortlists,
                        ranked,
                        m1000_costs,
                        token_budget=token_budget,
                        min_relative_score=ratio,
                        hierarchy_bonus=bonus,
                    )
                    rows.append(
                        _result(
                            f"m1000 dynamic c{size} t{token_budget} r{ratio:g} h{bonus:g}",
                            {
                                "max_chunk": 1000,
                                "family": "dynamic",
                                "candidates_per_article": size,
                                "token_budget": token_budget,
                                "min_relative_score": ratio,
                                "hierarchy_bonus": bonus,
                            },
                            cases,
                            rankings,
                            pools,
                        )
                    )

    best20 = _select([row for row in rows if row.spec["family"] != "pack30"])
    pack30 = _select([row for row in rows if row.spec["family"] == "pack30"])
    locked1000 = controlled["locked"]
    oracle = _oracle_breakdown(
        cases, shortlists, m1000_chunks, m1000_pools16, locked1000.rankings
    )
    if "exact_unavailable_in_m700" in oracle:
        oracle["exact_unavailable_in_m1000"] = oracle.pop("exact_unavailable_in_m700")
    m700_pairs = sum(len(pool) for pool in m700_pools)
    m1000_pairs = sum(len(pool) for pool in m1000_pools16)
    m700_rate = _cache_pairs_per_second(m700_fp32_cache, "dev")
    m1000_rate = _cache_pairs_per_second(m1000_cache, "dev")
    m700_ce_latency_ms = m700_pairs / m700_rate * 1000 / len(cases)
    m1000_ce_latency_ms = m1000_pairs / m1000_rate * 1000 / len(cases)
    state = {
        "version": STATE_VERSION,
        "split": "dev",
        "query": "blind",
        "precision": "fp32",
        "reranker_model": RERANKER_MODEL,
        "m700_seeded_original_fp32_pairs": seeded_pairs,
        "m700_cross_encoder_ms_per_query": m700_ce_latency_ms,
        "m1000_cross_encoder_ms_per_query": m1000_ce_latency_ms,
        "m700_pairs_per_second": m700_rate,
        "m1000_pairs_per_second": m1000_rate,
        "m1000_latency_ratio": m1000_ce_latency_ms / m700_ce_latency_ms,
        "n": len(cases),
        "count": len(rows),
        "m700_count": len(m700_rows),
        "m700_fixed20": _serialize(m700_fixed_result),
        "m700_locked_dynamic": _serialize(m700_result),
        "m700_best20": _serialize(m700_best20),
        "m700_pack30": _serialize(m700_pack30),
        "m1000_fixed20": _serialize(controlled["fixed20"]),
        "m1000_locked_dynamic": _serialize(locked1000),
        "m1000_best20": _serialize(best20),
        "m1000_pack30": _serialize(pack30),
        "locked_m1000_vs_m700": _comparison(cases, m700_result, locked1000),
        "fixed20_m1000_vs_m700": _comparison(
            cases, m700_fixed_result, controlled["fixed20"]
        ),
        "best_m1000_vs_m700": _comparison(cases, m700_result, best20),
        "best_m1000_vs_best_m700": _comparison(cases, m700_best20, best20),
        "oracle": oracle,
        "results": [_serialize(row) for row in rows],
        "m700_results": [_serialize(row) for row in m700_rows],
    }
    state_path = args.analysis_dir / "m1000_evidence_state.json"
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(state, args.report)
    _merge_aggregate(state, args.analysis_dir, args.report, args.aggregate_report)
    print(f"wrote {args.report}", file=sys.stderr)
    print(f"wrote {state_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
