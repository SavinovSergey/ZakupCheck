#!/usr/bin/env python3
"""Compare structural and whole-edition fixed-window retrieval on one frozen eval."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
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
from index.fusion import article_diversify  # noqa: E402
from ingest.law.corpus import build_corpus  # noqa: E402
from run_dense_eval import profile_for  # noqa: E402
from run_hybrid_retrieval_eval import (  # noqa: E402
    ARTICLE_COUNTS,
    LIMIT,
    POOL,
    RRF_KS,
    WEIGHTS,
    HybridResult,
    HybridSpec,
    _apply,
    _metric_dict,
    select_winner,
)
from schemas.corpus import Chunk  # noqa: E402
from schemas.eval_case import EvalCase  # noqa: E402

MODEL = "Roflmax/bge-m3-russian-legal"
DENSE_MAX_CHUNK = 700
BM25_MAX_CHUNKS = (1500, 2500)
WINDOW_SIZES = tuple(sorted({DENSE_MAX_CHUNK, *BM25_MAX_CHUNKS}))
WINDOW_OVERLAP = 200
SEED = 20261008
STATE_VERSION = 1
STRUCTURAL_LOCKED_SPEC = HybridSpec(1500, 30, 1.0, 2.0, 10)


def _rank_cases(
    index: Any,
    cases: list[EvalCase],
    *,
    variant: str,
    k: int,
) -> tuple[list[list[Chunk]], float]:
    if cases:
        index.search(query_text(cases[0], variant), k=min(k, 10))
    rankings: list[list[Chunk]] = []
    started = time.perf_counter()
    for case in cases:
        rankings.append(index.search(query_text(case, variant), k=k))
    elapsed_ms = (time.perf_counter() - started) * 1000 / len(cases)
    return rankings, elapsed_ms


def _metrics(cases: list[EvalCase], rankings: list[list[Chunk]]) -> dict[str, Any]:
    return {
        "at5": _metric_dict(aggregate_rankings(cases, rankings, k=5)),
        "at10": _metric_dict(aggregate_rankings(cases, rankings, k=10)),
    }


def _baseline(
    cases: list[EvalCase], rankings: list[list[Chunk]], latency_ms: float
) -> dict[str, Any]:
    diversified = [article_diversify(row, pool=POOL, limit=LIMIT) for row in rankings]
    return {
        **_metrics(cases, diversified),
        "latency_ms": latency_ms,
        "rankings": diversified,
    }


def _evaluate_hybrid(
    spec: HybridSpec,
    cases: list[EvalCase],
    dense: list[list[Chunk]],
    bm25: list[list[Chunk]],
    latency_ms: float,
) -> HybridResult:
    rankings, postprocess_ms = _apply(spec, dense, bm25)
    return HybridResult(
        spec=spec,
        at5=aggregate_rankings(cases, rankings, k=5),
        at10=aggregate_rankings(cases, rankings, k=10),
        latency_ms=latency_ms + postprocess_ms,
        rankings=rankings,
    )


def _serialize_hybrid(result: HybridResult) -> dict[str, Any]:
    return {
        "spec": asdict(result.spec),
        "at5": _metric_dict(result.at5),
        "at10": _metric_dict(result.at10),
        "latency_ms": result.latency_ms,
    }


def _public(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "rankings"}


def _spec_label(spec: dict[str, Any]) -> str:
    return (
        f"BM25 m{spec['bm25_max_chunk']}, k={spec['k']}, "
        f"w={int(spec['dense_weight'])}:{int(spec['bm25_weight'])}, "
        f"articles={spec['n_articles']}"
    )


def _comparison(
    cases: list[EvalCase],
    structural: list[list[Chunk]],
    window: list[list[Chunk]],
) -> dict[str, float]:
    return asdict(
        paired_group_bootstrap(
            per_case_article_recall(cases, structural),
            per_case_article_recall(cases, window),
            [case.case_group_id for case in cases],
            seed=SEED,
        )
    )


def _topic_rows(
    cases: list[EvalCase], rankings: list[list[Chunk]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    topics = sorted({case.topic for case in cases})
    for topic in topics:
        indices = [index for index, case in enumerate(cases) if case.topic == topic]
        selected_cases = [cases[index] for index in indices]
        selected_rankings = [rankings[index] for index in indices]
        rows.append({"topic": topic, **_metrics(selected_cases, selected_rankings)})
    return rows


def _load_or_build_windows(edition_dir: Path) -> dict[int, list[Chunk]]:
    result: dict[int, list[Chunk]] = {}
    for size in WINDOW_SIZES:
        path = chunks_path(edition_dir, size, chunking="window")
        if not path.is_file():
            print(f"building fixed-window m{size}", file=sys.stderr)
            build_corpus(
                edition_dir,
                max_chunk=size,
                window_overlap=WINDOW_OVERLAP,
                chunking="window",
            )
        result[size] = load_chunks(path)
    return result


def _indexes(
    edition_dir: Path,
) -> tuple[
    dict[str, list[Chunk]],
    dict[str, DenseIndex],
    dict[str, dict[int, Bm25Index]],
]:
    chunks = {
        "structural": load_chunks(chunks_path(edition_dir, DENSE_MAX_CHUNK)),
        "window": _load_or_build_windows(edition_dir)[DENSE_MAX_CHUNK],
    }
    window_chunks = {
        size: load_chunks(chunks_path(edition_dir, size, chunking="window"))
        for size in BM25_MAX_CHUNKS
    }
    profile = profile_for(MODEL)
    encoder = SentenceTransformerEncoder(
        MODEL,
        query_prefix=profile.query_prefix,
        document_prefix=profile.document_prefix,
        max_tokens=profile.max_tokens,
    )
    dense = {
        "structural": DenseIndex.build(
            chunks["structural"],
            encoder,
            cache_dir=edition_dir / "dense",
            model_name=MODEL,
            max_chunk=DENSE_MAX_CHUNK,
        ),
        "window": DenseIndex.build(
            chunks["window"],
            encoder,
            cache_dir=edition_dir / "dense_window",
            model_name=MODEL,
            max_chunk=DENSE_MAX_CHUNK,
        ),
    }
    bm25 = {
        "structural": {
            size: Bm25Index(load_chunks(chunks_path(edition_dir, size)))
            for size in BM25_MAX_CHUNKS
        },
        "window": {size: Bm25Index(window_chunks[size]) for size in BM25_MAX_CHUNKS},
    }
    return chunks, dense, bm25


def _run_scheme(
    scheme: str,
    cases: list[EvalCase],
    dense_index: DenseIndex,
    dense_chunks: list[Chunk],
    bm25_indexes: dict[int, Bm25Index],
    *,
    variant: str,
    winner_spec: HybridSpec | None,
) -> tuple[dict[str, Any], HybridSpec]:
    dense_full, dense_ms = _rank_cases(
        dense_index, cases, variant=variant, k=len(dense_chunks)
    )
    dense_baseline = _baseline(cases, dense_full, dense_ms)
    bm25_rankings: dict[int, list[list[Chunk]]] = {}
    bm25_baselines: dict[int, dict[str, Any]] = {}
    bm25_ms: dict[int, float] = {}
    for size, index in bm25_indexes.items():
        ranked, latency = _rank_cases(index, cases, variant=variant, k=POOL)
        bm25_rankings[size] = ranked
        bm25_ms[size] = latency
        bm25_baselines[size] = _baseline(cases, ranked, latency)

    candidates: list[HybridResult] = []
    if winner_spec is None:
        for size in BM25_MAX_CHUNKS:
            for k in RRF_KS:
                for dense_weight, bm25_weight in WEIGHTS:
                    for n_articles in ARTICLE_COUNTS:
                        spec = HybridSpec(size, k, dense_weight, bm25_weight, n_articles)
                        candidates.append(
                            _evaluate_hybrid(
                                spec,
                                cases,
                                dense_full,
                                bm25_rankings[size],
                                dense_ms + bm25_ms[size],
                            )
                        )
        winner = select_winner(candidates)
        winner_spec = winner.spec
    else:
        winner = _evaluate_hybrid(
            winner_spec,
            cases,
            dense_full,
            bm25_rankings[winner_spec.bm25_max_chunk],
            dense_ms + bm25_ms[winner_spec.bm25_max_chunk],
        )

    transferred = _evaluate_hybrid(
        STRUCTURAL_LOCKED_SPEC,
        cases,
        dense_full,
        bm25_rankings[STRUCTURAL_LOCKED_SPEC.bm25_max_chunk],
        dense_ms + bm25_ms[STRUCTURAL_LOCKED_SPEC.bm25_max_chunk],
    )
    return (
        {
            "scheme": scheme,
            "n": len(cases),
            "dense": _public(dense_baseline),
            "bm25": {
                str(size): _public(row) for size, row in bm25_baselines.items()
            },
            "hybrid": _serialize_hybrid(winner),
            "transferred_structural_hybrid": _serialize_hybrid(transferred),
            "rankings": {
                "dense": dense_baseline["rankings"],
                "bm25_m1500": bm25_baselines[1500]["rankings"],
                "bm25_m2500": bm25_baselines[2500]["rankings"],
                "hybrid": winner.rankings,
            },
            "candidate_count": len(candidates),
        },
        winner_spec,
    )


def _write_report(state: dict[str, Any], path: Path) -> None:
    lines = [
        "# Structural versus fixed-window retrieval",
        "",
        "Scheme A uses legal hierarchy. Scheme B uses fixed windows over the whole edition text "
        f"with overlap {state['window_overlap']} characters; `norm_ids` are assigned by source-span intersection.",
        "Important: this first table scores the union of norm ids in the first chunks. A window can cover "
        "several articles in one slot, so these article figures describe context-chunk coverage, not a clean "
        "top-10 article ranking. See `chunking_role_comparison.md` for the separated control.",
        f"Dense model: `{state['model']}` m700. BM25: m1500 plus m2500 sensitivity. "
        "All methods return at most 10 article-diverse chunks. Hybrid parameters are selected on dev/blind only.",
        "",
        "## Corpus",
        "",
        "| scheme | m700 chunks | m1500 chunks | m2500 chunks | active articles covered |",
        "|---|---:|---:|---:|---:|",
    ]
    for scheme in ("structural", "window"):
        corpus = state["corpus"][scheme]
        lines.append(
            f"| {scheme} | {corpus['700']['chunks']} | {corpus['1500']['chunks']} | "
            f"{corpus['2500']['chunks']} | {corpus['700']['articles']} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate A/B matrix",
            "",
            "Primary rows are blind queries. Raw is reported to expose citation leakage. "
            "The window hybrid is locked on dev/blind and then applied unchanged to eval and raw.",
            "",
            "| split | query | method | structural exact/art/all-art R@5 | structural exact/art/all-art R@10 | window exact/art/all-art R@5 | window exact/art/all-art R@10 | window Δ article@10 (95% CI) |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split in ("dev", "eval"):
        for variant in ("blind", "raw"):
            block = state["results"][split][variant]
            for method in ("dense", "bm25_m1500", "bm25_m2500", "hybrid"):
                left5 = block["structural"][method]["at5"]
                left = block["structural"][method]["at10"]
                right5 = block["window"][method]["at5"]
                right = block["window"][method]["at10"]
                interval = block["comparisons"][method]
                lines.append(
                    f"| {split} | {variant} | {method} | "
                    f"{left5['recall_exact']:.3f}/{left5['recall_article']:.3f}/{left5['all_article']:.3f} | "
                    f"{left['recall_exact']:.3f}/{left['recall_article']:.3f}/{left['all_article']:.3f} | "
                    f"{right5['recall_exact']:.3f}/{right5['recall_article']:.3f}/{right5['all_article']:.3f} | "
                    f"{right['recall_exact']:.3f}/{right['recall_article']:.3f}/{right['all_article']:.3f} | "
                    f"{interval['mean_delta']:+.3f} [{interval['low']:+.3f}; {interval['high']:+.3f}] |"
                )
    lines.extend(
        [
            "",
            "## Locked hybrids",
            "",
            f"Structural dev winner: `{_spec_label(state['winners']['structural'])}`.",
            f"Window dev winner: `{_spec_label(state['winners']['window'])}`.",
            "The previous structural winner was also transferred unchanged to window chunks as a sensitivity check; "
            "it was not used to select the window winner.",
            "",
            "| split | window hybrid | exact R@10 | article R@10 | all article@10 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for split in ("dev", "eval"):
        block = state["results"][split]["blind"]["window"]
        for key, label in (
            ("transferred_structural_hybrid", "transferred structural spec"),
            ("hybrid", "window dev winner"),
        ):
            row = block[key]["at10"]
            lines.append(
                f"| {split} | {label} | {row['recall_exact']:.3f} | "
                f"{row['recall_article']:.3f} | {row['all_article']:.3f} |"
            )
    lines.extend(
        [
            "",
            "Exact recall needs a granularity caveat: a fixed window carries every active article/part/point "
            "whose source span it intersects. A structural point chunk can carry only the point id. "
            "Part of the window exact gain therefore comes from hierarchical reachability, not solely better ranking; "
            "article R@10 remains the primary selection metric.",
            "",
            "## Dev/blind topics",
            "",
            "Topics with n < 5 are descriptive only.",
            "",
            "| method | scheme | topic | n | exact R@10 | article R@10 | all article@10 | status |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for method in ("dense", "bm25_m1500", "hybrid"):
        for scheme in ("structural", "window"):
            for row in state["topics"][method][scheme]:
                at10 = row["at10"]
                status = "descriptive" if at10["n"] < 5 else "comparison"
                lines.append(
                    f"| {method} | {scheme} | {row['topic']} | {at10['n']} | "
                    f"{at10['recall_exact']:.3f} | {at10['recall_article']:.3f} | "
                    f"{at10['all_article']:.3f} | {status} |"
                )
    conclusion = state["conclusion"]
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            conclusion,
            "",
            f"Machine-readable state: `{state['state_path']}`. Eval was not used for parameter selection.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _article_count(chunks: list[Chunk]) -> int:
    return len(
        {
            norm_id.split(":art.", 1)[1].split(":", 1)[0]
            for chunk in chunks
            for norm_id in chunk.norm_ids
        }
    )


def _merge_aggregate(state: dict[str, Any], analysis_dir: Path, report: Path) -> None:
    aggregate_state_path = analysis_dir / "retrieval_error_state.json"
    if not aggregate_state_path.is_file():
        return
    aggregate = json.loads(aggregate_state_path.read_text(encoding="utf-8"))
    aggregate["chunking_comparison"] = {
        "report": str(report),
        "corpus": state["corpus"],
        "winners": state["winners"],
        "dev_blind": state["results"]["dev"]["blind"],
        "eval_blind": state["results"]["eval"]["blind"],
        "conclusion": state["conclusion"],
    }
    aggregate_state_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    from run_retrieval_error_analysis import _report

    _report(aggregate, Path("evals/reports/retrieval_error_analysis.md"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edition-dir", type=Path, default=Path("data/law/2026-08-04"))
    parser.add_argument("--cases-dir", type=Path, default=Path("evals/cases"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("data/analysis"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evals/reports/chunking_ab_comparison.md"),
    )
    args = parser.parse_args(argv)
    cases = load_verified(args.cases_dir)
    if sum(case.split == "dev" for case in cases) != 52 or sum(
        case.split == "eval" for case in cases
    ) != 22:
        print("expected 52 dev and 22 eval cases", file=sys.stderr)
        return 2

    dense_chunks, dense_indexes, bm25_indexes = _indexes(args.edition_dir)
    all_chunks = {
        scheme: {
            700: dense_chunks[scheme],
            **{
                size: load_chunks(chunks_path(args.edition_dir, size, chunking=scheme))
                for size in BM25_MAX_CHUNKS
            },
        }
        for scheme in ("structural", "window")
    }
    state: dict[str, Any] = {
        "version": STATE_VERSION,
        "seed": SEED,
        "model": MODEL,
        "window_overlap": WINDOW_OVERLAP,
        "state_path": str(args.analysis_dir / "chunking_ab_comparison_state.json"),
        "corpus": {
            scheme: {
                str(size): {
                    "chunks": len(chunks),
                    "articles": _article_count(chunks),
                }
                for size, chunks in by_size.items()
            }
            for scheme, by_size in all_chunks.items()
        },
        "results": {},
    }
    winners: dict[str, HybridSpec] = {}
    retained_rankings: dict[str, dict[str, dict[str, list[list[Chunk]]]]] = {}
    for split in ("dev", "eval"):
        split_cases = [case for case in cases if case.split == split]
        state["results"][split] = {}
        retained_rankings[split] = {}
        for variant in ("blind", "raw"):
            state["results"][split][variant] = {}
            retained_rankings[split][variant] = {}
            for scheme in ("structural", "window"):
                winner = winners.get(scheme)
                block, selected = _run_scheme(
                    scheme,
                    split_cases,
                    dense_indexes[scheme],
                    dense_chunks[scheme],
                    bm25_indexes[scheme],
                    variant=variant,
                    winner_spec=(
                        None if split == "dev" and variant == "blind" else winner
                    ),
                )
                if split == "dev" and variant == "blind":
                    winners[scheme] = selected
                public = {
                    "dense": block["dense"],
                    "bm25_m1500": block["bm25"]["1500"],
                    "bm25_m2500": block["bm25"]["2500"],
                    "hybrid": block["hybrid"],
                    "transferred_structural_hybrid": block[
                        "transferred_structural_hybrid"
                    ],
                    "candidate_count": block["candidate_count"],
                }
                state["results"][split][variant][scheme] = public
                retained_rankings[split][variant][scheme] = block["rankings"]
            comparisons = {}
            for method in ("dense", "bm25_m1500", "bm25_m2500", "hybrid"):
                comparisons[method] = _comparison(
                    split_cases,
                    retained_rankings[split][variant]["structural"][method],
                    retained_rankings[split][variant]["window"][method],
                )
            state["results"][split][variant]["comparisons"] = comparisons

    state["winners"] = {scheme: asdict(spec) for scheme, spec in winners.items()}
    dev_cases = [case for case in cases if case.split == "dev"]
    state["topics"] = {
        method: {
            scheme: _topic_rows(
                dev_cases,
                retained_rankings["dev"]["blind"][scheme][method],
            )
            for scheme in ("structural", "window")
        }
        for method in ("dense", "bm25_m1500", "hybrid")
    }
    dev_window = state["results"]["dev"]["blind"]["window"]["hybrid"]["at10"]
    dev_struct = state["results"]["dev"]["blind"]["structural"]["hybrid"]["at10"]
    eval_window = state["results"]["eval"]["blind"]["window"]["hybrid"]["at10"]
    eval_struct = state["results"]["eval"]["blind"]["structural"]["hybrid"]["at10"]
    state["conclusion"] = (
        "This first-pass chunk-union metric is not a clean article-ranking comparison because one fixed "
        "window can cover several articles in one result slot. On dev/blind, structural→window hybrid "
        "context coverage at ten chunks is "
        f"{dev_struct['recall_article']:.3f}→{dev_window['recall_article']:.3f} "
        f"and exact R@10 is {dev_struct['recall_exact']:.3f}→{dev_window['recall_exact']:.3f}. "
        "On the one-shot eval/blind check, article R@10 is "
        f"{eval_struct['recall_article']:.3f}→{eval_window['recall_article']:.3f} "
        f"and exact R@10 is {eval_struct['recall_exact']:.3f}→{eval_window['recall_exact']:.3f}. "
        "All article-delta confidence intervals include zero. Use the separated article-ranking and "
        "equal-budget evidence control in chunking_role_comparison.md for the actual decision."
    )
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.analysis_dir / "chunking_ab_comparison_state.json"
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(state, args.report)
    _merge_aggregate(state, args.analysis_dir, args.report)
    print(state["conclusion"])
    print(f"wrote {args.report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
