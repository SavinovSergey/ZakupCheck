#!/usr/bin/env python3
"""Separate article retrieval from evidence packing for two chunking schemes."""

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

from index.chunking_roles import (  # noqa: E402
    covered_fraction,
    outside_selected_fraction,
    overlap_weighted_article_ranking,
    pack_article_evidence,
)
from index.error_analysis import paired_group_bootstrap  # noqa: E402
from index.eval_tables import load_verified  # noqa: E402
from index.fusion import fuse_article_rankings, project_article_ranking  # noqa: E402
from index.recall import article_key, score_case  # noqa: E402
from run_chunking_comparison import (  # noqa: E402
    BM25_MAX_CHUNKS,
    MODEL,
    POOL,
    RRF_KS,
    SEED,
    WEIGHTS,
    HybridSpec,
    _indexes,
    _rank_cases,
)
from schemas.corpus import Chunk  # noqa: E402
from schemas.eval_case import EvalCase  # noqa: E402

ARTICLE_LIMIT = 10
TOKEN_BUDGETS = (1500, 3000, 4500)
CHARS_PER_TOKEN = 3
USABLE_GOLD_COVERAGE = 0.5
STATE_VERSION = 1


def _load_spans(edition_dir: Path) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[int, int]]]:
    norm_spans: dict[str, tuple[int, int]] = {}
    article_spans: dict[str, tuple[int, int]] = {}
    for line in (edition_dir / "norm_units.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        if row["is_repealed"]:
            continue
        span = (row["char_start"], row["char_end"])
        norm_spans[row["norm_id"]] = span
        if row["level"] == "article":
            article_spans[article_key(row["norm_id"])] = span
    return norm_spans, article_spans


def _article_metrics(cases: list[EvalCase], rankings: list[list[str]]) -> dict[str, float | int]:
    recalls: list[float] = []
    all_hits: list[float] = []
    for case, ranked in zip(cases, rankings, strict=True):
        gold = {article_key(norm_id) for norm_id in case.gold_norms}
        found = gold & set(ranked[:ARTICLE_LIMIT])
        recalls.append(len(found) / len(gold))
        all_hits.append(float(found == gold))
    return {
        "n": len(cases),
        "recall_article": sum(recalls) / len(recalls),
        "all_article": sum(all_hits) / len(all_hits),
    }


def _article_values(cases: list[EvalCase], rankings: list[list[str]]) -> list[float]:
    values = []
    for case, ranked in zip(cases, rankings, strict=True):
        gold = {article_key(norm_id) for norm_id in case.gold_norms}
        values.append(len(gold & set(ranked[:ARTICLE_LIMIT])) / len(gold))
    return values


def _fuse(
    dense: list[list[str]],
    bm25: list[list[str]],
    spec: HybridSpec,
) -> list[list[str]]:
    output = []
    for dense_row, bm25_row in zip(dense, bm25, strict=True):
        candidates = list(dict.fromkeys([*dense_row, *bm25_row]))
        output.append(
            fuse_article_rankings(
                [dense_row, bm25_row],
                candidates=candidates,
                weights=[spec.dense_weight, spec.bm25_weight],
                k=spec.k,
                limit=ARTICLE_LIMIT,
            )
        )
    return output


def _select_hybrid(
    cases: list[EvalCase],
    dense: list[list[str]],
    bm25: dict[int, list[list[str]]],
) -> tuple[HybridSpec, list[list[str]], int]:
    candidates: list[tuple[dict[str, float | int], HybridSpec, list[list[str]]]] = []
    for size in BM25_MAX_CHUNKS:
        for k in RRF_KS:
            for dense_weight, bm25_weight in WEIGHTS:
                spec = HybridSpec(size, k, dense_weight, bm25_weight, ARTICLE_LIMIT)
                ranking = _fuse(dense, bm25[size], spec)
                candidates.append((_article_metrics(cases, ranking), spec, ranking))
    best_recall = max(float(row[0]["recall_article"]) for row in candidates)
    tied = [row for row in candidates if best_recall - float(row[0]["recall_article"]) < 0.01]
    best_all = max(float(row[0]["all_article"]) for row in tied)
    tied = [row for row in tied if float(row[0]["all_article"]) == best_all]
    metrics, spec, ranking = min(
        tied,
        key=lambda row: (
            row[1].bm25_max_chunk,
            row[1].k,
            row[1].dense_weight,
            row[1].bm25_weight,
        ),
    )
    del metrics
    return spec, ranking, len(candidates)


def _evidence_metrics(
    cases: list[EvalCase],
    packs: list[list[Chunk]],
    selected_articles: list[list[str]],
    norm_spans: dict[str, tuple[int, int]],
    article_spans: dict[str, tuple[int, int]],
) -> tuple[dict[str, float | int], list[float]]:
    exact = []
    article = []
    span_coverage = []
    usable = []
    chunks = []
    chars = []
    outside = []
    extra_articles = []
    for case, pack, selected in zip(cases, packs, selected_articles, strict=True):
        ids = {norm_id for chunk in pack for norm_id in chunk.norm_ids}
        score = score_case(case.gold_norms, ids)
        exact.append(score.exact.recall)
        article.append(score.article.recall)
        coverages = [
            covered_fraction(pack, norm_spans[norm_id]) if norm_id in norm_spans else 0.0
            for norm_id in case.gold_norms
        ]
        span_coverage.append(sum(coverages) / len(coverages))
        usable.append(sum(value >= USABLE_GOLD_COVERAGE for value in coverages) / len(coverages))
        chunks.append(len(pack))
        chars.append(sum(len(chunk.text) for chunk in pack))
        outside.append(outside_selected_fraction(pack, selected, article_spans))
        retrieved = {article_key(norm_id) for norm_id in ids}
        extra_articles.append(len(retrieved - set(selected)))
    n = len(cases)
    return (
        {
            "n": n,
            "label_exact_recall": sum(exact) / n,
            "article_recall": sum(article) / n,
            "gold_span_coverage": sum(span_coverage) / n,
            "usable_gold_recall": sum(usable) / n,
            "mean_chunks": sum(chunks) / n,
            "mean_chars": sum(chars) / n,
            "outside_selected_fraction": sum(outside) / n,
            "mean_extra_articles": sum(extra_articles) / n,
        },
        usable,
    )


def _pack_rows(
    cases: list[EvalCase],
    dense_chunks: list[list[Chunk]],
    selected_articles: list[list[str]],
    norm_spans: dict[str, tuple[int, int]],
    article_spans: dict[str, tuple[int, int]],
) -> tuple[dict[str, dict[str, float | int]], dict[int, list[float]]]:
    rows = {}
    values = {}
    for token_budget in TOKEN_BUDGETS:
        packs = [
            pack_article_evidence(
                ranked,
                selected,
                article_spans,
                char_budget=token_budget * CHARS_PER_TOKEN,
            )
            for ranked, selected in zip(dense_chunks, selected_articles, strict=True)
        ]
        row, usable = _evidence_metrics(
            cases, packs, selected_articles, norm_spans, article_spans
        )
        rows[str(token_budget)] = row
        values[token_budget] = usable
    return rows, values


def _spec_label(spec: dict[str, Any]) -> str:
    return (
        f"BM25 m{spec['bm25_max_chunk']}, k={spec['k']}, "
        f"w={int(spec['dense_weight'])}:{int(spec['bm25_weight'])}"
    )


def _write_report(state: dict[str, Any], path: Path) -> None:
    lines = [
        "# Article retrieval and evidence packing by chunking scheme",
        "",
        "This control separates two roles that the previous chunk-level R@10 mixed together. "
        "Article retrieval gives every article its own top-10 slot. Evidence packing keeps multi-article "
        "windows intact and compares equal context budgets.",
        "",
        "## Article retrieval",
        "",
        "A chunk contributes its rank to every intersected article, weighted by the fraction of chunk "
        "characters inside that article. Each article keeps only its best chunk contribution; this prevents "
        "a boundary sliver or a larger number of chunks from receiving a full extra rank.",
        "",
        "| split | method | structural article/all-hit R@10 | window article/all-hit R@10 | window Δ article (95% CI) |",
        "|---|---|---:|---:|---:|",
    ]
    for split in ("dev", "eval"):
        block = state["article_retrieval"][split]
        for method in ("dense", "bm25_m1500", "bm25_m2500", "hybrid"):
            left = block["structural"][method]
            right = block["window"][method]
            ci = block["comparisons"][method]
            lines.append(
                f"| {split} | {method} | {left['recall_article']:.3f}/{left['all_article']:.3f} | "
                f"{right['recall_article']:.3f}/{right['all_article']:.3f} | "
                f"{ci['mean_delta']:+.3f} [{ci['low']:+.3f}; {ci['high']:+.3f}] |"
            )
    lines.extend(
        [
            "",
            f"Structural dev winner: `{_spec_label(state['winners']['structural'])}`.  ",
            f"Window dev winner: `{_spec_label(state['winners']['window'])}`.",
            "",
            "### Projection sensitivity",
            "",
            "First-seen projection gives every article touched by a chunk the same rank. The overlap-weighted "
            "projection is the primary control because it discounts small boundary intersections.",
            "",
            "| split | scheme | method | first-seen article R@10 | overlap-weighted article R@10 |",
            "|---|---|---|---:|---:|",
        ]
    )
    for split in ("dev", "eval"):
        for scheme in ("structural", "window"):
            for method in ("dense", "bm25_m1500", "bm25_m2500"):
                row = state["article_retrieval"][split][scheme]
                lines.append(
                    f"| {split} | {scheme} | {method} | "
                    f"{row['first_seen'][method]['recall_article']:.3f} | "
                    f"{row[method]['recall_article']:.3f} |"
                )
    lines.extend(
        [
            "",
            "## Evidence packing under equal budgets",
            "",
            "The dev-selected article hybrid supplies ten articles. Dense m700 chunks are then packed with one "
            "representative per selected article first, followed by additional chunks in retrieval order. "
            "One token is approximated as three characters. `usable gold` requires at least 50% of the gold "
            "norm's source span, rather than merely carrying its `norm_id`.",
            "",
            "| split | budget | scheme | label exact | usable gold | gold span coverage | article recall | chunks | chars | outside selected | extra articles | window Δ usable (95% CI) |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in ("dev", "eval"):
        for budget in map(str, TOKEN_BUDGETS):
            for scheme in ("structural", "window"):
                row = state["evidence_packing"][split]["end_to_end"][scheme][budget]
                interval = state["evidence_packing"][split]["comparisons"][budget]
                interval_text = (
                    f"{interval['mean_delta']:+.3f} [{interval['low']:+.3f}; {interval['high']:+.3f}]"
                    if scheme == "window"
                    else "—"
                )
                lines.append(
                    f"| {split} | {budget} | {scheme} | {row['label_exact_recall']:.3f} | "
                    f"{row['usable_gold_recall']:.3f} | {row['gold_span_coverage']:.3f} | "
                    f"{row['article_recall']:.3f} | {row['mean_chunks']:.1f} | {row['mean_chars']:.0f} | "
                    f"{row['outside_selected_fraction']:.3f} | {row['mean_extra_articles']:.2f} | "
                    f"{interval_text} |"
                )
    lines.extend(
        [
            "",
            "### Conditional evidence control",
            "",
            "For this diagnostic only, both packers receive the gold articles. It isolates evidence selection "
            "from article-retrieval errors; it is not a deployable score.",
            "",
            "| split | budget | scheme | usable gold | gold span coverage | outside gold articles |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for split in ("dev", "eval"):
        for budget in map(str, TOKEN_BUDGETS):
            for scheme in ("structural", "window"):
                row = state["evidence_packing"][split]["oracle_articles"][scheme][budget]
                lines.append(
                    f"| {split} | {budget} | {scheme} | {row['usable_gold_recall']:.3f} | "
                    f"{row['gold_span_coverage']:.3f} | {row['outside_selected_fraction']:.3f} |"
                )
    lines.extend(["", "## Conclusion", "", state["conclusion"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _merge_aggregate(state: dict[str, Any], analysis_dir: Path, report: Path) -> None:
    aggregate_path = analysis_dir / "retrieval_error_state.json"
    if not aggregate_path.is_file():
        return
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["chunking_role_comparison"] = {
        "report": str(report),
        "winners": state["winners"],
        "article_retrieval": state["article_retrieval"],
        "evidence_packing": state["evidence_packing"],
        "conclusion": state["conclusion"],
    }
    aggregate_path.write_text(
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
        default=Path("evals/reports/chunking_role_comparison.md"),
    )
    args = parser.parse_args(argv)
    cases = load_verified(args.cases_dir)
    norm_spans, article_spans = _load_spans(args.edition_dir)
    dense_corpora, dense_indexes, bm25_indexes = _indexes(args.edition_dir)

    state: dict[str, Any] = {
        "version": STATE_VERSION,
        "seed": SEED,
        "model": MODEL,
        "token_budgets": list(TOKEN_BUDGETS),
        "chars_per_token": CHARS_PER_TOKEN,
        "usable_gold_coverage": USABLE_GOLD_COVERAGE,
        "article_retrieval": {},
        "evidence_packing": {},
    }
    winners: dict[str, HybridSpec] = {}
    for split in ("dev", "eval"):
        split_cases = [case for case in cases if case.split == split]
        raw: dict[str, dict[str, Any]] = {}
        state["article_retrieval"][split] = {}
        for scheme in ("structural", "window"):
            dense_ranked, _ = _rank_cases(
                dense_indexes[scheme], split_cases, variant="blind", k=len(dense_corpora[scheme])
            )
            bm25_ranked = {}
            for size, index in bm25_indexes[scheme].items():
                bm25_ranked[size], _ = _rank_cases(
                    index, split_cases, variant="blind", k=POOL
                )
            raw[scheme] = {"dense": dense_ranked, "bm25": bm25_ranked}
            dense_articles = [
                overlap_weighted_article_ranking(row, article_spans, pool=POOL)
                for row in dense_ranked
            ]
            bm25_articles = {
                size: [
                    overlap_weighted_article_ranking(row, article_spans, pool=POOL)
                    for row in ranked
                ]
                for size, ranked in bm25_ranked.items()
            }
            first_seen = {
                "dense": _article_metrics(
                    split_cases,
                    [project_article_ranking(row, pool=POOL) for row in dense_ranked],
                ),
                **{
                    f"bm25_m{size}": _article_metrics(
                        split_cases,
                        [project_article_ranking(row, pool=POOL) for row in ranked],
                    )
                    for size, ranked in bm25_ranked.items()
                },
            }
            if split == "dev":
                spec, hybrid_articles, candidate_count = _select_hybrid(
                    split_cases, dense_articles, bm25_articles
                )
                winners[scheme] = spec
            else:
                spec = winners[scheme]
                hybrid_articles = _fuse(dense_articles, bm25_articles[spec.bm25_max_chunk], spec)
                candidate_count = 0
            raw[scheme]["article_rankings"] = {
                "dense": dense_articles,
                **{f"bm25_m{size}": rows for size, rows in bm25_articles.items()},
                "hybrid": hybrid_articles,
            }
            state["article_retrieval"][split][scheme] = {
                "dense": _article_metrics(split_cases, dense_articles),
                **{
                    f"bm25_m{size}": _article_metrics(split_cases, rows)
                    for size, rows in bm25_articles.items()
                },
                "hybrid": _article_metrics(split_cases, hybrid_articles),
                "first_seen": first_seen,
                "hybrid_spec": asdict(spec),
                "candidate_count": candidate_count,
            }

        comparisons = {}
        for method in ("dense", "bm25_m1500", "bm25_m2500", "hybrid"):
            comparisons[method] = asdict(
                paired_group_bootstrap(
                    _article_values(split_cases, raw["structural"]["article_rankings"][method]),
                    _article_values(split_cases, raw["window"]["article_rankings"][method]),
                    [case.case_group_id for case in split_cases],
                    seed=SEED,
                )
            )
        state["article_retrieval"][split]["comparisons"] = comparisons

        state["evidence_packing"][split] = {"end_to_end": {}, "oracle_articles": {}}
        oracle_articles = [
            sorted({article_key(norm_id) for norm_id in case.gold_norms})
            for case in split_cases
        ]
        usable_values: dict[str, dict[int, list[float]]] = {}
        for scheme in ("structural", "window"):
            hybrid_articles = raw[scheme]["article_rankings"]["hybrid"]
            rows, values = _pack_rows(
                split_cases,
                raw[scheme]["dense"],
                hybrid_articles,
                norm_spans,
                article_spans,
            )
            state["evidence_packing"][split]["end_to_end"][scheme] = rows
            usable_values[scheme] = values
            oracle_rows, _ = _pack_rows(
                split_cases,
                raw[scheme]["dense"],
                oracle_articles,
                norm_spans,
                article_spans,
            )
            state["evidence_packing"][split]["oracle_articles"][scheme] = oracle_rows
        state["evidence_packing"][split]["comparisons"] = {
            str(budget): asdict(
                paired_group_bootstrap(
                    usable_values["structural"][budget],
                    usable_values["window"][budget],
                    [case.case_group_id for case in split_cases],
                    seed=SEED,
                )
            )
            for budget in TOKEN_BUDGETS
        }

    state["winners"] = {scheme: asdict(spec) for scheme, spec in winners.items()}
    dev_article = state["article_retrieval"]["dev"]
    dev_evidence = state["evidence_packing"]["dev"]["end_to_end"]
    state["conclusion"] = (
        "After separating roles, the comparison no longer treats several articles in one window as several "
        "free top-10 hits. Dev/blind hybrid article R@10 is "
        f"{dev_article['structural']['hybrid']['recall_article']:.3f} for structural and "
        f"{dev_article['window']['hybrid']['recall_article']:.3f} for window. Under a 3000-token evidence "
        "budget, usable gold recall is "
        f"{dev_evidence['structural']['3000']['usable_gold_recall']:.3f} versus "
        f"{dev_evidence['window']['3000']['usable_gold_recall']:.3f}; window outside-selected text is "
        f"{dev_evidence['window']['3000']['outside_selected_fraction']:.3f}. "
        "The two numbers answer different questions: article ranking chooses legal provisions, while evidence "
        "packing measures the context actually delivered to generation."
    )
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.analysis_dir / "chunking_role_comparison_state.json"
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(state, args.report)
    _merge_aggregate(state, args.analysis_dir, args.report)
    print(state["conclusion"])
    print(f"wrote {args.report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
