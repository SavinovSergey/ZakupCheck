#!/usr/bin/env python3
"""Case-level retrieval diagnostics and locked dev -> eval experiments.

The default ``dev`` phase selects a candidate without looking at eval.  The
``eval`` phase requires the saved dev state and evaluates only the two base
systems plus that locked winner.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from index.bm25 import Bm25Index, load_chunks  # noqa: E402
from index.dense import DenseIndex, SentenceTransformerEncoder  # noqa: E402
from index.error_analysis import (  # noqa: E402
    RANK_BUCKETS,
    aggregate_rankings,
    best_rank,
    complementarity,
    corpus_gold_statuses,
    paired_group_bootstrap,
    per_case_article_recall,
    rank_bucket,
    repeated_article_slots,
)
from index.eval_tables import chunks_path, load_verified, query_text  # noqa: E402
from index.fusion import article_diversify, interleave, is_pointer_text, weighted_rrf  # noqa: E402
from index.recall import AggregateRecall, article_key, score_case  # noqa: E402
from run_dense_eval import profile_for  # noqa: E402
from schemas.corpus import Chunk, NormUnit  # noqa: E402
from schemas.eval_case import EvalCase  # noqa: E402

MODEL = "Roflmax/bge-m3-russian-legal"
MAX_CHUNK = 700
BM25_SENSITIVITY_CHUNK = 1500
POOL = 50
LIMIT = 10
STATE_VERSION = 1


@dataclass(frozen=True)
class MethodResult:
    name: str
    spec: dict[str, Any]
    metrics: AggregateRecall
    latency_ms: float
    rankings: list[list[Chunk]]


def _metrics_dict(metrics: AggregateRecall) -> dict[str, float | int]:
    return asdict(metrics)


def _rank_cases(index: Any, cases: list[EvalCase], variant: str, k: int) -> tuple[list[list[Chunk]], float]:
    if cases:
        index.search(query_text(cases[0], variant), k=min(k, 10))
    rankings: list[list[Chunk]] = []
    elapsed = 0.0
    for case in cases:
        started = time.perf_counter()
        rankings.append(index.search(query_text(case, variant), k=k))
        elapsed += time.perf_counter() - started
    return rankings, elapsed * 1000 / len(cases)


def _title_prefixed(chunks: list[Chunk], edition_dir: Path) -> list[Chunk]:
    titles: dict[str, str] = {}
    for line in (edition_dir / "norm_units.jsonl").read_text(encoding="utf-8").splitlines():
        unit = NormUnit.model_validate_json(line)
        if unit.level == "article" and unit.title:
            titles[article_key(unit.norm_id)] = unit.title
    prefixed: list[Chunk] = []
    for chunk in chunks:
        articles = sorted({article_key(norm_id) for norm_id in chunk.norm_ids})
        title = next((titles[article] for article in articles if article in titles), None)
        text = f"{title}.\n{chunk.text}" if title else chunk.text
        prefixed.append(chunk.model_copy(update={"text": text, "article_title_prefix": bool(title)}))
    return prefixed


def _result(
    name: str,
    spec: dict[str, Any],
    cases: list[EvalCase],
    rankings: list[list[Chunk]],
    latency_ms: float,
) -> MethodResult:
    return MethodResult(name, spec, aggregate_rankings(cases, rankings), latency_ms, rankings)


def _winner(results: list[MethodResult]) -> MethodResult:
    best_article = max(item.metrics.recall_article for item in results)
    tied = [item for item in results if best_article - item.metrics.recall_article < 0.01]
    best_exact = max(item.metrics.recall_exact for item in tied)
    tied = [item for item in tied if item.metrics.recall_exact == best_exact]
    best_all = max(item.metrics.all_article for item in tied)
    tied = [item for item in tied if item.metrics.all_article == best_all]
    return min(tied, key=lambda item: (item.latency_ms, item.name))


def _build_dev_methods(
    cases: list[EvalCase],
    dense: list[list[Chunk]],
    bm25: list[list[Chunk]],
    dense_ms: float,
    bm25_ms: float,
    bm25_sensitivity: list[list[Chunk]],
    bm25_sensitivity_ms: float,
    dense_title: list[list[Chunk]],
    dense_title_ms: float,
    bm25_title: list[list[Chunk]],
    bm25_title_ms: float,
) -> list[MethodResult]:
    results = [
        _result("dense m700", {"kind": "dense"}, cases, dense, dense_ms),
        _result("BM25 m700", {"kind": "bm25"}, cases, bm25, bm25_ms),
        _result(
            "BM25 m1500 sensitivity",
            {"kind": "bm25_sensitivity"},
            cases,
            bm25_sensitivity,
            bm25_sensitivity_ms,
        ),
        _result("dense m700 + title", {"kind": "dense_title"}, cases, dense_title, dense_title_ms),
        _result("BM25 m700 + title", {"kind": "bm25_title"}, cases, bm25_title, bm25_title_ms),
    ]
    fusion_latency = dense_ms + bm25_ms
    rrf_results: list[MethodResult] = []
    for k in (0, 10, 30, 60):
        for weights in ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0)):
            rankings = [
                weighted_rrf([left, right], weights=weights, k=k, pool=POOL, limit=LIMIT)
                for left, right in zip(dense, bm25, strict=True)
            ]
            label = f"weighted RRF k={k} w={int(weights[0])}:{int(weights[1])}"
            rrf_results.append(
                _result(
                    label,
                    {"kind": "weighted_rrf", "k": k, "weights": list(weights)},
                    cases,
                    rankings,
                    fusion_latency,
                )
            )
    results.extend(rrf_results)
    for name, order in (("interleave dense-first", "dense-first"), ("interleave BM25-first", "bm25-first")):
        rankings = [
            interleave([left, right] if order == "dense-first" else [right, left], pool=POOL, limit=LIMIT)
            for left, right in zip(dense, bm25, strict=True)
        ]
        results.append(_result(name, {"kind": "interleave", "order": order}, cases, rankings, fusion_latency))
    results.append(
        _result(
            "dense article-diverse",
            {"kind": "diverse", "base": {"kind": "dense"}},
            cases,
            [article_diversify(ranked, pool=POOL, limit=LIMIT) for ranked in dense],
            dense_ms,
        )
    )
    results.append(
        _result(
            "BM25 article-diverse",
            {"kind": "diverse", "base": {"kind": "bm25"}},
            cases,
            [article_diversify(ranked, pool=POOL, limit=LIMIT) for ranked in bm25],
            bm25_ms,
        )
    )
    best_rrf = _winner(rrf_results)
    results.append(
        _result(
            f"{best_rrf.name} + article-diverse",
            {"kind": "diverse", "base": best_rrf.spec},
            cases,
            [article_diversify(ranked, pool=POOL, limit=LIMIT) for ranked in best_rrf.rankings],
            best_rrf.latency_ms,
        )
    )
    return results


def _apply_spec(
    spec: dict[str, Any],
    dense: list[list[Chunk]],
    bm25: list[list[Chunk]],
    *,
    bm25_sensitivity: list[list[Chunk]] | None = None,
    dense_title: list[list[Chunk]] | None = None,
    bm25_title: list[list[Chunk]] | None = None,
) -> list[list[Chunk]]:
    kind = spec["kind"]
    if kind == "dense":
        return dense
    if kind == "bm25":
        return bm25
    if kind == "bm25_sensitivity":
        if bm25_sensitivity is None:
            raise ValueError("winner needs BM25 m1500 rankings")
        return bm25_sensitivity
    if kind == "dense_title":
        if dense_title is None:
            raise ValueError("winner needs title-prefixed dense rankings")
        return dense_title
    if kind == "bm25_title":
        if bm25_title is None:
            raise ValueError("winner needs title-prefixed BM25 rankings")
        return bm25_title
    if kind == "weighted_rrf":
        return [
            weighted_rrf(
                [left, right],
                weights=spec["weights"],
                k=spec["k"],
                pool=POOL,
                limit=LIMIT,
            )
            for left, right in zip(dense, bm25, strict=True)
        ]
    if kind == "interleave":
        dense_first = spec["order"] == "dense-first"
        return [
            interleave([left, right] if dense_first else [right, left], pool=POOL, limit=LIMIT)
            for left, right in zip(dense, bm25, strict=True)
        ]
    if kind == "diverse":
        base = _apply_spec(
            spec["base"],
            dense,
            bm25,
            bm25_sensitivity=bm25_sensitivity,
            dense_title=dense_title,
            bm25_title=bm25_title,
        )
        return [article_diversify(ranked, pool=POOL, limit=LIMIT) for ranked in base]
    raise ValueError(f"unknown method kind: {kind}")


def _needs(spec: dict[str, Any], kind: str) -> bool:
    if spec.get("kind") == kind:
        return True
    return spec.get("kind") == "diverse" and _needs(spec["base"], kind)


def _top_chunks(ranked: list[Chunk]) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "chunk_id": chunk.chunk_id,
            "norm_ids": chunk.norm_ids,
            "text": " ".join(chunk.text.split())[:500],
        }
        for rank, chunk in enumerate(ranked[:LIMIT], start=1)
    ]


def _write_outcomes(
    path: Path,
    cases: list[EvalCase],
    baseline: list[list[Chunk]],
    winner: list[list[Chunk]],
) -> dict[str, Any]:
    """Persist paired case outcomes without putting local queries in the report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    gains = losses = ties = 0
    by_topic: dict[str, list[float]] = defaultdict(list)
    for case, before, after in zip(cases, baseline, winner, strict=True):
        before_ids = {norm_id for chunk in before[:LIMIT] for norm_id in chunk.norm_ids}
        after_ids = {norm_id for chunk in after[:LIMIT] for norm_id in chunk.norm_ids}
        before_score = score_case(case.gold_norms, before_ids)
        after_score = score_case(case.gold_norms, after_ids)
        delta = after_score.article.recall - before_score.article.recall
        gains += delta > 0
        losses += delta < 0
        ties += delta == 0
        by_topic[case.topic].append(delta)
        gold_articles = {article_key(norm_id) for norm_id in case.gold_norms}
        before_articles = {article_key(norm_id) for norm_id in before_ids}
        after_articles = {article_key(norm_id) for norm_id in after_ids}
        record = {
            "case_id": case.case_id,
            "case_group_id": case.case_group_id,
            "split": case.split,
            "topic": case.topic,
            "gold_articles": sorted(gold_articles),
            "baseline_article_recall": before_score.article.recall,
            "winner_article_recall": after_score.article.recall,
            "article_delta": delta,
            "baseline_exact_recall": before_score.exact.recall,
            "winner_exact_recall": after_score.exact.recall,
            "gained_articles": sorted((after_articles - before_articles) & gold_articles),
            "lost_articles": sorted((before_articles - after_articles) & gold_articles),
            "baseline_top10_articles": sorted(before_articles),
            "winner_top10_articles": sorted(after_articles),
        }
        lines.append(json.dumps(record, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "gains": gains,
        "losses": losses,
        "ties": ties,
        "mean_delta_by_topic": {
            topic: sum(values) / len(values) for topic, values in sorted(by_topic.items())
        },
    }


def _write_cases(
    path: Path,
    cases: list[EvalCase],
    rankings: dict[str, dict[str, list[list[Chunk]]]],
    gold_status: dict[str, str],
    pointer_ids: set[str],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    bucket_counts: dict[str, Counter[str]] = defaultdict(Counter)
    comp = Counter()
    wrong_pairs: Counter[tuple[str, str, str]] = Counter()
    review_cases: set[str] = set()
    lines: list[str] = []
    for index, case in enumerate(cases):
        raw_dense = rankings["raw"]["dense"][index]
        raw_bm25 = rankings["raw"]["bm25"][index]
        blind_dense = rankings["blind"]["dense"][index]
        blind_bm25 = rankings["blind"]["bm25"][index]
        raw_hits = {
            article_key(gold)
            for gold in case.gold_norms
            if complementarity(gold, raw_dense, raw_bm25) != "neither"
        }
        blind_hits = {
            article_key(gold)
            for gold in case.gold_norms
            if complementarity(gold, blind_dense, blind_bm25) != "neither"
        }
        for variant in ("raw", "blind"):
            dense = rankings[variant]["dense"][index]
            bm25 = rankings[variant]["bm25"][index]
            gold_rows = []
            for gold in case.gold_norms:
                dense_article_rank = best_rank(gold, dense, level="article")
                bm25_article_rank = best_rank(gold, bm25, level="article")
                dense_bucket = rank_bucket(dense_article_rank)
                bm25_bucket = rank_bucket(bm25_article_rank)
                state = complementarity(gold, dense, bm25)
                bucket_counts[f"{variant}:dense"][dense_bucket] += 1
                bucket_counts[f"{variant}:bm25"][bm25_bucket] += 1
                if variant == "blind":
                    comp[state] += 1
                    if state != "both":
                        review_cases.add(case.case_id)
                    gold_article = article_key(gold)
                    for system, ranked, bucket in (("dense", dense, dense_bucket), ("bm25", bm25, bm25_bucket)):
                        if bucket == "top-10":
                            continue
                        wrong = next(
                            (
                                article_key(norm_id)
                                for chunk in ranked[:LIMIT]
                                for norm_id in chunk.norm_ids
                                if article_key(norm_id) != gold_article
                            ),
                            "none",
                        )
                        wrong_pairs[(system, gold_article, wrong)] += 1
                gold_rows.append(
                    {
                        "gold_id": gold,
                        "corpus_status": gold_status[gold],
                        "pointer_text": gold in pointer_ids,
                        "dense_article_rank": dense_article_rank,
                        "dense_exact_rank": best_rank(gold, dense, level="exact"),
                        "dense_bucket": dense_bucket,
                        "bm25_article_rank": bm25_article_rank,
                        "bm25_exact_rank": best_rank(gold, bm25, level="exact"),
                        "bm25_bucket": bm25_bucket,
                        "complementarity": state,
                    }
                )
            tags = []
            query_words = len(query_text(case, variant).split())
            if query_words >= 100:
                tags.append("long-query")
            if query_words <= 20:
                tags.append("short-query")
            if case.complaint_argument_raw == case.complaint_argument_blind:
                tags.append("raw-equals-blind")
            if len(case.gold_norms) > 1:
                tags.append("multi-gold")
            if repeated_article_slots(dense, case.gold_norms) >= 2:
                tags.append("dense-crowding")
            if repeated_article_slots(bm25, case.gold_norms) >= 2:
                tags.append("bm25-crowding")
            if variant == "blind" and raw_hits != blind_hits:
                tags.append("citation-sensitive")
            record = {
                "case_id": case.case_id,
                "case_group_id": case.case_group_id,
                "split": case.split,
                "topic": case.topic,
                "query_variant": variant,
                "query": query_text(case, variant),
                "gold_norms": case.gold_norms,
                "gold": gold_rows,
                "tags": tags,
                "dense_repeated_wrong_article_slots": repeated_article_slots(dense, case.gold_norms),
                "bm25_repeated_wrong_article_slots": repeated_article_slots(bm25, case.gold_norms),
                "dense_top10": _top_chunks(dense),
                "bm25_top10": _top_chunks(bm25),
            }
            lines.append(json.dumps(record, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "buckets": {name: dict(counts) for name, counts in bucket_counts.items()},
        "complementarity": dict(comp),
        "wrong_pairs": [
            {"system": system, "gold": gold, "wrong": wrong, "count": count}
            for (system, gold, wrong), count in wrong_pairs.most_common(20)
        ],
        "review_cases": sorted(review_cases),
    }


def _parse_model_overview(report_dir: Path) -> list[dict[str, Any]]:
    overview = []
    for path in sorted(report_dir.glob("dense_*_by_max_chunk.md")):
        if path.name.endswith("_n6.md"):
            continue
        text = path.read_text(encoding="utf-8")
        match = re.search(r"\u041c\u043e\u0434\u0435\u043b\u044c: `([^`]+)`", text)
        if not match:
            continue
        model = match.group(1)
        active = False
        rows = []
        for line in text.splitlines():
            if line == "## dev / eval":
                active = True
                continue
            if active and line.startswith("## "):
                break
            if active and re.match(r"^\| \d+ \|", line):
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                if cells[2] == "dev":
                    rows.append({"chunk": int(cells[0]), "query": cells[1], "article_r10": float(cells[11])})
        blind = [row for row in rows if row["query"] == "blind"]
        if blind:
            best = max(blind, key=lambda row: (row["article_r10"], -row["chunk"]))
            overview.append({"model": model, **best})
    return sorted(overview, key=lambda row: (-row["article_r10"], row["model"]))


def _report(state: dict[str, Any], report_path: Path) -> None:
    dev = state["dev"]
    lines = [
        "# Error analysis retrieval",
        "",
        "\u041e\u0441\u043d\u043e\u0432\u043d\u043e\u0439 \u0441\u0446\u0435\u043d\u0430\u0440\u0438\u0439: `complaint_argument_blind`; \u043e\u0441\u043d\u043e\u0432\u043d\u0430\u044f \u043c\u0435\u0442\u0440\u0438\u043a\u0430: article R@10.",
        f"Dev \u0432\u044b\u0431\u0438\u0440\u0430\u0435\u0442 \u0433\u0438\u043f\u043e\u0442\u0435\u0437\u0443; eval \u0437\u0430\u043f\u0443\u0441\u043a\u0430\u0435\u0442\u0441\u044f \u0442\u043e\u043b\u044c\u043a\u043e \u0434\u043b\u044f \u0437\u0430\u0444\u0438\u043a\u0441\u0438\u0440\u043e\u0432\u0430\u043d\u043d\u043e\u0433\u043e \u043f\u043e\u0431\u0435\u0434\u0438\u0442\u0435\u043b\u044f. Seed bootstrap: `{state['seed']}`.",
        "",
        "## \u0412\u044b\u0432\u043e\u0434 \u043f\u043e dev",
        "",
        f"\u0412\u044b\u0431\u0440\u0430\u043d \u043c\u0435\u0442\u043e\u0434 **{dev['winner']['name']}** \u0434\u043b\u044f `{state['model']}`, "
        f"`max_chunk={state['max_chunk']}`: article R@10 "
        f"{dev['winner']['metrics']['recall_article']:.3f}, exact R@10 {dev['winner']['metrics']['recall_exact']:.3f}, "
        f"all article@10 {dev['winner']['metrics']['all_article']:.3f}.",
        f"\u041f\u0430\u0440\u043d\u0430\u044f \u0440\u0430\u0437\u043d\u0438\u0446\u0430 \u0441 dense baseline: {dev['bootstrap']['mean_delta']:+.3f} "
        f"(95% group-bootstrap [{dev['bootstrap']['low']:+.3f}; {dev['bootstrap']['high']:+.3f}]).",
        "",
        "## \u0413\u0438\u043f\u043e\u0442\u0435\u0437\u044b, dev / blind",
        "",
        "| \u043c\u0435\u0442\u043e\u0434 | R@10 exact | all@10 exact | R@10 art | all@10 art | ms/query* |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if "law_corpus_audit" in state:
        audit = state["law_corpus_audit"]
        stale = audit["stale_dense_caches"]
        cache_status = (
            f"**Важно:** {stale} существующих dense-кэшей относятся к предыдущей версии "
            "чанков. Dense, hybrid и cross-encoder результаты ниже являются историческими "
            "до пересчёта эмбеддингов. Отдельные BM25-отчёты уже пересчитаны; смешанная "
            "таблица error analysis ниже также остаётся исторической до полного перезапуска."
            if stale
            else "Все 34 базовых dense-кэша и отдельный title-prefix кэш обновлены для "
            "текущего корпуса; приведённые ниже dense, hybrid и cross-encoder эксперименты "
            "повторены после обновления."
        )
        lines[4:4] = [
            "",
            "## Покрытие корпуса",
            "",
            f"После исправления парсинга: {len(audit['active_articles'])} действующих статей, "
            f"{len(audit['repealed_articles'])} полностью утративших силу, "
            f"{len(audit['empty_articles'])} пустых; каждый структурный корпус покрывает "
            f"{len(audit['active_articles'])} из {len(audit['active_articles'])} действующих статей. "
            f"Полный аудит: `{audit['report']}`.",
            cache_status,
        ]
    for item in sorted(dev["methods"], key=lambda row: (-row["metrics"]["recall_article"], row["name"])):
        metrics = item["metrics"]
        lines.append(
            f"| {item['name']} | {metrics['recall_exact']:.3f} | {metrics['all_exact']:.3f} "
            f"| {metrics['recall_article']:.3f} | {metrics['all_article']:.3f} | {item['latency_ms']:.1f} |"
        )
    lines.extend(
        [
            "",
            "\\* latency \u0432 \u044d\u0442\u043e\u043c \u043e\u0442\u0447\u0451\u0442\u0435 \u0432\u043a\u043b\u044e\u0447\u0430\u0435\u0442 \u0441\u043e\u0440\u0442\u0438\u0440\u043e\u0432\u043a\u0443 \u0432\u0441\u0435\u0433\u043e \u043a\u043e\u0440\u043f\u0443\u0441\u0430 \u0434\u043b\u044f \u0434\u0438\u0430\u0433\u043d\u043e\u0441\u0442\u0438\u043a\u0438 \u0440\u0430\u043d\u0433\u0430; \u044d\u0442\u043e \u043d\u0435 production latency top-10.",
            "",
            "## \u041a\u0430\u0442\u0435\u0433\u043e\u0440\u0438\u0438 \u043e\u0448\u0438\u0431\u043e\u043a, dev / blind",
            "",
            "| \u0441\u0438\u0441\u0442\u0435\u043c\u0430 | top-10 | 11\u201320 | 21\u201350 | 51\u2013100 | >100 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    buckets = dev["diagnostics"]["buckets"]
    for system in ("blind:dense", "blind:bm25"):
        counts = buckets[system]
        lines.append("| " + system.split(":")[1] + " | " + " | ".join(str(counts.get(name, 0)) for name in RANK_BUCKETS) + " |")
    lines.extend(
        [
            "",
            "### \u0414\u043e\u043f\u043e\u043b\u043d\u044f\u0435\u043c\u043e\u0441\u0442\u044c top-10",
            "",
            "| \u043a\u043b\u0430\u0441\u0441 | gold-id, \u0441\u043f\u0440\u043e\u0435\u0446\u0438\u0440\u043e\u0432\u0430\u043d\u043d\u044b\u0445 \u043d\u0430 \u0441\u0442\u0430\u0442\u044c\u044e |",
            "|---|---:|",
        ]
    )
    for name in ("only-dense", "only-bm25", "both", "neither"):
        lines.append(f"| {name} | {dev['diagnostics']['complementarity'].get(name, 0)} |")
    lines.extend(["", "### \u0427\u0430\u0441\u0442\u044b\u0435 \u043e\u0448\u0438\u0431\u043e\u0447\u043d\u044b\u0435 \u043c\u0430\u0440\u0448\u0440\u0443\u0442\u044b", "", "| \u0441\u0438\u0441\u0442\u0435\u043c\u0430 | gold | \u043f\u0435\u0440\u0432\u0430\u044f \u0447\u0443\u0436\u0430\u044f \u0441\u0442\u0430\u0442\u044c\u044f | \u0441\u043b\u0443\u0447\u0430\u0435\u0432 |", "|---|---|---|---:|"])
    for row in dev["diagnostics"]["wrong_pairs"][:12]:
        lines.append(f"| {row['system']} | {row['gold']} | {row['wrong']} | {row['count']} |")
    outcomes = dev["outcomes"]
    lines.extend(
        [
            "",
            "## \u041a\u0430\u0447\u0435\u0441\u0442\u0432\u0435\u043d\u043d\u044b\u0439 \u0440\u0430\u0437\u0431\u043e\u0440 dev-\u043e\u0448\u0438\u0431\u043e\u043a",
            "",
            "1. **\u0413\u043b\u0430\u0432\u043d\u0430\u044f \u0440\u0430\u043d\u0436\u0438\u0440\u0443\u044e\u0449\u0430\u044f \u043e\u0448\u0438\u0431\u043a\u0430 \u2014 \u043f\u043e\u0432\u0442\u043e\u0440\u044b \u043e\u0434\u043d\u043e\u0439 \u0441\u0442\u0430\u0442\u044c\u0438.** "
            f"Dense \u0438\u043c\u0435\u0435\u0442 \u043d\u0435 \u043c\u0435\u043d\u0435\u0435 \u0434\u0432\u0443\u0445 \u043b\u0438\u0448\u043d\u0438\u0445 \u043f\u043e\u0432\u0442\u043e\u0440\u043e\u0432 \u0447\u0443\u0436\u0438\u0445 \u0441\u0442\u0430\u0442\u0435\u0439 \u0432 top-10 \u0443 39 \u0438\u0437 52 \u043a\u0435\u0439\u0441\u043e\u0432. "
            f"\u0414\u0438\u0432\u0435\u0440\u0441\u0438\u0444\u0438\u043a\u0430\u0446\u0438\u044f \u0434\u0430\u043b\u0430 \u0432\u044b\u0438\u0433\u0440\u044b\u0448 \u0432 {outcomes['gains']} \u043a\u0435\u0439\u0441\u0430\u0445, \u043f\u0440\u043e\u0438\u0433\u0440\u044b\u0448 \u0432 {outcomes['losses']} \u0438 \u043d\u0435 \u0438\u0437\u043c\u0435\u043d\u0438\u043b\u0430 {outcomes['ties']}.",
            "2. **Article recall \u0438 exact recall \u043a\u043e\u043d\u0444\u043b\u0438\u043a\u0442\u0443\u044e\u0442.** \u041e\u0434\u0438\u043d \u0447\u0430\u043d\u043a \u043d\u0430 \u0441\u0442\u0430\u0442\u044c\u044e \u043e\u0441\u0432\u043e\u0431\u043e\u0436\u0434\u0430\u0435\u0442 \u043c\u0435\u0441\u0442\u0430 \u0434\u043b\u044f \u043d\u043e\u0432\u044b\u0445 \u0441\u0442\u0430\u0442\u0435\u0439, \u043d\u043e \u043c\u043e\u0436\u0435\u0442 \u0432\u044b\u043a\u0438\u043d\u0443\u0442\u044c \u043d\u0443\u0436\u043d\u0443\u044e \u0447\u0430\u0441\u0442\u044c/\u043f\u0443\u043d\u043a\u0442. \u041f\u043e\u044d\u0442\u043e\u043c\u0443 \u044d\u0442\u043e \u0445\u043e\u0440\u043e\u0448\u0438\u0439 \u043f\u0435\u0440\u0432\u044b\u0439 \u044d\u0442\u0430\u043f \u043e\u0442\u0431\u043e\u0440\u0430 \u0441\u0442\u0430\u0442\u0435\u0439, \u043d\u043e \u043d\u0435 \u0433\u043e\u0442\u043e\u0432\u0430\u044f \u0437\u0430\u043c\u0435\u043d\u0430 point-level retriever.",
            "3. **\u041e\u043f\u0438\u0441\u0430\u043d\u0438\u0435 \u043e\u0431\u044a\u0435\u043a\u0442\u0430 \u0435\u0441\u0442\u0435\u0441\u0442\u0432\u0435\u043d\u043d\u043e \u0432\u0435\u0434\u0451\u0442 \u043a \u0441\u0442. 33, \u043d\u043e gold \u0447\u0430\u0441\u0442\u043e \u0442\u0440\u0435\u0431\u0443\u0435\u0442 \u0441\u0442. 42.** "
            f"\u0423 {dev['pointer_gold_cases']} dev-\u043a\u0435\u0439\u0441\u043e\u0432 gold \u0441\u043e\u0434\u0435\u0440\u0436\u0438\u0442 \u043a\u043e\u0440\u043e\u0442\u043a\u0443\u044e \u043d\u043e\u0440\u043c\u0443-\u043e\u0442\u0441\u044b\u043b\u043a\u0443; \u0432 \u043e\u0441\u043d\u043e\u0432\u043d\u043e\u043c \u044d\u0442\u043e `art.42:ch.2:p.1` \u0441\u043e \u0441\u0441\u044b\u043b\u043a\u043e\u0439 \u043d\u0430 \u0441\u0442. 33. \u041a\u0435\u0439\u0441\u044b `fas_2026_39000566`, `fas_2026_56001987` \u0438 `fas_2026_ooz_garantm_sc120_specs_010872` \u043f\u043e\u043a\u0430\u0437\u044b\u0432\u0430\u044e\u0442, \u0447\u0442\u043e \u0438\u0437\u0432\u043b\u0435\u0447\u0451\u043d\u043d\u0430\u044f \u0441\u0442. 33 \u0441\u0435\u043c\u0430\u043d\u0442\u0438\u0447\u0435\u0441\u043a\u0438 \u0443\u043c\u0435\u0441\u0442\u043d\u0430, \u0445\u043e\u0442\u044f \u043d\u0435 \u0437\u0430\u043a\u0440\u044b\u0432\u0430\u0435\u0442 \u0431\u0443\u043a\u0432\u0430\u043b\u044c\u043d\u044b\u0439 gold.",
            "4. **\u0414\u043b\u044f \u043e\u0442\u043a\u043b\u043e\u043d\u0435\u043d\u0438\u044f \u0437\u0430\u044f\u0432\u043e\u043a \u043c\u043e\u0434\u0435\u043b\u0438 \u043f\u0443\u0442\u0430\u044e\u0442 \u043f\u0440\u043e\u0446\u0435\u0434\u0443\u0440\u043d\u044b\u0435 \u0441\u0442\u0430\u0442\u044c\u0438.** Gold 48/49 \u0443\u0445\u043e\u0434\u0438\u0442 \u043f\u043e\u0434 \u0441\u0442. 43, 52, 73\u201376: \u0437\u0430\u043f\u0440\u043e\u0441 \u043e\u043f\u0438\u0441\u044b\u0432\u0430\u0435\u0442 \u0444\u0430\u043a\u0442 \u043d\u0435\u0434\u043e\u0441\u0442\u043e\u0432\u0435\u0440\u043d\u043e\u0441\u0442\u0438, \u0430 \u043d\u0435 \u0432\u0438\u0434 \u043f\u0440\u043e\u0446\u0435\u0434\u0443\u0440\u044b. \u0425\u0430\u0440\u0430\u043a\u0442\u0435\u0440\u043d\u044b `fas_2026_04002773`, `fas_2026_89016860`, `fas_2026_98006537`.",
            "5. **\u0421\u043b\u0438\u0448\u043a\u043e\u043c \u043a\u043e\u0440\u043e\u0442\u043a\u0438\u0435 \u0438\u043b\u0438 \u0444\u0430\u043a\u0442\u043e\u043b\u043e\u0433\u0438\u0447\u0435\u0441\u043a\u0438\u0435 \u0434\u043e\u0432\u043e\u0434\u044b \u043d\u0435 \u043d\u0435\u0441\u0443\u0442 \u043f\u0440\u0430\u0432\u043e\u0432\u043e\u0433\u043e \u0441\u0438\u0433\u043d\u0430\u043b\u0430.** `fas_2026_25001220` \u0441\u043e\u0434\u0435\u0440\u0436\u0438\u0442 \u043f\u043e\u0447\u0442\u0438 \u0442\u043e\u043b\u044c\u043a\u043e \u0445\u0430\u0440\u0430\u043a\u0442\u0435\u0440\u0438\u0441\u0442\u0438\u043a\u0443 \u0448\u043e\u0432\u043d\u043e\u0433\u043e \u043c\u0430\u0442\u0435\u0440\u0438\u0430\u043b\u0430 \u0438 \u0441\u0442\u0430\u0432\u0438\u0442 gold \u043d\u0438\u0436\u0435 90-\u0433\u043e \u043c\u0435\u0441\u0442\u0430 \u0443 BM25 \u0438 200-\u0433\u043e \u0443 dense. \u0417\u0434\u0435\u0441\u044c \u043d\u0443\u0436\u0435\u043d `issue_type`/\u043a\u0440\u0430\u0442\u043a\u0438\u0439 \u044e\u0440\u0438\u0434\u0438\u0447\u0435\u0441\u043a\u0438\u0439 paraphrase, \u0430 \u043d\u0435 \u0434\u0440\u0443\u0433\u043e\u0439 \u044d\u043c\u0431\u0435\u0434\u0434\u0435\u0440.",
            "6. **\u042f\u0432\u043d\u044b\u0435 \u0441\u0441\u044b\u043b\u043a\u0438 \u043d\u0435 \u043e\u0431\u044a\u044f\u0441\u043d\u044f\u044e\u0442 \u043e\u0441\u043d\u043e\u0432\u043d\u0443\u044e \u0447\u0430\u0441\u0442\u044c \u043a\u0430\u0447\u0435\u0441\u0442\u0432\u0430.** Raw \u0438 blind \u0440\u0430\u0437\u043b\u0438\u0447\u0430\u044e\u0442\u0441\u044f \u043f\u043e \u043f\u043e\u043a\u0440\u044b\u0442\u0438\u044e top-10 \u043b\u0438\u0448\u044c \u0432 3 \u0438\u0437 52 dev-\u043a\u0435\u0439\u0441\u043e\u0432; \u0443 32 \u0442\u0435\u043a\u0441\u0442 raw \u0438 blind \u0432\u043e\u043e\u0431\u0449\u0435 \u043e\u0434\u0438\u043d\u0430\u043a\u043e\u0432.",
            "7. **\u0417\u0430\u0433\u043e\u043b\u043e\u0432\u043e\u043a \u0441\u0442\u0430\u0442\u044c\u0438 \u0438 RRF \u043d\u0435 \u0438\u0441\u043f\u0440\u0430\u0432\u0438\u043b\u0438 \u043f\u0440\u043e\u0431\u043b\u0435\u043c\u0443.** \u041f\u0440\u0435\u0444\u0438\u043a\u0441 \u0437\u0430\u0433\u043e\u043b\u043e\u0432\u043a\u0430 \u0441\u043d\u0438\u0437\u0438\u043b dense 0.529\u21920.490 \u0438 BM25 0.442\u21920.385. \u041b\u0443\u0447\u0448\u0438\u0439 weighted RRF \u0434\u043e\u0448\u0451\u043b \u0442\u043e\u043b\u044c\u043a\u043e \u0434\u043e 0.452: \u043e\u043d \u043f\u043e-\u043f\u0440\u0435\u0436\u043d\u0435\u043c\u0443 \u0432\u044b\u0442\u0435\u0441\u043d\u044f\u0435\u0442 \u0443\u043d\u0438\u043a\u0430\u043b\u044c\u043d\u044b\u0435 \u043f\u043e\u043f\u0430\u0434\u0430\u043d\u0438\u044f \u043a\u0430\u0436\u0434\u043e\u0439 \u0441\u0438\u0441\u0442\u0435\u043c\u044b.",
            "",
            "## \u041e\u0431\u0437\u043e\u0440 dense-\u043c\u043e\u0434\u0435\u043b\u0435\u0439 \u043f\u043e dev / blind \u2014 baseline \u0431\u0435\u0437 \u0434\u0438\u0432\u0435\u0440\u0441\u0438\u0444\u0438\u043a\u0430\u0446\u0438\u0438",
            "",
            "| \u043c\u043e\u0434\u0435\u043b\u044c | \u043b\u0443\u0447\u0448\u0438\u0439 max_chunk | article R@10 |",
            "|---|---:|---:|",
        ]
    )
    for row in state["model_overview"]:
        lines.append(f"| {row['model']} | {row['chunk']} | {row['article_r10']:.3f} |")
    if "cross_model_diversification" in state:
        screen = state["cross_model_diversification"]
        global_winner = screen["dev"]["winner"]
        global_interval = screen["dev"]["bootstrap"]
        lines.extend(
            [
                "",
                "## \u0414\u0438\u0432\u0435\u0440\u0441\u0438\u0444\u0438\u043a\u0430\u0446\u0438\u044f \u0432\u0441\u0435\u0445 dense-\u043a\u043e\u043d\u0444\u0438\u0433\u0443\u0440\u0430\u0446\u0438\u0439",
                "",
                "\u041e\u0434\u0438\u043d\u0430\u043a\u043e\u0432\u0430\u044f article-diversification (\u043f\u0443\u043b 50, \u043b\u0438\u043c\u0438\u0442 10) \u043f\u0440\u043e\u0432\u0435\u0440\u0435\u043d\u0430 \u043d\u0430 \u0432\u0441\u0435\u0445 34 \u0437\u0430\u043a\u044d\u0448\u0438\u0440\u043e\u0432\u0430\u043d\u043d\u044b\u0445 \u0441\u043e\u0447\u0435\u0442\u0430\u043d\u0438\u044f\u0445 \u0448\u0435\u0441\u0442\u0438 \u044d\u043c\u0431\u0435\u0434\u0434\u0435\u0440\u043e\u0432 \u0438 max_chunk.",
                f"\u0413\u043b\u043e\u0431\u0430\u043b\u044c\u043d\u044b\u0439 dev-\u043f\u043e\u0431\u0435\u0434\u0438\u0442\u0435\u043b\u044c \u043d\u0435 \u0438\u0437\u043c\u0435\u043d\u0438\u043b\u0441\u044f: **{global_winner['model']}**, "
                f"`max_chunk={global_winner['max_chunk']}`, article R@10 "
                f"{global_winner['baseline']['recall_article']:.3f}\u2192{global_winner['diversified']['recall_article']:.3f}; "
                f"exact R@10 {global_winner['baseline']['recall_exact']:.3f}\u2192{global_winner['diversified']['recall_exact']:.3f}.",
                f"\u041f\u0430\u0440\u043d\u0430\u044f \u0440\u0430\u0437\u043d\u0438\u0446\u0430 article R@10: {global_interval['mean_delta']:+.3f} "
                f"(95% group-bootstrap [{global_interval['low']:+.3f}; {global_interval['high']:+.3f}]). "
                f"\u041f\u043e\u043b\u043d\u0430\u044f \u0442\u0430\u0431\u043b\u0438\u0446\u0430: `{screen['report']}`.",
            ]
        )
        if "eval" in screen:
            global_eval = screen["eval"]
            global_eval_interval = global_eval["bootstrap"]
            lines.append(
                f"\u041d\u0430 eval \u043f\u0440\u043e\u0432\u0435\u0440\u0435\u043d\u044b \u0442\u043e\u043b\u044c\u043a\u043e \u044d\u0442\u043e\u0442 baseline \u0438 \u0435\u0433\u043e diverse-\u0432\u0430\u0440\u0438\u0430\u043d\u0442: "
                f"{global_eval['baseline']['recall_article']:.3f}\u2192{global_eval['diversified']['recall_article']:.3f}, "
                f"95% CI [{global_eval_interval['low']:+.3f}; {global_eval_interval['high']:+.3f}]."
            )
    if "bm25_diversification" in state:
        screen = state["bm25_diversification"]
        bm25_winner = screen["dev"]["winner"]
        bm25_interval = screen["dev"]["bootstrap"]
        highest_article = max(
            screen["dev"]["candidates"],
            key=lambda row: (row["diversified"]["recall_article"], -row["max_chunk"]),
        )
        lines.extend(
            [
                "",
                "## \u0414\u0438\u0432\u0435\u0440\u0441\u0438\u0444\u0438\u043a\u0430\u0446\u0438\u044f BM25 \u043f\u043e \u0432\u0441\u0435\u043c max_chunk",
                "",
                "\u041e\u0434\u0438\u043d\u0430\u043a\u043e\u0432\u0430\u044f article-diversification (\u043f\u0443\u043b 50, \u043b\u0438\u043c\u0438\u0442 10) \u043f\u0440\u043e\u0432\u0435\u0440\u0435\u043d\u0430 \u0434\u043b\u044f max_chunk 300, 500, 700, 1000, 1500, 2000 \u0438 2500.",
                f"\u041c\u0430\u043a\u0441\u0438\u043c\u0430\u043b\u044c\u043d\u044b\u0439 article R@10 \u0434\u0430\u043b m{highest_article['max_chunk']}: "
                f"{highest_article['diversified']['recall_article']:.3f}. \u041f\u043e \u0437\u0430\u0440\u0430\u043d\u0435\u0435 \u0437\u0430\u0434\u0430\u043d\u043d\u044b\u043c tie-breakers \u0437\u0430\u0444\u0438\u043a\u0441\u0438\u0440\u043e\u0432\u0430\u043d **BM25, "
                f"`max_chunk={bm25_winner['max_chunk']}`**: article R@10 "
                f"{bm25_winner['baseline']['recall_article']:.3f}\u2192{bm25_winner['diversified']['recall_article']:.3f}, "
                f"exact R@10 {bm25_winner['baseline']['recall_exact']:.3f}\u2192{bm25_winner['diversified']['recall_exact']:.3f}.",
                f"Dev-\u0440\u0430\u0437\u043d\u0438\u0446\u0430 article R@10: {bm25_interval['mean_delta']:+.3f} "
                f"(95% group-bootstrap [{bm25_interval['low']:+.3f}; {bm25_interval['high']:+.3f}]). "
                f"\u041f\u043e\u043b\u043d\u0430\u044f \u0442\u0430\u0431\u043b\u0438\u0446\u0430: `{screen['report']}`.",
            ]
        )
        if "eval" in screen:
            bm25_eval = screen["eval"]
            bm25_eval_interval = bm25_eval["bootstrap"]
            lines.append(
                f"\u041d\u0430 eval BM25 m{bm25_winner['max_chunk']} \u0432\u044b\u0440\u043e\u0441 "
                f"{bm25_eval['baseline']['recall_article']:.3f}\u2192{bm25_eval['diversified']['recall_article']:.3f}; "
                f"95% CI [{bm25_eval_interval['low']:+.3f}; {bm25_eval_interval['high']:+.3f}] \u043a\u0430\u0441\u0430\u0435\u0442\u0441\u044f \u043d\u0443\u043b\u044f, "
                "\u043f\u043e\u044d\u0442\u043e\u043c\u0443 eval-\u0432\u044b\u0438\u0433\u0440\u044b\u0448 \u043f\u043e\u043a\u0430 \u043d\u0435\u043b\u044c\u0437\u044f \u0441\u0447\u0438\u0442\u0430\u0442\u044c \u0443\u0441\u0442\u043e\u0439\u0447\u0438\u0432\u043e \u043e\u0442\u043b\u0438\u0447\u043d\u044b\u043c \u043e\u0442 \u043d\u0443\u043b\u044f."
            )
        for sensitivity in screen.get("eval_sensitivity", []):
            sensitivity_interval = sensitivity["bootstrap"]
            lines.append(
                f"Post-hoc sensitivity BM25 m{sensitivity['max_chunk']} \u043d\u0430 eval: article R@10 "
                f"{sensitivity['baseline']['recall_article']:.3f}\u2192{sensitivity['diversified']['recall_article']:.3f}, "
                f"exact R@10 {sensitivity['baseline']['recall_exact']:.3f}\u2192{sensitivity['diversified']['recall_exact']:.3f}, "
                f"95% CI \u0434\u043b\u044f \u0440\u0430\u0437\u043d\u0438\u0446\u044b article "
                f"[{sensitivity_interval['low']:+.3f}; {sensitivity_interval['high']:+.3f}]."
            )
    lines.extend(
        [
            "",
            "## Архитектура article-level гибрида",
            "",
            "**\u0414\u043b\u044f dense \u043d\u0443\u0436\u043d\u043e \u0437\u0430\u0444\u0438\u043a\u0441\u0438\u0440\u043e\u0432\u0430\u0442\u044c `Roflmax/bge-m3-russian-legal`, `max_chunk=700`.** "
            "\u042d\u0442\u043e \u043b\u0443\u0447\u0448\u0430\u044f dense-\u043a\u043e\u043d\u0444\u0438\u0433\u0443\u0440\u0430\u0446\u0438\u044f \u043f\u043e\u0441\u043b\u0435 \u043e\u0434\u0438\u043d\u0430\u043a\u043e\u0432\u043e\u0439 \u0434\u0438\u0432\u0435\u0440\u0441\u0438\u0444\u0438\u043a\u0430\u0446\u0438\u0438 \u0432\u0441\u0435\u0445 \u044d\u043c\u0431\u0435\u0434\u0434\u0435\u0440\u043e\u0432 \u0438 \u0440\u0430\u0437\u043c\u0435\u0440\u043e\u0432.",
            "",
            "**В эксперимент гибрида включены BM25 m1500 и m2500.** "
            "m1500 \u0434\u0430\u043b \u043c\u0430\u043a\u0441\u0438\u043c\u0430\u043b\u044c\u043d\u044b\u0439 dev article R@10 (0.567), \u0430 m2500 \u0431\u044b\u043b \u0437\u0430\u0444\u0438\u043a\u0441\u0438\u0440\u043e\u0432\u0430\u043d \u043f\u043e \u043e\u0431\u0449\u0438\u043c tie-breakers \u0437\u0430 \u0431\u043e\u043b\u0435\u0435 \u0432\u044b\u0441\u043e\u043a\u0438\u0439 exact R@10. "
            "\u0415\u0434\u0438\u043d\u043e\u043b\u0438\u0447\u043d\u044b\u0439 \u043f\u043e\u0431\u0435\u0434\u0438\u0442\u0435\u043b\u044c \u043d\u0435 \u043e\u0431\u044f\u0437\u0430\u043d \u0431\u044b\u0442\u044c \u043b\u0443\u0447\u0448\u0438\u043c \u043a\u043e\u043c\u043f\u043e\u043d\u0435\u043d\u0442\u043e\u043c \u0433\u0438\u0431\u0440\u0438\u0434\u0430.",
            "",
            "\u0413\u0438\u0431\u0440\u0438\u0434 \u043d\u0435 \u0434\u043e\u043b\u0436\u0435\u043d \u0442\u0440\u0435\u0431\u043e\u0432\u0430\u0442\u044c \u043e\u0434\u0438\u043d\u0430\u043a\u043e\u0432\u043e\u0433\u043e \u0447\u0430\u043d\u043a\u0438\u043d\u0433\u0430. \u0427\u0430\u043d\u043a\u0438 m700, m1500 \u0438 m2500 \u043d\u0435 \u0438\u043c\u0435\u044e\u0442 \u043e\u0431\u0449\u0438\u0445 `chunk_id`, \u043f\u043e\u044d\u0442\u043e\u043c\u0443 \u043f\u0440\u044f\u043c\u043e\u0439 chunk-level RRF \u0438\u043b\u0438 interleave \u043d\u0435\u043a\u043e\u0440\u0440\u0435\u043a\u0442\u043d\u044b. "
            "\u041e\u0431\u0449\u0435\u0439 \u0435\u0434\u0438\u043d\u0438\u0446\u0435\u0439 fusion \u0434\u043e\u043b\u0436\u043d\u0430 \u0431\u044b\u0442\u044c \u0441\u0442\u0430\u0442\u044c\u044f.",
            "",
            "Проверенная схема:",
            "",
            "1. Dense m700 \u0438 BM25 m1500/m2500 \u043d\u0435\u0437\u0430\u0432\u0438\u0441\u0438\u043c\u043e \u0432\u043e\u0437\u0432\u0440\u0430\u0449\u0430\u044e\u0442 top-50 \u0447\u0430\u043d\u043a\u043e\u0432.",
            "2. \u041a\u0430\u0436\u0434\u044b\u0439 \u0441\u043f\u0438\u0441\u043e\u043a \u043f\u0440\u043e\u0435\u0446\u0438\u0440\u0443\u0435\u0442\u0441\u044f \u0432 \u0440\u0430\u043d\u0433 \u0441\u0442\u0430\u0442\u0435\u0439 \u043f\u043e \u043b\u0443\u0447\u0448\u0435\u043c\u0443 \u0447\u0430\u043d\u043a\u0443 \u0441\u0442\u0430\u0442\u044c\u0438. \u041e\u0434\u043d\u0430 \u0441\u0438\u0441\u0442\u0435\u043c\u0430 \u0434\u0430\u0451\u0442 \u043d\u0435 \u0431\u043e\u043b\u0435\u0435 \u043e\u0434\u043d\u043e\u0433\u043e \u0432\u043a\u043b\u0430\u0434\u0430 \u043d\u0430 \u0441\u0442\u0430\u0442\u044c\u044e, \u0447\u0442\u043e\u0431\u044b m700 \u043d\u0435 \u043f\u043e\u043b\u0443\u0447\u0430\u043b \u043f\u0440\u0435\u0438\u043c\u0443\u0449\u0435\u0441\u0442\u0432\u043e \u0437\u0430 \u0441\u0447\u0451\u0442 \u0431\u043e\u043b\u044c\u0448\u0435\u0433\u043e \u0447\u0438\u0441\u043b\u0430 \u0447\u0430\u043d\u043a\u043e\u0432.",
            "3. Ранги статей объединяются weighted RRF. На dev проверены BM25 m1500/m2500 и веса dense:BM25 `1:1`, `2:1`, `1:2` с фиксированным пулом 50.",
            "4. \u041f\u043e\u0441\u043b\u0435 \u0432\u044b\u0431\u043e\u0440\u0430 top-\u0441\u0442\u0430\u0442\u0435\u0439 \u0444\u0438\u043d\u0430\u043b\u044c\u043d\u044b\u0435 \u0444\u0440\u0430\u0433\u043c\u0435\u043d\u0442\u044b \u0431\u0435\u0440\u0443\u0442\u0441\u044f \u0438\u0437 m700-\u043a\u043e\u0440\u043f\u0443\u0441\u0430: \u0447\u0430\u0441\u0442\u0438 \u0438 \u043f\u0443\u043d\u043a\u0442\u044b \u043f\u043e\u0432\u0442\u043e\u0440\u043d\u043e \u0440\u0430\u043d\u0436\u0438\u0440\u0443\u044e\u0442\u0441\u044f \u0441 \u0443\u0447\u0451\u0442\u043e\u043c article-score. BM25-\u0447\u0430\u043d\u043a m1500/m2500 \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0435\u0442\u0441\u044f \u043a\u0430\u043a \u0441\u0438\u0433\u043d\u0430\u043b \u0441\u0442\u0430\u0442\u044c\u0438, \u0430 \u043d\u0435 \u043a\u0430\u043a \u043e\u0431\u044f\u0437\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u0439 \u0444\u0438\u043d\u0430\u043b\u044c\u043d\u044b\u0439 \u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442.",
            "5. Выбор сделан по dev/blind article R@10, затем exact R@10, all-hit article@10 и latency; на eval проверены baseline и зафиксированный гибрид.",
            "",
            "\u0422\u0430\u043a\u0430\u044f \u0441\u0445\u0435\u043c\u0430 \u0440\u0430\u0437\u0434\u0435\u043b\u044f\u0435\u0442 \u0434\u0432\u0435 \u0437\u0430\u0434\u0430\u0447\u0438: BM25 \u0434\u0430\u0451\u0442 \u0448\u0438\u0440\u043e\u043a\u043e\u0435 \u043b\u0435\u043a\u0441\u0438\u0447\u0435\u0441\u043a\u043e\u0435 \u043f\u043e\u043a\u0440\u044b\u0442\u0438\u0435 \u0441\u0442\u0430\u0442\u0435\u0439, \u0430 dense m700 \u2014 \u0431\u043e\u043b\u0435\u0435 \u0442\u043e\u0447\u043d\u044b\u0435 \u0444\u0440\u0430\u0433\u043c\u0435\u043d\u0442\u044b \u0434\u043b\u044f \u0444\u0438\u043d\u0430\u043b\u044c\u043d\u043e\u0433\u043e \u043e\u0442\u0432\u0435\u0442\u0430.",
        ]
    )
    if "article_hybrid" in state:
        hybrid = state["article_hybrid"]
        hybrid_dev = hybrid["dev"]
        hybrid_winner = hybrid_dev["winner"]
        hybrid_spec = hybrid_winner["spec"]
        hybrid_interval = hybrid_dev["bootstrap_vs_dense"]
        dense_baseline = hybrid_dev["baselines"]["dense m700 article-diverse"]["at10"]
        lines.extend(
            [
                "",
                "## Результат article-level гибрида",
                "",
                f"На dev / blind выбран BM25 m{hybrid_spec['bm25_max_chunk']}, `k={hybrid_spec['k']}`, "
                f"веса dense:BM25 `{int(hybrid_spec['dense_weight'])}:{int(hybrid_spec['bm25_weight'])}`, "
                f"{hybrid_spec['n_articles']} статей: article R@10 {dense_baseline['recall_article']:.3f}→"
                f"{hybrid_winner['at10']['recall_article']:.3f}, exact R@10 {dense_baseline['recall_exact']:.3f}→"
                f"{hybrid_winner['at10']['recall_exact']:.3f}. Разница article "
                f"{hybrid_interval['mean_delta']:+.3f} (95% CI [{hybrid_interval['low']:+.3f}; "
                f"{hybrid_interval['high']:+.3f}]).",
                f"Полный протокол и сетка: `{hybrid['report']}`.",
            ]
        )
        if "eval" in hybrid:
            hybrid_eval = hybrid["eval"]
            eval_dense = hybrid_eval["methods"]["dense m700 article-diverse"]["at10"]
            eval_hybrid = hybrid_eval["methods"]["locked article-level hybrid"]["at10"]
            eval_interval = hybrid_eval["bootstrap_vs_dense"]
            lines.extend(
                [
                    "",
                    f"На eval / blind выигрыш не перенёсся: article R@10 "
                    f"{eval_dense['recall_article']:.3f}→{eval_hybrid['recall_article']:.3f}, exact R@10 "
                    f"{eval_dense['recall_exact']:.3f}→{eval_hybrid['recall_exact']:.3f}; разница article "
                    f"{eval_interval['mean_delta']:+.3f} (95% CI [{eval_interval['low']:+.3f}; "
                    f"{eval_interval['high']:+.3f}]).",
                    "**Рекомендация:** оставить dense m700 article-diverse основным retriever. "
                    "Текущий article-level RRF считать отклонённой гипотезой, а BM25 использовать как отдельный резервный канал или позже обучать fusion на большем dev-наборе.",
                ]
            )
    if "conservative_hybrid" in state and "cross_encoder" in state:
        conservative = state["conservative_hybrid"]
        cross_encoder = state["cross_encoder"]
        dense = cross_encoder["dev"]["baseline"]["at10"]
        conservative_dev = conservative["dev"]["winner"]["at10"]
        cross_dev = cross_encoder["dev"]["winner"]["at10"]
        interval = cross_encoder["dev"]["bootstrap_vs_dense"]
        lines.extend(
            [
                "",
                "## Консервативный гибрид и cross-encoder",
                "",
                "Консервативная схема сохраняет первые 8 dense-статей, а BM25 m1500 "
                "заполняет оставшиеся позиции. Внутри выбранных статей сравниваются chunk-level "
                "RRF и `BAAI/bge-reranker-v2-m3` по m700-чанкам.",
                f"На dev / blind dense → conservative → cross-encoder: article R@10 "
                f"{dense['recall_article']:.3f}→{conservative_dev['recall_article']:.3f}→"
                f"{cross_dev['recall_article']:.3f}; exact R@10 "
                f"{dense['recall_exact']:.3f}→{conservative_dev['recall_exact']:.3f}→"
                f"{cross_dev['recall_exact']:.3f}. Article-разница к dense "
                f"{interval['mean_delta']:+.3f} (95% CI [{interval['low']:+.3f}; "
                f"{interval['high']:+.3f}]).",
                "Article-выигрыш целиком даёт выбор статей; cross-encoder "
                "улучшает только exact-выбор чанка внутри статьи. Полные отчёты: "
                f"`{conservative['report']}` и `{cross_encoder['report']}`.",
            ]
        )
        if "eval_posthoc" in cross_encoder:
            eval_dense = cross_encoder["eval_posthoc"]["methods"][
                "dense m700 article-diverse"
            ]["at10"]
            eval_conservative = cross_encoder["eval_posthoc"]["methods"][
                "locked conservative hybrid"
            ]["at10"]
            eval_cross = cross_encoder["eval_posthoc"]["methods"][
                "locked cross-encoder rerank"
            ]["at10"]
            eval_interval = cross_encoder["eval_posthoc"]["bootstrap_vs_dense"]
            lines.extend(
                [
                    "",
                    "Post-hoc eval не подтвердил перенос: dense → conservative → cross-encoder "
                    f"по article R@10 {eval_dense['recall_article']:.3f}→"
                    f"{eval_conservative['recall_article']:.3f}→{eval_cross['recall_article']:.3f}, "
                    f"по exact R@10 {eval_dense['recall_exact']:.3f}→"
                    f"{eval_conservative['recall_exact']:.3f}→{eval_cross['recall_exact']:.3f}; "
                    f"article-разница {eval_interval['mean_delta']:+.3f} "
                    f"(95% CI [{eval_interval['low']:+.3f}; {eval_interval['high']:+.3f}]).",
                    "**Рекомендация:** оставить dense m700 article-diverse production baseline. "
                    "Консервативный hybrid считать перспективной, но пока не подтверждённой гипотезой; "
                    "текущий cross-encoder не внедрять: post-hoc выигрыша нет, а CPU-latency около 22 секунд на запрос.",
                ]
            )
    if "wide_article_rerank" in state:
        wide = state["wide_article_rerank"]
        wide_dev = wide["dev"]
        wide_winner = wide_dev["winner"]
        wide_spec = wide_winner["spec"]
        wide_interval = wide_dev["bootstrap_vs_dense"]
        wide_dense = wide_dev["baseline"]["at10"]
        lines.extend(
            [
                "",
                "## Wide article-first cross-encoder",
                "",
                f"Dense m700 и BM25 m1500 сначала возвращают по 50 уникальных статей. "
                f"На dev выбраны квоты dense={wide_spec['dense_articles']} и "
                f"BM25={wide_spec['bm25_articles']}, затем `{wide['model']}` ранжирует до "
                f"{wide_spec['chunks_per_article']} m700-чанков на статью и оставляет один лучший.",
                f"На dev / blind article R@10 {wide_dense['recall_article']:.3f}→"
                f"{wide_winner['at10']['recall_article']:.3f}, exact R@10 "
                f"{wide_dense['recall_exact']:.3f}→{wide_winner['at10']['recall_exact']:.3f}; "
                f"Δ article {wide_interval['mean_delta']:+.3f} (95% CI "
                f"[{wide_interval['low']:+.3f}; {wide_interval['high']:+.3f}]).",
            ]
        )
        if "eval" in wide:
            wide_eval = wide["eval"]
            eval_dense = wide_eval["methods"]["dense m700 article-diverse"]["at10"]
            eval_wide = wide_eval["methods"]["locked wide cross-encoder"]["at10"]
            eval_interval = wide_eval["bootstrap_vs_dense"]
            lines.extend(
                [
                    f"На eval / blind article R@10 {eval_dense['recall_article']:.3f}→"
                    f"{eval_wide['recall_article']:.3f}, exact R@10 "
                    f"{eval_dense['recall_exact']:.3f}→{eval_wide['recall_exact']:.3f}; "
                    f"Δ article {eval_interval['mean_delta']:+.3f} (95% CI "
                    f"[{eval_interval['low']:+.3f}; {eval_interval['high']:+.3f}]).",
                    "**Рекомендация:** не заменять dense article-diverse этим вариантом: "
                    "article-выигрыш dev не перенёсся на eval. Схема полезна как evidence-reranker "
                    f"из-за роста exact и all-hit на eval, но CPU-latency около "
                    f"{wide_eval['cross_encoder_latency_ms'] / 1000:.1f} s/query слишком высока для "
                    "production без более узкого candidate generator или дистилляции.",
                    f"Полный отчёт: `{wide['report']}`.",
                ]
            )
    if "article_chunk_budget" in state:
        budget = state["article_chunk_budget"]
        surface = budget["surface_winner"]
        adaptive = budget["two_pass_winner"]
        surface_d = surface["diagnostics"]
        adaptive_d = adaptive["diagnostics"]
        adaptive_interval = budget["two_pass_vs_surface"]["interval"]
        lines.extend(
            [
                "",
                "## Совместный бюджет статей и чанков",
                "",
                f"На {budget['n']} dev/blind-кейсах проверены "
                f"{budget['counts']['surface']} одноэтапных и "
                f"{budget['counts']['two_pass']} двухпроходных конфигураций; "
                f"{budget['counts']['genuinely_adaptive']} из последних действительно сокращают "
                "shortlist минимум на пять статей в среднем.",
                f"Лучшая общая поверхность сохраняет dense="
                f"{surface['spec']['dense_articles']}, BM25="
                f"{surface['spec']['bm25_articles']}, "
                f"chunks/article={surface['spec']['chunks_per_article']}: article R@10 "
                f"{surface['at10']['recall_article']:.3f}, exact R@10 "
                f"{surface['at10']['recall_exact']:.3f}, mean/max CE-пар "
                f"{surface['mean_scored_chunks']:.1f}/{surface['max_scored_chunks']}.",
                f"Лучший настоящий two-pass берёт dense="
                f"{adaptive['spec']['dense_articles']} и BM25="
                f"{adaptive['spec']['bm25_articles']}, сначала "
                f"{adaptive['spec']['first_pass_chunks']} чанка на статью, затем "
                f"{adaptive['spec']['shortlist_articles']} статей × "
                f"{adaptive['spec']['final_chunks_per_article']} чанков. Candidate article recall "
                f"{adaptive_d['candidate_article_recall']:.3f} падает до "
                f"{adaptive_d['shortlist_article_recall']:.3f} уже при shortlist; итоговые "
                f"article/exact R@10 равны {adaptive['at10']['recall_article']:.3f}/"
                f"{adaptive['at10']['recall_exact']:.3f} при "
                f"{adaptive['mean_scored_chunks']:.1f} CE-пары в среднем.",
                f"Разница two-pass к лучшей общей поверхности по article R@10 "
                f"{adaptive_interval['mean_delta']:+.3f} (95% group-bootstrap "
                f"[{adaptive_interval['low']:+.3f}; {adaptive_interval['high']:+.3f}]).",
                f"Диагностика exact показывает отдельное узкое место: даже у победителя поверхности "
                f"в пуле достижим exact recall {surface_d['pool_exact_recall']:.3f}, но в финальном "
                f"top-10 остаётся {surface_d['final_exact_recall']:.3f}. Поэтому следующий шаг — "
                "улучшать выбор shortlist и evidence-чанка, а не просто расширять число статей.",
                f"Полный отчёт: `{budget['report']}`.",
            ]
        )
    if "article_shortlist_fusion" in state:
        fusion = state["article_shortlist_fusion"]
        winner = fusion["winner"]
        diagnostics = winner["diagnostics"]
        ce_only = fusion["winners_by_strategy"]["ce_only"]
        ce_interval = fusion["winner_vs_ce_only"]["interval"]
        surface = fusion["joint_surface_baseline"]
        surface_interval = fusion["winner_vs_joint_surface"]["interval"]
        lines.extend(
            [
                "",
                "## Retrieval-aware shortlist с cross-encoder",
                "",
                f"На dev/blind проверено {fusion['count']} конфигураций: CE-only, "
                "трёхсторонний RRF dense+BM25+CE, защищённые квоты, "
                "CE с retrieval-бонусом и dense+BM25 RRF с CE только для выбора чанка.",
                f"Победитель: dense={winner['spec']['dense_articles']}, "
                f"BM25={winner['spec']['bm25_articles']}, first-pass="
                f"{winner['spec']['first_pass_chunks']}, shortlist="
                f"{winner['spec']['shortlist_articles']}, RRF k={winner['spec']['k']}, "
                f"веса dense:BM25:CE={winner['spec']['dense_weight']}:"
                f"{winner['spec']['bm25_weight']}:{winner['spec']['ce_weight']}. Candidate article recall "
                f"{diagnostics['candidate_article_recall']:.3f}→"
                f"{diagnostics['shortlist_article_recall']:.3f}; финальные article/exact R@10 "
                f"{winner['at10']['recall_article']:.3f}/{winner['at10']['recall_exact']:.3f}.",
                f"Против CE-only article R@10 {ce_only['at10']['recall_article']:.3f}→"
                f"{winner['at10']['recall_article']:.3f}, Δ {ce_interval['mean_delta']:+.3f} "
                f"(95% CI [{ce_interval['low']:+.3f}; {ce_interval['high']:+.3f}]). "
                f"Против прежней поверхности article R@10 "
                f"{surface['at10']['recall_article']:.3f}→{winner['at10']['recall_article']:.3f}, "
                f"Δ {surface_interval['mean_delta']:+.3f} (95% CI "
                f"[{surface_interval['low']:+.3f}; {surface_interval['high']:+.3f}]); exact R@10 "
                f"{surface['at10']['recall_exact']:.3f}→{winner['at10']['recall_exact']:.3f}.",
                "Вывод: retrieval-ранги действительно исправляют ошибки узкого CE-pass: "
                "потеря candidate→shortlist сокращается с 0.096 до 0.038. Оставшаяся "
                "проблема — выбор exact evidence внутри статьи.",
                f"Полный отчёт: `{fusion['report']}`.",
            ]
        )
        if "bf16_comparison" in fusion:
            precision = fusion["bf16_comparison"]
            bf16 = precision["winner"]
            lines.extend(
                [
                    f"Контрольный пересчёт зафиксированной архитектуры в FP32 сохранил победителя "
                    f"и метрики: BF16→FP32 article R@10 "
                    f"{bf16['at10']['recall_article']:.3f}→{winner['at10']['recall_article']:.3f}, "
                    f"exact R@10 {bf16['at10']['recall_exact']:.3f}→"
                    f"{winner['at10']['recall_exact']:.3f}. Полностью одинаковый порядок top-10 "
                    f"чанков получен в {precision['rankings_identical']}/{precision['n']} кейсах, "
                    f"но различия в остальных кейсах не изменили recall.",
                    f"FP32 оказался медленнее: throughput "
                    f"{precision['bf16_pairs_per_second']:.2f}→"
                    f"{precision['fp32_pairs_per_second']:.2f} пар/с, расчётная CE latency "
                    f"{precision['bf16_estimated_ms_per_query'] / 1000:.1f}→"
                    f"{precision['fp32_estimated_ms_per_query'] / 1000:.1f} s/query.",
                ]
            )
        if "eval" in fusion:
            evaluated = fusion["eval"]
            eval_dense = evaluated["dense_baseline"]
            eval_winner = evaluated["winner"]["at10"]
            eval_interval = evaluated["bootstrap_vs_dense"]
            lines.extend(
                [
                    f"Однократный eval/blind: dense article-diverse → locked RRF "
                    f"по article R@10 {eval_dense['recall_article']:.3f}→"
                    f"{eval_winner['recall_article']:.3f}, по exact R@10 "
                    f"{eval_dense['recall_exact']:.3f}→{eval_winner['recall_exact']:.3f}; "
                    f"Δ article {eval_interval['mean_delta']:+.3f} (95% CI "
                    f"[{eval_interval['low']:+.3f}; {eval_interval['high']:+.3f}])."
                ]
            )
        elif "eval_reporting_snapshot" in fusion:
            evaluated = fusion["eval_reporting_snapshot"]
            dense = evaluated["dense_baseline"]
            winner_eval = evaluated["winner"]
            interval = evaluated["bootstrap_vs_dense"]
            lines.append(
                "Reporting-only BF16 eval snapshot (не перезапускался): "
                f"dense→RRF article R@10 {dense['recall_article']:.3f}→"
                f"{winner_eval['recall_article']:.3f}, exact R@10 "
                f"{dense['recall_exact']:.3f}→{winner_eval['recall_exact']:.3f}; "
                f"Δ article {interval['mean_delta']:+.3f} (95% CI "
                f"[{interval['low']:+.3f}; {interval['high']:+.3f}])."
            )
    if "evidence_selection" in state:
        evidence = state["evidence_selection"]
        baseline = evidence["baseline"]
        top10 = evidence["best_top10"]
        exact_top10 = evidence["best_exact_top10"]
        pack20 = evidence["best_pack20"]
        pack30 = evidence["best_pack30"]
        top10_interval = evidence["top10_vs_baseline_exact"]
        exact_top10_interval = evidence["exact_top10_vs_baseline_exact"]
        pack20_interval = evidence["pack20_vs_baseline_exact"]
        pack30_interval = evidence["pack30_vs_baseline_exact"]
        lines.extend(
            [
                "",
                "## Evidence selection внутри зафиксированных статей",
                "",
                f"На {evidence['n']} dev/blind-кейсах проверено {evidence['count']} конфигураций: "
                "interleave, chunk-level RRF, diversity-aware RRF, retrieval-бонусы "
                "и evidence packs с лимитом на статью.",
                f"При жёстком top-10 расширение пула с 8 до 16 чанков на статью "
                f"подняло exact R@10 {baseline['metrics']['recall_exact']:.3f}→"
                f"{top10['metrics']['recall_exact']:.3f}; Δ {top10_interval['mean_delta']:+.3f} "
                f"(95% CI [{top10_interval['low']:+.3f}; {top10_interval['high']:+.3f}]). "
                "Обычный CE оказался лучше chunk-level fusion и retrieval-бонусов.",
                f"Если под тем же лимитом 10 чанков отдать часть слотов вторым фрагментам, "
                f"exact R@10 растёт до {exact_top10['metrics']['recall_exact']:.3f} (Δ "
                f"{exact_top10_interval['mean_delta']:+.3f}, 95% CI "
                f"[{exact_top10_interval['low']:+.3f}; {exact_top10_interval['high']:+.3f}]), "
                f"но article R@10 падает до {exact_top10['metrics']['recall_article']:.3f}. "
                "Это явный конфликт за один и тот же бюджет.",
                f"Отдельный evidence pack дал намного больший эффект: exact recall "
                f"{pack20['metrics']['recall_exact']:.3f} при 20 чанках (Δ "
                f"{pack20_interval['mean_delta']:+.3f}, 95% CI "
                f"[{pack20_interval['low']:+.3f}; {pack20_interval['high']:+.3f}]) и "
                f"{pack30['metrics']['recall_exact']:.3f} при 30 (Δ "
                f"{pack30_interval['mean_delta']:+.3f}, 95% CI "
                f"[{pack30_interval['low']:+.3f}; {pack30_interval['high']:+.3f}]).",
                "Вывод: article top-10 и evidence context нужно разделить. "
                "Для article-метрики оставлять один чанк на статью; для генератора "
                "добавлять второй/третий недублирующий evidence-чанк. Diversity помогла именно "
                "широкому pack-30, но не выбору одного лучшего чанка.",
                f"Полный отчёт: `{evidence['report']}`. Eval для этого подбора не использовался.",
            ]
        )
    if "m700_evidence_refinement" in state:
        refinement = state["m700_evidence_refinement"]
        baseline = refinement["baseline_pack20"]
        neighbor = refinement["best_neighbor"]
        context_plain = refinement["context_plain_pack20"]
        context = refinement["context_pack20"]
        hierarchy = refinement["best_hierarchy"]
        dynamic = refinement["best_dynamic"]
        neighbor_interval = refinement["neighbor_vs_baseline"]
        context_interval = refinement["context_vs_plain"]
        dynamic_interval = refinement["dynamic_vs_baseline"]
        oracle = refinement["oracle"]
        lines.extend(
            [
                "",
                "## Дополнительное улучшение evidence на m700",
                "",
                f"На dev/blind проверено {refinement['count']} варианта: соседние структурные чанки, "
                "префикс статья/часть/пункт для cross-encoder, hierarchy bonus и динамический "
                "pack с порогом и токен-бюджетом.",
                f"Соседи не помогли: exact {baseline['metrics']['recall_exact']:.3f}→"
                f"{neighbor['metrics']['recall_exact']:.3f}, Δ {neighbor_interval['mean_delta']:+.3f} "
                f"(95% CI [{neighbor_interval['low']:+.3f}; {neighbor_interval['high']:+.3f}]). "
                f"Структурный CE-префикс также ухудшил exact на одинаковом пуле "
                f"{context_plain['metrics']['recall_exact']:.3f}→{context['metrics']['recall_exact']:.3f}, "
                f"Δ {context_interval['mean_delta']:+.3f} (95% CI "
                f"[{context_interval['low']:+.3f}; {context_interval['high']:+.3f}]).",
                f"Hierarchy bonus сохранил exact {hierarchy['metrics']['recall_exact']:.3f}, "
                "но не дал прироста. Полезный результат — динамический pack на исходном "
                f"пуле из 8 кандидатов/статью: exact остался "
                f"{baseline['metrics']['recall_exact']:.3f}→{dynamic['metrics']['recall_exact']:.3f} "
                f"(Δ {dynamic_interval['mean_delta']:+.3f}), а средний выход сократился "
                f"с {baseline['mean_chunks']:.1f} до {dynamic['mean_chunks']:.1f} чанка и с "
                f"{baseline['mean_estimated_tokens']:.0f} до {dynamic['mean_estimated_tokens']:.0f} условных токенов.",
                f"Oracle-разложение {sum(oracle.values())} gold-id: hit={oracle['hit']}, "
                f"article miss={oracle['article_not_selected']}, exact недостижим в m700="
                f"{oracle['exact_unavailable_in_m700']}, candidate miss={oracle['candidate_miss']}, "
                f"selection miss={oracle['selection_miss']}. Значительная часть оставшихся потерь "
                "находится до evidence selection или в гранулярности корпуса.",
                f"Полный отчёт: `{refinement['report']}`. Eval не использовался.",
            ]
        )
    if "m1000_evidence" in state:
        comparison = state["m1000_evidence"]
        m700_fixed = comparison["m700_fixed20"]
        m700_locked = comparison["m700_locked_dynamic"]
        m1000_fixed = comparison["m1000_fixed20"]
        m1000_locked = comparison["m1000_locked_dynamic"]
        m1000_best = comparison["m1000_best20"]
        fixed_interval = comparison["fixed20_m1000_vs_m700"]
        locked_interval = comparison["locked_m1000_vs_m700"]
        best_interval = comparison["best_m1000_vs_m700"]
        lines.extend(
            [
                "",
                "## Проверка evidence-чанков m1000 в FP32",
                "",
                f"На тех же {comparison['n']} dev/blind-кейсах и с неизменным top-10 статей "
                f"cross-encoder `{comparison['reranker_model']}` заново запущен в FP32. "
                "BF16-оценки в это сравнение не входят.",
                f"При одинаковом fixed pack-20 переход m700→m1000 изменил exact recall "
                f"{m700_fixed['metrics']['recall_exact']:.3f}→"
                f"{m1000_fixed['metrics']['recall_exact']:.3f}, Δ {fixed_interval['mean_delta']:+.3f} "
                f"(95% group-bootstrap [{fixed_interval['low']:+.3f}; "
                f"{fixed_interval['high']:+.3f}]). Интервал включает ноль.",
                f"Старая динамическая политика не переносится автоматически: exact "
                f"{m700_locked['metrics']['recall_exact']:.3f}→"
                f"{m1000_locked['metrics']['recall_exact']:.3f}, Δ "
                f"{locked_interval['mean_delta']:+.3f} (95% CI "
                f"[{locked_interval['low']:+.3f}; {locked_interval['high']:+.3f}]). "
                "Причина — более длинные чанки меняют заполнение того же token budget.",
                f"Лучший исследовательский m1000 pack на dev дал exact "
                f"{m1000_best['metrics']['recall_exact']:.3f} при "
                f"{m1000_best['mean_chunks']:.1f} чанка и "
                f"{m1000_best['mean_estimated_tokens']:.0f} условных токенах: Δ к locked m700 "
                f"{best_interval['mean_delta']:+.3f} (95% CI "
                f"[{best_interval['low']:+.3f}; {best_interval['high']:+.3f}]). Это dev-подбор, "
                "не подтверждённый на eval.",
                f"Цена по CPU выше: m700 {comparison['m700_cross_encoder_ms_per_query'] / 1000:.1f} "
                f"s/query против m1000 {comparison['m1000_cross_encoder_ms_per_query'] / 1000:.1f} "
                f"s/query ({comparison['m1000_latency_ratio']:.2f}× медленнее), хотя в m1000-пуле "
                "даже меньше пар. Поэтому m1000 пока не заменяет m700: прирост мал и статистически "
                "не установлен, а стоимость дополнительно выше.",
                f"Полный отчёт: `{comparison['report']}`. Eval не использовался.",
            ]
        )
    if "chunking_comparison" in state:
        chunking = state["chunking_comparison"]
        chunk_dev = chunking["dev_blind"]
        evaluated = chunking["eval_blind"]
        dev_struct = chunk_dev["structural"]["hybrid"]["at10"]
        dev_window = chunk_dev["window"]["hybrid"]["at10"]
        eval_struct = evaluated["structural"]["hybrid"]["at10"]
        eval_window = evaluated["window"]["hybrid"]["at10"]
        interval = chunk_dev["comparisons"]["hybrid"]
        lines.extend(
            [
                "",
                "## Structural против fixed-window",
                "",
                "Первичный chunk-level срез. Схема B построена фиксированными окнами по всему "
                "тексту редакции с overlap 200; "
                "norm-id назначаются по пересечению оффсетов. Dense m700, BM25 m1500/m2500 и "
                "article-level hybrid сравнены с structural на одних и тех же 52 dev и 22 eval кейсах.",
                f"Dev/blind hybrid structural→window: article R@10 "
                f"{dev_struct['recall_article']:.3f}→{dev_window['recall_article']:.3f}, exact "
                f"{dev_struct['recall_exact']:.3f}→{dev_window['recall_exact']:.3f}; Δ article "
                f"{interval['mean_delta']:+.3f} (95% CI [{interval['low']:+.3f}; "
                f"{interval['high']:+.3f}]).",
                f"One-shot eval/blind: article R@10 {eval_struct['recall_article']:.3f}→"
                f"{eval_window['recall_article']:.3f}, exact {eval_struct['recall_exact']:.3f}→"
                f"{eval_window['recall_exact']:.3f}. Все article-интервалы включают ноль; "
                "fixed-window не заменяет structural как основную схему.",
                "Это не чистая метрика article retrieval: одно окно может занять один chunk-slot, но засчитать "
                "несколько статей. Рост exact также частично объясняется гранулярностью: window-чанк несёт все "
                "пересечённые article/part/point id, тогда как structural point-чанк может нести "
                "только id пункта. Этот срез следует читать как coverage первых 10 контекстных фрагментов.",
                f"Полный отчёт: `{chunking['report']}`.",
            ]
        )
    if "chunking_role_comparison" in state:
        roles = state["chunking_role_comparison"]
        dev_article = roles["article_retrieval"]["dev"]
        eval_article = roles["article_retrieval"]["eval"]
        dev_pack = roles["evidence_packing"]["dev"]["end_to_end"]
        pack_interval = roles["evidence_packing"]["dev"]["comparisons"]["3000"]
        interval = dev_article["comparisons"]["hybrid"]
        lines.extend(
            [
                "",
                "### Разделение article retrieval и evidence packing",
                "",
                "В контрольной оценке каждая статья занимает отдельный top-10 slot, а score мультистатейного "
                "окна взвешивается по доле его текста внутри статьи. Dev/blind hybrid structural→window: article R@10 "
                f"{dev_article['structural']['hybrid']['recall_article']:.3f}→"
                f"{dev_article['window']['hybrid']['recall_article']:.3f}, Δ "
                f"{interval['mean_delta']:+.3f} (95% CI [{interval['low']:+.3f}; {interval['high']:+.3f}]).",
                "One-shot eval/blind: "
                f"{eval_article['structural']['hybrid']['recall_article']:.3f}→"
                f"{eval_article['window']['hybrid']['recall_article']:.3f}.",
                "При одинаковом evidence-бюджете 3000 условных токенов usable gold recall "
                f"{dev_pack['structural']['3000']['usable_gold_recall']:.3f}→"
                f"{dev_pack['window']['3000']['usable_gold_recall']:.3f}; window тратит "
                f"{dev_pack['window']['3000']['outside_selected_fraction']:.1%} текста вне выбранных статей. Δ usable "
                f"{pack_interval['mean_delta']:+.3f} (95% CI [{pack_interval['low']:+.3f}; "
                f"{pack_interval['high']:+.3f}]). "
                "По dev structural лучше в обеих ролях, но интервал article-разницы включает ноль.",
                f"Полный отчёт: `{roles['report']}`.",
            ]
        )
    if "baseline_e" in state:
        baseline_e = state["baseline_e"]
        dev_e = baseline_e["dev"]["group_holdout"]["at10"]
        dev_ci = baseline_e["dev"]["group_holdout"]["article_at10_ci"]
        lines.extend(
            [
                "",
                "## Baseline E: таблица issue_type → нормы",
                "",
                "Baseline E получает ручной gold `issue_type`; это oracle upper bound "
                "для lookup-таблицы, а не end-to-end качество поиска нарушений.",
                f"Честная dev-оценка leave-one-case-group-out: article R@10 "
                f"{dev_e['recall_article']:.3f} (95% group-bootstrap "
                f"[{dev_ci['low']:.3f}; {dev_ci['high']:.3f}]), exact R@10 "
                f"{dev_e['recall_exact']:.3f}, all-hit article@10 {dev_e['all_article']:.3f}.",
            ]
        )
        if "eval" in baseline_e:
            eval_e = baseline_e["eval"]["e_dev"]["at10"]
            seen = baseline_e["eval"]["seen_types"]["at10"]
            zero = baseline_e["eval"]["zero_shot"]["at10"]
            lines.extend(
                [
                    f"Замороженная full-dev таблица на eval: article/exact R@10 "
                    f"{eval_e['recall_article']:.3f}/{eval_e['recall_exact']:.3f}; на типах, "
                    f"встречавшихся в dev, article R@10 {seen['recall_article']:.3f}.",
                    f"Post-hoc zero-shot overlay для двух eval-only типов даёт "
                    f"article/exact R@10 {zero['recall_article']:.3f}/{zero['recall_exact']:.3f}, "
                    "но не является слепым eval-результатом.",
                ]
            )
        lines.extend(
            [
                "Вывод: при идеально известном типе таблица очень сильна. "
                "Следующий end-to-end эксперимент должен заменить gold-тип на "
                "предсказание эвристики/LLM и измерить всю цепочку.",
                f"Полный отчёт: `{baseline_e['report']}`.",
            ]
        )
    if "reranker_runtime_comparison" in state:
        runtime = state["reranker_runtime_comparison"]
        by_name = {row["runtime"]: row for row in runtime["rows"]}
        int8 = by_name["onnx-int8"]
        gguf = by_name["gguf-q8"]
        onnx_fp32 = by_name["onnx-fp32"]
        pytorch_en_ru = by_name.get("pytorch-fp32-en-ru")
        lines.extend(
            [
                "",
                "## CPU runtime и квантование cross-encoder",
                "",
                *(
                    [
                        f"Truncated PyTorch FP32 (`qilowoq/bge-reranker-v2-m3-en-ru`) уменьшает "
                        f"веса с {by_name['pytorch-fp32']['model_mb']:.0f} до "
                        f"{pytorch_en_ru['model_mb']:.0f} MB, но inference speedup равен только "
                        f"{pytorch_en_ru['speedup_vs_pytorch']:.2f}× (95% bootstrap CI "
                        f"[{pytorch_en_ru['paired_latency']['low']:.3f}×; "
                        f"{pytorch_en_ru['paired_latency']['high']:.3f}×]). На 10 контрольных "
                        f"кейсах Spearman {pytorch_en_ru['quality']['mean_spearman']:.4f}, "
                        f"chunk top-10 overlap "
                        f"{pytorch_en_ru['quality']['mean_chunk_top10_overlap']:.3f}.",
                    ]
                    if pytorch_en_ru is not None
                    else []
                ),
                f"Bilingual ONNX FP32 на {onnx_fp32['quality']['n']} контрольных кейсах практически "
                f"эквивалентен PyTorch FP32: Spearman {onnx_fp32['quality']['mean_spearman']:.4f}, "
                f"chunk top-10 overlap {onnx_fp32['quality']['mean_chunk_top10_overlap']:.3f}, "
                "но на CPU оказался медленнее.",
                f"ONNX INT8 ускоряет reranking в {int8['speedup_vs_pytorch']:.2f}×, однако chunk "
                f"top-10 overlap равен {int8['quality']['mean_chunk_top10_overlap']:.3f}, article top-5 "
                f"overlap {int8['quality']['mean_article_top5_overlap']:.3f}, а Δ article R@5 "
                f"{int8['quality']['article_r5_bootstrap']['mean_delta']:+.3f}.",
                f"GGUF Q8_0 ускоряет в {gguf['speedup_vs_pytorch']:.2f}× и снижает peak RSS до "
                f"{gguf['peak_rss_mb']:.0f} MB, но fidelity ещё ниже: chunk overlap "
                f"{gguf['quality']['mean_chunk_top10_overlap']:.3f}, article overlap "
                f"{gguf['quality']['mean_article_top5_overlap']:.3f}, Δ article R@5 "
                f"{gguf['quality']['article_r5_bootstrap']['mean_delta']:+.3f}.",
                "**Вывод:** truncated PyTorch полезен меньшим артефактом и cold start, но почти не "
                "ускоряет длинный reranking; перед production-заменой нужен полный 52-case fidelity-run. "
                "INT8 и GGUF Q8_0 не считать эквивалентной заменой; использовать их можно только "
                f"как явный quality–speed trade-off. Полный отчёт: `{runtime['report']}`.",
            ]
        )
    if "eval" in state:
        lines.extend(["", "## \u041e\u0434\u043d\u043e\u043a\u0440\u0430\u0442\u043d\u0430\u044f \u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 eval", "", "| \u043c\u0435\u0442\u043e\u0434 | R@10 exact | all@10 exact | R@10 art | all@10 art |", "|---|---:|---:|---:|---:|"])
        for item in state["eval"]["methods"]:
            metrics = item["metrics"]
            lines.append(
                f"| {item['name']} | {metrics['recall_exact']:.3f} | {metrics['all_exact']:.3f} "
                f"| {metrics['recall_article']:.3f} | {metrics['all_article']:.3f} |"
            )
        baseline_e = state.get("baseline_e")
        if baseline_e is not None and "eval" in baseline_e:
            for label, result in (
                ("Baseline E-dev frozen (oracle issue_type)", baseline_e["eval"]["e_dev"]),
                (
                    "Baseline E-dev + zero-shot (post-hoc)",
                    baseline_e["eval"]["zero_shot"],
                ),
            ):
                metrics = result["at10"]
                lines.append(
                    f"| {label} | {metrics['recall_exact']:.3f} | {metrics['all_exact']:.3f} "
                    f"| {metrics['recall_article']:.3f} | {metrics['all_article']:.3f} |"
                )
        bm25_screen = state.get("bm25_diversification")
        if bm25_screen is not None and "eval" in bm25_screen:
            bm25_chunk = bm25_screen["dev"]["winner"]["max_chunk"]
            bm25_eval = bm25_screen["eval"]
            for label, metrics in (
                (f"BM25 m{bm25_chunk}", bm25_eval["baseline"]),
                (f"BM25 m{bm25_chunk} article-diverse", bm25_eval["diversified"]),
            ):
                lines.append(
                    f"| {label} | {metrics['recall_exact']:.3f} | {metrics['all_exact']:.3f} "
                    f"| {metrics['recall_article']:.3f} | {metrics['all_article']:.3f} |"
                )
            for sensitivity in bm25_screen.get("eval_sensitivity", []):
                sensitivity_chunk = sensitivity["max_chunk"]
                for label, metrics in (
                    (f"BM25 m{sensitivity_chunk} baseline (post-hoc)", sensitivity["baseline"]),
                    (
                        f"BM25 m{sensitivity_chunk} article-diverse (post-hoc)",
                        sensitivity["diversified"],
                    ),
                ):
                    lines.append(
                        f"| {label} | {metrics['recall_exact']:.3f} | {metrics['all_exact']:.3f} "
                        f"| {metrics['recall_article']:.3f} | {metrics['all_article']:.3f} |"
                    )
        hybrid_screen = state.get("article_hybrid")
        if hybrid_screen is not None and "eval" in hybrid_screen:
            hybrid_metrics = hybrid_screen["eval"]["methods"]["locked article-level hybrid"]["at10"]
            lines.append(
                f"| locked article-level hybrid | {hybrid_metrics['recall_exact']:.3f} "
                f"| {hybrid_metrics['all_exact']:.3f} | {hybrid_metrics['recall_article']:.3f} "
                f"| {hybrid_metrics['all_article']:.3f} |"
            )
        cross_screen = state.get("cross_encoder")
        if cross_screen is not None and "eval_posthoc" in cross_screen:
            for label in ("locked conservative hybrid", "locked cross-encoder rerank"):
                metrics = cross_screen["eval_posthoc"]["methods"][label]["at10"]
                lines.append(
                    f"| {label} (post-hoc) | {metrics['recall_exact']:.3f} "
                    f"| {metrics['all_exact']:.3f} | {metrics['recall_article']:.3f} "
                    f"| {metrics['all_article']:.3f} |"
                )
        eval_outcomes = state["eval"]["outcomes"]
        interval = state["eval"]["bootstrap"]
        lines.extend(
            [
                "",
                f"\u041d\u0430 eval \u0434\u0438\u0432\u0435\u0440\u0441\u0438\u0444\u0438\u043a\u0430\u0446\u0438\u044f dense m700 \u0434\u0430\u043b\u0430 \u0432\u044b\u0438\u0433\u0440\u044b\u0448 \u0432 {eval_outcomes['gains']} \u043a\u0435\u0439\u0441\u0430\u0445, \u043f\u0440\u043e\u0438\u0433\u0440\u044b\u0448 \u0432 {eval_outcomes['losses']} \u0438 tie \u0432 {eval_outcomes['ties']}.",
                f"\u041f\u0430\u0440\u043d\u0430\u044f \u0440\u0430\u0437\u043d\u0438\u0446\u0430 article R@10: {interval['mean_delta']:+.3f} "
                f"(95% group-bootstrap [{interval['low']:+.3f}; {interval['high']:+.3f}]).",
                "\u041f\u0435\u0440\u0435\u043d\u043e\u0441 \u043f\u043e article recall \u043f\u043e\u0434\u0442\u0432\u0435\u0440\u0436\u0434\u0451\u043d, \u043d\u043e exact R@10 \u0441\u043d\u0438\u0437\u0438\u043b\u0441\u044f: \u0434\u043b\u044f point-level \u0432\u044b\u0434\u0430\u0447\u0438 \u043d\u0443\u0436\u043d\u043e \u0432\u043e\u0437\u0432\u0440\u0430\u0449\u0430\u0442\u044c \u0434\u043e\u043f\u043e\u043b\u043d\u0438\u0442\u0435\u043b\u044c\u043d\u044b\u0435 \u0447\u0430\u043d\u043a\u0438 \u0432\u043d\u0443\u0442\u0440\u0438 \u043e\u0442\u043e\u0431\u0440\u0430\u043d\u043d\u044b\u0445 \u0441\u0442\u0430\u0442\u0435\u0439.",
            ]
        )
    lines.extend(
        [
            "",
            "## \u0410\u0440\u0442\u0435\u0444\u0430\u043a\u0442\u044b \u043c\u0435\u0442\u0440\u0438\u043a\u0438",
            "",
            f"\u0421\u0442\u0430\u0442\u0443\u0441\u044b gold-id \u0432 \u043a\u043e\u0440\u043f\u0443\u0441\u0435 m700: `{json.dumps(dev['gold_status_counts'], ensure_ascii=False)}`. ",
            "Gold \u0441\u043e \u0441\u0442\u0430\u0442\u0443\u0441\u043e\u043c `descendant` \u043d\u0435 \u043c\u043e\u0436\u0435\u0442 \u0434\u0430\u0442\u044c literal exact-hit, \u0434\u0430\u0436\u0435 \u0435\u0441\u043b\u0438 \u0440\u0435\u043b\u0435\u0432\u0430\u043d\u0442\u043d\u044b\u0439 \u0434\u043e\u0447\u0435\u0440\u043d\u0438\u0439 \u043f\u0443\u043d\u043a \u043f\u043e\u043f\u0430\u043b \u0432 \u0432\u044b\u0434\u0430\u0447\u0443. \u041f\u043e\u044d\u0442\u043e\u043c\u0443 exact \u043d\u0435\u043b\u044c\u0437\u044f \u0441\u043c\u0435\u0448\u0438\u0432\u0430\u0442\u044c \u0441 \u0447\u0438\u0441\u0442\u043e\u0439 \u043e\u0448\u0438\u0431\u043a\u043e\u0439 \u0440\u0430\u043d\u0436\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u044f.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _serializable_result(item: MethodResult) -> dict[str, Any]:
    return {"name": item.name, "spec": item.spec, "metrics": _metrics_dict(item.metrics), "latency_ms": item.latency_ms}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="\u041f\u043e\u043a\u0435\u0439\u0441\u043e\u0432\u044b\u0439 error analysis retriever")
    parser.add_argument("--phase", choices=["dev", "eval"], default="dev")
    parser.add_argument("--edition-dir", type=Path, default=Path("data/law/2026-08-04"))
    parser.add_argument("--cases-dir", type=Path, default=Path("evals/cases"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("data/analysis"))
    parser.add_argument("--report", type=Path, default=Path("evals/reports/retrieval_error_analysis.md"))
    args = parser.parse_args(argv)
    state_path = args.analysis_dir / "retrieval_error_state.json"
    if args.phase == "eval" and not state_path.is_file():
        print("\u0441\u043d\u0430\u0447\u0430\u043b\u0430 \u0437\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u0435 --phase dev", file=sys.stderr)
        return 2

    all_cases = load_verified(args.cases_dir)
    cases = [case for case in all_cases if case.split == args.phase]
    expected = 52 if args.phase == "dev" else 22
    if len(cases) != expected:
        print(f"\u043e\u0436\u0438\u0434\u0430\u043b\u043e\u0441\u044c {expected} {args.phase}-\u043a\u0435\u0439\u0441\u0430, \u043f\u043e\u043b\u0443\u0447\u0435\u043d\u043e {len(cases)}", file=sys.stderr)
        return 2

    base_chunks = load_chunks(chunks_path(args.edition_dir, MAX_CHUNK))
    sensitivity_chunks = load_chunks(chunks_path(args.edition_dir, BM25_SENSITIVITY_CHUNK))
    profile = profile_for(MODEL)
    print(f"loading {MODEL}", file=sys.stderr)
    encoder = SentenceTransformerEncoder(
        MODEL,
        query_prefix=profile.query_prefix,
        document_prefix=profile.document_prefix,
        max_tokens=profile.max_tokens,
    )
    dense_index = DenseIndex.load_cached(
        base_chunks,
        encoder,
        cache_dir=args.edition_dir / "dense",
        model_name=MODEL,
        max_chunk=MAX_CHUNK,
    )
    if dense_index is None:
        print("\u043d\u0435\u0442 \u0431\u0430\u0437\u043e\u0432\u043e\u0433\u043e dense-\u043a\u044d\u0448\u0430 m700", file=sys.stderr)
        return 2
    bm25_index = Bm25Index(base_chunks)
    bm25_sensitivity_index = Bm25Index(sensitivity_chunks)

    full_rankings: dict[str, dict[str, list[list[Chunk]]]] = {}
    base_latency: dict[str, dict[str, float]] = {}
    for variant in ("raw", "blind"):
        dense_ranked, dense_ms = _rank_cases(dense_index, cases, variant, len(base_chunks))
        bm25_ranked, bm25_ms = _rank_cases(bm25_index, cases, variant, len(base_chunks))
        full_rankings[variant] = {"dense": dense_ranked, "bm25": bm25_ranked}
        base_latency[variant] = {"dense": dense_ms, "bm25": bm25_ms}
        print(f"{variant}: dense {dense_ms:.1f} ms, BM25 {bm25_ms:.1f} ms", file=sys.stderr)

    gold_status = corpus_gold_statuses(cases, base_chunks)
    pointer_ids = {
        norm_id
        for chunk in base_chunks
        if is_pointer_text(chunk.text)
        for norm_id in chunk.norm_ids
    }
    diagnostics = _write_cases(
        args.analysis_dir / f"retrieval_error_cases_{args.phase}.jsonl",
        cases,
        full_rankings,
        gold_status,
        pointer_ids,
    )

    if args.phase == "dev":
        bm25_sensitivity, sensitivity_ms = _rank_cases(
            bm25_sensitivity_index, cases, "blind", len(sensitivity_chunks)
        )
        title_chunks = _title_prefixed(base_chunks, args.edition_dir)
        title_dense_index = DenseIndex.build(
            title_chunks,
            encoder,
            cache_dir=args.edition_dir / "dense_title_prefix",
            model_name=MODEL,
            max_chunk=MAX_CHUNK,
        )
        title_bm25_index = Bm25Index(title_chunks)
        dense_title, dense_title_ms = _rank_cases(title_dense_index, cases, "blind", len(title_chunks))
        bm25_title, bm25_title_ms = _rank_cases(title_bm25_index, cases, "blind", len(title_chunks))
        methods = _build_dev_methods(
            cases,
            full_rankings["blind"]["dense"],
            full_rankings["blind"]["bm25"],
            base_latency["blind"]["dense"],
            base_latency["blind"]["bm25"],
            bm25_sensitivity,
            sensitivity_ms,
            dense_title,
            dense_title_ms,
            bm25_title,
            bm25_title_ms,
        )
        winner = _winner(methods)
        dense_baseline = next(item for item in methods if item.spec == {"kind": "dense"})
        interval = paired_group_bootstrap(
            per_case_article_recall(cases, dense_baseline.rankings),
            per_case_article_recall(cases, winner.rankings),
            [case.case_group_id for case in cases],
        )
        state = {
            "version": STATE_VERSION,
            "seed": 20261008,
            "model": MODEL,
            "max_chunk": MAX_CHUNK,
            "primary_query": "blind",
            "primary_metric": "article R@10",
            "model_overview": _parse_model_overview(args.report.parent),
            "dev": {
                "n": len(cases),
                "methods": [_serializable_result(item) for item in methods],
                "winner": _serializable_result(winner),
                "bootstrap": asdict(interval),
                "outcomes": _write_outcomes(
                    args.analysis_dir / "retrieval_experiment_outcomes_dev.jsonl",
                    cases,
                    dense_baseline.rankings,
                    winner.rankings,
                ),
                "diagnostics": diagnostics,
                "gold_status_counts": dict(Counter(gold_status.values())),
                "pointer_gold_cases": sum(
                    bool(set(case.gold_norms) & pointer_ids) for case in cases
                ),
            },
        }
        args.analysis_dir.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"locked winner: {winner.name}", file=sys.stderr)
    else:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("version") != STATE_VERSION:
            print("\u043d\u0435\u0441\u043e\u0432\u043c\u0435\u0441\u0442\u0438\u043c\u0430\u044f \u0432\u0435\u0440\u0441\u0438\u044f dev-state", file=sys.stderr)
            return 2
        spec = state["dev"]["winner"]["spec"]
        kwargs: dict[str, Any] = {}
        winner_latency = base_latency["blind"]["dense"] + base_latency["blind"]["bm25"]
        if _needs(spec, "bm25_sensitivity"):
            kwargs["bm25_sensitivity"], winner_latency = _rank_cases(
                bm25_sensitivity_index, cases, "blind", len(sensitivity_chunks)
            )
        if _needs(spec, "dense_title") or _needs(spec, "bm25_title"):
            title_chunks = _title_prefixed(base_chunks, args.edition_dir)
            if _needs(spec, "dense_title"):
                title_dense_index = DenseIndex.load_cached(
                    title_chunks,
                    encoder,
                    cache_dir=args.edition_dir / "dense_title_prefix",
                    model_name=MODEL,
                    max_chunk=MAX_CHUNK,
                )
                if title_dense_index is None:
                    print("\u043d\u0435\u0442 title-prefix dense-\u043a\u044d\u0448\u0430 \u0438\u0437 dev", file=sys.stderr)
                    return 2
                kwargs["dense_title"], winner_latency = _rank_cases(
                    title_dense_index, cases, "blind", len(title_chunks)
                )
            if _needs(spec, "bm25_title"):
                kwargs["bm25_title"], winner_latency = _rank_cases(
                    Bm25Index(title_chunks), cases, "blind", len(title_chunks)
                )
        winner_rankings = _apply_spec(
            spec,
            full_rankings["blind"]["dense"],
            full_rankings["blind"]["bm25"],
            **kwargs,
        )
        eval_methods = [
            _result(
                "dense m700",
                {"kind": "dense"},
                cases,
                full_rankings["blind"]["dense"],
                base_latency["blind"]["dense"],
            ),
            _result(
                "BM25 m700",
                {"kind": "bm25"},
                cases,
                full_rankings["blind"]["bm25"],
                base_latency["blind"]["bm25"],
            ),
            _result(state["dev"]["winner"]["name"], spec, cases, winner_rankings, winner_latency),
        ]
        state["eval"] = {
            "n": len(cases),
            "methods": [_serializable_result(item) for item in eval_methods],
            "outcomes": _write_outcomes(
                args.analysis_dir / "retrieval_experiment_outcomes_eval.jsonl",
                cases,
                full_rankings["blind"]["dense"],
                winner_rankings,
            ),
            "bootstrap": asdict(
                paired_group_bootstrap(
                    per_case_article_recall(cases, full_rankings["blind"]["dense"]),
                    per_case_article_recall(cases, winner_rankings),
                    [case.case_group_id for case in cases],
                )
            ),
            "diagnostics": diagnostics,
            "gold_status_counts": dict(Counter(gold_status.values())),
            "pointer_gold_cases": sum(
                bool(set(case.gold_norms) & pointer_ids) for case in cases
            ),
        }
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _report(state, args.report)
    print(f"wrote {args.report}", file=sys.stderr)
    print(f"wrote {state_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
