#!/usr/bin/env python3
"""Audit where gold articles and exact norms disappear in a frozen dev run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from schemas.corpus import Chunk  # noqa: E402
from schemas.eval_case import EvalCase  # noqa: E402
from scripts.run_grounded_dev_comparison import _prepared_rows  # noqa: E402


def article(norm_id: str) -> str:
    return norm_id.split(":")[2]


def primary_loss(result: dict) -> str:
    """Assign one earliest actionable loss, with a terminal available class."""
    if not result["corpus_exact"]:
        return "corpus_granularity"
    if not result["shortlist_article"]:
        return "article_shortlist"
    if not result["scored_exact"]:
        return "within_article_candidate_pool"
    if not result["pack_exact"]:
        return "evidence_pack"
    return "available_to_generator"


def analyze(prepared: Path, cases_dir: Path, corpus_path: Path) -> list[dict]:
    corpus = {
        chunk.chunk_id: chunk
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for chunk in [Chunk.model_validate_json(line)]
    }
    rows = []
    for row in _prepared_rows(prepared):
        case = EvalCase.model_validate_json(
            (cases_dir / f"{row['case_id']}.json").read_text(encoding="utf-8")
        )
        gold = set(case.gold_norms)
        gold_articles = {article(norm) for norm in gold}
        shortlist = row["retrieval"]["article_ids"]
        scored = [corpus[chunk_id] for chunk_id in row["retrieval"]["scored_chunk_ids"]]
        pack = [Chunk.model_validate(chunk) for chunk in row["retrieval"]["chunks"]]
        scored_norms = {norm for chunk in scored for norm in chunk.norm_ids}
        pack_norms = {norm for chunk in pack for norm in chunk.norm_ids}
        corpus_norms = {norm for chunk in corpus.values() for norm in chunk.norm_ids}
        article_rank = next((i + 1 for i, value in enumerate(shortlist) if value in gold_articles), None)
        result = {
            "case_id": row["case_id"],
            "topic": case.topic,
            "gold_articles": sorted(gold_articles),
            "corpus_exact": bool(gold & corpus_norms),
            "shortlist_article": article_rank is not None,
            "shortlist_article_rank": article_rank,
            "scored_article": bool(gold_articles & {article(norm) for norm in scored_norms}),
            "scored_exact": bool(gold & scored_norms),
            "pack_article": bool(gold_articles & {article(norm) for norm in pack_norms}),
            "pack_exact": bool(gold & pack_norms),
            "shortlist_size": len(shortlist),
            "scored_chunks": len(scored),
            "pack_chunks": len(pack),
        }
        result["primary_loss"] = primary_loss(result)
        rows.append(result)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/analysis/generation_10_dev_prepared.jsonl")
    parser.add_argument("--cases-dir", type=Path, default=ROOT / "evals/cases")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/law/2026-08-04/chunks_structural_m700.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "data/analysis/deepseek_retrieval_funnel.json")
    args = parser.parse_args()
    rows = analyze(args.prepared, args.cases_dir, args.corpus)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(
            row["case_id"],
            f"shortlist_rank={row['shortlist_article_rank']}",
            f"scored_exact={row['scored_exact']}",
            f"pack_article={row['pack_article']}",
            f"pack_exact={row['pack_exact']}",
            f"loss={row['primary_loss']}",
        )


if __name__ == "__main__":
    main()
