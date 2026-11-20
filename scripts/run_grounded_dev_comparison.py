#!/usr/bin/env python3
"""Prepare a fixed dev sample once, then run several generators on identical evidence.

The manifest and all detailed outputs belong in gitignored data/analysis/. Gold
labels are read only by `summarize`, never by `prepare` or `run` scoring logic.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate.llama_server import LlamaServerGenerator  # noqa: E402
from index.grounded_retrieval import FrozenArticleRetriever, RetrievalEvidence  # noqa: E402
from pipeline import GroundedArgumentPipeline  # noqa: E402
from schemas.corpus import Chunk  # noqa: E402
from schemas.eval_case import EvalCase  # noqa: E402
from schemas.grounded import ArgumentRequest, DocumentFragment  # noqa: E402
from validate.grounding import load_fragments  # noqa: E402


def _fragment(spec: dict, documents_root: Path) -> DocumentFragment:
    path = spec["path"]
    source = (documents_root / path).resolve(strict=True)
    if not source.is_relative_to(documents_root.resolve(strict=True)):
        raise ValueError(f"source escapes documents root: {path}")
    text = source.read_text(encoding="utf-8")
    if "anchor" in spec:
        anchor = spec["anchor"]
        occurrence = spec.get("occurrence")
        count = text.count(anchor)
        if count != 1 and occurrence is None:
            raise ValueError(f"anchor must occur exactly once: {path}: {anchor!r}")
        if occurrence is not None and (not isinstance(occurrence, int) or occurrence < 0 or occurrence >= count):
            raise ValueError(f"invalid anchor occurrence: {path}: {anchor!r}")
        index = 0
        for position in range((occurrence or 0) + 1):
            index = text.index(anchor, index)
            if position < (occurrence or 0):
                index += len(anchor)
        rough_start = max(0, index - spec.get("before", 0))
        rough_end = min(len(text), index + len(anchor) + spec.get("after", 0))
        start = text.rfind("\n", 0, rough_start) + 1
        next_newline = text.find("\n", rough_end)
        end = len(text) if next_newline < 0 else next_newline
    else:
        start_marker, end_marker = spec["start"], spec["end"]
        if text.count(start_marker) != 1:
            raise ValueError(f"start marker must occur exactly once: {path}")
        start = text.index(start_marker)
        end = text.index(end_marker, start + len(start_marker))
    return DocumentFragment(
        fragment_id=spec["fragment_id"], role=spec["role"], path=path,
        char_start=start, char_end=end,
    )


def _sample(manifest: Path, cases_dir: Path, documents_root: Path) -> list[tuple[str, ArgumentRequest]]:
    selectors = json.loads(manifest.read_text(encoding="utf-8"))
    if len(selectors) != 10 or len({row["case_id"] for row in selectors}) != 10:
        raise ValueError("comparison manifest must contain 10 distinct dev arguments")
    sample = []
    groups = set()
    for row in selectors:
        case_id = row["case_id"]
        case = EvalCase.model_validate_json((cases_dir / f"{case_id}.json").read_text(encoding="utf-8"))
        if case.split != "dev" or case.annotation_status != "verified" or not case.complaint_argument_blind:
            raise ValueError(f"not a verified dev/blind case: {case_id}")
        if case.case_group_id in groups:
            raise ValueError(f"duplicate case group: {case.case_group_id}")
        groups.add(case.case_group_id)
        request = ArgumentRequest(
            complaint_argument=case.complaint_argument_blind,
            edition_id=case.edition_id,
            fragments=[_fragment(spec, documents_root) for spec in row["fragments"]],
        )
        load_fragments(request, documents_root)
        if not any(item.role == "claim" for item in request.fragments):
            raise ValueError(f"no claim fragment: {case_id}")
        sample.append((case_id, request))
    return sample


def prepare(args: argparse.Namespace) -> None:
    sample = _sample(args.manifest, args.cases_dir, args.documents_root)
    existing = (
        {row["case_id"]: row for row in _prepared_rows(args.prepared)}
        if args.reuse_existing and args.prepared.exists() else {}
    )
    retriever = None
    rows = []
    for case_id, item in sample:
        old = existing.get(case_id)
        if old:
            previous = ArgumentRequest.model_validate(old["request"])
            if (previous.complaint_argument, previous.edition_id) != (item.complaint_argument, item.edition_id):
                raise ValueError(f"cannot reuse retrieval after query/edition change: {case_id}")
            old["request"] = item.model_dump(mode="json")
            rows.append(old)
            print(f"reused {case_id}: {len(old['retrieval']['chunks'])} chunks", flush=True)
            continue
        if retriever is None:
            retriever = FrozenArticleRetriever.from_local(
                args.edition_dir, cross_encoder_path=args.cross_encoder_path,
                evidence_max_chunks=args.evidence_max_chunks,
            )
        started = time.perf_counter()
        evidence = retriever.retrieve(item.complaint_argument)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        rows.append({
            "case_id": case_id,
            "request": item.model_dump(mode="json"),
            "retrieval_ms": elapsed_ms,
            "retrieval": {
                "article_ids": evidence.article_ids,
                "chunks": [chunk.model_dump(mode="json") for chunk in evidence.chunks],
                "scored_chunk_ids": evidence.scored_chunk_ids,
            },
        })
        print(f"prepared {case_id}: {len(evidence.chunks)} chunks, {elapsed_ms:.0f} ms", flush=True)
    args.prepared.parent.mkdir(parents=True, exist_ok=True)
    args.prepared.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )


class StaticRetriever:
    def __init__(self, evidence: RetrievalEvidence) -> None:
        self.evidence = evidence

    def retrieve(self, query: str) -> RetrievalEvidence:
        return self.evidence


def _prepared_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 10 or len({row["case_id"] for row in rows}) != 10:
        raise ValueError("prepared input must contain 10 distinct cases")
    return rows


def _article_id(norm_id: str) -> str:
    parts = norm_id.split(":")
    if len(parts) < 3 or not parts[2].startswith("art."):
        raise ValueError(f"invalid norm id: {norm_id}")
    return ":".join(parts[:3])


def refresh(args: argparse.Namespace) -> None:
    """Update only source spans after selector corrections; preserve frozen retrieval."""
    sample = _sample(args.manifest, args.cases_dir, args.documents_root)
    rows = _prepared_rows(args.prepared)
    if [row["case_id"] for row in rows] != [case_id for case_id, _ in sample]:
        raise ValueError("manifest order does not match prepared retrieval")
    for row, (_, item) in zip(rows, sample, strict=True):
        previous = ArgumentRequest.model_validate(row["request"])
        if (previous.complaint_argument, previous.edition_id) != (item.complaint_argument, item.edition_id):
            raise ValueError("retrieval query or edition changed")
        row["request"] = item.model_dump(mode="json")
    args.prepared.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    print("refreshed 10 source-span requests; frozen retrieval unchanged")


def run(args: argparse.Namespace) -> None:
    rows = _prepared_rows(args.prepared)
    reused = {}
    if args.reuse_run_from:
        reused = {
            trace["case_id"]: trace
            for line in args.reuse_run_from.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for trace in [json.loads(line)]
        }
    generator = LlamaServerGenerator(
        base_url=args.base_url, model=args.model, temperature=args.temperature,
        max_tokens=args.max_tokens, timeout_seconds=args.timeout_seconds,
        seed=args.seed, enable_thinking=args.thinking,
    )
    output = []
    for row in rows:
        prior = reused.get(row["case_id"])
        if prior:
            item = ArgumentRequest.model_validate(row["request"])
            if (
                prior["generation_metadata"].get("model") != args.model
                or prior["complaint_argument"] != item.complaint_argument
                or prior["edition_id"] != item.edition_id
                or [chunk["chunk_id"] for chunk in prior["retrieval"]["evidence"]]
                != [chunk["chunk_id"] for chunk in row["retrieval"]["chunks"]]
                or [
                    (fragment["fragment_id"], fragment["path"], fragment["char_start"], fragment["char_end"])
                    for fragment in prior["fragments"]
                ] != [
                    (fragment.fragment_id, fragment.path, fragment.char_start, fragment.char_end)
                    for fragment in item.fragments
                ]
            ):
                raise ValueError(f"cannot reuse changed run trace: {row['case_id']}")
            output.append(prior)
            print(f"reused run {row['case_id']}", flush=True)
            continue
        frozen = row["retrieval"]
        evidence = RetrievalEvidence(
            article_ids=frozen["article_ids"],
            chunks=[Chunk.model_validate(chunk) for chunk in frozen["chunks"]],
            scored_chunk_ids=frozen["scored_chunk_ids"],
        )
        pipeline = GroundedArgumentPipeline(
            edition_dir=args.edition_dir,
            documents_root=args.documents_root,
            retriever=StaticRetriever(evidence),
            generator=generator,
        )
        item = ArgumentRequest.model_validate(row["request"])
        result, trace = pipeline.run(item)
        trace["case_id"] = row["case_id"]
        trace["frozen_retrieval_ms"] = row["retrieval_ms"]
        output.append(trace)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in output) + "\n",
            encoding="utf-8",
        )
        print(f"{row['case_id']}: {result.status} / {result.reason_code or 'remark'}", flush=True)
    if len(output) == len(rows):
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in output) + "\n",
            encoding="utf-8",
        )


def summarize(args: argparse.Namespace) -> None:
    cases = {
        case_id: EvalCase.model_validate_json((args.cases_dir / f"{case_id}.json").read_text(encoding="utf-8"))
        for case_id in [row["case_id"] for row in _prepared_rows(args.prepared)]
    }
    results = {}
    for label, path in args.run:
        traces = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        if {row["case_id"] for row in traces} != set(cases) or len(traces) != 10:
            raise ValueError(f"run {label} does not match the fixed sample")
        results[label] = {}
        for trace in traces:
            case_id = trace["case_id"]
            gold = set(cases[case_id].gold_norms)
            shown = {norm for chunk in trace["retrieval"]["evidence"] for norm in chunk["norm_ids"]}
            gold_articles = {_article_id(norm) for norm in gold}
            shown_articles = {_article_id(norm) for norm in shown}
            result = trace["result"]
            norm_id = result.get("remark", {}).get("norm_id") if result.get("remark") else None
            results[label][case_id] = {
                "topic": cases[case_id].topic,
                "gold_in_pack": bool(gold & shown),
                "gold_article_in_pack": bool(gold_articles & shown_articles),
                "status": result["status"],
                "reason_code": result.get("reason_code"),
                "selected_norm_id": norm_id,
                "gold_norm_selected": norm_id in gold if norm_id else False,
                "gold_article_selected": _article_id(norm_id) in gold_articles if norm_id else False,
                "generation_ms": trace["timings_ms"].get("generation"),
                "doc_quote": result.get("remark", {}).get("doc_quote") if result.get("remark") else None,
                "remark_text": result.get("remark", {}).get("remark_text") if result.get("remark") else None,
            }
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for label, cases_for_model in results.items():
        values = list(cases_for_model.values())
        print(
            label,
            f"gold_in_pack={sum(item['gold_in_pack'] for item in values)}/10",
            f"gold_article_in_pack={sum(item['gold_article_in_pack'] for item in values)}/10",
            f"remark={sum(item['status'] == 'remark' for item in values)}/10",
            f"gold_norm_selected={sum(item['gold_norm_selected'] for item in values)}/10",
            f"error={sum(item['status'] == 'error' for item in values)}/10",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check", "prepare", "refresh", "run", "summarize"])
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/analysis/generation_10_dev_manifest.json")
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/analysis/generation_10_dev_prepared.jsonl")
    parser.add_argument("--cases-dir", type=Path, default=ROOT / "evals/cases")
    parser.add_argument("--edition-dir", type=Path, default=ROOT / "data/law/2026-08-04")
    parser.add_argument("--documents-root", type=Path, default=ROOT / "data/raw/notices")
    parser.add_argument("--cross-encoder-path", type=Path, default=ROOT / "data/analysis/models/bge-reranker-v2-m3-en-ru-pytorch")
    parser.add_argument("--evidence-max-chunks", type=int, default=5)
    parser.add_argument("--reuse-existing", action="store_true", help="reuse frozen retrieval for unchanged case queries")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="T-lite-it-2.1")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "data/analysis/generation_10_dev_trace.jsonl")
    parser.add_argument("--reuse-run-from", type=Path, help="reuse matching case traces from an earlier run of the same model")
    parser.add_argument("--run", action="append", nargs=2, metavar=("LABEL", "PATH"), default=[])
    parser.add_argument("--summary-out", type=Path)
    args = parser.parse_args()
    if args.stage == "check":
        for case_id, item in _sample(args.manifest, args.cases_dir, args.documents_root):
            print(case_id, [(fragment.fragment_id, fragment.char_end - fragment.char_start) for fragment in item.fragments])
    elif args.stage == "prepare":
        prepare(args)
    elif args.stage == "refresh":
        refresh(args)
    elif args.stage == "run":
        run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
