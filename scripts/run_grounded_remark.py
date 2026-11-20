#!/usr/bin/env python3
"""Run one grounded argument; --demo-case uses only dev input, never gold labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate.llama_server import LlamaServerGenerator  # noqa: E402
from index.grounded_retrieval import FrozenArticleRetriever  # noqa: E402
from pipeline import GroundedArgumentPipeline  # noqa: E402
from schemas.eval_case import EvalCase  # noqa: E402
from schemas.grounded import ArgumentRequest, DocumentFragment  # noqa: E402

DEMO_CASE_ID = "fas_2026_59011718"
DEMO_NOTICE = "0372200208826000004"
DEMO_SCOPE = f"{DEMO_NOTICE}/text/1-Описание_объекта_закупки.txt"
DEMO_REQUIREMENT = f"{DEMO_NOTICE}/text/3-Требованияк_содержанию_составу.txt"


def _between(text: str, start: str, end: str) -> tuple[int, int]:
    left = text.index(start)
    right = text.index(end, left + len(start))
    return left, right


def demo_request(
    *,
    cases_dir: Path,
    documents_root: Path,
    omit_requirement: bool = False,
) -> ArgumentRequest:
    case = EvalCase.model_validate_json(
        (cases_dir / f"{DEMO_CASE_ID}.json").read_text(encoding="utf-8")
    )
    if case.split != "dev" or not case.complaint_argument_blind:
        raise ValueError("demo requires the verified dev blind argument")
    scope = (documents_root / DEMO_SCOPE).read_text(encoding="utf-8")
    requirement = (documents_root / DEMO_REQUIREMENT).read_text(encoding="utf-8")
    scope_start, scope_end = _between(
        scope,
        "3.2.1. Объёмные требования:",
        "- запрет открывания преграждающего устройства",
    )
    requirement_start, requirement_end = _between(
        requirement,
        "__- реквизиты лицензии",
        "\nИнформацию и документы, подтверждающие соответствие участника закупки дополнительным требованиям",
    )
    fragments = [
        DocumentFragment(
            fragment_id="scope", path=DEMO_SCOPE, role="context",
            char_start=scope_start, char_end=scope_end,
        ),
        DocumentFragment(
            fragment_id="license_requirement", path=DEMO_REQUIREMENT,
            char_start=requirement_start, char_end=requirement_end,
        ),
    ]
    if omit_requirement:
        fragments = fragments[:1]
    return ArgumentRequest(
        complaint_argument=case.complaint_argument_blind,
        edition_id=case.edition_id,
        fragments=fragments,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="JSON ArgumentRequest")
    source.add_argument("--demo-case", action="store_true", help=f"use dev case {DEMO_CASE_ID}")
    parser.add_argument("--omit-requirement", action="store_true", help="demo negative control")
    parser.add_argument("--cases-dir", type=Path, default=ROOT / "evals/cases")
    parser.add_argument("--edition-dir", type=Path, default=ROOT / "data/law/2026-08-04")
    parser.add_argument("--documents-root", type=Path, default=ROOT / "data/raw/notices")
    parser.add_argument(
        "--cross-encoder-path", type=Path,
        default=ROOT / "data/analysis/models/bge-reranker-v2-m3-en-ru-pytorch",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="T-lite-it-2.1")
    parser.add_argument("--generation-temperature", type=float, default=0)
    parser.add_argument("--generation-max-tokens", type=int, default=768)
    parser.add_argument("--generation-top-p", type=float)
    parser.add_argument("--generation-top-k", type=int)
    parser.add_argument("--generation-timeout-seconds", type=int, default=240)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--generation-seed", type=int)
    parser.add_argument(
        "--evidence-max-chunks", type=int, default=20,
        help="maximum chunks shown to the generator (default: 20)",
    )
    parser.add_argument("--trace-out", type=Path, default=ROOT / "data/analysis/grounded_remark_trace.json")
    args = parser.parse_args(argv)
    if args.generation_max_tokens < 1 or args.generation_timeout_seconds < 1:
        parser.error("generation token budget and timeout must be positive")
    if args.omit_requirement and not args.demo_case:
        parser.error("--omit-requirement requires --demo-case")

    try:
        item = (
            demo_request(
                cases_dir=args.cases_dir,
                documents_root=args.documents_root,
                omit_requirement=args.omit_requirement,
            )
            if args.demo_case
            else ArgumentRequest.model_validate_json(args.input.read_text(encoding="utf-8"))
        )
        retriever = FrozenArticleRetriever.from_local(
            args.edition_dir,
            cross_encoder_path=args.cross_encoder_path,
            evidence_max_chunks=args.evidence_max_chunks,
        )
        pipeline = GroundedArgumentPipeline(
            edition_dir=args.edition_dir,
            documents_root=args.documents_root,
            retriever=retriever,
            generator=LlamaServerGenerator(
                base_url=args.base_url,
                model=args.model,
                timeout_seconds=args.generation_timeout_seconds,
                temperature=args.generation_temperature,
                max_tokens=args.generation_max_tokens,
                top_p=args.generation_top_p,
                top_k=args.generation_top_k,
                enable_thinking=args.thinking,
                seed=args.generation_seed,
            ),
        )
        result, trace = pipeline.run(item)
    except (OSError, ValueError) as exc:
        print(f"setup error: {exc}", file=sys.stderr)
        return 2
    args.trace_out.parent.mkdir(parents=True, exist_ok=True)
    args.trace_out.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(result.model_dump_json(indent=2))
    print(f"Local trace: {args.trace_out}", file=sys.stderr)
    return 0 if result.status != "error" else 2


if __name__ == "__main__":
    raise SystemExit(main())
