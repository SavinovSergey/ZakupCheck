#!/usr/bin/env python3
"""Compare DeepSeek-V4-Pro on the frozen 10-case dev evidence, with redacted payloads.

`preview` never calls the API. Inspect its gitignored output before `run`.
The model sees no gold labels, FAS decisions, raw source paths or unredacted
personal/structured identifiers. Original local spans remain authoritative for
the existing quote validator; the model only chooses stable span IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from urllib import error, request

from pydantic import TypeAdapter, ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from generate.llama_server import GenerationError, build_messages  # noqa: E402
from index.grounded_retrieval import RetrievalEvidence  # noqa: E402
from pipeline import GroundedArgumentPipeline  # noqa: E402
from privacy.anonymize import ManualRedaction, anonymize_fields, load_manual_redactions, scan_text  # noqa: E402
from schemas.corpus import Chunk  # noqa: E402
from schemas.grounded import ArgumentRequest, ModelDraft  # noqa: E402
from validate.grounding import LoadedFragment, load_fragments  # noqa: E402

from scripts.run_grounded_dev_comparison import StaticRetriever, _prepared_rows  # noqa: E402

MODEL = "deepseek-v4-pro"
ENDPOINT = "https://api.deepseek.com/chat/completions"
_DRAFT = TypeAdapter(ModelDraft)
_JSON_EXAMPLE = (
    'Формат JSON для замечания: {"decision":"remark","remark_text":"одно предложение",'
    '"source_fragment_id":"id","doc_span_id":"s0","norm_chunk_id":"chunk_id",'
    '"norm_id":"norm_id","norm_span_id":"s0"}. '
    'Для отказа: {"decision":"abstain","reason":"одно предложение"}. '
    "Не добавляй других полей."
)


def api_key(env_file: Path) -> str:
    """Read the key without logging it; process environment takes precedence."""
    value = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if value:
        return value
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key.removeprefix("export ").strip() == "DEEPSEEK_API_KEY":
                return value.strip().strip('"\'')
    raise RuntimeError("DEEPSEEK_API_KEY is missing from environment and local .env")


def outgoing_messages(
    complaint_argument: str,
    fragments: Sequence[LoadedFragment],
    evidence: Sequence[Chunk],
    *,
    manual_redactions: Sequence[ManualRedaction] = (),
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Redact outbound text while retaining the original span IDs for validation."""
    messages = build_messages(complaint_argument, fragments, evidence)
    user_text = messages[1]["content"]
    for fragment in fragments:
        user_text = user_text.replace(fragment.spec.path, "<LOCAL_SOURCE>")
    result = anonymize_fields({"payload": user_text}, manual_redactions=manual_redactions)
    safe_text = result.fields["payload"]
    if scan_text(safe_text, manual_redactions=manual_redactions):
        raise ValueError("detectable sensitive text survived outbound redaction")
    messages[0]["content"] += " Верни JSON. " + _JSON_EXAMPLE
    messages[1]["content"] = safe_text
    counts = dict(Counter(item.kind for item in result.detections["payload"]))
    return messages, counts


def _evidence(row: dict) -> RetrievalEvidence:
    frozen = row["retrieval"]
    return RetrievalEvidence(
        article_ids=frozen["article_ids"],
        chunks=[Chunk.model_validate(chunk) for chunk in frozen["chunks"]],
        scored_chunk_ids=frozen["scored_chunk_ids"],
    )


class DeepSeekGenerator:
    def __init__(
        self, *, token: str, effort: str, max_tokens: int,
        timeout_seconds: int, manual_redactions: Sequence[ManualRedaction],
    ) -> None:
        self.token = token
        self.effort = effort
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.manual_redactions = manual_redactions
        self.last_generation_metadata: dict | None = None

    def generate(
        self, complaint_argument: str, fragments: Sequence[LoadedFragment],
        evidence: Sequence[Chunk],
    ) -> ModelDraft:
        self.last_generation_metadata = None
        messages, redactions = outgoing_messages(
            complaint_argument, fragments, evidence,
            manual_redactions=self.manual_redactions,
        )
        payload = {
            "model": MODEL,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "enabled"},
            "reasoning_effort": self.effort,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        outgoing = request.Request(
            ENDPOINT, data=encoded, method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.token,
            },
        )
        try:
            with request.urlopen(outgoing, timeout=self.timeout_seconds) as response:
                body = json.load(response)
            choice = body["choices"][0]
            if choice.get("finish_reason") == "length":
                raise GenerationError("DeepSeek response truncated at max_tokens")
            message = choice["message"]
            content = message["content"]
            if not isinstance(content, str) or not content.strip():
                raise GenerationError("DeepSeek returned empty non-reasoning content")
            draft = _DRAFT.validate_json(content)
            reasoning = message.get("reasoning_content")
            self.last_generation_metadata = {
                "model": body.get("model", MODEL),
                "effort": self.effort,
                "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else 0,
                "usage": body.get("usage"),
                "outgoing_sha256": hashlib.sha256(encoded).hexdigest(),
                "outgoing_redactions": redactions,
            }
            return draft
        except error.HTTPError as exc:
            raise GenerationError(f"DeepSeek HTTP {exc.code}") from exc
        except (error.URLError, TimeoutError, OSError) as exc:
            raise GenerationError(f"DeepSeek transport error: {type(exc).__name__}") from exc
        except (KeyError, IndexError, ValueError, ValidationError) as exc:
            raise GenerationError(f"invalid DeepSeek response: {type(exc).__name__}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["preview", "run"])
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/analysis/generation_10_dev_prepared.jsonl")
    parser.add_argument("--edition-dir", type=Path, default=ROOT / "data/law/2026-08-04")
    parser.add_argument("--documents-root", type=Path, default=ROOT / "data/raw/notices")
    parser.add_argument("--redactions", type=Path, default=ROOT / "data/analysis/public_redactions.json")
    parser.add_argument("--extra-redactions", type=Path, default=ROOT / "data/analysis/deepseek_redactions.json")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--preview-out", type=Path, default=ROOT / "data/analysis/deepseek_v4_pro_outgoing_preview.jsonl")
    parser.add_argument("--trace-out", type=Path, default=ROOT / "data/analysis/generation_10_dev_deepseek_v4_pro_trace.jsonl")
    parser.add_argument("--effort", choices=["low", "high", "max"], default="high")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--resume", action="store_true", help="skip previously completed cases in trace-out")
    args = parser.parse_args()

    rows = _prepared_rows(args.prepared)
    global_redactions, by_case = load_manual_redactions(args.redactions)
    if args.extra_redactions.is_file():
        extra_global, extra_by_case = load_manual_redactions(args.extra_redactions)
        global_redactions.extend(extra_global)
        for case_id, items in extra_by_case.items():
            by_case.setdefault(case_id, []).extend(items)
    if args.stage == "preview":
        output = []
        for row in rows:
            item = ArgumentRequest.model_validate(row["request"])
            fragments = load_fragments(item, args.documents_root)
            messages, counts = outgoing_messages(
                item.complaint_argument, fragments, _evidence(row).chunks,
                manual_redactions=[*global_redactions, *by_case.get(row["case_id"], [])],
            )
            output.append({"case_id": row["case_id"], "messages": messages, "redactions": counts})
            print(f"preview {row['case_id']}: {counts}", flush=True)
        args.preview_out.parent.mkdir(parents=True, exist_ok=True)
        args.preview_out.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in output) + "\n",
            encoding="utf-8",
        )
        print(f"review outbound messages locally: {args.preview_out}")
        return

    if not args.preview_out.is_file():
        raise RuntimeError("run preview and review its outbound messages before API calls")
    preview = [json.loads(line) for line in args.preview_out.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [row["case_id"] for row in rows] != [row["case_id"] for row in preview]:
        raise RuntimeError("preview does not match prepared case order")
    token = api_key(args.env_file)
    completed = {}
    if args.resume and args.trace_out.is_file():
        completed = {
            trace["case_id"]: trace
            for line in args.trace_out.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for trace in [json.loads(line)]
        }
    traces = []
    for row, preview_row in zip(rows, preview, strict=True):
        item = ArgumentRequest.model_validate(row["request"])
        evidence = _evidence(row)
        fragments = load_fragments(item, args.documents_root)
        messages, _ = outgoing_messages(
            item.complaint_argument, fragments, evidence.chunks,
            manual_redactions=[*global_redactions, *by_case.get(row["case_id"], [])],
        )
        if messages != preview_row["messages"]:
            raise RuntimeError(f"outbound payload changed since review: {row['case_id']}")
        prior = completed.get(row["case_id"])
        if prior and prior["result"]["status"] == "error":
            prior = None
        if prior:
            if (
                prior["complaint_argument"] != item.complaint_argument
                or prior["edition_id"] != item.edition_id
                or [chunk["chunk_id"] for chunk in prior["retrieval"]["evidence"]]
                != [chunk.chunk_id for chunk in evidence.chunks]
                or prior.get("generation_metadata", {}).get("model") != MODEL
            ):
                raise RuntimeError(f"cannot reuse changed or failed trace: {row['case_id']}")
            traces.append(prior)
            print(f"reused API result {row['case_id']}", flush=True)
            continue
        generator = DeepSeekGenerator(
            token=token, effort=args.effort, max_tokens=args.max_tokens,
            timeout_seconds=args.timeout_seconds,
            manual_redactions=[*global_redactions, *by_case.get(row["case_id"], [])],
        )
        pipeline = GroundedArgumentPipeline(
            edition_dir=args.edition_dir,
            documents_root=args.documents_root,
            retriever=StaticRetriever(evidence),
            generator=generator,
        )
        started = time.perf_counter()
        result, trace = pipeline.run(item)
        trace["case_id"] = row["case_id"]
        trace["wall_ms"] = round((time.perf_counter() - started) * 1000, 1)
        trace["frozen_retrieval_ms"] = row["retrieval_ms"]
        traces.append(trace)
        args.trace_out.parent.mkdir(parents=True, exist_ok=True)
        args.trace_out.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in traces) + "\n",
            encoding="utf-8",
        )
        print(f"{row['case_id']}: {result.status} / {result.reason_code or 'remark'}", flush=True)
        if result.status == "error":
            raise RuntimeError(f"stopped after API/pipeline error on {row['case_id']}: {result.reason_code}")


if __name__ == "__main__":
    main()
