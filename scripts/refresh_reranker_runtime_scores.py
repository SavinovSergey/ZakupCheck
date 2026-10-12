#!/usr/bin/env python3
"""Migrate runtime-comparison scores after the article coverage fix.

Unchanged m700 chunks keep their exact text but receive shifted sequential ids.
The script remaps those ids and computes only genuinely missing ONNX/GGUF
scores before the runtime comparison report is regenerated.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from index.bm25 import load_chunks  # noqa: E402
from index.dense import corpus_fingerprint  # noqa: E402
from index.eval_tables import chunks_path, load_verified, query_text  # noqa: E402
from run_cross_encoder_rerank_eval import legacy_score_id_map  # noqa: E402
from run_reranker_runtime_comparison import (  # noqa: E402
    GGUF_MODEL,
    _request_json,
    _wait_for_server,
)
from schemas.corpus import Chunk  # noqa: E402
from schemas.eval_case import EvalCase  # noqa: E402


def migrate_scores(
    scores: dict[str, dict[str, float]],
    id_map: dict[str, str],
) -> dict[str, dict[str, float]]:
    return {
        case_id: {
            id_map[chunk_id]: float(score)
            for chunk_id, score in case_scores.items()
            if chunk_id in id_map
        }
        for case_id, case_scores in scores.items()
    }


def missing_pairs(
    scores: dict[str, dict[str, float]],
    reference: dict[str, dict[str, float]],
) -> dict[str, list[str]]:
    return {
        case_id: sorted(set(reference[case_id]) - set(case_scores))
        for case_id, case_scores in scores.items()
        if set(reference[case_id]) - set(case_scores)
    }


def repair_onnx_int8(
    *,
    cases: dict[str, EvalCase],
    chunks: dict[str, Chunk],
    missing: dict[str, list[str]],
    onnx_dir: Path,
    threads: int,
) -> dict[str, dict[str, float]]:
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(onnx_dir, local_files_only=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_dir / "model.int8.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_names = {item.name for item in session.get_inputs()}
    repaired: dict[str, dict[str, float]] = {}
    for case_id, chunk_ids in missing.items():
        case = cases[case_id]
        encoded = tokenizer(
            [query_text(case, "blind")] * len(chunk_ids),
            [chunks[chunk_id].text for chunk_id in chunk_ids],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        inputs = {
            name: np.asarray(encoded[name], dtype=np.int64)
            for name in input_names
            if name in encoded
        }
        logits = np.asarray(session.run(None, inputs)[0], dtype=np.float32).reshape(-1)
        repaired[case_id] = dict(zip(chunk_ids, logits.tolist(), strict=True))
    return repaired


def repair_gguf(
    *,
    cases: dict[str, EvalCase],
    chunks: dict[str, Chunk],
    missing: dict[str, list[str]],
    llama_server: Path,
    gguf_model: Path,
    threads: int,
    port: int,
    log_path: Path,
) -> dict[str, dict[str, float]]:
    base_url = f"http://127.0.0.1:{port}"
    command = [
        str(llama_server),
        "--model", str(gguf_model),
        "--embedding", "--reranking", "--pooling", "rank",
        "--ctx-size", "512",
        "--threads", str(threads),
        "--threads-batch", str(threads),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--no-webui",
    ]
    repaired: dict[str, dict[str, float]] = {}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        try:
            _wait_for_server(process, base_url)
            for case_id, chunk_ids in missing.items():
                response = _request_json(
                    f"{base_url}/v1/rerank",
                    {
                        "model": GGUF_MODEL,
                        "query": query_text(cases[case_id], "blind"),
                        "documents": [chunks[chunk_id].text for chunk_id in chunk_ids],
                        "top_n": len(chunk_ids),
                    },
                )
                by_index = {
                    int(item["index"]): float(item["relevance_score"])
                    for item in response["results"]
                }
                repaired[case_id] = {
                    chunk_id: by_index[index]
                    for index, chunk_id in enumerate(chunk_ids)
                }
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    return repaired


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh reranker runtime score ids and missing pairs")
    parser.add_argument("--edition-dir", type=Path, default=Path("data/law/2026-08-04"))
    parser.add_argument("--cases-dir", type=Path, default=Path("evals/cases"))
    parser.add_argument("--reference", type=Path, default=Path("data/analysis/cross_encoder_scores.json"))
    parser.add_argument("--runtime-dir", type=Path, default=Path("data/analysis/reranker_runtime"))
    parser.add_argument("--onnx-dir", type=Path, default=Path("data/analysis/models/bge-reranker-v2-m3-en-ru-onnx"))
    parser.add_argument("--gguf-model", type=Path, default=Path("data/analysis/models/bge-reranker-v2-m3-gguf-q8/bge-reranker-v2-m3-q8_0.gguf"))
    parser.add_argument("--llama-server", type=Path, default=Path("/tmp/llama.cpp-zakupcheck/build/bin/llama-server"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--port", type=int, default=18082)
    args = parser.parse_args(argv)

    chunk_list = load_chunks(chunks_path(args.edition_dir, 700))
    chunks = {chunk.chunk_id: chunk for chunk in chunk_list}
    fingerprint = corpus_fingerprint(chunk_list)
    id_map = legacy_score_id_map(chunk_list)
    cases = {
        case.case_id: case
        for case in load_verified(args.cases_dir)
        if case.split == "dev"
    }
    reference = json.loads(args.reference.read_text(encoding="utf-8"))["splits"]["dev"]

    repairers: dict[str, Callable[..., dict[str, dict[str, float]]]] = {
        "onnx-int8": lambda **kwargs: repair_onnx_int8(
            **kwargs, onnx_dir=args.onnx_dir, threads=args.threads
        ),
        "gguf-q8": lambda **kwargs: repair_gguf(
            **kwargs,
            llama_server=args.llama_server,
            gguf_model=args.gguf_model,
            threads=args.threads,
            port=args.port,
            log_path=args.runtime_dir / "llama-server-refresh.log",
        ),
    }
    for path in sorted(args.runtime_dir.glob("*.json")):
        state: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if state.get("corpus_fingerprint") not in (None, fingerprint):
            raise ValueError(f"unknown corpus fingerprint in {path}")
        if state.get("corpus_fingerprint") is None:
            state["scores"] = migrate_scores(state["scores"], id_map)
        missing = missing_pairs(state["scores"], reference)
        if missing:
            runtime = state["runtime"]
            if runtime not in repairers:
                raise ValueError(f"{runtime} has {sum(map(len, missing.values()))} missing pairs")
            additions = repairers[runtime](cases=cases, chunks=chunks, missing=missing)
            for case_id, values in additions.items():
                state["scores"][case_id].update(values)
        remaining = missing_pairs(state["scores"], reference)
        if remaining:
            raise AssertionError(f"unrepaired pairs in {path}: {remaining}")
        state["corpus_fingerprint"] = fingerprint
        state.setdefault("metadata", {})["coverage_refresh"] = {
            "migrated_sequential_ids": True,
            "new_pairs_scored": sum(map(len, missing.values())),
        }
        path.write_text(json.dumps(state, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{state['runtime']}: repaired {sum(map(len, missing.values()))} pairs", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
