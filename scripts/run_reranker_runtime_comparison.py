#!/usr/bin/env python3
"""Compare PyTorch, ONNX and GGUF reranker runtimes on identical pairs."""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from index.bm25 import load_chunks  # noqa: E402
from index.error_analysis import aggregate_rankings, paired_group_bootstrap  # noqa: E402
from index.eval_tables import chunks_path, load_verified, query_text  # noqa: E402
from index.rerank import SentenceTransformerCrossEncoder  # noqa: E402
from index.rerank_comparison import (  # noqa: E402
    article_evidence_ranking,
    article_ranking,
    ranked_ids,
    spearman_rank,
    top_overlap,
)
from schemas.corpus import Chunk  # noqa: E402
from schemas.eval_case import EvalCase  # noqa: E402

REFERENCE_MODEL = "BAAI/bge-reranker-v2-m3"
TRUNCATED_MODEL = "qilowoq/bge-reranker-v2-m3-en-ru"
ONNX_MODEL = "metalmon80/bge-reranker-v2-m3-en-ru-onnx"
GGUF_MODEL = "Felladrin/gguf-Q8_0-bge-reranker-v2-m3"
STATE_VERSION = 1


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _peak_rss_mb() -> float:
    # Linux reports ru_maxrss in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _paired_latency_interval(
    reference_ms: list[float],
    candidate_ms: list[float],
    *,
    seed: int = 20261009,
    samples: int = 100_000,
) -> dict[str, float | int]:
    """Bootstrap total-throughput speedup over identical query batches."""
    if len(reference_ms) != len(candidate_ms) or not reference_ms:
        raise ValueError("paired non-empty latency samples are required")
    left = np.asarray(reference_ms, dtype=np.float64)
    right = np.asarray(candidate_ms, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(left), size=(samples, len(left)))
    ratios = left[indices].sum(axis=1) / right[indices].sum(axis=1)
    return {
        "n": len(left),
        "speedup": float(left.sum() / right.sum()),
        "median_paired_speedup": float(np.median(left / right)),
        "wins": int(np.sum(right < left)),
        "low": float(np.quantile(ratios, 0.025)),
        "high": float(np.quantile(ratios, 0.975)),
    }


def _load_dataset(
    cases_dir: Path,
    edition_dir: Path,
    reference_path: Path,
    *,
    split: str,
) -> tuple[list[EvalCase], dict[str, Chunk], dict[str, dict[str, float]]]:
    cases = [case for case in load_verified(cases_dir) if case.split == split]
    chunks = {
        chunk.chunk_id: chunk
        for chunk in load_chunks(chunks_path(edition_dir, 700))
    }
    cache = json.loads(reference_path.read_text(encoding="utf-8"))
    if cache.get("model") != REFERENCE_MODEL:
        raise ValueError(f"unexpected reference model: {cache.get('model')}")
    split_cache = cache["splits"][split]
    scores: dict[str, dict[str, float]] = {}
    for case in cases:
        case_scores = {
            chunk_id: float(score)
            for chunk_id, score in split_cache[case.case_id].items()
        }
        missing = set(case_scores) - set(chunks)
        if missing:
            raise ValueError(f"{case.case_id}: unknown chunks: {sorted(missing)[:3]}")
        scores[case.case_id] = case_scores
    return cases, chunks, scores


def _result_path(output_dir: Path, runtime: str) -> Path:
    return output_dir / f"{runtime}.json"


def _write_runtime(
    path: Path,
    *,
    runtime: str,
    model: str,
    scores: dict[str, dict[str, float]],
    case_ms: dict[str, float],
    load_ms: float,
    model_bytes: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "version": STATE_VERSION,
        "runtime": runtime,
        "model": model,
        "scores": scores,
        "case_ms": case_ms,
        "load_ms": load_ms,
        "peak_rss_mb": _peak_rss_mb(),
        "model_bytes": model_bytes,
        "metadata": metadata or {},
    }
    path.write_text(json.dumps(state, ensure_ascii=False) + "\n", encoding="utf-8")


def _token_parity(
    cases: list[EvalCase],
    chunks: dict[str, Chunk],
    reference: dict[str, dict[str, float]],
    candidate_tokenizer: Any,
    original_dir: Path,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    original = AutoTokenizer.from_pretrained(original_dir, local_files_only=True)
    exact_pairs = 0
    total = 0
    candidate_unknown = 0
    candidate_tokens = 0
    mismatches: list[dict[str, Any]] = []
    for case in cases:
        query = query_text(case, "blind")
        original_query = original.tokenize(query)
        candidate_query = candidate_tokenizer.tokenize(query)
        for chunk_id in reference[case.case_id]:
            text = chunks[chunk_id].text
            original_tokens = original_query + ["</s>", "</s>"] + original.tokenize(text)
            candidate_tokens_list = (
                candidate_query + ["</s>", "</s>"] + candidate_tokenizer.tokenize(text)
            )
            same = original_tokens[:510] == candidate_tokens_list[:510]
            exact_pairs += int(same)
            total += 1
            candidate_unknown += sum(token == "<unk>" for token in candidate_tokens_list[:510])
            candidate_tokens += min(510, len(candidate_tokens_list))
            if not same and len(mismatches) < 10:
                mismatches.append({"case_id": case.case_id, "chunk_id": chunk_id})
    return {
        "pairs": total,
        "exact_token_sequence_fraction": exact_pairs / total,
        "unknown_token_fraction": candidate_unknown / max(1, candidate_tokens),
        "mismatch_examples": mismatches,
    }


def _score_onnx(args: argparse.Namespace) -> None:
    import onnxruntime as ort
    from transformers import AutoTokenizer

    cases, chunks, reference = _load_dataset(
        args.cases_dir, args.edition_dir, args.reference_scores, split=args.split
    )
    model_path = args.onnx_dir / ("model.onnx" if args.runtime == "onnx-fp32" else "model.int8.onnx")
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.onnx_dir, local_files_only=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    load_ms = (time.perf_counter() - started) * 1000
    input_names = {item.name for item in session.get_inputs()}
    token_parity = _token_parity(cases, chunks, reference, tokenizer, args.pytorch_dir)

    def score_case(query: str, documents: list[str]) -> list[float]:
        values: list[float] = []
        for start in range(0, len(documents), args.batch_size):
            batch = documents[start : start + args.batch_size]
            encoded = tokenizer(
                [query] * len(batch),
                batch,
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
            logits = session.run(None, inputs)[0]
            values.extend(np.asarray(logits, dtype=np.float32).reshape(-1).tolist())
        return values

    first_case = cases[0]
    first_chunk_id = next(iter(reference[first_case.case_id]))
    score_case(query_text(first_case, "blind"), [chunks[first_chunk_id].text])

    scores: dict[str, dict[str, float]] = {}
    case_ms: dict[str, float] = {}
    score_cases = cases[: args.quality_cases] if args.quality_cases else cases
    for index, case in enumerate(score_cases, start=1):
        chunk_ids = sorted(reference[case.case_id])
        documents = [chunks[chunk_id].text for chunk_id in chunk_ids]
        started = time.perf_counter()
        values = score_case(query_text(case, "blind"), documents)
        case_ms[case.case_id] = (time.perf_counter() - started) * 1000
        scores[case.case_id] = dict(zip(chunk_ids, values, strict=True))
        print(f"{args.runtime}: {index}/{len(score_cases)}", file=sys.stderr)

    metadata = {
        "threads": args.threads,
        "batch_size": args.batch_size,
        "max_length": 512,
        "quality_cases": len(score_cases),
        "token_parity": token_parity,
        "onnxruntime": ort.__version__,
    }
    model_bytes = model_path.stat().st_size
    data_path = model_path.with_name(model_path.name + ".data")
    if data_path.is_file():
        model_bytes += data_path.stat().st_size
    _write_runtime(
        _result_path(args.output_dir, args.runtime),
        runtime=args.runtime,
        model=ONNX_MODEL,
        scores=scores,
        case_ms=case_ms,
        load_ms=load_ms,
        model_bytes=model_bytes,
        metadata=metadata,
    )


def _score_pytorch(args: argparse.Namespace) -> None:
    import torch

    cases, chunks, reference = _load_dataset(
        args.cases_dir, args.edition_dir, args.reference_scores, split=args.split
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    is_truncated = args.runtime == "pytorch-fp32-en-ru"
    runtime = "pytorch-fp32-en-ru" if is_truncated else "pytorch-fp32"
    model_name = TRUNCATED_MODEL if is_truncated else REFERENCE_MODEL
    model_dir = args.pytorch_en_ru_dir if is_truncated else args.pytorch_dir
    started = time.perf_counter()
    scorer = SentenceTransformerCrossEncoder(
        str(model_dir),
        max_length=512,
        batch_size=args.batch_size,
    )
    load_ms = (time.perf_counter() - started) * 1000
    first_case = cases[0]
    first_chunk_id = next(iter(reference[first_case.case_id]))
    scorer.score(query_text(first_case, "blind"), [chunks[first_chunk_id]])
    scores: dict[str, dict[str, float]] = {}
    case_ms: dict[str, float] = {}
    for index, case in enumerate(cases[: args.latency_cases], start=1):
        chunk_ids = sorted(reference[case.case_id])
        documents = [chunks[chunk_id] for chunk_id in chunk_ids]
        started = time.perf_counter()
        values = scorer.score(query_text(case, "blind"), documents)
        case_ms[case.case_id] = (time.perf_counter() - started) * 1000
        scores[case.case_id] = dict(zip(chunk_ids, values, strict=True))
        print(f"{runtime} latency: {index}/{args.latency_cases}", file=sys.stderr)
    model_file = model_dir / "model.safetensors"
    metadata: dict[str, Any] = {
        "threads": args.threads,
        "batch_size": args.batch_size,
        "max_length": 512,
        "quality_scores": (
            "fresh scores cover latency subset"
            if is_truncated
            else "reference cache; fresh scores cover latency subset"
        ),
        "torch": torch.__version__,
    }
    if is_truncated:
        metadata["token_parity"] = _token_parity(
            cases,
            chunks,
            reference,
            scorer.model.tokenizer,
            args.pytorch_dir,
        )
    _write_runtime(
        _result_path(args.output_dir, runtime),
        runtime=runtime,
        model=model_name,
        scores=scores,
        case_ms=case_ms,
        load_ms=load_ms,
        model_bytes=model_file.stat().st_size,
        metadata=metadata,
    )


def _request_json(url: str, payload: dict[str, Any] | None = None) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_for_server(process: subprocess.Popen[str], base_url: str) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with {process.returncode}")
        try:
            _request_json(f"{base_url}/health")
            return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            time.sleep(0.25)
    raise TimeoutError("llama-server did not become healthy")


def _child_peak_rss_mb(pid: int) -> float:
    status = Path(f"/proc/{pid}/status")
    if not status.is_file():
        return 0.0
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmHWM:"):
            return float(line.split()[1]) / 1024.0
    return 0.0


def _score_gguf(args: argparse.Namespace) -> None:
    cases, chunks, reference = _load_dataset(
        args.cases_dir, args.edition_dir, args.reference_scores, split=args.split
    )
    base_url = f"http://127.0.0.1:{args.port}"
    command = [
        str(args.llama_server),
        "--model",
        str(args.gguf_model),
        "--embedding",
        "--reranking",
        "--pooling",
        "rank",
        "--ctx-size",
        "512",
        "--threads",
        str(args.threads),
        "--threads-batch",
        str(args.threads),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--no-webui",
    ]
    log_path = args.output_dir / "llama-server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        try:
            _wait_for_server(process, base_url)
            load_ms = (time.perf_counter() - started) * 1000
            scores: dict[str, dict[str, float]] = {}
            case_ms: dict[str, float] = {}
            for index, case in enumerate(cases, start=1):
                chunk_ids = sorted(reference[case.case_id])
                documents = [chunks[chunk_id].text for chunk_id in chunk_ids]
                started = time.perf_counter()
                response = _request_json(
                    f"{base_url}/v1/rerank",
                    {
                        "model": GGUF_MODEL,
                        "query": query_text(case, "blind"),
                        "documents": documents,
                        "top_n": len(documents),
                    },
                )
                case_ms[case.case_id] = (time.perf_counter() - started) * 1000
                values = {int(item["index"]): float(item["relevance_score"]) for item in response["results"]}
                scores[case.case_id] = {
                    chunk_id: values[position]
                    for position, chunk_id in enumerate(chunk_ids)
                }
                print(f"gguf-q8: {index}/{len(cases)}", file=sys.stderr)
            peak_rss = _child_peak_rss_mb(process.pid)
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    _write_runtime(
        _result_path(args.output_dir, "gguf-q8"),
        runtime="gguf-q8",
        model=GGUF_MODEL,
        scores=scores,
        case_ms=case_ms,
        load_ms=load_ms,
        model_bytes=args.gguf_model.stat().st_size,
        metadata={
            "threads": args.threads,
            "ctx_size": 512,
            "llama_cpp_commit": args.llama_server_commit,
            "server_peak_rss_mb": peak_rss,
        },
    )


def _runtime_quality(
    cases: list[EvalCase],
    chunks: dict[str, Chunk],
    reference: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
) -> dict[str, Any]:
    spearman: list[float] = []
    chunk_overlap: list[float] = []
    chunk_top1: list[bool] = []
    article_overlap: list[float] = []
    article_top1: list[bool] = []
    ref_rankings: list[list[Chunk]] = []
    candidate_rankings: list[list[Chunk]] = []
    used_cases: list[EvalCase] = []
    score_differences: list[float] = []
    for case in cases:
        if case.case_id not in candidate:
            continue
        left_scores = reference[case.case_id]
        right_scores = candidate[case.case_id]
        if set(left_scores) != set(right_scores):
            raise ValueError(f"{case.case_id}: candidate ids differ from reference")
        score_differences.extend(
            right_scores[chunk_id] - left_scores[chunk_id]
            for chunk_id in left_scores
        )
        left = ranked_ids(left_scores)
        right = ranked_ids(right_scores)
        spearman.append(spearman_rank(left, right))
        chunk_overlap.append(top_overlap(left, right, k=10))
        chunk_top1.append(left[0] == right[0])
        left_articles, _ = article_ranking(chunks, left_scores)
        right_articles, _ = article_ranking(chunks, right_scores)
        article_overlap.append(top_overlap(left_articles, right_articles, k=5))
        article_top1.append(left_articles[0] == right_articles[0])
        ref_rankings.append(article_evidence_ranking(chunks, left_scores))
        candidate_rankings.append(article_evidence_ranking(chunks, right_scores))
        used_cases.append(case)
    ref5 = aggregate_rankings(used_cases, ref_rankings, k=5)
    candidate5 = aggregate_rankings(used_cases, candidate_rankings, k=5)
    ref10 = aggregate_rankings(used_cases, ref_rankings, k=10)
    candidate10 = aggregate_rankings(used_cases, candidate_rankings, k=10)
    interval = paired_group_bootstrap(
        [
            aggregate_rankings([case], [ranking], k=5).recall_article
            for case, ranking in zip(used_cases, ref_rankings, strict=True)
        ],
        [
            aggregate_rankings([case], [ranking], k=5).recall_article
            for case, ranking in zip(used_cases, candidate_rankings, strict=True)
        ],
        [case.case_group_id for case in used_cases],
        seed=20261008,
    )
    return {
        "n": len(used_cases),
        "mean_spearman": statistics.fmean(spearman),
        "mean_chunk_top10_overlap": statistics.fmean(chunk_overlap),
        "chunk_top1_agreement": statistics.fmean(chunk_top1),
        "mean_article_top5_overlap": statistics.fmean(article_overlap),
        "article_top1_agreement": statistics.fmean(article_top1),
        "reference_at5": asdict(ref5),
        "candidate_at5": asdict(candidate5),
        "reference_at10": asdict(ref10),
        "candidate_at10": asdict(candidate10),
        "article_r5_bootstrap": asdict(interval),
        "score_mae": statistics.fmean(abs(value) for value in score_differences),
        "score_max_abs": max(abs(value) for value in score_differences),
    }


def _report(args: argparse.Namespace) -> None:
    cases, chunks, reference = _load_dataset(
        args.cases_dir, args.edition_dir, args.reference_scores, split=args.split
    )
    states: list[dict[str, Any]] = []
    for runtime in (
        "pytorch-fp32",
        "pytorch-fp32-en-ru",
        "onnx-fp32",
        "onnx-int8",
        "gguf-q8",
    ):
        path = _result_path(args.output_dir, runtime)
        if not path.is_file():
            raise FileNotFoundError(path)
        states.append(json.loads(path.read_text(encoding="utf-8")))

    rows: list[dict[str, Any]] = []
    for state in states:
        runtime = state["runtime"]
        quality_scores = reference if runtime == "pytorch-fp32" else state["scores"]
        quality = _runtime_quality(cases, chunks, reference, quality_scores)
        latency_case_ids = [case.case_id for case in cases[: args.latency_cases]]
        latencies = [state["case_ms"][case_id] for case_id in latency_case_ids]
        pair_count = sum(len(reference[case_id]) for case_id in latency_case_ids)
        peak_rss = float(
            state.get("metadata", {}).get("server_peak_rss_mb", state["peak_rss_mb"])
        )
        rows.append(
            {
                "runtime": runtime,
                "model": state["model"],
                "quality": quality,
                "latency": {
                    "n": len(latencies),
                    "p50_ms": _percentile(latencies, 50),
                    "p95_ms": _percentile(latencies, 95),
                    "mean_ms": statistics.fmean(latencies),
                    "pairs_per_second": pair_count / (sum(latencies) / 1000),
                    "load_ms": state["load_ms"],
                },
                "peak_rss_mb": peak_rss,
                "model_mb": state["model_bytes"] / 1_000_000,
                "metadata": state.get("metadata", {}),
            }
        )

    reference_row = rows[0]
    ref_latency = reference_row["latency"]["mean_ms"]
    for row in rows:
        row["speedup_vs_pytorch"] = ref_latency / row["latency"]["mean_ms"]

    truncated_row = next(row for row in rows if row["runtime"] == "pytorch-fp32-en-ru")
    reference_state = states[0]
    truncated_state = next(
        state for state in states if state["runtime"] == "pytorch-fp32-en-ru"
    )
    latency_case_ids = [case.case_id for case in cases[: args.latency_cases]]
    truncated_latency = _paired_latency_interval(
        [reference_state["case_ms"][case_id] for case_id in latency_case_ids],
        [truncated_state["case_ms"][case_id] for case_id in latency_case_ids],
    )
    truncated_row["paired_latency"] = truncated_latency

    summary = {
        "version": STATE_VERSION,
        "split": args.split,
        "pairs": sum(len(scores) for scores in reference.values()),
        "latency_cases": args.latency_cases,
        "rows": rows,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Сравнение runtimes для bge-reranker-v2-m3",
        "",
        f"INT8 и GGUF измерены на {len(cases)} dev/blind кейсах и {summary['pairs']} одинаковых "
        "парах «запрос—m700-чанк». Bilingual PyTorch и ONNX FP32 проверены на 10 кейсах "
        "как контроль эквивалентности; PyTorch FP32 — эталонные scores из первичного прогона. "
        f"Latency измерена свежим запуском на первых {args.latency_cases} кейсах, 8 физических CPU-ядер, "
        "max_length/context 512.",
        "",
        "Текущий пул содержит до 10 статей на запрос. Поэтому чувствительная article-метрика здесь — "
        "R@5 и overlap top-5; article R@10 почти фиксирован candidate generation.",
        "",
        "## Итог",
        "",
        "| runtime | quality n | model MB | peak RSS MB | load s | p50 ms/query | p95 ms/query | pairs/s | speedup | Spearman | chunk top-10 overlap | article top-5 overlap | exact R@10 | article R@5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        quality = row["quality"]
        lines.append(
            f"| {row['runtime']} | {quality['n']} | {row['model_mb']:.0f} | {row['peak_rss_mb']:.0f} | "
            f"{row['latency']['load_ms'] / 1000:.2f} | "
            f"{row['latency']['p50_ms']:.0f} | {row['latency']['p95_ms']:.0f} | "
            f"{row['latency']['pairs_per_second']:.1f} | {row['speedup_vs_pytorch']:.2f}x | "
            f"{quality['mean_spearman']:.4f} | {quality['mean_chunk_top10_overlap']:.3f} | "
            f"{quality['mean_article_top5_overlap']:.3f} | "
            f"{quality['candidate_at10']['recall_exact']:.3f} | "
            f"{quality['candidate_at5']['recall_article']:.3f} |"
        )
    lines.extend(["", "## Fidelity к PyTorch FP32", ""])
    for row in rows[1:]:
        quality = row["quality"]
        interval = quality["article_r5_bootstrap"]
        lines.append(
            f"- `{row['runtime']}`: chunk top-1 {quality['chunk_top1_agreement']:.3f}, "
            f"article top-1 {quality['article_top1_agreement']:.3f}; Δ article R@5 "
            f"{interval['mean_delta']:+.3f}, 95% CI [{interval['low']:+.3f}; {interval['high']:+.3f}]."
        )
    truncated_quality = truncated_row["quality"]
    truncated_token = truncated_row["metadata"].get("token_parity")
    onnx_row = next(row for row in rows if row["runtime"] == "onnx-fp32")
    onnx_token = onnx_row["metadata"].get("token_parity")
    if truncated_token:
        lines.extend(
            [
                "",
                "## Прямое сравнение PyTorch FP32",
                "",
                f"Truncated en-ru: speedup {truncated_row['speedup_vs_pytorch']:.2f}x, "
                f"парный bootstrap 95% CI "
                f"[{truncated_latency['low']:.3f}x; {truncated_latency['high']:.3f}x], "
                f"быстрее в {truncated_latency['wins']} из {truncated_latency['n']} кейсов. "
                f"model {truncated_row['model_mb']:.0f} MB против {reference_row['model_mb']:.0f} MB, "
                f"peak RSS {truncated_row['peak_rss_mb']:.0f} MB против "
                f"{reference_row['peak_rss_mb']:.0f} MB. Fidelity на 10 одинаковых кейсах: "
                f"Spearman {truncated_quality['mean_spearman']:.4f}, chunk top-10 overlap "
                f"{truncated_quality['mean_chunk_top10_overlap']:.3f}, article top-5 overlap "
                f"{truncated_quality['mean_article_top5_overlap']:.3f}; MAE logits "
                f"{truncated_quality['score_mae']:.6f}, max abs "
                f"{truncated_quality['score_max_abs']:.6f}.",
                "",
                f"Tokenizer en-ru полностью совпал с оригинальным на "
                f"{truncated_token['exact_token_sequence_fraction']:.2%} пар; `<unk>`: "
                f"{truncated_token['unknown_token_fraction']:.6f}.",
            ]
        )
    if onnx_token:
        lines.extend(
            [
                "",
                "## Tokenizer parity en-ru",
                "",
                f"Доля пар с полностью совпадающей последовательностью токенов: "
                f"{onnx_token['exact_token_sequence_fraction']:.4f}; `<unk>`: "
                f"{onnx_token['unknown_token_fraction']:.6f}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Решение",
            "",
            "Критерий допуска: Δ article R@5 не хуже -0.02, chunk top-10 overlap ≥0.95, "
            "article top-5 overlap ≥0.95 и измеримое ускорение. Победитель определяется по этим "
            "ограничениям, затем по latency и памяти.",
            "",
            "**Ни ONNX INT8, ни GGUF Q8_0 критерий fidelity не прошли.** Их актуальные "
            "speedup, overlap и retrieval-метрики приведены в таблице выше; пересчёт speedup "
            "использует тот же свежий PyTorch baseline.",
            "",
            "Bilingual PyTorch FP32 уменьшает файл весов примерно на треть и ускоряет cold start. "
            "После этого контроля зафиксированная article-архитектура была повторена на 52 dev-кейсах: "
            "BF16 и FP32 дали одинаковые article/exact R@10 0.731/0.202, а ordered top-10 совпал в 38/52 кейсах. "
            "На длинном CPU-прогоне BF16 быстрее: 6.44 против 2.52 пары/с, или около 14.0 против 35.7 с/запрос. "
            "Поэтому BF16 — CPU deployment-кандидат, FP32 — контрольный эталон. Bilingual ONNX FP32 на 10 кейсах "
            "практически эквивалентен оригиналу, но медленнее PyTorch. INT8 и GGUF Q8_0 остаются "
            "явным quality–speed trade-off, а не эквивалентной заменой.",
            "",
            "GGUF-прогон одновременно меняет квантование и runtime/tokenization, поэтому расхождение "
            "нельзя приписать только Q8_0 без FP16/FP32 GGUF-контроля.",
            "",
            "Полные scores и профили хранятся локально в `data/analysis/reranker_runtime/`; "
            "этот каталог gitignored.",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    main_path = args.output_dir.parent / "retrieval_error_state.json"
    if main_path.is_file():
        main = json.loads(main_path.read_text(encoding="utf-8"))
        main["reranker_runtime_comparison"] = {**summary, "report": str(args.report)}
        main_path.write_text(json.dumps(main, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        from run_retrieval_error_analysis import _report as write_aggregate_report

        write_aggregate_report(main, args.aggregate_report)
    print(f"wrote {args.report}", file=sys.stderr)
    print(f"wrote {args.summary}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare reranker runtimes")
    parser.add_argument("action", choices=("score", "report"))
    parser.add_argument(
        "--runtime",
        choices=(
            "pytorch-fp32",
            "pytorch-fp32-en-ru",
            "onnx-fp32",
            "onnx-int8",
            "gguf-q8",
        ),
    )
    parser.add_argument("--split", default="dev", choices=("dev",))
    parser.add_argument("--cases-dir", type=Path, default=Path("evals/cases"))
    parser.add_argument("--edition-dir", type=Path, default=Path("data/law/2026-08-04"))
    parser.add_argument(
        "--reference-scores",
        type=Path,
        default=Path("data/analysis/cross_encoder_scores.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analysis/reranker_runtime"),
    )
    parser.add_argument(
        "--pytorch-dir",
        type=Path,
        default=Path("data/analysis/models/bge-reranker-v2-m3-pytorch"),
    )
    parser.add_argument(
        "--pytorch-en-ru-dir",
        type=Path,
        default=Path("data/analysis/models/bge-reranker-v2-m3-en-ru-pytorch"),
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=Path("data/analysis/models/bge-reranker-v2-m3-en-ru-onnx"),
    )
    parser.add_argument(
        "--gguf-model",
        type=Path,
        default=Path("data/analysis/models/bge-reranker-v2-m3-gguf-q8/bge-reranker-v2-m3-q8_0.gguf"),
    )
    parser.add_argument(
        "--llama-server",
        type=Path,
        default=Path("/tmp/llama.cpp-zakupcheck/build/bin/llama-server"),
    )
    parser.add_argument("--llama-server-commit", default="de7fa0a")
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--latency-cases", type=int, default=10)
    parser.add_argument(
        "--quality-cases",
        type=int,
        default=0,
        help="score only the first N cases; zero means the complete split",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evals/reports/reranker_runtime_comparison.md"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/analysis/reranker_runtime_summary.json"),
    )
    parser.add_argument(
        "--aggregate-report",
        type=Path,
        default=Path("evals/reports/retrieval_error_analysis.md"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "report":
        _report(args)
        return 0
    if args.runtime is None:
        print("--runtime is required for score", file=sys.stderr)
        return 2
    if args.runtime.startswith("pytorch-fp32"):
        _score_pytorch(args)
    elif args.runtime.startswith("onnx-"):
        _score_onnx(args)
    else:
        _score_gguf(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
