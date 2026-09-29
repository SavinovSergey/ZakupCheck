#!/usr/bin/env bash
# Wrapper around llama-bench. Binary lives outside this repo (see .env.example).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  # shellcheck disable=SC1091
  set -a && source "$ROOT/.env" && set +a
fi

BENCH_BIN="${LLAMA_BENCH_BIN:-${LLAMA_CPP_DIR:-$HOME/llama.cpp}/build/bin/llama-bench}"
MODEL="${MODEL_PATH:-$HOME/models/t-lite/T-lite-it-2.1-Q5_K_M.gguf}"
OUT_DIR="$ROOT/evals/reports/llama-bench"
mkdir -p "$OUT_DIR"

if [[ ! -x "$BENCH_BIN" ]]; then
  echo "llama-bench not found: $BENCH_BIN" >&2
  echo "Build llama.cpp outside the repo and set LLAMA_BENCH_BIN in .env" >&2
  exit 1
fi

NGL="${NGL:-99}"
TAG="${TAG:-run}"
OUT="$OUT_DIR/bench_${TAG}.md"

"$BENCH_BIN" -m "$MODEL" -ngl "$NGL" -p 512,2048 -n 128 -r 5 -o md | tee "$OUT"
echo "Wrote $OUT"
