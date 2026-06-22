#!/bin/bash
set -euo pipefail

ROOT="${ROOT:-/home/hyang/mediQ}"
PYTHON="${PYTHON:-/home/hyang/anaconda3/envs/scope/bin/python}"
OUT_DIR="${OUT_DIR:-$ROOT/code-scope/output/thinking_baseline}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CODE_SCOPE_GENERATION_DEVICE="${CODE_SCOPE_GENERATION_DEVICE:-cuda:0}"
export CODE_SCOPE_ENABLE_THINKING=1
export CODE_SCOPE_MAX_NEW_TOKENS="${CODE_SCOPE_MAX_NEW_TOKENS:-16384}"

mkdir -p "$OUT_DIR"
cd "$ROOT"

"$PYTHON" code-scope/evaluate_humaneval.py \
  --mode baseline \
  --output "$OUT_DIR/qwen3_4b_baseline_thinking.jsonl" \
  "$@"
