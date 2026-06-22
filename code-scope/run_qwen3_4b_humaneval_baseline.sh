#!/bin/bash
set -euo pipefail

ROOT="${ROOT:-/home/hyang/mediQ}"
PYTHON="${PYTHON:-/home/hyang/miniconda3/envs/scope/bin/python}"
OUT_DIR="${OUT_DIR:-$ROOT/new_outputs/code-scope}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CODE_SCOPE_GENERATION_DEVICE="${CODE_SCOPE_GENERATION_DEVICE:-cuda:0}"

mkdir -p "$OUT_DIR"
cd "$ROOT"

"$PYTHON" code-scope/evaluate_humaneval.py \
  --mode baseline \
  --output "$OUT_DIR/qwen3_4b_baseline.jsonl" \
  "$@"
