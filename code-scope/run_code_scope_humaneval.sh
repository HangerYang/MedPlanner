#!/bin/bash
set -euo pipefail

ROOT="${ROOT:-/home/hyang/mediQ}"
PYTHON="${PYTHON:-/home/hyang/miniconda3/envs/scope/bin/python}"
OUT_DIR="${OUT_DIR:-$ROOT/new_outputs/code-scope}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CODE_SCOPE_GENERATION_DEVICE="${CODE_SCOPE_GENERATION_DEVICE:-cuda:0}"
export CODE_SCOPE_TRANSITION_DEVICE="${CODE_SCOPE_TRANSITION_DEVICE:-cuda:1}"
export CODE_SCOPE_REWARD_DEVICE="${CODE_SCOPE_REWARD_DEVICE:-cuda:1}"
export CODE_SCOPE_TRANSITION_DIR="${CODE_SCOPE_TRANSITION_DIR:-/home/hyang/mediQ/mediQ_model_files/code-moe}"
export CODE_SCOPE_REWARD_PATH="${CODE_SCOPE_REWARD_PATH:-/home/hyang/mediQ/mediQ_model_files/code_feedback_cumulative_reward_mlp.pt}"
export CODE_SCOPE_NUM_CANDIDATES="${CODE_SCOPE_NUM_CANDIDATES:-5}"
export CODE_SCOPE_PLANNING_ROUNDS="${CODE_SCOPE_PLANNING_ROUNDS:-10}"
export CODE_SCOPE_MCTS_TIME="${CODE_SCOPE_MCTS_TIME:-30}"

mkdir -p "$OUT_DIR"
cd "$ROOT"

"$PYTHON" code-scope/evaluate_humaneval.py \
  --mode scope \
  --output "$OUT_DIR/qwen3_4b_scope.jsonl" \
  "$@"
