#!/bin/bash
set -euo pipefail

ROOT="${ROOT:-/home/hyang/mediQ}"
PYTHON="${PYTHON:-/home/hyang/anaconda3/envs/scope/bin/python}"
OUT_DIR="${OUT_DIR:-$ROOT/code-scope/output/moe4_rerun}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CODE_SCOPE_GENERATION_DEVICE="${CODE_SCOPE_GENERATION_DEVICE:-cuda:0}"
export CODE_SCOPE_TRANSITION_DEVICE="${CODE_SCOPE_TRANSITION_DEVICE:-cuda:1}"
export CODE_SCOPE_REWARD_DEVICE="${CODE_SCOPE_REWARD_DEVICE:-cuda:1}"
export CODE_SCOPE_TRANSITION_DIR="${ROOT}/mediQ_model_files/code-moe"
export CODE_SCOPE_REWARD_PATH="${ROOT}/mediQ_model_files/code_feedback_cumulative_reward_mlp.pt"

# Entropy + trajectory logging: save per-rollout entropy, features, and pass/fail correlation
export CODE_SCOPE_ENTROPY_LOGGING="${CODE_SCOPE_ENTROPY_LOGGING:-1}"
export CODE_SCOPE_TRAJECTORY_JSONL="${CODE_SCOPE_TRAJECTORY_JSONL:-$OUT_DIR/qwen3_4b_scope_trajectory.jsonl}"

# Entropy-based early stopping during MCTS rollout simulation.
# Low-H: stop when H < 0.05 — saves ~75% of rollout compute (state converged, no new signal).
# High-H: stop when H > 1.5 — discard OOD-drifted rollouts (set to "" to disable).
export CODE_SCOPE_ROLLOUT_LOW_H="${CODE_SCOPE_ROLLOUT_LOW_H:-0.05}"
export CODE_SCOPE_ROLLOUT_HIGH_H="${CODE_SCOPE_ROLLOUT_HIGH_H:-1.5}"

# Task indices: 3 selector failures (oracle=True) + 2 all-candidates failures
TASK_IDS="5 6 10 37 38"

mkdir -p "$OUT_DIR"
cd "$ROOT"

"$PYTHON" code-scope/evaluate_humaneval.py \
  --mode scope \
  --output "$OUT_DIR/qwen3_4b_scope.jsonl" \
  --task-ids $TASK_IDS \
  "$@"
