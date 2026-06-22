#!/bin/bash
# One-shot final-answer eval: initial context vs full context (no interaction).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-/home/hyang/mediQ}"
PYTHON="${PYTHON:-/home/hyang/miniconda3/envs/scope/bin/python}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$REPO_DIR/scripts/one_shot_full_context_vllm_eval.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output}"
DATA_FILE="${DATA_FILE:-$REPO_DIR/data/med_data/all_test_convo_medqa.jsonl}"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-4B}"
MAX_EXAMPLES="${MAX_EXAMPLES:-175}"
SEED="${SEED:-0}"
NUM_SEEDS="${NUM_SEEDS:-5}"
TEMP="${TEMP:-0.8}"
TOP_P="${TOP_P:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_DEVICES="${GPU_DEVICES:-2}"

if [[ -n "$GPU_DEVICES" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
fi

run_baseline() {
  local mode="$1"
  local folder="$OUTPUT_ROOT/one-shot-${mode}-context"
  local tag="test_eval_qwen3_4b_${mode}_context_hf_replay"
  local out_jsonl="$folder/${tag}.jsonl"

  mkdir -p "$folder"

  echo
  echo "=== ${QWEN_MODEL} :: one-shot ${mode} context ==="
  echo "DATA:   $DATA_FILE"
  echo "Output: $out_jsonl"

  PYTHONPATH="$REPO_DIR/medical-scope:$REPO_DIR/src:${PYTHONPATH:-}" \
  MEDIQ_ENABLE_THINKING=0 \
  "$PYTHON" "$EVAL_SCRIPT" \
    --data "$DATA_FILE" \
    --context-mode "$mode" \
    --model "$QWEN_MODEL" \
    --output "$out_jsonl" \
    --max-examples "$MAX_EXAMPLES" \
    --temperature "$TEMP" \
    --top-p "$TOP_P" \
    --max-tokens "$MAX_TOKENS" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    --num-seeds "$NUM_SEEDS" \
    --do-sample
}

run_baseline initial
run_baseline full

echo
echo "Done."
echo "  $OUTPUT_ROOT/one-shot-initial-context/"
echo "  $OUTPUT_ROOT/one-shot-full-context/"
