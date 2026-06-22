#!/bin/bash
# One-shot eval on shadow min/max fact cuts (med-scope-diverse by default).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-/home/hyang/mediQ}"
PYTHON="${PYTHON:-/home/hyang/miniconda3/envs/scope/bin/python}"
CUT_SCRIPT="${CUT_SCRIPT:-$REPO_DIR/scripts/shadow_fact_cut_rows.py}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$REPO_DIR/scripts/one_shot_full_context_vllm_eval.py}"

RUN_FOLDER="${RUN_FOLDER:-$SCRIPT_DIR/output/med-scope-diverse}"
MAX_EXAMPLES="${MAX_EXAMPLES:-175}"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-4B}"
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

run_cut_eval() {
  local mode="$1"
  local rows_jsonl="$RUN_FOLDER/shadow_fact_${mode}_eval_rows.jsonl"
  local out_jsonl="$RUN_FOLDER/test_eval_qwen3_4b_shadow_fact_${mode}_hf.jsonl"

  echo
  echo "=== shadow fact ${mode} :: build rows ==="
  PYTHONPATH="$REPO_DIR/scripts:$REPO_DIR/src:${PYTHONPATH:-}" \
  "$PYTHON" "$CUT_SCRIPT" \
    --run-folder "$RUN_FOLDER" \
    --cut-mode "$mode" \
    --output "$rows_jsonl" \
    --max-examples "$MAX_EXAMPLES"

  echo
  echo "=== ${QWEN_MODEL} :: one-shot shadow_fact_${mode} ==="
  echo "Rows:   $rows_jsonl"
  echo "Output: $out_jsonl"

  PYTHONPATH="$REPO_DIR/medical-scope:$REPO_DIR/src:${PYTHONPATH:-}" \
  MEDIQ_ENABLE_THINKING=0 \
  "$PYTHON" "$EVAL_SCRIPT" \
    --rows-jsonl "$rows_jsonl" \
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

run_cut_eval min
run_cut_eval max

echo
echo "Done."
echo "  $RUN_FOLDER/shadow_fact_min_eval_rows.jsonl"
echo "  $RUN_FOLDER/test_eval_qwen3_4b_shadow_fact_min_hf.jsonl"
echo "  $RUN_FOLDER/shadow_fact_max_eval_rows.jsonl"
echo "  $RUN_FOLDER/test_eval_qwen3_4b_shadow_fact_max_hf.jsonl"
