#!/bin/bash
# Replay final-answer eval (Qwen3-4B) for each run folder under medical-scope/output.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-/home/hyang/mediQ}"
PYTHON="${PYTHON:-/home/hyang/miniconda3/envs/scope/bin/python}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$REPO_DIR/scripts/one_shot_full_context_vllm_eval.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output}"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-4B}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
SEED="${SEED:-0}"
NUM_SEEDS="${NUM_SEEDS:-5}"
TEMP="${TEMP:-0.8}"
TOP_P="${TOP_P:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_DEVICES="${GPU_DEVICES:-2}"
TAG="${TAG:-test_eval_qwen3_4b_hf_replay}"

if [[ -n "$GPU_DEVICES" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
fi

run_folder() {
  local folder="$1"
  local results_file="$2"
  local out_jsonl="$folder/${TAG}.jsonl"

  if [[ ! -f "$results_file" ]]; then
    echo "SKIP $folder — missing $results_file" >&2
    return 0
  fi

  echo
  echo "=== ${QWEN_MODEL} :: replay :: $(basename "$folder") ==="
  echo "RESULTS: $results_file"
  echo "Output:  $out_jsonl"

  PYTHONPATH="$REPO_DIR/medical-scope:$REPO_DIR/src:${PYTHONPATH:-}" \
  MEDIQ_ENABLE_THINKING=0 \
  "$PYTHON" "$EVAL_SCRIPT" \
    --rows-from-results "$results_file" \
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

run_one() {
  local folder="$1"
  results=()
  while IFS= read -r f; do
    results+=("$f")
  done < <(find "$folder" -maxdepth 1 -name '*_results.jsonl' | sort)
  if [[ ${#results[@]} -ne 1 ]]; then
    echo "SKIP $(basename "$folder") — expected 1 *_results.jsonl, found ${#results[@]}" >&2
    return 0
  fi
  run_folder "$folder" "${results[0]}"
}

# Pass a single run folder as $1, or default to all subfolders under OUTPUT_ROOT.
if [[ -n "${1:-}" ]]; then
  run_one "$1"
else
  for folder in "$OUTPUT_ROOT"/*/; do
    [[ -d "$folder" ]] || continue
    run_one "$folder"
  done
fi

echo
echo "Done. Replay outputs under: $OUTPUT_ROOT/*/${TAG}.jsonl"
