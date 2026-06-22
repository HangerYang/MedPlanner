#!/bin/bash
set -euo pipefail

# Exact final-answer replays for both:
#   1. SCOPE-Medical results in new_outputs/med
#   2. CondensedScaleExpert results in new_outputs/condensed
# Each replay uses the saved/reconstructed final condensed patient information and
# calls mediQ's final_choice_with_options through a sampled HuggingFace helper path.

PYTHON=/home/hyang/miniconda3/envs/scope/bin/python
SCRIPT=/home/hyang/mediQ/scripts/one_shot_full_context_vllm_eval.py
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/home/hyang/mediQ/new_outputs}"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-4B}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
SEED="${SEED:-0}"
NUM_SEEDS="${NUM_SEEDS:-5}"

# Keep these equal to the original mediQ/SCOPE launch args. In the HuggingFace
# final-answer path, temperature/top_p are present in args but generation is
# sampled because this eval passes --do-sample and uses temperature/top_p.
TEMP="${TEMP:-0.8}"
TOP_P="${TOP_P:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TOP_LOGPROBS="${TOP_LOGPROBS:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_DEVICES="${GPU_DEVICES:-0,1}"

run_replay() {
  local label="$1"
  local results_file="$2"
  local out_dir="$3"
  local tag="$4"

  mkdir -p "$out_dir"

  if [[ ! -f "$results_file" ]]; then
    echo "Missing ${label} results file: $results_file" >&2
    return 1
  fi

  echo
  echo "=== ${QWEN_MODEL} :: exact ${label} final-answer replay ==="
  echo "RESULTS: ${results_file}"
  echo "Output: ${out_dir}/${tag}.jsonl"
  echo "temperature_arg=${TEMP} top_p_arg=${TOP_P} max_tokens=${MAX_TOKENS} max_examples=${MAX_EXAMPLES}"
  echo "seed=${SEED} num_seeds=${NUM_SEEDS}"
  echo "backend=HuggingFace mediQ helper, do_sample=True"

  PYTHONPATH="/home/hyang/mediQ/medical-scope:/home/hyang/mediQ/src:${PYTHONPATH:-}" \
  MEDIQ_ENABLE_THINKING=0 \
  "$PYTHON" "$SCRIPT" \
    --rows-from-results "$results_file" \
    --model "$QWEN_MODEL" \
    --output "$out_dir/${tag}.jsonl" \
    --max-examples "$MAX_EXAMPLES" \
    --temperature "$TEMP" \
    --top-p "$TOP_P" \
    --max-tokens "$MAX_TOKENS" \
    --top-logprobs "$TOP_LOGPROBS" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    --num-seeds "$NUM_SEEDS" \
    --do-sample
}

if [[ -n "$GPU_DEVICES" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
fi

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

run_replay \
  "scope-medical" \
  "${SCOPE_RESULTS:-$BASE_OUTPUT_DIR/med/medical_scope_qwen3_4b_results.jsonl}" \
  "${SCOPE_OUT_DIR:-$BASE_OUTPUT_DIR/med}" \
  "${SCOPE_TAG:-test_eval_qwen3_4b_scope_facts_hf_sample}"

run_replay \
  "condensed" \
  "${CONDENSED_RESULTS:-$BASE_OUTPUT_DIR/condensed/condensed_qwen3_4b_no_reasoning_results.jsonl}" \
  "${CONDENSED_OUT_DIR:-$BASE_OUTPUT_DIR/condensed}" \
  "${CONDENSED_TAG:-test_eval_qwen3_4b_condensed_facts_hf_sample}"
