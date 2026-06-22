#!/bin/bash
# MediQ benchmark -- CondensedScaleExpert, no reasoning, 2-GPU data-parallel.
# Confidence check uses full conv log; final answer uses initial_info + extracted facts only.

set -e

START_TIME=$(date +%s)

PYTHON=/home/hyang/miniconda3/envs/scope/bin/python
SRC=/home/hyang/mediQ/src
SCRIPTS=/home/hyang/mediQ/scripts
DATA=/home/hyang/mediQ/data/med_data
FILE="${FILE:-/home/hyang/mediQ/data/med_data/all_test_convo_medqa.jsonl}"
NEW_OUTPUTS="${NEW_OUTPUTS:-/home/hyang/mediQ/new_outputs/condensed}"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-4B}"
EXPERT_MODEL="${EXPERT_MODEL:-$QWEN_MODEL}"
PATIENT_MODEL="${PATIENT_MODEL:-$QWEN_MODEL}"

TAG="${TAG:-condensed_qwen3_4b_no_reasoning}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
TEMP="${TEMP:-0.8}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-5.0}"
MAX_QUESTIONS="${MAX_QUESTIONS:-10}"

NUM_SHARDS="${NUM_SHARDS:-2}"
SHARD_GPUS="${SHARD_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1}}"
TP_SIZE=1

export MEDIQ_ENABLE_THINKING=0
export MEDIQ_VLLM_GPU_MEMORY_UTILIZATION="${MEDIQ_VLLM_GPU_MEMORY_UTILIZATION:-0.8}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
BATCH_SIZE="${BATCH_SIZE:-4}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
MAX_TOKENS="${MAX_TOKENS:-2048}"

BENCH_RETRY_SLEEP="${BENCH_RETRY_SLEEP:-600}"
BENCH_MAX_ATTEMPTS="${BENCH_MAX_ATTEMPTS:-5}"

mkdir -p "$NEW_OUTPUTS"

IFS=',' read -r -a GPU_ARRAY <<< "$SHARD_GPUS"
if [[ "${#GPU_ARRAY[@]}" -lt "$NUM_SHARDS" ]]; then
  echo "[run_condensed.sh] FATAL: NUM_SHARDS=${NUM_SHARDS}, but SHARD_GPUS only has ${#GPU_ARRAY[@]} GPU ids: ${SHARD_GPUS}" >&2
  exit 1
fi

run_benchmark() {
  local tag="$1"
  local shard_idx="$2"
  local num_shards="$3"
  local gpu_id="$4"

  echo
  echo "=== CondensedScaleExpert :: ${tag} ==="
  echo "gpu=${gpu_id} shard=${shard_idx}/${num_shards} temp=${TEMP} threshold=${CONFIDENCE_THRESHOLD} max_questions=${MAX_QUESTIONS}"

  cd "$SRC" && CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON" mediQ_benchmark.py \
    --expert_class CondensedScaleExpert \
    --expert_module expert \
    --expert_model "$EXPERT_MODEL" \
    --expert_model_question_generator "$EXPERT_MODEL" \
    --patient_class FactSelectPatient \
    --patient_module patient \
    --patient_model "$PATIENT_MODEL" \
    --data_dir "$DATA" \
    --dev_filename "$FILE" \
    --max_examples "$MAX_EXAMPLES" \
    --num_shards "$num_shards" \
    --shard_idx "$shard_idx" \
    --output_filename "$NEW_OUTPUTS/${tag}_results.jsonl" \
    --max_questions "$MAX_QUESTIONS" \
    --abstain_threshold "$CONFIDENCE_THRESHOLD" \
    --option_mode yes-option \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEMP" \
    --top_p 0.9 \
    --use_vllm \
    --vllm_max_model_len "$VLLM_MAX_MODEL_LEN" \
    --batch_size "$BATCH_SIZE" \
    --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS" \
    --tensor_parallel_size "$TP_SIZE" \
    --convo_log_filename "$NEW_OUTPUTS/${tag}_convo.txt" \
    --doctor_log_filename "$NEW_OUTPUTS/${tag}_doctor_view.txt"
}

run_benchmark_with_retry() {
  local tag="$1" shard_idx="$2" num_shards="$3" gpu_id="$4"
  local attempt=1
  while [[ $attempt -le $BENCH_MAX_ATTEMPTS ]]; do
    if run_benchmark "$tag" "$shard_idx" "$num_shards" "$gpu_id"; then
      return 0
    fi
    if [[ $attempt -eq $BENCH_MAX_ATTEMPTS ]]; then
      echo "[run_condensed.sh] FATAL: ${tag} failed after ${BENCH_MAX_ATTEMPTS} attempts." >&2
      return 1
    fi
    echo "[run_condensed.sh] ${tag} failed on attempt ${attempt}/${BENCH_MAX_ATTEMPTS}; sleeping ${BENCH_RETRY_SLEEP}s then retrying..."
    sleep "$BENCH_RETRY_SLEEP"
    attempt=$((attempt + 1))
  done
}

merge_shards() {
  echo
  echo "=== Merge shards :: ${TAG} ==="

  "$PYTHON" - "$NEW_OUTPUTS" "$TAG" "$NUM_SHARDS" <<'PYMERGE'
import json
import re
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
tag = sys.argv[2]
num_shards = int(sys.argv[3])

result_rows = []
for shard_idx in range(num_shards):
    path = out_dir / f"{tag}_shard{shard_idx}_results.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    for line in path.read_text(errors="replace").splitlines():
        if line.strip():
            result_rows.append(json.loads(line))
result_rows.sort(key=lambda row: int(row["id"]))
merged_results = out_dir / f"{tag}_results.jsonl"
merged_results.write_text("".join(json.dumps(row) + "
" for row in result_rows))
correct = sum(1 for r in result_rows if r.get("interactive_system", {}).get("correct", False))
total = len(result_rows)
print(f"Merged {total} rows -> {merged_results}")
print(f"Accuracy: {correct}/{total} = {correct/total:.4f}")

header_re = re.compile(r"^Patient #(\d+)\s+\|", re.MULTILINE)

def split_blocks(text):
    matches = list(header_re.finditer(text))
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        yield int(match.group(1)), text[start:end].strip() + "
"

def merge_text_logs(suffix):
    blocks = []
    for shard_idx in range(num_shards):
        path = out_dir / f"{tag}_shard{shard_idx}_{suffix}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        blocks.extend(split_blocks(path.read_text(errors="replace")))
    blocks.sort(key=lambda item: item[0])
    merged = out_dir / f"{tag}_{suffix}.txt"
    merged.write_text("
".join(block for _, block in blocks))
    print(f"Merged {len(blocks)} patient blocks -> {merged}")

merge_text_logs("convo")
merge_text_logs("doctor_view")
PYMERGE
}

cleanup_shards() {
  for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    local shard_tag="${TAG}_shard${shard_idx}"
    rm -f "$NEW_OUTPUTS/${shard_tag}_results.jsonl" \
          "$NEW_OUTPUTS/${shard_tag}_convo.txt" \
          "$NEW_OUTPUTS/${shard_tag}_doctor_view.txt" \
          "$NEW_OUTPUTS/${shard_tag}_run.log"
  done
}

echo
echo "############################################################"
echo "# CondensedScaleExpert | no reasoning | Tag: ${TAG}"
echo "# GPUs: ${SHARD_GPUS} | shards=${NUM_SHARDS} | TP_SIZE=${TP_SIZE}"
echo "# max_examples=${MAX_EXAMPLES} temp=${TEMP} threshold=${CONFIDENCE_THRESHOLD} max_questions=${MAX_QUESTIONS}"
echo "############################################################"

PIDS=()
SHARD_TAGS=()
for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu_id="${GPU_ARRAY[$shard_idx]}"
  shard_tag="${TAG}_shard${shard_idx}"
  SHARD_TAGS+=("$shard_tag")
  echo "[run_condensed.sh] launching ${shard_tag} on GPU ${gpu_id}; log: ${NEW_OUTPUTS}/${shard_tag}_run.log"
  run_benchmark_with_retry "$shard_tag" "$shard_idx" "$NUM_SHARDS" "$gpu_id" \
    > "$NEW_OUTPUTS/${shard_tag}_run.log" 2>&1 &
  PIDS+=("$!")
done

FAILED=0
while true; do
  all_done=1
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      all_done=0
    fi
  done
  if [[ "$all_done" -eq 1 ]]; then break; fi
  sleep 30
done

for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  shard_tag="${SHARD_TAGS[$i]}"
  if wait "$pid"; then
    echo "[run_condensed.sh] ${shard_tag} finished successfully."
  else
    echo "[run_condensed.sh] ${shard_tag} failed. See ${NEW_OUTPUTS}/${shard_tag}_run.log" >&2
    FAILED=1
  fi
done

if [[ "$FAILED" -ne 0 ]]; then
  echo "[run_condensed.sh] FATAL: at least one shard failed." >&2
  exit 1
fi

merge_shards
cleanup_shards

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
printf -v ELAPSED_HMS '%02d:%02d:%02d' "$((ELAPSED / 3600))" "$(((ELAPSED % 3600) / 60))" "$((ELAPSED % 60))"
echo
echo "Done. Artifacts in: $NEW_OUTPUTS"
echo "Total elapsed: ${ELAPSED}s (${ELAPSED_HMS})"
