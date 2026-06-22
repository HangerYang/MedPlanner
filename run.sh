#!/bin/bash
# MediQ benchmark -- 4-GPU data-parallel conf5 branch-d3 high-temp reasoning run.

set -e

START_TIME=$(date +%s)

# ---- Easy config -----------------------------------------------------------
PYTHON=/home/hyang/miniconda3/envs/scope/bin/python
SRC=/home/hyang/mediQ/src
SCRIPTS=/home/hyang/mediQ/scripts
DATA=/home/hyang/mediQ/data/med_data
FILE="${FILE:-/home/hyang/mediQ/data/med_data/all_train_convo_medqa.jsonl}"
NEW_OUTPUTS="${NEW_OUTPUTS:-/home/hyang/mediQ/new_outputs}"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-4B}"
EXPERT_MODEL="${EXPERT_MODEL:-$QWEN_MODEL}"
PATIENT_MODEL="${PATIENT_MODEL:-$QWEN_MODEL}"

TAG="${TAG:-scale_qwen3_4b_branch_d3_regulartemp_conf5_100q_reasoning}"
MAX_EXAMPLES="${MAX_EXAMPLES:-100}"
TEMP="${TEMP:-0.6}"
BRANCH_DEPTH="${BRANCH_DEPTH:-3}"
BRANCH_TOP_K="${BRANCH_TOP_K:-2}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-5.0}"

# Data parallelism: one independent single-GPU vLLM process per shard.
NUM_SHARDS="${NUM_SHARDS:-4}"
SHARD_GPUS="${SHARD_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
TP_SIZE=1

# vLLM tuning. 0.8 is intentionally aggressive for this conf5 run.
export MEDIQ_VLLM_GPU_MEMORY_UTILIZATION="${MEDIQ_VLLM_GPU_MEMORY_UTILIZATION:-0.8}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
BATCH_SIZE="${BATCH_SIZE:-4}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
MAX_TOKENS="${MAX_TOKENS:-2048}"

BENCH_RETRY_SLEEP="${BENCH_RETRY_SLEEP:-600}"
BENCH_MAX_ATTEMPTS="${BENCH_MAX_ATTEMPTS:-5}"

mkdir -p "$NEW_OUTPUTS"
export MEDIQ_ENABLE_THINKING="${MEDIQ_ENABLE_THINKING:-1}"

IFS=',' read -r -a GPU_ARRAY <<< "$SHARD_GPUS"
if [[ "${#GPU_ARRAY[@]}" -lt "$NUM_SHARDS" ]]; then
  echo "[run.sh] FATAL: NUM_SHARDS=${NUM_SHARDS}, but SHARD_GPUS only has ${#GPU_ARRAY[@]} GPU ids: ${SHARD_GPUS}" >&2
  exit 1
fi

# ---- Benchmark runner ------------------------------------------------------
run_benchmark() {
  local tag="$1"
  local shard_idx="$2"
  local num_shards="$3"
  local gpu_id="$4"

  echo
  echo "=== Benchmark :: ${tag} ==="
  echo "gpu=${gpu_id} shard=${shard_idx}/${num_shards} temp=${TEMP} branch_depth=${BRANCH_DEPTH} branch_top_k=${BRANCH_TOP_K} confidence_threshold=${CONFIDENCE_THRESHOLD}"

  local branch_args=()
  if [[ "$BRANCH_DEPTH" -gt 0 ]]; then
    branch_args=(--branch_depth "$BRANCH_DEPTH" --branch_top_k "$BRANCH_TOP_K")
  fi

  cd "$SRC" && CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON" mediQ_benchmark.py \
    --expert_class ScaleExpert \
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
    --max_questions 10 \
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
    --doctor_log_filename "$NEW_OUTPUTS/${tag}_doctor_view.txt" \
    "${branch_args[@]}"
}

run_benchmark_with_retry() {
  local tag="$1" shard_idx="$2" num_shards="$3" gpu_id="$4"
  local attempt=1
  while [[ $attempt -le $BENCH_MAX_ATTEMPTS ]]; do
    if run_benchmark "$tag" "$shard_idx" "$num_shards" "$gpu_id"; then
      return 0
    fi
    if [[ $attempt -eq $BENCH_MAX_ATTEMPTS ]]; then
      echo "[run.sh] FATAL: benchmark ${tag} failed after ${BENCH_MAX_ATTEMPTS} attempts." >&2
      return 1
    fi
    echo "[run.sh] ${tag} failed on attempt ${attempt}/${BENCH_MAX_ATTEMPTS}; sleeping ${BENCH_RETRY_SLEEP}s then retrying..."
    sleep "$BENCH_RETRY_SLEEP"
    attempt=$((attempt + 1))
  done
}

# ---- Merge and analysis ----------------------------------------------------
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
merged_results.write_text("".join(json.dumps(row) + "\n" for row in result_rows))
print(f"Merged {len(result_rows)} result rows -> {merged_results}")

header_re = re.compile(r"^Patient #(\d+)\s+\|", re.MULTILINE)

def split_blocks(text):
    matches = list(header_re.finditer(text))
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        yield int(match.group(1)), text[start:end].strip() + "\n"

def merge_text_logs(suffix):
    blocks = []
    for shard_idx in range(num_shards):
        path = out_dir / f"{tag}_shard{shard_idx}_{suffix}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        blocks.extend(split_blocks(path.read_text(errors="replace")))
    blocks.sort(key=lambda item: item[0])
    merged = out_dir / f"{tag}_{suffix}.txt"
    merged.write_text("\n".join(block for _, block in blocks))
    print(f"Merged {len(blocks)} patient blocks -> {merged}")

merge_text_logs("convo")
merge_text_logs("doctor_view")
PYMERGE
}

cleanup_shards() {
  echo
  echo "=== Cleanup shard artifacts :: ${TAG} ==="
  for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    local shard_tag="${TAG}_shard${shard_idx}"
    rm -f       "$NEW_OUTPUTS/${shard_tag}_results.jsonl"       "$NEW_OUTPUTS/${shard_tag}_convo.txt"       "$NEW_OUTPUTS/${shard_tag}_doctor_view.txt"       "$NEW_OUTPUTS/${shard_tag}_run.log"
  done
  echo "Removed shard JSONL/log/text files for ${NUM_SHARDS} shards."
}

summarize_confidence5() {
  local results_jsonl="$NEW_OUTPUTS/${TAG}_results.jsonl"

  echo
  echo "=== Confidence-5 summary :: ${TAG} ==="

  "$PYTHON" - "$results_jsonl" "$MAX_EXAMPLES" "$BRANCH_DEPTH" <<'PYCONF'
import json
import sys
from collections import Counter
from pathlib import Path

results_path = Path(sys.argv[1])
max_examples = int(sys.argv[2])
branch_depth = int(sys.argv[3])

leaf_counts = []
conf5_total = 0
conf5_patients = set()
conf5_by_depth = Counter()
leaf_reason_counts = Counter()

with results_path.open() as f:
    for line in f:
        row = json.loads(line)
        patient_id = int(row["id"])
        branches = row.get("info", {}).get("branches") or []
        leaves = [branch for branch in branches if branch.get("is_leaf")]
        leaf_counts.append(len(leaves))
        for leaf in leaves:
            leaf_reason_counts[leaf.get("leaf_reason")] += 1
            if float(leaf.get("confidence", -1)) == 5.0:
                conf5_total += 1
                conf5_patients.add(patient_id)
                conf5_by_depth[leaf.get("depth")] += 1

expected_full = max_examples * (2 ** branch_depth)
actual_leaves = sum(leaf_counts)
print(f"Results JSONL: {results_path}")
print(f"Expected full depth-{branch_depth} leaf conversations: {expected_full}")
print(f"Actual final leaf conversations: {actual_leaves}")
print(f"Missing vs full expansion: {expected_full - actual_leaves}")
print(f"Confidence == 5 leaf conversations: {conf5_total}")
print(f"Patients with at least one confidence-5 leaf: {len(conf5_patients)}")
print(f"Confidence-5 by depth: {dict(sorted(conf5_by_depth.items()))}")
print(f"Leaf reason counts: {dict(leaf_reason_counts)}")
print(f"Leaf count distribution: {dict(sorted(Counter(leaf_counts).items()))}")
PYCONF
}

run_trajectory_eval() {
  echo
  echo "=== Trajectory eval :: ${TAG} ==="

  "$PYTHON" "$SCRIPTS/analyze_convo_answer_trajectory.py" \
    --input "$NEW_OUTPUTS/${TAG}_convo.txt" \
    --results-jsonl "$NEW_OUTPUTS/${TAG}_results.jsonl" \
    --output-jsonl "$NEW_OUTPUTS/${TAG}_answer_trajectory.jsonl" \
    --summary "$NEW_OUTPUTS/${TAG}_answer_trajectory_summary.txt"
}

# ---- Launch ----------------------------------------------------------------
echo
echo "############################################################"
echo "# Run: 4-GPU data parallel Qwen3 conf5 (MEDIQ_ENABLE_THINKING=${MEDIQ_ENABLE_THINKING})"
echo "# Tag: ${TAG}"
echo "# GPUs: ${SHARD_GPUS} | shards=${NUM_SHARDS} | TP_SIZE=${TP_SIZE}"
echo "# max_examples=${MAX_EXAMPLES} temp=${TEMP} branch_depth=${BRANCH_DEPTH} branch_top_k=${BRANCH_TOP_K} threshold=${CONFIDENCE_THRESHOLD}"
echo "# MEDIQ_VLLM_GPU_MEMORY_UTILIZATION=${MEDIQ_VLLM_GPU_MEMORY_UTILIZATION}"
echo "############################################################"

PIDS=()
SHARD_TAGS=()
for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu_id="${GPU_ARRAY[$shard_idx]}"
  shard_tag="${TAG}_shard${shard_idx}"
  SHARD_TAGS+=("$shard_tag")
  echo "[run.sh] launching ${shard_tag} on GPU ${gpu_id}; log: ${NEW_OUTPUTS}/${shard_tag}_run.log"
  run_benchmark_with_retry "$shard_tag" "$shard_idx" "$NUM_SHARDS" "$gpu_id" \
    > "$NEW_OUTPUTS/${shard_tag}_run.log" 2>&1 &
  PIDS+=("$!")
done

print_progress() {
  local total_done=0
  local parts=()
  for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    local shard_tag="${TAG}_shard${shard_idx}"
    local results_file="${NEW_OUTPUTS}/${shard_tag}_results.jsonl"
    local done=0
    if [[ -f "$results_file" ]]; then
      done=$(wc -l < "$results_file")
    fi
    total_done=$((total_done + done))
    parts+=("shard${shard_idx}=${done}/$(((MAX_EXAMPLES + NUM_SHARDS - 1 - shard_idx) / NUM_SHARDS))")
  done
  local now elapsed hms
  now=$(date +%s)
  elapsed=$((now - START_TIME))
  printf -v hms '%02d:%02d:%02d' "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))"
  echo "[progress ${hms}] ${parts[*]} total=${total_done}/${MAX_EXAMPLES}"
}

FAILED=0
while true; do
  all_done=1
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    if kill -0 "$pid" 2>/dev/null; then
      all_done=0
    fi
  done
  print_progress
  if [[ "$all_done" -eq 1 ]]; then
    break
  fi
  sleep "${PROGRESS_INTERVAL:-30}"
done

for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  shard_tag="${SHARD_TAGS[$i]}"
  if wait "$pid"; then
    echo "[run.sh] ${shard_tag} finished successfully."
  else
    echo "[run.sh] ${shard_tag} failed. See ${NEW_OUTPUTS}/${shard_tag}_run.log" >&2
    FAILED=1
  fi
done

if [[ "$FAILED" -ne 0 ]]; then
  echo "[run.sh] FATAL: at least one shard failed; skipping merge/eval." >&2
  exit 1
fi

merge_shards
summarize_confidence5
run_trajectory_eval
cleanup_shards

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
printf -v ELAPSED_HMS '%02d:%02d:%02d' "$((ELAPSED / 3600))" "$(((ELAPSED % 3600) / 60))" "$((ELAPSED % 60))"

echo
echo "Run + merge + confidence summary + trajectory eval complete. Artifacts in: $NEW_OUTPUTS"
echo "Total elapsed time: ${ELAPSED}s (${ELAPSED_HMS})"
