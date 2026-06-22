#!/bin/bash
# MediQ benchmark -- SCOPE-Medical doctor with Qwen3-4B.
# Matches run.sh's mediQ prompt/data/model/generation surface where applicable.
# Expert : ScopeMedicalExpert (SCOPE-style semantic MCTS via medical-scope/)
# Patient: FactSelectPatient (Qwen3-4B)
# Each shard occupies 2 GPUs: one for the LLM, one for embedding/reward/transition.

set -e

START_TIME=$(date +%s)

# ---- Easy config -----------------------------------------------------------
PYTHON=/home/hyang/miniconda3/envs/scope/bin/python
SRC=/home/hyang/mediQ/src
SCRIPTS=/home/hyang/mediQ/scripts
MEDICAL_SCOPE=/home/hyang/mediQ/medical-scope
DATA=/home/hyang/mediQ/data/med_data
FILE="${FILE:-/home/hyang/mediQ/data/med_data/all_test_convo_medqa.jsonl}"
NEW_OUTPUTS="${NEW_OUTPUTS:-/home/hyang/mediQ/new_outputs}"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-4B}"
PATIENT_MODEL="${PATIENT_MODEL:-$QWEN_MODEL}"

TAG="${TAG:-scope_qwen3_4b_seq_conf5}"
MAX_EXAMPLES="${MAX_EXAMPLES:-100}"
TEMP="${TEMP:-0.6}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-5.0}"
# branch_depth=0: sequential (no outer branching); SCOPE's internal MCTS supplies look-ahead.
BRANCH_DEPTH=0
BRANCH_TOP_K=0

# SCOPE MCTS time budget per doctor decision (seconds).
SCOPE_MCTS_TIME="${SCOPE_MCTS_TIME:-30}"

# Data parallelism: each shard uses 2 GPUs (LLM + Q).
# Default: 2 shards × 2 GPUs = 4 GPUs (same 4-GPU server as run.sh).
NUM_SHARDS="${NUM_SHARDS:-2}"
SHARD_GPUS="${SHARD_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"

MAX_TOKENS="${MAX_TOKENS:-2048}"
SCOPE_TRANSITION_DIR="${SCOPE_TRANSITION_DIR:-/home/hyang/mediQ/scope_saved/transition_models}"
SCOPE_REWARD_PATH="${SCOPE_REWARD_PATH:-/home/hyang/mediQ/scope_saved/reward/embedding_mediQ_reward_cumulative.pt}"


mkdir -p "$NEW_OUTPUTS"
export MEDIQ_ENABLE_THINKING="${MEDIQ_ENABLE_THINKING:-0}"

IFS=',' read -r -a GPU_ARRAY <<< "$SHARD_GPUS"
GPUS_NEEDED=$((NUM_SHARDS * 2))
if [[ "${#GPU_ARRAY[@]}" -lt "$GPUS_NEEDED" ]]; then
  echo "[run_scope.sh] FATAL: need ${NUM_SHARDS} shards × 2 GPUs = ${GPUS_NEEDED} GPU ids, but SHARD_GPUS only has ${#GPU_ARRAY[@]}: ${SHARD_GPUS}" >&2
  exit 1
fi

for required_path in "$FILE" "$MEDICAL_SCOPE/medical_scope/expert.py" "$SCOPE_REWARD_PATH" "$SCOPE_TRANSITION_DIR"; do
  if [[ ! -e "$required_path" ]]; then
    echo "[run_scope.sh] FATAL: required path not found: $required_path" >&2
    exit 1
  fi
done

if [[ "$MAX_EXAMPLES" -gt 0 ]]; then
  TARGET_EXAMPLES="$MAX_EXAMPLES"
else
  TARGET_EXAMPLES=$(wc -l < "$FILE")
fi

# ---- Benchmark runner ------------------------------------------------------
run_scope_benchmark() {
  local tag="$1"
  local shard_idx="$2"
  local num_shards="$3"
  local llm_gpu="$4"   # physical GPU for Qwen3-4B LLM weights
  local q_gpu="$5"     # physical GPU for embedding / reward MLP / transition model

  echo
  echo "=== SCOPE Benchmark :: ${tag} ==="
  echo "llm_gpu=${llm_gpu} q_gpu=${q_gpu} shard=${shard_idx}/${num_shards} mcts_time=${SCOPE_MCTS_TIME}s threshold=${CONFIDENCE_THRESHOLD}"

  : > "$NEW_OUTPUTS/${tag}_scope_trace.jsonl"

  # CUDA_VISIBLE_DEVICES restricts the process to exactly these two GPUs.
  # Within the process, logical index 0 = llm_gpu, logical index 1 = q_gpu.
  cd "$SRC" && \
    PYTHONPATH="$MEDICAL_SCOPE:$SRC:${PYTHONPATH:-}" \
    SCOPE_MEDICAL_EMBED_DEVICE=cuda:1 \
    SCOPE_MEDICAL_TRANSITION_DEVICE=cuda:1 \
    SCOPE_MEDICAL_REWARD_DEVICE=cuda:1 \
    SCOPE_MEDICAL_MCTS_TIME="$SCOPE_MCTS_TIME" \
    SCOPE_MEDICAL_TRANSITION_DIR="$SCOPE_TRANSITION_DIR" \
    SCOPE_MEDICAL_REWARD_PATH="$SCOPE_REWARD_PATH" \
    SCOPE_MEDICAL_TRACE_JSONL="$NEW_OUTPUTS/${tag}_scope_trace.jsonl" \
    CUDA_VISIBLE_DEVICES="${llm_gpu},${q_gpu}" \
    "$PYTHON" mediQ_benchmark.py \
      --expert_class ScopeMedicalExpert \
      --expert_module medical_scope.expert \
      --expert_model "$QWEN_MODEL" \
      --expert_model_question_generator "$QWEN_MODEL" \
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
      --branch_depth "$BRANCH_DEPTH" \
      --convo_log_filename "$NEW_OUTPUTS/${tag}_convo.txt" \
      --doctor_log_filename "$NEW_OUTPUTS/${tag}_doctor_view.txt"
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

trace_rows = []
for shard_idx in range(num_shards):
    path = out_dir / f"{tag}_shard{shard_idx}_scope_trace.jsonl"
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            if line.strip():
                row = json.loads(line)
                row["shard_idx"] = shard_idx
                trace_rows.append(row)
merged_trace = out_dir / f"{tag}_scope_trace.jsonl"
merged_trace.write_text("".join(json.dumps(row) + "\n" for row in trace_rows))
print(f"Merged {len(trace_rows)} SCOPE trace rows -> {merged_trace}")
PYMERGE
}

cleanup_shards() {
  echo
  echo "=== Cleanup shard artifacts :: ${TAG} ==="
  for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    local shard_tag="${TAG}_shard${shard_idx}"
    rm -f \
      "$NEW_OUTPUTS/${shard_tag}_results.jsonl" \
      "$NEW_OUTPUTS/${shard_tag}_convo.txt" \
      "$NEW_OUTPUTS/${shard_tag}_doctor_view.txt" \
      "$NEW_OUTPUTS/${shard_tag}_scope_trace.jsonl"
  done
  echo "Removed shard JSONL/text files for ${NUM_SHARDS} shards; kept shard run logs."
}

summarize_results() {
  local results_jsonl="$NEW_OUTPUTS/${TAG}_results.jsonl"

  echo
  echo "=== SCOPE results summary :: ${TAG} ==="

  "$PYTHON" - "$results_jsonl" "$MAX_EXAMPLES" <<'PYSUM'
import json
import sys
from pathlib import Path
from collections import Counter

results_path = Path(sys.argv[1])
max_examples = int(sys.argv[2])

correct = 0
total = 0
turn_lengths = []

with results_path.open() as f:
    for line in f:
        row = json.loads(line)
        total += 1
        info = row.get("info", {})
        interactive = row.get("interactive_system", {})
        if interactive.get("correct"):
            correct += 1
        qs = info.get("questions") or []
        turn_lengths.append(len(qs))

accuracy = correct / total if total else 0.0
avg_turns = sum(turn_lengths) / len(turn_lengths) if turn_lengths else 0.0
print(f"Results JSONL     : {results_path}")
print(f"Total patients    : {total}")
print(f"Correct           : {correct} / {total} = {accuracy:.4f}")
print(f"Avg turns         : {avg_turns:.2f}")
print(f"Turn distribution : {dict(sorted(Counter(turn_lengths).items()))}")
PYSUM
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
echo "# Run: SCOPE-Medical Qwen3-4B (MEDIQ_ENABLE_THINKING=${MEDIQ_ENABLE_THINKING})"
echo "# Tag: ${TAG}"
echo "# File: ${FILE}"
echo "# GPUs: ${SHARD_GPUS} | shards=${NUM_SHARDS} (2 GPUs each) | mcts_time=${SCOPE_MCTS_TIME}s"
echo "# SCOPE transition: ${SCOPE_TRANSITION_DIR}"
echo "# SCOPE reward: ${SCOPE_REWARD_PATH}"
echo "# max_examples=${MAX_EXAMPLES} target_examples=${TARGET_EXAMPLES} temp=${TEMP} branch_depth=${BRANCH_DEPTH} threshold=${CONFIDENCE_THRESHOLD}"
echo "############################################################"

PIDS=()
SHARD_TAGS=()
for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
  llm_gpu="${GPU_ARRAY[$((shard_idx * 2))]}"
  q_gpu="${GPU_ARRAY[$((shard_idx * 2 + 1))]}"
  shard_tag="${TAG}_shard${shard_idx}"
  SHARD_TAGS+=("$shard_tag")
  echo "[run_scope.sh] launching ${shard_tag} on LLM GPU ${llm_gpu}, Q GPU ${q_gpu}; log: ${NEW_OUTPUTS}/${shard_tag}_run.log"
  run_scope_benchmark "$shard_tag" "$shard_idx" "$NUM_SHARDS" "$llm_gpu" "$q_gpu" \
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
    parts+=("shard${shard_idx}=${done}/$(((TARGET_EXAMPLES + NUM_SHARDS - 1 - shard_idx) / NUM_SHARDS))")
  done
  local now elapsed hms
  now=$(date +%s)
  elapsed=$((now - START_TIME))
  printf -v hms '%02d:%02d:%02d' "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))"
  echo "[progress ${hms}] ${parts[*]} total=${total_done}/${TARGET_EXAMPLES}"
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
    echo "[run_scope.sh] ${shard_tag} finished successfully."
  else
    echo "[run_scope.sh] ${shard_tag} failed. See ${NEW_OUTPUTS}/${shard_tag}_run.log" >&2
    FAILED=1
  fi
done

if [[ "$FAILED" -ne 0 ]]; then
  echo "[run_scope.sh] FATAL: at least one shard failed; skipping merge/eval." >&2
  exit 1
fi

merge_shards
summarize_results
run_trajectory_eval
cleanup_shards

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
printf -v ELAPSED_HMS '%02d:%02d:%02d' "$((ELAPSED / 3600))" "$(((ELAPSED % 3600) / 60))" "$((ELAPSED % 60))"

echo
echo "SCOPE run + merge + trajectory eval complete. Artifacts in: $NEW_OUTPUTS"
echo "Total elapsed time: ${ELAPSED}s (${ELAPSED_HMS})"
