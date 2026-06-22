#!/usr/bin/env bash
# Train all 15 transition-model jobs (3 sources × MoE×4 seeds + MDN×1 seed)
# across 4 GPUs.  At most 4 jobs run concurrently.
set -euo pipefail

cd /home/hyang/mediQ

PYTHON="${PYTHON:-/home/hyang/miniconda3/envs/scope/bin/python}"
SCRIPT="scripts/train_scope_transition.py"
LOG_DIR="${LOG_DIR:-scope_saved/transition_models/new/logs}"
mkdir -p scope_saved/transition_models/new "${LOG_DIR}"
MOE_BATCH="${MOE_BATCH:-2048}"
MDN_BATCH="${MDN_BATCH:-512}"
EPOCHS="${EPOCHS:-30}"
NUM_GPUS="${NUM_GPUS:-4}"

mkdir -p "${LOG_DIR}"

run_job() {
  local gpu="$1"
  local scope="$2"
  local embedding_mode="$3"   # accumulative | turn_independent | - (code)
  local model_type="$4"
  local seed="$5"
  local batch_size="$6"
  local tag="$7"
  local log="${LOG_DIR}/${tag}.log"

  echo "[$(date -Is)] START gpu=${gpu} ${tag} -> ${log}"
  if [[ "${scope}" == "code" ]]; then
    "${PYTHON}" "${SCRIPT}" \
      --scope code \
      --model-type "${model_type}" \
      --seed "${seed}" \
      --batch-size "${batch_size}" \
      --epochs "${EPOCHS}" \
      --device "cuda:${gpu}" \
      > "${log}" 2>&1
  else
    "${PYTHON}" "${SCRIPT}" \
      --scope medical \
      --embedding-mode "${embedding_mode}" \
      --model-type "${model_type}" \
      --seed "${seed}" \
      --batch-size "${batch_size}" \
      --epochs "${EPOCHS}" \
      --device "cuda:${gpu}" \
      > "${log}" 2>&1
  fi
  echo "[$(date -Is)] DONE  gpu=${gpu} ${tag}"
}

# job lines: scope|embedding_mode|model_type|seed|batch|tag
JOBS=(
  "medical|accumulative|moe|0|${MOE_BATCH}|medical_accumulative_moe_seed_0"
  "medical|accumulative|moe|1|${MOE_BATCH}|medical_accumulative_moe_seed_1"
  "medical|accumulative|moe|2|${MOE_BATCH}|medical_accumulative_moe_seed_2"
  "medical|accumulative|moe|3|${MOE_BATCH}|medical_accumulative_moe_seed_3"
  "medical|turn_independent|moe|0|${MOE_BATCH}|medical_turn_independent_moe_seed_0"
  "medical|turn_independent|moe|1|${MOE_BATCH}|medical_turn_independent_moe_seed_1"
  "medical|turn_independent|moe|2|${MOE_BATCH}|medical_turn_independent_moe_seed_2"
  "medical|turn_independent|moe|3|${MOE_BATCH}|medical_turn_independent_moe_seed_3"
  "code|-|moe|0|${MOE_BATCH}|code_feedback_moe_seed_0"
  "code|-|moe|1|${MOE_BATCH}|code_feedback_moe_seed_1"
  "code|-|moe|2|${MOE_BATCH}|code_feedback_moe_seed_2"
  "code|-|moe|3|${MOE_BATCH}|code_feedback_moe_seed_3"
  "medical|accumulative|mdn|0|${MDN_BATCH}|medical_accumulative_mdn_seed_0"
  "medical|turn_independent|mdn|0|${MDN_BATCH}|medical_turn_independent_mdn_seed_0"
  "code|-|mdn|0|${MDN_BATCH}|code_feedback_mdn_seed_0"
)

echo "Launching ${#JOBS[@]} jobs on ${NUM_GPUS} GPUs (logs -> ${LOG_DIR}/)"

running=0
job_idx=0
failed=0

for job_line in "${JOBS[@]}"; do
  IFS='|' read -r scope emb model seed batch tag <<< "${job_line}"
  gpu=$((job_idx % NUM_GPUS))

  # throttle: wait until fewer than NUM_GPUS jobs are running
  while (( running >= NUM_GPUS )); do
    if ! wait -n; then
      failed=$((failed + 1))
    fi
    running=$((running - 1))
  done

  run_job "${gpu}" "${scope}" "${emb}" "${model}" "${seed}" "${batch}" "${tag}" &
  running=$((running + 1))
  job_idx=$((job_idx + 1))
done

# wait for remaining background jobs
while (( running > 0 )); do
  if ! wait -n; then
    failed=$((failed + 1))
  fi
  running=$((running - 1))
done

if (( failed > 0 )); then
  echo "${failed} job(s) failed — check logs in ${LOG_DIR}/"
  exit 1
fi

echo "All ${#JOBS[@]} jobs finished successfully."
