#!/usr/bin/env bash
# Retrain transition models — v2
#
# What runs and why:
#   medical_accumulative_moe   seeds 0-3  100 epochs  fresh start (was 30)
#   code_feedback_moe          seeds 0-3  100 epochs  fresh start (was 30)
#   medical_turn_independent_mdn seeds 0-1 100 epochs  lr=1e-4  (seed 0 diverged at lr=1e-3)
#   code_feedback_mdn          seeds 0-1  100 epochs  lr=1e-4  (seed 0 llm_human poor at lr=1e-3)
#
# What is kept untouched:
#   medical_accumulative_mdn_seed_0_batch_512  — converged cleanly, not deleted
#   medical_turn_independent_moe_*             — poor formulation, not worth retraining
#
# Old folders for retrained models are deleted before training starts.

set -euo pipefail

cd /home/hyang/mediQ

PYTHON="${PYTHON:-/home/hyang/miniconda3/envs/scope/bin/python}"
SCRIPT="scripts/train_scope_transition.py"
NEW_DIR="scope_saved/transition_models/new"
LOG_DIR="${LOG_DIR:-${NEW_DIR}/logs}"
MOE_BATCH="${MOE_BATCH:-2048}"
MDN_BATCH="${MDN_BATCH:-512}"
EPOCHS="${EPOCHS:-100}"
NUM_GPUS="${NUM_GPUS:-4}"

mkdir -p "${NEW_DIR}" "${LOG_DIR}"
LAUNCHER_LOG="${LOG_DIR}/launcher_v2.log"
exec > >(tee -a "${LAUNCHER_LOG}") 2>&1

# ── Delete old folders for models being replaced ──────────────────────────────
echo "Removing old checkpoints for models being retrained..."
for seed in 0 1 2 3; do
    rm -rf "${NEW_DIR}/medical_accumulative_moe_seed_${seed}_batch_${MOE_BATCH}"
    rm -rf "${NEW_DIR}/code_feedback_moe_seed_${seed}_batch_${MOE_BATCH}"
done
rm -rf "${NEW_DIR}/medical_turn_independent_mdn_seed_0_batch_${MDN_BATCH}"
rm -rf "${NEW_DIR}/code_feedback_mdn_seed_0_batch_${MDN_BATCH}"
echo "Done. Keeping: medical_accumulative_mdn_seed_0_batch_${MDN_BATCH} (already converged)"
echo ""

# ── Job runner ────────────────────────────────────────────────────────────────
run_job() {
    local gpu="$1"
    local tag="$2"
    local log="${LOG_DIR}/${tag}.log"
    shift 2
    echo "[$(date -Is)] START gpu=${gpu} ${tag} -> ${log}"
    "${PYTHON}" "${SCRIPT}" --device "cuda:${gpu}" "$@" > "${log}" 2>&1
    echo "[$(date -Is)] DONE  gpu=${gpu} ${tag}"
}

# ── Job definitions: tag | extra args ────────────────────────────────────────
# Format: "tag|arg1|arg2|..."
JOBS=(
    # medical accumulative MoE — seeds 0-3, 100 epochs
    "medical_accumulative_moe_seed_0|--scope|medical|--embedding-mode|accumulative|--model-type|moe|--seed|0|--batch-size|${MOE_BATCH}|--epochs|${EPOCHS}"
    "medical_accumulative_moe_seed_1|--scope|medical|--embedding-mode|accumulative|--model-type|moe|--seed|1|--batch-size|${MOE_BATCH}|--epochs|${EPOCHS}"
    "medical_accumulative_moe_seed_2|--scope|medical|--embedding-mode|accumulative|--model-type|moe|--seed|2|--batch-size|${MOE_BATCH}|--epochs|${EPOCHS}"
    "medical_accumulative_moe_seed_3|--scope|medical|--embedding-mode|accumulative|--model-type|moe|--seed|3|--batch-size|${MOE_BATCH}|--epochs|${EPOCHS}"

    # code feedback MoE — seeds 0-3, 100 epochs
    "code_feedback_moe_seed_0|--scope|code|--model-type|moe|--seed|0|--batch-size|${MOE_BATCH}|--epochs|${EPOCHS}"
    "code_feedback_moe_seed_1|--scope|code|--model-type|moe|--seed|1|--batch-size|${MOE_BATCH}|--epochs|${EPOCHS}"
    "code_feedback_moe_seed_2|--scope|code|--model-type|moe|--seed|2|--batch-size|${MOE_BATCH}|--epochs|${EPOCHS}"
    "code_feedback_moe_seed_3|--scope|code|--model-type|moe|--seed|3|--batch-size|${MOE_BATCH}|--epochs|${EPOCHS}"

    # medical turn_independent MDN — seeds 0-1, lr=1e-4 (was 1e-3, human_llm diverged)
    "medical_turn_independent_mdn_seed_0|--scope|medical|--embedding-mode|turn_independent|--model-type|mdn|--seed|0|--batch-size|${MDN_BATCH}|--epochs|${EPOCHS}|--lr|1e-4"
    "medical_turn_independent_mdn_seed_1|--scope|medical|--embedding-mode|turn_independent|--model-type|mdn|--seed|1|--batch-size|${MDN_BATCH}|--epochs|${EPOCHS}|--lr|1e-4"

    # code feedback MDN — seeds 0-1, lr=1e-4 (was 1e-3, llm_human NLL=392)
    "code_feedback_mdn_seed_0|--scope|code|--model-type|mdn|--seed|0|--batch-size|${MDN_BATCH}|--epochs|${EPOCHS}|--lr|1e-4"
    "code_feedback_mdn_seed_1|--scope|code|--model-type|mdn|--seed|1|--batch-size|${MDN_BATCH}|--epochs|${EPOCHS}|--lr|1e-4"
)

echo "Launching ${#JOBS[@]} jobs on ${NUM_GPUS} GPUs (logs -> ${LOG_DIR}/)"

running=0
job_idx=0
failed=0

for job_line in "${JOBS[@]}"; do
    IFS='|' read -ra parts <<< "${job_line}"
    tag="${parts[0]}"
    extra_args=("${parts[@]:1}")
    gpu=$((job_idx % NUM_GPUS))

    while (( running >= NUM_GPUS )); do
        if ! wait -n; then
            failed=$((failed + 1))
        fi
        running=$((running - 1))
    done

    run_job "${gpu}" "${tag}" "${extra_args[@]}" &
    running=$((running + 1))
    job_idx=$((job_idx + 1))
done

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
