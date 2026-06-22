#!/bin/bash
# mediQ benchmark with ScopeMedicalExpert.
# All new code lives under medical-scope/; mediQ prompts are imported from src/.

set -e

REPO_DIR="${REPO_DIR:-/home/hyang/mediQ}"
PYTHON="${PYTHON:-/home/hyang/miniconda3/envs/scope/bin/python}"
DATA_DIR="${DATA_DIR:-$REPO_DIR/data/med_data}"
DEV_FILENAME="${DEV_FILENAME:-all_test_convo_medqa.jsonl}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/new_outputs/med}"
TAG="${TAG:-medical_scope_qwen3_4b}"

EXPERT_MODEL="${EXPERT_MODEL:-Qwen/Qwen3-4B}"
PATIENT_MODEL="${PATIENT_MODEL:-$EXPERT_MODEL}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
MAX_QUESTIONS="${MAX_QUESTIONS:-10}"
TEMP="${TEMP:-0.8}"
TOP_P="${TOP_P:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-5.0}"

export PYTHONPATH="$REPO_DIR/medical-scope:$REPO_DIR/src:${PYTHONPATH}"
export SCOPE_MEDICAL_TRACE_JSONL="${SCOPE_MEDICAL_TRACE_JSONL:-$OUT_DIR/${TAG}_scope_trace.jsonl}"
export SCOPE_MEDICAL_NUM_CANDIDATES="${SCOPE_MEDICAL_NUM_CANDIDATES:-5}"
export SCOPE_MEDICAL_CANDIDATE_MAX_NEW_TOKENS="${SCOPE_MEDICAL_CANDIDATE_MAX_NEW_TOKENS:-500}"
export SCOPE_MEDICAL_CANDIDATE_NUM_BEAM_GROUPS="${SCOPE_MEDICAL_CANDIDATE_NUM_BEAM_GROUPS:-5}"
export SCOPE_MEDICAL_CANDIDATE_DIVERSITY_PENALTY="${SCOPE_MEDICAL_CANDIDATE_DIVERSITY_PENALTY:-1.0}"
export SCOPE_MEDICAL_CANDIDATE_REPETITION_PENALTY="${SCOPE_MEDICAL_CANDIDATE_REPETITION_PENALTY:-1.0}"
export SCOPE_MEDICAL_PLANNING_DEPTH="${SCOPE_MEDICAL_PLANNING_DEPTH:-8}"
export SCOPE_MEDICAL_MCTS_TIME="${SCOPE_MEDICAL_MCTS_TIME:-30}"
export SCOPE_MEDICAL_TRANSITION_DIR="${SCOPE_MEDICAL_TRANSITION_DIR:-$REPO_DIR/scope_saved/transition_models}"
export SCOPE_MEDICAL_REWARD_PATH="${SCOPE_MEDICAL_REWARD_PATH:-$REPO_DIR/scope_saved/reward/embedding_mediQ_reward_cumulative.pt}"

mkdir -p "$OUT_DIR"
: > "$SCOPE_MEDICAL_TRACE_JSONL"

cd "$REPO_DIR/src"
"$PYTHON" mediQ_benchmark.py \
  --expert_module medical_scope.expert \
  --expert_class ScopeMedicalExpert \
  --expert_model "$EXPERT_MODEL" \
  --expert_model_question_generator "$EXPERT_MODEL" \
  --patient_module patient \
  --patient_class FactSelectPatient \
  --patient_model "$PATIENT_MODEL" \
  --data_dir "$DATA_DIR" \
  --dev_filename "$DEV_FILENAME" \
  --output_filename "$OUT_DIR/${TAG}_results.jsonl" \
  --convo_log_filename "$OUT_DIR/${TAG}_convo.txt" \
  --doctor_log_filename "$OUT_DIR/${TAG}_doctor_view.txt" \
  --max_examples "$MAX_EXAMPLES" \
  --max_questions "$MAX_QUESTIONS" \
  --abstain_threshold "$CONFIDENCE_THRESHOLD" \
  --option_mode yes-option \
  --temperature "$TEMP" \
  --top_p "$TOP_P" \
  --max_tokens "$MAX_TOKENS"
