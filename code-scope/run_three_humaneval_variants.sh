#!/bin/bash
set -euo pipefail

ROOT="${ROOT:-/home/hyang/mediQ}"
PYTHON="${PYTHON:-/home/hyang/anaconda3/envs/scope/bin/python}"
OUT_ROOT="${OUT_ROOT:-$ROOT/code-scope/output}"
LIMIT="${LIMIT:-0}"
START="${START:-0}"
DRY_RUN="${DRY_RUN:-1}"

REWARD_PATH="${REWARD_PATH:-$ROOT/mediQ_model_files/code_feedback_cumulative_reward_mlp.pt}"
MOE_TRANSITION_DIR="${MOE_TRANSITION_DIR:-$ROOT/mediQ_model_files/code-moe}"
MDN_TRANSITION_DIR="${MDN_TRANSITION_DIR:-$ROOT/mediQ_model_files/code_feedback_mdn_seed_0_batch_512}"

COMMON_ARGS=(--start "$START")
if [[ "$LIMIT" != "0" ]]; then
  COMMON_ARGS+=(--limit "$LIMIT")
fi

preflight() {
  echo "PYTHON=$PYTHON"
  test -x "$PYTHON"
  echo "REWARD_PATH=$REWARD_PATH"
  test -f "$REWARD_PATH"
  echo "MOE_TRANSITION_DIR=$MOE_TRANSITION_DIR"
  test -d "$MOE_TRANSITION_DIR"
  echo "MDN_TRANSITION_DIR=$MDN_TRANSITION_DIR"
  test -d "$MDN_TRANSITION_DIR"

  "$PYTHON" - <<PY
from pathlib import Path
import torch

moe = Path("$MOE_TRANSITION_DIR")
mdn = Path("$MDN_TRANSITION_DIR")
reward = Path("$REWARD_PATH")

def count_direction(root, kind):
    roots = [root] + [p for p in root.iterdir() if p.is_dir()]
    found = []
    for base in roots:
        path = base / kind / "model_min_train.pth"
        if not path.exists():
            path = base / kind / "model_min_val.pth"
        if path.exists() and path not in found:
            found.append(path)
    return found

moe_llm = count_direction(moe, "human_llm")
moe_human = count_direction(moe, "llm_human")
mdn_llm = count_direction(mdn, "human_llm")
mdn_human = count_direction(mdn, "llm_human")
print(f"moe_llm_models={len(moe_llm)}")
print(f"moe_human_models={len(moe_human)}")
print(f"mdn_llm_models={len(mdn_llm)}")
print(f"mdn_human_models={len(mdn_human)}")
if len(moe_llm) != 4 or len(moe_human) != 4:
    raise SystemExit("expected 4 MoE models per direction")
if len(mdn_llm) != 1 or len(mdn_human) != 1:
    raise SystemExit("expected 1 MDN model per direction")

moe_state = torch.load(moe_llm[0], map_location="cpu")["model_state_dict"]
mdn_state = torch.load(mdn_llm[0], map_location="cpu")["model_state_dict"]
reward_state = torch.load(reward, map_location="cpu", weights_only=False)
print("moe_dim=" + str(moe_state["input_mean"].numel()))
print("moe_outer_experts=" + str(moe_state["model.gate_outer.w_gating"].shape[1]))
print("moe_inner_experts=" + str(moe_state["model.gate_inner.w_gating"].shape[2]))
print("mdn_dim=" + str(mdn_state["input_mean"].numel()))
print("mdn_components=" + str(mdn_state["model.pi_network.6.weight"].shape[0]))
print("mdn_hidden=" + str(mdn_state["model.pi_network.0.weight"].shape[0]))
print("reward_features=" + str(len(reward_state["feature_keys"])))
PY
}

preflight
if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1, preflight complete. Set DRY_RUN=0 to run evals."
  exit 0
fi

mkdir -p "$OUT_ROOT/baseline" "$OUT_ROOT/moe4" "$OUT_ROOT/mdn_seed0"
cd "$ROOT"

PYTHON="$PYTHON" OUT_DIR="$OUT_ROOT/baseline" \
  "$ROOT/code-scope/run_qwen3_4b_humaneval_baseline.sh" "${COMMON_ARGS[@]}"

PYTHON="$PYTHON" OUT_DIR="$OUT_ROOT/moe4" \
  CODE_SCOPE_TRANSITION_DIR="$MOE_TRANSITION_DIR" \
  CODE_SCOPE_REWARD_PATH="$REWARD_PATH" \
  "$ROOT/code-scope/run_code_scope_humaneval.sh" "${COMMON_ARGS[@]}"

PYTHON="$PYTHON" OUT_DIR="$OUT_ROOT/mdn_seed0" \
  CODE_SCOPE_TRANSITION_DIR="$MDN_TRANSITION_DIR" \
  CODE_SCOPE_REWARD_PATH="$REWARD_PATH" \
  "$ROOT/code-scope/run_code_scope_humaneval.sh" "${COMMON_ARGS[@]}"

"$PYTHON" "$ROOT/code-scope/summarize_humaneval.py" \
  "$OUT_ROOT/baseline/qwen3_4b_baseline.jsonl" \
  "$OUT_ROOT/moe4/qwen3_4b_scope.jsonl" \
  "$OUT_ROOT/mdn_seed0/qwen3_4b_scope.jsonl"
