"""
scope_mediq_runner.py

Integrates SCOPE's MCTS planner as the expert in mediQ's benchmark loop.

Two simulators (clearly separated):
  1. SCOPE's transition model  — inner simulator used inside MCTS for semantic-space planning
  2. mediQ's FactSelectPatient — outer simulator that gives real patient answers to SCOPE's chosen question

Usage:
  python scope_mediq_runner.py [--data_file PATH] [--output_filename PATH] [--max_questions N]
"""

import sys, os, json, re, glob, argparse, time, copy
import torch
import numpy as np
import random

# ---------- paths ----------
SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_DIR  = os.path.dirname(SRC_DIR)
SCOPE_DIR = os.path.join(REPO_DIR, 'convo-plan-SCOPE')

sys.path.insert(0, SRC_DIR)
sys.path.insert(0, SCOPE_DIR)

# ---------- mediQ imports ----------
from expert import Expert
from patient import FactSelectPatient
import expert_functions
import helper as mediq_helper        # we inject SCOPE's model into its cache below

# ---------- SCOPE imports ----------
from agent.Model import create_human_and_llm
from agent.Conversation import Conversation
from monte_carlo_tree_search.policy_agent import OnlineAgent
from monte_carlo_tree_search.qtable import AccumulatedRewardTable
from monte_carlo_tree_search.conversation_env import conversation_state
from transition_models.transition_model import TransitionModelMOE
from transition_models.qwen3_embedding import Qwen3Embedding
from reward.Embedding_Scope_Reward import Embedding_Scope_Reward


# ============================================================
#  CLI args
# ============================================================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str,
                        default=os.path.join(REPO_DIR, 'data/med_data/all_dev_convo.jsonl'))
    parser.add_argument('--output_filename', type=str,
                        default=os.path.join(SRC_DIR, 'results', 'scope_mediq.jsonl'))
    parser.add_argument('--max_questions', type=int, default=5)
    parser.add_argument('--max_examples', type=int, default=0,
                        help='Stop after this many examples (0 = all)')
    return parser.parse_args()


# ============================================================
#  Fix transition model paths
# ============================================================
def prepare_transition_models(cuda_q, transition_model_dir):
    for val_pth in glob.glob(os.path.join(transition_model_dir, '**', 'model_min_val.pth'), recursive=True):
        train_pth = val_pth.replace('model_min_val.pth', 'model_min_train.pth')
        if not os.path.exists(train_pth):
            os.symlink(val_pth, train_pth)
            print(f"[setup] symlinked model_min_val.pth -> model_min_train.pth in {transition_model_dir}")
    model = TransitionModelMOE(noise=0.005, cuda=cuda_q, transition_model_dir=transition_model_dir)
    return model


# ============================================================
#  GPU assignment (override via SCOPE_CUDA_LLM / SCOPE_CUDA_Q env vars)
# ============================================================
CUDA_LLM = int(os.environ.get("SCOPE_CUDA_LLM", "0"))
CUDA_Q   = int(os.environ.get("SCOPE_CUDA_Q",   "1"))


# ============================================================
#  Initialize SCOPE components once
# ============================================================
config_path = os.path.join(SCOPE_DIR, 'agent', 'mediq_qwen3_scope_config.yaml')

# Transition model: trained on Qwen3-4B 2560-dim embeddings.
# TransitionModelMOE expects .../<experiment>/seed_*/{human_llm,llm_human}/, not the seed dir itself.
TRANSITION_MODEL_DIR = os.path.join(
    SCOPE_DIR, 'transition_models', 'deterministic_train4k_qwen3'
)
REWARD_MODEL_PATH = os.path.join(SCOPE_DIR, 'reward', 'embedding_mediQ_reward_cumulative')

print("[init] Loading SCOPE LLM models (Qwen3-4B)…")
human_sim, human_eval, llm_agent = create_human_and_llm(config=config_path, cuda=CUDA_LLM)

print("[init] Loading Qwen3 embedding model (2560-dim, no projection)…")
# ── Embedding model: same Qwen3-4B used for training transition + reward MLP ──
embed_model = Qwen3Embedding(
    model_name="Qwen/Qwen3-4B",
    device_map=CUDA_Q,
    random_projection=None,   # native 2560-dim — matches trained transition + reward MLP
)
dim = embed_model.output_dim  # 2560

print("[init] Loading cumulative reward MLP (Embedding_Scope_Reward)…")
reward_function = Embedding_Scope_Reward(
    path_to_model=REWARD_MODEL_PATH,
    device_map=CUDA_Q,
)

print("[init] Loading Qwen3 transition model…")
# ── Simulator 1: SCOPE's transition model (inner MCTS planning) ──────────────
transition_model = prepare_transition_models(CUDA_Q, TRANSITION_MODEL_DIR)

print("[init] Building SCOPE OnlineAgent…")
semanticqfunction = AccumulatedRewardTable()
scope_agent = OnlineAgent(
    semanticqfunction,
    search_depth=8,
    mcts_time_limit=int(os.environ.get("SCOPE_MCTS_TIME", "30")),
    llm_agent=llm_agent,
    human_simulator=human_sim,           # SCOPE's inner human simulator
    reward_function_for_mcts=reward_function,
    search_space="semantic_space",
    transition_model=transition_model,   # SCOPE's learned transition model
    embedding_model=embed_model,
)

# ============================================================
#  Inject SCOPE's loaded HF model into mediQ's helper cache
#  so FactSelectPatient reuses it without loading a second copy.
#
#  ── Simulator 2: mediQ's FactSelectPatient (outer real patient) ────────────
# ============================================================
class _SCOPEModelAdapter:
    """Adapts SCOPE's Local_LLM to the interface expected by mediQ helper.py."""
    def __init__(self, local_llm):
        self._local_llm = local_llm
        self.use_vllm   = False
        self.use_api    = None
        self.args       = {"temperature": 0.6, "max_tokens": 2048, "top_p": 0.9, "top_logprobs": 0}

    def generate(self, messages):
        tokenizer = self._local_llm.tokenizer
        hf_model  = self._local_llm.model
        max_new   = self.args.get("max_tokens", 2048)

        tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_attention_mask=True, return_dict=True,
            **mediq_helper._chat_template_kwargs(MODEL_NAME),
        ).to(hf_model.device)

        with torch.no_grad():
            output = hf_model.generate(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                max_new_tokens=max_new,
                do_sample=True,
                temperature=self.args["temperature"],
                top_p=self.args["top_p"],
                pad_token_id=tokenizer.eos_token_id,
            )
        generated     = output[:, tokens["input_ids"].shape[-1]:]
        response_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        usage = {
            "input_tokens":  int(tokens["input_ids"].shape[-1]),
            "output_tokens": int(generated.shape[-1]),
        }
        print(f"[LLM OUTPUT]: {response_text}\n")
        return response_text, None, usage


# Register in helper's model cache — FactSelectPatient will find it here.
# Uses the same Qwen3-4B weights already loaded by llm_agent (no extra VRAM).
MODEL_NAME = "Qwen/Qwen3-4B"
mediq_helper.models[MODEL_NAME] = _SCOPEModelAdapter(llm_agent.model)
print("[init] Qwen3-4B injected into mediQ helper cache — FactSelectPatient ready.\n")


TRACE_JSONL = os.environ.get("SCOPE_TRACE_JSONL")



SCOPE_EXHAUSTIVE_TRACE_DEPTH = int(os.environ.get("SCOPE_EXHAUSTIVE_TRACE_DEPTH", "0"))
SCOPE_EXHAUSTIVE_MAX_PATHS = int(os.environ.get("SCOPE_EXHAUSTIVE_MAX_PATHS", "20000"))


def _summarize_values(values):
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def _semantic_reward(prev_state, next_state):
    return reward_function.get_reward(prev_state, None, next_state)


def _trace_exhaustive_candidate_rewards(convo, actions, max_depth=None, max_paths=None):
    """Trace true transition/reward rollouts per root candidate, independent of Q.

    max_depth counts semantic doctor/patient transitions after the root doctor action.
    depth=1 means root action -> simulated patient response only.
    depth=2 adds one simulated doctor action and patient response, etc.
    """
    if not actions:
        return []
    if max_depth is None:
        max_depth = SCOPE_EXHAUSTIVE_TRACE_DEPTH
    if max_paths is None:
        max_paths = SCOPE_EXHAUSTIVE_MAX_PATHS
    if max_depth <= 0:
        return []

    with torch.no_grad():
        state_embedding = embed_model.embed(convo).cpu().numpy().reshape(-1)
    state_tuple = tuple(np.asarray(state_embedding, dtype=np.float32))

    rows = []
    action_deltas = []
    for action_idx, action_text in enumerate(actions):
        with torch.no_grad():
            action_embedding = embed_model.embed(convo + action_text).cpu().numpy().reshape(-1)
        action_delta_array = np.asarray(action_embedding - state_embedding, dtype=np.float32)
        action_delta = tuple(action_delta_array)
        action_deltas.append(action_delta_array)
        root_next_states = transition_model.transit(state_tuple, action_delta)

        frontier = []
        root_rewards = []
        for next_state in root_next_states:
            reward = _semantic_reward(state_tuple, next_state)
            root_rewards.append(reward)
            frontier.append((next_state, reward, [reward]))

        truncated = False
        for _depth in range(2, max_depth + 1):
            next_frontier = []
            for current_state, cumulative_reward, path_rewards in frontier:
                llm_actions = transition_model.sample_actions(current_state)
                for llm_action in llm_actions:
                    human_states = transition_model.transit(current_state, llm_action)
                    for human_state in human_states:
                        step_reward = _semantic_reward(current_state, human_state)
                        next_frontier.append(
                            (
                                human_state,
                                cumulative_reward + step_reward,
                                path_rewards + [step_reward],
                            )
                        )
                        if len(next_frontier) >= max_paths:
                            truncated = True
                            break
                    if truncated:
                        break
                if truncated:
                    break
            frontier = next_frontier
            if truncated or not frontier:
                break

        cumulative_rewards = [item[1] for item in frontier] if frontier else root_rewards
        rows.append(
            {
                "index": action_idx,
                "action": action_text,
                "action_delta_norm": float(np.linalg.norm(action_delta_array)),
                "action_delta_first10": [float(x) for x in action_delta_array[:10]],
                "action_delta_summary": {
                    "min": float(action_delta_array.min()),
                    "max": float(action_delta_array.max()),
                    "mean": float(action_delta_array.mean()),
                    "std": float(action_delta_array.std()),
                },
                "root_next_state_count": len(root_next_states),
                "root_reward_summary": _summarize_values(root_rewards),
                "leaf_depth": max_depth,
                "leaf_path_count": len(cumulative_rewards),
                "leaf_truncated": truncated,
                "leaf_cumulative_reward_summary": _summarize_values(cumulative_rewards),
            }
        )

    for i in range(len(rows)):
        distances = {}
        for j in range(len(rows)):
            if i == j:
                continue
            distances[str(j)] = float(np.linalg.norm(action_deltas[i] - action_deltas[j]))
        rows[i]["pairwise_action_delta_distance"] = distances
    return rows



def _q_readout_diagnostics(qfunction, semantic_state, action_semantics):
    if not action_semantics:
        return {"num_actions": 0}
    # AccumulatedRewardTable: just report the per-action accumulated average rewards.
    if not hasattr(qfunction, "q_network"):
        qs = qfunction.get_qs(semantic_state, action_semantics)
        return {
            "num_actions": len(action_semantics),
            "type": "accumulated_reward_table",
            "q_values": [float(q) for q in qs],
        }
    with torch.no_grad():
        inputs = torch.cat(
            [qfunction.merge(semantic_state, action).to(qfunction.cuda) for action in action_semantics],
            dim=0,
        )
        rows = {
            "num_actions": int(inputs.shape[0]),
            "input_norms": [float(torch.linalg.vector_norm(x).detach().cpu()) for x in inputs],
            "input_pairwise_distances": {},
            "layers": [],
        }
        for i in range(inputs.shape[0]):
            for j in range(i + 1, inputs.shape[0]):
                rows["input_pairwise_distances"][f"{i}-{j}"] = float(
                    torch.linalg.vector_norm(inputs[i] - inputs[j]).detach().cpu()
                )

        x = inputs
        for idx, layer in enumerate(qfunction.q_network):
            x = layer(x)
            layer_info = {
                "index": idx,
                "type": layer.__class__.__name__,
                "output_norms": [float(torch.linalg.vector_norm(row).detach().cpu()) for row in x],
                "output_pairwise_distances": {},
            }
            if isinstance(layer, torch.nn.ReLU):
                layer_info["nonzero_counts"] = [int((row != 0).sum().detach().cpu()) for row in x]
            for i in range(x.shape[0]):
                for j in range(i + 1, x.shape[0]):
                    layer_info["output_pairwise_distances"][f"{i}-{j}"] = float(
                        torch.linalg.vector_norm(x[i] - x[j]).detach().cpu()
                    )
            rows["layers"].append(layer_info)
        rows["final_outputs"] = [float(v.detach().cpu()) for v in x.reshape(-1)]
        return rows


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _answer_reward_probe(convo, options):
    """Score committing to each answer letter from the current text conversation."""
    with torch.no_grad():
        before = embed_model.embed(convo)
        before_value = reward_function.value(before)
        rows = []
        for letter in options:
            answer_convo = convo + f"FINAL ANSWER: {letter}"
            after = embed_model.embed(answer_convo)
            after_value = reward_function.value(after)
            rows.append(
                {
                    "answer": letter,
                    "option": options.get(letter),
                    "value_before": before_value,
                    "value_after": after_value,
                    "reward": after_value - before_value,
                }
            )
    return rows


def _write_scope_trace(row):
    if not TRACE_JSONL:
        return
    out_dir = os.path.dirname(TRACE_JSONL)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(TRACE_JSONL, "a") as f:
        f.write(json.dumps(_jsonable(row)) + "\n")


class _ScaleExpertActionGenerator:
    """Generate SCOPE candidates by replaying the exact ScaleExpert path."""

    def __init__(self):
        self.expert = None
        self.patient_state = None
        self.last_candidate_records = []
        self.cached_actions = None
        self.num_passes = int(os.environ.get("SCOPE_CANDIDATE_PASSES", "5"))

    def configure(self, expert, patient_state, confidence_result):
        self.expert = expert
        self.patient_state = patient_state
        self.confidence_result = confidence_result
        self.last_candidate_records = []
        self.cached_actions = None
        model_cache = mediq_helper.models.get(MODEL_NAME)
        if model_cache is not None:
            model_cache.args.update(expert.get_inference_kwargs())

    def sample_actions(self, _prompt, **_kwargs):
        if self.expert is None or self.patient_state is None:
            raise RuntimeError("SCOPE action generator was used before configure().")
        if self.cached_actions is not None:
            return self.cached_actions

        actions = []
        records = []
        confidence_result = self.confidence_result
        for pass_idx in range(self.num_passes):
            record = {
                "pass": pass_idx,
                "abstain": confidence_result.get("abstain"),
                "confidence": confidence_result.get("confidence"),
                "letter_choice": confidence_result.get("letter_choice"),
                "shadow_answer": confidence_result.get("shadow_answer"),
            }

            if confidence_result.get("abstain"):
                question_response = self.expert.ask_question(
                    self.patient_state, copy.deepcopy(confidence_result["messages"])
                )
                action = question_response.get("atomic_question")
                record["type"] = "question"
                record["action"] = action
                record["question_rationale"] = question_response.get("question_rationale")
                record["usage"] = question_response.get("usage")
                if action and action not in actions:
                    actions.append(action)
            else:
                letter, usage = expert_functions.final_choice_with_options(
                    self.patient_state,
                    self.expert.inquiry,
                    self.expert.options,
                    **self.expert.get_inference_kwargs(),
                )
                action = f"FINAL ANSWER: {letter}" if letter else None
                record["type"] = "choice"
                record["action"] = action
                record["letter_choice"] = letter
                record["usage"] = usage
                if action and action not in actions:
                    actions.append(action)

            records.append(record)

        if not actions:
            fallback_letter = confidence_result.get("letter_choice") or "A"
            actions.append(f"FINAL ANSWER: {fallback_letter}")
        self.last_candidate_records = records
        self.cached_actions = actions
        return actions


scale_action_generator = _ScaleExpertActionGenerator()
scope_agent.llm_agent = scale_action_generator


# ============================================================
#  Expert: wraps SCOPE's OnlineAgent as a mediQ Expert
# ============================================================
class SCOPEExpert(Expert):

    def set_trace_context(self, **context):
        self._trace_context = context

    def _build_conversation(self, patient_state):
        # This is only the semantic state consumed by SCOPE's reward/transition
        # models. Candidate generation below uses the exact ScaleExpert prompt
        # functions, not this representation.
        first_msg = patient_state["initial_info"]
        convo = Conversation(first_msg, start_with_human=True)
        for qa in patient_state["interaction_history"]:
            convo = convo.add_llm_response(qa["question"], copy=False)
            convo = convo.add_human_response(qa["answer"], copy=False)
        return convo

    def _parse_action(self, text):
        m = re.search(r'FINAL\s*ANSWER\s*[:\-]?\s*([ABCD])', text.upper())
        if m:
            return "choice", m.group(1)
        m = re.search(r'(?<![A-Z])([ABCD])(?![A-Z])\s*$', text.strip().upper())
        if m and len(text.strip()) < 30:
            return "choice", m.group(1)
        return "question", None

    def respond(self, patient_state):
        convo  = self._build_conversation(patient_state)
        last_r = convo.full_convo[-1] if convo.full_convo else ""
        state  = conversation_state(last_r, convo)
        state.depth = len(patient_state['interaction_history']) * 2 + 1

        confidence_result = expert_functions.scale_abstention_decision(
            **self.get_abstain_kwargs(patient_state)
        )
        scope_agent.qfunction.reset()
        scale_action_generator.configure(self, patient_state, confidence_result)
        results = {}
        turn_index = len(patient_state["interaction_history"])
        answer_probe = _answer_reward_probe(convo, self.options)
        action  = scope_agent.generate_action(state, results=results)
        print(f"\n  [SCOPE action]: {action!r}")

        possible_actions = results.get("possible_actions") or []
        possible_q_values = results.get("possible_actions_reward") or []
        greedy_rewards = results.get("greedy_rewards") or []
        selected_idx = results.get("selected_action_index")

        q_readout_diagnostics = None
        if possible_actions:
            with torch.no_grad():
                state_embedding = embed_model.embed(convo).cpu().numpy().reshape(-1)
            semantic_state_for_diag = copy.deepcopy(state)
            semantic_state_for_diag.conversation = tuple(np.asarray(state_embedding, dtype=np.float32))
            action_semantics_for_diag = []
            for candidate_action in possible_actions:
                with torch.no_grad():
                    action_embedding = embed_model.embed(convo + candidate_action).cpu().numpy().reshape(-1)
                action_semantics_for_diag.append(
                    tuple(np.asarray(action_embedding - state_embedding, dtype=np.float32))
                )
            q_readout_diagnostics = _q_readout_diagnostics(
                scope_agent.qfunction, semantic_state_for_diag, action_semantics_for_diag
            )

        exhaustive_reward_trace = _trace_exhaustive_candidate_rewards(
            convo, possible_actions
        )
        trace_row = {
            **getattr(self, "_trace_context", {}),
            "turn_index": turn_index,
            "state_depth": state.depth,
            "num_observed_questions": len(patient_state["interaction_history"]),
            "answer_rewards": answer_probe,
            "candidate_generation": scale_action_generator.last_candidate_records,
            "greedy_rewards": results.get("greedy_rewards"),
            "greedy_action_index": results.get("greedy_action_index"),
            "possible_actions": possible_actions,
            "possible_actions_reward": possible_q_values,
            "selected_action_index": selected_idx,
            "selected_action": action,
            "exhaustive_reward_trace": exhaustive_reward_trace,
            "q_readout_diagnostics": q_readout_diagnostics,
            "mcts_trace": results.get("mcts_trace", []),
        }
        _write_scope_trace(trace_row)

        candidate_rewards = []
        for idx, candidate_action in enumerate(possible_actions):
            candidate_rewards.append({
                "index": idx,
                "action": candidate_action,
                "embedding_reward": greedy_rewards[idx] if idx < len(greedy_rewards) else None,
                "mcts_q": possible_q_values[idx] if idx < len(possible_q_values) else None,
                "selected": idx == selected_idx,
            })

        base_meta = {
            "raw_action": action,
            "confidence": confidence_result.get("confidence"),
            "confidence_rationale": confidence_result.get("confidence_rationale"),
            "shadow_answer": confidence_result.get("shadow_answer"),
            "boxed_answer": confidence_result.get("boxed_answer"),
            "scope_candidate_generation": scale_action_generator.last_candidate_records,
            "scope_candidate_rewards": _jsonable(candidate_rewards),
            "scope_selected_action_index": _jsonable(selected_idx),
            "usage": confidence_result.get("usage", {"input_tokens": 0, "output_tokens": 0}),
        }

        action_type, letter = self._parse_action(action)
        if action_type == "choice":
            return {**base_meta, "type": "choice", "letter_choice": letter}
        else:
            return {**base_meta, "type": "question", "question": action,
                    "letter_choice": confidence_result.get("letter_choice") or list(self.options.keys())[0]}


# ============================================================
#  Interaction loop
# ============================================================
class RunArgs:
    use_vllm      = False    # FactSelectPatient reuses the injected HF Qwen3 model
    use_api       = None
    temperature   = 0.6
    max_tokens    = 2048
    top_p         = 0.9
    top_logprobs  = 0
    api_account   = "mediQ"
    patient_model = MODEL_NAME   # "Qwen/Qwen3-4B" — found in helper.models cache


def run_interaction(sample, max_questions):
    run_args = RunArgs()
    run_args.max_questions = max_questions

    expert  = SCOPEExpert(run_args, sample["question"], sample["options"])
    # ── Simulator 2: mediQ's FactSelectPatient ────────────────────────────────
    patient = FactSelectPatient(run_args, sample)

    temp_choices    = []
    temp_additional = []

    while len(patient.get_questions()) < max_questions:
        patient_state = patient.get_state()
        resp = expert.respond(patient_state)
        temp_additional.append({k: v for k, v in resp.items()
                                 if k not in ["type", "letter_choice", "question"]})

        if resp["type"] == "question":
            temp_choices.append(resp["letter_choice"])
            patient_answer = patient.respond(resp["question"])
            print(f"  [Patient]: {patient_answer[:200]}")

        elif resp["type"] == "choice":
            temp_choices.append(resp["letter_choice"])
            return resp["letter_choice"], patient.get_questions(), patient.get_answers(), \
                   temp_choices, temp_additional
        else:
            raise ValueError(f"Unknown response type: {resp['type']}")

    print(f"\n  [max questions ({max_questions}) reached — forcing final answer]")
    patient_state = patient.get_state()
    resp = expert.respond(patient_state)
    final = resp["letter_choice"]
    temp_choices.append(final)
    temp_additional.append({k: v for k, v in resp.items()
                             if k not in ["type", "letter_choice", "question"]})
    return final, patient.get_questions(), patient.get_answers(), temp_choices, temp_additional


# ============================================================
#  Main
# ============================================================
if __name__ == "__main__":
    args = get_args()

    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    print(f"[run] data         : {args.data_file}")
    print(f"[run] output       : {args.output_filename}")
    print(f"[run] max_questions: {args.max_questions}")
    print(f"[run] inner sim    : SCOPE transition model (semantic space)")
    print(f"[run] outer sim    : mediQ FactSelectPatient\n")

    with open(args.data_file) as f:
        data = [json.loads(line) for line in f]

    os.makedirs(os.path.dirname(args.output_filename), exist_ok=True)
    processed_ids = set()
    correct_history, timeout_history, turn_lengths = [], [], []
    if os.path.exists(args.output_filename):
        with open(args.output_filename) as f:
            for line in f:
                rec = json.loads(line)
                processed_ids.add(rec["id"])
                correct_history.append(rec["interactive_system"]["correct"])
                timeout_history.append(
                    len(rec["interactive_system"]["intermediate_choices"]) > args.max_questions)
                turn_lengths.append(rec["interactive_system"]["num_questions"])

    for sample in data:
        if args.max_examples and len(correct_history) >= args.max_examples:
            break
        pid = sample["id"]
        if pid in processed_ids:
            print(f"Skipping {pid} (already processed)")
            continue

        print(f"\n{'='*65}")
        print(f"Sample {pid} | GT: {sample['answer_idx']} ({sample['answer']})")
        print(f"{'='*65}")

        t0 = time.time()
        letter, questions, answers, choices, extra = run_interaction(sample, args.max_questions)
        elapsed = time.time() - t0

        correct = letter == sample["answer_idx"]
        correct_history.append(correct)
        timeout_history.append(len(choices) > args.max_questions)
        turn_lengths.append(len(questions))

        n = len(correct_history)
        accuracy     = sum(correct_history) / n
        timeout_rate = sum(timeout_history) / n
        avg_turns    = sum(turn_lengths)    / n
        print(f"\n  Predicted: {letter} | GT: {sample['answer_idx']} | Correct: {correct} | {elapsed:.1f}s")
        print(f"  Running — Acc: {accuracy:.3f}  Timeout: {timeout_rate:.3f}  "
              f"AvgTurns: {avg_turns:.1f}  ({n}/{len(data)})")

        output_dict = {
            "id": pid,
            "interactive_system": {
                "correct": correct,
                "letter_choice": letter,
                "questions": questions,
                "answers": answers,
                "num_questions": len(questions),
                "intermediate_choices": choices,
                "temp_additional_info": extra,
            },
            "info": {
                "initial_info": (sample["context"][0] if sample["context"] else "") if isinstance(sample["context"], list)
                                else sample["context"].split(". ")[0],
                "correct_answer": sample["answer"],
                "correct_answer_idx": sample["answer_idx"],
                "question": sample["question"],
                "options": sample["options"],
                "context": sample["context"],
                "facts": sample.get("atomic_facts"),
            }
        }
        with open(args.output_filename, "a") as f:
            f.write(json.dumps(output_dict) + "\n")

    n = len(correct_history)
    accuracy     = sum(correct_history) / n if n else 0
    timeout_rate = sum(timeout_history) / n if n else 0
    avg_turns    = sum(turn_lengths)    / n if n else 0
    print(f"\n{'='*65}")
    print(f"FINAL SUMMARY  ({len(data)} examples)")
    print(f"  Accuracy     : {sum(correct_history)} / {n} = {accuracy:.4f}")
    print(f"  Timeout Rate : {sum(timeout_history)} / {n} = {timeout_rate:.4f}")
    print(f"  Avg. Turns   : {avg_turns:.2f}")
    print(f"  Output       : {args.output_filename}")
    print(f"{'='*65}")
