import json
import os
import textwrap
import time
import logging
import copy
from args import get_args
from patient import Patient
import expert_functions
import importlib

def _block(label, text, width=80, indent=4):
    pad = " " * indent
    if text is None:
        return f"{pad}{label}: (none)\n"
    text = str(text)
    if "\n" not in text and len(text) <= width - indent - len(label) - 2:
        return f"{pad}{label}: {text}\n"
    wrapped = textwrap.indent(textwrap.fill(text.strip(), width=width - indent), pad + "  ")
    return f"{pad}{label}:\n{wrapped}\n"


def _block_lines(label, text, width=80, indent=4):
    """Like _block but preserves newlines (wrap each physical line separately)."""
    pad = " " * indent
    inner_w = max(20, width - indent - 2)
    if text is None:
        return f"{pad}{label}: (none)\n"
    out = [f"{pad}{label}:"]
    for ln in str(text).splitlines():
        s = ln.strip()
        if not s:
            out.append(pad + "  ")
            continue
        chunks = textwrap.wrap(s, width=inner_w) or [s]
        for ch in chunks:
            out.append(pad + "  " + ch)
    return "\n".join(out) + "\n"


def _scope_candidate_reward_lines(meta, indent=4, width=80):
    rows = (meta or {}).get("scope_candidate_rewards") or []
    if not rows:
        return []
    pad = " " * indent
    lines = [f"{pad}SCOPE Candidate Scores:"]
    inner_w = max(20, width - indent - 6)
    for row in rows:
        idx = row.get("index")
        mcts_q = row.get("mcts_q")
        selected = ""
        action = str(row.get("action") or "")
        mcts_q_str = f"{mcts_q:.4f}" if isinstance(mcts_q, float) else str(mcts_q)
        prefix = f"{pad}  [{idx}] mean_reward={mcts_q_str}{selected}: "
        wrapped = textwrap.wrap(action, width=inner_w) or [""]
        lines.append(prefix + wrapped[0])
        for chunk in wrapped[1:]:
            lines.append(" " * len(prefix) + chunk)
    return lines


def _format_full_context(context):
    """Turn sample context (list or str) into a single string for logging."""
    if context is None:
        return None
    if isinstance(context, list):
        if not context:
            return "(empty list)"
        return "\n".join(f"[{i}] {item}" for i, item in enumerate(context, start=1))
    return str(context).strip() or "(empty)"

def setup_logger(name, file):
    if not file: return None
    logger = logging.getLogger(name)
    handler = logging.FileHandler(file, mode='a')
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def log_info(message, print_to_std=False):
    if history_logger: history_logger.info(message)
    if detail_logger: detail_logger.info(message)
    if print_to_std: print(message + "\n")

def load_data(filename):
    with open(filename, "r") as json_file:
        json_list = list(json_file)
    data = [json.loads(line) for line in json_list]
    data = {item['id']: item for item in data}
    return data

def add_mode_suffix(filename, option_mode):
    if not filename or option_mode == "yes-option":
        return filename
    base, ext = os.path.splitext(filename)
    suffix = f"_{option_mode}"
    if base.endswith(suffix):
        return filename
    return f"{base}{suffix}{ext}"

def _continue_leaf_chain(expert_system, patient, sample, leaf_depth):
    """Continue a depth-cap leaf as a sequential abstain chain.

    Picks up from the leaf's accumulated patient state and asks follow-up
    questions one at a time (same loop as the no-branch path) until the
    expert is confident or the total number of doctor questions hits
    args.max_questions. Returns a dict with the leaf's updated final state.
    """
    total_questions_cap = max(args.max_questions, leaf_depth)
    last_response = None

    while len(patient.get_questions()) < total_questions_cap:
        patient_state = patient.get_state()
        response_dict = expert_system.respond(patient_state)
        last_response = response_dict

        if response_dict["type"] == "question":
            patient.respond(response_dict["question"])
            continue
        if response_dict["type"] == "choice":
            break
        raise ValueError("Invalid response type from expert_system during leaf continuation.")
    else:
        patient_state = patient.get_state()
        last_response = expert_system.respond(patient_state)

    letter_choice = last_response.get("letter_choice")
    if (
        last_response.get("type") == "choice"
        and args.option_mode == "option-in-the-end"
    ):
        letter_choice, _ = expert_functions.final_choice_with_options(
            patient.get_state(), sample["question"], sample["options"],
            **expert_system.get_inference_kwargs(),
        )

    return {
        "confidence": last_response.get("confidence"),
        "confidence_rationale": last_response.get("confidence_rationale"),
        "shadow_answer": last_response.get("shadow_answer"),
        "boxed_answer": last_response.get("boxed_answer"),
        "letter_choice": letter_choice,
        "final_answer": letter_choice,
        "questions": patient.get_questions(),
        "answers": patient.get_answers(),
    }


def _aggregate_leaf_vote(leaves):
    """Pick a canonical letter_choice from the per-leaf final_answers.

    Majority vote across leaves. Ties broken by the leaf with the highest
    confidence (preserves first-in-traversal on confidence ties). Returns
    None if no leaf has a parseable letter.
    """
    valid = [n for n in leaves if n.get("final_answer") in {"A", "B", "C", "D"}]
    if not valid:
        return None
    from collections import Counter
    counts = Counter(n["final_answer"] for n in valid)
    top_count = counts.most_common(1)[0][1]
    tied = {letter for letter, c in counts.items() if c == top_count}
    if len(tied) == 1:
        return next(iter(tied))
    best = max(
        (n for n in valid if n["final_answer"] in tied),
        key=lambda n: (n.get("confidence") or 0.0),
    )
    return best["final_answer"]


def run_branch_interaction(expert_class, patient_class, sample):
    """Tree-then-chain conversation explorer.

    Phase 1 (tree, depth ≤ args.branch_depth): at each node compute confidence.
      If confident → LEAF, commit. Else expand top_k follow-up questions and
      recurse. Hitting branch_depth without confidence forces a LEAF that is
      flagged for continuation.
    Phase 2 (per-leaf chain): every leaf that was NOT confidence-triggered
      (depth cap or top-k parse failure) continues as a sequential abstain
      chain on its own deep-copied patient, up to args.max_questions total
      doctor questions. Confidence-triggered leaves stay as-is.
    Aggregation: canonical letter_choice = majority vote across all leaves'
      final answers (tiebreak: highest confidence). The lenient "any leaf
      correct" benchmark grader is unchanged.
    """
    expert_system = expert_class(args, sample["question"], sample["options"])
    patient_system = patient_class(args, sample)
    all_nodes = []
    # Patient deep-copies held alongside each LEAF node for Phase-2 continuation.
    leaf_patients = {}

    def _recurse(patient, branch_id, depth):
        label = branch_id if branch_id else "Root"
        log_info(f"==================== Branch {label} | Depth {depth} ====================")
        abstain_kwargs = expert_system.get_abstain_kwargs(patient.get_state())
        abstain_result = expert_functions.scale_abstention_decision(**abstain_kwargs)
        log_info(f"[Branch {label}] confidence={abstain_result['confidence']}, abstain={abstain_result['abstain']}")

        confident = not abstain_result["abstain"]
        at_max = depth >= args.branch_depth

        if confident or at_max:
            letter_choice = abstain_result["letter_choice"]
            if confident and args.option_mode == "option-in-the-end":
                letter_choice, _ = expert_functions.final_choice_with_options(
                    patient.get_state(), sample["question"], sample["options"],
                    **expert_system.get_inference_kwargs(),
                )
            node = {
                "branch_id": branch_id or "root",
                "depth": depth,
                "is_leaf": True,
                "leaf_reason": "confidence" if confident else "depth",
                "confidence": abstain_result["confidence"],
                "confidence_rationale": abstain_result.get("confidence_rationale"),
                "shadow_answer": abstain_result.get("shadow_answer"),
                "letter_choice": letter_choice,
                "boxed_answer": abstain_result.get("boxed_answer"),
                "final_answer": letter_choice,
                "top_k_raw": None,
                "top_k_questions": None,
                "questions": patient.get_questions(),
                "answers": patient.get_answers(),
            }
            all_nodes.append(node)
            leaf_patients[id(node)] = patient
            log_info(
                f"[Branch {label}] LEAF ({node['leaf_reason']}) → {letter_choice}"
            )
            return

        q_kwargs = dict(expert_system.get_inference_kwargs(),
                        model_name=args.expert_model_question_generator or args.expert_model)
        top_k_result = expert_functions.top_k_question_generation(
            patient_state=patient.get_state(),
            inquiry=sample["question"],
            options_dict=sample["options"],
            option_mode=args.option_mode,
            top_k=args.branch_top_k,
            **q_kwargs,
        )
        node = {
            "branch_id": branch_id or "root",
            "depth": depth,
            "is_leaf": False,
            "leaf_reason": None,
            "confidence": abstain_result["confidence"],
            "confidence_rationale": abstain_result.get("confidence_rationale"),
            "shadow_answer": abstain_result.get("shadow_answer"),
            "letter_choice": abstain_result["letter_choice"],
            "boxed_answer": abstain_result.get("boxed_answer"),
            "final_answer": None,
            "top_k_raw": top_k_result["raw_response"],
            "top_k_questions": top_k_result["questions"],
            "questions": patient.get_questions(),
            "answers": patient.get_answers(),
        }
        if not top_k_result["questions"]:
            node["is_leaf"] = True
            node["leaf_reason"] = "parse_fail"
            node["final_answer"] = node["letter_choice"]
            all_nodes.append(node)
            leaf_patients[id(node)] = patient
            log_info(f"[Branch {label}] top-k parse failed → forced LEAF")
            return

        all_nodes.append(node)
        for i, q_item in enumerate(top_k_result["questions"]):
            child_patient = copy.deepcopy(patient)
            child_patient.respond(q_item["question"])
            child_id = f"{branch_id}-{i + 1}" if branch_id else str(i + 1)
            _recurse(child_patient, child_id, depth + 1)

    _recurse(patient_system, "", 0)

    leaves = [n for n in all_nodes if n["is_leaf"]]

    # Phase 2 — continue every non-confidence leaf as a sequential chain.
    for leaf in leaves:
        if leaf["leaf_reason"] == "confidence":
            continue
        if leaf["depth"] >= args.max_questions:
            continue
        leaf_patient = leaf_patients.get(id(leaf))
        if leaf_patient is None:
            continue
        log_info(
            f"[Branch {leaf['branch_id']}] continuing from depth={leaf['depth']} "
            f"(reason={leaf['leaf_reason']}) up to max_questions={args.max_questions}"
        )
        updated = _continue_leaf_chain(expert_system, leaf_patient, sample, leaf["depth"])
        leaf["confidence"] = updated["confidence"]
        leaf["confidence_rationale"] = updated["confidence_rationale"]
        leaf["shadow_answer"] = updated["shadow_answer"]
        leaf["boxed_answer"] = updated["boxed_answer"]
        leaf["letter_choice"] = updated["letter_choice"]
        leaf["final_answer"] = updated["final_answer"]
        leaf["questions"] = updated["questions"]
        leaf["answers"] = updated["answers"]

    sample_info = {
        "initial_info": patient_system.initial_info,
        "correct_answer": sample["answer"],
        "correct_answer_idx": sample["answer_idx"],
        "question": sample["question"],
        "options": sample["options"],
        "context": sample["context"],
        "facts": patient_system.facts,
        "branches": all_nodes,
    }

    voted_letter = _aggregate_leaf_vote(leaves)
    first_leaf = leaves[0] if leaves else all_nodes[-1]
    letter_choice = voted_letter if voted_letter is not None else first_leaf.get("final_answer")
    # Surface aggregation details for downstream logging/analysis.
    sample_info["aggregation"] = {
        "rule": "majority_vote_then_max_confidence",
        "voted_letter": voted_letter,
        "leaf_finals": [n.get("final_answer") for n in leaves],
        "leaf_confidences": [n.get("confidence") for n in leaves],
        "leaf_reasons": [n.get("leaf_reason") for n in leaves],
    }
    # Report the deepest leaf's Q&A as the "primary" chain for the JSONL row
    # (used by interactive_system.questions/answers); fall back to first leaf.
    primary_leaf = max(leaves, key=lambda n: len(n["questions"])) if leaves else first_leaf
    questions = primary_leaf["questions"]
    answers = primary_leaf["answers"]
    temp_choice_list = [n.get("final_answer") for n in leaves]
    return letter_choice, questions, answers, temp_choice_list, [], sample_info


def write_branching_convo_log(filename, pid, sample_info, is_correct, letter_choice):
    correct_str = "CORRECT" if is_correct else "WRONG"
    opts = "  ".join(f"{k}: {v}" for k, v in sample_info["options"].items())
    lines = []
    lines.append("=" * 80)
    lines.append(f"Patient #{pid}  |  {correct_str}  |  Predicted: {letter_choice}  |  True: {sample_info['correct_answer_idx']} ({sample_info['correct_answer']})")
    lines.append("=" * 80)
    lines.append(_block("Initial", sample_info["initial_info"]))
    lines.append(_block("Question", sample_info["question"]))
    if args.option_mode == "yes-option":
        lines.append(f"    Options: {opts}\n")

    for node in sample_info["branches"]:
        bid = node["branch_id"]
        depth = node["depth"]
        is_leaf = node["is_leaf"]
        label = "Root" if bid == "root" else f"Branch {bid}"
        kind = "LEAF" if is_leaf else "BRANCHING POINT"
        lines.append(f"  {'─' * 22} {label} | Depth {depth} | {kind} {'─' * 22}")

        if not node["questions"]:
            lines.append("    (no Q&A yet at this point)\n")
        else:
            for i, (q, a) in enumerate(zip(node["questions"], node["answers"])):
                new_marker = " ← new" if i == len(node["questions"]) - 1 and depth > 0 else ""
                lines.append(f"    Turn {i + 1}{new_marker}:")
                lines.append(_block("Doctor Q", q, indent=6))
                lines.append(_block("Patient", a, indent=6))

        lines.append(f"    Confidence: {node['confidence']}")
        if node.get("confidence_rationale"):
            lines.append(_block("Confidence Rationale", node["confidence_rationale"]))
        if node.get("shadow_answer"):
            lines.append(_block("Shadow Answer", node["shadow_answer"]))
        if args.option_mode != "yes-option" and node.get("boxed_answer"):
            lines.append(_block("Boxed Answer", node["boxed_answer"]))

        if not is_leaf:
            lines.append(f"\n    [TOP-{args.branch_top_k} PROPOSALS — not in doctor view]")
            if node.get("top_k_raw"):
                lines.append(_block_lines("Raw", node["top_k_raw"]))
            for q_item in (node.get("top_k_questions") or []):
                lines.append(f"    Q{q_item['rank']}:")
                if q_item.get("reason"):
                    lines.append(_block("Reason", q_item["reason"], indent=6))
                lines.append(_block("Question", q_item["question"], indent=6))
        else:
            final = node.get("final_answer") or node.get("letter_choice")
            lines.append(f"    → Final Answer: {final}\n")

    out_dir = os.path.dirname(filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(filename, 'a') as f:
        f.write("\n".join(lines) + "\n")


def write_branching_doctor_log(filename, pid, sample_info, is_correct, letter_choice, judgment, judgment_rationale):
    correct_str = "CORRECT" if is_correct else "WRONG"
    opts = "  ".join(f"{k}: {v}" for k, v in sample_info["options"].items())
    lines = []
    lines.append("=" * 80)
    lines.append(f"Patient #{pid}  |  {correct_str}  |  Predicted: {letter_choice}  |  True: {sample_info['correct_answer_idx']} ({sample_info['correct_answer']})")
    lines.append("=" * 80)
    lines.append(_block("Initial", sample_info["initial_info"]))
    lines.append(_block("Question", sample_info["question"]))
    lines.append(_block_lines(
        "Full context (all segments)",
        _format_full_context(sample_info.get("context")),
    ))
    if args.option_mode == "yes-option":
        lines.append(f"    Options: {opts}\n")

    leaves = [n for n in sample_info["branches"] if n["is_leaf"]]
    for node in leaves:
        bid = node["branch_id"]
        final = node.get("final_answer") or node.get("letter_choice")
        branch_correct = final == sample_info["correct_answer_idx"]
        bc_str = "CORRECT" if branch_correct else "WRONG"
        lines.append(f"\n  {'─' * 18} Branch {bid} | {bc_str} | Final: {final} {'─' * 18}")
        for i, (q, a) in enumerate(zip(node["questions"], node["answers"])):
            lines.append(f"  --- Turn {i + 1} " + "-" * 60)
            lines.append(_block("Doctor", q))
            lines.append(_block("Patient", a))
        lines.append(f"  → Final Answer: {final}")
        if judgment is not None and bid == leaves[0]["branch_id"]:
            lines.append(f"  → Judgment: {judgment}  |  {judgment_rationale}")

    lines.append("")
    out_dir = os.path.dirname(filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(filename, 'a') as f:
        f.write("\n".join(lines) + "\n")


def main():
    args.output_filename = add_mode_suffix(args.output_filename, args.option_mode)
    args.convo_log_filename = add_mode_suffix(args.convo_log_filename, args.option_mode)
    args.doctor_log_filename = add_mode_suffix(args.doctor_log_filename, args.option_mode)

    if args.overwrite and os.path.exists(args.output_filename):
        open(args.output_filename, 'w').close()
    if args.overwrite and args.convo_log_filename and os.path.exists(args.convo_log_filename):
        open(args.convo_log_filename, 'w').close()

    if os.path.exists(args.output_filename):
        with open(args.output_filename, "r") as f:
            lines = f.readlines()
        output_data = [json.loads(line) for line in lines]
        if len(lines) == 0: processed_ids = []
        else: processed_ids = {sample["id"]: {"correct": sample["interactive_system"]["correct"],
                                              "timeout": len(sample["interactive_system"]["intermediate_choices"]) > args.max_questions,
                                              "turns": sample["interactive_system"]["num_questions"]}
                                for sample in output_data}
    else:
        processed_ids = []

    expert_module = importlib.import_module(args.expert_module)
    expert_class = getattr(expert_module, args.expert_class)
    patient_module = importlib.import_module(args.patient_module)
    patient_class = getattr(patient_module, args.patient_class)
    
    patient_data_path = os.path.join(args.data_dir, args.dev_filename)
    patient_data = load_data(patient_data_path)
    if args.max_examples > 0:
        patient_data = dict(list(patient_data.items())[:args.max_examples])

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_idx < 0 or args.shard_idx >= args.num_shards:
        raise ValueError("--shard_idx must be in [0, num_shards)")
    if args.num_shards > 1:
        patient_data = dict(
            (pid, sample)
            for idx, (pid, sample) in enumerate(patient_data.items())
            if idx % args.num_shards == args.shard_idx
        )
        print(
            f"Running shard {args.shard_idx}/{args.num_shards}: "
            f"{len(patient_data)} examples after max_examples={args.max_examples}"
        )

    num_processed = 0
    correct_history, timeout_history, turn_lengths = [], [], []

    for pid, sample in patient_data.items():
        if pid in processed_ids:
            print(f"Skipping patient {pid} as it has already been processed.")
            correct_history.append(processed_ids[pid]["correct"])
            timeout_history.append(processed_ids[pid]["timeout"])
            turn_lengths.append(processed_ids[pid]["turns"])
            continue

        log_info(f"|||||||||||||||||||| PATIENT #{pid} | GT: {sample['answer_idx']} ({sample['answer']}) ||||||||||||||||||||")
        letter_choice, questions, answers, temp_choice_list, temp_additional_info, sample_info = run_patient_interaction(expert_class, patient_class, sample, pid=pid)

        judgment, judgment_rationale = None, None
        if args.option_mode == "no-option":
            final_meta = temp_additional_info[-1] if temp_additional_info else {}
            boxed_answer = final_meta.get("boxed_answer")
            if boxed_answer:
                judge_kwargs = dict(model_name=args.patient_model, use_vllm=args.use_vllm,
                                    use_api=args.use_api, temperature=args.temperature,
                                    max_tokens=args.max_tokens, top_p=args.top_p,
                                    tensor_parallel_size=args.tensor_parallel_size,
                                    batch_size=args.batch_size,
                                    gpu_memory_utilization=getattr(args, "gpu_memory_utilization", None),
                                    vllm_max_model_len=getattr(args, "vllm_max_model_len", None),
                                    vllm_max_num_seqs=getattr(args, "vllm_max_num_seqs", None),
                                    vllm_enforce_eager=getattr(args, "vllm_enforce_eager", False))
                judgment, judgment_rationale, _ = expert_functions.judge_answer(
                    boxed_answer, sample_info["correct_answer"], sample_info["question"], **judge_kwargs)
            is_correct = judgment == "YES"
        else:
            is_correct = letter_choice == sample["answer_idx"]
            if sample_info.get("branches"):
                is_correct = any(
                    branch.get("final_answer") == sample["answer_idx"]
                    for branch in sample_info["branches"]
                    if branch.get("is_leaf")
                )

        log_info(f"|||||||||||||||||||| Interaction ended for patient #{pid} | Predicted: {letter_choice} | GT: {sample['answer_idx']} | Correct: {is_correct} ||||||||||||||||||||\n\n\n")

        output_dict = {
            "id": pid,
            "interactive_system": {
                "correct": is_correct,
                "letter_choice": letter_choice,
                "judgment": judgment,
                "judgment_rationale": judgment_rationale,
                "questions": questions,
                "answers": answers,
                "num_questions": len(questions),
                "intermediate_choices": temp_choice_list,
                "temp_additional_info": temp_additional_info
            },
            "info": sample_info,
            # TODO: add additional evaluation metrics for analysis, some metrics can be found in src/evaluate.py
            # "eval": {
            #     "confidence_scores": [],
            #     "repeat_question_score": [],
            #     "repeat_answer_score": [],
            #     "relevancy_score": [],
            #     "delta_confidence_score": [],
            #     "specificity_score": []
            # }
        }

        out_dir = os.path.dirname(args.output_filename)
        if out_dir: os.makedirs(out_dir, exist_ok=True)
        with open(args.output_filename, 'a+') as f:
            f.write(json.dumps(output_dict) + '\n')

        if args.convo_log_filename and sample_info.get("branches"):
            write_branching_convo_log(
                args.convo_log_filename, pid, sample_info, is_correct, letter_choice
            )
        elif args.convo_log_filename:
            correct_str = "CORRECT" if is_correct else "WRONG"
            opts = "  ".join(f"{k}: {v}" for k, v in sample_info["options"].items())
            lines = []
            lines.append("=" * 80)
            lines.append(f"Patient #{pid}  |  {correct_str}  |  Predicted: {letter_choice}  |  True: {sample_info['correct_answer_idx']} ({sample_info['correct_answer']})")
            lines.append("=" * 80)
            lines.append(_block("Initial", sample_info["initial_info"]))
            lines.append(_block("Question", sample_info["question"]))
            if args.option_mode == "yes-option":
                lines.append(f"    Options: {opts}\n")
            for i, (q, a, meta) in enumerate(zip(questions, answers, temp_additional_info)):
                lines.append(f"  --- Turn {i+1} " + "-" * 60)
                lines.append(f"    Confidence: {meta.get('confidence')}")
                lines.append(_block("Confidence Rationale", meta.get("confidence_rationale")))
                lines.append(_block("Shadow Answer", meta.get("shadow_answer")))
                if args.option_mode != "yes-option":
                    lines.append(_block("Boxed Answer", meta.get("boxed_answer")))
                if meta.get("question_rationale"):
                    lines.append(_block("Question Rationale", meta.get("question_rationale")))
                lines.extend(_scope_candidate_reward_lines(meta))
                lines.append(_block("Doctor Question", q))
                lines.append(_block("Patient", a))
            # final decision turn (always one more meta entry than questions)
            if len(temp_additional_info) > len(questions):
                meta = temp_additional_info[len(questions)]
                lines.append(f"  --- Turn {len(questions)+1} (Final Decision) " + "-" * 45)
                lines.append(f"    Confidence: {meta.get('confidence')}")
                lines.append(_block("Confidence Rationale", meta.get("confidence_rationale")))
                lines.append(_block("Shadow Answer", meta.get("shadow_answer")))
                lines.extend(_scope_candidate_reward_lines(meta))
                if args.option_mode == "no-option":
                    lines.append(_block("Boxed Answer", meta.get("boxed_answer")))
                    lines.append(f"    → Judgment: {judgment}  |  {judgment_rationale}")
                elif args.option_mode == "option-in-the-end":
                    lines.append(_block("Boxed Answer", meta.get("boxed_answer")))
                    lines.append(f"    → Committed to answer: {letter_choice}")
                else:
                    lines.append(f"    → Committed to answer: {letter_choice}")
            lines.append("")
            convo_dir = os.path.dirname(args.convo_log_filename)
            if convo_dir:
                os.makedirs(convo_dir, exist_ok=True)
            with open(args.convo_log_filename, 'a') as f:
                f.write("\n".join(lines) + "\n")

        if args.doctor_log_filename and sample_info.get("branches"):
            write_branching_doctor_log(
                args.doctor_log_filename, pid, sample_info, is_correct, letter_choice, judgment, judgment_rationale
            )
        elif args.doctor_log_filename:
            final_meta = temp_additional_info[-1] if temp_additional_info else {}
            correct_str = "CORRECT" if is_correct else "WRONG"
            final_answer = letter_choice if args.option_mode != "no-option" else final_meta.get("boxed_answer")
            opts = "  ".join(f"{k}: {v}" for k, v in sample_info["options"].items())
            doc_lines = []
            doc_lines.append("=" * 80)
            doc_lines.append(f"Patient #{pid}  |  {correct_str}  |  Predicted: {final_answer}  |  True: {sample_info['correct_answer_idx']} ({sample_info['correct_answer']})")
            doc_lines.append("=" * 80)
            doc_lines.append(_block("Initial", sample_info["initial_info"]))
            doc_lines.append(_block("Question", sample_info["question"]))
            doc_lines.append(
                _block_lines(
                    "Full context (all segments)",
                    _format_full_context(sample_info.get("context")),
                )
            )
            if args.option_mode == "yes-option":
                doc_lines.append(f"    Options: {opts}\n")
            for i, (q, a) in enumerate(zip(questions, answers)):
                doc_lines.append(f"  --- Turn {i+1} " + "-" * 60)
                turn_meta = temp_additional_info[i] if i < len(temp_additional_info) else {}
                if args.rationale_generation:
                    if turn_meta.get("question_rationale"):
                        doc_lines.append(_block("Question Rationale", turn_meta["question_rationale"]))
                doc_lines.extend(_scope_candidate_reward_lines(turn_meta))
                doc_lines.append(_block("Doctor", q))
                doc_lines.append(_block("Patient", a))
            doc_lines.extend(_scope_candidate_reward_lines(final_meta))
            doc_lines.append(f"  → Final Answer: {final_answer}")
            if judgment is not None:
                doc_lines.append(f"  → Judgment: {judgment}  |  {judgment_rationale}")
            doc_lines.append("")
            doctor_dir = os.path.dirname(args.doctor_log_filename)
            if doctor_dir:
                os.makedirs(doctor_dir, exist_ok=True)
            with open(args.doctor_log_filename, 'a') as f:
                f.write("\n".join(doc_lines) + "\n")

        correct_history.append(is_correct)
        timeout_history.append(len(temp_choice_list) > args.max_questions)
        turn_lengths.append(len(temp_choice_list))
        num_processed += 1
        accuracy = sum(correct_history) / len(correct_history) if len(correct_history) > 0 else None
        timeout_rate = sum(timeout_history) / len(timeout_history) if len(timeout_history) > 0 else None
        avg_turns = sum(turn_lengths) / len(turn_lengths) if len(turn_lengths) > 0 else None

        if results_logger: results_logger.info(f'Processed {num_processed}/{len(patient_data)} patients | Accuracy: {accuracy}')
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processed {num_processed}/{len(patient_data)} patients | Accuracy: {accuracy} | Timeout Rate: {timeout_rate} | Avg. Turns: {avg_turns}")
    accuracy = sum(correct_history) / len(correct_history) if len(correct_history) > 0 else None
    timeout_rate = sum(timeout_history) / len(timeout_history) if len(timeout_history) > 0 else None
    avg_turns = sum(turn_lengths) / len(turn_lengths) if len(turn_lengths) > 0 else None
    print(f"Accuracy: {sum(correct_history)} / {len(correct_history)} = {accuracy}")
    print(f"Timeout Rate: {sum(timeout_history)} / {len(timeout_history)} = {timeout_rate}")
    print(f"Avg. Turns: {avg_turns}")
    

def run_patient_interaction(expert_class, patient_class, sample, pid=None):
    if args.branch_depth > 0:
        return run_branch_interaction(expert_class, patient_class, sample)

    expert_system = expert_class(args, sample["question"], sample["options"])
    if hasattr(expert_system, "set_trace_context"):
        expert_system.set_trace_context(patient_id=pid)
    patient_system = patient_class(args, sample)  # Assuming the patient_system is initialized with the sample which includes necessary context
    temp_choice_list = []
    temp_additional_info = []  # To store optional data like confidence scores

    while len(patient_system.get_questions()) < args.max_questions:
        log_info(f"==================== Turn {len(patient_system.get_questions()) + 1} ====================")
        patient_state = patient_system.get_state()
        response_dict = expert_system.respond(patient_state)
        log_info(f"[Expert System]: {response_dict}")
        
        # Optional return values for analysis, e.g., confidence score, logprobs
        temp_additional_info.append({k: v for k, v in response_dict.items() if k not in ["type", "letter_choice", "question"]})

        if response_dict["type"] == "question":
            # still make the Expert generate a choice based on the current state for intermediate evaluation, log the question as an intermediate choice
            temp_choice_list.append(response_dict["letter_choice"])
            # Patient generates an answer based on the last question asked, and add to memory
            patient_response = patient_system.respond(response_dict["question"])
            log_info(f"[Patient System]: {patient_response}")

        elif response_dict["type"] == "choice":
            expert_decision = response_dict["letter_choice"]
            temp_choice_list.append(expert_decision)
            sample_info = {
                "initial_info": patient_system.initial_info,
                "correct_answer": sample["answer"],
                "correct_answer_idx": sample["answer_idx"],
                "question": sample["question"],
                "options": sample["options"],
                "context": sample["context"],
                "facts": patient_system.facts, # if the FactSelectPatient patient module is used, this will store the atomic facts the patient used to answer questions for reproducibility
            }
            return expert_decision, patient_system.get_questions(), patient_system.get_answers(), temp_choice_list, temp_additional_info, sample_info
        
        else:
            raise ValueError("Invalid response type from expert_system.")
    
    # If max questions are reached and no final decision has been made
    log_info(f"==================== Max Interaction Length ({args.max_questions} turns) Reached --> Force Final Answer ====================")
    patient_state = patient_system.get_state()
    response_dict = expert_system.respond(patient_state)
    log_info(f"[Expert System]: {response_dict}")
    stuck_response = response_dict["letter_choice"]
    # Optional return values for analysis, e.g., confidence score, logprobs
    temp_additional_info.append({k: v for k, v in response_dict.items() if k != "letter_choice"})

    sample_info = {
        "initial_info": patient_system.initial_info,
        "correct_answer": sample["answer"],
        "correct_answer_idx": sample["answer_idx"],
        "question": sample["question"],
        "options": sample["options"],
        "context": sample["context"],
        "facts": patient_system.facts, # if the FactSelectPatient patient module is used, this will store the atomic facts the patient used to answer questions for reproducibility
    }
    
    return stuck_response, patient_system.get_questions(), patient_system.get_answers(), temp_choice_list + [stuck_response], temp_additional_info, sample_info


if __name__ == "__main__":
    args = get_args()
    results_logger = setup_logger('results_logger', args.log_filename)
    history_logger = setup_logger('history_logger', args.history_log_filename)
    detail_logger = setup_logger('detail_logger', args.detail_log_filename)
    message_logger = setup_logger('message_logger', args.message_log_filename)
    main()
