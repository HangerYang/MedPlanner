"""
Interactive human-in-the-loop session with an LLM acting as the expert.
The human supplies the scenario and answers any follow-up questions the LLM asks.
All sessions are appended to a JSONL file.

Run from the repo root:
    python src/human_interactive.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --use_vllm \
        --output_file src/results/human_sessions.jsonl

Or without vLLM (uses HuggingFace directly):
    python src/human_interactive.py --model meta-llama/Llama-3.1-8B-Instruct
"""

import sys
import os
import json
import argparse
from datetime import datetime

# Ensure src/ imports resolve whether the script is run from root or src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prompts
import expert_basics

SEP = "=" * 60
THIN = "-" * 60


def get_args():
    p = argparse.ArgumentParser(description="Human-in-the-loop MediQ interactive session.")
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                   help="Model name or path (HuggingFace or vLLM).")
    p.add_argument("--use_vllm", action="store_true",
                   help="Use a running vLLM server to generate responses.")
    p.add_argument("--use_api", type=str, default=None, choices=["openai"],
                   help="Use an OpenAI-compatible API instead of a local model.")
    p.add_argument("--max_questions", type=int, default=10,
                   help="Maximum follow-up questions before forcing a final answer.")
    p.add_argument("--output_file", type=str, default="src/results/human_sessions.jsonl",
                   help="JSONL file where sessions are saved.")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--top_logprobs", type=int, default=0)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=None,
        help="vLLM GPU memory fraction (0–1). Lower when sharing the GPU. Env MEDIQ_VLLM_GPU_MEMORY_UTILIZATION if unset.",
    )
    p.add_argument(
        "--vllm_max_model_len",
        type=int,
        default=8192,
        help="vLLM max_model_len (KV). Default 8192 avoids 128k-context OOM on one GPU.",
    )
    p.add_argument(
        "--vllm_max_num_seqs",
        type=int,
        default=None,
        help="vLLM max_num_seqs; default uses --batch_size.",
    )
    p.add_argument(
        "--vllm_enforce_eager",
        action="store_true",
        help="vLLM enforce_eager (lower peak VRAM, slower).",
    )
    p.add_argument("--api_account", type=str, default="mediQ")
    return p.parse_args()


def build_prompt(initial_info, history, question, options):
    conv_log = "\n".join(
        f"Doctor Question: {qa['question']}\nPatient Response: {qa['answer']}"
        for qa in history
    ) or "None"
    options_text = ", ".join(f"{k}: {v}" for k, v in options.items())
    return prompts.expert_system["curr_template"].format(
        initial_info, conv_log, question, options_text, prompts.expert_system["implicit"]
    )


def build_forced_prompt(initial_info, history, question, options):
    conv_log = "\n".join(
        f"Doctor Question: {qa['question']}\nPatient Response: {qa['answer']}"
        for qa in history
    ) or "None"
    options_text = ", ".join(f"{k}: {v}" for k, v in options.items())
    return prompts.expert_system["curr_template"].format(
        initial_info, conv_log, question, options_text, prompts.expert_system["answer"]
    )


def collect_scenario():
    print(SEP)
    print("  New Session — enter the clinical scenario")
    print(SEP)
    initial_info = input("Patient information: ").strip()
    question = input("Clinical question:   ").strip()

    print("Answer options (press Enter to skip an option):")
    options = {}
    for letter in ["A", "B", "C", "D"]:
        val = input(f"  {letter}: ").strip()
        if val:
            options[letter] = val

    if len(options) < 2:
        print("(Fewer than 2 options entered — using generic A/B/C/D placeholders.)")
        options = {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}

    return initial_info, question, options


def run_session(args, llm_kwargs):
    initial_info, question, options = collect_scenario()

    history = []
    final_choice = None
    turn = 0

    while turn < args.max_questions:
        print(f"\n{THIN}")
        print(f"  Turn {turn + 1} — thinking...")
        print(THIN)

        messages = [
            {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
            {"role": "user",   "content": build_prompt(initial_info, history, question, options)},
        ]

        _, atomic_question, letter_choice, conf_score, _, _ = \
            expert_basics.expert_response_choice_or_question(messages, options, **llm_kwargs)

        if letter_choice is not None and atomic_question is None:
            final_choice = letter_choice
            print(f"\n  Final answer: [{final_choice}] {options.get(final_choice, '')}")
            print(f"  Confidence:   {conf_score:.0%}")
            break

        if atomic_question is not None:
            print(f"\n  Doctor's question: {atomic_question}")
            answer = input("  Your answer:       ").strip()
            if not answer:
                answer = "(no answer provided)"
            history.append({"question": atomic_question, "answer": answer})
            turn += 1
        else:
            print("  LLM returned an unparseable response. Ending session.")
            break

    # Force a final answer if max turns reached without a choice
    if final_choice is None:
        print(f"\n{SEP}")
        print(f"  Max questions ({args.max_questions}) reached — forcing final answer.")
        messages = [
            {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
            {"role": "user",   "content": build_forced_prompt(initial_info, history, question, options)},
        ]
        _, final_choice, _ = expert_basics.expert_response_choice(messages, options, **llm_kwargs)
        print(f"  Final answer: [{final_choice}] {options.get(final_choice, '')}")

    session = {
        "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model": args.model,
        "initial_info": initial_info,
        "question": question,
        "options": options,
        "history": history,
        "final_choice": final_choice,
        "num_turns": len(history),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "a") as f:
        f.write(json.dumps(session) + "\n")

    print(f"\n  Session saved → {args.output_file}")
    return session


def main():
    args = get_args()

    llm_kwargs = {
        "model_name":          args.model,
        "use_vllm":            args.use_vllm,
        "use_api":             args.use_api,
        "temperature":         args.temperature,
        "max_tokens":          args.max_tokens,
        "top_p":               args.top_p,
        "top_logprobs":        args.top_logprobs,
        "api_account":         args.api_account,
        "tensor_parallel_size": args.tensor_parallel_size,
        "batch_size":          args.batch_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "vllm_max_model_len":     args.vllm_max_model_len,
        "vllm_max_num_seqs":      args.vllm_max_num_seqs,
        "vllm_enforce_eager":     args.vllm_enforce_eager,
    }

    print(SEP)
    print("  MediQ Human-Interactive Mode")
    print(f"  Model:       {args.model}")
    print(f"  Backend:     {'vLLM' if args.use_vllm else ('OpenAI API' if args.use_api else 'HuggingFace')}")
    print(f"  Max turns:   {args.max_questions}")
    print(f"  Output file: {args.output_file}")
    print(SEP)

    try:
        while True:
            run_session(args, llm_kwargs)
            print()
            again = input("Run another session? (y/n): ").strip().lower()
            if again != "y":
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
