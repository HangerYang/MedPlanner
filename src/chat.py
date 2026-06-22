"""
Simple interactive chat with a local LLM (vLLM or HuggingFace).
All conversations are saved to a JSONL file.

Usage:
    python src/chat.py --model meta-llama/Llama-3.1-8B-Instruct --use_vllm
    python src/chat.py --model meta-llama/Llama-3.1-8B-Instruct
"""

import sys
import os
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helper import get_response


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                   choices=["meta-llama/Llama-3.1-8B-Instruct", "google/medgemma-27b-text-it"])
    p.add_argument("--use_vllm", action="store_true")
    p.add_argument("--use_api", type=str, default=None, choices=["openai"])
    p.add_argument("--output_file", type=str, default=None,
                   help="Output JSONL file. Defaults to src/results/<model_short_name>_chat_sessions.jsonl.")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_tokens", type=int, default=512)
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
        help="vLLM max_num_seqs; default uses --batch_size. Env MEDIQ_VLLM_MAX_NUM_SEQS caps.",
    )
    p.add_argument(
        "--vllm_enforce_eager",
        action="store_true",
        help="vLLM enforce_eager (lower peak VRAM, slower).",
    )
    p.add_argument("--api_account", type=str, default="mediQ")
    p.add_argument("--system", type=str, default="You are a helpful assistant. Answer the user's questions directly and concisely.",
                   help="System prompt.")
    p.add_argument("--append", type=str,
                   default="Given the information so far, you can either provide the final answer in \\boxed{} if you are confident, or ask ONE SPECIFIC ATOMIC QUESTION to the patient if you need more information. The question should be bite-sized, NOT ask for too much at once, and NOT repeat what has already been asked. In this case, respond with the atomic question and NOTHING ELSE.",
                   help="Text appended to every user message.")
    args = p.parse_args()
    if args.output_file is None:
        short = args.model.split("/")[-1].lower().replace("-", "_")
        args.output_file = f"src/results/{short}_chat_sessions.jsonl"
    return args


def main():
    args = get_args()
    kwargs = dict(
        model_name=args.model,
        use_vllm=args.use_vllm,
        use_api=args.use_api,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_max_num_seqs=args.vllm_max_num_seqs,
        vllm_enforce_eager=args.vllm_enforce_eager,
        api_account=args.api_account,
    )

    print(f"Model: {args.model} | type 'quit' to exit\n")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    messages = [{"role": "system", "content": args.system}]
    history = []  # clean list of {user, response} for saving

    def save():
        # Load existing sessions, replace this session_id if present, else append.
        sessions = []
        if os.path.exists(args.output_file):
            with open(args.output_file) as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get("session_id") != session_id:
                        sessions.append(entry)
        sessions.append({
            "session_id": session_id,
            "model": args.model,
            "conversation": history,
        })
        with open(args.output_file, "w") as f:
            for s in sessions:
                f.write(json.dumps(s) + "\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        content = f"{user_input}\n\n{args.append}" if args.append else user_input
        messages.append({"role": "user", "content": content})
        response, _, _ = get_response(messages, **kwargs)
        messages.append({"role": "assistant", "content": response})

        print(f"\nModel: {response}\n")

        history.append({"user": user_input, "response": response})
        save()


if __name__ == "__main__":
    main()
