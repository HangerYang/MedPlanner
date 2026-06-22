#!/usr/bin/env python3
"""One-shot eval using the exact mediQ curr_template + meditron system prompt."""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import prompts

from vllm import LLM, SamplingParams


def load_jsonl(path, limit=0):
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if limit and len(rows) >= limit:
            break
    return rows


def build_prompt(row):
    context = row.get("context") or []
    initial = context[0] if context else ""
    extra_facts = context[1:] if len(context) > 1 else []

    patient_info = initial
    if extra_facts:
        patient_info += "\n\nKnown useful patient facts:\n" + "\n".join(extra_facts)

    inquiry = row["question"]
    options = row["options"]
    options_text = f'A: {options["A"]}, B: {options["B"]}, C: {options["C"]}, D: {options["D"]}'
    conv_log = "None"

    user_content = prompts.expert_system["curr_template"].format(
        patient_info, conv_log, inquiry, options_text, prompts.expert_system["answer"]
    )
    return [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user", "content": user_content},
    ]


def apply_chat_template(llm, messages):
    tokenizer = llm.get_tokenizer()
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def parse_choice(text, options):
    text = re.sub(r"<think>.*?</think>", " ", text or "", flags=re.DOTALL | re.IGNORECASE).strip()
    for pat in [r"\bLETTER\s+CHOICE\s*[:\-]?\s*([A-D])\b",
                r"\bFINAL\s+ANSWER\b[^A-D]*([A-D])\b",
                r"\bANSWER\s*[:\-]?\s*([A-D])\b",
                r"^\s*([A-D])\s*$"]:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m and m.group(1).upper() in options:
            return m.group(1).upper()
    m = re.search(r"\b([A-D])\b", text)
    if m and m.group(1) in options:
        return m.group(1)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    rows = load_jsonl(args.data, args.max_examples)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)

    prompts_list = [apply_chat_template(llm, build_prompt(row)) for row in rows]
    outputs = llm.generate(prompts_list, sampling)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    with open(args.output, "w") as f:
        for row, out in zip(rows, outputs):
            text = out.outputs[0].text.strip()
            pred = parse_choice(text, row["options"])
            is_correct = pred == row["answer_idx"] if pred else False
            correct += int(is_correct)
            f.write(json.dumps({
                "id": row["id"], "question": row["question"],
                "true_letter": row["answer_idx"], "parsed_letter": pred,
                "correct": is_correct, "full_response": text,
            }) + "\n")

    print(f"Rows: {len(rows)}")
    print(f"Correct: {correct}/{len(rows)} = {correct/len(rows):.4f}")


if __name__ == "__main__":
    main()
