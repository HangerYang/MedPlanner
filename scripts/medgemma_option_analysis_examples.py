import argparse
import json
import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


DEFAULT_DATA_PATH = "/home/hyang/mediQ/data/med_data/all_dev_convo.jsonl"
DEFAULT_OUTPUT_PATH = "/home/hyang/mediQ/outputs/medgemma27b_examples2-5_option_analysis.txt"
DEFAULT_JSONL_PATH = "/home/hyang/mediQ/outputs/medgemma27b_examples2-5_option_analysis.jsonl"
MODEL_NAME = "google/medgemma-27b-text-it"


PROMPT_TEMPLATE = """You are an expert clinical reasoning assistant.

You will be given:
- A clinical question with multiple answer options
- The initial clinical context available so far

Your task is to analyze each answer option carefully. For each option:

1. **Status**: Is this option eliminated or still in contention?
   - "Eliminated": The current context makes this option impossible or highly implausible.
     Give a clear reason grounded in the context.
   - "In contention": This option has not been ruled out yet.

2. **If in contention -- rank its plausibility** relative to other in-contention options:
   - Assign a rank (1 = most plausible, higher = less plausible).
   - Give a brief justification: what in the current context supports or weakens this option?

3. **Missing facts**: What specific clinical facts would change this option's status or ranking?
   - Be concrete (e.g., "presence of night sweats", "TSH level", "duration of symptoms").
   - For each missing fact, state: if X were true, this option would be [eliminated / confirmed / elevated in rank / lowered in rank].

---

Clinical Question:
{question}

Answer Options:
{options}

Current Context:
{context_so_far}

---

Analyze each option systematically, then provide a final ranked list of in-contention
options from most to least plausible.
"""


def load_examples(data_path, start_index, num_examples):
    examples = []
    end_index = start_index + num_examples
    with open(data_path, "r", encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            if line_index < start_index:
                continue
            if line_index >= end_index:
                break
            examples.append((line_index, json.loads(line)))
    return examples


def build_prompt(sample):
    context_so_far = sample["context"][0]
    options = "\n".join(f"{key}. {value}" for key, value in sample["options"].items())
    return PROMPT_TEMPLATE.format(
        question=sample["question"],
        options=options,
        context_so_far=context_so_far,
    )


def clean_output(text):
    text = text.strip()
    if "<unused95>" in text:
        text = text.split("<unused95>", 1)[1].strip()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--jsonl_path", default=DEFAULT_JSONL_PATH)
    parser.add_argument("--start_index", type=int, default=1)
    parser.add_argument("--num_examples", type=int, default=4)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.35)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_num_seqs", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=2048)
    args = parser.parse_args()

    examples = load_examples(args.data_path, args.start_index, args.num_examples)
    if not examples:
        raise ValueError("No examples found for the requested range.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompts = []
    rendered_prompts = []
    for _, sample in examples:
        prompt = build_prompt(sample)
        prompts.append(prompt)
        rendered_prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=False,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
    )
    generations = llm.generate(rendered_prompts, sampling_params)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.jsonl_path), exist_ok=True)

    records = []
    with open(args.output_path, "w", encoding="utf-8") as txt_f:
        txt_f.write("MODEL: " + MODEL_NAME + "\n")
        txt_f.write("DATA: " + args.data_path + "\n")
        txt_f.write(
            "EXAMPLES: JSONL records "
            + str(args.start_index)
            + "-"
            + str(args.start_index + len(examples) - 1)
            + "\n"
        )
        txt_f.write("CONTEXT USED: context[0]\n")

        for (line_index, sample), prompt, generation in zip(examples, prompts, generations):
            output = clean_output(generation.outputs[0].text)
            record = {
                "line_index": line_index,
                "id": sample.get("id"),
                "answer": sample.get("answer"),
                "answer_idx": sample.get("answer_idx"),
                "prompt": prompt,
                "output": output,
            }
            records.append(record)

            txt_f.write("\n\n" + "=" * 80 + "\n")
            txt_f.write("EXAMPLE: JSONL record " + str(line_index) + ", id=" + str(sample.get("id")) + "\n")
            txt_f.write("ANSWER: " + str(sample.get("answer_idx")) + ". " + str(sample.get("answer")) + "\n")
            txt_f.write("\n===== PROMPT =====\n")
            txt_f.write(prompt)
            txt_f.write("\n\n===== MODEL OUTPUT =====\n")
            txt_f.write(output)
            txt_f.write("\n")

    with open(args.jsonl_path, "w", encoding="utf-8") as jsonl_f:
        for record in records:
            jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(args.output_path)
    print(args.jsonl_path)


if __name__ == "__main__":
    main()
