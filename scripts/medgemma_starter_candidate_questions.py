import json
import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


DATA_PATH = "/home/hyang/mediQ/data/med_data/all_dev_convo.jsonl"
OUTPUT_TXT_PATH = "/home/hyang/mediQ/outputs/medgemma27b_starter_candidate_questions_10.txt"
OUTPUT_JSONL_PATH = "/home/hyang/mediQ/outputs/medgemma27b_starter_candidate_questions_10.jsonl"
MODEL_NAME = "google/medgemma-27b-text-it"
NUM_EXAMPLES = 10
NUM_QUESTIONS = 5


PROMPT_TEMPLATE = """You are an expert clinical reasoning assistant.

You are given a clinical question with answer options 
and an initial patient context.

Your task: generate {num_questions} candidate questions that a 
clinician would want to ask next to determine the 
correct answer.

Rules:
- Questions should be clinically natural
- Each question should target different aspects 
  of the diagnosis/treatment
- Do NOT assume you know any information beyond 
  what is given
- Questions should be specific and answerable in 
  a real clinical encounter

Clinical Question: {question}
Answer Options: {options}
Current Context: {context_0}

Output: a numbered list of {num_questions} candidate questions, 
each with one sentence justifying why it is 
clinically relevant.
"""


def clean_generation(text):
    text = text.strip()
    if "<unused95>" in text:
        text = text.split("<unused95>", 1)[1].strip()
    if "<unused94>" in text:
        text = text.split("<unused94>", 1)[0].strip()
    return text


def load_samples():
    samples = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if len(samples) >= NUM_EXAMPLES:
                break
            samples.append(json.loads(line))
    return samples


def build_prompt(sample):
    options = "\n".join(f"{key}. {value}" for key, value in sample["options"].items())
    return PROMPT_TEMPLATE.format(
        num_questions=NUM_QUESTIONS,
        question=sample["question"],
        options=options,
        context_0=sample["context"][0],
    )


def main():
    samples = load_samples()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompts = [build_prompt(sample) for sample in samples]
    rendered_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=4,
        enable_prefix_caching=False,
        max_num_seqs=NUM_EXAMPLES,
        max_model_len=4096,
        gpu_memory_utilization=0.35,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
    )
    generations = llm.generate(rendered_prompts, sampling_params)
    records = []
    for sample, prompt, generation in zip(samples, prompts, generations):
        records.append(
            {
                "id": sample["id"],
                "question": sample["question"],
                "options": sample["options"],
                "context_0": sample["context"][0],
                "prompt": prompt,
                "model_output": clean_generation(generation.outputs[0].text),
            }
        )

    os.makedirs(os.path.dirname(OUTPUT_TXT_PATH), exist_ok=True)
    with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("MODEL: " + MODEL_NAME + "\n")
        f.write("DATA: " + DATA_PATH + "\n")
        f.write(f"EXAMPLES: first {NUM_EXAMPLES} JSONL records\n")
        f.write("CONTEXT USED: context[0]\n")
        f.write(f"CANDIDATE QUESTIONS PER EXAMPLE: {NUM_QUESTIONS}\n\n")
        for index, record in enumerate(records, start=1):
            f.write("=" * 80 + "\n")
            f.write(f"EXAMPLE {index} | id={record['id']}\n")
            f.write("=" * 80 + "\n")
            f.write("Clinical Question:\n")
            f.write(record["question"] + "\n\n")
            f.write("Answer Options:\n")
            for key, value in record["options"].items():
                f.write(f"{key}. {value}\n")
            f.write("\nCurrent Context:\n")
            f.write(record["context_0"] + "\n\n")
            f.write("Starter Candidate Questions:\n")
            f.write(record["model_output"] + "\n\n")

    print(OUTPUT_TXT_PATH)
    print(OUTPUT_JSONL_PATH)


if __name__ == "__main__":
    main()
