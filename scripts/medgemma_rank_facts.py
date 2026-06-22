import json
import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


DATA_PATH = "/home/hyang/mediQ/data/med_data/all_dev_convo.jsonl"
OUTPUT_TXT_PATH = "/home/hyang/mediQ/outputs/medgemma27b_ranked_facts_10.txt"
OUTPUT_JSONL_PATH = "/home/hyang/mediQ/outputs/medgemma27b_ranked_facts_10.jsonl"
MODEL_NAME = "google/medgemma-27b-text-it"
NUM_EXAMPLES = 10


PROMPT_TEMPLATE = """You are an expert clinical reasoning assistant with 
access to the full patient case.

You are given:
- A clinical question with answer options
- The complete patient context (all facts)
- The correct answer

Your task: rank ALL available facts by their 
importance in determining the correct answer.

For each fact:
  a) Importance rank (1 = most important)
  b) Which options does it eliminate? Why?
  c) Which options does it elevate? Why?
  d) Is this fact sufficient alone to determine 
     the answer, or does it need other facts?

Clinical Question: {question}
Answer Options: {options}
Correct Answer: {correct_answer}
Full Context: {all_contexts}

Available Facts To Rank:
{facts}

Output as structured JSON:
{{
  "ranked_facts": [
    {{
      "fact": "...",
      "importance_rank": 1,
      "eliminates": [...],
      "elevates": [...],
      "sufficient_alone": true/false,
      "reasoning": "..."
    }},
    ...
  ]
}}
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


def get_facts(sample):
    facts = sample.get("atomic_facts") or sample.get("facts_old") or sample["context"]
    return "\n".join(str(fact) for fact in facts)


def build_prompt(sample):
    options = "\n".join(f"{key}. {value}" for key, value in sample["options"].items())
    all_contexts = "\n".join(f"{idx + 1}. {context}" for idx, context in enumerate(sample["context"]))
    return PROMPT_TEMPLATE.format(
        question=sample["question"],
        options=options,
        correct_answer=f"{sample['answer_idx']}. {sample['answer']}",
        all_contexts=all_contexts,
        facts=get_facts(sample),
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
        max_tokens=2048,
    )
    generations = llm.generate(rendered_prompts, sampling_params)

    records = []
    for sample, prompt, generation in zip(samples, prompts, generations):
        records.append(
            {
                "id": sample["id"],
                "question": sample["question"],
                "options": sample["options"],
                "correct_answer": sample["answer"],
                "correct_answer_idx": sample["answer_idx"],
                "full_context": sample["context"],
                "facts_to_rank": sample.get("atomic_facts") or sample.get("facts_old") or sample["context"],
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
        f.write("FULL CONTEXT USED: all entries in sample['context']\n")
        f.write("FACTS RANKED: sample['atomic_facts'] when available, else facts_old/context\n\n")
        for index, record in enumerate(records, start=1):
            f.write("=" * 80 + "\n")
            f.write(f"EXAMPLE {index} | id={record['id']}\n")
            f.write("=" * 80 + "\n")
            f.write("Clinical Question:\n")
            f.write(record["question"] + "\n\n")
            f.write("Answer Options:\n")
            for key, value in record["options"].items():
                f.write(f"{key}. {value}\n")
            f.write("\nCorrect Answer:\n")
            f.write(f"{record['correct_answer_idx']}. {record['correct_answer']}\n\n")
            f.write("Full Context:\n")
            for idx, context in enumerate(record["full_context"], start=1):
                f.write(f"{idx}. {context}\n")
            f.write("\nRanked Facts JSON:\n")
            f.write(record["model_output"] + "\n\n")

    print(OUTPUT_TXT_PATH)
    print(OUTPUT_JSONL_PATH)


if __name__ == "__main__":
    main()
