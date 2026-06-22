import json
import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


DATA_PATH = "/home/hyang/mediQ/data/med_data/all_dev_convo.jsonl"
OUTPUT_PATH = "/home/hyang/mediQ/outputs/medgemma27b_example1_option_analysis.txt"
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

2. **If in contention — rank its plausibility** relative to other in-contention options:
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


def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        sample = json.loads(f.readline())

    context_so_far = sample["context"][0]
    options = "\n".join(f"{key}. {value}" for key, value in sample["options"].items())
    prompt = PROMPT_TEMPLATE.format(
        question=sample["question"],
        options=options,
        context_so_far=context_so_far,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    messages = [{"role": "user", "content": prompt}]
    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=4,
        enable_prefix_caching=False,
        max_num_seqs=1,
        max_model_len=4096,
        gpu_memory_utilization=0.35,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
    )
    result = llm.generate([rendered_prompt], sampling_params)[0].outputs[0].text.strip()
    if "<unused95>" in result:
        result = result.split("<unused95>", 1)[1].strip()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("MODEL: " + MODEL_NAME + "\n")
        f.write("DATA: " + DATA_PATH + "\n")
        f.write("EXAMPLE: first JSONL record, id=" + str(sample["id"]) + "\n")
        f.write("CONTEXT USED: context[0]\n\n")
        f.write("===== PROMPT =====\n")
        f.write(prompt)
        f.write("\n\n===== MODEL OUTPUT =====\n")
        f.write(result)
        f.write("\n")

    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
