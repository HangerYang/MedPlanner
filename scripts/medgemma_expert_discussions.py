import json
import os
import re

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


STARTER_PATH = "/home/hyang/mediQ/outputs/medgemma27b_starter_candidate_questions_10.jsonl"
OUTPUT_JSONL_PATH = "/home/hyang/mediQ/outputs/medgemma27b_expert_discussions_10x5_n3.jsonl"
OUTPUT_TXT_PATH = "/home/hyang/mediQ/outputs/medgemma27b_expert_discussions_10x5_n3.txt"
MODEL_NAME = "google/medgemma-27b-text-it"
NUM_TURNS = 3
MAX_CANDIDATES_PER_SAMPLE = 5


PROMPT_A = """You are Expert A, a clinical reasoning specialist 
in a consultation with Expert B.

Your goal: given a candidate question, discuss all 
possible answers to it, reason through what each 
answer means clinically, which options it eliminates 
or elevates, and what the next best question to ask 
would be.

You only have access to:
- Initial patient context
- Clinical question and options
- The candidate question being discussed
- Conversation history so far

Each turn you must:
- Analyze possible answers to the candidate question
- State which options are eliminated/elevated and why
- Challenge Expert B if you disagree
- Propose what the next best question should be

Clinical Question: {question}
Options: {options}
Context: {context_0}
Candidate Question: {candidate_question}
Conversation History: {history}

Respond as Expert A only. Be concise and specific.
"""


PROMPT_B = """You are Expert B, a clinical reasoning specialist 
in a consultation with Expert A.

Your goal: given a candidate question, discuss all 
possible answers to it, reason through what each 
answer means clinically, which options it eliminates 
or elevates, and what the next best question to ask 
would be.

You only have access to:
- Initial patient context
- Clinical question and options
- The candidate question being discussed
- Conversation history so far

Each turn you must:
- Critically evaluate Expert A's reasoning
- Add clinical perspectives they may have missed
- Disagree explicitly if you think they are wrong
- Propose what the next best question should be

Clinical Question: {question}
Options: {options}
Context: {context_0}
Candidate Question: {candidate_question}
Conversation History: {history}

Respond as Expert B only. Be concise and specific.
"""


CONCLUSION_PROMPT = """You are a clinical reasoning adjudicator.

You are given a conversation between Expert A and Expert B about one candidate
question. Force a conclusion from the discussion.

Clinical Question: {question}
Options: {options}
Context: {context_0}
Candidate Question: {candidate_question}
Conversation History: {history}

Output exactly these fields:
Forced Conclusion:
1. Best interpretation of the candidate question:
2. Possible answers and what each would mean:
3. Options eliminated or elevated:
4. Best next question to ask:
5. Current best answer option if forced to choose:
"""


def clean_generation(text):
    text = text.strip()
    if "<unused95>" in text:
        text = text.split("<unused95>", 1)[1].strip()
    if "<unused94>" in text:
        text = text.split("<unused94>", 1)[0].strip()
    return text


def clean_candidate_question(text):
    text = re.sub(r"^\s*\d+[\.)]\s*", "", text).strip()
    text = re.sub(r"^\*\*", "", text).strip()
    text = re.sub(r"\*\*.*$", "", text).strip()
    text = text.strip('"').strip()
    return text


def extract_candidate_questions(model_output):
    candidates = []
    for line in model_output.splitlines():
        line = line.strip()
        if not re.match(r"^\d+[\.)]\s+", line):
            continue
        candidate = clean_candidate_question(line)
        if candidate:
            candidates.append(candidate)
    return candidates[:MAX_CANDIDATES_PER_SAMPLE]


def format_options(options):
    return "\n".join(f"{key}. {value}" for key, value in options.items())


def format_history(history):
    if not history:
        return "[]"
    return "\n".join(f"Expert {turn['expert']}: {turn['content']}" for turn in history)


def build_prompt(template, trajectory):
    return template.format(
        question=trajectory["question"],
        options=format_options(trajectory["options"]),
        context_0=trajectory["context_0"],
        candidate_question=trajectory["candidate_question"],
        history=format_history(trajectory["discussion"]),
    )


def render_prompts(tokenizer, prompts):
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]


def load_trajectories():
    trajectories = []
    with open(STARTER_PATH, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            candidates = extract_candidate_questions(record["model_output"])
            for idx, candidate_question in enumerate(candidates, start=1):
                trajectories.append(
                    {
                        "id": record["id"],
                        "candidate_index": idx,
                        "question": record["question"],
                        "options": record["options"],
                        "context_0": record["context_0"],
                        "candidate_question": candidate_question,
                        "discussion": [],
                        "forced_conclusion": "",
                    }
                )
    return trajectories


def main():
    trajectories = load_trajectories()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=4,
        enable_prefix_caching=False,
        max_num_seqs=16,
        max_model_len=4096,
        gpu_memory_utilization=0.35,
    )
    turn_sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=384,
    )
    conclusion_sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
    )

    for _ in range(NUM_TURNS):
        prompts_a = [build_prompt(PROMPT_A, trajectory) for trajectory in trajectories]
        outputs_a = llm.generate(render_prompts(tokenizer, prompts_a), turn_sampling_params)
        for trajectory, output in zip(trajectories, outputs_a):
            trajectory["discussion"].append(
                {"expert": "A", "content": clean_generation(output.outputs[0].text)}
            )

        prompts_b = [build_prompt(PROMPT_B, trajectory) for trajectory in trajectories]
        outputs_b = llm.generate(render_prompts(tokenizer, prompts_b), turn_sampling_params)
        for trajectory, output in zip(trajectories, outputs_b):
            trajectory["discussion"].append(
                {"expert": "B", "content": clean_generation(output.outputs[0].text)}
            )

    conclusion_prompts = [build_prompt(CONCLUSION_PROMPT, trajectory) for trajectory in trajectories]
    conclusion_outputs = llm.generate(render_prompts(tokenizer, conclusion_prompts), conclusion_sampling_params)
    for trajectory, output in zip(trajectories, conclusion_outputs):
        trajectory["forced_conclusion"] = clean_generation(output.outputs[0].text)

    os.makedirs(os.path.dirname(OUTPUT_JSONL_PATH), exist_ok=True)
    with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as f:
        for trajectory in trajectories:
            f.write(json.dumps(trajectory, ensure_ascii=True) + "\n")

    with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("MODEL: " + MODEL_NAME + "\n")
        f.write("STARTER QUESTIONS: " + STARTER_PATH + "\n")
        f.write(f"TRAJECTORIES: {len(trajectories)}\n")
        f.write(f"A/B ROUNDS PER TRAJECTORY: {NUM_TURNS}\n")
        f.write("FINAL FORCED CONCLUSION: yes\n\n")
        for idx, trajectory in enumerate(trajectories, start=1):
            f.write("=" * 80 + "\n")
            f.write(
                f"TRAJECTORY {idx} | sample_id={trajectory['id']} "
                f"| candidate={trajectory['candidate_index']}\n"
            )
            f.write("=" * 80 + "\n")
            f.write("Clinical Question:\n")
            f.write(trajectory["question"] + "\n\n")
            f.write("Options:\n")
            f.write(format_options(trajectory["options"]) + "\n\n")
            f.write("Context:\n")
            f.write(trajectory["context_0"] + "\n\n")
            f.write("Candidate Question:\n")
            f.write(trajectory["candidate_question"] + "\n\n")
            f.write("Discussion:\n")
            for turn in trajectory["discussion"]:
                f.write(f"\nExpert {turn['expert']}:\n{turn['content']}\n")
            f.write("\nForced Conclusion:\n")
            f.write(trajectory["forced_conclusion"] + "\n\n")

    print(OUTPUT_JSONL_PATH)
    print(OUTPUT_TXT_PATH)


if __name__ == "__main__":
    main()
