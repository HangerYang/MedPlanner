import argparse
import json
import os
import re
from datetime import datetime

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


MODEL_NAME = "google/medgemma-27b-text-it"


CANDIDATE_PROMPT = """You are an expert clinical reasoning assistant.

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


FACT_RANKING_PROMPT = """You are an expert clinical reasoning assistant with 
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
- If the discussion has reached a clear conclusion, explicitly write:
  CONCLUSION_REACHED: yes
- Otherwise write:
  CONCLUSION_REACHED: no

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
- If the discussion has reached a clear conclusion, explicitly write:
  CONCLUSION_REACHED: yes
- Otherwise write:
  CONCLUSION_REACHED: no

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home/hyang/mediQ/data/med_data/all_train_convo.jsonl")
    parser.add_argument("--output_dir", default="/home/hyang/mediQ/outputs/scope_train_1k_medgemma27b")
    parser.add_argument("--num_examples", type=int, default=1000)
    parser.add_argument("--num_candidate_questions", type=int, default=5)
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.35)
    parser.add_argument("--max_num_seqs", type=int, default=16)
    parser.add_argument("--candidate_max_tokens", type=int, default=768)
    parser.add_argument("--fact_max_tokens", type=int, default=2048)
    parser.add_argument("--turn_max_tokens", type=int, default=384)
    parser.add_argument("--conclusion_max_tokens", type=int, default=512)
    return parser.parse_args()


def clean_generation(text):
    text = text.strip()
    if "<unused95>" in text:
        text = text.split("<unused95>", 1)[1].strip()
    if "<unused94>" in text:
        text = text.split("<unused94>", 1)[0].strip()
    return text.strip()


def strip_code_fence(text):
    text = clean_generation(text)
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


def extract_json_object(text):
    text = strip_code_fence(text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def clean_candidate_question(line):
    line = re.sub(r"^\s*\d+[\.)]\s*", "", line).strip()
    line = re.sub(r"^\*\*", "", line).strip()
    if "**" in line:
        line = line.split("**", 1)[0].strip()
    if " - " in line:
        line = line.split(" - ", 1)[0].strip()
    if " This " in line:
        line = line.split(" This ", 1)[0].strip()
    return line.strip().strip('"').strip()


def extract_candidate_questions(model_output, limit):
    candidates = []
    for line in clean_generation(model_output).splitlines():
        if not re.match(r"^\s*\d+[\.)]\s+", line):
            continue
        question = clean_candidate_question(line)
        if question and question not in candidates:
            candidates.append(question)
        if len(candidates) >= limit:
            break
    return candidates


def conclusion_reached(text):
    return bool(re.search(r"CONCLUSION_REACHED\s*:\s*yes", text, flags=re.IGNORECASE))


def normalize_context(context):
    if isinstance(context, list):
        return [str(item) for item in context]
    return [str(context)]


def format_options(options):
    return "\n".join(f"{key}. {value}" for key, value in options.items())


def format_history(history):
    if not history:
        return "[]"
    return "\n".join(f"Expert {turn['expert']}: {turn['content']}" for turn in history)


def get_facts(sample):
    return sample.get("atomic_facts") or sample.get("facts_old") or normalize_context(sample["context"])


def load_samples(path, limit):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(samples) >= limit:
                break
            sample = json.loads(line)
            contexts = normalize_context(sample["context"])
            sample["_context_list"] = contexts
            sample["_context_0"] = contexts[0]
            samples.append(sample)
    return samples


def render_prompts(tokenizer, prompts):
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]


def generate_batch(llm, tokenizer, prompts, max_tokens):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outputs = llm.generate(render_prompts(tokenizer, prompts), sampling_params)
    return [clean_generation(output.outputs[0].text) for output in outputs]


def build_candidate_prompt(sample, num_questions):
    return CANDIDATE_PROMPT.format(
        num_questions=num_questions,
        question=sample["question"],
        options=format_options(sample["options"]),
        context_0=sample["_context_0"],
    )


def build_fact_prompt(sample):
    return FACT_RANKING_PROMPT.format(
        question=sample["question"],
        options=format_options(sample["options"]),
        correct_answer=f"{sample['answer_idx']}. {sample['answer']}",
        all_contexts="\n".join(f"{idx + 1}. {ctx}" for idx, ctx in enumerate(sample["_context_list"])),
        facts="\n".join(str(fact) for fact in get_facts(sample)),
    )


def build_discussion_prompt(template, trajectory):
    return template.format(
        question=trajectory["question"],
        options=format_options(trajectory["options"]),
        context_0=trajectory["context_0"],
        candidate_question=trajectory["candidate_question"],
        history=format_history(trajectory["discussion"]),
    )


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_candidate_txt(path, records, args):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"MODEL: {MODEL_NAME}\nDATA: {args.data_path}\nEXAMPLES: {len(records)}\n\n")
        for index, record in enumerate(records, start=1):
            f.write("=" * 80 + f"\nEXAMPLE {index} | id={record['id']}\n" + "=" * 80 + "\n")
            f.write(record["question"] + "\n\n")
            f.write(format_options(record["options"]) + "\n\n")
            f.write("Context:\n" + record["context_0"] + "\n\n")
            f.write("Candidate Questions:\n")
            for q_idx, question in enumerate(record["candidate_questions"], start=1):
                f.write(f"{q_idx}. {question}\n")
            f.write("\nRaw Output:\n" + record["raw_model_output"] + "\n\n")


def write_fact_txt(path, records, args):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"MODEL: {MODEL_NAME}\nDATA: {args.data_path}\nEXAMPLES: {len(records)}\n\n")
        for index, record in enumerate(records, start=1):
            f.write("=" * 80 + f"\nEXAMPLE {index} | id={record['id']}\n" + "=" * 80 + "\n")
            f.write(record["question"] + "\n\n")
            f.write("Correct Answer: " + record["correct_answer_idx"] + ". " + record["correct_answer"] + "\n\n")
            f.write("Ranked Facts:\n")
            f.write(json.dumps(record["ranked_facts_json"], indent=2, ensure_ascii=True) if record["ranked_facts_json"] else record["raw_model_output"])
            f.write("\n\n")


def write_discussion_txt(path, records, args):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"MODEL: {MODEL_NAME}\nTRAJECTORIES: {len(records)}\nMAX_TURNS: {args.max_turns}\n")
        f.write("EARLY STOP: enabled when an expert writes CONCLUSION_REACHED: yes\n\n")
        for index, record in enumerate(records, start=1):
            f.write("=" * 80 + f"\nTRAJECTORY {index} | sample_id={record['id']} | candidate={record['candidate_index']}\n")
            f.write("=" * 80 + "\n")
            f.write("Clinical Question:\n" + record["question"] + "\n\n")
            f.write("Options:\n" + format_options(record["options"]) + "\n\n")
            f.write("Context:\n" + record["context_0"] + "\n\n")
            f.write("Candidate Question:\n" + record["candidate_question"] + "\n\n")
            f.write(f"Stopped Early: {record['stopped_early']} | Rounds Completed: {record['rounds_completed']}\n")
            for turn in record["discussion"]:
                f.write(f"\nExpert {turn['expert']}:\n{turn['content']}\n")
            f.write("\nForced Conclusion:\n" + record["forced_conclusion"] + "\n\n")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"started_at": datetime.now().isoformat(), "args": vars(args)}, f, indent=2)

    samples = load_samples(args.data_path, args.num_examples)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=False,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    candidate_prompts = [build_candidate_prompt(sample, args.num_candidate_questions) for sample in samples]
    candidate_outputs = generate_batch(llm, tokenizer, candidate_prompts, args.candidate_max_tokens)
    candidate_records = []
    trajectories = []
    for sample, prompt, output in zip(samples, candidate_prompts, candidate_outputs):
        questions = extract_candidate_questions(output, args.num_candidate_questions)
        record = {
            "id": sample["id"],
            "question": sample["question"],
            "options": sample["options"],
            "context_0": sample["_context_0"],
            "prompt": prompt,
            "raw_model_output": output,
            "candidate_questions": questions,
        }
        candidate_records.append(record)
        for idx, candidate_question in enumerate(questions, start=1):
            trajectories.append(
                {
                    "id": sample["id"],
                    "candidate_index": idx,
                    "question": sample["question"],
                    "options": sample["options"],
                    "context_0": sample["_context_0"],
                    "candidate_question": candidate_question,
                    "discussion": [],
                    "forced_conclusion": "",
                    "stopped_early": False,
                    "rounds_completed": 0,
                }
            )

    candidate_jsonl = os.path.join(args.output_dir, "candidate_questions.jsonl")
    candidate_txt = os.path.join(args.output_dir, "candidate_questions.txt")
    write_jsonl(candidate_jsonl, candidate_records)
    write_candidate_txt(candidate_txt, candidate_records, args)
    print(candidate_jsonl, flush=True)

    fact_prompts = [build_fact_prompt(sample) for sample in samples]
    fact_outputs = generate_batch(llm, tokenizer, fact_prompts, args.fact_max_tokens)
    fact_records = []
    for sample, prompt, output in zip(samples, fact_prompts, fact_outputs):
        fact_records.append(
            {
                "id": sample["id"],
                "question": sample["question"],
                "options": sample["options"],
                "correct_answer": sample["answer"],
                "correct_answer_idx": sample["answer_idx"],
                "full_context": sample["_context_list"],
                "facts_to_rank": get_facts(sample),
                "prompt": prompt,
                "raw_model_output": output,
                "ranked_facts_json": extract_json_object(output),
            }
        )

    facts_jsonl = os.path.join(args.output_dir, "ranked_facts.jsonl")
    facts_txt = os.path.join(args.output_dir, "ranked_facts.txt")
    write_jsonl(facts_jsonl, fact_records)
    write_fact_txt(facts_txt, fact_records, args)
    print(facts_jsonl, flush=True)

    active = trajectories
    for turn_idx in range(args.max_turns):
        if not active:
            break

        prompts_a = [build_discussion_prompt(PROMPT_A, trajectory) for trajectory in active]
        outputs_a = generate_batch(llm, tokenizer, prompts_a, args.turn_max_tokens)
        for trajectory, output in zip(active, outputs_a):
            trajectory["discussion"].append({"expert": "A", "content": output})

        prompts_b = [build_discussion_prompt(PROMPT_B, trajectory) for trajectory in active]
        outputs_b = generate_batch(llm, tokenizer, prompts_b, args.turn_max_tokens)

        next_active = []
        for trajectory, output in zip(active, outputs_b):
            trajectory["discussion"].append({"expert": "B", "content": output})
            trajectory["rounds_completed"] = turn_idx + 1
            last_a = trajectory["discussion"][-2]["content"]
            last_b = trajectory["discussion"][-1]["content"]
            if conclusion_reached(last_a) or conclusion_reached(last_b):
                trajectory["stopped_early"] = True
            else:
                next_active.append(trajectory)
        active = next_active
        print(f"discussion_round={turn_idx + 1} active_remaining={len(active)}", flush=True)

    conclusion_prompts = [build_discussion_prompt(CONCLUSION_PROMPT, trajectory) for trajectory in trajectories]
    conclusion_outputs = generate_batch(llm, tokenizer, conclusion_prompts, args.conclusion_max_tokens)
    for trajectory, output in zip(trajectories, conclusion_outputs):
        trajectory["forced_conclusion"] = output

    discussion_jsonl = os.path.join(args.output_dir, "expert_discussions.jsonl")
    discussion_txt = os.path.join(args.output_dir, "expert_discussions.txt")
    write_jsonl(discussion_jsonl, trajectories)
    write_discussion_txt(discussion_txt, trajectories, args)
    print(discussion_jsonl, flush=True)

    with open(manifest_path, "r+", encoding="utf-8") as f:
        manifest = json.load(f)
        manifest.update(
            {
                "completed_at": datetime.now().isoformat(),
                "num_samples": len(samples),
                "num_candidate_records": len(candidate_records),
                "num_fact_records": len(fact_records),
                "num_discussion_trajectories": len(trajectories),
                "files": {
                    "candidate_questions_jsonl": candidate_jsonl,
                    "candidate_questions_txt": candidate_txt,
                    "ranked_facts_jsonl": facts_jsonl,
                    "ranked_facts_txt": facts_txt,
                    "expert_discussions_jsonl": discussion_jsonl,
                    "expert_discussions_txt": discussion_txt,
                },
            }
        )
        f.seek(0)
        json.dump(manifest, f, indent=2)
        f.truncate()
    print(manifest_path, flush=True)


if __name__ == "__main__":
    main()
