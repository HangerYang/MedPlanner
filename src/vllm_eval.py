"""
vllm_eval.py — Standalone MedQA evaluation against a running vLLM OpenAI server.

Prompt format:
  [system]  You are a medical doctor ...
  [user]    <case context>
             Question: ...
             Options: A) ... B) ... C) ... D) ...
             In conclusion, the answer is:

Usage:
  python src/vllm_eval.py --config vllm_eval_config.json [--mode infer|eval|all]
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

SYSTEM_PROMPT = (
    "You are a medical doctor with expertise in clinical reasoning. "
    "Read the patient case carefully, then select the single best answer "
    "from the options provided. Respond with only the letter (A, B, C, or D)."
)

MEDIQ_SYSTEM_PROMPT = (
    "You are a medical doctor trying to reason through a real-life clinical case. "
    "Based on your understanding of basic and clinical science, medical knowledge, "
    "and mechanisms underlying health, disease, patient care, and modes of therapy, "
    "respond according to the task specified by the user. Base your response on the "
    "current and standard practices referenced in medical guidelines."
)

MEDIQ_TEMPLATE = (
    "A patient comes into the clinic presenting with a symptom as described in the "
    "conversation log below:\n    \n"
    "PATIENT INFORMATION: {patient_info}\n"
    "CONVERSATION LOG:\n{conv_log}\n"
    "QUESTION: {question}\n"
    "OPTIONS: {options_text}\n"
    "YOUR TASK: Assume that you already have enough information from the above "
    "question-answer pairs to answer the patient inquiry, use the above information "
    "to produce a factual conclusion. Respond with the correct letter choice "
    "(A, B, C, or D) and NOTHING ELSE.\n"
    "LETTER CHOICE: "
)


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_prompt(row: dict) -> list[dict]:
    context = row.get("context", [])
    question = row.get("question", "")
    options = row.get("options", {})

    lines = []
    if context:
        lines.append("Patient case:")
        for sent in context:
            lines.append(f"  {sent}")
        lines.append("")

    lines.append(f"Question: {question}")
    lines.append("")
    lines.append("Options:")
    for letter, text in sorted(options.items()):
        lines.append(f"  {letter}: {text}")
    lines.append("")
    lines.append("In conclusion, the answer is:")

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": "\n".join(lines)},
    ]


def build_prompt_mediq(row: dict) -> list[dict]:
    """Exact mediQ FixedExpert + FullContextPatient prompt."""
    context = row.get("context", [])
    question = row.get("question", "")
    options = row.get("options", {})

    patient_info = " ".join(context) if context else ""
    options_text = ", ".join(f"{k}: {v}" for k, v in sorted(options.items()))

    user_content = MEDIQ_TEMPLATE.format(
        patient_info=patient_info,
        conv_log="None",
        question=question,
        options_text=options_text,
    )
    return [
        {"role": "system", "content": MEDIQ_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


# ── Response parser ───────────────────────────────────────────────────────────

def strip_thinking(text: str) -> str:
    """Remove medgemma thinking blocks: <unused94>...<unused95>ANSWER"""
    stripped = re.sub(r'<[^>]+>.*?<[^>]+>', '', text, flags=re.DOTALL).strip()
    if not stripped:
        # thinking was cut off — take everything after the last '>'
        parts = re.split(r'>[^>]*$', text)
        stripped = parts[-1].strip() if len(parts) > 1 else text.strip()
    return stripped


def parse_choice(response_text: str, options: dict) -> str | None:
    text = strip_thinking(response_text).strip()
    # exact single letter
    if text in options:
        return text
    # "A:" / "A." / "(A)" / "Answer: A"
    m = re.search(r'\b([A-D])\b', text)
    if m and m.group(1) in options:
        return m.group(1)
    # full option text in response
    for letter, opt_text in options.items():
        if opt_text.lower() in text.lower():
            return letter
    return None


# ── Single inference call ─────────────────────────────────────────────────────

def infer_one(client: OpenAI, model: str, row: dict, cfg: dict, prompt_style: str = "default") -> dict:
    messages = build_prompt_mediq(row) if prompt_style == "mediq" else build_prompt(row)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
        )
        response_text = resp.choices[0].message.content.strip()
        input_tokens  = resp.usage.prompt_tokens
        output_tokens = resp.usage.completion_tokens
    except Exception as e:
        response_text = ""
        input_tokens = output_tokens = 0
        print(f"  [ERROR] row {row['id']}: {e}")

    pred = parse_choice(response_text, row["options"])
    correct = (pred == row["answer_idx"]) if pred else False

    return {
        "id":             row["id"],
        "question":       row["question"],
        "answer":         row["answer"],
        "answer_idx":     row["answer_idx"],
        "response_text":  response_text,
        "pred":           pred,
        "correct":        correct,
        "input_tokens":   input_tokens,
        "output_tokens":  output_tokens,
    }


# ── Infer mode ────────────────────────────────────────────────────────────────

def run_infer(cfg: dict, rows: list[dict], output_path: Path, prompt_style: str = "default"):
    base_url = f"http://{cfg['host']}:{cfg['port']}/v1"
    client   = OpenAI(api_key="sk-EMPTY", base_url=base_url)
    model    = cfg["served_model_name"]
    nproc    = cfg["api_nproc"]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"Resuming: {len(done_ids)} already done.")

    todo = [r for r in rows if r["id"] not in done_ids]
    total = len(rows)
    correct_count = sum(
        1 for line in (open(output_path).readlines() if output_path.exists() else [])
        if json.loads(line).get("correct")
    )

    t0 = time.time()
    with open(output_path, "a") as out_f:
        with ThreadPoolExecutor(max_workers=nproc) as pool:
            futures = {pool.submit(infer_one, client, model, row, cfg, prompt_style): row for row in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                result = fut.result()
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
                if result["correct"]:
                    correct_count += 1
                done_total = len(done_ids) + i
                acc = correct_count / done_total
                elapsed = time.time() - t0
                print(
                    f"\r[{done_total}/{total}] acc={acc:.3f} | "
                    f"{elapsed:.0f}s elapsed",
                    end="", flush=True,
                )
    print()


# ── Eval mode ─────────────────────────────────────────────────────────────────

def run_eval(output_path: Path):
    results = []
    with open(output_path) as f:
        for line in f:
            results.append(json.loads(line))

    results.sort(key=lambda x: x["id"])
    total   = len(results)
    correct = sum(1 for r in results if r["correct"])
    no_pred = sum(1 for r in results if r["pred"] is None)

    print(f"\n{'='*50}")
    print(f"Results: {output_path}")
    print(f"  Total:      {total}")
    print(f"  Correct:    {correct}")
    print(f"  Accuracy:   {correct/total:.4f} ({correct/total*100:.2f}%)")
    print(f"  No pred:    {no_pred} ({no_pred/total*100:.1f}%)")
    print(f"{'='*50}\n")
    return correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        required=True, help="Path to vllm_eval_config.json")
    parser.add_argument("--mode",          default="all", choices=["infer", "eval", "all"])
    parser.add_argument("--prompt_style",  default="default", choices=["default", "mediq"])
    parser.add_argument("--output_file",   default=None, help="Override output file from config")
    parser.add_argument("--max_questions", type=int, default=None, help="Limit rows (0 = all)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    repo_root = Path(args.config).parent
    data_path = repo_root / cfg["data_file"]
    out_path  = repo_root / (args.output_file if args.output_file else cfg["output_file"])

    with open(data_path) as f:
        rows = [json.loads(line) for line in f]

    limit = args.max_questions if args.max_questions is not None else cfg.get("max_questions", 0)
    if limit and limit > 0:
        rows = rows[:limit]

    print(f"Model:  {cfg['served_model_name']}  @ {cfg['host']}:{cfg['port']}")
    print(f"Data:   {data_path.name}  ({len(rows)} rows)")
    print(f"Output: {out_path}")
    print(f"Prompt: {args.prompt_style}")

    if args.mode in ("infer", "all"):
        run_infer(cfg, rows, out_path, args.prompt_style)
    if args.mode in ("eval", "all"):
        run_eval(out_path)


if __name__ == "__main__":
    main()
