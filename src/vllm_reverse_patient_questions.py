"""
For each row in all_dev_convo.jsonl, skip the first context sentence (opening vignette).
For every remaining context sentence, ask the served model for 5 distinct **physician**
questions addressed **to the patient** (history-taking / probing) that would reasonably
surface that fact in the patient's reply.

Uses the same OpenAI-compatible vLLM endpoint pattern as vllm_eval.py (parallel HTTP calls).

Example:
  python src/vllm_reverse_patient_questions.py --config vllm_eval_config.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

SYSTEM_PROMPT = (
    "You write realistic clinician-to-patient questions for medical education data. "
    "The expert is the physician: each line must be something a doctor would ask the "
    "patient or a family member during history-taking or clarifying questions "
    "(not multiple-choice, not board-exam stems). "
    "Always follow the output format requested in the user message exactly."
)


def build_user_prompt(row: dict, target_sentence: str) -> str:
    ctx = row.get("context") or []
    opening = ctx[0] if ctx else ""
    mcq = row.get("question", "")
    lines = []
    if opening:
        lines.append("Opening case line (for demographic grounding only; do NOT ask about this line):")
        lines.append(opening)
        lines.append("")
    lines.append("Board-style question being studied (context only; do not quote it in the questions):")
    lines.append(mcq)
    lines.append("")
    lines.append(
        "Target case fact (written as chart/vignette narrative; the patient may reveal "
        "this when answering a good history question):"
    )
    lines.append(target_sentence)
    lines.append("")
    lines.append(
        "Write exactly 5 different questions a **physician would ask the patient** "
        "(or a caregiver), with different wording and angle, that would most naturally "
        "lead the patient to volunteer or confirm the target fact above "
        "(or an equivalent paraphrase)."
    )
    lines.append("")
    lines.append(
        'Return ONLY valid JSON, one line, no markdown fences, with this exact shape:\n'
        '{"doctor_questions":["...","...","...","...","..."]}'
    )
    return "\n".join(lines)


def parse_doctor_questions_json(response_text: str) -> list[str] | None:
    text = response_text.strip()
    # Strip optional ```json fences
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    blob = m.group(0)
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return None
    qs = obj.get("doctor_questions")
    if not isinstance(qs, list):
        return None
    out = [str(q).strip() for q in qs if str(q).strip()]
    if len(out) != 5:
        return None
    return out


def infer_one(
    client: OpenAI,
    model: str,
    row: dict,
    context_idx: int,
    target_sentence: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(row, target_sentence)},
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        response_text = (resp.choices[0].message.content or "").strip()
        input_tokens = resp.usage.prompt_tokens
        output_tokens = resp.usage.completion_tokens
    except Exception as e:
        response_text = ""
        input_tokens = output_tokens = 0
        err = str(e)
        print(f"  [ERROR] id={row.get('id')} ctx_idx={context_idx}: {err}")
        return {
            "id": row["id"],
            "context_idx": context_idx,
            "target_sentence": target_sentence,
            "doctor_questions": None,
            "parse_ok": False,
            "response_text": response_text,
            "error": err,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    parsed = parse_doctor_questions_json(response_text)
    return {
        "id": row["id"],
        "context_idx": context_idx,
        "target_sentence": target_sentence,
        "doctor_questions": parsed,
        "parse_ok": parsed is not None,
        "response_text": response_text if parsed is None else "",
        "error": None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def load_done_keys(path: Path) -> set[tuple[int, int]]:
    done: set[tuple[int, int]] = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                if o.get("parse_ok"):
                    done.add((int(o["id"]), int(o["context_idx"])))
            except Exception:
                pass
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config (same host/port/model keys as vllm_eval)")
    parser.add_argument(
        "--data",
        default=None,
        help="Override data JSONL path (default: repo_root / data_file from config)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override output JSONL path (default: repo_root / reverse_doctor_output_file or src/results/reverse_doctor_questions.jsonl)",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Limit rows by id order (0 = all)")
    parser.add_argument("--api-nproc", type=int, default=None, help="Parallel workers (default: config api_nproc or 32)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.65)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    repo_root = Path(args.config).parent.resolve()
    data_path = Path(args.data) if args.data else repo_root / cfg["data_file"]
    if args.output:
        out_path = Path(args.output)
    else:
        rel = cfg.get("reverse_doctor_output_file", "src/results/reverse_doctor_questions.jsonl")
        out_path = repo_root / rel

    base_url = f"http://{cfg['host']}:{cfg['port']}/v1"
    client = OpenAI(api_key="sk-EMPTY", base_url=base_url)
    model = cfg["served_model_name"]
    nproc = args.api_nproc or cfg.get("api_nproc", 32)

    with open(data_path) as f:
        rows = [json.loads(line) for line in f]

    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]

    tasks: list[tuple[dict, int, str]] = []
    for row in rows:
        ctx = row.get("context") or []
        if len(ctx) <= 1:
            continue
        for j, sent in enumerate(ctx[1:], start=1):
            s = (sent or "").strip()
            if not s:
                continue
            tasks.append((row, j, s))

    done = load_done_keys(out_path)
    todo = [(r, j, s) for (r, j, s) in tasks if (int(r["id"]), j) not in done]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Model:  {model}  @ {cfg['host']}:{cfg['port']}")
    print(f"Data:   {data_path}  ({len(rows)} rows, {len(tasks)} targets, {len(todo)} todo)")
    print(f"Output: {out_path}")

    t0 = time.time()
    with open(out_path, "a") as out_f:
        with ThreadPoolExecutor(max_workers=nproc) as pool:
            futs = {
                pool.submit(
                    infer_one,
                    client,
                    model,
                    r,
                    j,
                    s,
                    args.max_tokens,
                    args.temperature,
                    args.top_p,
                ): (r, j)
                for (r, j, s) in todo
            }
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                elapsed = time.time() - t0
                print(f"\r[{i}/{len(todo)}] {elapsed:.0f}s", end="", flush=True)
    print()


if __name__ == "__main__":
    main()
