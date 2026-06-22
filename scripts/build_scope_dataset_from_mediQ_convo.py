#!/usr/bin/env python3
"""Build a SCOPE-compatible training dataset from MediQ branching convo logs.

Reads the human-readable ``*_convo.txt`` and ``*_doctor_view.txt`` outputs
(plus optional ``*_results.jsonl``) and writes:

  - ``*_scope_trajectories.jsonl`` — one row per leaf branch with labels + conversation
  - ``*_scope_hf/`` — HuggingFace dataset on disk (``conversation`` column for embed_dataset)
  - ``*_scope_dataset_summary.txt`` — counts and label statistics

Conversation format matches ``scope_mediq_runner.py`` / lmsys-chat-1m:
  user (patient case) -> assistant (doctor Q) -> user (patient A) -> ...
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_branch_convo import (  # noqa: E402
    HEADER_RE,
    OPTIONS_RE,
    extract_labeled_block,
    split_patient_blocks,
)


DOCTOR_BRANCH_RE = re.compile(
    r"^\s*─+\s*Branch\s+(?P<branch_id>[\w-]+)\s+\|\s+(?P<branch_label>CORRECT|WRONG)\s+\|\s+"
    r"Final:\s+(?P<final_letter>[A-D])\s+─+\s*$",
    re.MULTILINE,
)
TURN_HEADER_RE = re.compile(r"^\s*--- Turn\s+(?P<turn>\d+)\s+-+", re.MULTILINE)
INITIAL_RE = re.compile(r"^\s*Initial:\s*(?P<body>.*?)(?=^\s*Question:|\Z)", re.MULTILINE | re.DOTALL)
QUESTION_RE = re.compile(r"^\s*Question:\s*(?P<body>.*?)(?=^\s*Full context|\s*Options:|\Z)", re.MULTILINE | re.DOTALL)
FULL_CONTEXT_RE = re.compile(
    r"^\s*Full context \(all segments\):\s*(?P<body>.*?)(?=^\s*Options:|\Z)",
    re.MULTILINE | re.DOTALL,
)


def _collapse_ws(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip())


def extract_options(block: str) -> dict[str, str]:
    match = OPTIONS_RE.search(block)
    if not match:
        return {}
    return {letter: match.group(letter).strip() for letter in ("A", "B", "C", "D")}


def extract_patient_case(block: str) -> dict[str, str]:
    initial = INITIAL_RE.search(block)
    question = QUESTION_RE.search(block)
    full_context = FULL_CONTEXT_RE.search(block)
    options = extract_options(block)
    return {
        "initial_info": _collapse_ws(initial.group("body")) if initial else "",
        "question": _collapse_ws(question.group("body")) if question else "",
        "full_context": _collapse_ws(full_context.group("body")) if full_context else "",
        "options": options,
    }


def build_starter_text(case: dict[str, str]) -> str:
    context = case["full_context"] or case["initial_info"]
    opts = ", ".join(f"{k}: {v}" for k, v in case["options"].items())
    parts = [f"A patient presents with: {context}"]
    if case["question"]:
        parts.append(f"Clinical question: {case['question']}")
    if opts:
        parts.append(f"Options: {opts}")
    return "\n\n".join(parts)


def extract_doctor_patient_turns(branch_block: str) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    turn_matches = list(TURN_HEADER_RE.finditer(branch_block))
    for idx, match in enumerate(turn_matches):
        turn_start = match.end()
        turn_end = (
            turn_matches[idx + 1].start() if idx + 1 < len(turn_matches) else len(branch_block)
        )
        sub = branch_block[turn_start:turn_end]
        doctor = extract_labeled_block(
            sub,
            "Doctor",
            ["Patient", "--- Turn", "→", "─"],
        )
        patient = extract_labeled_block(
            sub,
            "Patient",
            ["Doctor", "--- Turn", "→", "─"],
        )
        if doctor and patient:
            turns.append(
                {
                    "turn_index": int(match.group("turn")),
                    "question": _collapse_ws(doctor),
                    "answer": _collapse_ws(patient),
                    "cannot_answer": "i cannot answer this question" in patient.lower(),
                }
            )
    return turns


def build_conversation(starter: str, qa_turns: list[dict[str, str]], final_letter: str | None) -> list[dict[str, str]]:
    conversation = [{"role": "user", "content": starter}]
    for turn in qa_turns:
        conversation.append({"role": "assistant", "content": turn["question"]})
        conversation.append({"role": "user", "content": turn["answer"]})
    if final_letter:
        conversation.append(
            {"role": "assistant", "content": f"FINAL ANSWER: {final_letter}"}
        )
    return conversation


def parse_doctor_view_patient(header_match: re.Match[str], block: str) -> list[dict]:
    patient_id = int(header_match.group("id"))
    true_letter = header_match.group("true_letter").strip()
    true_answer = header_match.group("true_answer").strip()
    predicted_header = header_match.group("predicted").strip()
    case = extract_patient_case(block)
    starter = build_starter_text(case)

    rows: list[dict] = []
    branch_matches = list(DOCTOR_BRANCH_RE.finditer(block))
    for branch_idx, branch_match in enumerate(branch_matches):
        branch_start = branch_match.end()
        branch_end = (
            branch_matches[branch_idx + 1].start()
            if branch_idx + 1 < len(branch_matches)
            else len(block)
        )
        branch_block = block[branch_start:branch_end]
        branch_id = branch_match.group("branch_id").strip()
        final_letter = branch_match.group("final_letter").strip()
        branch_label = branch_match.group("branch_label").strip()
        qa_turns = extract_doctor_patient_turns(branch_block)
        num_turns = len(qa_turns)
        cannot_count = sum(turn["cannot_answer"] for turn in qa_turns)
        final_correct = final_letter == true_letter
        conversation = build_conversation(starter, qa_turns, final_letter)

        rows.append(
            {
                "patient_id": patient_id,
                "branch_id": branch_id,
                "branch_label": branch_label,
                "final_correct": final_correct,
                "final_answer_letter": final_letter,
                "ground_truth_letter": true_letter,
                "ground_truth_answer": true_answer,
                "header_predicted_letter": predicted_header
                if predicted_header in {"A", "B", "C", "D"}
                else None,
                "header_correct_label": header_match.group("label").strip(),
                "num_turns": num_turns,
                "num_messages": len(conversation),
                "num_cannot_answer_turns": cannot_count,
                "options": case["options"],
                "question": case["question"],
                "initial_info": case["initial_info"],
                "full_context": case["full_context"],
                "starter_text": starter,
                "qa_turns": qa_turns,
                "conversation": conversation,
            }
        )
    return rows


def load_results_index(path: Path | None) -> dict[int, dict]:
    if path is None or not path.exists():
        return {}
    rows: dict[int, dict] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[int(row["id"])] = row
    return rows


def attach_results_metadata(rows: list[dict], results_by_id: dict[int, dict]) -> None:
    for row in rows:
        result = results_by_id.get(row["patient_id"])
        if not result:
            continue
        info = result.get("info") or {}
        interactive = result.get("interactive_system") or {}
        row["benchmark_id"] = result.get("id")
        row["benchmark_final_correct"] = interactive.get("correct")
        row["benchmark_predicted_letter"] = interactive.get("letter_choice")
        row["benchmark_num_questions"] = interactive.get("num_questions")


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    patients = {row["patient_id"] for row in rows}
    final_correct = sum(row["final_correct"] for row in rows)
    turn_counts = [row["num_turns"] for row in rows]
    msg_counts = [row["num_messages"] for row in rows]
    cannot_rows = sum(row["num_cannot_answer_turns"] > 0 for row in rows)
    return {
        "num_trajectories": n,
        "num_patients": len(patients),
        "num_final_correct": final_correct,
        "leaf_accuracy": final_correct / n if n else None,
        "avg_turns": sum(turn_counts) / n if n else None,
        "min_turns": min(turn_counts) if turn_counts else 0,
        "max_turns": max(turn_counts) if turn_counts else 0,
        "avg_messages": sum(msg_counts) / n if n else None,
        "trajectories_with_cannot_answer": cannot_rows,
        "turn_distribution": {
            str(t): turn_counts.count(t) for t in sorted(set(turn_counts))
        },
    }


def write_summary(summary: dict, path: Path, args: argparse.Namespace) -> None:
    lines = [
        "MediQ -> SCOPE dataset summary",
        f"convo_log: {args.convo}",
        f"doctor_view: {args.doctor_view}",
        f"results_jsonl: {args.results_jsonl or '(not provided)'}",
        "",
        f"Trajectories (leaf branches): {summary['num_trajectories']}",
        f"Unique patients: {summary['num_patients']}",
        f"Leaf final correct: {summary['num_final_correct']}",
        f"Leaf accuracy: {summary['leaf_accuracy']:.4f}"
        if summary["leaf_accuracy"] is not None
        else "Leaf accuracy: n/a",
        f"Avg doctor-patient turns per leaf: {summary['avg_turns']:.2f}"
        if summary["avg_turns"] is not None
        else "Avg turns: n/a",
        f"Min/max turns: {summary['min_turns']} / {summary['max_turns']}",
        f"Avg messages per conversation: {summary['avg_messages']:.2f}"
        if summary["avg_messages"] is not None
        else "Avg messages: n/a",
        f"Trajectories with >=1 cannot-answer patient reply: {summary['trajectories_with_cannot_answer']}",
        "Turn count distribution: "
        + ", ".join(f"{k}:{v}" for k, v in summary["turn_distribution"].items()),
        "",
        "SCOPE training pipeline:",
        "  1. Embed conversations:",
        "     cd convo-plan-SCOPE && python3 train/embed_mediQ_scope_dataset.py \\",
        f"       --dataset {args.output_hf}",
        "  2. Train transition model:",
        "     python3 train/train_transition.py --dataset embeddings/<name> --seed=0",
        "",
        "Conversation roles: user=patient, assistant=doctor (matches scope_mediq_runner.py).",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_hf_dataset(rows: list[dict], path: Path) -> None:
    import datasets

    hf_rows = {
        "patient_id": [r["patient_id"] for r in rows],
        "branch_id": [r["branch_id"] for r in rows],
        "conversation": [r["conversation"] for r in rows],
        "final_correct": [r["final_correct"] for r in rows],
        "final_answer_letter": [r["final_answer_letter"] for r in rows],
        "ground_truth_letter": [r["ground_truth_letter"] for r in rows],
        "ground_truth_answer": [r["ground_truth_answer"] for r in rows],
        "num_turns": [r["num_turns"] for r in rows],
        "num_messages": [r["num_messages"] for r in rows],
    }
    ds = datasets.Dataset.from_dict(hf_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(path))


def default_paths_from_convo(convo_path: Path) -> dict[str, Path]:
    stem = convo_path.name
    if stem.endswith("_convo.txt"):
        prefix = stem[: -len("_convo.txt")]
    else:
        prefix = convo_path.stem
    parent = convo_path.parent
    return {
        "doctor_view": parent / f"{prefix}_doctor_view.txt",
        "results_jsonl": parent / f"{prefix}_results.jsonl",
        "output_jsonl": parent / f"{prefix}_scope_trajectories.jsonl",
        "output_hf": parent / f"{prefix}_scope_hf",
        "summary": parent / f"{prefix}_scope_dataset_summary.txt",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MediQ branching convo logs into a SCOPE training dataset."
    )
    parser.add_argument(
        "--convo",
        type=Path,
        required=True,
        help="Path to *_convo.txt (used for default output names; optional metadata).",
    )
    parser.add_argument(
        "--doctor-view",
        type=Path,
        default=None,
        help="Path to *_doctor_view.txt (primary source for leaf Q&A).",
    )
    parser.add_argument(
        "--results-jsonl",
        type=Path,
        default=None,
        help="Optional benchmark results JSONL for extra metadata.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Output JSONL with labels + full conversation per leaf branch.",
    )
    parser.add_argument(
        "--output-hf",
        type=Path,
        default=None,
        help="HuggingFace dataset directory (conversation column for embedding).",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Plain-text summary output path.",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=0,
        help="If >0, only process the first N patient blocks (for debugging).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = default_paths_from_convo(args.convo)
    doctor_view_path = args.doctor_view or defaults["doctor_view"]
    results_path = args.results_jsonl if args.results_jsonl is not None else defaults["results_jsonl"]
    output_jsonl = args.output_jsonl or defaults["output_jsonl"]
    output_hf = args.output_hf or defaults["output_hf"]
    summary_path = args.summary or defaults["summary"]

    if not doctor_view_path.exists():
        raise FileNotFoundError(
            f"doctor_view not found: {doctor_view_path}\n"
            "Pass --doctor-view explicitly or ensure the file sits next to the convo log."
        )

    text = doctor_view_path.read_text(encoding="utf-8", errors="replace")
    patient_blocks = list(split_patient_blocks(text))
    if args.max_patients > 0:
        patient_blocks = patient_blocks[: args.max_patients]

    all_rows: list[dict] = []
    for header, block in patient_blocks:
        all_rows.extend(parse_doctor_view_patient(header, block))

    results_by_id = load_results_index(results_path)
    attach_results_metadata(all_rows, results_by_id)

    for sample_idx, row in enumerate(all_rows):
        row["sample_id"] = sample_idx

    summary = summarize(all_rows)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    save_hf_dataset(all_rows, output_hf)
    write_summary(summary, summary_path, argparse.Namespace(
        convo=args.convo,
        doctor_view=doctor_view_path,
        results_jsonl=results_path,
        output_hf=output_hf,
    ))

    print(f"Wrote {len(all_rows)} trajectories to {output_jsonl}")
    print(f"Wrote HuggingFace dataset to {output_hf}")
    print(f"Wrote summary to {summary_path}")
    if summary["leaf_accuracy"] is not None:
        print(
            f"Leaf accuracy: {summary['num_final_correct']}/{summary['num_trajectories']} "
            f"= {summary['leaf_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
