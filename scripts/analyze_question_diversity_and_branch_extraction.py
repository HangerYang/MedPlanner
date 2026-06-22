#!/usr/bin/env python3
import argparse
import difflib
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_branch_convo import parse_patient as parse_branch_patient, split_patient_blocks as split_branch_patient_blocks  # noqa: E402
from analyze_convo_answer_trajectory import (  # noqa: E402
    HEADER_RE,
    extract_shadow_answer,
    iter_turn_blocks,
    normalize_for_match,
    split_patient_blocks,
)


LINEAR_RUNS = [
    {
        "name": "medgemma_normal",
        "kind": "linear",
        "path": "/home/hyang/mediQ/logs/scale_medgemma4b_yes_options_100q_convo.txt",
    },
    {
        "name": "llama_normal",
        "kind": "linear",
        "path": "/home/hyang/mediQ/logs/scale_llama31_8b_100q_convo.txt",
    },
    {
        "name": "medgemma_hightemp",
        "kind": "linear",
        "path": "/home/hyang/mediQ/logs/scale_medgemma4b_hightemp_100q_convo.txt",
    },
    {
        "name": "llama_hightemp",
        "kind": "linear",
        "path": "/home/hyang/mediQ/logs/scale_llama31_8b_hightemp_100q_convo.txt",
    },
]

BRANCH_RUNS = [
    {
        "name": "medgemma_branch_d3",
        "kind": "branch",
        "path": "/home/hyang/mediQ/logs/scale_medgemma4b_branch_d3_5ex_convo.txt",
    },
    {
        "name": "llama_branch_d3",
        "kind": "branch",
        "path": "/home/hyang/mediQ/logs/scale_llama31_8b_branch_d3_5ex_convo.txt",
    },
]


def norm_question(question):
    text = (question or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def near_unique_count(questions, threshold):
    reps = []
    near_duplicates = 0
    for question in questions:
        normalized = norm_question(question)
        if not normalized:
            continue
        if any(difflib.SequenceMatcher(None, normalized, rep).ratio() >= threshold for rep in reps):
            near_duplicates += 1
        else:
            reps.append(normalized)
    return len(reps), near_duplicates


def diversity_stats(questions, threshold):
    normalized = [norm_question(question) for question in questions if norm_question(question)]
    total = len(normalized)
    exact_unique = len(set(normalized))
    near_unique, near_duplicates = near_unique_count(questions, threshold)
    return {
        "total_questions": total,
        "exact_unique_questions": exact_unique,
        "exact_duplicate_questions": total - exact_unique,
        "exact_unique_ratio": exact_unique / total if total else 0.0,
        "near_unique_questions": near_unique,
        "near_duplicate_questions": near_duplicates,
        "near_unique_ratio": near_unique / total if total else 0.0,
    }


def extract_labeled_block(text, label, stop_labels):
    marker = re.search(rf"^\s*{re.escape(label)}:\s*(?P<inline>.*)$", text, re.MULTILINE)
    if not marker:
        return None
    inline = marker.group("inline").strip()
    rest = text[marker.end():]
    stop_pattern = r"^\s*(?:" + "|".join(stop_labels) + r")"
    stop = re.search(stop_pattern, rest, re.MULTILINE)
    body = rest[: stop.start()] if stop else rest
    out = inline
    if body.strip():
        out = (out + "\n" + body.strip()).strip() if out else body.strip()
    return re.sub(r"\s+", " ", out).strip() or None


def extract_doctor_question(turn_block):
    return extract_labeled_block(
        turn_block,
        "Doctor Question",
        [r"Patient:", r"Shadow Answer:", r"Boxed Answer:", r"→\s*Committed", r"--- Turn"],
    )


def linear_questions(path):
    text = Path(path).read_text(errors="replace")
    rows = []
    for header, block in split_patient_blocks(text):
        patient_id = int(header.group("id"))
        for turn_match, turn_block in iter_turn_blocks(block):
            question = extract_doctor_question(turn_block)
            if question:
                rows.append(
                    {
                        "patient_id": patient_id,
                        "branch": None,
                        "depth": int(turn_match.group("turn")),
                        "question": question,
                    }
                )
    return rows


def branch_cases(path):
    text = Path(path).read_text(errors="replace")
    return [
        parse_branch_patient(header, block)
        for header, block in split_branch_patient_blocks(text)
    ]


def branch_new_question_rows(cases):
    rows = []
    for case in cases:
        for node in case["nodes"]:
            if node["depth"] <= 0 or not node["doctor_questions"]:
                continue
            rows.append(
                {
                    "patient_id": case["patient_id"],
                    "branch": node["branch"],
                    "depth": node["depth"],
                    "question": node["doctor_questions"][-1],
                }
            )
    return rows


def branch_leaf_question_rows(cases):
    rows = []
    for case in cases:
        for node in case["nodes"]:
            if not node["is_leaf"]:
                continue
            for idx, question in enumerate(node["doctor_questions"], start=1):
                rows.append(
                    {
                        "patient_id": case["patient_id"],
                        "branch": node["branch"],
                        "depth": idx,
                        "question": question,
                    }
                )
    return rows


def load_facts_by_id(data_path, limit):
    rows = {}
    for line in Path(data_path).read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if limit and len(rows) >= limit:
            break
        facts = row.get("atomic_facts") or row.get("facts") or row.get("context") or []
        rows[int(row["id"])] = facts
    return rows


def fact_body(fact):
    return re.sub(r"^\s*\d+\.\s*", "", fact or "").strip()


def extracted_fact_indices_from_text(text, facts):
    norm_answer = normalize_for_match(text)
    found = set()
    for idx, fact in enumerate(facts):
        norm_full = normalize_for_match(fact)
        norm_body = normalize_for_match(fact_body(fact))
        if (norm_full and norm_full in norm_answer) or (norm_body and norm_body in norm_answer):
            found.add(idx)
    return found


def branch_turn_extraction(cases, facts_by_id, max_depth=3):
    per_depth = {
        depth: {
            "depth": depth,
            "patient_answers": 0,
            "cannot_answers": 0,
            "extracted_facts": 0,
            "total_facts": 0,
        }
        for depth in range(0, max_depth + 1)
    }
    per_patient_rows = []
    for case in cases:
        facts = facts_by_id.get(case["patient_id"], [])
        total_facts = len(facts)
        exact_by_depth = {depth: [] for depth in range(1, max_depth + 1)}
        for node in case["nodes"]:
            depth = node["depth"]
            if depth <= 0 or depth > max_depth or not node["patient_answers"]:
                continue
            exact_by_depth[depth].append(node["patient_answers"][-1])

        cumulative_found = set()
        for depth in range(0, max_depth + 1):
            if depth > 0:
                for answer in exact_by_depth.get(depth, []):
                    cumulative_found.update(extracted_fact_indices_from_text(answer, facts))
            patient_answers = sum(len(exact_by_depth.get(d, [])) for d in range(1, depth + 1))
            cannot_answers = sum(
                "i cannot answer this question" in answer.lower()
                for d in range(1, depth + 1)
                for answer in exact_by_depth.get(d, [])
            )
            per_depth[depth]["patient_answers"] += patient_answers
            per_depth[depth]["cannot_answers"] += cannot_answers
            per_depth[depth]["extracted_facts"] += len(cumulative_found)
            per_depth[depth]["total_facts"] += total_facts
            per_patient_rows.append(
                {
                    "patient_id": case["patient_id"],
                    "depth": depth,
                    "patient_answers_cumulative": patient_answers,
                    "cannot_answers_cumulative": cannot_answers,
                    "extracted_facts_cumulative": len(cumulative_found),
                    "total_facts": total_facts,
                    "extracted_ratio": len(cumulative_found) / total_facts if total_facts else 0.0,
                }
            )
    for stats in per_depth.values():
        stats["fact_extraction_rate"] = (
            stats["extracted_facts"] / stats["total_facts"]
            if stats["total_facts"]
            else 0.0
        )
        stats["cannot_answer_rate"] = (
            stats["cannot_answers"] / stats["patient_answers"]
            if stats["patient_answers"]
            else 0.0
        )
    return list(per_depth.values()), per_patient_rows


def summarize_run_questions(run_name, run_kind, question_rows, threshold):
    questions = [row["question"] for row in question_rows]
    run_stats = {
        "run": run_name,
        "kind": run_kind,
        **diversity_stats(questions, threshold),
    }
    per_patient = []
    by_patient = {}
    for row in question_rows:
        by_patient.setdefault(row["patient_id"], []).append(row["question"])
    patient_stats = []
    for patient_id, patient_questions in sorted(by_patient.items()):
        stats = diversity_stats(patient_questions, threshold)
        stats.update({"run": run_name, "kind": run_kind, "patient_id": patient_id})
        per_patient.append(stats)
        patient_stats.append(stats)
    run_stats["patients_with_questions"] = len(patient_stats)
    run_stats["avg_questions_per_patient"] = (
        sum(item["total_questions"] for item in patient_stats) / len(patient_stats)
        if patient_stats
        else 0.0
    )
    run_stats["avg_exact_unique_ratio_per_patient"] = (
        sum(item["exact_unique_ratio"] for item in patient_stats) / len(patient_stats)
        if patient_stats
        else 0.0
    )
    run_stats["avg_near_unique_ratio_per_patient"] = (
        sum(item["near_unique_ratio"] for item in patient_stats) / len(patient_stats)
        if patient_stats
        else 0.0
    )
    return run_stats, per_patient


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path, run_rows, branch_extract_rows):
    lines = ["Question diversity summary", ""]
    for row in run_rows:
        lines.extend(
            [
                f"{row['run']} ({row['kind']}):",
                f"  Total questions: {row['total_questions']}",
                f"  Exact unique: {row['exact_unique_questions']}/{row['total_questions']} = {row['exact_unique_ratio']:.4f}",
                f"  Near unique: {row['near_unique_questions']}/{row['total_questions']} = {row['near_unique_ratio']:.4f}",
                f"  Exact duplicates: {row['exact_duplicate_questions']}",
                f"  Near duplicates: {row['near_duplicate_questions']}",
                f"  Avg questions/patient: {row['avg_questions_per_patient']:.2f}",
                f"  Avg exact unique ratio/patient: {row['avg_exact_unique_ratio_per_patient']:.4f}",
                f"  Avg near unique ratio/patient: {row['avg_near_unique_ratio_per_patient']:.4f}",
                "",
            ]
        )

    lines.extend(["Branch cumulative fact extraction by parallel depth", ""])
    by_run = {}
    for row in branch_extract_rows:
        by_run.setdefault(row["run"], []).append(row)
    for run, rows in by_run.items():
        lines.append(f"{run}:")
        for row in sorted(rows, key=lambda item: item["depth"]):
            lines.append(
                f"  Depth {row['depth']}: facts {row['extracted_facts']}/{row['total_facts']} = "
                f"{row['fact_extraction_rate']:.4f}; cannot-answer "
                f"{row['cannot_answers']}/{row['patient_answers']} = {row['cannot_answer_rate']:.4f}"
                if row["patient_answers"]
                else f"  Depth {row['depth']}: facts {row['extracted_facts']}/{row['total_facts']} = {row['fact_extraction_rate']:.4f}; cannot-answer n/a"
            )
        lines.append("")
    Path(path).write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="/home/hyang/mediQ/results")
    parser.add_argument("--data", default="/home/hyang/mediQ/data/med_data/all_dev_convo.jsonl")
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--near-threshold", type=float, default=0.85)
    parser.add_argument("--max-branch-depth", type=int, default=3)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    run_rows = []
    patient_rows = []
    branch_extract_rows = []
    branch_extract_patient_rows = []

    for run in LINEAR_RUNS:
        path = Path(run["path"])
        if not path.exists():
            continue
        question_rows = linear_questions(path)
        stats, per_patient = summarize_run_questions(
            run["name"], run["kind"], question_rows, args.near_threshold
        )
        run_rows.append(stats)
        patient_rows.extend(per_patient)

    facts_by_id = load_facts_by_id(args.data, args.max_examples)
    for run in BRANCH_RUNS:
        path = Path(run["path"])
        if not path.exists():
            continue
        cases = branch_cases(path)
        new_question_rows = branch_new_question_rows(cases)
        leaf_question_rows = branch_leaf_question_rows(cases)

        stats, per_patient = summarize_run_questions(
            run["name"] + "_new_branch_questions",
            "branch_new_questions",
            new_question_rows,
            args.near_threshold,
        )
        run_rows.append(stats)
        patient_rows.extend(per_patient)

        leaf_stats, leaf_per_patient = summarize_run_questions(
            run["name"] + "_leaf_path_questions",
            "branch_leaf_path_questions",
            leaf_question_rows,
            args.near_threshold,
        )
        run_rows.append(leaf_stats)
        patient_rows.extend(leaf_per_patient)

        depth_rows, depth_patient_rows = branch_turn_extraction(
            cases, facts_by_id, max_depth=args.max_branch_depth
        )
        for row in depth_rows:
            row["run"] = run["name"]
        for row in depth_patient_rows:
            row["run"] = run["name"]
        branch_extract_rows.extend(depth_rows)
        branch_extract_patient_rows.extend(depth_patient_rows)

    write_jsonl(results_dir / "question_diversity_by_run.jsonl", run_rows)
    write_jsonl(results_dir / "question_diversity_by_patient.jsonl", patient_rows)
    write_jsonl(results_dir / "branch_cumulative_fact_extraction_by_depth.jsonl", branch_extract_rows)
    write_jsonl(results_dir / "branch_cumulative_fact_extraction_by_patient.jsonl", branch_extract_patient_rows)
    write_summary(
        results_dir / "question_diversity_and_branch_extraction_summary.txt",
        run_rows,
        branch_extract_rows,
    )
    print(f"Wrote summary to {results_dir / 'question_diversity_and_branch_extraction_summary.txt'}")
    print(f"Wrote {len(run_rows)} run diversity rows")
    print(f"Wrote {len(branch_extract_rows)} branch extraction depth rows")


if __name__ == "__main__":
    main()
