#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_convo_answer_trajectory import parse_shadow_letter  # noqa: E402


HEADER_RE = re.compile(
    r"^Patient #(?P<id>\d+)\s+\|\s+(?P<label>\w+)\s+\|\s+"
    r"Predicted:\s*(?P<predicted>.*?)\s+\|\s+True:\s*(?P<true_letter>[A-D])\s+\((?P<true_answer>.*)\)\s*$",
    re.MULTILINE,
)
OPTIONS_RE = re.compile(
    r"^\s*Options:\s*A:\s*(?P<A>.*?)\s+B:\s*(?P<B>.*?)\s+C:\s*(?P<C>.*?)\s+D:\s*(?P<D>.*?)\s*$",
    re.MULTILINE,
)
NODE_RE = re.compile(
    r"^\s*─+\s*(?P<name>Root|Branch\s+[0-9-]+)\s+\|\s+Depth\s+(?P<depth>\d+)\s+\|\s+(?P<kind>LEAF|BRANCHING POINT)\s+─+\s*$",
    re.MULTILINE,
)


def split_patient_blocks(text):
    matches = list(HEADER_RE.finditer(text))
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        yield match, text[start:end]


def split_node_blocks(block):
    matches = list(NODE_RE.finditer(block))
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        yield match, block[start:end]


def extract_options(block):
    match = OPTIONS_RE.search(block)
    if not match:
        return {}
    return {letter: match.group(letter).strip() for letter in ("A", "B", "C", "D")}


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
    return out.strip() or None


def extract_shadow_answer(node_block):
    return extract_labeled_block(
        node_block,
        "Shadow Answer",
        [
            r"\[TOP-2",
            r"→\s*Final Answer:",
            r"Confidence:",
            r"Confidence Rationale:",
            r"Turn\s+\d+",
            r"Q1:",
            r"Q2:",
            r"─+",
        ],
    )


def extract_doctor_questions(node_block):
    questions = []
    for match in re.finditer(
        r"^\s*Doctor Q:\s*(?P<inline>.*?)(?=^\s*Patient:)",
        node_block,
        flags=re.MULTILINE | re.DOTALL,
    ):
        question = match.group("inline").strip()
        if question:
            questions.append(re.sub(r"\s+", " ", question))
    return questions


def extract_patient_answers(node_block):
    answers = []
    for match in re.finditer(
        r"^\s*Patient:\s*(?P<body>.*?)(?=^\s*(?:Turn\s+\d+|Confidence:|Shadow Answer:|→\s*Final Answer:|─+))",
        node_block,
        flags=re.MULTILINE | re.DOTALL,
    ):
        answer = match.group("body").strip()
        if answer:
            answers.append(re.sub(r"\s+", " ", answer))
    return answers


def extract_top_questions(node_block):
    questions = []
    for idx in (1, 2):
        section_match = re.search(
            rf"^\s*Q{idx}:\s*(?P<section>.*?)(?=^\s*(?:Q{idx + 1}:|─+|Turn\s+\d+|Confidence:|Shadow Answer:|→\s*Final Answer:)|\Z)",
            node_block,
            flags=re.MULTILINE | re.DOTALL,
        )
        if not section_match:
            continue
        section = section_match.group("section")
        question_match = re.search(
            r"^\s*Question:\s*(?P<question>.*)",
            section,
            flags=re.MULTILINE,
        )
        if not question_match:
            continue
        start = question_match.start("question")
        after = section[start:]
        stop = re.search(r"^\s*(?:Reason:|Q\d+:|─+|Turn\s+\d+)", after, re.MULTILINE)
        question = after[: stop.start()] if stop else after
        question = re.sub(r"\s+", " ", question).strip()
        if question:
            questions.append(question)
    return questions


def normalize_question(question):
    return re.sub(r"[^a-z0-9]+", " ", (question or "").lower()).strip()


def parse_patient(header, block):
    true_letter = header.group("true_letter")
    options = extract_options(block)
    nodes = []
    for node_match, node_block in split_node_blocks(block):
        raw_name = node_match.group("name")
        branch = "root" if raw_name == "Root" else raw_name.replace("Branch ", "")
        kind = node_match.group("kind")
        shadow_text = extract_shadow_answer(node_block)
        shadow_letter = parse_shadow_letter(shadow_text, options)
        final_match = re.search(r"→\s*Final Answer:\s*([A-D]|None)\b", node_block)
        final_answer = final_match.group(1) if final_match else None
        doctor_questions = extract_doctor_questions(node_block)
        patient_answers = extract_patient_answers(node_block)
        top_questions = extract_top_questions(node_block)
        nodes.append(
            {
                "branch": branch,
                "depth": int(node_match.group("depth")),
                "kind": kind,
                "is_leaf": kind == "LEAF",
                "num_turns": len(doctor_questions),
                "doctor_questions": doctor_questions,
                "patient_answers": patient_answers,
                "cannot_answer_count": sum(
                    "i cannot answer this question" in answer.lower()
                    for answer in patient_answers
                ),
                "shadow_answer_text": shadow_text,
                "shadow_answer_letter": shadow_letter,
                "shadow_answer_correct": (
                    shadow_letter == true_letter if shadow_letter is not None else None
                ),
                "final_answer": final_answer,
                "final_correct": final_answer == true_letter if final_answer else None,
                "top_questions": top_questions,
                "top_questions_parsed": len(top_questions),
            }
        )

    leaf_nodes = [node for node in nodes if node["is_leaf"]]
    all_questions = [
        question for node in leaf_nodes for question in node["doctor_questions"]
    ]
    normalized = [normalize_question(question) for question in all_questions]
    unique_questions = {question for question in normalized if question}

    return {
        "patient_id": int(header.group("id")),
        "header_label": header.group("label"),
        "header_predicted": header.group("predicted").strip(),
        "true_letter": true_letter,
        "true_answer": header.group("true_answer").strip(),
        "nodes": nodes,
        "num_nodes": len(nodes),
        "num_branching_nodes": sum(not node["is_leaf"] for node in nodes),
        "num_leaf_branches": len(leaf_nodes),
        "leaf_turn_counts": [node["num_turns"] for node in leaf_nodes],
        "leaf_final_answers": [node["final_answer"] for node in leaf_nodes],
        "any_leaf_final_correct": any(node["final_correct"] is True for node in leaf_nodes),
        "all_leaf_final_correct": bool(leaf_nodes) and all(node["final_correct"] is True for node in leaf_nodes),
        "any_shadow_correct": any(node["shadow_answer_correct"] is True for node in nodes),
        "last_shadow_correct": (
            nodes[-1]["shadow_answer_correct"] if nodes else None
        ),
        "num_shadow_answers": sum(node["shadow_answer_text"] is not None for node in nodes),
        "num_parseable_shadow_answers": sum(node["shadow_answer_letter"] is not None for node in nodes),
        "leaf_patient_answers": sum(len(node["patient_answers"]) for node in leaf_nodes),
        "leaf_cannot_answers": sum(node["cannot_answer_count"] for node in leaf_nodes),
        "unique_leaf_questions": len(unique_questions),
        "duplicate_leaf_questions": max(0, len(normalized) - len(unique_questions)),
    }


def mean(values):
    return sum(values) / len(values) if values else 0.0


def answer_stats_for_nodes(nodes):
    patient_answers = sum(len(node["patient_answers"]) for node in nodes)
    cannot_answers = sum(node["cannot_answer_count"] for node in nodes)
    return {
        "branches": len(nodes),
        "patient_answers": patient_answers,
        "cannot_answers": cannot_answers,
        "cannot_rate": cannot_answers / patient_answers if patient_answers else 0.0,
        "avg_turns": mean([node["num_turns"] for node in nodes]),
    }


def answer_stats_for_cases(cases):
    return {
        "cases": len(cases),
        "leaf_branches": sum(case["num_leaf_branches"] for case in cases),
        "patient_answers": sum(case["leaf_patient_answers"] for case in cases),
        "cannot_answers": sum(case["leaf_cannot_answers"] for case in cases),
    }


def summarize(cases):
    all_nodes = [node for case in cases for node in case["nodes"]]
    leaf_nodes = [node for node in all_nodes if node["is_leaf"]]
    leaf_turns = [node["num_turns"] for node in leaf_nodes]
    shadow_total = sum(node["shadow_answer_text"] is not None for node in all_nodes)
    shadow_parseable = sum(node["shadow_answer_letter"] is not None for node in all_nodes)
    branching_nodes = [node for node in all_nodes if not node["is_leaf"]]
    top_expected = 2 * len(branching_nodes)
    top_parsed = sum(node["top_questions_parsed"] for node in branching_nodes)
    leaf_patient_answers = sum(case["leaf_patient_answers"] for case in cases)
    leaf_cannot_answers = sum(case["leaf_cannot_answers"] for case in cases)
    correct_leaf_nodes = [node for node in leaf_nodes if node["final_correct"] is True]
    wrong_leaf_nodes = [node for node in leaf_nodes if node["final_correct"] is False]
    unparseable_leaf_nodes = [node for node in leaf_nodes if node["final_correct"] is None]
    any_correct_cases = [case for case in cases if case["any_leaf_final_correct"]]
    no_correct_cases = [case for case in cases if not case["any_leaf_final_correct"]]
    any_shadow_cases = [case for case in cases if case["any_shadow_correct"]]
    no_shadow_cases = [case for case in cases if not case["any_shadow_correct"]]

    return {
        "num_cases": len(cases),
        "num_header_correct": sum(case["header_label"] == "CORRECT" for case in cases),
        "num_nodes": len(all_nodes),
        "num_branching_nodes": len(branching_nodes),
        "num_leaf_branches": len(leaf_nodes),
        "avg_leaf_branches_per_case": mean([case["num_leaf_branches"] for case in cases]),
        "avg_leaf_turns": mean(leaf_turns),
        "min_leaf_turns": min(leaf_turns) if leaf_turns else 0,
        "max_leaf_turns": max(leaf_turns) if leaf_turns else 0,
        "leaf_turn_distribution": {
            str(turns): sum(turn == turns for turn in leaf_turns)
            for turns in sorted(set(leaf_turns))
        },
        "leaf_final_correct": sum(node["final_correct"] is True for node in leaf_nodes),
        "leaf_final_total": sum(node["final_answer"] is not None for node in leaf_nodes),
        "patients_any_leaf_final_correct": sum(case["any_leaf_final_correct"] for case in cases),
        "patients_all_leaf_final_correct": sum(case["all_leaf_final_correct"] for case in cases),
        "patients_never_leaf_final_correct": sum(not case["any_leaf_final_correct"] for case in cases),
        "patients_any_shadow_correct": sum(case["any_shadow_correct"] for case in cases),
        "patients_last_shadow_correct": sum(case["last_shadow_correct"] is True for case in cases),
        "patients_never_shadow_correct": sum(not case["any_shadow_correct"] for case in cases),
        "shadow_parseable": shadow_parseable,
        "shadow_total": shadow_total,
        "top_questions_parsed": top_parsed,
        "top_questions_expected": top_expected,
        "leaf_patient_answers": leaf_patient_answers,
        "leaf_cannot_answers": leaf_cannot_answers,
        "cannot_answer_rate": (
            leaf_cannot_answers / leaf_patient_answers if leaf_patient_answers else 0.0
        ),
        "cannot_by_leaf_final": {
            "correct_leaf": answer_stats_for_nodes(correct_leaf_nodes),
            "wrong_leaf": answer_stats_for_nodes(wrong_leaf_nodes),
            "unparseable_leaf": answer_stats_for_nodes(unparseable_leaf_nodes),
        },
        "cannot_by_patient_outcome": {
            "any_leaf_final_correct": answer_stats_for_cases(any_correct_cases),
            "no_leaf_final_correct": answer_stats_for_cases(no_correct_cases),
            "any_shadow_correct": answer_stats_for_cases(any_shadow_cases),
            "no_shadow_correct": answer_stats_for_cases(no_shadow_cases),
        },
        "unique_leaf_questions": sum(case["unique_leaf_questions"] for case in cases),
        "duplicate_leaf_questions": sum(case["duplicate_leaf_questions"] for case in cases),
    }


def write_summary(summary, path):
    leaf_acc = (
        summary["leaf_final_correct"] / summary["leaf_final_total"]
        if summary["leaf_final_total"]
        else 0.0
    )
    def fmt_node_stats(label, stats):
        return (
            f"{label}: {stats['cannot_answers']}/{stats['patient_answers']} = "
            f"{stats['cannot_rate']:.4f} "
            f"({stats['branches']} branches, avg turns {stats['avg_turns']:.2f})"
            if stats["patient_answers"]
            else f"{label}: n/a ({stats['branches']} branches)"
        )

    def fmt_case_stats(label, stats):
        rate = (
            stats["cannot_answers"] / stats["patient_answers"]
            if stats["patient_answers"]
            else 0.0
        )
        return (
            f"{label}: {stats['cannot_answers']}/{stats['patient_answers']} = "
            f"{rate:.4f} ({stats['cases']} cases, {stats['leaf_branches']} leaf branches)"
            if stats["patient_answers"]
            else f"{label}: n/a ({stats['cases']} cases, {stats['leaf_branches']} leaf branches)"
        )

    lines = [
        f"Cases parsed: {summary['num_cases']}",
        f"Header correct: {summary['num_header_correct']}/{summary['num_cases']} = {summary['num_header_correct'] / summary['num_cases']:.4f}" if summary["num_cases"] else "Header correct: n/a",
        f"Total branch nodes: {summary['num_nodes']}",
        f"Branching nodes: {summary['num_branching_nodes']}",
        f"Leaf branches: {summary['num_leaf_branches']}",
        f"Avg leaf branches per case: {summary['avg_leaf_branches_per_case']:.2f}",
        f"Leaf final correct: {summary['leaf_final_correct']}/{summary['leaf_final_total']} = {leaf_acc:.4f}",
        f"Patients with any leaf final correct: {summary['patients_any_leaf_final_correct']}/{summary['num_cases']} = {summary['patients_any_leaf_final_correct'] / summary['num_cases']:.4f}" if summary["num_cases"] else "Patients with any leaf final correct: n/a",
        f"Patients with all leaves final correct: {summary['patients_all_leaf_final_correct']}/{summary['num_cases']} = {summary['patients_all_leaf_final_correct'] / summary['num_cases']:.4f}" if summary["num_cases"] else "Patients with all leaves final correct: n/a",
        f"Patients with no leaf final correct: {summary['patients_never_leaf_final_correct']}/{summary['num_cases']} = {summary['patients_never_leaf_final_correct'] / summary['num_cases']:.4f}" if summary["num_cases"] else "Patients with no leaf final correct: n/a",
        f"Patients with any shadow correct: {summary['patients_any_shadow_correct']}/{summary['num_cases']} = {summary['patients_any_shadow_correct'] / summary['num_cases']:.4f}" if summary["num_cases"] else "Patients with any shadow correct: n/a",
        f"Patients with last shadow correct: {summary['patients_last_shadow_correct']}/{summary['num_cases']} = {summary['patients_last_shadow_correct'] / summary['num_cases']:.4f}" if summary["num_cases"] else "Patients with last shadow correct: n/a",
        f"Patients with no shadow correct: {summary['patients_never_shadow_correct']}/{summary['num_cases']} = {summary['patients_never_shadow_correct'] / summary['num_cases']:.4f}" if summary["num_cases"] else "Patients with no shadow correct: n/a",
        f"Shadow answer parse rate: {summary['shadow_parseable']}/{summary['shadow_total']} = {summary['shadow_parseable'] / summary['shadow_total']:.4f}" if summary["shadow_total"] else "Shadow answer parse rate: n/a",
        f"Top-2 proposal parse rate: {summary['top_questions_parsed']}/{summary['top_questions_expected']} = {summary['top_questions_parsed'] / summary['top_questions_expected']:.4f}" if summary["top_questions_expected"] else "Top-2 proposal parse rate: n/a",
        f"Avg turns per leaf branch: {summary['avg_leaf_turns']:.2f}",
        f"Min/max turns per leaf branch: {summary['min_leaf_turns']} / {summary['max_leaf_turns']}",
        "Leaf turn distribution: "
        + ", ".join(f"{k}:{v}" for k, v in summary["leaf_turn_distribution"].items()),
        f"Leaf patient answers: {summary['leaf_patient_answers']}",
        f"Leaf cannot-answer responses: {summary['leaf_cannot_answers']}/{summary['leaf_patient_answers']} = {summary['cannot_answer_rate']:.4f}" if summary["leaf_patient_answers"] else "Leaf cannot-answer responses: n/a",
        "",
        "Cannot-answer responses by leaf final outcome:",
        "  " + fmt_node_stats("Correct leaf finals", summary["cannot_by_leaf_final"]["correct_leaf"]),
        "  " + fmt_node_stats("Wrong leaf finals", summary["cannot_by_leaf_final"]["wrong_leaf"]),
        "  " + fmt_node_stats("Unparseable leaf finals", summary["cannot_by_leaf_final"]["unparseable_leaf"]),
        "",
        "Cannot-answer responses by patient-level branch outcome:",
        "  " + fmt_case_stats("Patients with any leaf final correct", summary["cannot_by_patient_outcome"]["any_leaf_final_correct"]),
        "  " + fmt_case_stats("Patients with no leaf final correct", summary["cannot_by_patient_outcome"]["no_leaf_final_correct"]),
        "  " + fmt_case_stats("Patients with any shadow correct", summary["cannot_by_patient_outcome"]["any_shadow_correct"]),
        "  " + fmt_case_stats("Patients with no shadow correct", summary["cannot_by_patient_outcome"]["no_shadow_correct"]),
        "",
        f"Unique leaf doctor questions summed per case: {summary['unique_leaf_questions']}",
        f"Duplicate leaf doctor questions summed per case: {summary['duplicate_leaf_questions']}",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze branching MediQ convo logs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    text = Path(args.input).read_text(errors="replace")
    cases = [parse_patient(header, block) for header, block in split_patient_blocks(text)]

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    summary = summarize(cases)
    write_summary(summary, Path(args.summary))

    print(f"Wrote {len(cases)} branch patient records to {out_path}")
    print(f"Wrote summary to {args.summary}")
    print(
        f"Patients with any leaf final correct: {summary['patients_any_leaf_final_correct']}/{summary['num_cases']}"
    )
    print(
        f"Patients with any shadow correct: {summary['patients_any_shadow_correct']}/{summary['num_cases']}"
    )


if __name__ == "__main__":
    main()
