#!/usr/bin/env python3
import argparse
import difflib
import json
import re
from pathlib import Path


HEADER_RE = re.compile(
    r"^Patient #(?P<id>\d+)\s+\|\s+(?P<label>\w+)\s+\|\s+"
    r"Predicted:\s*(?P<predicted>.*?)\s+\|\s+True:\s*(?P<true_letter>[A-D])\s+\((?P<true_answer>.*)\)\s*$",
    re.MULTILINE,
)
TURN_RE = re.compile(
    r"^\s*--- Turn (?P<turn>\d+)(?P<final>\s+\(Final Decision\))?.*$",
    re.MULTILINE,
)
# Branching convo nodes look like:
#   ───────── Root | Depth 0 | LEAF ─────────
#   ───────── Branch 2-2-1 | Depth 3 | BRANCHING POINT ─────────
BRANCH_NODE_RE = re.compile(
    r"^\s*[─\-]+\s*(?P<branch_id>Root|Branch\s+\S+)\s*\|\s*Depth\s+(?P<depth>\d+)\s*\|\s*"
    r"(?P<kind>LEAF|BRANCHING POINT)\s*[─\-]+\s*$",
    re.MULTILINE,
)
COMMITTED_RE = re.compile(
    r"^\s*→ Committed to answer:\s*(?P<answer>[A-D]|None)\s*$",
    re.MULTILINE,
)
# Branching convos write "→ Final Answer: X" at every LEAF instead of a single
# "→ Committed to answer:" line.
FINAL_ANSWER_RE = re.compile(
    r"^\s*→ Final Answer:\s*(?P<answer>[A-D]|None)\s*$",
    re.MULTILINE,
)
OPTIONS_RE = re.compile(
    r"^\s*Options:\s*A:\s*(?P<A>.*?)\s+B:\s*(?P<B>.*?)\s+C:\s*(?P<C>.*?)\s+D:\s*(?P<D>.*?)\s*$",
    re.MULTILINE,
)


def split_patient_blocks(text):
    matches = list(HEADER_RE.finditer(text))
    for i, match in enumerate(matches):
        block_start = match.start()
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        yield match, text[block_start:block_end]


def iter_turn_blocks(block):
    """Yield (turn_number, is_final_decision, sub_block) tuples.

    Supports two convo formats:
      * Flat: "--- Turn N (Final Decision)?" markers (legacy ScaleExpert).
      * Branching: "─── Root|Branch X | Depth N | LEAF|BRANCHING POINT ───"
        node headers. Depth is used as the turn number; LEAF nodes are treated
        as final-decision turns since they emit a "→ Final Answer:" line.
    """
    flat_matches = list(TURN_RE.finditer(block))
    if flat_matches:
        for i, match in enumerate(flat_matches):
            turn_start = match.start()
            turn_end = (
                flat_matches[i + 1].start() if i + 1 < len(flat_matches) else len(block)
            )
            yield {
                "turn": int(match.group("turn")),
                "is_final": bool(match.group("final")),
                "kind": "flat",
                "branch_id": None,
            }, block[turn_start:turn_end]
        return

    branch_matches = list(BRANCH_NODE_RE.finditer(block))
    for i, match in enumerate(branch_matches):
        turn_start = match.start()
        turn_end = (
            branch_matches[i + 1].start() if i + 1 < len(branch_matches) else len(block)
        )
        yield {
            "turn": int(match.group("depth")),
            "is_final": match.group("kind") == "LEAF",
            "kind": "branch",
            "branch_id": match.group("branch_id").strip(),
        }, block[turn_start:turn_end]


def extract_shadow_answer(turn_block):
    marker = re.search(r"^\s*Shadow Answer:(?P<inline>.*)$", turn_block, re.MULTILINE)
    if not marker:
        return None

    inline = marker.group("inline").strip()
    rest = turn_block[marker.end() :]
    stop = re.search(
        r"^\s*(?:Boxed Answer:|Doctor Question:|Patient:|→ Committed to answer:"
        r"|→ Final Answer:|--- Turn |[─\-]{3,})",
        rest,
        re.MULTILINE,
    )
    body = rest[: stop.start()] if stop else rest
    text = inline
    if body.strip():
        text = (text + "\n" + body.strip()).strip() if text else body.strip()
    return text.strip() or None


def extract_options(block):
    match = OPTIONS_RE.search(block)
    if not match:
        return {}
    return {letter: match.group(letter).strip() for letter in ("A", "B", "C", "D")}


def normalize_for_match(text):
    text = re.sub(r"(\w)-\s+(\w)", r"\1-\2", text or "")
    text = re.sub(r"<[^>]*>", " ", text)
    text = text.replace("```", " ").replace("**", " ")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def conclusion_span(text):
    clean = re.sub(r"(\w)-\s+(\w)", r"\1-\2", text or "")
    clean = re.sub(r"<[^>]*>", " ", clean)
    markers = [
        "final answer",
        "therefore",
        "thus",
        "in conclusion",
        "most likely",
        "best treatment",
        "most probable cause",
        "correct answer",
        "best course",
    ]
    lower = clean.lower()
    positions = [lower.rfind(marker) for marker in markers]
    pos = max(positions)
    if pos >= 0:
        return clean[max(0, pos - 500):]
    return clean[-900:]


def option_text_match(text, options):
    if not text or not options:
        return None

    span = conclusion_span(text)
    norm_span = normalize_for_match(span)
    if not norm_span:
        return None

    normalized_options = {
        letter: normalize_for_match(option_text)
        for letter, option_text in options.items()
        if option_text
    }
    exact_matches = [
        (norm_span.rfind(norm_option), letter)
        for letter, norm_option in normalized_options.items()
        if norm_option and norm_option in norm_span
    ]
    if exact_matches:
        return max(exact_matches)[1]

    scores = []
    for letter, norm_option in normalized_options.items():
        if not norm_option:
            continue
        score = difflib.SequenceMatcher(None, norm_span, norm_option).ratio()
        token_hits = sum(
            1 for token in norm_option.split() if len(token) > 3 and token in norm_span
        )
        scores.append((score, token_hits, letter))
    if not scores:
        return None

    scores.sort(reverse=True)
    best_score, best_hits, best_letter = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else 0.0
    if best_score >= 0.72 and best_score - second_score >= 0.12:
        return best_letter
    if best_hits >= 2 and best_score - second_score >= 0.08:
        return best_letter
    return None


def parse_shadow_letter(text, options=None):
    if not text:
        return None

    tag_match = re.search(r"<unused\d+>\s*([A-D])\b", text)
    if tag_match:
        return tag_match.group(1).upper()

    clean = re.sub(r"<[^>]*>", " ", text)
    clean = clean.replace("```", " ")

    patterns = [
        r"\bLETTER\s+CHOICE\s*[:\-]?\s*([A-D])\b",
        r"\bANSWER\s*[:\-]?\s*([A-D])\b",
        r"\bFINAL\s+ANSWER\b[\s:*_\-]*(?:the\s+final\s+answer\s+is\s*)?(?:\$?\\boxed?\{)?\s*([A-D])\b",
        r"\\boxed?\{([A-D])\}",
        r"\bmost\s+likely(?:\s+\w+){0,8}\s+is\s+\**([A-D])\s*:",
        r"\bbest\s+treatment(?:\s+\w+){0,4}\s+is\s+\**([A-D])\s*:",
        r"\b(?:answer|choice|option)\s+(?:is|:)\s*\**([A-D])\b",
        r"\b(?:choose|select|pick)\s+\**([A-D])\b",
        r"\bOPTION\s*([A-D])\b",
        r"\bCHOICE\s*([A-D])\b",
    ]
    tail = clean[-800:]
    for pattern in patterns:
        match = re.search(pattern, tail, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    for pattern in patterns:
        match = re.search(pattern, clean, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    stripped = clean.strip().strip(".:;,*_`'\"")
    if re.fullmatch(r"[A-D]", stripped):
        return stripped

    # Last resort: if the final non-empty line is just a letter-like answer.
    for line in reversed(clean.splitlines()):
        stripped_line = line.strip().strip(".:;,*_`'\"")
        if re.fullmatch(r"[A-D]", stripped_line):
            return stripped_line

    return option_text_match(text, options or {})


def parse_case(header_match, block):
    true_letter = header_match.group("true_letter")
    options = extract_options(block)

    # Committed answer precedence:
    #   1. Header "Predicted: X" — canonical, always written from
    #      interactive_system.letter_choice (matches the JSONL result row).
    #   2. "→ Committed to answer:" line (flat ScaleExpert convo format).
    #   3. Last "→ Final Answer:" line (branching format fallback; only used
    #      when the header letter was unparseable since branching emits one
    #      "→ Final Answer:" per LEAF and the canonical pick is a vote, not
    #      necessarily the last one).
    committed = None
    predicted_header = (header_match.group("predicted") or "").strip()
    if predicted_header in {"A", "B", "C", "D"}:
        committed = predicted_header
    else:
        committed_match = COMMITTED_RE.search(block)
        if committed_match:
            answer = committed_match.group("answer")
            committed = None if answer == "None" else answer
        else:
            final_matches = list(FINAL_ANSWER_RE.finditer(block))
            if final_matches:
                answer = final_matches[-1].group("answer")
                committed = None if answer == "None" else answer

    turns = []
    for turn_info, turn_block in iter_turn_blocks(block):
        shadow_text = extract_shadow_answer(turn_block)
        shadow_letter = parse_shadow_letter(shadow_text, options)
        turns.append(
            {
                "turn": turn_info["turn"],
                "is_final_decision_turn": turn_info["is_final"],
                "branch_id": turn_info["branch_id"],
                "shadow_answer_letter": shadow_letter,
                "shadow_answer_correct": (
                    shadow_letter == true_letter if shadow_letter is not None else None
                ),
                "shadow_answer_text": shadow_text,
            }
        )

    correct_turns = [
        turn["turn"] for turn in turns if turn["shadow_answer_correct"] is True
    ]
    final_correct = committed == true_letter if committed is not None else False

    return {
        "patient_id": int(header_match.group("id")),
        "header_correct_label": header_match.group("label"),
        "header_predicted": header_match.group("predicted").strip(),
        "true_letter": true_letter,
        "true_answer": header_match.group("true_answer").strip(),
        "final_committed_answer": committed,
        "final_correct": final_correct,
        "ever_correct": bool(correct_turns),
        "first_correct_turn": correct_turns[0] if correct_turns else None,
        "num_turns": len(turns),
        "num_parseable_shadow_answers": sum(
            turn["shadow_answer_letter"] is not None for turn in turns
        ),
        "turns": turns,
    }


def load_results(path):
    if not path:
        return {}
    result_path = Path(path)
    if not result_path.exists():
        return {}
    rows = {}
    for line in result_path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[int(row["id"])] = row
    return rows


def answer_is_cannot_answer(answer):
    return "i cannot answer this question" in (answer or "").lower()


def fact_body(fact):
    return re.sub(r"^\s*\d+\.\s*", "", fact or "").strip()


def extracted_fact_indices(answers, facts, max_questions=10):
    if not facts:
        return set()
    answer_text = "\n".join(answers[:max_questions])
    norm_answer = normalize_for_match(answer_text)
    found = set()
    for idx, fact in enumerate(facts):
        norm_full = normalize_for_match(fact)
        norm_body = normalize_for_match(fact_body(fact))
        if (norm_full and norm_full in norm_answer) or (norm_body and norm_body in norm_answer):
            found.add(idx)
    return found


def result_for_case(case, results_by_id):
    return results_by_id.get(case["patient_id"], results_by_id.get(case["patient_id"] - 1))


def patient_response_stats(cases, results_by_id, max_questions=10):
    groups = {
        "all_cases": cases,
        "final_correct": [case for case in cases if case["final_correct"]],
        "middle_correct_but_final_wrong": [
            case for case in cases if case["ever_correct"] and not case["final_correct"]
        ],
        "never_correct": [case for case in cases if not case["ever_correct"]],
    }
    stats = {}
    for name, group_cases in groups.items():
        total_answers = 0
        cannot_answers = 0
        total_facts = 0
        total_extracted_facts = 0
        per_case_cannot = []
        per_case_extracted_counts = []
        per_case_extracted_ratios = []
        question_counts = []
        cases_with_results = 0

        for case in group_cases:
            row = result_for_case(case, results_by_id)
            if not row:
                continue
            cases_with_results += 1
            answers = row["interactive_system"].get("answers", [])
            facts = row["info"].get("facts") or row["info"].get("context") or []
            capped_answers = answers[:max_questions]
            question_count = len(capped_answers)
            cannot_count = sum(
                answer_is_cannot_answer(answer) for answer in capped_answers
            )
            extracted = extracted_fact_indices(answers, facts, max_questions=max_questions)
            fact_count = len(facts)
            extracted_count = len(extracted)

            total_answers += question_count
            cannot_answers += cannot_count
            total_facts += fact_count
            total_extracted_facts += extracted_count
            per_case_cannot.append(cannot_count)
            per_case_extracted_counts.append(extracted_count)
            per_case_extracted_ratios.append(
                extracted_count / fact_count if fact_count else 0.0
            )
            question_counts.append(question_count)

        stats[name] = {
            "num_cases": len(group_cases),
            "num_cases_with_results": cases_with_results,
            "max_possible_patient_answers": len(group_cases) * max_questions,
            "patient_answers": total_answers,
            "cannot_answer_count": cannot_answers,
            "cannot_answer_proportion": cannot_answers / total_answers if total_answers else 0.0,
            "min_cannot_answers_per_case": min(per_case_cannot) if per_case_cannot else 0,
            "max_cannot_answers_per_case": max(per_case_cannot) if per_case_cannot else 0,
            "total_facts": total_facts,
            "total_extracted_facts": total_extracted_facts,
            "context_extracted_proportion": (
                total_extracted_facts / total_facts if total_facts else 0.0
            ),
            "min_extracted_facts_per_case": (
                min(per_case_extracted_counts) if per_case_extracted_counts else 0
            ),
            "max_extracted_facts_per_case": (
                max(per_case_extracted_counts) if per_case_extracted_counts else 0
            ),
            "min_extracted_proportion_per_case": (
                min(per_case_extracted_ratios) if per_case_extracted_ratios else 0.0
            ),
            "max_extracted_proportion_per_case": (
                max(per_case_extracted_ratios) if per_case_extracted_ratios else 0.0
            ),
            "avg_questions_considered": (
                sum(question_counts) / len(question_counts) if question_counts else 0.0
            ),
            "min_questions_considered": min(question_counts) if question_counts else 0,
            "max_questions_considered": max(question_counts) if question_counts else 0,
        }
    return stats


def summarize(cases, results_by_id=None):
    results_by_id = results_by_id or {}
    n = len(cases)
    final_parseable = [case for case in cases if case["final_committed_answer"] is not None]
    ever_correct = [case for case in cases if case["ever_correct"]]
    final_correct = [case for case in cases if case["final_correct"]]
    correct_then_wrong = [
        case
        for case in cases
        if case["ever_correct"] and case["final_correct"] is False
    ]
    never_correct = [case for case in cases if not case["ever_correct"]]
    first_turns = [case["first_correct_turn"] for case in ever_correct]
    total_shadow = sum(case["num_turns"] for case in cases)
    parseable_shadow = sum(case["num_parseable_shadow_answers"] for case in cases)
    never_parseable_shadow = [
        case for case in cases if case["num_parseable_shadow_answers"] == 0
    ]
    unparseable_shadow_turns = [
        {"patient_id": case["patient_id"], "turn": turn["turn"]}
        for case in cases
        for turn in case["turns"]
        if turn["shadow_answer_text"] is not None
        and turn["shadow_answer_letter"] is None
    ]

    return {
        "num_cases": n,
        "num_final_answers_parseable": len(final_parseable),
        "num_final_correct": len(final_correct),
        "final_accuracy": len(final_correct) / n if n else None,
        "num_ever_correct": len(ever_correct),
        "ever_correct_rate": len(ever_correct) / n if n else None,
        "num_correct_then_final_wrong": len(correct_then_wrong),
        "num_never_correct": len(never_correct),
        "avg_first_correct_turn": (
            sum(first_turns) / len(first_turns) if first_turns else None
        ),
        "num_shadow_answer_turns": total_shadow,
        "num_parseable_shadow_answers": parseable_shadow,
        "shadow_answer_parse_rate": parseable_shadow / total_shadow if total_shadow else None,
        "never_parseable_shadow_patient_ids": [
            case["patient_id"] for case in never_parseable_shadow
        ],
        "unparseable_shadow_turns": unparseable_shadow_turns,
        "correct_then_final_wrong_patient_ids": [
            case["patient_id"] for case in correct_then_wrong
        ],
        "never_correct_patient_ids": [case["patient_id"] for case in never_correct],
        "patient_response_stats": patient_response_stats(cases, results_by_id),
    }


def write_summary(summary, path):
    overall_turn_stats = summary["patient_response_stats"]["all_cases"]
    total_turns_line = (
        f"Total turns across all cases: {overall_turn_stats['patient_answers']} "
        f"(avg {overall_turn_stats['avg_questions_considered']:.2f}, "
        f"min {overall_turn_stats['min_questions_considered']}, "
        f"max {overall_turn_stats['max_questions_considered']})"
    )
    lines = [
        f"Cases parsed: {summary['num_cases']}",
        f"Final answers parsed: {summary['num_final_answers_parseable']}",
        f"Final correct: {summary['num_final_correct']}",
        f"Final accuracy: {summary['final_accuracy']:.4f}"
        if summary["final_accuracy"] is not None
        else "Final accuracy: n/a",
        f"Ever correct: {summary['num_ever_correct']}",
        f"Ever-correct rate: {summary['ever_correct_rate']:.4f}"
        if summary["ever_correct_rate"] is not None
        else "Ever-correct rate: n/a",
        f"Correct at some point, final wrong: {summary['num_correct_then_final_wrong']}",
        f"Never correct: {summary['num_never_correct']}",
        total_turns_line,
        f"Average first-correct turn: {summary['avg_first_correct_turn']:.2f}"
        if summary["avg_first_correct_turn"] is not None
        else "Average first-correct turn: n/a",
        f"Shadow answer parse rate: {summary['num_parseable_shadow_answers']}/"
        f"{summary['num_shadow_answer_turns']} = {summary['shadow_answer_parse_rate']:.4f}"
        if summary["shadow_answer_parse_rate"] is not None
        else "Shadow answer parse rate: n/a",
        "",
        "Never-parseable shadow-answer patient ids:",
        ", ".join(map(str, summary["never_parseable_shadow_patient_ids"])) or "(none)",
        "",
        "Unparseable shadow-answer turns:",
        ", ".join(
            f"{item['patient_id']}:{item['turn']}"
            for item in summary["unparseable_shadow_turns"]
        )
        or "(none)",
        "",
        "Correct at some point, final wrong patient ids:",
        ", ".join(map(str, summary["correct_then_final_wrong_patient_ids"])) or "(none)",
        "",
        "Never correct patient ids:",
        ", ".join(map(str, summary["never_correct_patient_ids"])) or "(none)",
        "",
        "Patient response/context extraction stats by trajectory group (first 10 questions):",
    ]
    group_labels = {
        "final_correct": "Final correct",
        "middle_correct_but_final_wrong": "Middle correct but wrong in the end",
        "never_correct": "Never correct",
    }
    for key, label in group_labels.items():
        stats = summary["patient_response_stats"][key]
        lines.extend(
            [
                f"{label}:",
                f"  Cases: {stats['num_cases']} ({stats['num_cases_with_results']} with result rows)",
                f"  Patient answers considered: {stats['patient_answers']} actual / {stats['max_possible_patient_answers']} max possible",
                f"  Total turns: {stats['patient_answers']}",
                f"  Questions per case: avg {stats['avg_questions_considered']:.2f}, min {stats['min_questions_considered']}, max {stats['max_questions_considered']}",
                f"  Cannot-answer responses: {stats['cannot_answer_count']}/{stats['patient_answers']} = {stats['cannot_answer_proportion']:.4f}",
                f"  Cannot-answer responses per case: min {stats['min_cannot_answers_per_case']}, max {stats['max_cannot_answers_per_case']}",
                f"  Context/facts extracted: {stats['total_extracted_facts']}/{stats['total_facts']} = {stats['context_extracted_proportion']:.4f}",
                f"  Extracted facts per case: min {stats['min_extracted_facts_per_case']}, max {stats['max_extracted_facts_per_case']}",
                f"  Extracted fact proportion per case: min {stats['min_extracted_proportion_per_case']:.4f}, max {stats['max_extracted_proportion_per_case']:.4f}",
            ]
        )
    stats = summary["patient_response_stats"]["all_cases"]
    lines.extend(
        [
            "",
            "Overall patient response/context extraction stats across all cases (first 10 questions):",
            f"  Cases: {stats['num_cases']} ({stats['num_cases_with_results']} with result rows)",
            f"  Patient answers considered: {stats['patient_answers']} actual / {stats['max_possible_patient_answers']} max possible",
            f"  Total turns: {stats['patient_answers']}",
            f"  Questions per case: avg {stats['avg_questions_considered']:.2f}, min {stats['min_questions_considered']}, max {stats['max_questions_considered']}",
            f"  Cannot-answer responses: {stats['cannot_answer_count']}/{stats['patient_answers']} = {stats['cannot_answer_proportion']:.4f}",
            f"  Cannot-answer responses per case: min {stats['min_cannot_answers_per_case']}, max {stats['max_cannot_answers_per_case']}",
            f"  Context/facts extracted: {stats['total_extracted_facts']}/{stats['total_facts']} = {stats['context_extracted_proportion']:.4f}",
            f"  Extracted facts per case: min {stats['min_extracted_facts_per_case']}, max {stats['max_extracted_facts_per_case']}",
            f"  Extracted fact proportion per case: min {stats['min_extracted_proportion_per_case']:.4f}, max {stats['max_extracted_proportion_per_case']:.4f}",
        ]
    )
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-turn shadow-answer correctness in a MediQ convo text log."
    )
    parser.add_argument(
        "--input",
        default="/home/hyang/mediQ/logs/scale_medgemma4b_yes_options_100q_convo.txt",
        help="Human-readable convo log produced by mediQ_benchmark.py.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="/home/hyang/mediQ/logs/scale_medgemma4b_yes_options_100q_answer_trajectory.jsonl",
        help="One JSON object per patient with answer trajectory details.",
    )
    parser.add_argument(
        "--summary",
        default="/home/hyang/mediQ/logs/scale_medgemma4b_yes_options_100q_answer_trajectory_summary.txt",
        help="Plain-text summary statistics.",
    )
    parser.add_argument(
        "--results-jsonl",
        default="/home/hyang/mediQ/results/scale_medgemma4b_yes_options_100q_results.jsonl",
        help="Benchmark result JSONL used for patient answers/context extraction stats.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary)

    text = input_path.read_text(errors="replace")
    cases = [parse_case(header, block) for header, block in split_patient_blocks(text)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    summary = summarize(cases, load_results(args.results_jsonl))
    write_summary(summary, summary_path)

    print(f"Wrote {len(cases)} patient records to {output_path}")
    print(f"Wrote summary to {summary_path}")
    print(
        "Final accuracy: "
        + (
            f"{summary['num_final_correct']}/{summary['num_cases']} "
            f"= {summary['final_accuracy']:.4f}"
            if summary["final_accuracy"] is not None
            else "n/a"
        )
    )
    print(
        "Ever-correct rate: "
        + (
            f"{summary['num_ever_correct']}/{summary['num_cases']} "
            f"= {summary['ever_correct_rate']:.4f}"
            if summary["ever_correct_rate"] is not None
            else "n/a"
        )
    )


if __name__ == "__main__":
    main()
