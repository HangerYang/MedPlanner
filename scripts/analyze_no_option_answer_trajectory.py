#!/usr/bin/env python3
import argparse
import difflib
import json
import re
from pathlib import Path

from analyze_convo_answer_trajectory import (
    COMMITTED_RE,
    HEADER_RE,
    TURN_RE,
    extract_options,
    split_patient_blocks,
)


LABELS = (
    "Initial",
    "Question",
    "Options",
    "Confidence",
    "Confidence Rationale",
    "Shadow Answer",
    "Boxed Answer",
    "Doctor Question",
    "Patient",
)


def iter_turn_blocks(block):
    matches = list(TURN_RE.finditer(block))
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        yield match, block[start:end]


def clean_wrapped_text(text):
    text = re.sub(r"(\w)-\s+(\w)", r"\1-\2", text or "")
    text = re.sub(r"<[^>]*>", " ", text)
    text = text.replace("```", " ").replace("\\box", " ").replace("\\boxed", " ")
    text = re.sub(r"[{}$*_`\"']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_for_ratio(text):
    text = clean_wrapped_text(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_label(block, label):
    pattern = rf"^\s*{re.escape(label)}:(?P<inline>.*)$"
    match = re.search(pattern, block, re.MULTILINE)
    if not match:
        return None

    inline = match.group("inline").strip()
    rest = block[match.end() :]
    stop_labels = "|".join(re.escape(item) for item in LABELS if item != label)
    stop = re.search(rf"^\s*(?:{stop_labels}|--- Turn|→ Judgment|→ Committed)\b", rest, re.MULTILINE)
    body = rest[: stop.start()] if stop else rest
    text = inline
    if body.strip():
        text = f"{text}\n{body.strip()}".strip() if text else body.strip()
    return text.strip() or None


def conclusion_span(text):
    text = clean_wrapped_text(text)
    if not text:
        return None

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
    lower = text.lower()
    positions = [lower.rfind(marker) for marker in markers]
    pos = max(positions)
    if pos >= 0:
        return text[pos:]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    tail = " ".join(sentence for sentence in sentences[-2:] if sentence).strip()
    return tail or text[-700:]


def answer_span(turn_block):
    boxed = extract_label(turn_block, "Boxed Answer")
    if boxed:
        return clean_wrapped_text(boxed), "boxed_answer"

    shadow = extract_label(turn_block, "Shadow Answer")
    if shadow:
        return conclusion_span(shadow), "shadow_conclusion"

    return None, "missing"


def fuzzy_match_to_options(answer, options):
    norm_answer = normalize_for_ratio(answer)
    scores = []
    for letter, option_text in options.items():
        norm_option = normalize_for_ratio(option_text)
        score = difflib.SequenceMatcher(None, norm_answer, norm_option).ratio() if norm_answer and norm_option else 0.0
        scores.append(
            {
                "letter": letter,
                "option_text": option_text,
                "score": score,
            }
        )
    scores.sort(key=lambda item: item["score"], reverse=True)
    best = scores[0] if scores else None
    second = scores[1] if len(scores) > 1 else {"score": 0.0}
    margin = (best["score"] - second["score"]) if best else 0.0
    return {
        "matched_letter": best["letter"] if best else None,
        "matched_option_text": best["option_text"] if best else None,
        "score": best["score"] if best else None,
        "second_score": second["score"],
        "margin": margin,
        "accepted": bool(best and best["score"] >= 0.45 and margin >= 0.03),
        "scores": scores,
    }


def parse_reference_cases(yes_text):
    refs = {}
    for header, block in split_patient_blocks(yes_text):
        pid = int(header.group("id"))
        refs[pid] = {
            "initial": extract_label(block, "Initial"),
            "question": extract_label(block, "Question"),
            "options": extract_options(block),
            "true_letter": header.group("true_letter"),
            "true_answer": header.group("true_answer").strip(),
        }
    return refs


def similarity(a, b):
    return difflib.SequenceMatcher(None, normalize_for_ratio(a), normalize_for_ratio(b)).ratio()


def parse_no_option_case(header, block, ref):
    pid = int(header.group("id"))
    true_letter = ref["true_letter"] if ref else header.group("true_letter")
    options = ref["options"] if ref else {}

    initial = extract_label(block, "Initial")
    question = extract_label(block, "Question")
    validation = {
        "initial_similarity": similarity(initial, ref["initial"]) if ref else None,
        "question_similarity": similarity(question, ref["question"]) if ref else None,
        "options_found": bool(options),
    }
    validation["warning"] = bool(
        not ref
        or validation["initial_similarity"] < 0.85
        or validation["question_similarity"] < 0.85
        or not options
    )

    turns = []
    for turn_match, turn_block in iter_turn_blocks(block):
        span, source = answer_span(turn_block)
        match = fuzzy_match_to_options(span, options) if span and options else {
            "matched_letter": None,
            "matched_option_text": None,
            "score": None,
            "second_score": None,
            "margin": None,
            "accepted": False,
            "scores": [],
        }
        letter = match["matched_letter"]
        turns.append(
            {
                "turn": int(turn_match.group("turn")),
                "is_final_decision_turn": bool(turn_match.group("final")),
                "answer_source": source,
                "answer_span": span,
                "matched_letter": letter,
                "matched_option_text": match["matched_option_text"],
                "match_score": match["score"],
                "second_score": match["second_score"],
                "margin": match["margin"],
                "accepted": match["accepted"],
                "correct": letter == true_letter if letter else None,
                "scores": match["scores"],
            }
        )

    final_turn = next((turn for turn in reversed(turns) if turn["is_final_decision_turn"]), turns[-1] if turns else None)
    final_letter = final_turn["matched_letter"] if final_turn else None
    committed_match = COMMITTED_RE.search(block)
    committed_answer = None
    if committed_match:
        raw_committed = committed_match.group("answer")
        committed_answer = None if raw_committed == "None" else raw_committed
    correct_turns = [turn["turn"] for turn in turns if turn["correct"] is True]
    return {
        "patient_id": pid,
        "true_letter": true_letter,
        "true_answer": ref["true_answer"] if ref else header.group("true_answer").strip(),
        "original_header_label": header.group("label"),
        "original_header_predicted": header.group("predicted").strip(),
        "validation": validation,
        "final_matched_letter": final_letter,
        "final_matched_option_text": final_turn["matched_option_text"] if final_turn else None,
        "final_match_score": final_turn["match_score"] if final_turn else None,
        "final_margin": final_turn["margin"] if final_turn else None,
        "final_match_accepted": final_turn["accepted"] if final_turn else False,
        "final_correct": final_letter == true_letter if final_letter else False,
        "final_committed_answer": committed_answer,
        "final_committed_correct": committed_answer == true_letter if committed_answer else None,
        "ever_correct": bool(correct_turns),
        "first_correct_turn": correct_turns[0] if correct_turns else None,
        "num_turns": len(turns),
        "num_matched_turns": sum(turn["matched_letter"] is not None for turn in turns),
        "turns": turns,
    }


def summarize(cases):
    n = len(cases)
    final_correct = [case for case in cases if case["final_correct"]]
    committed_known = [case for case in cases if case["final_committed_answer"] is not None]
    committed_correct = [case for case in committed_known if case["final_committed_correct"]]
    ever_correct = [case for case in cases if case["ever_correct"]]
    validation_warnings = [case for case in cases if case["validation"]["warning"]]
    low_conf_final = [case for case in cases if not case["final_match_accepted"]]
    unmatched_turns = [
        {"patient_id": case["patient_id"], "turn": turn["turn"]}
        for case in cases
        for turn in case["turns"]
        if turn["matched_letter"] is None
    ]
    return {
        "num_cases": n,
        "num_final_correct": len(final_correct),
        "final_accuracy": len(final_correct) / n if n else None,
        "num_final_committed_parseable": len(committed_known),
        "num_final_committed_correct": len(committed_correct),
        "final_committed_accuracy": len(committed_correct) / len(committed_known) if committed_known else None,
        "num_ever_correct": len(ever_correct),
        "ever_correct_rate": len(ever_correct) / n if n else None,
        "num_validation_warnings": len(validation_warnings),
        "validation_warning_patient_ids": [case["patient_id"] for case in validation_warnings],
        "num_low_confidence_final_matches": len(low_conf_final),
        "low_confidence_final_patient_ids": [case["patient_id"] for case in low_conf_final],
        "num_unmatched_turns": len(unmatched_turns),
        "unmatched_turns": unmatched_turns,
    }


def write_summary(summary, path):
    lines = [
        f"Cases parsed: {summary['num_cases']}",
        f"Final correct: {summary['num_final_correct']}",
        f"Final accuracy: {summary['final_accuracy']:.4f}" if summary["final_accuracy"] is not None else "Final accuracy: n/a",
        f"Final committed answers parsed: {summary['num_final_committed_parseable']}",
        f"Final committed correct: {summary['num_final_committed_correct']}",
        f"Final committed accuracy: {summary['final_committed_accuracy']:.4f}" if summary["final_committed_accuracy"] is not None else "Final committed accuracy: n/a",
        f"Ever correct: {summary['num_ever_correct']}",
        f"Ever-correct rate: {summary['ever_correct_rate']:.4f}" if summary["ever_correct_rate"] is not None else "Ever-correct rate: n/a",
        f"Validation warnings: {summary['num_validation_warnings']}",
        ", ".join(map(str, summary["validation_warning_patient_ids"])) or "(none)",
        "",
        f"Low-confidence final matches: {summary['num_low_confidence_final_matches']}",
        ", ".join(map(str, summary["low_confidence_final_patient_ids"])) or "(none)",
        "",
        f"Unmatched turns: {summary['num_unmatched_turns']}",
        ", ".join(f"{item['patient_id']}:{item['turn']}" for item in summary["unmatched_turns"]) or "(none)",
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Analyze no-option MediQ answers by fuzzy matching to yes-option options.")
    parser.add_argument("--no-option-log", default="/home/hyang/mediQ/logs/scale_rg_medgemma4b_convo_no_options.txt")
    parser.add_argument("--yes-option-log", default="/home/hyang/mediQ/logs/scale_medgemma4b_yes_options_100q_convo.txt")
    parser.add_argument("--output-jsonl", default="/home/hyang/mediQ/logs/scale_rg_medgemma4b_no_options_answer_trajectory.jsonl")
    parser.add_argument("--summary", default="/home/hyang/mediQ/logs/scale_rg_medgemma4b_no_options_answer_trajectory_summary.txt")
    args = parser.parse_args()

    no_text = Path(args.no_option_log).read_text(errors="replace")
    yes_text = Path(args.yes_option_log).read_text(errors="replace")
    refs = parse_reference_cases(yes_text)

    cases = []
    for header, block in split_patient_blocks(no_text):
        pid = int(header.group("id"))
        cases.append(parse_no_option_case(header, block, refs.get(pid)))

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    summary = summarize(cases)
    write_summary(summary, Path(args.summary))
    print(f"Wrote {len(cases)} patient records to {output_path}")
    print(f"Wrote summary to {args.summary}")
    print(f"Final accuracy: {summary['num_final_correct']}/{summary['num_cases']} = {summary['final_accuracy']:.4f}")
    if summary["final_committed_accuracy"] is not None:
        print(
            f"Final committed accuracy: {summary['num_final_committed_correct']}/"
            f"{summary['num_final_committed_parseable']} = {summary['final_committed_accuracy']:.4f}"
        )
    print(f"Ever-correct rate: {summary['num_ever_correct']}/{summary['num_cases']} = {summary['ever_correct_rate']:.4f}")


if __name__ == "__main__":
    main()
