#!/usr/bin/env python3
"""Compute SCOPE-Medical run metrics from convo.txt (+ optional results.jsonl for facts).

Intermediate accuracy: fraction of patients whose shadow answer was correct on at
least one turn (ever-correct), not per-turn micro accuracy.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Reuse convo parsers from the trajectory analysis script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_convo_answer_trajectory import (  # noqa: E402
    HEADER_RE,
    answer_is_cannot_answer,
    extract_options,
    extract_shadow_answer,
    iter_turn_blocks,
    load_results,
    parse_shadow_letter,
    split_patient_blocks,
    extracted_fact_indices,
)

PATIENT_RE = re.compile(
    r"^\s*Patient:\s*(?P<body>.*?)"
    r"(?=^\s*(?:--- Turn |→ Committed to answer:|→ Final Answer:|\Z))",
    re.MULTILINE | re.DOTALL,
)


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def extract_patient_answer(turn_block: str) -> str | None:
    match = PATIENT_RE.search(turn_block)
    if not match:
        return None
    return _collapse_ws(match.group("body"))


def infer_results_path(convo_path: Path) -> Path | None:
    name = convo_path.name
    if name.endswith("_convo.txt"):
        candidate = convo_path.with_name(name.replace("_convo.txt", "_results.jsonl"))
        if candidate.exists():
            return candidate
    parent = convo_path.parent
    matches = sorted(parent.glob("*_results.jsonl"))
    return matches[0] if len(matches) == 1 else None


def discover_run_folder(folder: Path) -> tuple[Path, Path]:
    """Return (convo.txt, results.jsonl) from a single run output directory."""
    folder = folder.resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder}")

    convo_files = sorted(folder.glob("*_convo.txt"))
    if not convo_files:
        raise FileNotFoundError(f"No *_convo.txt in {folder}")
    if len(convo_files) > 1:
        raise ValueError(f"Expected one *_convo.txt in {folder}, found {len(convo_files)}")

    convo_path = convo_files[0]
    results_path = infer_results_path(convo_path)
    if results_path is None:
        results_files = sorted(folder.glob("*_results.jsonl"))
        if len(results_files) == 1:
            results_path = results_files[0]
        else:
            raise FileNotFoundError(f"No matching *_results.jsonl in {folder}")
    return convo_path, results_path


def process_run_folder(
    folder: Path,
    *,
    max_questions: int = 10,
    temperature: float | None = None,
    confidence_threshold: float | None = None,
) -> dict:
    convo_path, results_path = discover_run_folder(folder)
    tag = folder.name
    out_json = folder / "metrics.json"
    metrics = process_convo(
        convo_path,
        results_jsonl=results_path,
        max_questions=max_questions,
        temperature=temperature,
        confidence_threshold=confidence_threshold,
        output_json=out_json,
    )
    metrics["tag"] = tag
    metrics["run_folder"] = str(folder)
    return metrics


def parse_convo_cases(convo_text: str) -> list[dict]:
    cases = []
    for header_match, block in split_patient_blocks(convo_text):
        patient_id = int(header_match.group("id"))
        true_letter = header_match.group("true_letter").strip()
        options = extract_options(block)
        predicted = (header_match.group("predicted") or "").strip()
        final_correct = predicted == true_letter if predicted in {"A", "B", "C", "D"} else False

        turns = []
        patient_answers: list[str] = []
        num_questions = 0

        for turn_info, turn_block in iter_turn_blocks(block):
            shadow_text = extract_shadow_answer(turn_block)
            shadow_letter = parse_shadow_letter(shadow_text, options)
            is_final = turn_info["is_final"]

            patient_answer = None
            if not is_final:
                patient_answer = extract_patient_answer(turn_block)
                if patient_answer is not None:
                    patient_answers.append(patient_answer)
                    num_questions += 1

            turns.append(
                {
                    "turn": turn_info["turn"],
                    "is_final": is_final,
                    "shadow_letter": shadow_letter,
                    "shadow_correct": (
                        shadow_letter == true_letter if shadow_letter is not None else None
                    ),
                    "patient_answer": patient_answer,
                }
            )

        cases.append(
            {
                "patient_id": patient_id,
                "true_letter": true_letter,
                "predicted": predicted,
                "final_correct": final_correct,
                "num_questions": num_questions,
                "patient_answers": patient_answers,
                "turns": turns,
            }
        )
    return cases


def _final_correct_from_results(row: dict) -> bool | None:
    isys = row.get("interactive_system") or {}
    if "correct" in isys:
        return bool(isys["correct"])
    letter = isys.get("letter_choice")
    gold = (row.get("info") or {}).get("correct_answer_idx")
    if letter and gold:
        return letter == gold
    return None


def supplement_cases_from_results(
    cases: list[dict],
    results_by_id: dict[int, dict],
) -> tuple[list[dict], list[int]]:
    """Add stub cases from results.jsonl when convo.txt is missing patients."""
    present = {c["patient_id"] for c in cases}
    added_ids: list[int] = []
    supplemented = list(cases)
    for pid, row in sorted(results_by_id.items()):
        if pid in present:
            continue
        final_correct = _final_correct_from_results(row)
        if final_correct is None:
            continue
        isys = row.get("interactive_system") or {}
        supplemented.append(
            {
                "patient_id": pid,
                "true_letter": (row.get("info") or {}).get("correct_answer_idx", ""),
                "predicted": isys.get("letter_choice", ""),
                "final_correct": final_correct,
                "num_questions": int(isys.get("num_questions") or 0),
                "patient_answers": list(isys.get("answers") or []),
                "turns": [],
                "from_results_only": True,
            }
        )
        added_ids.append(pid)
    return supplemented, added_ids


def compute_metrics(
    cases: list[dict],
    results_by_id: dict[int, dict],
    *,
    max_questions: int = 10,
    temperature: float | None = None,
    confidence_threshold: float | None = None,
) -> dict:
    convo_n = len(cases)
    if convo_n == 0 and not results_by_id:
        raise ValueError("No patient cases parsed from convo or results file.")

    cases, added_from_results = supplement_cases_from_results(cases, results_by_id)
    n = len(cases)

    # 1. Final accuracy (all patients with a known final label)
    final_correct = sum(1 for c in cases if c["final_correct"])

    # 2. Null answer ratio (patient idk / all patient answers to doctor questions; convo only)
    total_patient_answers = 0
    cannot_answer_count = 0
    for case in cases:
        if case.get("from_results_only"):
            continue
        for answer in case["patient_answers"]:
            total_patient_answers += 1
            if answer_is_cannot_answer(answer):
                cannot_answer_count += 1
    null_answer_ratio = (
        cannot_answer_count / total_patient_answers if total_patient_answers else 0.0
    )

    # 3. Avg questions per patient (convo-parsed cases only)
    convo_cases = [c for c in cases if not c.get("from_results_only")]
    avg_questions = (
        sum(c["num_questions"] for c in convo_cases) / len(convo_cases) if convo_cases else 0.0
    )

    # 4. Avg facts extracted (% of atomic fact pool), unique only
    fact_ratios: list[float] = []
    fact_counts: list[int] = []
    fact_pool_sizes: list[int] = []
    missing_facts = 0
    for case in cases:
        if case.get("from_results_only"):
            continue
        row = results_by_id.get(case["patient_id"])
        if not row:
            missing_facts += 1
            continue
        facts = row.get("info", {}).get("facts") or []
        pool_size = len(facts)
        if pool_size == 0:
            continue
        extracted = extracted_fact_indices(case["patient_answers"], facts, max_questions=max_questions)
        fact_counts.append(len(extracted))
        fact_pool_sizes.append(pool_size)
        fact_ratios.append(len(extracted) / pool_size)

    avg_facts_pct = sum(fact_ratios) / len(fact_ratios) if fact_ratios else 0.0
    avg_facts_count = sum(fact_counts) / len(fact_counts) if fact_counts else 0.0

    # 5. Intermediate accuracy: per patient, count as correct if shadow was ever right.
    def case_ever_shadow_correct(case: dict) -> bool:
        if case.get("from_results_only"):
            row = results_by_id.get(case["patient_id"])
            if not row:
                return False
            gold = (row.get("info") or {}).get("correct_answer_idx")
            choices = (row.get("interactive_system") or {}).get("intermediate_choices") or []
            return any(choice == gold for choice in choices if choice)
        return any(turn["shadow_correct"] is True for turn in case["turns"])

    ever_correct_count = sum(1 for c in cases if case_ever_shadow_correct(c))
    intermediate_acc = ever_correct_count / n if n else 0.0

    # 6–7. Max questions reached & max-question accuracy
    max_reached_cases = [c for c in cases if c["num_questions"] >= max_questions]
    max_questions_reached_rate = len(max_reached_cases) / n
    max_question_acc = (
        sum(1 for c in max_reached_cases if c["final_correct"]) / len(max_reached_cases)
        if max_reached_cases
        else 0.0
    )

    # 8. Middle accuracy (confident early stop: final answer before hitting cap)
    middle_cases = [c for c in cases if c["num_questions"] < max_questions]
    middle_acc = (
        sum(1 for c in middle_cases if c["final_correct"]) / len(middle_cases)
        if middle_cases
        else 0.0
    )

    return {
        "tag": None,
        "n_patients": n,
        "n_patients_convo": convo_n,
        "patients_added_from_results": added_from_results,
        "max_questions": max_questions,
        "temperature": temperature,
        "confidence_threshold": confidence_threshold,
        "accuracy": final_correct / n,
        "accuracy_n": f"{final_correct}/{n}",
        "null_answer_ratio": null_answer_ratio,
        "null_answer_n": f"{cannot_answer_count}/{total_patient_answers}",
        "avg_questions": avg_questions,
        "avg_facts_pct_of_pool": avg_facts_pct,
        "avg_facts_extracted_count": avg_facts_count,
        "avg_facts_pool_size": (
            sum(fact_pool_sizes) / len(fact_pool_sizes) if fact_pool_sizes else 0.0
        ),
        "facts_cases_with_results": len(fact_ratios),
        "facts_cases_missing_results": missing_facts,
        "intermediate_acc": intermediate_acc,
        "intermediate_acc_n": f"{ever_correct_count}/{n}",
        "intermediate_acc_definition": "per_patient_ever_correct_shadow",
        "max_questions_reached_rate": max_questions_reached_rate,
        "max_questions_reached_n": f"{len(max_reached_cases)}/{n}",
        "max_question_acc": max_question_acc,
        "max_question_acc_n": (
            f"{sum(1 for c in max_reached_cases if c['final_correct'])}/{len(max_reached_cases)}"
            if max_reached_cases
            else "0/0"
        ),
        "middle_acc": middle_acc,
        "middle_acc_n": (
            f"{sum(1 for c in middle_cases if c['final_correct'])}/{len(middle_cases)}"
            if middle_cases
            else "0/0"
        ),
        "middle_cases": len(middle_cases),
        "max_reached_cases": len(max_reached_cases),
    }


def _fmt_metric(value, fmt: str = ".4f") -> str:
    if value is None:
        return "n/a"
    if fmt == ".2f":
        return f"{value:.2f}"
    return f"{value:{fmt}}"


def format_table_row(metrics: dict, tag: str) -> str:
    return "\t".join(
        [
            tag,
            _fmt_metric(metrics.get("null_answer_ratio")),
            _fmt_metric(metrics.get("avg_questions"), ".2f"),
            _fmt_metric(metrics.get("avg_facts_pct_of_pool")),
            _fmt_metric(metrics.get("intermediate_acc")),
            str(metrics.get("temperature", "")),
            str(metrics.get("confidence_threshold", "")),
            _fmt_metric(metrics.get("max_questions_reached_rate")),
            _fmt_metric(metrics.get("max_question_acc")),
            _fmt_metric(metrics.get("middle_acc")),
            _fmt_metric(metrics.get("accuracy")),
        ]
    )


EVAL_LOG_OUTPUT_RE = re.compile(r"^Output:\s*(.+)$", re.MULTILINE)
EVAL_LOG_ROWS_RE = re.compile(r"^Rows:\s*(\d+)\s*$", re.MULTILINE)
EVAL_LOG_CORRECT_RE = re.compile(r"^Correct:\s*(\d+)/(\d+)\s*=\s*([\d.]+)", re.MULTILINE)
EVAL_LOG_PARSED_RE = re.compile(r"^Parsed:\s*(\d+)/(\d+)", re.MULTILINE)


def parse_eval_log(log_path: Path) -> dict:
    text = log_path.read_text(errors="replace")
    output_match = EVAL_LOG_OUTPUT_RE.search(text)
    correct_match = EVAL_LOG_CORRECT_RE.search(text)
    parsed_match = EVAL_LOG_PARSED_RE.search(text)
    rows_match = EVAL_LOG_ROWS_RE.search(text)
    return {
        "output_jsonl": output_match.group(1).strip() if output_match else None,
        "rows": int(rows_match.group(1)) if rows_match else None,
        "parsed_n": int(parsed_match.group(1)) if parsed_match else None,
        "parsed_total": int(parsed_match.group(2)) if parsed_match else None,
        "correct_n": int(correct_match.group(1)) if correct_match else None,
        "correct_total": int(correct_match.group(2)) if correct_match else None,
        "accuracy": float(correct_match.group(3)) if correct_match else None,
    }


def facts_pct_from_context_segments(context: list, facts: list) -> float | None:
    if not facts:
        return None
    segments = context[1:] if len(context) > 1 else []
    if not segments:
        return 0.0
    extracted = extracted_fact_indices(segments, facts, max_questions=max(len(segments), 1))
    return len(extracted) / len(facts)


def _use_scope_context_for_facts(source: str) -> bool:
    name = Path(source).name.lower()
    if "initial_context" in name:
        return False
    return any(token in name for token in ("facts", "replay", "eval_facts", "condensed"))


def compute_oneshot_metrics(
    eval_rows: list[dict],
    results_by_id: dict[int, dict] | None = None,
    *,
    tag: str,
    source: str,
    log_summary: dict | None = None,
    use_scope_context_for_facts: bool | None = None,
) -> dict:
    """Metrics for non-interactive one-shot vLLM eval (no doctor/patient turns)."""
    n = len(eval_rows)
    correct = sum(1 for row in eval_rows if row.get("correct"))
    accuracy = correct / n if n else 0.0
    if log_summary and log_summary.get("accuracy") is not None and n == 0:
        accuracy = log_summary["accuracy"]
        correct = log_summary.get("correct_n", 0)
        n = log_summary.get("correct_total", 0)

    if use_scope_context_for_facts is None:
        use_scope_context_for_facts = _use_scope_context_for_facts(source)

    fact_ratios: list[float] = []
    for row in eval_rows:
        pid = int(row["id"])
        scope_row = results_by_id.get(pid) if results_by_id else None
        if use_scope_context_for_facts and scope_row:
            info = scope_row.get("info") or {}
            facts = info.get("facts") or []
            context = info.get("context") or []
        else:
            facts = (scope_row.get("info") or {}).get("facts") or [] if scope_row else []
            context = row.get("context") or []
        ratio = facts_pct_from_context_segments(context, facts)
        if ratio is not None:
            fact_ratios.append(ratio)

    return {
        "tag": tag,
        "run_mode": "oneshot",
        "source": source,
        "n_patients": n or (log_summary or {}).get("correct_total"),
        "max_questions": 0,
        "temperature": None,
        "confidence_threshold": None,
        "accuracy": accuracy if n else (log_summary or {}).get("accuracy"),
        "accuracy_n": f"{correct}/{n}" if n else (
            f"{log_summary.get('correct_n')}/{log_summary.get('correct_total')}"
            if log_summary and log_summary.get("correct_n") is not None
            else "n/a"
        ),
        "null_answer_ratio": None,
        "null_answer_n": "n/a",
        "avg_questions": 0.0,
        "avg_facts_pct_of_pool": (
            sum(fact_ratios) / len(fact_ratios) if fact_ratios else None
        ),
        "avg_facts_note": (
            "facts_present_in_prompt / atomic_fact_pool (no interactive extraction)"
            if fact_ratios
            else None
        ),
        "intermediate_acc": None,
        "intermediate_acc_n": "n/a",
        "max_questions_reached_rate": None,
        "max_questions_reached_n": "n/a",
        "max_question_acc": None,
        "max_question_acc_n": "n/a",
        "middle_acc": None,
        "middle_acc_n": "n/a",
        "eval_output_jsonl": (log_summary or {}).get("output_jsonl"),
        "parsed_rate": (
            log_summary["parsed_n"] / log_summary["parsed_total"]
            if log_summary and log_summary.get("parsed_total")
            else None
        ),
    }


def process_convo(
    convo_path: Path,
    *,
    results_jsonl: Path | None = None,
    max_questions: int = 10,
    temperature: float | None = None,
    confidence_threshold: float | None = None,
    output_json: Path | None = None,
) -> dict:
    convo_text = convo_path.read_text(errors="replace")
    cases = parse_convo_cases(convo_text)
    results_path = results_jsonl or infer_results_path(convo_path)
    results_by_id = load_results(str(results_path)) if results_path and results_path.exists() else {}

    tag = convo_path.name.replace("_convo.txt", "")
    metrics = compute_metrics(
        cases,
        results_by_id,
        max_questions=max_questions,
        temperature=temperature,
        confidence_threshold=confidence_threshold,
    )
    metrics["tag"] = tag
    metrics["run_mode"] = "interactive"
    metrics["source"] = str(convo_path)
    metrics["convo"] = str(convo_path)
    metrics["results_jsonl"] = str(results_path) if results_path else None

    out_json = output_json or convo_path.with_suffix(".metrics.json")
    out_json.write_text(json.dumps(metrics, indent=2) + "\n")
    metrics["_output_json"] = str(out_json)
    return metrics


def process_eval_log(
    log_path: Path,
    *,
    scope_results_jsonl: Path | None = None,
    output_json: Path | None = None,
) -> dict:
    summary = parse_eval_log(log_path)
    eval_path = Path(summary["output_jsonl"]) if summary.get("output_jsonl") else None
    eval_rows = []
    if eval_path and eval_path.exists():
        eval_rows = [json.loads(line) for line in eval_path.read_text(errors="replace").splitlines() if line.strip()]

    results_path = scope_results_jsonl
    if not results_path or not results_path.exists():
        dummy = log_path.parent / "eval.jsonl"
        results_path = infer_scope_results_for_eval(dummy)
    results_by_id = load_results(str(results_path)) if results_path and results_path.exists() else {}

    tag = log_path.stem.replace("_eval", "").replace("_run", "")
    metrics = compute_oneshot_metrics(
        eval_rows,
        results_by_id,
        tag=tag,
        source=str(log_path),
        log_summary=summary,
    )
    if eval_rows and results_by_id and metrics.get("avg_facts_pct_of_pool") is None:
        # Build fact coverage from scope results for ids in eval subset.
        ratios = []
        eval_ids = {int(r["id"]) for r in eval_rows}
        for pid in eval_ids:
            row = results_by_id.get(pid)
            if not row:
                continue
            facts = (row.get("info") or {}).get("facts") or []
            context = (row.get("info") or {}).get("context") or []
            ratio = facts_pct_from_context_segments(context, facts)
            if ratio is not None:
                ratios.append(ratio)
        if ratios:
            metrics["avg_facts_pct_of_pool"] = sum(ratios) / len(ratios)
            metrics["avg_facts_note"] = (
                "unique scope context facts / atomic_fact_pool (eval jsonl missing; used scope results)"
            )

    use_scope_facts = _use_scope_context_for_facts(str(log_path))
    if metrics.get("avg_facts_pct_of_pool") is None and results_by_id and use_scope_facts:
        ratios = []
        limit = summary.get("rows") or len(results_by_id)
        for pid, row in sorted(results_by_id.items()):
            if len(ratios) >= limit:
                break
            facts = (row.get("info") or {}).get("facts") or []
            context = (row.get("info") or {}).get("context") or []
            ratio = facts_pct_from_context_segments(context, facts)
            if ratio is not None:
                ratios.append(ratio)
        if ratios:
            metrics["avg_facts_pct_of_pool"] = sum(ratios) / len(ratios)
            metrics["avg_facts_note"] = (
                "unique scope context facts / atomic_fact_pool"
                + (" (eval output jsonl missing)" if not eval_rows else "")
            )

    metrics["scope_results_jsonl"] = str(results_path) if results_path else None
    out_json = output_json or log_path.with_suffix(".metrics.json")
    out_json.write_text(json.dumps(metrics, indent=2) + "\n")
    metrics["_output_json"] = str(out_json)
    return metrics


def infer_scope_results_for_eval(eval_path: Path) -> Path | None:
    parent = eval_path.parent
    name = eval_path.name.lower()
    if "condensed" in name:
        preferred = ("condensed_qwen3_4b_no_reasoning_results.jsonl",)
    else:
        preferred = (
            "medical_scope_qwen3_4b_scope_decode_top175_results.jsonl",
            "medical_scope_qwen3_4b_results.jsonl",
        )
    for pattern in preferred + (
        "condensed_qwen3_4b_no_reasoning_results.jsonl",
        "medical_scope_qwen3_4b_results.jsonl",
    ):
        candidate = parent / pattern
        if candidate.exists():
            return candidate
    for candidate in sorted(parent.glob("*_results.jsonl")):
        if "test_eval" not in candidate.name:
            return candidate
    return None


def process_eval_jsonl(
    eval_path: Path,
    *,
    scope_results_jsonl: Path | None = None,
    output_json: Path | None = None,
) -> dict:
    eval_rows = [json.loads(line) for line in eval_path.read_text(errors="replace").splitlines() if line.strip()]
    results_path = infer_scope_results_for_eval(eval_path) or scope_results_jsonl
    results_by_id = load_results(str(results_path)) if results_path and results_path.exists() else {}
    tag = eval_path.stem
    metrics = compute_oneshot_metrics(
        eval_rows,
        results_by_id,
        tag=tag,
        source=str(eval_path),
    )
    metrics["scope_results_jsonl"] = str(results_path) if results_path else None
    out_json = output_json or eval_path.with_suffix(".metrics.json")
    out_json.write_text(json.dumps(metrics, indent=2) + "\n")
    metrics["_output_json"] = str(out_json)
    return metrics


def process_path(path: Path, **kwargs) -> dict | None:
    path = path.resolve()
    if path.is_dir():
        return None
    if path.suffix == ".log" and "eval" in path.name:
        return process_eval_log(path, **kwargs)
    if path.name.endswith("_convo.txt"):
        return process_convo(path, **kwargs)
    if path.suffix == ".jsonl" and "test_eval" in path.name:
        return process_eval_jsonl(path, **kwargs)
    return None


def collect_run_folders(paths: list[Path]) -> list[Path]:
    """Expand inputs into run folders (dirs containing *_convo.txt + *_results.jsonl)."""
    folders: list[Path] = []
    for path in paths:
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            if list(path.glob("*_convo.txt")):
                folders.append(path)
            else:
                for child in sorted(path.iterdir()):
                    if child.is_dir() and list(child.glob("*_convo.txt")):
                        folders.append(child)
        else:
            raise ValueError(f"Expected a run folder, got file: {path}")
    return folders


def collect_targets(paths: list[Path]) -> list[Path]:
    targets: list[Path] = []
    for path in paths:
        path = path.resolve()
        if path.is_dir():
            for pattern in ("*_convo.txt", "*eval*.log", "test_eval_*.jsonl"):
                targets.extend(sorted(path.glob(pattern)))
        else:
            targets.append(path)
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Run folder(s) with *_convo.txt + *_results.jsonl, or a parent directory",
    )
    parser.add_argument(
        "--convo",
        default=None,
        help="(Legacy) Path to a single *_convo.txt file",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Legacy mode: scan dirs for loose convo/log/jsonl files",
    )
    parser.add_argument(
        "--results-jsonl",
        default=None,
        help="Path to matching *_results.jsonl (default: infer from convo path)",
    )
    parser.add_argument(
        "--scope-results-jsonl",
        default=None,
        help="Scope results JSONL for fact pools in one-shot eval logs",
    )
    parser.add_argument("--max-questions", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument(
        "--output-json",
        default=None,
        help="Write full metrics JSON (only when a single convo input is given)",
    )
    args = parser.parse_args()

    if not args.inputs and not args.convo:
        parser.error("Provide one or more run folders (or a parent directory).")

    header = (
        "tag\tnull_answer_ratio\tavg_questions\tavg_facts_pct\tintermediate_acc\t"
        "temperature\tconfidence\tmax_q_reached\tmax_q_acc\tmiddle_acc\taccuracy"
    )
    print(header)

    all_metrics: list[dict] = []

    if args.legacy or args.convo:
        paths: list[Path] = []
        if args.convo:
            paths.append(Path(args.convo))
        paths.extend(Path(p) for p in args.inputs)
        scope_results = Path(args.scope_results_jsonl) if args.scope_results_jsonl else None
        kwargs = {
            "results_jsonl": Path(args.results_jsonl) if args.results_jsonl else None,
            "scope_results_jsonl": scope_results,
            "max_questions": args.max_questions,
            "temperature": args.temperature,
            "confidence_threshold": args.confidence_threshold,
            "output_json": Path(args.output_json) if args.output_json else None,
        }
        for target in collect_targets(paths):
            if target.name.endswith("_convo.txt"):
                metrics = process_convo(target, **{k: v for k, v in kwargs.items() if k != "scope_results_jsonl"})
            elif target.suffix == ".log":
                metrics = process_eval_log(target, scope_results_jsonl=scope_results)
            elif target.suffix == ".jsonl":
                metrics = process_eval_jsonl(target, scope_results_jsonl=scope_results)
            else:
                continue
            all_metrics.append(metrics)
            print(format_table_row(metrics, metrics["tag"]))
    else:
        run_folders = collect_run_folders([Path(p) for p in args.inputs])
        for folder in run_folders:
            metrics = process_run_folder(
                folder,
                max_questions=args.max_questions,
                temperature=args.temperature,
                confidence_threshold=args.confidence_threshold,
            )
            all_metrics.append(metrics)
            print(format_table_row(metrics, metrics["tag"]))

        if len(run_folders) > 1:
            parent = run_folders[0].parent
            summary_path = parent / "metrics_summary.json"
            summary_path.write_text(json.dumps(all_metrics, indent=2) + "\n")
            print(f"\nWrote {summary_path}")

    if len(all_metrics) == 1:
        print(f"\nWrote {all_metrics[0].get('_output_json')}")


if __name__ == "__main__":
    main()
