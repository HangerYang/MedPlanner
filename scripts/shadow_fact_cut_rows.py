#!/usr/bin/env python3
"""Build one-shot eval rows with min/max patient-fact cuts from a SCOPE convo log.

Uses patient answer lines (option A), not reassembled atomic_fact pool indices.

Cut rules (shadow evaluated before each turn's patient reply):
  min — facts from turns with turn_number < first_shadow_correct_turn;
        if never shadow-correct, full useful facts from the run.
  max — if ever shadow-correct and final wrong: turns < last_shadow_correct_turn;
        else full useful facts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_convo_answer_trajectory import (  # noqa: E402
    extract_options,
    extract_shadow_answer,
    iter_turn_blocks,
    parse_shadow_letter,
    split_patient_blocks,
)
from compute_scope_medical_metrics import discover_run_folder, extract_patient_answer  # noqa: E402

import expert_functions  # noqa: E402


PATIENT_RE = re.compile(
    r"^\s*Patient:\s*(?P<body>.*?)"
    r"(?=^\s*(?:--- Turn |→ Committed to answer:|→ Final Answer:|\Z))",
    re.MULTILINE | re.DOTALL,
)


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def parse_case_shadow_turns(header_match, block: str) -> dict:
    true_letter = header_match.group("true_letter").strip()
    options = extract_options(block)
    predicted = (header_match.group("predicted") or "").strip()
    final_correct = predicted == true_letter if predicted in {"A", "B", "C", "D"} else False

    turn_answers: list[tuple[int, str]] = []
    shadow_correct_turns: list[int] = []

    for turn_info, turn_block in iter_turn_blocks(block):
        turn_num = int(turn_info["turn"])
        shadow_text = extract_shadow_answer(turn_block)
        shadow_letter = parse_shadow_letter(shadow_text, options)
        if shadow_letter == true_letter:
            shadow_correct_turns.append(turn_num)

        if not turn_info["is_final"]:
            answer = extract_patient_answer(turn_block)
            if answer is not None:
                turn_answers.append((turn_num, answer))

    first_correct = min(shadow_correct_turns) if shadow_correct_turns else None
    last_correct = max(shadow_correct_turns) if shadow_correct_turns else None

    return {
        "patient_id": int(header_match.group("id")),
        "final_correct": final_correct,
        "ever_shadow_correct": bool(shadow_correct_turns),
        "first_shadow_correct_turn": first_correct,
        "last_shadow_correct_turn": last_correct,
        "num_turns": sum(1 for _ in iter_turn_blocks(block)),
        "turn_answers": turn_answers,
    }


def compute_turn_exclusive_bound(cut_mode: str, case: dict) -> int | None:
    """Return max turn_number to include (exclusive upper bound on turn ids), or None = all."""
    if cut_mode == "min":
        first = case["first_shadow_correct_turn"]
        return first if first is not None else None

    if cut_mode == "max":
        if not case["ever_shadow_correct"]:
            return None
        if case["final_correct"]:
            return None
        return case["last_shadow_correct_turn"]

    raise ValueError(f"Unknown cut_mode: {cut_mode!r}")


def build_patient_information(
    initial_info: str,
    turn_answers: list[tuple[int, str]],
    *,
    exclusive_turn_bound: int | None,
) -> str:
    if exclusive_turn_bound is None:
        selected = turn_answers
    else:
        selected = [(t, a) for t, a in turn_answers if t < exclusive_turn_bound]

    history = [{"question": "", "answer": answer} for _, answer in selected]
    patient_state = {
        "initial_info": initial_info or "",
        "interaction_history": history,
    }
    return str(expert_functions.condensed_patient_state(patient_state)["initial_info"])


def build_eval_rows(
    convo_path: Path,
    results_path: Path,
    *,
    cut_mode: str,
    only_shadow_correct_final_wrong: bool = False,
    limit: int | None = None,
) -> list[dict]:
    results_by_id = {}
    for line in results_path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        results_by_id[int(row["id"])] = row
        if limit and len(results_by_id) >= limit:
            break

    convo_text = convo_path.read_text(errors="replace")
    rows = []
    for header_match, block in split_patient_blocks(convo_text):
        case = parse_case_shadow_turns(header_match, block)
        if only_shadow_correct_final_wrong and (
            not case["ever_shadow_correct"] or case["final_correct"]
        ):
            continue

        pid = case["patient_id"]
        scope_row = results_by_id.get(pid)
        if scope_row is None:
            continue

        info = scope_row.get("info") or {}
        bound = compute_turn_exclusive_bound(cut_mode, case)
        patient_information = build_patient_information(
            str(info.get("initial_info") or ""),
            case["turn_answers"],
            exclusive_turn_bound=bound,
        )

        interactive = scope_row.get("interactive_system") or {}
        rows.append(
            {
                "id": pid,
                "question": info.get("question"),
                "options": info.get("options"),
                "answer_idx": info.get("correct_answer_idx"),
                "answer": info.get("correct_answer"),
                "patient_information": patient_information,
                "context_mode": f"shadow_fact_{cut_mode}",
                "fact_cut_mode": cut_mode,
                "exclusive_turn_bound": bound,
                "first_shadow_correct_turn": case["first_shadow_correct_turn"],
                "last_shadow_correct_turn": case["last_shadow_correct_turn"],
                "ever_shadow_correct": case["ever_shadow_correct"],
                "final_correct_interactive": case["final_correct"],
                "num_turn_answers": len(case["turn_answers"]),
                "scope_letter_choice": interactive.get("letter_choice"),
                "scope_correct": interactive.get("correct"),
                "subset": (
                    "shadow_correct_final_wrong"
                    if only_shadow_correct_final_wrong
                    else "all"
                ),
            }
        )
        if limit and len(rows) >= limit:
            break

    rows.sort(key=lambda r: r["id"])
    return rows


def write_rows_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-folder", type=Path, help="SCOPE output folder with convo + results")
    parser.add_argument("--convo", type=Path, default=None)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument(
        "--cut-mode",
        choices=("min", "max"),
        required=True,
        help="min=earliest shadow-correct; max=last shadow-correct if final wrong else full",
    )
    parser.add_argument("--output", type=Path, required=True, help="Eval rows JSONL for one_shot eval")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument(
        "--only-shadow-correct-final-wrong",
        action="store_true",
        help="Keep only cases where an intermediate shadow answer was correct but the final answer was wrong.",
    )
    args = parser.parse_args()

    if args.run_folder:
        convo_path, results_path = discover_run_folder(args.run_folder)
    else:
        if not args.convo or not args.results:
            raise SystemExit("Provide --run-folder or both --convo and --results")
        convo_path = args.convo
        results_path = args.results
        if not convo_path.exists():
            raise FileNotFoundError(convo_path)
        if not results_path.exists():
            raise FileNotFoundError(results_path)

    limit = args.max_examples or None
    rows = build_eval_rows(
        convo_path,
        results_path,
        cut_mode=args.cut_mode,
        only_shadow_correct_final_wrong=args.only_shadow_correct_final_wrong,
        limit=limit,
    )
    write_rows_jsonl(rows, args.output)
    subset = (
        "shadow_correct_final_wrong"
        if args.only_shadow_correct_final_wrong
        else "all"
    )
    print(
        f"Wrote {len(rows)} rows -> {args.output} "
        f"(cut_mode={args.cut_mode}, subset={subset})"
    )


if __name__ == "__main__":
    main()
