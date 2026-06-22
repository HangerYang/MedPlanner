#!/usr/bin/env python3
"""Annotate SCOPE trajectories with per-turn rewards.

For each leaf trajectory produced by ``build_scope_dataset_from_mediQ_convo.py``:

  - Each (doctor question -> patient answer) turn gets an ``info_gain`` reward:
      info_gain = (# atomic facts first revealed in this answer) / total_facts
  - The LAST turn's reward is overridden with the correctness signal:
      final_reward = 1.0 if final_committed_answer == ground_truth else 0.0
  - The ``reward`` field per turn is the value to optimize directly:
      reward[i]   = info_gain[i]   for non-final turns
      reward[-1]  = final_reward

Inputs (all derived automatically from the convo log if you only pass ``--convo``):
  - <prefix>_scope_trajectories.jsonl   (text conversations + qa_turns)
  - <prefix>_results.jsonl              (per-patient atomic facts)

Outputs:
  - <prefix>_scope_reward_trajectories.jsonl   (full per-turn record)
  - <prefix>_scope_reward_hf/                  (HF dataset for SCOPE embed step)
  - <prefix>_scope_reward_summary.txt          (label/reward stats)
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

from analyze_convo_answer_trajectory import (  # noqa: E402
    fact_body,
    normalize_for_match,
)


def detect_fact_indices(answer_text: str, facts: list[str]) -> set[int]:
    """Return indices of facts revealed by this single patient answer."""
    if not answer_text or not facts:
        return set()
    norm_answer = normalize_for_match(answer_text)
    if not norm_answer:
        return set()
    hits: set[int] = set()
    for idx, fact in enumerate(facts):
        norm_full = normalize_for_match(fact)
        norm_body = normalize_for_match(fact_body(fact))
        if (norm_full and norm_full in norm_answer) or (
            norm_body and norm_body in norm_answer
        ):
            hits.add(idx)
    return hits


def annotate_trajectory(traj: dict, facts: list[str]) -> dict:
    total_facts = len(facts)
    qa_turns = traj.get("qa_turns") or []
    final_correct = bool(traj.get("final_correct"))
    final_reward = 1.0 if final_correct else 0.0

    seen: set[int] = set()
    annotated_turns: list[dict] = []
    info_gains: list[float] = []
    for i, turn in enumerate(qa_turns):
        is_final = i == len(qa_turns) - 1
        revealed = detect_fact_indices(turn.get("answer") or "", facts)
        new_indices = sorted(revealed - seen)
        seen |= revealed
        info_gain = (len(new_indices) / total_facts) if total_facts else 0.0
        reward = final_reward if is_final else info_gain
        annotated_turns.append(
            {
                "turn_index": turn.get("turn_index", i + 1),
                "doctor_question": turn.get("question", ""),
                "patient_answer": turn.get("answer", ""),
                "cannot_answer": bool(turn.get("cannot_answer")),
                "new_fact_indices": new_indices,
                "num_new_facts": len(new_indices),
                "cumulative_facts_revealed": len(seen),
                "info_gain": info_gain,
                "is_final_turn": is_final,
                "correctness_reward": final_reward if is_final else None,
                "reward": reward,
            }
        )
        info_gains.append(info_gain)

    if not annotated_turns:
        # Zero-turn trajectory (model committed at root). Emit a single
        # placeholder turn so reward consumers always have one signal slot.
        annotated_turns.append(
            {
                "turn_index": 0,
                "doctor_question": None,
                "patient_answer": None,
                "cannot_answer": False,
                "new_fact_indices": [],
                "num_new_facts": 0,
                "cumulative_facts_revealed": 0,
                "info_gain": 0.0,
                "is_final_turn": True,
                "correctness_reward": final_reward,
                "reward": final_reward,
            }
        )
        info_gains.append(0.0)

    return {
        **{k: v for k, v in traj.items() if k != "qa_turns"},
        "num_facts_total": total_facts,
        "facts": facts,
        "final_reward": final_reward,
        "turn_rewards": [t["reward"] for t in annotated_turns],
        "info_gain_per_turn": info_gains,
        "cumulative_information_gain": sum(info_gains),
        "facts_revealed_total": len(seen),
        "facts_revealed_proportion": (len(seen) / total_facts) if total_facts else 0.0,
        "turns": annotated_turns,
    }


def load_facts_index(path: Path) -> dict[int, list[str]]:
    facts: dict[int, list[str]] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            info = row.get("info") or {}
            patient_facts = info.get("facts") or info.get("atomic_facts") or []
            facts[int(row["id"])] = list(patient_facts)
    return facts


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def save_hf_dataset(rows: list[dict], path: Path) -> None:
    import datasets

    hf_rows = {
        "patient_id": [r["patient_id"] for r in rows],
        "branch_id": [r["branch_id"] for r in rows],
        "conversation": [r["conversation"] for r in rows],
        # turn_rewards[t] = info_gain for t < last_turn; correctness for last turn (legacy format).
        # Use info_gain_per_turn for pure per-turn information gain without the correctness override.
        "turn_rewards": [r["turn_rewards"] for r in rows],
        "info_gain_per_turn": [r["info_gain_per_turn"] for r in rows],
        "final_reward": [r["final_reward"] for r in rows],
        "ground_truth_letter": [r["ground_truth_letter"] for r in rows],
        "final_answer_letter": [r["final_answer_letter"] for r in rows],
        "final_correct": [r["final_correct"] for r in rows],
        "num_turns": [r["num_turns"] for r in rows],
        "num_facts_total": [r["num_facts_total"] for r in rows],
        "facts_revealed_total": [r["facts_revealed_total"] for r in rows],
        "cumulative_information_gain": [r["cumulative_information_gain"] for r in rows],
    }
    ds = datasets.Dataset.from_dict(hf_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(path))


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    final_correct = sum(r["final_correct"] for r in rows)
    avg_total_facts = sum(r["num_facts_total"] for r in rows) / n if n else 0.0
    avg_revealed = sum(r["facts_revealed_total"] for r in rows) / n if n else 0.0
    avg_cig = sum(r["cumulative_information_gain"] for r in rows) / n if n else 0.0
    info_gains = [g for r in rows for g in r["info_gain_per_turn"]]
    return {
        "num_trajectories": n,
        "num_final_correct": final_correct,
        "final_accuracy": (final_correct / n) if n else None,
        "avg_total_facts_per_case": avg_total_facts,
        "avg_facts_revealed_per_case": avg_revealed,
        "avg_cumulative_information_gain": avg_cig,
        "num_turns_total": len(info_gains),
        "avg_info_gain_per_turn": (sum(info_gains) / len(info_gains))
        if info_gains
        else 0.0,
        "max_info_gain_per_turn": max(info_gains) if info_gains else 0.0,
        "num_zero_info_turns": sum(1 for g in info_gains if g == 0.0),
    }


def write_summary(summary: dict, path: Path) -> None:
    lines = [
        f"Trajectories: {summary['num_trajectories']}",
        f"Final correct: {summary['num_final_correct']} "
        + (
            f"(acc={summary['final_accuracy']:.4f})"
            if summary["final_accuracy"] is not None
            else "(acc=n/a)"
        ),
        f"Avg total facts / case:        {summary['avg_total_facts_per_case']:.2f}",
        f"Avg facts revealed / case:     {summary['avg_facts_revealed_per_case']:.2f}",
        f"Avg cumulative info gain:      {summary['avg_cumulative_information_gain']:.4f}",
        f"Total Q&A turns:               {summary['num_turns_total']}",
        f"Avg info gain per turn:        {summary['avg_info_gain_per_turn']:.4f}",
        f"Max info gain per turn:        {summary['max_info_gain_per_turn']:.4f}",
        f"Zero-info turns:               {summary['num_zero_info_turns']} / {summary['num_turns_total']}",
        "",
        "Reward layout per trajectory:",
        "  turn_rewards[i]  = (new facts revealed at turn i) / total_facts",
        "  turn_rewards[-1] = 1.0 if final answer correct else 0.0   (overrides info_gain at last turn)",
        "  final_reward     = correctness reward (same as turn_rewards[-1])",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_paths_from_convo(convo_path: Path) -> dict[str, Path]:
    stem = convo_path.name
    if stem.endswith("_convo.txt"):
        prefix = stem[: -len("_convo.txt")]
    elif stem.endswith("_scope_trajectories.jsonl"):
        prefix = stem[: -len("_scope_trajectories.jsonl")]
    else:
        prefix = convo_path.stem
    parent = convo_path.parent
    return {
        "trajectories": parent / f"{prefix}_scope_trajectories.jsonl",
        "results": parent / f"{prefix}_results.jsonl",
        "output_jsonl": parent / f"{prefix}_scope_reward_trajectories.jsonl",
        "output_hf": parent / f"{prefix}_scope_reward_hf",
        "summary": parent / f"{prefix}_scope_reward_summary.txt",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate SCOPE trajectories with per-turn information-gain rewards "
            "(plus a 0/1 correctness reward at the final turn)."
        )
    )
    parser.add_argument(
        "--convo",
        type=Path,
        required=True,
        help="Path to *_convo.txt (or *_scope_trajectories.jsonl). Only used for "
        "default output paths.",
    )
    parser.add_argument("--trajectories", type=Path, default=None)
    parser.add_argument("--results-jsonl", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--output-hf", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = default_paths_from_convo(args.convo)
    traj_path = args.trajectories or defaults["trajectories"]
    results_path = args.results_jsonl or defaults["results"]
    output_jsonl = args.output_jsonl or defaults["output_jsonl"]
    output_hf = args.output_hf or defaults["output_hf"]
    summary_path = args.summary or defaults["summary"]

    if not traj_path.exists():
        raise FileNotFoundError(
            f"Trajectories JSONL not found: {traj_path}\n"
            "Run scripts/build_scope_dataset_from_mediQ_convo.py first."
        )
    if not results_path.exists():
        raise FileNotFoundError(
            f"Results JSONL not found: {results_path}\n"
            "Atomic facts are read from <patient>['info']['facts']."
        )

    facts_index = load_facts_index(results_path)
    annotated: list[dict] = []
    missing_facts = 0
    for traj in iter_jsonl(traj_path):
        facts = facts_index.get(int(traj["patient_id"]), [])
        if not facts:
            missing_facts += 1
        annotated.append(annotate_trajectory(traj, facts))

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in annotated:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    save_hf_dataset(annotated, output_hf)
    summary = summarize(annotated)
    summary["trajectories_missing_facts"] = missing_facts
    write_summary(summary, summary_path)

    print(f"Wrote {len(annotated)} reward-annotated trajectories to {output_jsonl}")
    print(f"Wrote HuggingFace dataset to {output_hf}")
    print(f"Wrote summary to {summary_path}")
    if summary["final_accuracy"] is not None:
        print(
            f"Final accuracy: {summary['num_final_correct']}/{summary['num_trajectories']} "
            f"= {summary['final_accuracy']:.4f}"
        )
    if missing_facts:
        print(f"WARNING: {missing_facts} trajectories had no atomic_facts in results.jsonl.")


if __name__ == "__main__":
    main()
