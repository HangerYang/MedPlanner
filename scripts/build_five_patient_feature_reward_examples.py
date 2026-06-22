#!/usr/bin/env python3
"""Build five readable, fully joined conversation examples."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path


IDENTITY_KEYS = {"patient_id", "branch_id", "turn_idx", "role", "reward"}


def select_feature_groups(path: Path, num_patients: int) -> OrderedDict:
    groups: OrderedDict[tuple[int, str], list[dict]] = OrderedDict()
    current_key = None
    current_rows: list[dict] = []
    selected_patients: set[int] = set()

    def consider(key, rows):
        if key is None or key[0] in selected_patients:
            return
        # At least one Q&A pair, but keep the examples compact.
        if 4 <= len(rows) <= 12:
            groups[key] = rows
            selected_patients.add(key[0])

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (int(row["patient_id"]), str(row["branch_id"]))
            if current_key is not None and key != current_key:
                consider(current_key, current_rows)
                if len(selected_patients) >= num_patients:
                    break
                current_rows = []
            current_key = key
            current_rows.append(row)
        if len(selected_patients) < num_patients:
            consider(current_key, current_rows)
    if len(groups) != num_patients:
        raise RuntimeError(f"Selected {len(groups)} patients, expected {num_patients}")
    return groups


def load_trajectories(path: Path, keys: set[tuple[int, str]]) -> dict:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (int(row["patient_id"]), str(row["branch_id"]))
            if key in keys and key not in rows:
                rows[key] = row
                if len(rows) == len(keys):
                    break
    return rows


def load_facts(path: Path, patient_ids: set[int]) -> dict[int, list[str]]:
    facts = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            patient_id = int(row["id"])
            if patient_id in patient_ids:
                facts[patient_id] = list(row.get("atomic_facts") or [])
                if len(facts) == len(patient_ids):
                    break
    return facts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--num-patients", type=int, default=5)
    args = parser.parse_args()

    groups = select_feature_groups(args.features, args.num_patients)
    trajectories = load_trajectories(args.trajectories, set(groups))
    facts = load_facts(args.facts, {key[0] for key in groups})

    examples = []
    for example_index, (key, feature_rows) in enumerate(groups.items(), start=1):
        patient_id, branch_id = key
        trajectory = trajectories[key]
        conversation = trajectory["conversation"]
        if len(conversation) != len(feature_rows):
            raise RuntimeError(
                f"{key}: conversation={len(conversation)} features={len(feature_rows)}"
            )

        steps = []
        for message, feature_row in zip(conversation, feature_rows):
            if (
                message["role"] != feature_row["role"]
                or feature_row["turn_idx"] != len(steps)
            ):
                raise RuntimeError(f"Step alignment mismatch for {key}")
            steps.append(
                {
                    "step_index": feature_row["turn_idx"],
                    "role": feature_row["role"],
                    "content": message["content"],
                    "reward": feature_row["reward"],
                    "conversation_features": {
                        name: value
                        for name, value in feature_row.items()
                        if name not in IDENTITY_KEYS
                    },
                }
            )

        message_count = len(conversation)
        examples.append(
            {
                "example_index": example_index,
                "patient_id": patient_id,
                "branch_id": branch_id,
                "embedding_source_row_index": trajectory["sample_id"],
                "question": trajectory["question"],
                "ground_truth_letter": trajectory["ground_truth_letter"],
                "final_answer_letter": trajectory["final_answer_letter"],
                "final_correct": trajectory["final_correct"],
                "atomic_facts": facts[patient_id],
                "num_atomic_facts": len(facts[patient_id]),
                "conversation": conversation,
                "total_turns": message_count,
                "num_qa_turns": trajectory["num_turns"],
                "embedding_count": message_count,
                "embedding_dimension": 2560,
                "steps": steps,
            }
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    lines = [
        "# Five Patient Conversation Examples",
        "",
        "Each conversation message has one 2560-dimensional embedding, one cumulative fact reward, and one 26-value conversation-feature snapshot.",
        "",
        "| Example | Patient | Branch | Total turns | Q&A turns | Embeddings | Facts | Final | Correct |",
        "|---:|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for example in examples:
        lines.append(
            f"| {example['example_index']} | {example['patient_id']} | {example['branch_id']} "
            f"| {example['total_turns']} | {example['num_qa_turns']} "
            f"| {example['embedding_count']} x {example['embedding_dimension']} "
            f"| {example['num_atomic_facts']} | {example['final_answer_letter']} "
            f"| {example['final_correct']} |"
        )
    for example in examples:
        lines.extend(
            [
                "",
                f"## Example {example['example_index']}: Patient {example['patient_id']}, Branch {example['branch_id']}",
                "",
                f"- Question: {example['question']}",
                f"- Total turns / embeddings: {example['total_turns']}",
                f"- Q&A turns: {example['num_qa_turns']}",
                f"- Atomic facts: {example['num_atomic_facts']}",
                "",
                "| Step | Role | Reward | Content |",
                "|---:|---|---:|---|",
            ]
        )
        for step in example["steps"]:
            content = step["content"].replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {step['step_index']} | {step['role']} | {step['reward']:.6f} | {content} |"
            )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(examples)} examples -> {args.output_jsonl}")
    print(f"Wrote readable summary -> {args.output_md}")


if __name__ == "__main__":
    main()
