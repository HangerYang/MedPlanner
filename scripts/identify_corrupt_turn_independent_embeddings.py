#!/usr/bin/env python3
"""Identify embedding rows corrupted by missed multiline patient headers.

The original parser only recognizes patient headers whose true-answer text is
contained on one line. When a header is missed, that patient's branches are
parsed as part of the previous recognized patient. This script reproduces the
original trajectory ordering, marks branches belonging to missed headers, and
maps them to row ranges in the two-GPU turn-independent embedding output.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_branch_convo import HEADER_RE
from build_scope_dataset_from_mediQ_convo import DOCTOR_BRANCH_RE, TURN_HEADER_RE


RAW_PATIENT_RE = re.compile(r"^Patient #(?P<id>\d+)\s+\|", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doctor-view", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-bad-indices", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.doctor_view.read_text(encoding="utf-8", errors="replace")
    raw_headers = list(RAW_PATIENT_RE.finditer(text))
    recognized_starts = {match.start(): match for match in HEADER_RE.finditer(text)}

    trajectory_index = 0
    trajectories: list[dict] = []
    current_assigned_patient: int | None = None

    for raw_idx, raw_header in enumerate(raw_headers):
        source_patient = int(raw_header.group("id"))
        raw_end = (
            raw_headers[raw_idx + 1].start()
            if raw_idx + 1 < len(raw_headers)
            else len(text)
        )
        block = text[raw_header.start():raw_end]

        if raw_header.start() in recognized_starts:
            current_assigned_patient = source_patient
        if current_assigned_patient is None:
            raise RuntimeError("First patient header was not recognized")

        corrupt = source_patient != current_assigned_patient
        for branch in DOCTOR_BRANCH_RE.finditer(block):
            branch_start = branch.end()
            next_branch = DOCTOR_BRANCH_RE.search(block, branch_start)
            branch_end = next_branch.start() if next_branch else len(block)
            branch_block = block[branch_start:branch_end]
            num_turns = len(TURN_HEADER_RE.findall(branch_block))
            message_count = 2 * num_turns + 2
            trajectories.append(
                {
                    "trajectory_index": trajectory_index,
                    "assigned_patient_id": current_assigned_patient,
                    "source_patient_id": source_patient,
                    "branch_id": branch.group("branch_id"),
                    "num_turns": num_turns,
                    "message_count": message_count,
                    "corrupt": corrupt,
                }
            )
            trajectory_index += 1

    # embed_turns_independent.py sends even trajectory indices to GPU 0 and odd
    # indices to GPU 1, then concatenates GPU 0 followed by GPU 1.
    parity_totals = {
        parity: sum(t["message_count"] for t in trajectories if t["trajectory_index"] % 2 == parity)
        for parity in (0, 1)
    }
    parity_offsets = {0: 0, 1: parity_totals[0]}
    parity_positions = {0: 0, 1: 0}
    corrupt_rows: list[dict] = []
    bad_indices: list[int] = []

    for trajectory in trajectories:
        parity = trajectory["trajectory_index"] % 2
        row_start = parity_offsets[parity] + parity_positions[parity]
        row_end = row_start + trajectory["message_count"]
        parity_positions[parity] += trajectory["message_count"]
        if not trajectory["corrupt"]:
            continue
        output = {
            **{k: v for k, v in trajectory.items() if k != "corrupt"},
            "embedding_row_start": row_start,
            "embedding_row_end_exclusive": row_end,
        }
        corrupt_rows.append(output)
        bad_indices.extend(range(row_start, row_end))

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in corrupt_rows:
            handle.write(json.dumps(row) + "\n")

    with args.output_bad_indices.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["embedding_row_index"])
        writer.writerows([index] for index in bad_indices)

    missing_patients = sorted({row["source_patient_id"] for row in corrupt_rows})
    assigned_patients = sorted({row["assigned_patient_id"] for row in corrupt_rows})
    summary = {
        "total_trajectories": len(trajectories),
        "total_embedding_rows": sum(t["message_count"] for t in trajectories),
        "corrupt_source_patients": len(missing_patients),
        "corrupt_assigned_patient_ids": len(assigned_patients),
        "corrupt_trajectories": len(corrupt_rows),
        "corrupt_embedding_rows": len(bad_indices),
        "valid_embedding_rows": sum(t["message_count"] for t in trajectories) - len(bad_indices),
        "missing_source_patient_ids": missing_patients,
        "assigned_patient_ids_to_filter_wholesale": assigned_patients,
    }
    args.output_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
