#!/usr/bin/env python3
"""Check per-conversation source, retained-embedding, and feature row counts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_scope_dataset_from_mediQ_convo import DOCTOR_BRANCH_RE, TURN_HEADER_RE


RAW_PATIENT_RE = re.compile(r"^Patient #(?P<id>\d+)\s+\|", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doctor-view", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--bad-indices", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    return parser.parse_args()


def source_counts(path: Path) -> dict[tuple[int, str], int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    headers = list(RAW_PATIENT_RE.finditer(text))
    counts: dict[tuple[int, str], int] = {}
    for idx, header in enumerate(headers):
        patient_id = int(header.group("id"))
        end = headers[idx + 1].start() if idx + 1 < len(headers) else len(text)
        block = text[header.start():end]
        branches = list(DOCTOR_BRANCH_RE.finditer(block))
        for branch_idx, branch in enumerate(branches):
            branch_end = (
                branches[branch_idx + 1].start()
                if branch_idx + 1 < len(branches)
                else len(block)
            )
            branch_block = block[branch.end():branch_end]
            key = (patient_id, branch.group("branch_id"))
            if key in counts:
                raise RuntimeError(f"Duplicate source conversation key: {key}")
            counts[key] = 2 * len(TURN_HEADER_RE.findall(branch_block)) + 2
    return counts


def load_bad_indices(path: Path) -> set[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["embedding_row_index"]) for row in csv.DictReader(handle)}


def embedding_counts(
    path: Path, bad_indices: set[int]
) -> tuple[Counter, dict[tuple[int, str], set[int]], int]:
    counts: Counter = Counter()
    turns: dict[tuple[int, str], set[int]] = defaultdict(set)
    physical_index = 0
    skipped = 0
    for arrow_path in sorted(path.glob("data-*.arrow")):
        table = ipc.open_stream(pa.memory_map(str(arrow_path), "r")).read_all()
        patients = table.column("patient_id").to_pylist()
        branches = table.column("branch_id").to_pylist()
        turn_indices = table.column("turn_idx").to_pylist()
        for patient_id, branch_id, turn_idx in zip(patients, branches, turn_indices):
            if physical_index in bad_indices:
                skipped += 1
            else:
                key = (patient_id, branch_id)
                counts[key] += 1
                turns[key].add(turn_idx)
            physical_index += 1
    if skipped != len(bad_indices):
        raise RuntimeError(f"Skipped {skipped} rows, expected {len(bad_indices)}")
    return counts, turns, physical_index


def feature_counts(
    path: Path,
) -> tuple[Counter, dict[tuple[int, str], set[int]], int, int]:
    counts: Counter = Counter()
    turns: dict[tuple[int, str], set[int]] = defaultdict(set)
    number_of_turn_mismatches = 0
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["patient_id"], row["branch_id"])
            counts[key] += 1
            turns[key].add(row["turn_idx"])
            number_of_turn_mismatches += (
                float(row["Number of Turns"]) != row["turn_idx"] + 1
            )
            rows += 1
    return counts, turns, rows, number_of_turn_mismatches


def contiguous_failures(
    counts: Counter, turns: dict[tuple[int, str], set[int]]
) -> list[tuple[int, str]]:
    return [
        key
        for key, count in counts.items()
        if turns[key] != set(range(count))
    ]


def main() -> None:
    args = parse_args()
    expected = source_counts(args.doctor_view)
    bad_indices = load_bad_indices(args.bad_indices)
    embeddings, embedding_turns, raw_embedding_rows = embedding_counts(
        args.embeddings, bad_indices
    )
    features, feature_turns, feature_rows, number_of_turn_mismatches = feature_counts(
        args.features
    )

    all_keys = set(expected) | set(embeddings) | set(features)
    mismatches = [
        {
            "patient_id": key[0],
            "branch_id": key[1],
            "source": expected.get(key, 0),
            "embeddings": embeddings.get(key, 0),
            "features": features.get(key, 0),
        }
        for key in sorted(all_keys)
        if not (
            expected.get(key, 0)
            == embeddings.get(key, 0)
            == features.get(key, 0)
        )
    ]
    embedding_noncontiguous = contiguous_failures(embeddings, embedding_turns)
    feature_noncontiguous = contiguous_failures(features, feature_turns)

    result = {
        "source_conversations": len(expected),
        "retained_embedding_conversations": len(embeddings),
        "feature_conversations": len(features),
        "source_expected_rows": sum(expected.values()),
        "raw_embedding_rows": raw_embedding_rows,
        "excluded_corrupt_embedding_rows": len(bad_indices),
        "retained_embedding_rows": sum(embeddings.values()),
        "feature_rows": feature_rows,
        "conversation_count_mismatches": len(mismatches),
        "embedding_noncontiguous_turn_sequences": len(embedding_noncontiguous),
        "feature_noncontiguous_turn_sequences": len(feature_noncontiguous),
        "feature_number_of_turn_mismatches": number_of_turn_mismatches,
        "first_count_mismatches": mismatches[:20],
    }
    print(json.dumps(result, indent=2))
    if (
        mismatches
        or embedding_noncontiguous
        or feature_noncontiguous
        or number_of_turn_mismatches
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
