#!/usr/bin/env python3
"""Independently verify corrected conversation features against raw source."""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_scope_dataset_from_mediQ_convo import DOCTOR_BRANCH_RE, TURN_HEADER_RE
from conversation_feature import IncrementalFeatureState


RAW_PATIENT_RE = re.compile(r"^Patient #(?P<id>\d+)\s+\|", re.MULTILINE)


class ArrowRows:
    def __init__(self, dataset_path: Path) -> None:
        self.tables = []
        self.ends = []
        total = 0
        for path in sorted(dataset_path.glob("data-*.arrow")):
            table = ipc.open_stream(pa.memory_map(str(path), "r")).read_all()
            self.tables.append(table)
            total += table.num_rows
            self.ends.append(total)
        self.total = total

    def row(self, index: int) -> dict:
        table_idx = bisect.bisect_right(self.ends, index)
        start = 0 if table_idx == 0 else self.ends[table_idx - 1]
        table = self.tables[table_idx]
        offset = index - start
        return {
            "patient_id": table.column("patient_id")[offset].as_py(),
            "branch_id": table.column("branch_id")[offset].as_py(),
            "turn_idx": table.column("turn_idx")[offset].as_py(),
            "role": table.column("role")[offset].as_py(),
            "embedding": table.column("embedding")[offset].as_py(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doctor-view", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    return parser.parse_args()


def expected_clean_keys(doctor_view: Path) -> set[tuple[int, str, int]]:
    text = doctor_view.read_text(encoding="utf-8", errors="replace")
    headers = list(RAW_PATIENT_RE.finditer(text))
    keys: set[tuple[int, str, int]] = set()
    duplicate_keys = 0
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
            messages = 2 * len(TURN_HEADER_RE.findall(branch_block)) + 2
            for turn_idx in range(messages):
                key = (patient_id, branch.group("branch_id"), turn_idx)
                duplicate_keys += key in keys
                keys.add(key)
    if duplicate_keys:
        raise RuntimeError(f"Raw source unexpectedly contains {duplicate_keys} duplicate keys")
    return keys


def main() -> None:
    args = parse_args()
    expected = expected_clean_keys(args.doctor_view)

    actual_rows: dict[tuple[int, str, int], dict] = {}
    actual_duplicates = 0
    for line in args.features.open(encoding="utf-8"):
        row = json.loads(line)
        key = (row["patient_id"], row["branch_id"], row["turn_idx"])
        actual_duplicates += key in actual_rows
        actual_rows[key] = row

    actual = set(actual_rows)
    missing = expected - actual
    unexpected = actual - expected

    # Find physical embeddings matching valid source keys. Corrupt rows reuse
    # predecessor keys; retain the first occurrence for direct spot checks.
    arrows = ArrowRows(args.embeddings)
    physical_by_key: dict[tuple[int, str, int], int] = {}
    for physical_index in range(arrows.total):
        row = arrows.row(physical_index)
        key = (row["patient_id"], row["branch_id"], row["turn_idx"])
        physical_by_key.setdefault(key, physical_index)

    samples = [
        (0, "root"),
        (139, "1-1-1"),
        (3607, "1-1-1"),
        (3607, "2"),
        (7999, "2-2-2"),
    ]
    max_abs_feature_diff = 0.0
    checked_rows = 0
    sample_results = []
    for patient_id, branch_id in samples:
        keys = sorted(
            (key for key in actual if key[:2] == (patient_id, branch_id)),
            key=lambda key: key[2],
        )
        state = IncrementalFeatureState()
        sample_max = 0.0
        for key in keys:
            embedding = arrows.row(physical_by_key[key])
            role = "user" if embedding["role"] == "user" else "agent"
            recomputed = state.add_turn(
                torch.tensor(embedding["embedding"], dtype=torch.float32), role
            )
            stored = actual_rows[key]
            for name, value in recomputed.items():
                diff = abs(float(stored[name]) - float(value))
                sample_max = max(sample_max, diff)
                max_abs_feature_diff = max(max_abs_feature_diff, diff)
            checked_rows += 1
        sample_results.append(
            {
                "patient_id": patient_id,
                "branch_id": branch_id,
                "rows": len(keys),
                "max_abs_feature_diff": sample_max,
            }
        )

    result = {
        "expected_clean_rows_from_raw_source": len(expected),
        "actual_corrected_feature_rows": len(actual),
        "actual_duplicate_keys": actual_duplicates,
        "missing_expected_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "sample_feature_rows_recomputed": checked_rows,
        "sample_max_abs_feature_diff": max_abs_feature_diff,
        "sample_results": sample_results,
    }
    print(json.dumps(result, indent=2))
    if actual_duplicates or missing or unexpected or max_abs_feature_diff > 1e-6:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
