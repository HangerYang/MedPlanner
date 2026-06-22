#!/usr/bin/env python3
"""Remove corrupt embedding rows and repair affected conversation features.

Unaffected feature rows are copied unchanged. For patients that received
branches from a missed multiline header, valid rows are recomputed from the
existing embeddings after corrupt rows are removed.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from conversation_feature import IncrementalFeatureState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--bad-indices", type=Path, required=True)
    parser.add_argument("--manifest-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


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


def load_bad_indices(path: Path) -> set[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["embedding_row_index"]) for row in csv.DictReader(handle)}


def load_sort_indices(embeddings: Path) -> list[int]:
    cache_paths = sorted(embeddings.glob("cache-*.arrow"))
    if len(cache_paths) != 1:
        raise RuntimeError(
            f"Expected exactly one sorted-index cache in {embeddings}, found {cache_paths}"
        )
    table = ipc.open_stream(pa.memory_map(str(cache_paths[0]), "r")).read_all()
    if table.column_names != ["indices"]:
        raise RuntimeError(f"Unexpected cache schema: {table.schema}")
    return table.column("indices").to_pylist()


def main() -> None:
    args = parse_args()
    bad_indices = load_bad_indices(args.bad_indices)
    summary = json.loads(args.manifest_summary.read_text(encoding="utf-8"))
    affected_patients = set(summary["assigned_patient_ids_to_filter_wholesale"])
    sort_indices = load_sort_indices(args.embeddings)
    arrow_rows = ArrowRows(args.embeddings)

    if len(sort_indices) != arrow_rows.total:
        raise RuntimeError(
            f"Sort index length {len(sort_indices)} != embedding rows {arrow_rows.total}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    recomputed = 0
    previous_key = None
    state: IncrementalFeatureState | None = None

    with args.features.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as destination:
        for sorted_index, line in enumerate(source):
            physical_index = sort_indices[sorted_index]
            if physical_index in bad_indices:
                skipped += 1
                continue

            feature_row = json.loads(line)
            if feature_row["patient_id"] not in affected_patients:
                destination.write(line)
                written += 1
                continue

            embedding_row = arrow_rows.row(physical_index)
            metadata = (
                feature_row["patient_id"],
                feature_row["branch_id"],
                feature_row["turn_idx"],
                feature_row["role"],
            )
            embedding_metadata = (
                embedding_row["patient_id"],
                embedding_row["branch_id"],
                embedding_row["turn_idx"],
                embedding_row["role"],
            )
            if metadata != embedding_metadata:
                raise RuntimeError(
                    f"Feature/embedding mismatch at sorted row {sorted_index}: "
                    f"{metadata} != {embedding_metadata}"
                )

            key = (embedding_row["patient_id"], embedding_row["branch_id"])
            if key != previous_key:
                state = IncrementalFeatureState()
                previous_key = key
            assert state is not None
            role = "user" if embedding_row["role"] == "user" else "agent"
            repaired = state.add_turn(
                torch.tensor(embedding_row["embedding"], dtype=torch.float32), role
            )
            feature_row.update(repaired)
            destination.write(json.dumps(feature_row, ensure_ascii=False) + "\n")
            written += 1
            recomputed += 1

    if sorted_index + 1 != len(sort_indices):
        raise RuntimeError(
            f"Feature rows {sorted_index + 1} != sort indices {len(sort_indices)}"
        )
    expected_written = arrow_rows.total - len(bad_indices)
    if written != expected_written or skipped != len(bad_indices):
        raise RuntimeError(
            f"Output mismatch: written={written}/{expected_written}, "
            f"skipped={skipped}/{len(bad_indices)}"
        )
    print(
        json.dumps(
            {
                "input_rows": arrow_rows.total,
                "skipped_corrupt_rows": skipped,
                "recomputed_valid_rows": recomputed,
                "output_rows": written,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
