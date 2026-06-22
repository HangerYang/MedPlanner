#!/usr/bin/env python3
"""Convert feature reward CSV to reward-training JSONL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.input.open(newline="", encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as destination:
        for row in csv.DictReader(source):
            output = {
                "patient_id": int(row["patient_id"]),
                "branch_id": row["branch_id"],
                "turn_idx": int(row["turn_idx"]),
                "role": row["role"],
                "reward": float(row["fact_reward"]),
            }
            for key, value in row.items():
                if key not in output and key != "fact_reward":
                    output[key] = float(value)
            destination.write(json.dumps(output, ensure_ascii=False) + "\n")
            written += 1
    print(f"Wrote {written} rows -> {args.output}")


if __name__ == "__main__":
    main()
