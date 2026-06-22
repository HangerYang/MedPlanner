#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def rate(rows, key):
    return sum(bool(row.get(key)) for row in rows) / len(rows) if rows else 0.0


parser = argparse.ArgumentParser()
parser.add_argument("paths", nargs="+", type=Path)
args = parser.parse_args()
for path in args.paths:
    rows = [json.loads(line) for line in path.open()]
    summary = {"path": str(path), "tasks": len(rows), "pass@1": rate(rows, "execution")}
    summary["pass@1"] = sum(row["execution"]["passed"] for row in rows) / len(rows) if rows else 0.0
    if rows and rows[0]["mode"] == "scope":
        summary.update(
            {
                "first_beam_pass@1": rate(rows, "first_beam_passed"),
                "immediate_reward_pass@1": rate(rows, "immediate_selected_passed"),
                "oracle_beam_pass@1": rate(rows, "oracle_beam_passed"),
            }
        )
    print(json.dumps(summary, indent=2))
