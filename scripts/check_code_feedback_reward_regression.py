#!/usr/bin/env python3
"""Check annotated Code-Feedback rewards against reviewed regression cases."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rewards",
        type=Path,
        default=Path("data/med_data/code_feedback_rewards.jsonl"),
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("data/med_data/code_feedback_reward_regression.json"),
    )
    args = parser.parse_args()

    expected = {
        (case["id"], case["turn"]): case
        for case in json.loads(args.fixture.read_text(encoding="utf-8"))["cases"]
    }
    found = {}
    for line in args.rewards.open(encoding="utf-8"):
        conversation = json.loads(line)
        for message in conversation["messages"]:
            key = (conversation["id"], message["turn"])
            if key in expected:
                found[key] = message

    errors = []
    for key, case in expected.items():
        actual = found.get(key)
        if actual is None:
            errors.append(f"{key}: missing")
            continue
        for field in ("reward", "reward_event"):
            if actual.get(field) != case[field]:
                errors.append(
                    f"{key}: {field} expected {case[field]!r}, "
                    f"got {actual.get(field)!r}"
                )
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Regression checks passed: {len(expected)} cases")


if __name__ == "__main__":
    main()
