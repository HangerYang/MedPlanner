#!/usr/bin/env python3
"""Compute fact-based cumulative reward per turn and join with features.jsonl.

Reward rules (cumulative):
  - Patient turn: += new_unique_facts_revealed / total_facts
  - Last doctor turn per branch: += 1.0 if branch is CORRECT, else 0
  - All other turns: carry previous cumulative value

Parses doctor_view.txt directly — no embedding model needed.

Usage:
  python fact_reward.py \
    --txt new_outputs/.../scale_..._doctor_view.txt \
    --facts data/med_data/all_train_convo_medqa.jsonl \
    --features scope_saved/conversation_features/features.jsonl \
    --output scope_saved/conversation_features/features_with_fact_reward.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Txt parser
# ---------------------------------------------------------------------------

def parse_doctor_view(txt_path: Path) -> dict[tuple[int, str], dict]:
    """Parse doctor_view.txt into per-branch data.

    Returns {(patient_id, branch_id): {
        'is_correct': bool,
        'patient_responses': [text, ...],  # in order, Turn 1..N responses
    }}
    """
    result: dict[tuple[int, str], dict] = {}
    patient_id: int | None = None
    branch_id: str | None = None
    current: dict | None = None
    state = None  # 'doctor' | 'patient' | None
    buf: list[str] = []

    def flush_buf():
        nonlocal buf
        text = " ".join(buf).strip()
        buf = []
        return text

    with txt_path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            stripped = line.strip()

            # New patient
            m = re.match(r"Patient #(\d+)\s*\|", stripped)
            if m:
                patient_id = int(m.group(1))
                branch_id = None
                current = None
                state = None
                buf = []
                continue

            # New branch
            m = re.search(r"Branch\s+(\S+)\s*\|\s*(CORRECT|WRONG)", stripped)
            if m and patient_id is not None:
                # flush any pending patient text
                if state == "patient" and current is not None:
                    current["patient_responses"].append(flush_buf())
                branch_id = m.group(1)
                is_correct = m.group(2) == "CORRECT"
                current = {"is_correct": is_correct, "patient_responses": []}
                result[(patient_id, branch_id)] = current
                state = None
                buf = []
                continue

            if current is None:
                continue

            # Turn marker
            if re.match(r"---\s*Turn\s+\d+", stripped):
                if state == "patient":
                    current["patient_responses"].append(flush_buf())
                state = None
                buf = []
                continue

            # Final answer marker
            if stripped.startswith("→ Final Answer"):
                if state == "patient":
                    current["patient_responses"].append(flush_buf())
                state = None
                buf = []
                continue

            # Role markers
            if stripped.startswith("Doctor:"):
                if state == "patient":
                    current["patient_responses"].append(flush_buf())
                state = "doctor"
                rest = stripped[len("Doctor:"):].strip()
                buf = [rest] if rest else []
                continue

            if stripped.startswith("Patient:"):
                state = "patient"
                rest = stripped[len("Patient:"):].strip()
                buf = [rest] if rest else []
                continue

            # Continuation lines
            if state == "patient" and stripped:
                buf.append(stripped)

    # flush last patient buffer
    if state == "patient" and current is not None:
        text = flush_buf()
        if text:
            current["patient_responses"].append(text)

    return result


# ---------------------------------------------------------------------------
# Fact extraction
# ---------------------------------------------------------------------------

def extract_fact_numbers(text: str) -> set[int]:
    """Extract 1-indexed fact numbers from patient response like '3. text 5. text'."""
    return {int(n) for n in re.findall(r"\b(\d+)\.\s+\w", text)}


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def compute_rewards(
    branch_data: dict,        # from parse_doctor_view
    total_facts: int,
    patient_turn_idxs: list[int],   # turn_idx values of ALL patient turns, in order
    doctor_turn_idxs: list[int],    # turn_idx values of ALL doctor turns, in order
) -> dict[int, float]:
    """Returns {turn_idx: cumulative_fact_reward}."""
    responses = branch_data["patient_responses"]
    is_correct = branch_data["is_correct"]

    # Initial patient turns = those before the first doctor turn
    first_doctor_tidx = doctor_turn_idxs[0] if doctor_turn_idxs else float("inf")
    response_patient_idxs = [t for t in patient_turn_idxs if t > first_doctor_tidx]

    revealed: set[int] = set()
    cumulative = 0.0
    rewards: dict[int, float] = {}

    # Initial patient turns get reward=0
    for t in patient_turn_idxs:
        if t <= first_doctor_tidx:
            rewards[t] = 0.0

    # Interleave doctor and response patient turns in order
    all_turns = sorted(
        [(t, "doctor") for t in doctor_turn_idxs] +
        [(t, "patient") for t in response_patient_idxs]
    )

    response_idx = 0
    last_doctor_tidx = doctor_turn_idxs[-1] if doctor_turn_idxs else None

    for tidx, role in all_turns:
        if role == "patient" and response_idx < len(responses):
            text = responses[response_idx]
            response_idx += 1
            nums = extract_fact_numbers(text)
            new = nums - revealed
            revealed |= new
            if total_facts > 0:
                cumulative += len(new) / total_facts
            rewards[tidx] = cumulative
        elif role == "doctor":
            if tidx == last_doctor_tidx and is_correct:
                cumulative += 1.0
            rewards[tidx] = cumulative

    return rewards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt", type=Path, required=True, help="doctor_view.txt")
    parser.add_argument("--facts", type=Path, required=True, help="all_train_convo_medqa.jsonl")
    parser.add_argument("--features", type=Path, required=True, help="features.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # 1. Parse txt
    print("Parsing doctor_view.txt...")
    branch_data = parse_doctor_view(args.txt)
    print(f"  {len(branch_data)} branches parsed")

    # 2. Load total facts per patient
    print("Loading facts...")
    total_facts: dict[int, int] = {}
    with args.facts.open() as f:
        for line in f:
            item = json.loads(line)
            total_facts[item["id"]] = len(item["atomic_facts"])
    print(f"  {len(total_facts)} patients")

    # 3. Load features.jsonl, group turns by (patient_id, branch_id)
    print("Loading features.jsonl...")
    chunks = []
    for chunk in tqdm(
        pd.read_json(args.features, lines=True, chunksize=100_000),
        desc="Reading features",
    ):
        chunks.append(chunk)
    fdf = pd.concat(chunks, ignore_index=True)
    print(f"  {len(fdf)} rows")

    # Group turn_idx by role per conversation
    conv_patient: dict[tuple, list[int]] = defaultdict(list)
    conv_doctor: dict[tuple, list[int]] = defaultdict(list)
    for _, row in tqdm(fdf[["patient_id", "branch_id", "turn_idx", "role"]].iterrows(),
                       desc="Grouping turns", total=len(fdf)):
        key = (int(row["patient_id"]), row["branch_id"])
        if row["role"] == "user":
            conv_patient[key].append(int(row["turn_idx"]))
        else:
            conv_doctor[key].append(int(row["turn_idx"]))

    # 4. Compute rewards per conversation
    print("Computing rewards...")
    all_rewards: dict[tuple, dict[int, float]] = {}
    for key in tqdm(set(conv_patient) | set(conv_doctor), desc="Conversations"):
        pid, bid = key
        bd = branch_data.get(key)
        if bd is None:
            continue
        tf = total_facts.get(pid, 1)
        ptidxs = sorted(conv_patient.get(key, []))
        dtidxs = sorted(conv_doctor.get(key, []))
        all_rewards[key] = compute_rewards(bd, tf, ptidxs, dtidxs)

    # 5. Map rewards back to feature rows
    def get_reward(row):
        key = (int(row["patient_id"]), row["branch_id"])
        return all_rewards.get(key, {}).get(int(row["turn_idx"]), None)

    tqdm.pandas(desc="Mapping rewards")
    fdf["fact_reward"] = fdf.progress_apply(get_reward, axis=1).fillna(0.0)
    missing = fdf["fact_reward"].isna().sum()
    print(f"  {missing} missing rewards (filled with 0.0)")

    # 6. Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fdf.to_csv(args.output, index=False)
    print(f"Saved {len(fdf)} rows -> {args.output}")


if __name__ == "__main__":
    main()
