#!/usr/bin/env python3
"""Print FixedExpert turn-1 prompts (no LLM). Run from mediQ/src: python demo_first_turn_prompt.py"""
import json
import sys
from pathlib import Path

import prompts


def main():
    root = Path(__file__).resolve().parent
    default = root.parent / "data" / "med_data" / "all_dev_convo.jsonl"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    line = path.read_text().splitlines()[0]
    sample = json.loads(line)

    # Same initial_info as Patient when context is a list and no initial_info
    patient_info = sample["context"][0]
    conv_log = "None"
    inquiry = sample["question"]
    options = sample["options"]
    options_text = f'A: {options["A"]}, B: {options["B"]}, C: {options["C"]}, D: {options["D"]}'

    answer_task = prompts.expert_system["answer"]
    user_answer = prompts.expert_system["curr_template"].format(
        patient_info, conv_log, inquiry, options_text, answer_task
    )

    atomic_task = prompts.expert_system["atomic_question_improved"]

    print("=" * 80)
    print("SYSTEM (all expert calls)")
    print("=" * 80)
    print(prompts.expert_system["meditron_system_msg"])
    print()
    print("=" * 80)
    print("USER — fixed_abstention_decision: letter-choice probe (turn 1)")
    print("=" * 80)
    print(user_answer)
    print()
    print("(Then model returns assistant text with a letter; that is appended to messages.)")
    print()
    print("=" * 80)
    print("USER — question_generation append (independent_modules=False)")
    print("=" * 80)
    print(atomic_task)
    print()
    print("Note: RandomPatient never calls the patient_model; it picks random context lines.")


if __name__ == "__main__":
    main()
