#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

from analyze_convo_answer_trajectory import (
    HEADER_RE,
    COMMITTED_RE,
    extract_options,
    extract_shadow_answer,
    iter_turn_blocks,
    parse_shadow_letter,
    split_patient_blocks,
)


def corrected_block(header_match, block):
    true_letter = header_match.group("true_letter")
    options = extract_options(block)
    final_shadow_letter = None

    turns = list(iter_turn_blocks(block))
    final_turns = [(m, b) for m, b in turns if m.group("final")]
    target_turn = final_turns[-1][1] if final_turns else (turns[-1][1] if turns else None)
    if target_turn:
        final_shadow_letter = parse_shadow_letter(
            extract_shadow_answer(target_turn), options
        )

    if not final_shadow_letter:
        return block, None

    new_label = "CORRECT" if final_shadow_letter == true_letter else "WRONG"
    new_header = (
        f"Patient #{header_match.group('id')}  |  {new_label}  |  "
        f"Predicted: {final_shadow_letter}  |  True: {true_letter} "
        f"({header_match.group('true_answer')})"
    )
    old_header_line = header_match.group(0)
    block = block.replace(old_header_line, new_header, 1)

    committed_match = COMMITTED_RE.search(block)
    if committed_match:
        block = (
            block[: committed_match.start("answer")]
            + final_shadow_letter
            + block[committed_match.end("answer") :]
        )

    return block, {
        "patient_id": int(header_match.group("id")),
        "old_predicted": header_match.group("predicted").strip(),
        "new_predicted": final_shadow_letter,
        "true_letter": true_letter,
        "old_label": header_match.group("label"),
        "new_label": new_label,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite MediQ convo headers/committed letters from final-turn shadow answers."
    )
    parser.add_argument(
        "--input",
        default="/home/hyang/mediQ/logs/scale_medgemma4b_yes_options_100q_convo.txt",
    )
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument(
        "--output",
        default="/home/hyang/mediQ/logs/scale_medgemma4b_yes_options_100q_convo_corrected.txt",
    )
    parser.add_argument(
        "--changes",
        default="/home/hyang/mediQ/logs/scale_medgemma4b_yes_options_100q_convo_corrections.txt",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    text = input_path.read_text(errors="replace")
    pieces = []
    changes = []
    last_end = 0

    for header_match, block in split_patient_blocks(text):
        pieces.append(text[last_end : header_match.start()])
        new_block, change = corrected_block(header_match, block)
        pieces.append(new_block)
        if change and (
            change["old_predicted"] != change["new_predicted"]
            or change["old_label"] != change["new_label"]
        ):
            changes.append(change)
        last_end = header_match.start() + len(block)
    pieces.append(text[last_end:])

    output_text = "".join(pieces)
    output_path = input_path if args.in_place else Path(args.output)
    output_path.write_text(output_text)

    changes_path = Path(args.changes)
    lines = [
        "patient_id\told_predicted\tnew_predicted\ttrue_letter\told_label\tnew_label"
    ]
    for change in changes:
        lines.append(
            "{patient_id}\t{old_predicted}\t{new_predicted}\t{true_letter}\t{old_label}\t{new_label}".format(
                **change
            )
        )
    changes_path.write_text("\n".join(lines) + "\n")

    print(f"Wrote corrected convo log to {output_path}")
    print(f"Wrote {len(changes)} changed records to {changes_path}")


if __name__ == "__main__":
    main()
