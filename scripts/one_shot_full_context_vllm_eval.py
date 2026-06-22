#!/usr/bin/env python3
"""Replay SCOPE-Medical final-answer prompts exactly.

Despite the historical filename, this script intentionally uses the same mediQ
HuggingFace final-answer path as ScopeMedicalExpert:

    expert_functions.final_choice_with_options
      -> expert_basics.expert_response_choice
      -> helper.get_response
      -> ModelCache.huggingface_generate(... do_sample=False)

Rows are read from a SCOPE results JSONL. Each row's final saved
`condensed_evidence` is used verbatim as PATIENT INFORMATION, matching the
SCOPE final-answer call.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
MEDICAL_SCOPE_DIR = REPO_ROOT / "medical-scope"
for item in (str(MEDICAL_SCOPE_DIR), str(SRC_DIR)):
    if item not in sys.path:
        sys.path.insert(0, item)

# Import for side effect: patches helper.ModelCache._to_content_list for Qwen,
# exactly as run_scope_medical.sh does when importing ScopeMedicalExpert.
import medical_scope.expert  # noqa: F401,E402
import expert_functions  # noqa: E402
import helper as mediq_helper  # noqa: E402


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict]:
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if limit and len(rows) >= limit:
            break
    return rows


def final_condensed_evidence(scope_row: dict) -> str:
    extras = (scope_row.get("interactive_system") or {}).get("temp_additional_info") or []
    for item in reversed(extras):
        value = item.get("condensed_evidence")
        if value:
            return str(value)

    info = scope_row.get("info") or {}
    system = scope_row.get("interactive_system") or {}
    questions = system.get("questions") or []
    answers = system.get("answers") or []
    patient_state = {
        "initial_info": info.get("initial_info") or "",
        "interaction_history": [
            {"question": question, "answer": answer}
            for question, answer in zip(questions, answers)
        ],
    }
    return str(expert_functions.condensed_patient_state(patient_state)["initial_info"])


def patient_information_for_mode(sample: dict, context_mode: str) -> str:
    """Build PATIENT INFORMATION for one-shot mediQ final-answer eval."""
    context = sample.get("context") or []
    if context_mode == "initial":
        if isinstance(context, list) and context:
            return str(context[0])
        if isinstance(context, str) and context.strip():
            return context.split(". ")[0]
        return str(sample.get("initial_info") or "")
    if context_mode == "full":
        if isinstance(context, list) and context:
            return " ".join(str(part) for part in context)
        return str(context or sample.get("initial_info") or "")
    raise ValueError(f"Unknown context_mode: {context_mode!r} (use 'initial' or 'full')")


def load_data_eval_rows(
    path: str | Path,
    *,
    context_mode: str,
    limit: int | None = None,
) -> list[dict]:
    rows = []
    for sample in load_jsonl(path, limit=limit):
        rows.append(
            {
                "id": sample["id"],
                "question": sample.get("question"),
                "options": sample.get("options"),
                "answer_idx": sample.get("answer_idx"),
                "answer": sample.get("answer"),
                "patient_information": patient_information_for_mode(sample, context_mode),
                "context_mode": context_mode,
                "scope_letter_choice": None,
                "scope_correct": None,
            }
        )
    return rows


def load_scope_eval_rows(path: str | Path, limit: int | None = None) -> list[dict]:
    rows = []
    for scope_row in load_jsonl(path, limit=limit):
        info = scope_row.get("info") or {}
        rows.append(
            {
                "id": scope_row["id"],
                "question": info.get("question"),
                "options": info.get("options"),
                "answer_idx": info.get("correct_answer_idx"),
                "answer": info.get("correct_answer"),
                "patient_information": final_condensed_evidence(scope_row),
                "context_mode": "condensed",
                "scope_letter_choice": (scope_row.get("interactive_system") or {}).get("letter_choice"),
                "scope_correct": (scope_row.get("interactive_system") or {}).get("correct"),
            }
        )
    return rows


def load_rows_jsonl(path: str | Path, limit: int | None = None) -> list[dict]:
    rows = []
    for row in load_jsonl(path, limit=limit):
        rows.append(
            {
                "id": row["id"],
                "question": row.get("question"),
                "options": row.get("options"),
                "answer_idx": row.get("answer_idx"),
                "answer": row.get("answer"),
                "patient_information": row["patient_information"],
                "context_mode": row.get("context_mode") or row.get("fact_cut_mode"),
                "scope_letter_choice": row.get("scope_letter_choice"),
                "scope_correct": row.get("scope_correct"),
            }
        )
    return rows


def load_eval_rows(args: argparse.Namespace) -> list[dict]:
    limit = args.max_examples or None
    if args.data:
        return load_data_eval_rows(args.data, context_mode=args.context_mode, limit=limit)
    if args.rows_from_results:
        return load_scope_eval_rows(args.rows_from_results, limit=limit)
    if args.rows_jsonl:
        return load_rows_jsonl(args.rows_jsonl, limit=limit)
    raise SystemExit(
        "Provide --data, --rows-from-results, or --rows-jsonl (prebuilt patient_information rows)."
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_hf_sampling() -> None:
    if getattr(mediq_helper.ModelCache, "_replay_sampling_patch", False):
        return

    original = mediq_helper.ModelCache.huggingface_generate

    def huggingface_generate(self, messages):
        if not bool(self.args.get("do_sample", False)):
            return original(self, messages)

        tmpl_kwargs = mediq_helper._chat_template_kwargs(self.model_name)
        inputs = self.tokenizer.apply_chat_template(
            self._to_content_list(messages),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **tmpl_kwargs,
        ).to(self.model.device)
        inputs = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
            )

        raw_text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        response_text = mediq_helper._strip_thinking(raw_text)
        usage = {"input_tokens": input_len, "output_tokens": outputs.shape[-1] - input_len}
        return response_text, None, usage

    mediq_helper.ModelCache.huggingface_generate = huggingface_generate
    mediq_helper.ModelCache._replay_sampling_patch = True


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def run_eval(args: argparse.Namespace) -> None:
    if args.do_sample:
        enable_hf_sampling()

    rows = load_eval_rows(args)
    if not rows:
        raise SystemExit("No evaluation rows loaded.")
    has_scope_reference = rows[0].get("scope_letter_choice") is not None

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed + i for i in range(args.num_seeds)]
    summaries = []

    with output_path.open("w") as f:
        for seed in seeds:
            set_seed(seed)
            correct = 0
            parsed = 0
            scope_correct = 0
            agreement = 0

            for row in rows:
                patient_state = {
                    "initial_info": row["patient_information"],
                    "interaction_history": [],
                }
                letter, usage = expert_functions.final_choice_with_options(
                    patient_state,
                    row["question"],
                    row["options"],
                    model_name=args.model,
                    use_vllm=False,
                    use_api=None,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    top_p=args.top_p,
                    top_logprobs=args.top_logprobs,
                    api_account=args.api_account,
                    tensor_parallel_size=args.tensor_parallel_size,
                    batch_size=args.batch_size,
                    do_sample=args.do_sample,
                )
                is_correct = letter == row["answer_idx"] if letter else False
                correct += int(is_correct)
                parsed += int(letter is not None)
                if has_scope_reference:
                    scope_correct += int(bool(row.get("scope_correct")))
                    agreement += int(letter == row.get("scope_letter_choice"))

                record = {
                    "seed": seed,
                    "id": row["id"],
                    "question": row["question"],
                    "options": row["options"],
                    "true_letter": row["answer_idx"],
                    "true_answer": row["answer"],
                    "context_mode": row.get("context_mode"),
                    "patient_information": row["patient_information"],
                    "model": args.model,
                    "backend": "huggingface_mediq_helper",
                    "do_sample": args.do_sample,
                    "temperature_arg": args.temperature,
                    "top_p_arg": args.top_p,
                    "max_tokens": args.max_tokens,
                    "parsed_letter": letter,
                    "model_output": letter,
                    "correct": is_correct,
                    "scope_letter_choice": row.get("scope_letter_choice"),
                    "scope_correct": row.get("scope_correct"),
                    "agrees_with_scope": letter == row.get("scope_letter_choice"),
                    "usage": usage,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            total = len(rows)
            summary = {
                "seed": seed,
                "rows": total,
                "parsed": parsed,
                "parsed_rate": parsed / total,
                "correct": correct,
                "accuracy": correct / total,
            }
            if has_scope_reference:
                summary.update(
                    {
                        "scope_correct": scope_correct,
                        "scope_accuracy": scope_correct / total,
                        "agreement": agreement,
                        "agreement_rate": agreement / total,
                    }
                )
            summaries.append(summary)
            msg = (
                f"Seed {seed}: parsed={parsed}/{total} ({summary['parsed_rate']:.4f}) "
                f"correct={correct}/{total} ({summary['accuracy']:.4f})"
            )
            if has_scope_reference:
                msg += f" agreement={agreement}/{total} ({summary['agreement_rate']:.4f})"
            print(msg)

    acc_mean, acc_std = mean_std([item["accuracy"] for item in summaries])
    parsed_mean, parsed_std = mean_std([item["parsed_rate"] for item in summaries])
    total = len(rows)
    aggregate = {
        "model": args.model,
        "context_mode": rows[0].get("context_mode"),
        "data": str(args.data) if args.data else None,
        "rows_from_results": str(args.rows_from_results) if args.rows_from_results else None,
        "rows_jsonl": str(args.rows_jsonl) if args.rows_jsonl else None,
        "rows": total,
        "num_seeds": args.num_seeds,
        "seeds": seeds,
        "output": str(output_path),
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "parsed_rate_mean": parsed_mean,
        "parsed_rate_std": parsed_std,
        "per_seed": summaries,
    }
    if has_scope_reference:
        agree_mean, agree_std = mean_std([item["agreement_rate"] for item in summaries])
        aggregate.update(
            {
                "agreement_rate_mean": agree_mean,
                "agreement_rate_std": agree_std,
                "scope_accuracy": summaries[0]["scope_accuracy"],
            }
        )
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n")

    print(f"Model: {args.model}")
    print(f"Rows per seed: {total}")
    print(f"Seeds: {seeds}")
    print(f"Output: {output_path}")
    print(f"Summary: {summary_path}")
    print(f"Accuracy mean±std: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"Parsed mean±std: {parsed_mean:.4f} ± {parsed_std:.4f}")
    if has_scope_reference:
        print(f"Agreement mean±std: {aggregate['agreement_rate_mean']:.4f} ± {aggregate['agreement_rate_std']:.4f}")
        print(f"Source run accuracy: {aggregate['scope_accuracy']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay SCOPE final-answer calls using SCOPE condensed_evidence and mediQ HF generation."
    )
    parser.add_argument("--rows-from-results", default=None, help="SCOPE result JSONL to replay.")
    parser.add_argument(
        "--rows-jsonl",
        default=None,
        help="Prebuilt eval rows JSONL (e.g. shadow min/max fact cuts from shadow_fact_cut_rows.py).",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="MedQA JSONL for one-shot initial/full context eval (e.g. all_test_convo_medqa.jsonl).",
    )
    parser.add_argument(
        "--context-mode",
        choices=("initial", "full", "condensed"),
        default="condensed",
        help="Patient info source when using --data: initial=first context sentence, full=all context.",
    )
    parser.add_argument("--model", required=True, help="HuggingFace model id or local path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--top-logprobs", type=int, default=0)
    parser.add_argument("--api-account", default="mediQ")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=1)

    # Compatibility no-ops for old run_eval.sh invocations.
    parser.add_argument("--prompt-format", default=None)
    parser.add_argument("--reference-results", nargs="*", default=[])
    parser.add_argument("--no-reasoning", action="store_true")
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)

    args = parser.parse_args()
    sources = sum(bool(x) for x in (args.data, args.rows_from_results, args.rows_jsonl))
    if sources != 1:
        raise SystemExit("Provide exactly one of --data, --rows-from-results, or --rows-jsonl.")
    if args.data and args.context_mode not in {"initial", "full"}:
        raise SystemExit("--data requires --context-mode initial or full.")
    if args.rows_from_results and not args.data:
        args.context_mode = "condensed"
    run_eval(args)


if __name__ == "__main__":
    main()
