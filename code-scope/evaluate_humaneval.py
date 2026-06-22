#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code_scope.config import CodeScopeConfig
from code_scope.execution import check_completion
from code_scope.model import QwenCodeModel
from code_scope.planner import CodeScopePlanner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scope", "baseline"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execution-timeout", type=float, default=5.0)
    parser.add_argument("--dataset", default="openai/openai_humaneval")
    parser.add_argument(
        "--task-ids", type=int, nargs="+", default=None,
        help="Run only these HumanEval indices (e.g. --task-ids 5 6 10 37 38)",
    )
    return parser.parse_args()


def load_problems(name: str):
    dataset = load_dataset(name, split="test")
    return [dict(row) for row in dataset]


def existing_task_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["task_id"] for line in path.open()}


# ---------------------------------------------------------------------------
# Trajectory correlation helpers
# ---------------------------------------------------------------------------

def _mean(vals: list[float]) -> float | None:
    return float(np.mean(vals)) if vals else None



def build_trajectory_record(
    problem: dict,
    candidates: list[str],
    executions: list[dict],
    planning: dict,
) -> dict | None:
    """Merge execution pass/fail into the planner's trajectory data and compute correlations."""
    traj = planning.get("trajectory")
    if traj is None:
        return None

    candidate_records = traj.get("candidates", [])
    for i, rec in enumerate(candidate_records):
        rec["passed"] = executions[i]["passed"] if i < len(executions) else None
        rec["candidate_index"] = i

    # Entropy correlation split by pass/fail
    passed_entropy = [r["entropy"] for r in candidate_records if r.get("passed") and "entropy" in r]
    failed_entropy = [r["entropy"] for r in candidate_records if r.get("passed") is False and "entropy" in r]

    correlation = {
        "passed_entropy_mean": _mean(passed_entropy),
        "failed_entropy_mean": _mean(failed_entropy),
        "entropy_delta": (
            (_mean(passed_entropy) - _mean(failed_entropy))
            if passed_entropy and failed_entropy
            else None
        ),
    }

    return {
        "task_id": problem["task_id"],
        "n_candidates": len(candidates),
        "selected_index": planning["selected_index"],
        "selected_passed": executions[planning["selected_index"]]["passed"],
        "oracle_passed": any(e["passed"] for e in executions),
        "prompt_entropy": traj.get("prompt_entropy"),
        "candidates": candidate_records,
        "rollouts": traj.get("rollouts", []),
        "rollout_summary": traj.get("summary", {}),
        "correlation": correlation,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    config = CodeScopeConfig()
    problems = load_problems(args.dataset)
    end = None if not args.limit else args.start + args.limit
    problems = problems[args.start:end]
    if args.task_ids is not None:
        task_id_set = {f"HumanEval/{i}" for i in args.task_ids}
        problems = [p for p in problems if p["task_id"] in task_id_set]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = existing_task_ids(args.output)

    model = QwenCodeModel(config.model_name, config.generation_device, config.enable_thinking, config.thinking_budget)
    lm_head = model.model.lm_head if args.mode == "scope" and config.enable_entropy_logging else None
    planner = CodeScopePlanner(config, lm_head=lm_head) if args.mode == "scope" else None

    # Trajectory JSONL: derive from output path unless overridden via env
    traj_path: Path | None = None
    if config.trajectory_jsonl:
        traj_path = Path(config.trajectory_jsonl)
    elif config.enable_entropy_logging and args.mode == "scope":
        traj_path = args.output.with_name(args.output.stem + "_trajectory.jsonl")
    if traj_path:
        traj_path.parent.mkdir(parents=True, exist_ok=True)

    traj_file = traj_path.open("a") if traj_path else None

    try:
        with args.output.open("a") as output:
            for problem in tqdm(problems, desc=f"HumanEval {args.mode}"):
                if problem["task_id"] in completed:
                    continue
                seed = args.seed + int(problem["task_id"].split("/")[-1])
                random.seed(seed)

                if args.mode == "baseline":
                    completion = model.generate_greedy(problem["prompt"], config.max_new_tokens)
                    result = {
                        "task_id": problem["task_id"],
                        "mode": "baseline",
                        "completion": completion,
                        "execution": check_completion(problem, completion, args.execution_timeout),
                    }
                else:
                    candidates = model.generate_beams(
                        problem["prompt"],
                        config.num_candidates,
                        config.max_new_tokens,
                        config.diversity_penalty,
                        config.repetition_penalty,
                    )
                    prompt_embedding = model.embed_turn(problem["prompt"])
                    candidate_embeddings = [model.embed_turn(c) for c in candidates]
                    planning = planner.choose(prompt_embedding, candidate_embeddings, seed)
                    executions = [
                        check_completion(problem, candidate, args.execution_timeout)
                        for candidate in candidates
                    ]
                    selected = planning["selected_index"]
                    result = {
                        "task_id": problem["task_id"],
                        "mode": "scope",
                        "candidates": candidates,
                        "candidate_executions": executions,
                        "selected_index": selected,
                        "completion": candidates[selected],
                        "execution": executions[selected],
                        "planning": {k: v for k, v in planning.items() if k != "trajectory"},
                        "first_beam_passed": executions[0]["passed"],
                        "immediate_selected_passed": executions[planning["immediate_selected_index"]]["passed"],
                        "oracle_beam_passed": any(item["passed"] for item in executions),
                    }

                    # Save full trajectory record to separate file
                    if traj_file is not None:
                        traj_record = build_trajectory_record(problem, candidates, executions, planning)
                        if traj_record is not None:
                            traj_file.write(json.dumps(traj_record) + "\n")
                            traj_file.flush()

                output.write(json.dumps(result) + "\n")
                output.flush()
    finally:
        if traj_file is not None:
            traj_file.close()


if __name__ == "__main__":
    main()
