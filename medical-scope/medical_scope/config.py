from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ScopeMedicalConfig:
    model_name: str = os.environ.get("SCOPE_MEDICAL_EMBED_MODEL", "Qwen/Qwen3-4B")
    embedding_device: str = os.environ.get("SCOPE_MEDICAL_EMBED_DEVICE", os.environ.get("SCOPE_CUDA_Q", "cuda:1"))
    transition_device: str = os.environ.get("SCOPE_MEDICAL_TRANSITION_DEVICE", os.environ.get("SCOPE_CUDA_Q", "cuda:1"))
    reward_device: str = os.environ.get("SCOPE_MEDICAL_REWARD_DEVICE", os.environ.get("SCOPE_CUDA_Q", "cuda:1"))
    transition_model_dir: str = os.environ.get(
        "SCOPE_MEDICAL_TRANSITION_DIR",
        str(REPO_ROOT / "scope_saved" / "transition_models"),
    )
    reward_model_path: str = os.environ.get(
        "SCOPE_MEDICAL_REWARD_PATH",
        str(REPO_ROOT / "scope_saved" / "reward" / "embedding_mediQ_reward_cumulative.pt"),
    )
    mcts_time: float = float(os.environ.get("SCOPE_MEDICAL_MCTS_TIME", os.environ.get("SCOPE_MCTS_TIME", "30")))
    planning_depth: int = int(os.environ.get("SCOPE_MEDICAL_PLANNING_DEPTH", "8"))
    num_candidates: int = int(os.environ.get("SCOPE_MEDICAL_NUM_CANDIDATES", os.environ.get("SCOPE_CANDIDATE_PASSES", "5")))
    candidate_max_new_tokens: int = int(os.environ.get("SCOPE_MEDICAL_CANDIDATE_MAX_NEW_TOKENS", "500"))
    candidate_num_beam_groups: int | None = (
        int(os.environ["SCOPE_MEDICAL_CANDIDATE_NUM_BEAM_GROUPS"])
        if os.environ.get("SCOPE_MEDICAL_CANDIDATE_NUM_BEAM_GROUPS")
        else None
    )
    candidate_diversity_penalty: float = float(os.environ.get("SCOPE_MEDICAL_CANDIDATE_DIVERSITY_PENALTY", "1.0"))
    candidate_repetition_penalty: float = float(os.environ.get("SCOPE_MEDICAL_CANDIDATE_REPETITION_PENALTY", "1.0"))
    transition_samples: int = int(os.environ.get("SCOPE_MEDICAL_TRANSITION_SAMPLES", "4"))
    transition_noise: float = float(os.environ.get("SCOPE_MEDICAL_TRANSITION_NOISE", "0.005"))
    trace_jsonl: str | None = os.environ.get("SCOPE_MEDICAL_TRACE_JSONL", os.environ.get("SCOPE_TRACE_JSONL"))
    use_feature_reward: bool = os.environ.get("SCOPE_MEDICAL_USE_FEATURE_REWARD", "0") == "1"
    feature_reward_model_path: str = os.environ.get(
        "SCOPE_MEDICAL_FEATURE_REWARD_PATH",
        str(REPO_ROOT / "scope_saved" / "reward" / "feature_reward_mlp.pt"),
    )


def normalize_device(value):
    if value is None:
        return "cpu"
    value = str(value)
    if value.isdigit():
        return f"cuda:{value}" if os.environ.get("CUDA_VISIBLE_DEVICES") is None else f"cuda:{value}"
    return value
