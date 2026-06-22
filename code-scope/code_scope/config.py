from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CODE_SCOPE_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CodeScopeConfig:
    model_name: str = os.environ.get("CODE_SCOPE_MODEL", "Qwen/Qwen3-4B")
    generation_device: str = os.environ.get("CODE_SCOPE_GENERATION_DEVICE", "cuda:0")
    transition_device: str = os.environ.get("CODE_SCOPE_TRANSITION_DEVICE", "cuda:1")
    reward_device: str = os.environ.get("CODE_SCOPE_REWARD_DEVICE", "cuda:1")
    transition_dir: str = os.environ.get(
        "CODE_SCOPE_TRANSITION_DIR",
        str(REPO / "mediQ_model_files/code-moe"),
    )
    reward_path: str = os.environ.get(
        "CODE_SCOPE_REWARD_PATH",
        str(REPO / "mediQ_model_files/code_feedback_cumulative_reward_mlp.pt"),
    )
    num_candidates: int = int(os.environ.get("CODE_SCOPE_NUM_CANDIDATES", "5"))
    max_new_tokens: int = int(os.environ.get("CODE_SCOPE_MAX_NEW_TOKENS", "512"))
    enable_thinking: bool = os.environ.get("CODE_SCOPE_ENABLE_THINKING", "0") == "1"
    thinking_budget: int = int(os.environ.get("CODE_SCOPE_THINKING_BUDGET", "0"))
    planning_rounds: int = int(os.environ.get("CODE_SCOPE_PLANNING_ROUNDS", "10"))
    mcts_time: float = float(os.environ.get("CODE_SCOPE_MCTS_TIME", "30"))
    transition_samples: int = int(os.environ.get("CODE_SCOPE_TRANSITION_SAMPLES", "4"))
    transition_noise: float = float(os.environ.get("CODE_SCOPE_TRANSITION_NOISE", "0.005"))
    diversity_penalty: float = float(os.environ.get("CODE_SCOPE_DIVERSITY_PENALTY", "1.0"))
    repetition_penalty: float = float(os.environ.get("CODE_SCOPE_REPETITION_PENALTY", "1.0"))
    enable_entropy_logging: bool = os.environ.get("CODE_SCOPE_ENTROPY_LOGGING", "0") == "1"
    trajectory_jsonl: str | None = os.environ.get("CODE_SCOPE_TRAJECTORY_JSONL")
    # Entropy-based early stopping during MCTS rollout simulation.
    # low_H: stop when H < threshold (state converged, ~75% step savings at 0.05).
    # high_H: stop when H > threshold (embedding drifted OOD, default off).
    rollout_low_H_threshold: float | None = (
        float(os.environ["CODE_SCOPE_ROLLOUT_LOW_H"]) if os.environ.get("CODE_SCOPE_ROLLOUT_LOW_H") else None
    )
    rollout_high_H_threshold: float | None = (
        float(os.environ["CODE_SCOPE_ROLLOUT_HIGH_H"]) if os.environ.get("CODE_SCOPE_ROLLOUT_HIGH_H") else None
    )

    @property
    def planning_depth(self) -> int:
        # Semantic depth starts at one and advances by two per assistant/user round.
        return 1 + 2 * self.planning_rounds
