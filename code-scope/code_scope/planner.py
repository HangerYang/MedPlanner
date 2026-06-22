from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
MEDICAL_SCOPE = REPO / "medical-scope"
for path in (REPO, MEDICAL_SCOPE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from medical_scope.conversation import SemanticState
from medical_scope.mcts import UpperConfidenceBounds
from medical_scope.q_function import AccumulatedRewardTable
from medical_scope.semantic_env import SemanticConversationEnvironment
from medical_scope.transition_model import TransitionModelMDN, TransitionModelMOE

from .config import CodeScopeConfig
from .entropy import EmbeddingEntropyScorer, EntropyTrackingMCTS
from .reward import CodeFeatureReward


class CodeScopePlanner:
    def __init__(self, config: CodeScopeConfig, lm_head: Optional[nn.Linear] = None) -> None:
        self.config = config
        self.reward = CodeFeatureReward(config.reward_path, config.reward_device)
        transition_cls = (
            TransitionModelMDN
            if "mdn" in str(config.transition_dir).lower()
            else TransitionModelMOE
        )
        self.transition = transition_cls(
            samples=config.transition_samples,
            noise=config.transition_noise,
            cuda=config.transition_device,
            transition_model_dir=config.transition_dir,
        )
        self.entropy_scorer: EmbeddingEntropyScorer | None = None
        if config.enable_entropy_logging and lm_head is not None:
            print("[init] Building EmbeddingEntropyScorer from lm_head...")
            self.entropy_scorer = EmbeddingEntropyScorer(lm_head)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _candidate_metrics(
        self,
        prompt_embedding: np.ndarray,
        candidate_embeddings: list[np.ndarray],
    ) -> list[dict]:
        """Return entropy for each candidate (no conversation features)."""
        metrics = []
        for candidate in candidate_embeddings:
            entry: dict = {}
            candidate_t = tuple(candidate.astype(np.float32))
            if self.entropy_scorer is not None:
                entry["entropy"] = self.entropy_scorer.entropy(candidate_t)
            metrics.append(entry)
        return metrics

    # ------------------------------------------------------------------
    # Main planner entry point
    # ------------------------------------------------------------------

    def choose(
        self,
        prompt_embedding: np.ndarray,
        candidate_embeddings: list[np.ndarray],
        seed: int,
    ) -> dict:
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32))
        random.seed(seed)

        root = tuple(prompt_embedding.astype(np.float32))
        initial_history = (root,)
        actions = [
            tuple((candidate.astype(np.float32) - prompt_embedding.astype(np.float32)).tolist())
            for candidate in candidate_embeddings
        ]
        root_action_to_index = {action: i for i, action in enumerate(actions)}
        immediate_values = [
            self.reward.value(initial_history + (tuple(candidate.astype(np.float32)),))
            - self.reward.value(initial_history)
            for candidate in candidate_embeddings
        ]

        # Per-candidate entropy; prompt entropy as root baseline
        candidate_metrics = self._candidate_metrics(prompt_embedding, candidate_embeddings)
        prompt_entropy: float | None = None
        if self.entropy_scorer is not None:
            prompt_entropy = self.entropy_scorer.entropy(root)

        environment = SemanticConversationEnvironment(
            transition_model=self.transition,
            initial_embedding=root,
            max_depth=self.config.planning_depth,
            reward_function=self.reward,
            initial_embedding_history=initial_history,
        )
        environment.state_to_action_map[root] = actions

        qfunction = AccumulatedRewardTable()
        mcts = EntropyTrackingMCTS(
            environment,
            qfunction,
            UpperConfidenceBounds(),
            entropy_scorer=self.entropy_scorer,
            low_H_threshold=self.config.rollout_low_H_threshold,
            high_H_threshold=self.config.rollout_high_H_threshold,
            root_action_to_index=root_action_to_index,
        )
        mcts.mcts(timeout=self.config.mcts_time, seed=seed)

        root_state = SemanticState(root, depth=1, embedding_history=initial_history)
        q_values = [float(v) for v in qfunction.get_qs(root_state, actions)]
        selected_index = int(np.argmax(q_values))

        return {
            "selected_index": selected_index,
            "q_values": q_values,
            "immediate_values": [float(v) for v in immediate_values],
            "immediate_selected_index": int(np.argmax(immediate_values)),
            "planning_depth": self.config.planning_depth,
            "planning_rounds": self.config.planning_rounds,
            # Trajectory data — present whenever entropy logging is on, else None
            "trajectory": {
                "prompt_entropy": prompt_entropy,
                "candidates": candidate_metrics,
                "rollouts": mcts.rollout_data,
                "summary": mcts.rollout_summary(),
            } if self.entropy_scorer is not None else None,
        }
