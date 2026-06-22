from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ScopeMedicalConfig, normalize_device
from .conversation import MedicalConversation, SemanticState
from .embedding import Qwen3Embedding
from .mcts import SingleAgentMCTS, UpperConfidenceBounds
from .q_function import AccumulatedRewardTable
from .reward import EmbeddingScopeReward, FeatureScopeReward
from .semantic_env import SemanticConversationEnvironment
from .transition_model import TransitionModelMOE


class ScopeMedicalPlanner:
    def __init__(self, config: ScopeMedicalConfig | None = None) -> None:
        self.config = config or ScopeMedicalConfig()
        self.embedding_model = Qwen3Embedding(
            model_name=self.config.model_name,
            device_map=normalize_device(self.config.embedding_device),
        )
        if self.config.use_feature_reward:
            self.reward_function = FeatureScopeReward(
                path_to_model=self.config.feature_reward_model_path,
                device_map=normalize_device(self.config.reward_device),
            )
        else:
            self.reward_function = EmbeddingScopeReward(
                path_to_model=self.config.reward_model_path,
                device_map=normalize_device(self.config.reward_device),
            )
        self.transition_model = TransitionModelMOE(
            samples=self.config.transition_samples,
            noise=self.config.transition_noise,
            cuda=normalize_device(self.config.transition_device),
            transition_model_dir=self.config.transition_model_dir,
        )
        self.qfunction = AccumulatedRewardTable()

    def _embed_conversation(self, convo: MedicalConversation) -> np.ndarray:
        with torch.no_grad():
            return self.embedding_model.embed(convo).cpu().numpy().reshape(-1).astype(np.float32)

    def _action_semantics(self, convo: MedicalConversation, state_embedding: np.ndarray, actions: list[str],
                          initial_history: tuple = ()):
        semantics = []
        greedy_rewards = []
        for action in actions:
            action_embedding = self._embed_conversation(convo.with_expert_action(action))
            if self.config.use_feature_reward:
                action_history = initial_history + (tuple(action_embedding.tolist()),)
                reward = self.reward_function.value(action_history) - self.reward_function.value(initial_history)
            else:
                reward = self.reward_function.value(action_embedding) - self.reward_function.value(state_embedding)
            greedy_rewards.append(float(reward))
            semantics.append(tuple((action_embedding - state_embedding).astype(np.float32)))
        return semantics, greedy_rewards

    def choose_action(self, convo: MedicalConversation, candidate_actions: list[str], seed=None, trace_context=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed % (2**32))
            random.seed(seed)
        if not candidate_actions:
            raise ValueError("ScopeMedicalPlanner requires at least one candidate action.")

        print("generating action in realtime...")
        self.qfunction.reset()
        state_embedding = self._embed_conversation(convo)

        # Seed embedding history with one entry per message prefix in the current conversation
        initial_history = tuple(
            tuple(self._embed_conversation(
                type(convo)(convo.messages[: i + 1])
            ).tolist())
            for i in range(len(convo.messages))
        ) if self.config.use_feature_reward else ()

        action_semantics, greedy_rewards = self._action_semantics(convo, state_embedding, candidate_actions,
                                                                    initial_history=initial_history)

        env = SemanticConversationEnvironment(
            transition_model=self.transition_model,
            initial_embedding=tuple(state_embedding),
            max_depth=self.config.planning_depth,
            reward_function=self.reward_function,
            initial_embedding_history=initial_history,
        )
        env.state_to_action_map[tuple(state_embedding)] = action_semantics

        print("performing MCTS search...")
        mcts = SingleAgentMCTS(env, self.qfunction, UpperConfidenceBounds())
        mcts.mcts(timeout=self.config.mcts_time, seed=seed)

        print("getting best action from Q function...")
        semantic_state = SemanticState(tuple(state_embedding), depth=1)
        qs = self.qfunction.get_qs(semantic_state, action_semantics)
        best_idx = int(np.argmax(qs))
        best_action = candidate_actions[best_idx]
        print("possible actions generated: ", candidate_actions)
        print(f"action selected by online agent: {best_action}")

        result = {
            "possible_actions": candidate_actions,
            "possible_actions_reward": [float(q) for q in qs],
            "selected_action_index": best_idx,
            "selected_action": best_action,
            "greedy_rewards": greedy_rewards,
            "greedy_action_index": int(np.argmax(greedy_rewards)) if greedy_rewards else None,
        }
        self._write_trace({**(trace_context or {}), **result})
        return best_action, result

    def _write_trace(self, row: dict[str, Any]) -> None:
        if not self.config.trace_jsonl:
            return
        path = Path(self.config.trace_jsonl)
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")


_PLANNER = None


def get_planner() -> ScopeMedicalPlanner:
    global _PLANNER
    if _PLANNER is None:
        print("[init] Building SCOPE-Medical planner...")
        _PLANNER = ScopeMedicalPlanner()
    return _PLANNER
