from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from conversation_feature import IncrementalFeatureState  # noqa: E402

_MEDICAL_SCOPE = _REPO / "medical-scope"
if str(_MEDICAL_SCOPE) not in sys.path:
    sys.path.insert(0, str(_MEDICAL_SCOPE))

from medical_scope.mcts import SingleAgentMCTS  # noqa: E402


# ---------------------------------------------------------------------------
# Entropy scorer
# ---------------------------------------------------------------------------

class EmbeddingEntropyScorer:
    """Project a hidden-state embedding through a frozen lm_head and compute Shannon entropy.

    z           ∈  ℝ^hidden        last-layer embedding
    logits  =  W_lm · z            ∈  ℝ^vocab_size
    p       =  softmax(logits)
    H       =  -Σ p · log(p)       scalar entropy in nats
    """

    def __init__(self, lm_head: nn.Linear) -> None:
        # Detach and cast to float32 for numerical stability.
        # Keep weight on whatever device lm_head already lives on.
        self.weight = lm_head.weight.detach().float()  # [vocab_size, hidden_size]
        self.device = self.weight.device

    @torch.inference_mode()
    def entropy(self, embedding: tuple | np.ndarray) -> float:
        z = torch.tensor(np.asarray(embedding, dtype=np.float32), device=self.device)
        logits = z @ self.weight.T       # [vocab_size]
        log_p = torch.log_softmax(logits, dim=-1)
        return float(-(log_p.exp() * log_p).sum().item())


# ---------------------------------------------------------------------------
# Conversation feature helper
# ---------------------------------------------------------------------------

def features_from_history(embedding_history: tuple) -> dict[str, float]:
    """Run embedding_history through IncrementalFeatureState and return 26 features.

    embedding_history alternates user / agent starting from index 0 (user = prompt).
    This matches the SemanticConversationEnvironment._extend_history layout.
    """
    state = IncrementalFeatureState()
    for i, emb in enumerate(embedding_history):
        role = "user" if i % 2 == 0 else "agent"
        t = torch.tensor(np.asarray(emb, dtype=np.float32), dtype=torch.float32)
        state.add_turn(t, role)
    return dict(state._f)


# ---------------------------------------------------------------------------
# MCTS subclass with full trajectory logging and entropy-based early stopping
# ---------------------------------------------------------------------------

class EntropyTrackingMCTS(SingleAgentMCTS):
    """MCTS subclass that records per-step entropy and supports entropy-based early stopping.

    Two optional stopping criteria (applied only when entropy_scorer is set):

      low_H_threshold  — stop when H drops below this value.
        The state has converged to a near-deterministic region; continuing yields
        no new reward information. Cuts ~75–78% of rollout steps in practice
        (based on empirical data: 54% of states have H<0.05 by depth 7, 74% by depth 9).

      high_H_threshold — stop when H exceeds this value.
        The simulated embedding has drifted far out of the model's learned distribution;
        the transition model output is unreliable. Returns reward accumulated so far.

    Rewards and Q-function updates are unchanged by early stopping — the rollout
    simply returns its partial cumulative reward at the stopping point.
    """

    def __init__(
        self,
        mdp,
        qfunction,
        bandit=None,
        entropy_scorer: EmbeddingEntropyScorer | None = None,
        low_H_threshold: float | None = None,
        high_H_threshold: float | None = None,
        root_action_to_index: dict[tuple, int] | None = None,
    ) -> None:
        super().__init__(mdp, qfunction, bandit)
        self.entropy_scorer = entropy_scorer
        self.low_H_threshold = low_H_threshold
        self.high_H_threshold = high_H_threshold
        self.root_action_to_index = root_action_to_index or {}
        self.rollout_data: list[dict] = []

    def _root_child(self, node):
        root_child = node
        while root_child.parent is not None and root_child.parent.parent is not None:
            root_child = root_child.parent
        return root_child

    def _root_action_index(self, node) -> int | None:
        root_child = self._root_child(node)
        if root_child.parent is None or root_child.action is None:
            return None
        return self.root_action_to_index.get(tuple(root_child.action))

    def _root_action_entropy(self, node) -> float | None:
        if self.entropy_scorer is None:
            return None
        root_child = self._root_child(node)
        if root_child.parent is None or root_child.action is None:
            return None
        candidate_embedding = np.array(root_child.parent.state.conversation) + np.array(root_child.action)
        return self.entropy_scorer.entropy(tuple(candidate_embedding.astype(np.float32)))

    def simulate(self, node, seed=None):
        if seed is not None:
            random.seed(seed)

        record: dict = {
            "rollout_index": len(self.rollout_data),
            "root_action_index": self._root_action_index(node),
            "root_action_entropy": self._root_action_entropy(node),
            "start_depth": node.state.depth,
            "start_entropy": (
                self.entropy_scorer.entropy(node.state.conversation)
                if self.entropy_scorer is not None
                else None
            ),
            "cumulative_reward": 0.0,
            "steps": [],          # list of {depth, entropy} per simulated step
            "n_steps": 0,
            "terminal_entropy": None,
            "early_stop_reason": None,
        }

        state = node.state
        cumulative_reward = 0.0

        while not self.mdp.is_terminal(state):
            actions = self.mdp.get_actions(state)
            if not actions:
                break
            action = random.choice(list(actions))
            state, reward = self.mdp.execute_in_expansion(state, action)
            cumulative_reward += reward

            step_entry: dict = {"depth": state.depth}

            if self.entropy_scorer is not None:
                h = self.entropy_scorer.entropy(state.conversation)
                step_entry["entropy"] = h

                if self.low_H_threshold is not None and h < self.low_H_threshold:
                    record["steps"].append(step_entry)
                    record["early_stop_reason"] = "low_H"
                    break

                if self.high_H_threshold is not None and h > self.high_H_threshold:
                    record["steps"].append(step_entry)
                    record["early_stop_reason"] = "high_H"
                    break

            record["steps"].append(step_entry)

        record["cumulative_reward"] = cumulative_reward
        record["n_steps"] = len(record["steps"])
        if record["steps"] and self.entropy_scorer is not None:
            record["terminal_entropy"] = record["steps"][-1].get("entropy")

        self.rollout_data.append(record)
        return cumulative_reward

    def rollout_summary(self) -> dict:
        """Aggregate statistics across all recorded rollouts."""
        if not self.rollout_data:
            return {}

        terminal_H = [r["terminal_entropy"] for r in self.rollout_data if r["terminal_entropy"] is not None]
        all_step_H = [
            s["entropy"]
            for r in self.rollout_data
            for s in r["steps"]
            if "entropy" in s
        ]
        rewards = [r["cumulative_reward"] for r in self.rollout_data]
        stop_counts = {
            "low_H": sum(1 for r in self.rollout_data if r.get("early_stop_reason") == "low_H"),
            "high_H": sum(1 for r in self.rollout_data if r.get("early_stop_reason") == "high_H"),
            "natural": sum(1 for r in self.rollout_data if r.get("early_stop_reason") is None),
        }

        result: dict = {
            "n_rollouts": len(self.rollout_data),
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "early_stop_counts": stop_counts,
        }
        if terminal_H:
            result["terminal_H_mean"] = float(np.mean(terminal_H))
            result["terminal_H_std"] = float(np.std(terminal_H))
        if all_step_H:
            result["step_H_mean"] = float(np.mean(all_step_H))
            result["step_H_std"] = float(np.std(all_step_H))
        return result
