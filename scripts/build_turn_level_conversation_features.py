#!/usr/bin/env python3
"""Build turn-level conversation features dataset.

For each conversation, compute features incrementally after each message,
recomputing only the features that depend on the newly added turn's role:

  ALWAYS      : Number of Turns, turn-to-turn distance stats, Final Turn Distance from Goal
  ONCE        : Initial Response Distance (frozen after first pair)
  EXPERT turn : Model Self-Similarity, Model Distance from User, Min Model Distance to
                User Prompt, Trend in Model Relevance, Model Adherence to Initial Prompt,
                + all model-to-goal features (using current goal)
  USER turn   : User Self-Consistency, User Distance from Model,
                + all goal features (goal = last user message, so goal just changed)

Output: JSONL with one row per message:
  source_idx, msg_index, role, reward, <26 features>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.ipc as ipc
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

REPO = Path("/home/hyang/mediQ")
DEFAULT_EMB = REPO / (
    "scope_saved/embeddings/"
    "scale_qwen3_4b_branch_d3_hightemp_conf5_train4k_no_reasoning_scope_reward_hf_qwen3_2560_full"
)
DEFAULT_REWARD = REPO / (
    "scope_saved/reward_datasets/"
    "scale_qwen3_4b_branch_d3_hightemp_conf5_train4k_no_reasoning_scope_reward_qwen3_flat_cumulative_full"
)
DEFAULT_OUTPUT = REPO / "results/turn_level_conversation_features/features.jsonl"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _slope(vals: list[float]) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(vals) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(vals))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if abs(den) > 1e-12 else 0.0


def _safe_div(a: float, b: float) -> float:
    return a / b if abs(b) > 1e-12 else 0.0


def _cdist(a: Tensor, b: Tensor) -> float:
    a = F.normalize(a.float().reshape(1, -1), p=2, dim=1)
    b = F.normalize(b.float().reshape(1, -1), p=2, dim=1)
    return float((1.0 - (a @ b.T).clamp(-1, 1)).item())


class IncrementalFeatureState:
    """Maintains conversation features with role-aware cache invalidation."""

    def __init__(self) -> None:
        self.user_embs: list[Tensor] = []
        self.agent_embs: list[Tensor] = []
        self.all_embs: list[Tensor] = []
        self._tt_dists: list[float] = []          # consecutive turn-to-turn distances
        self._model_dist_user: list[float] = []   # agent[i] vs user[i]
        self._user_dist_model: list[float] = []   # user[i] vs agent[i-1]
        self._model_dist_goal: list[float] = []   # agent[i] vs current goal
        self._initial_response_set = False
        self._f: dict[str, float] = {
            "Number of Turns": 0.0,
            "Model Self-Similarity": 0.0,
            "Max Model Self-Similarity": 0.0,
            "Initial Response Distance": 0.0,
            "Avg Model Distance from User": 0.0,
            "Max Model Distance from User": 0.0,
            "Avg User Distance from Model": 0.0,
            "Max User Distance from Model": 0.0,
            "Min Model Distance to User Prompt": 0.0,
            "Trend in Model Relevance": 0.0,
            "Semantic Cohesion": 0.0,
            "Conversation Volatility": 0.0,
            "Max Turn-to-Turn Distance": 0.0,
            "Late Conversation Volatility": 0.0,
            "User Self-Consistency": 0.0,
            "Model Adherence to Goal": 0.0,
            "User Adherence to Goal": 1.0,
            "Min Model Distance to Goal": 0.0,
            "Max Model Distance from Goal": 0.0,
            "Final Model Response to Goal Distance": 0.0,
            "Final Turn Distance from Goal": 0.0,
            "Model Adherence to Initial Prompt": 0.0,
            "Goal vs Initial Prompt Distance": 0.0,
            "Conversation Drift from Goal": 0.0,
            "Trend in Goal Adherence": 0.0,
            "Goal Convergence Ratio": 0.0,
        }

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def add_turn(self, emb: Tensor, role: str) -> dict[str, float]:
        """Add one message embedding ('user' or 'agent'), return updated features."""
        e = F.normalize(emb.float().reshape(-1), p=2, dim=0)

        # Turn-to-turn distance (always)
        if self.all_embs:
            self._tt_dists.append(_cdist(self.all_embs[-1], e))
        self.all_embs.append(e)
        self._f["Number of Turns"] = float(len(self.all_embs))
        self._update_tt_features()

        if role == "user":
            self.user_embs.append(e)
            self._update_user_features()
            self._update_goal_features()          # goal changed → recompute everything goal-related
            # Final turn is now this user message — distance to itself = 0
            self._f["Final Turn Distance from Goal"] = 0.0

        else:  # agent / expert
            self.agent_embs.append(e)

            # Initial Response Distance — set once
            if not self._initial_response_set and self.user_embs:
                self._f["Initial Response Distance"] = _cdist(self.user_embs[0], e)
                self._initial_response_set = True

            # New agent[i] vs user[i] pair
            i = len(self.agent_embs) - 1
            if i < len(self.user_embs):
                d = _cdist(e, self.user_embs[i])
                if len(self._model_dist_user) <= i:
                    self._model_dist_user.append(d)
                else:
                    self._model_dist_user[i] = d

            self._update_expert_features()
            self._update_model_goal_features()    # recompute model-to-goal with current goal

            # Final turn is now this agent message
            if self.user_embs:
                self._f["Final Turn Distance from Goal"] = _cdist(e, self.user_embs[-1])

        return dict(self._f)

    # ------------------------------------------------------------------
    # Private update methods
    # ------------------------------------------------------------------

    def _update_tt_features(self) -> None:
        dists = self._tt_dists
        if not dists:
            return
        avg = sum(dists) / len(dists)
        self._f["Semantic Cohesion"] = 1.0 - min(avg, 1.0)
        self._f["Conversation Volatility"] = (
            sum((x - avg) ** 2 for x in dists) / len(dists)
        ) ** 0.5
        self._f["Max Turn-to-Turn Distance"] = max(dists)
        half = len(dists) // 2
        late = dists[half:]
        if late:
            late_avg = sum(late) / len(late)
            self._f["Late Conversation Volatility"] = (
                sum((x - late_avg) ** 2 for x in late) / len(late)
            ) ** 0.5

    def _update_user_features(self) -> None:
        # User Self-Consistency
        if len(self.user_embs) >= 2:
            ut = torch.stack(self.user_embs)
            sim = ut @ ut.T
            n = sim.shape[0]
            mask = ~torch.eye(n, dtype=torch.bool)
            self._f["User Self-Consistency"] = float(sim[mask].mean().item())

        # User[i] vs Agent[i-1] distance (new user turn adds one new pair)
        i = len(self.user_embs) - 1
        if i >= 1 and i - 1 < len(self.agent_embs):
            d = _cdist(self.user_embs[i], self.agent_embs[i - 1])
            if len(self._user_dist_model) < i:
                self._user_dist_model.append(d)
            else:
                self._user_dist_model[i - 1] = d
        if self._user_dist_model:
            self._f["Avg User Distance from Model"] = sum(self._user_dist_model) / len(self._user_dist_model)
            self._f["Max User Distance from Model"] = max(self._user_dist_model)

    def _update_expert_features(self) -> None:
        # Model Self-Similarity
        if len(self.agent_embs) >= 2:
            at = torch.stack(self.agent_embs)
            sim = at @ at.T
            n = sim.shape[0]
            mask = ~torch.eye(n, dtype=torch.bool)
            off = sim[mask]
            self._f["Model Self-Similarity"] = float(off.mean().item())
            self._f["Max Model Self-Similarity"] = float(off.max().item())

        # Avg/Max Model Distance from User
        if self._model_dist_user:
            self._f["Avg Model Distance from User"] = sum(self._model_dist_user) / len(self._model_dist_user)
            self._f["Max Model Distance from User"] = max(self._model_dist_user)
            self._f["Trend in Model Relevance"] = _slope(self._model_dist_user)

        # Min Model Distance to User Prompt (user_embs[0])
        if self.user_embs:
            d = _cdist(self.agent_embs[-1], self.user_embs[0])
            prev = self._f["Min Model Distance to User Prompt"]
            self._f["Min Model Distance to User Prompt"] = min(prev, d) if self._initial_response_set else d

        # Model Adherence to Initial Prompt
        if self.user_embs:
            init = self.user_embs[0]
            at = torch.stack(self.agent_embs)
            self._f["Model Adherence to Initial Prompt"] = float(
                (at @ init.unsqueeze(0).T).clamp(-1, 1).mean().item()
            )

    def _update_goal_features(self) -> None:
        """Recompute all goal-dependent features. Called when user turn added (goal changed)."""
        if not self.user_embs:
            return
        goal = self.user_embs[-1]
        init = self.user_embs[0]

        self._f["Goal vs Initial Prompt Distance"] = _cdist(goal, init)

        # User Adherence to Goal
        if len(self.user_embs) >= 2:
            ut = torch.stack(self.user_embs)
            sims = (ut @ goal.unsqueeze(0).T).clamp(-1, 1)
            self._f["User Adherence to Goal"] = float(sims.mean().item())
        else:
            self._f["User Adherence to Goal"] = 1.0

        self._update_model_goal_features()

    def _update_model_goal_features(self) -> None:
        """Recompute model-to-goal distances. Called on expert turn or when goal changes."""
        if not self.agent_embs or not self.user_embs:
            return
        goal = self.user_embs[-1]
        at = torch.stack(self.agent_embs)
        sims = (at @ goal.unsqueeze(0).T).squeeze().clamp(-1, 1)
        if sims.dim() == 0:
            sims = sims.unsqueeze(0)
        dists = (1.0 - sims).tolist()

        self._model_dist_goal = dists
        self._f["Model Adherence to Goal"] = float(sims.mean().item())
        self._f["Min Model Distance to Goal"] = min(dists)
        self._f["Max Model Distance from Goal"] = max(dists)
        self._f["Final Model Response to Goal Distance"] = dists[-1]

        if len(dists) >= 2:
            self._f["Trend in Goal Adherence"] = -_slope(dists)
            d0, dl = dists[0], dists[-1]
            self._f["Goal Convergence Ratio"] = _safe_div(d0 - dl, d0)
        mean_d = sum(dists) / len(dists)
        self._f["Conversation Drift from Goal"] = (
            sum((x - mean_d) ** 2 for x in dists) / len(dists)
        ) ** 0.5


# ------------------------------------------------------------------
# Dataset I/O
# ------------------------------------------------------------------

def iter_emb_batches(dataset_dir: Path):
    state = json.loads((dataset_dir / "state.json").read_text())
    for item in state["_data_files"]:
        reader = ipc.open_stream(str(dataset_dir / item["filename"]))
        yield from reader


def load_rewards_by_source(dataset_dir: Path) -> dict[int, list[dict]]:
    """Returns {source_idx: [rows sorted by msg_index]}."""
    state = json.loads((dataset_dir / "state.json").read_text())
    by_source: dict[int, list[dict]] = {}
    for item in state["_data_files"]:
        reader = ipc.open_stream(str(dataset_dir / item["filename"]))
        for batch in reader:
            names = [n for n in batch.schema.names if n != "embedding"]
            for i in range(batch.num_rows):
                row = {n: batch.column(n)[i].as_py() for n in names}
                src = int(row["source_idx"])
                by_source.setdefault(src, []).append(row)
    for rows in by_source.values():
        rows.sort(key=lambda r: r["msg_index"])
    return by_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emb-dataset", type=Path, default=DEFAULT_EMB)
    parser.add_argument("--reward-dataset", type=Path, default=DEFAULT_REWARD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-conversations", type=int, default=None)
    args = parser.parse_args()

    print("Loading reward metadata...", flush=True)
    rewards_by_source = load_rewards_by_source(args.reward_dataset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    conv_idx = 0
    written = 0

    with args.output.open("w", encoding="utf-8") as out_f:
        for batch in tqdm(iter_emb_batches(args.emb_dataset), desc="Batches"):
            for i in range(batch.num_rows):
                if args.max_conversations is not None and conv_idx >= args.max_conversations:
                    break

                reward_rows = rewards_by_source.get(conv_idx)
                if reward_rows is None:
                    conv_idx += 1
                    continue

                emb_matrix = torch.tensor(
                    np.array(batch.column("embeddings")[i].as_py()), dtype=torch.float32
                )

                if emb_matrix.shape[0] != len(reward_rows):
                    conv_idx += 1
                    continue

                state = IncrementalFeatureState()
                for reward_row in reward_rows:
                    msg_idx = reward_row["msg_index"]
                    role = "user" if msg_idx % 2 == 0 else "agent"
                    emb = emb_matrix[msg_idx]
                    feats = state.add_turn(emb, role)

                    row: dict[str, Any] = {
                        "source_idx": conv_idx,
                        "patient_id": reward_row["patient_id"],
                        "branch_id": reward_row["branch_id"],
                        "msg_index": msg_idx,
                        "role": role,
                        "reward": reward_row["reward"],
                        "is_final_turn": reward_row["is_final_turn"],
                        **feats,
                    }
                    out_f.write(json.dumps(row) + "\n")
                    written += 1

                conv_idx += 1

    print(f"Wrote {written} rows ({conv_idx} conversations) to {args.output}")


if __name__ == "__main__":
    main()
