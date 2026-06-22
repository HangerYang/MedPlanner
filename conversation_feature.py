#!/usr/bin/env python3
"""Extract conversation-level features from per-turn embeddings (.pt file).

Features are computed per conversation. Run:
  python conversation_feature.py --pt-path outputs/wildfeedback_per_turn_4b.pt --output outputs/features.json

Feature categories:
- Inefficiency and Repetition: Number of Turns, Model Self-Similarity, Max Model Self-Similarity
- Semantic Cohesion and Relevance: Initial Response Distance, Avg/Max Model Distance from User,
  Avg/Max User Distance from Model, Min Model Distance to User Prompt, Trend in Model Relevance,
  Semantic Cohesion, Conversation Volatility, Max Turn-to-Turn Distance,
  Late Conversation Volatility, User Self-Consistency
- Goal Orientation: Model/User Adherence to Goal, Min/Max Model Distance to Goal,
  Final Turn/Model Response Distance from Goal, Model Adherence to Initial Prompt,
  Goal vs Initial Prompt Distance, Conversation Drift from Goal,
  Trend in Goal Adherence, Goal Convergence Ratio
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Incremental feature computation (role-aware cache invalidation)
# ---------------------------------------------------------------------------

class IncrementalFeatureState:
    """Maintains conversation features incrementally with role-aware updates.

    Call add_turn(embedding, role) after each message ('user' or 'agent').
    Only the features that depend on the new turn's role are recomputed.
    """

    FEATURE_KEYS = [
        "Number of Turns", "Model Self-Similarity", "Max Model Self-Similarity",
        "Initial Response Distance", "Avg Model Distance from User",
        "Max Model Distance from User", "Avg User Distance from Model",
        "Max User Distance from Model", "Min Model Distance to User Prompt",
        "Trend in Model Relevance", "Semantic Cohesion", "Conversation Volatility",
        "Max Turn-to-Turn Distance", "Late Conversation Volatility",
        "User Self-Consistency", "Model Adherence to Goal", "User Adherence to Goal",
        "Min Model Distance to Goal", "Max Model Distance from Goal",
        "Final Model Response to Goal Distance", "Final Turn Distance from Goal",
        "Model Adherence to Initial Prompt", "Goal vs Initial Prompt Distance",
        "Conversation Drift from Goal", "Trend in Goal Adherence", "Goal Convergence Ratio",
    ]

    def __init__(self) -> None:
        self.user_embs: list[Tensor] = []
        self.agent_embs: list[Tensor] = []
        self.all_embs: list[Tensor] = []
        self._tt_dists: list[float] = []
        self._model_dist_user: list[float] = []
        self._user_dist_model: list[float] = []
        self._initial_response_set = False
        self._f: dict[str, float] = {k: 0.0 for k in self.FEATURE_KEYS}
        self._f["User Adherence to Goal"] = 1.0

    def add_turn(self, emb: Tensor, role: str) -> dict[str, float]:
        e = F.normalize(emb.float().reshape(-1), p=2, dim=0)
        if self.all_embs:
            self._tt_dists.append(self._cd(self.all_embs[-1], e))
        self.all_embs.append(e)
        self._f["Number of Turns"] = float(len(self.all_embs))
        self._update_tt()

        if role == "user":
            self.user_embs.append(e)
            self._update_user()
            self._update_goal()
            self._f["Final Turn Distance from Goal"] = 0.0
        else:
            self.agent_embs.append(e)
            if not self._initial_response_set and self.user_embs:
                self._f["Initial Response Distance"] = self._cd(self.user_embs[0], e)
                self._initial_response_set = True
            i = len(self.agent_embs) - 1
            if i < len(self.user_embs):
                d = self._cd(e, self.user_embs[i])
                if len(self._model_dist_user) <= i:
                    self._model_dist_user.append(d)
                else:
                    self._model_dist_user[i] = d
            self._update_expert()
            self._update_model_goal()
            if self.user_embs:
                self._f["Final Turn Distance from Goal"] = self._cd(e, self.user_embs[-1])

        return dict(self._f)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _cd(a: Tensor, b: Tensor) -> float:
        a = F.normalize(a.float().reshape(1, -1), p=2, dim=1)
        b = F.normalize(b.float().reshape(1, -1), p=2, dim=1)
        return float((1.0 - (a @ b.T).clamp(-1, 1)).item())

    @staticmethod
    def _slope(vals: list[float]) -> float:
        n = len(vals)
        if n < 2:
            return 0.0
        xm = (n - 1) / 2.0
        ym = sum(vals) / n
        num = sum((i - xm) * (v - ym) for i, v in enumerate(vals))
        den = sum((i - xm) ** 2 for i in range(n))
        return num / den if abs(den) > 1e-12 else 0.0

    # ------------------------------------------------------------------ update groups
    def _update_tt(self) -> None:
        d = self._tt_dists
        if not d:
            return
        avg = sum(d) / len(d)
        self._f["Semantic Cohesion"] = 1.0 - min(avg, 1.0)
        self._f["Conversation Volatility"] = (sum((x - avg) ** 2 for x in d) / len(d)) ** 0.5
        self._f["Max Turn-to-Turn Distance"] = max(d)
        late = d[len(d) // 2:]
        if late:
            la = sum(late) / len(late)
            self._f["Late Conversation Volatility"] = (sum((x - la) ** 2 for x in late) / len(late)) ** 0.5

    def _update_user(self) -> None:
        if len(self.user_embs) >= 2:
            ut = torch.stack(self.user_embs)
            sim = ut @ ut.T
            mask = ~torch.eye(sim.shape[0], dtype=torch.bool)
            self._f["User Self-Consistency"] = float(sim[mask].mean().item())
        i = len(self.user_embs) - 1
        if i >= 1 and i - 1 < len(self.agent_embs):
            d = self._cd(self.user_embs[i], self.agent_embs[i - 1])
            if len(self._user_dist_model) < i:
                self._user_dist_model.append(d)
            else:
                self._user_dist_model[i - 1] = d
        if self._user_dist_model:
            self._f["Avg User Distance from Model"] = sum(self._user_dist_model) / len(self._user_dist_model)
            self._f["Max User Distance from Model"] = max(self._user_dist_model)

    def _update_expert(self) -> None:
        if len(self.agent_embs) >= 2:
            at = torch.stack(self.agent_embs)
            sim = at @ at.T
            mask = ~torch.eye(sim.shape[0], dtype=torch.bool)
            off = sim[mask]
            self._f["Model Self-Similarity"] = float(off.mean().item())
            self._f["Max Model Self-Similarity"] = float(off.max().item())
        if self._model_dist_user:
            self._f["Avg Model Distance from User"] = sum(self._model_dist_user) / len(self._model_dist_user)
            self._f["Max Model Distance from User"] = max(self._model_dist_user)
            self._f["Trend in Model Relevance"] = self._slope(self._model_dist_user)
        if self.user_embs:
            d = self._cd(self.agent_embs[-1], self.user_embs[0])
            prev = self._f["Min Model Distance to User Prompt"]
            self._f["Min Model Distance to User Prompt"] = min(prev, d) if self._initial_response_set else d
            at = torch.stack(self.agent_embs)
            init = self.user_embs[0]
            self._f["Model Adherence to Initial Prompt"] = float(
                (at @ init.unsqueeze(0).T).clamp(-1, 1).mean().item()
            )

    def _update_goal(self) -> None:
        if not self.user_embs:
            return
        goal, init = self.user_embs[-1], self.user_embs[0]
        self._f["Goal vs Initial Prompt Distance"] = self._cd(goal, init)
        if len(self.user_embs) >= 2:
            ut = torch.stack(self.user_embs)
            self._f["User Adherence to Goal"] = float(
                (ut @ goal.unsqueeze(0).T).clamp(-1, 1).mean().item()
            )
        else:
            self._f["User Adherence to Goal"] = 1.0
        self._update_model_goal()

    def _update_model_goal(self) -> None:
        if not self.agent_embs or not self.user_embs:
            return
        goal = self.user_embs[-1]
        at = torch.stack(self.agent_embs)
        sims = (at @ goal.unsqueeze(0).T).squeeze().clamp(-1, 1)
        if sims.dim() == 0:
            sims = sims.unsqueeze(0)
        dists = (1.0 - sims).tolist()
        self._f["Model Adherence to Goal"] = float(sims.mean().item())
        self._f["Min Model Distance to Goal"] = min(dists)
        self._f["Max Model Distance from Goal"] = max(dists)
        self._f["Final Model Response to Goal Distance"] = dists[-1]
        if len(dists) >= 2:
            self._f["Trend in Goal Adherence"] = -self._slope(dists)
            d0, dl = dists[0], dists[-1]
            self._f["Goal Convergence Ratio"] = (d0 - dl) / d0 if abs(d0) > 1e-12 else 0.0
        md = sum(dists) / len(dists)
        self._f["Conversation Drift from Goal"] = (sum((x - md) ** 2 for x in dists) / len(dists)) ** 0.5


def cosine_similarity_matrix(emb: Tensor) -> Tensor:
    """(n, d) -> (n, n) pairwise cosine sim (emb assumed L2-normalized)."""
    return emb @ emb.T


def cosine_distance(emb_a: Tensor, emb_b: Tensor) -> Tensor:
    """Compute 1 - cosine similarity between rows. Returns scalar or vector."""
    if emb_a.dim() == 1:
        emb_a = emb_a.unsqueeze(0)
    if emb_b.dim() == 1:
        emb_b = emb_b.unsqueeze(0)
    emb_a = F.normalize(emb_a.float(), p=2, dim=1)
    emb_b = F.normalize(emb_b.float(), p=2, dim=1)
    sim = (emb_a @ emb_b.T).squeeze()
    return (1.0 - sim.clamp(-1, 1)).float()


def _slope(x: list[float], y: list[float]) -> float:
    """Linear regression slope. Returns 0 if insufficient points."""
    n = len(x)
    if n < 2:
        return 0.0
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    num = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    den = sum((x[i] - x_mean) ** 2 for i in range(n))
    if abs(den) < 1e-12:
        return 0.0
    return num / den


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b is None or abs(b) < 1e-12:
        return default
    return a / b


def compute_features_for_conversation(
    user_embs: list[Tensor],
    agent_embs: list[Tensor],
    all_embs_ordered: list[Tensor],
) -> dict[str, float]:
    """Compute all features for one conversation. Assumes turns alternate User, Agent."""
    out: dict[str, float] = {}
    device = user_embs[0].device if user_embs else agent_embs[0].device if agent_embs else None
    dtype = torch.float32

    def to_scalar(t: Tensor) -> float:
        if t.numel() == 1:
            return float(t.item())
        return float(t.mean().item())

    # ---- Inefficiency and Repetition ----
    n_turns = len(user_embs) + len(agent_embs)
    out["Number of Turns"] = float(n_turns)

    if len(agent_embs) >= 2:
        agent_t = torch.stack([e.float().squeeze() for e in agent_embs])
        if agent_t.dim() == 1:
            agent_t = agent_t.unsqueeze(0)
        agent_t = F.normalize(agent_t, p=2, dim=1)
        sim_m = cosine_similarity_matrix(agent_t)
        n_a = sim_m.shape[0]
        mask = ~torch.eye(n_a, dtype=torch.bool, device=sim_m.device)
        off = sim_m[mask]
        out["Model Self-Similarity"] = float(off.mean().item())
        out["Max Model Self-Similarity"] = float(off.max().item())
    else:
        out["Model Self-Similarity"] = 0.0
        out["Max Model Self-Similarity"] = 0.0

    # ---- Semantic Cohesion and Relevance ----
    if user_embs and agent_embs:
        u0 = user_embs[0].float().squeeze()
        a0 = agent_embs[0].float().squeeze()
        if u0.dim() == 0:
            u0 = u0.unsqueeze(0)
        if a0.dim() == 0:
            a0 = a0.unsqueeze(0)
        out["Initial Response Distance"] = to_scalar(cosine_distance(u0.unsqueeze(0), a0.unsqueeze(0)))
    else:
        out["Initial Response Distance"] = 0.0

    # Model distance from preceding user (per Agent turn)
    model_dist_from_user: list[float] = []
    for i, a_emb in enumerate(agent_embs):
        if i < len(user_embs):
            u = user_embs[i].float().squeeze()
            a = a_emb.float().squeeze()
            d = cosine_distance(
                u.unsqueeze(0) if u.dim() == 1 else u.unsqueeze(0),
                a.unsqueeze(0) if a.dim() == 1 else a.unsqueeze(0),
            )
            model_dist_from_user.append(to_scalar(d))
    if model_dist_from_user:
        out["Avg Model Distance from User"] = sum(model_dist_from_user) / len(model_dist_from_user)
        out["Max Model Distance from User"] = max(model_dist_from_user)
    else:
        out["Avg Model Distance from User"] = 0.0
        out["Max Model Distance from User"] = 0.0

    # User distance from preceding model (User turns after first)
    user_dist_from_model: list[float] = []
    for i in range(1, len(user_embs)):
        if i <= len(agent_embs):
            u = user_embs[i].float().squeeze()
            a = agent_embs[i - 1].float().squeeze()
            d = cosine_distance(
                u.unsqueeze(0) if u.dim() == 1 else u.unsqueeze(0),
                a.unsqueeze(0) if a.dim() == 1 else a.unsqueeze(0),
            )
            user_dist_from_model.append(to_scalar(d))
    if user_dist_from_model:
        out["Avg User Distance from Model"] = sum(user_dist_from_model) / len(user_dist_from_model)
        out["Max User Distance from Model"] = max(user_dist_from_model)
    else:
        out["Avg User Distance from Model"] = 0.0
        out["Max User Distance from Model"] = 0.0

    # Min Model Distance to User Prompt (first user)
    if user_embs and agent_embs:
        u0 = user_embs[0].float().squeeze()
        if u0.dim() == 0:
            u0 = u0.unsqueeze(0)
        u0_ = u0.unsqueeze(0) if u0.dim() == 1 else u0
        dists = [
            to_scalar(cosine_distance(a.float().squeeze().unsqueeze(0), u0_))
            for a in agent_embs
        ]
        out["Min Model Distance to User Prompt"] = min(dists) if dists else 0.0
    else:
        out["Min Model Distance to User Prompt"] = 0.0

    # Trend in Model Relevance (slope of model_dist_from_user over turn index)
    if len(model_dist_from_user) >= 2:
        out["Trend in Model Relevance"] = _slope(list(range(len(model_dist_from_user))), model_dist_from_user)
    else:
        out["Trend in Model Relevance"] = 0.0

    # Turn-to-turn distances (consecutive turns)
    turn_to_turn_dists: list[float] = []
    for i in range(len(all_embs_ordered) - 1):
        a = all_embs_ordered[i].float().squeeze()
        b = all_embs_ordered[i + 1].float().squeeze()
        d = cosine_distance(
            a.unsqueeze(0) if a.dim() == 1 else a.unsqueeze(0),
            b.unsqueeze(0) if b.dim() == 1 else b.unsqueeze(0),
        )
        turn_to_turn_dists.append(to_scalar(d))

    if turn_to_turn_dists:
        avg_tt = sum(turn_to_turn_dists) / len(turn_to_turn_dists)
        out["Semantic Cohesion"] = 1.0 - min(avg_tt, 1.0)  # higher = more cohesive
        out["Conversation Volatility"] = (
            (sum((x - avg_tt) ** 2 for x in turn_to_turn_dists) / len(turn_to_turn_dists)) ** 0.5
        )
        out["Max Turn-to-Turn Distance"] = max(turn_to_turn_dists)
        half = len(turn_to_turn_dists) // 2
        late_dists = turn_to_turn_dists[half:]
        if late_dists:
            late_avg = sum(late_dists) / len(late_dists)
            out["Late Conversation Volatility"] = (
                (sum((x - late_avg) ** 2 for x in late_dists) / len(late_dists)) ** 0.5
            )
        else:
            out["Late Conversation Volatility"] = 0.0
    else:
        out["Semantic Cohesion"] = 0.0
        out["Conversation Volatility"] = 0.0
        out["Max Turn-to-Turn Distance"] = 0.0
        out["Late Conversation Volatility"] = 0.0

    # User Self-Consistency (pairwise similarity of User turns)
    if len(user_embs) >= 2:
        user_t = torch.stack([e.float().squeeze() for e in user_embs])
        if user_t.dim() == 1:
            user_t = user_t.unsqueeze(0)
        user_t = F.normalize(user_t, p=2, dim=1)
        sim_u = cosine_similarity_matrix(user_t)
        n_u = sim_u.shape[0]
        mask = ~torch.eye(n_u, dtype=torch.bool, device=sim_u.device)
        off_u = sim_u[mask]
        out["User Self-Consistency"] = float(off_u.mean().item())
    else:
        out["User Self-Consistency"] = 0.0

    # ---- Goal Orientation ----
    # Goal = last User message; Initial prompt = first User message
    if not user_embs:
        for k in [
            "Model Adherence to Goal", "User Adherence to Goal", "Min Model Distance to Goal",
            "Max Model Distance from Goal", "Final Turn Distance from Goal",
            "Final Model Response to Goal Distance", "Model Adherence to Initial Prompt",
            "Goal vs Initial Prompt Distance", "Conversation Drift from Goal",
            "Trend in Goal Adherence", "Goal Convergence Ratio",
        ]:
            out[k] = 0.0
        return out

    goal_emb = user_embs[-1].float().squeeze()
    if goal_emb.dim() == 0:
        goal_emb = goal_emb.unsqueeze(0)
    goal_emb = F.normalize(goal_emb.unsqueeze(0), p=2, dim=1)

    init_emb = user_embs[0].float().squeeze()
    if init_emb.dim() == 0:
        init_emb = init_emb.unsqueeze(0)
    init_emb = F.normalize(init_emb.unsqueeze(0), p=2, dim=1)

    # Model Adherence to Goal (alignment = 1 - distance, higher = better)
    if agent_embs:
        agent_t = torch.stack([e.float().squeeze() for e in agent_embs])
        if agent_t.dim() == 1:
            agent_t = agent_t.unsqueeze(0)
        agent_t = F.normalize(agent_t, p=2, dim=1)
        sim_to_goal = (agent_t @ goal_emb.T).squeeze()
        align = sim_to_goal.clamp(-1, 1)
        out["Model Adherence to Goal"] = float(align.mean().item())
        dists_to_goal = 1.0 - align
        out["Min Model Distance to Goal"] = float(dists_to_goal.min().item())
        out["Max Model Distance from Goal"] = float(dists_to_goal.max().item())
        out["Model Adherence to Initial Prompt"] = float(
            (agent_t @ init_emb.T).squeeze().clamp(-1, 1).mean().item()
        )
        raw = dists_to_goal.tolist()
        dist_list = raw if isinstance(raw, list) else [raw]
        if not dist_list:
            out["Final Model Response to Goal Distance"] = 0.0
            out["Trend in Goal Adherence"] = 0.0
            out["Goal Convergence Ratio"] = 0.0
            out["Conversation Drift from Goal"] = 0.0
        else:
            out["Final Model Response to Goal Distance"] = float(dist_list[-1])
            if len(dist_list) >= 2:
                out["Trend in Goal Adherence"] = -_slope(list(range(len(dist_list))), dist_list)
            else:
                out["Trend in Goal Adherence"] = 0.0
            d_first, d_last = dist_list[0], dist_list[-1]
            out["Goal Convergence Ratio"] = _safe_div(d_first - d_last, d_first, 0.0)
            mean_d = sum(dist_list) / len(dist_list)
            out["Conversation Drift from Goal"] = (
                (sum((x - mean_d) ** 2 for x in dist_list) / len(dist_list)) ** 0.5
            )
    else:
        for k in [
            "Model Adherence to Goal", "Min Model Distance to Goal", "Max Model Distance from Goal",
            "Final Model Response to Goal Distance", "Model Adherence to Initial Prompt",
            "Trend in Goal Adherence", "Goal Convergence Ratio", "Conversation Drift from Goal",
        ]:
            out[k] = 0.0

    # User Adherence to Goal
    if len(user_embs) >= 2:
        user_t = torch.stack([e.float().squeeze() for e in user_embs])
        if user_t.dim() == 1:
            user_t = user_t.unsqueeze(0)
        user_t = F.normalize(user_t, p=2, dim=1)
        sim_u_goal = (user_t @ goal_emb.T).squeeze()
        out["User Adherence to Goal"] = float(sim_u_goal.clamp(-1, 1).mean().item())
    else:
        out["User Adherence to Goal"] = 1.0  # single user turn

    # Final turn distance to goal (last turn in conversation)
    if all_embs_ordered:
        last = all_embs_ordered[-1].float().squeeze()
        if last.dim() == 0:
            last = last.unsqueeze(0)
        last_ = F.normalize(last.unsqueeze(0), p=2, dim=1)
        out["Final Turn Distance from Goal"] = to_scalar(cosine_distance(last_, goal_emb))
    else:
        out["Final Turn Distance from Goal"] = 0.0

    # Goal vs Initial Prompt Distance
    out["Goal vs Initial Prompt Distance"] = to_scalar(cosine_distance(goal_emb, init_emb))

    return out


def load_pt_and_split_by_role(
    pt_path: Path,
) -> tuple[list[list[Tensor]], list[list[Tensor]], list[list[Tensor]], list[Any]]:
    """Load .pt, split embeddings by conversation and role. Returns (user_per_conv, agent_per_conv, all_ordered_per_conv, metadata_per_conv)."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    emb = data["embeddings"]
    if emb.dtype in (torch.bfloat16, torch.float16):
        emb = emb.float()
    conv_offsets = data["conversation_offsets"]
    turn_indices = data["turn_indices"]
    turn_metadata = data.get("turn_metadata")

    user_per_conv: list[list[Tensor]] = []
    agent_per_conv: list[list[Tensor]] = []
    all_ordered_per_conv: list[list[Tensor]] = []
    metadata_per_conv: list[Any] = []

    for conv_idx, (start, end) in enumerate(conv_offsets):
        user_embs: list[Tensor] = []
        agent_embs: list[Tensor] = []
        all_ordered: list[Tensor] = []

        for i in range(start, end):
            e = emb[i]
            turn_idx = turn_indices[i][1] if i < len(turn_indices) else (i - start)
            role = "User" if turn_idx % 2 == 0 else "Agent"
            if turn_metadata and i < len(turn_metadata):
                role = turn_metadata[i].get("Role", role)
            all_ordered.append(e)
            if role in ("User", "user"):
                user_embs.append(e)
            else:
                agent_embs.append(e)

        user_per_conv.append(user_embs)
        agent_per_conv.append(agent_embs)
        all_ordered_per_conv.append(all_ordered)
        metadata_per_conv.append(
            turn_metadata[start:end] if turn_metadata else [{"turn_idx": i - start} for i in range(start, end)]
        )

    return user_per_conv, agent_per_conv, all_ordered_per_conv, metadata_per_conv


def load_embeddings_with_index_jsonl(
    jsonl_path: Path,
) -> tuple[list[list[Tensor]], list[list[Tensor]], list[list[Tensor]], list[dict[str, Any]]]:
    """Load embeddings_with_index.jsonl. Groups by hash_id. For each conversation with chosen+rejected,
    produces TWO rows: chosen (satisfaction=1) and rejected (satisfaction=0).
    Returns (user_per_conv, agent_per_conv, all_ordered_per_conv, metadata_per_conv).
    metadata includes satisfaction_score, hash_id, original_index.
    """
    from collections import defaultdict

    groups: dict[str, dict] = defaultdict(lambda: {"history": [], "chosen": [], "rejected": []})
    orig_index_by_hash: dict[str, int] = {}

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading embeddings", unit="lines"):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            h = item.get("hash_id")
            if not h:
                continue
            emb = item.get("embedding")
            if emb is None or not isinstance(emb, (list, tuple)):
                continue
            try:
                e = torch.tensor(emb, dtype=torch.float32)
            except (TypeError, ValueError):
                continue
            t = item.get("type", "")
            ti = item.get("turn_index", 0)
            orig_index_by_hash[h] = item.get("original_index", 0)

            if t == "history":
                groups[h]["history"].append((ti, e))
            elif t == "chosen":
                groups[h]["chosen"].append((ti, e))
            elif t == "rejected":
                groups[h]["rejected"].append((ti, e))

    user_per_conv: list[list[Tensor]] = []
    agent_per_conv: list[list[Tensor]] = []
    all_ordered_per_conv: list[list[Tensor]] = []
    metadata_per_conv: list[dict[str, Any]] = []

    for h, g in tqdm(groups.items(), desc="Building conversations"):
        hist = sorted(g["history"], key=lambda x: (x[0], 0))
        chosen_list = sorted(g["chosen"], key=lambda x: (x[0], 0))
        rejected_list = sorted(g["rejected"], key=lambda x: (x[0], 0))
        if not chosen_list or not rejected_list:
            continue

        # History: 2 per turn (user, agent). Flatten by turn_index.
        hist_by_turn: dict[int, list[Tensor]] = defaultdict(list)
        for ti, emb in hist:
            hist_by_turn[ti].append(emb)
        turns_sorted = sorted(hist_by_turn.keys())
        hist_user: list[Tensor] = []
        hist_agent: list[Tensor] = []
        hist_all: list[Tensor] = []
        for ti in turns_sorted:
            embs = hist_by_turn[ti]
            if len(embs) >= 2:
                hist_user.append(embs[0])
                hist_agent.append(embs[1])
                hist_all.extend(embs)
            elif len(embs) == 1:
                hist_user.append(embs[0])
                hist_agent.append(embs[0])
                hist_all.append(embs[0])

        # Final turn: chosen has (user_prompt, chosen_resp), rejected has (user_prompt, rejected_resp)
        user4 = chosen_list[0][1] if chosen_list else rejected_list[0][1]
        chosen_resp = chosen_list[1][1] if len(chosen_list) >= 2 else chosen_list[0][1]
        rejected_resp = rejected_list[1][1] if len(rejected_list) >= 2 else rejected_list[0][1]

        orig_idx = orig_index_by_hash.get(h, 0)

        # Chosen variant (satisfaction=1)
        u_chosen = hist_user + [user4]
        a_chosen = hist_agent + [chosen_resp]
        all_chosen = hist_all + [user4, chosen_resp]
        user_per_conv.append(u_chosen)
        agent_per_conv.append(a_chosen)
        all_ordered_per_conv.append(all_chosen)
        metadata_per_conv.append({
            "satisfaction_score": 1,
            "hash_id": h,
            "original_index": orig_idx,
            "variant": "chosen",
        })

        # Rejected variant (satisfaction=0)
        u_rejected = hist_user + [user4]
        a_rejected = hist_agent + [rejected_resp]
        all_rejected = hist_all + [user4, rejected_resp]
        user_per_conv.append(u_rejected)
        agent_per_conv.append(a_rejected)
        all_ordered_per_conv.append(all_rejected)
        metadata_per_conv.append({
            "satisfaction_score": 0,
            "hash_id": h,
            "original_index": orig_idx,
            "variant": "rejected",
        })

    return user_per_conv, agent_per_conv, all_ordered_per_conv, metadata_per_conv


def load_jsonl_with_embeddings(
    jsonl_path: Path,
    turns_field: str = "turns",
    embedding_field: str = "embedding",
) -> tuple[list[list[Tensor]], list[list[Tensor]], list[list[Tensor]], list[Any]]:
    """Load JSONL where each turn has 'embedding'. Returns same format as load_pt_and_split_by_role."""
    user_per_conv: list[list[Tensor]] = []
    agent_per_conv: list[list[Tensor]] = []
    all_ordered_per_conv: list[list[Tensor]] = []
    metadata_per_conv: list[Any] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            turns = item.get(turns_field) or []

            user_embs: list[Tensor] = []
            agent_embs: list[Tensor] = []
            all_ordered: list[Tensor] = []
            meta: list[dict] = []

            for t in turns:
                if not isinstance(t, dict):
                    continue
                emb = t.get(embedding_field)
                if emb is None:
                    continue
                e = torch.tensor(emb, dtype=torch.float32)
                all_ordered.append(e)
                role = t.get("Role", "User" if len(all_ordered) % 2 == 1 else "Agent")
                if role in ("User", "user"):
                    user_embs.append(e)
                else:
                    agent_embs.append(e)
                meta.append({k: v for k, v in t.items() if k != embedding_field})

            user_per_conv.append(user_embs)
            agent_per_conv.append(agent_embs)
            all_ordered_per_conv.append(all_ordered)
            metadata_per_conv.append(meta)

    return user_per_conv, agent_per_conv, all_ordered_per_conv, metadata_per_conv


def join_features_with_reward(
    features_jsonl: Path,
    reward_dataset_path: Path,
    output_path: Path,
) -> None:
    """Replace reward dataset embeddings with conversation features from features.jsonl.
    Joins on (patient_id, branch_id, msg_index == turn_idx)."""
    import pandas as pd
    from datasets import load_from_disk

    print("Loading reward dataset (dropping embeddings)...")
    rds = load_from_disk(str(reward_dataset_path))
    rds = rds.remove_columns(["embedding"])
    rdf = rds.to_pandas()
    print(f"  -> {len(rdf)} reward rows loaded")

    print("Loading features...")
    chunks = []
    chunk_size = 100_000
    with tqdm(total=887_974, desc="Reading features.jsonl", unit="rows") as pbar:
        for chunk in pd.read_json(features_jsonl, lines=True, chunksize=chunk_size):
            chunks.append(chunk)
            pbar.update(len(chunk))
    fdf = pd.concat(chunks, ignore_index=True)
    print(f"  -> {len(fdf)} feature rows loaded")

    merged = rdf.merge(
        fdf.rename(columns={"turn_idx": "msg_index"}),
        on=["patient_id", "branch_id", "msg_index"],
        how="inner",
    )

    assert len(merged) == len(rdf) == len(fdf), \
        f"Row count mismatch after merge: {len(merged)} vs {len(rdf)} reward vs {len(fdf)} features"
    assert merged["reward"].isna().sum() == 0, "Missing rewards after merge"

    print(f"Merge OK: {len(merged)} rows, 0 misses")

    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / "features_with_reward.csv"
    merged.to_csv(out_file, index=False)
    print(f"Saved -> {out_file}")


def compute_per_turn_features_hf(dataset_path: Path, output_path: Path) -> None:
    """Load HF dataset, replay each conversation through IncrementalFeatureState,
    write one JSONL row per turn with all features at that point in the conversation."""
    from datasets import load_from_disk

    ds = load_from_disk(str(dataset_path))
    ds = ds.sort(["patient_id", "branch_id", "turn_idx"])

    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / "features.jsonl"

    total = 0
    prev_key: tuple | None = None
    state: IncrementalFeatureState | None = None

    with out_file.open("w", encoding="utf-8") as f:
        for row in tqdm(ds, desc="Computing per-turn features", total=len(ds)):
            key = (row["patient_id"], row["branch_id"])
            if key != prev_key:
                state = IncrementalFeatureState()
                prev_key = key

            emb = torch.tensor(row["embedding"], dtype=torch.float32)
            role_raw = row["role"]
            role = "user" if role_raw == "user" else "agent"

            feats = state.add_turn(emb, role)
            out_row = {
                "patient_id": row["patient_id"],
                "branch_id": row["branch_id"],
                "turn_idx": row["turn_idx"],
                "role": role_raw,
                **feats,
            }
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            total += 1

    print(f"Wrote {total} per-turn feature rows to {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract conversation features from per-turn embeddings (.pt or JSONL with embeddings)"
    )
    inp = parser.add_mutually_exclusive_group(required=True)
    inp.add_argument("--pt-path", type=Path, help="Path to .pt file (per_turn embeddings)")
    inp.add_argument("--jsonl-with-embeddings", type=Path, help="JSONL where each turn has 'embedding' field")
    inp.add_argument(
        "--embeddings-with-index",
        type=Path,
        help="embeddings_with_index.jsonl (chosen=1, rejected=0 satisfaction); outputs 2 rows per conversation",
    )
    inp.add_argument(
        "--hf-dataset",
        type=Path,
        help="HuggingFace dataset dir with patient_id/branch_id/turn_idx/role/embedding columns; outputs one row per turn",
    )
    inp.add_argument(
        "--join-reward",
        type=Path,
        help="Join features.jsonl with a reward HF dataset; requires --features-jsonl",
    )
    parser.add_argument("--features-jsonl", type=Path, help="features.jsonl to join (used with --join-reward)")
    parser.add_argument("--output", type=Path, required=True, help="Output dir or JSON/JSONL path")
    parser.add_argument("--jsonl", action="store_true", help="Output JSONL (one object per line)")
    parser.add_argument("--max-convos", type=int, default=None, help="Limit number of conversations")
    parser.add_argument("--turns-field", default="turns", help="Turns field when using --jsonl-with-embeddings")
    args = parser.parse_args()

    if args.hf_dataset is not None:
        compute_per_turn_features_hf(args.hf_dataset, args.output)
        return

    if args.join_reward is not None:
        features_jsonl = args.features_jsonl or (args.output / "features.jsonl")
        join_features_with_reward(features_jsonl, args.join_reward, args.output)
        return

    if args.pt_path is not None:
        user_per_conv, agent_per_conv, all_ordered_per_conv, metadata_per_conv = load_pt_and_split_by_role(
            args.pt_path
        )
    elif args.embeddings_with_index is not None:
        user_per_conv, agent_per_conv, all_ordered_per_conv, metadata_per_conv = load_embeddings_with_index_jsonl(
            args.embeddings_with_index
        )
    else:
        user_per_conv, agent_per_conv, all_ordered_per_conv, metadata_per_conv = load_jsonl_with_embeddings(
            args.jsonl_with_embeddings, turns_field=args.turns_field
        )
    if args.max_convos is not None:
        user_per_conv = user_per_conv[: args.max_convos]
        agent_per_conv = agent_per_conv[: args.max_convos]
        all_ordered_per_conv = all_ordered_per_conv[: args.max_convos]
        metadata_per_conv = metadata_per_conv[: args.max_convos]

    results: list[dict[str, Any]] = []
    for i in tqdm(range(len(user_per_conv)), desc="Computing features"):
        feats = compute_features_for_conversation(
            user_per_conv[i],
            agent_per_conv[i],
            all_ordered_per_conv[i],
        )
        row: dict[str, Any] = {"conversation_idx": i, **feats}
        if metadata_per_conv and metadata_per_conv[i]:
            meta = metadata_per_conv[i]
            if isinstance(meta, dict):
                for k in ("satisfaction_score", "hash_id", "original_index", "variant", "Domain", "label"):
                    if k in meta:
                        row[k] = meta[k]
            else:
                meta_list = meta
                first_meta = meta_list[0]
                last_meta = meta_list[-1] if len(meta_list) > 1 else first_meta
                for k in ("Domain", "label"):
                    if k in first_meta:
                        row[k] = first_meta[k]
                if "Sentiment" in last_meta:
                    row["Sentiment"] = last_meta["Sentiment"]
                elif "Sentiment" in first_meta:
                    row["Sentiment"] = first_meta["Sentiment"]
        results.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.jsonl:
        with args.output.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(results)} feature rows to {args.output}")


if __name__ == "__main__":
    main()
