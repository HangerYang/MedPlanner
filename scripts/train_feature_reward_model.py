#!/usr/bin/env python3
"""Train a reward MLP on 26 conversation features.

Input : results/turn_level_conversation_features/features.jsonl
Target: reward (cumulative, per turn)
Output: scope_saved/reward/feature_reward_mlp.pt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO = Path("/home/hyang/mediQ")
DEFAULT_INPUT  = REPO / "results/turn_level_conversation_features/features.jsonl"
DEFAULT_OUTPUT = REPO / "scope_saved/reward/feature_reward_mlp.pt"

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
    "Conversation Drift from Goal", "Trend in Goal Adherence",
    "Goal Convergence Ratio",
]


class RewardMLP(nn.Module):
    def __init__(self, input_dim: int = 26) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            feats = [float(row[k]) for k in FEATURE_KEYS]
            X.append(feats)
            y.append(float(row["reward"]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",  type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch-size", type=int,   default=4096)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val-frac",   type=float, default=0.05)
    parser.add_argument("--device",     default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("Loading data...", flush=True)
    X, y = load_data(args.input)
    print(f"  {len(X)} rows, {X.shape[1]} features", flush=True)

    # Normalize features
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    X = (X - mean) / std

    # Train / val split (by conversation to avoid leakage)
    n = len(X)
    idx = list(range(n))
    random.shuffle(idx)
    split = int(n * (1 - args.val_frac))
    tr_idx, va_idx = idx[:split], idx[split:]

    X_tr = torch.tensor(X[tr_idx])
    y_tr = torch.tensor(y[tr_idx]).unsqueeze(1)
    X_va = torch.tensor(X[va_idx])
    y_va = torch.tensor(y[va_idx]).unsqueeze(1)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)

    device = torch.device(args.device)
    model  = RewardMLP(input_dim=X.shape[1]).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    X_va_d = X_va.to(device)
    y_va_d = y_va.to(device)

    print(f"Training on {device} for {args.epochs} epochs...", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        train_mse = total_loss / len(tr_idx)

        model.eval()
        with torch.no_grad():
            val_mse = loss_fn(model(X_va_d), y_va_d).item()

        print(f"  epoch {epoch:>3}/{args.epochs}  train_mse={train_mse:.6f}  val_mse={val_mse:.6f}", flush=True)

    # Save model weights + normalisation stats
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "net": model.net.state_dict(),
        "feature_keys": FEATURE_KEYS,
        "norm_mean": mean.tolist(),
        "norm_std":  std.tolist(),
    }, args.output)
    print(f"Saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
