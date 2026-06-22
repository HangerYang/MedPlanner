#!/usr/bin/env python3
"""Train and evaluate an MLP that predicts cumulative Code-Feedback reward."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from conversation_feature import IncrementalFeatureState

FEATURE_KEYS = IncrementalFeatureState.FEATURE_KEYS


class RewardMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--target", default="cumulative_reward")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_rows(path: Path, target: str) -> tuple[np.ndarray, np.ndarray, set[int]]:
    features: list[list[float]] = []
    targets: list[float] = []
    conversation_ids: set[int] = set()
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            features.append([float(row[key]) for key in FEATURE_KEYS])
            targets.append(float(row[target]))
            conversation_ids.add(int(row["dataset_index"]))
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        conversation_ids,
    )


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    return {
        "mse": float(np.mean(error ** 2)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "rounded_exact_accuracy": float(np.mean(np.rint(prediction) == target)),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    x_train, y_train, train_conversations = load_rows(args.train, args.target)
    x_test, y_test, test_conversations = load_rows(args.test, args.target)
    overlap = train_conversations & test_conversations
    if overlap:
        raise ValueError(f"Train/test conversation overlap: {len(overlap)}")
    if len(test_conversations) != 1000:
        raise ValueError(f"Expected 1000 test conversations, found {len(test_conversations)}")

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-8
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train[:, None]))
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device(args.device)
    model = RewardMLP(len(FEATURE_KEYS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(xb)
        print(f"epoch={epoch:03d} train_mse={total_loss / len(train_dataset):.6f}", flush=True)

    model.eval()
    with torch.no_grad():
        train_prediction = model(torch.from_numpy(x_train).to(device)).cpu().numpy().reshape(-1)
        test_prediction = model(torch.from_numpy(x_test).to(device)).cpu().numpy().reshape(-1)

    metrics = {
        "target": args.target,
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "train_conversations": len(train_conversations),
        "test_conversations": len(test_conversations),
        "train": regression_metrics(train_prediction, y_train),
        "test": regression_metrics(test_prediction, y_test),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_keys": FEATURE_KEYS,
            "norm_mean": mean.tolist(),
            "norm_std": std.tolist(),
            "target": args.target,
            "metrics": metrics,
        },
        args.output,
    )
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
