#!/usr/bin/env python3
"""Train a Qwen3-embedding reward model for patient response text length.

The saved model matches medical-scope's EmbeddingScopeReward architecture:
2560 -> 512 -> 256 -> 64 -> 32 -> 1.  It predicts cumulative patient-response
length value, where each non-initial patient/user message contributes
len(content) / 100.  At runtime, medical-scope uses value differences:

    V(new_state_embedding) - V(prev_state_embedding)

which approximates the patient response length reward for that transition.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


class RewardMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2560, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EmbeddingLabelDataset(Dataset):
    def __init__(self, dataset, labels: np.ndarray) -> None:
        self.dataset = dataset
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        row = self.dataset[int(idx)]
        x = row["embedding"]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.to(dtype=torch.float32)
        y = torch.tensor([self.labels[int(idx)]], dtype=torch.float32)
        return x, y


def build_labels(flat_ds, text_ds):
    print("Building cumulative patient response length labels...", flush=True)
    started = time.time()

    source_idx = np.asarray(flat_ds["source_idx"], dtype=np.int64)
    msg_idx = np.asarray(flat_ds["msg_index"], dtype=np.int64)

    max_messages = max(len(row["conversation"]) for row in text_ds)
    label_table = np.zeros((len(text_ds), max_messages), dtype=np.float32)
    answer_rewards: list[float] = []

    for row_idx, row in enumerate(text_ds):
        cumulative = 0.0
        for message_idx, message in enumerate(row["conversation"]):
            if message_idx > 0 and message.get("role") == "user":
                reward = len(message.get("content") or "") / 100.0
                cumulative += reward
                answer_rewards.append(reward)
            label_table[row_idx, message_idx] = cumulative

    labels = label_table[source_idx, msg_idx].astype(np.float32, copy=False)
    stats = {
        "label_mean": float(labels.mean()),
        "label_std": float(labels.std()),
        "label_min": float(labels.min()),
        "label_max": float(labels.max()),
        "patient_answer_reward_mean": float(np.mean(answer_rewards)) if answer_rewards else 0.0,
        "patient_answer_reward_std": float(np.std(answer_rewards)) if answer_rewards else 0.0,
        "patient_answer_reward_max": float(np.max(answer_rewards)) if answer_rewards else 0.0,
        "label_build_sec": time.time() - started,
    }
    print(
        "Labels:",
        f"n={len(labels)}",
        f"mean={stats['label_mean']:.4f}",
        f"std={stats['label_std']:.4f}",
        f"min={stats['label_min']:.4f}",
        f"max={stats['label_max']:.4f}",
        f"built_in={stats['label_build_sec']:.1f}s",
        flush=True,
    )
    return labels, stats


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    loss_sum = 0.0
    abs_sum = 0.0
    count = 0
    mse = nn.MSELoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            batch = x.shape[0]
            loss_sum += float(mse(pred, y).detach().cpu()) * batch
            abs_sum += float(torch.abs(pred - y).sum().detach().cpu())
            count += batch
    return loss_sum / count, abs_sum / count


def parse_args() -> argparse.Namespace:
    repo = Path("/home/hyang/mediQ")
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path("/home/hyang/mediQ")
    parser.add_argument(
        "--flat-dataset",
        type=Path,
        default=repo
        / "scope_saved/reward_datasets/scale_qwen3_4b_branch_d3_hightemp_conf5_train4k_no_reasoning_scope_reward_qwen3_flat_cumulative_full",
        help="Flat dataset containing 2560-d embeddings plus source_idx/msg_index metadata.",
    )
    parser.add_argument(
        "--text-dataset",
        type=Path,
        default=repo
        / "new_outputs/train_hightemp_no_reasoning/scale_qwen3_4b_branch_d3_hightemp_conf5_train4k_no_reasoning_scope_reward_hf",
        help="Row-aligned text dataset containing conversation messages.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "scope_saved/reward/embedding_patient_response_length_reward.pt",
    )
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Primary torch device. Use cuda:0 with --data-parallel-gpus for multi-GPU.",
    )
    parser.add_argument(
        "--data-parallel-gpus",
        default="",
        help="Comma-separated CUDA device ids for torch.nn.DataParallel, e.g. 0,1,2,3.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("Loading datasets...", flush=True)
    flat_ds = load_from_disk(str(args.flat_dataset))
    text_ds = load_from_disk(str(args.text_dataset))
    print(flat_ds, flush=True)
    print(text_ds, flush=True)

    labels, label_stats = build_labels(flat_ds, text_ds)

    flat_ds = flat_ds.with_format("torch", columns=["embedding"])
    n = len(labels)
    rng = np.random.default_rng(args.seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    val_size = min(args.val_size, max(1, n // 10))
    val_idx = indices[:val_size].tolist()
    train_idx = indices[val_size:].tolist()

    base_ds = EmbeddingLabelDataset(flat_ds, labels)
    train_loader = DataLoader(
        Subset(base_ds, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        Subset(base_ds, val_idx),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device(args.device)
    model = RewardMLP().to(device)
    gpu_ids = [int(item) for item in args.data_parallel_gpus.split(",") if item.strip()]
    if gpu_ids:
        print(f"Using DataParallel on CUDA devices: {gpu_ids}", flush=True)
        model = nn.DataParallel(model, device_ids=gpu_ids)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = mse(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch = x.shape[0]
            loss_sum += float(loss.detach().cpu()) * batch
            count += batch
            if step % 50 == 0:
                print(
                    f"epoch {epoch} step {step}/{len(train_loader)} "
                    f"train_mse={loss_sum / count:.6f}",
                    flush=True,
                )

        train_mse = loss_sum / count
        val_mse, val_mae = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse, "val_mae": val_mae}
        history.append(row)
        print(
            f"epoch {epoch}: train_mse={train_mse:.6f} "
            f"val_mse={val_mse:.6f} val_mae={val_mae:.6f}",
            flush=True,
        )
        if val_mse < best_val:
            best_val = val_mse
            net = model.module.net if isinstance(model, nn.DataParallel) else model.net
            best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.output)
    summary_path = args.summary or args.output.with_suffix(args.output.suffix + ".summary.json")
    summary = {
        "output_model": str(args.output),
        "flat_dataset": str(args.flat_dataset),
        "text_dataset": str(args.text_dataset),
        "target": "cumulative sum over non-initial patient/user messages of len(content)/100",
        "num_rows": n,
        "train_rows": len(train_idx),
        "val_rows": len(val_idx),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "device": args.device,
        "data_parallel_gpus": gpu_ids,
        "seed": args.seed,
        "best_val_mse": best_val,
        "history": history,
        "elapsed_sec": time.time() - started,
        **label_stats,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved model: {args.output}", flush=True)
    print(f"Saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
