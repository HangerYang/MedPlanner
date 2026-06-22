#!/usr/bin/env python3
"""Compute row-aligned rewards for a saved SCOPE embedding dataset.

The embedding dataset stores one row per conversation/branch.  Each row has a
variable-length list of 2560-d embeddings, one per conversation state/message.
The reward checkpoint is an MLP that maps each 2560-d embedding to a cumulative
reward value.  This script writes a lightweight dataset whose row i corresponds
to embedding row i and contains the per-embedding values and transition rewards.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.ipc as ipc
import torch
from datasets import Dataset
from torch import nn


DEFAULT_EMBEDDINGS = Path(
    "/home/hyang/mediQ/scope_saved/embeddings/"
    "scale_qwen3_4b_branch_d3_hightemp_conf5_train4k_no_reasoning_scope_reward_hf_qwen3_2560_full"
)
DEFAULT_REWARD = Path(
    "/home/hyang/mediQ/scope_saved/reward/embedding_mediQ_reward_cumulative.pt"
)
DEFAULT_OUTPUT = Path(
    "/home/hyang/mediQ/scope_saved/reward_aligned/"
    "scale_qwen3_4b_branch_d3_hightemp_conf5_train4k_no_reasoning_scope_reward_hf_qwen3_2560_full__embedding_mediQ_reward_cumulative"
)


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


def load_model(path: Path, device: torch.device) -> nn.Module:
    model = RewardMLP().to(device)
    state = torch.load(path, map_location=device)
    if any(str(key).startswith("net.") for key in state):
        state = {str(key).removeprefix("net."): value for key, value in state.items()}
    model.net.load_state_dict(state)
    model.eval()
    return model


def iter_embedding_rows(dataset_dir: Path):
    state = json.loads((dataset_dir / "state.json").read_text())
    for item in state["_data_files"]:
        path = dataset_dir / item["filename"]
        reader = ipc.open_stream(str(path))
        for batch in reader:
            col = batch.column("embeddings")
            for row_idx in range(batch.num_rows):
                yield col[row_idx].as_py()


def predict_values(
    embeddings: list[list[float]],
    model: nn.Module,
    device: torch.device,
    batch_size: int,
) -> list[float]:
    values: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(embeddings), batch_size):
            array = np.asarray(embeddings[start : start + batch_size], dtype=np.float32)
            tensor = torch.from_numpy(array).to(device)
            pred = model(tensor).reshape(-1).detach().cpu().numpy()
            values.extend(float(x) for x in pred)
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--reward-model", type=Path, default=DEFAULT_REWARD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device used for reward inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    device = torch.device(args.device)
    model = load_model(args.reward_model, device)

    row_ids: list[int] = []
    num_embeddings: list[int] = []
    cumulative_values: list[list[float]] = []
    transition_rewards: list[list[float]] = []

    all_values: list[float] = []
    total_embeddings = 0
    for row_id, embeddings in enumerate(iter_embedding_rows(args.embeddings)):
        values = predict_values(embeddings, model, device, args.batch_size)
        rewards = [values[0]] + [
            values[i] - values[i - 1] for i in range(1, len(values))
        ]
        row_ids.append(row_id)
        num_embeddings.append(len(values))
        cumulative_values.append(values)
        transition_rewards.append(rewards)
        all_values.extend(values)
        total_embeddings += len(values)
        if (row_id + 1) % 1000 == 0:
            print(
                f"processed rows={row_id + 1} embeddings={total_embeddings}",
                flush=True,
            )

    ds = Dataset.from_dict(
        {
            "embedding_row_index": row_ids,
            "num_embeddings": num_embeddings,
            "cumulative_reward_values": cumulative_values,
            "transition_rewards": transition_rewards,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(args.output))

    values_np = np.asarray(all_values, dtype=np.float32)
    summary = {
        "embeddings": str(args.embeddings),
        "reward_model": str(args.reward_model),
        "output": str(args.output),
        "num_conversation_rows": len(row_ids),
        "num_embedding_values": total_embeddings,
        "value_mean": float(values_np.mean()) if len(values_np) else None,
        "value_std": float(values_np.std()) if len(values_np) else None,
        "value_min": float(values_np.min()) if len(values_np) else None,
        "value_max": float(values_np.max()) if len(values_np) else None,
        "elapsed_sec": time.time() - started,
        "correspondence": (
            "output row i corresponds to embeddings dataset row i; each list "
            "position j corresponds to embeddings[i]['embeddings'][j]"
        ),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
