#!/usr/bin/env python3
"""Build cumulative-reward, per-turn Code-Feedback conversation features."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from conversation_feature import IncrementalFeatureState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rewards", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--test-conversations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--limit-conversations", type=int, default=None)
    return parser.parse_args()


def load_eligible(path: Path) -> tuple[list[dict[str, Any]], int]:
    eligible: list[dict[str, Any]] = []
    discarded_null = 0
    with path.open() as handle:
        for line in handle:
            conversation = json.loads(line)
            user_rewards = [
                message.get("reward")
                for message in conversation["messages"]
                if message["role"] == "user"
            ]
            if any(reward is None for reward in user_rewards):
                discarded_null += 1
                continue
            eligible.append(conversation)
    return eligible, discarded_null


def load_or_create_split(
    path: Path,
    eligible_ids: list[int],
    test_conversations: int,
    seed: int,
) -> tuple[set[int], set[int]]:
    eligible_set = set(eligible_ids)
    if path.exists():
        manifest = json.loads(path.read_text())
        train_ids = set(manifest["train_dataset_indices"])
        test_ids = set(manifest["test_dataset_indices"])
        if train_ids | test_ids != eligible_set or train_ids & test_ids:
            raise ValueError(
                f"Existing split {path} does not match the current eligible conversations."
            )
        if len(test_ids) != test_conversations:
            raise ValueError(
                f"Existing split has {len(test_ids)} test conversations, expected {test_conversations}."
            )
        return train_ids, test_ids

    if len(eligible_ids) <= test_conversations:
        raise ValueError("Not enough eligible conversations for the requested test split.")
    shuffled = sorted(eligible_ids)
    random.Random(seed).shuffle(shuffled)
    test_ids = set(shuffled[:test_conversations])
    train_ids = eligible_set - test_ids
    manifest = {
        "version": 1,
        "seed": seed,
        "test_conversations": test_conversations,
        "eligible_conversations": len(eligible_ids),
        "train_dataset_indices": sorted(train_ids),
        "test_dataset_indices": sorted(test_ids),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return train_ids, test_ids


def resolve_embedding_path(conversation: dict[str, Any], embedding_dir: Path | None) -> Path:
    if embedding_dir is not None:
        return embedding_dir / f"conv_{conversation['dataset_index']:06d}.npz"
    path = Path(conversation["embedding_path"])
    return path if path.is_absolute() else REPO / path


def normalize_role(role: str) -> str:
    return "user" if role == "user" else "agent"


def main() -> None:
    args = parse_args()
    conversations, discarded_null = load_eligible(args.rewards)
    train_ids, test_ids = load_or_create_split(
        args.split_manifest,
        [int(conversation["dataset_index"]) for conversation in conversations],
        args.test_conversations,
        args.seed,
    )
    if args.limit_conversations is not None:
        conversations = conversations[: args.limit_conversations]

    for path in (args.train_output, args.test_output, args.summary):
        path.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    cumulative_values: Counter[str] = Counter()
    with args.train_output.open("w") as train_handle, args.test_output.open("w") as test_handle:
        for conversation in tqdm(conversations, desc="Building turn features"):
            dataset_index = int(conversation["dataset_index"])
            split = "test" if dataset_index in test_ids else "train"
            output = test_handle if split == "test" else train_handle
            embedding_path = resolve_embedding_path(conversation, args.embedding_dir)
            with np.load(embedding_path, allow_pickle=False) as data:
                embeddings = data["embeddings"]
                roles = [str(role) for role in data["roles"].tolist()]

            messages = conversation["messages"]
            if len(messages) != len(embeddings) or len(messages) != len(roles):
                raise ValueError(f"Turn-count mismatch for dataset_index={dataset_index}")

            state = IncrementalFeatureState()
            cumulative_reward = 0
            for position, (message, embedding, embedding_role) in enumerate(
                zip(messages, embeddings, roles)
            ):
                if message["role"] != embedding_role:
                    raise ValueError(
                        f"Role mismatch at dataset_index={dataset_index}, turn={position + 1}: "
                        f"{message['role']} != {embedding_role}"
                    )
                turn_reward = int(message["reward"]) if message["role"] == "user" else 0
                cumulative_reward += turn_reward
                features = state.add_turn(torch.from_numpy(embedding), normalize_role(message["role"]))
                row = {
                    "conversation_id": conversation["id"],
                    "dataset_index": dataset_index,
                    "split": split,
                    "turn": int(message.get("turn", position + 1)),
                    "embedding_index": int(message.get("embedding_index", position)),
                    "role": message["role"],
                    "turn_reward": turn_reward,
                    "cumulative_reward": cumulative_reward,
                    **features,
                }
                output.write(json.dumps(row, separators=(",", ":")) + "\n")
                counts[f"{split}_turns"] += 1
                counts[f"{split}_{message['role']}_turns"] += 1
                cumulative_values[str(cumulative_reward)] += 1
            counts[f"{split}_conversations"] += 1

    summary = {
        "reward_source": str(args.rewards),
        "feature_source": str(REPO / "conversation_feature.py"),
        "split_manifest": str(args.split_manifest),
        "discarded_null_conversations": discarded_null,
        "eligible_conversations": len(train_ids) + len(test_ids),
        "written": dict(sorted(counts.items())),
        "cumulative_reward_counts": dict(
            sorted(cumulative_values.items(), key=lambda item: int(item[0]))
        ),
        "feature_keys": IncrementalFeatureState.FEATURE_KEYS,
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
