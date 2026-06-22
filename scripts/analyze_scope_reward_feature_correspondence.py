#!/usr/bin/env python3
"""Analyze whether SCOPE reward rows correspond to conversation features.

Reads the flat reward dataset:
  one row = embedding + reward + source_idx/msg_index metadata

Reconstructs each conversation by grouping on source_idx and sorting by
msg_index, computes the feature set from conversation_feature.py, then ranks
feature correlations against conversation-level reward summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.ipc as ipc
import torch
from scipy.stats import pearsonr, spearmanr

REPO = Path("/home/hyang/mediQ")
DEFAULT_DATASET = REPO / (
    "scope_saved/reward_datasets/"
    "scale_qwen3_4b_branch_d3_hightemp_conf5_train4k_no_reasoning_scope_reward_qwen3_flat_cumulative_full"
)
DEFAULT_OUTPUT_DIR = REPO / "results/scope_reward_feature_correspondence"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from conversation_feature import compute_features_for_conversation  # noqa: E402


def iter_batches(dataset_dir: Path):
    state = json.loads((dataset_dir / "state.json").read_text())
    for item in state["_data_files"]:
        reader = ipc.open_stream(str(dataset_dir / item["filename"]))
        yield from reader


def finalize_group(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    rows.sort(key=lambda row: row["msg_index"])
    all_embs = [torch.tensor(row["embedding"], dtype=torch.float32) for row in rows]
    user_embs = [
        emb for emb, row in zip(all_embs, rows, strict=True) if int(row["msg_index"]) % 2 == 0
    ]
    agent_embs = [
        emb for emb, row in zip(all_embs, rows, strict=True) if int(row["msg_index"]) % 2 == 1
    ]
    features = compute_features_for_conversation(user_embs, agent_embs, all_embs)

    rewards = np.asarray([float(row["reward"]) for row in rows], dtype=np.float64)
    msg_indices = np.asarray([int(row["msg_index"]) for row in rows], dtype=np.int64)
    turn_indices = np.asarray([int(row["turn_index"]) for row in rows], dtype=np.int64)
    final_flags = np.asarray([bool(row["is_final_turn"]) for row in rows], dtype=bool)

    reward_deltas = np.diff(rewards) if len(rewards) >= 2 else np.asarray([], dtype=np.float64)
    final_turn_rewards = rewards[final_flags]
    source_idx = int(rows[0]["source_idx"])
    out: dict[str, Any] = {
        "source_idx": source_idx,
        "patient_id": int(rows[0]["patient_id"]),
        "branch_id": str(rows[0]["branch_id"]),
        "num_messages": int(len(rows)),
        "min_msg_index": int(msg_indices.min()),
        "max_msg_index": int(msg_indices.max()),
        "min_turn_index": int(turn_indices.min()),
        "max_turn_index": int(turn_indices.max()),
        "reward_final": float(rewards[-1]),
        "reward_mean": float(rewards.mean()),
        "reward_max": float(rewards.max()),
        "reward_sum": float(rewards.sum()),
        "reward_delta_sum": float(reward_deltas.sum()) if len(reward_deltas) else 0.0,
        "reward_delta_max": float(reward_deltas.max()) if len(reward_deltas) else 0.0,
        "final_turn_reward_mean": float(final_turn_rewards.mean())
        if len(final_turn_rewards)
        else float(rewards[-1]),
        **features,
    }
    return out


def iter_conversations(dataset_dir: Path, max_conversations: int | None = None):
    current_source: int | None = None
    current_rows: list[dict[str, Any]] = []
    yielded = 0

    for batch in iter_batches(dataset_dir):
        names = batch.column_names
        cols = {name: batch.column(name) for name in names}
        for i in range(batch.num_rows):
            source_idx = int(cols["source_idx"][i].as_py())
            if current_source is not None and source_idx != current_source:
                conv = finalize_group(current_rows)
                if conv is not None:
                    yield conv
                    yielded += 1
                    if max_conversations is not None and yielded >= max_conversations:
                        return
                current_rows = []
            current_source = source_idx
            current_rows.append({name: cols[name][i].as_py() for name in names})

    conv = finalize_group(current_rows)
    if conv is not None and (max_conversations is None or yielded < max_conversations):
        yield conv


def finite_pair(xs: list[float], ys: list[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def corr(x: list[float], y: list[float]) -> dict[str, float | int | None]:
    x_arr, y_arr = finite_pair(x, y)
    if len(x_arr) < 3 or float(np.std(x_arr)) == 0.0 or float(np.std(y_arr)) == 0.0:
        return {"n": int(len(x_arr)), "pearson": None, "spearman": None}
    pr = pearsonr(x_arr, y_arr).statistic
    sr = spearmanr(x_arr, y_arr).statistic
    return {
        "n": int(len(x_arr)),
        "pearson": None if math.isnan(float(pr)) else float(pr),
        "spearman": None if math.isnan(float(sr)) else float(sr),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-conversations", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for idx, conv in enumerate(iter_conversations(args.dataset, args.max_conversations), start=1):
        rows.append(conv)
        if idx % 1000 == 0:
            print(f"processed conversations={idx}", flush=True)

    reward_targets = [
        "reward_final",
        "reward_mean",
        "reward_max",
        "reward_sum",
        "reward_delta_sum",
        "reward_delta_max",
        "final_turn_reward_mean",
    ]
    metadata_keys = {
        "source_idx",
        "patient_id",
        "branch_id",
        "num_messages",
        "min_msg_index",
        "max_msg_index",
        "min_turn_index",
        "max_turn_index",
        *reward_targets,
    }
    feature_keys = [key for key in rows[0].keys() if key not in metadata_keys] if rows else []

    corr_rows: list[dict[str, Any]] = []
    for target in reward_targets:
        y = [float(row[target]) for row in rows]
        for feature in feature_keys:
            x = [float(row[feature]) for row in rows]
            stats = corr(x, y)
            corr_rows.append(
                {
                    "target": target,
                    "feature": feature,
                    **stats,
                    "abs_spearman": abs(stats["spearman"]) if stats["spearman"] is not None else None,
                    "abs_pearson": abs(stats["pearson"]) if stats["pearson"] is not None else None,
                }
            )

    corr_rows.sort(
        key=lambda row: (
            row["abs_spearman"] is not None,
            row["abs_spearman"] if row["abs_spearman"] is not None else -1,
        ),
        reverse=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "conversation_features_with_rewards.csv", rows)
    write_csv(args.output_dir / "feature_reward_correlations.csv", corr_rows)

    summary = {
        "dataset": str(args.dataset),
        "num_conversations": len(rows),
        "num_features": len(feature_keys),
        "reward_targets": reward_targets,
        "top_by_abs_spearman": corr_rows[:25],
        "outputs": {
            "features": str(args.output_dir / "conversation_features_with_rewards.csv"),
            "correlations": str(args.output_dir / "feature_reward_correlations.csv"),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
