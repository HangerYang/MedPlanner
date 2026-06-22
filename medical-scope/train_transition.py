#!/usr/bin/env python3
"""Train Qwen3-4B transition models compatible with medical-scope.

The input dataset contains one row per leaf conversation and an ``embeddings``
matrix containing accumulated conversation-prefix embeddings. Corrupted rows
caused by missed multiline patient headers are excluded directly from the raw
doctor-view log.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
MEDICAL_SCOPE = Path(__file__).resolve().parent
for path in (REPO, MEDICAL_SCOPE, MEDICAL_SCOPE / "medical_scope", REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_branch_convo import HEADER_RE  # noqa: E402
from build_scope_dataset_from_mediQ_convo import DOCTOR_BRANCH_RE  # noqa: E402
from transition_model import HierarchicalMoE, RegressionWrapper  # noqa: E402


RAW_PATIENT_RE = re.compile(r"^Patient #(?P<id>\d+)\s+\|", re.MULTILINE)
TRANSITIONS = {
    "human_llm": (0, 1),
    "llm_human": (1, 1),
}


def corrupt_trajectory_indices(doctor_view: Path) -> tuple[set[int], int]:
    text = doctor_view.read_text(encoding="utf-8", errors="replace")
    raw_headers = list(RAW_PATIENT_RE.finditer(text))
    recognized_starts = {match.start() for match in HEADER_RE.finditer(text)}
    corrupt: set[int] = set()
    trajectory_index = 0
    current_recognized = False
    for idx, header in enumerate(raw_headers):
        if header.start() in recognized_starts:
            current_recognized = True
        else:
            current_recognized = False
        end = raw_headers[idx + 1].start() if idx + 1 < len(raw_headers) else len(text)
        branches = list(DOCTOR_BRANCH_RE.finditer(text[header.start():end]))
        for _ in branches:
            if not current_recognized:
                corrupt.add(trajectory_index)
            trajectory_index += 1
    return corrupt, trajectory_index


def embedding_files(dataset_dir: Path) -> list[Path]:
    state = json.loads((dataset_dir / "state.json").read_text())
    return [dataset_dir / item["filename"] for item in state["_data_files"]]


def iter_conversations(dataset_dir: Path, selected=None, worker_id=0, num_workers=1):
    trajectory_offset = 0
    for file_idx, path in enumerate(embedding_files(dataset_dir)):
        table = ipc.open_stream(pa.memory_map(str(path), "r")).read_all()
        if file_idx % num_workers == worker_id:
            column = table.column("embeddings")
            for row_idx in range(table.num_rows):
                trajectory_idx = trajectory_offset + row_idx
                if selected is None or trajectory_idx in selected:
                    yield trajectory_idx, np.asarray(column[row_idx].as_py(), dtype=np.float32)
        trajectory_offset += table.num_rows


class TransitionPairs(IterableDataset):
    def __init__(self, dataset_dir, trajectory_indices, start, step, stats=None):
        self.dataset_dir = dataset_dir
        self.trajectory_indices = trajectory_indices
        self.start = start
        self.step = step
        self.stats = stats

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        num_workers = worker.num_workers if worker else 1
        for trajectory_idx, matrix in iter_conversations(self.dataset_dir, self.trajectory_indices, worker_id, num_workers):
            for output_idx in range(self.start + self.step, len(matrix), 2):
                input_idx = output_idx - self.step
                inputs = matrix[input_idx]
                outputs = matrix[output_idx] - matrix[input_idx]
                if self.stats is not None:
                    input_mean, input_std, output_mean, output_std = self.stats
                    inputs = (inputs - input_mean) / input_std
                    outputs = (outputs - output_mean) / output_std
                yield {"inputs": torch.from_numpy(np.asarray(inputs, dtype=np.float32)),
                       "outputs": torch.from_numpy(np.asarray(outputs, dtype=np.float32))}


def count_pairs(dataset_dir, trajectory_indices, start, step, desc="Counting transition pairs"):
    conversations = iter_conversations(dataset_dir, trajectory_indices)
    return sum(
        len(range(start + step, len(matrix), 2))
        for _, matrix in tqdm(conversations, total=len(trajectory_indices), desc=desc, unit="conversation")
    )


def feature_stats(dataset, total_pairs, batch_size=4096):
    loader = DataLoader(dataset, batch_size=batch_size)
    count = 0
    input_sum = output_sum = input_sq_sum = output_sq_sum = None
    for batch in tqdm(loader, total=(total_pairs + batch_size - 1) // batch_size, desc="Calculating normalization stats", unit="batch"):
        inputs = batch["inputs"].float()
        outputs = batch["outputs"].float()
        count += len(inputs)
        values = (inputs.sum(0), outputs.sum(0), inputs.square().sum(0), outputs.square().sum(0))
        if input_sum is None:
            input_sum, output_sum, input_sq_sum, output_sq_sum = values
        else:
            input_sum += values[0]; output_sum += values[1]
            input_sq_sum += values[2]; output_sq_sum += values[3]
    input_mean = input_sum / count
    output_mean = output_sum / count
    input_std = (input_sq_sum / count - input_mean.square()).clamp_min(1e-12).sqrt()
    output_std = (output_sq_sum / count - output_mean.square()).clamp_min(1e-12).sqrt()
    return input_mean, input_std, output_mean, output_std


def evaluate(model, loader, criterion, device, total_batches, desc):
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for batch in tqdm(loader, total=total_batches, desc=desc, unit="batch", leave=False):
            inputs = batch["inputs"].to(device).float()
            targets = batch["outputs"].to(device).float()
            outputs = model(inputs[:, None])[0][:, 0, :]
            loss = criterion(outputs, targets)
            total += float(loss.detach().cpu()) * len(inputs)
            count += len(inputs)
    return total / count


def save_checkpoint(path, model, stats, optimizer, epoch, use_residuals=True):
    dim = stats[0].numel()
    wrapper = RegressionWrapper(model, embedding_size=dim)
    wrapper.input_mean = nn.Parameter(stats[0].cpu(), requires_grad=False)
    wrapper.input_std = nn.Parameter(stats[1].cpu(), requires_grad=False)
    wrapper.output_mean = nn.Parameter(stats[2].cpu(), requires_grad=False)
    wrapper.output_std = nn.Parameter(stats[3].cpu(), requires_grad=False)
    wrapper.use_residuals = nn.Parameter(torch.tensor(use_residuals), requires_grad=False)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": wrapper.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def train_one(args, clean_indices, transition_type):
    start, step = TRANSITIONS[transition_type]
    indices = list(clean_indices)
    random.Random(args.seed).shuffle(indices)
    val_size = max(1, int(len(indices) * args.val_frac))
    val_indices = set(indices[:val_size])
    train_indices = set(indices[val_size:])
    print(f"{transition_type}: counting transition pairs...", flush=True)
    train_pairs = count_pairs(args.dataset, train_indices, start, step, f"{transition_type}: counting train pairs")
    val_pairs = count_pairs(args.dataset, val_indices, start, step, f"{transition_type}: counting validation pairs")
    stats = feature_stats(TransitionPairs(args.dataset, train_indices, start, step), train_pairs)
    input_mean, input_std, output_mean, output_std = stats
    np_stats = tuple(value.numpy() for value in stats)
    train_ds = TransitionPairs(args.dataset, train_indices, start, step, np_stats)
    val_ds = TransitionPairs(args.dataset, val_indices, start, step, np_stats)
    print(f"{transition_type}: train_conversations={len(train_indices)} validation_conversations={len(val_indices)} train_pairs={train_pairs} validation_pairs={val_pairs}", flush=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    dim = int(input_mean.numel())
    device = torch.device(args.device)
    model = HierarchicalMoE(
        dim=dim,
        outer_experts=args.outer_experts,
        inner_experts=args.inner_experts,
        hidden=args.hidden,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    output_dir = args.output_dir / transition_type
    output_dir.mkdir(parents=True, exist_ok=True)
    best_train = float("inf")
    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        train_batches = (train_pairs + args.batch_size - 1) // args.batch_size
        for batch in tqdm(train_loader, total=train_batches, desc=f"{transition_type} epoch {epoch}/{args.epochs} train", unit="batch"):
            inputs = batch["inputs"].to(device).float()
            targets = batch["outputs"].to(device).float()
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs[:, None])[0][:, 0, :]
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(inputs)
            count += len(inputs)
        train_mse = total / count
        val_batch_size = args.batch_size * 2
        val_batches = (val_pairs + val_batch_size - 1) // val_batch_size
        val_mse = evaluate(model, val_loader, criterion, device, val_batches, f"{transition_type} epoch {epoch}/{args.epochs} validation")
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})
        print(f"{transition_type} epoch={epoch} train_mse={train_mse:.6f} val_mse={val_mse:.6f}", flush=True)
        if train_mse < best_train:
            best_train = train_mse
            save_checkpoint(output_dir / "model_min_train.pth", model, stats, optimizer, epoch)
        if val_mse < best_val:
            best_val = val_mse
            save_checkpoint(output_dir / "model_min_val.pth", model, stats, optimizer, epoch)

    (output_dir / "results.json").write_text(
        json.dumps(
            {
                "transition_type": transition_type,
                "embedding_dim": dim,
                "train_pairs": train_pairs,
                "validation_pairs": val_pairs,
                "best_train_mse": best_train,
                "best_validation_mse": best_val,
                "history": history,
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--doctor-view", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transition-type", choices=["both", *TRANSITIONS], default="both")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outer-experts", type=int, default=4)
    parser.add_argument("--inner-experts", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    corrupt, expected_rows = corrupt_trajectory_indices(args.doctor_view)
    files = embedding_files(args.dataset)
    dataset_rows = sum(ipc.open_stream(pa.memory_map(str(path), "r")).read_all().num_rows for path in files)
    if dataset_rows != expected_rows:
        raise RuntimeError(f"Dataset rows {dataset_rows} != parsed trajectories {expected_rows}")
    clean_indices = set(range(dataset_rows)) - corrupt
    first_table = ipc.open_stream(pa.memory_map(str(files[0]), "r")).read_all()
    embedding_dim = len(first_table.column("embeddings")[0].as_py()[0])
    print(
        json.dumps(
            {
                "dataset_rows": dataset_rows,
                "excluded_corrupt_trajectories": len(corrupt),
                "clean_trajectories": len(clean_indices),
                "embedding_dim": embedding_dim,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return
    types = list(TRANSITIONS) if args.transition_type == "both" else [args.transition_type]
    started = time.time()
    for transition_type in types:
        train_one(args, clean_indices, transition_type)
    print(f"Training completed in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
