#!/usr/bin/env python3
"""Train MoE or MDN transition models for medical-scope and code-scope.

Supports three embedding sources:
  - medical accumulative HF matrices (prefix embeddings)
  - medical turn-independent HF flat rows (grouped by conversation)
  - code-feedback NPZ files (conv_*.npz)

Trains human_llm and llm_human transitions with residual targets.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
MEDICAL_SCOPE = REPO / "medical-scope"
CONVO_SCOPE = REPO / "convo-plan-SCOPE"
for path in (
    REPO,
    MEDICAL_SCOPE,
    MEDICAL_SCOPE / "medical_scope",
    CONVO_SCOPE / "mdn" / "src",
    CONVO_SCOPE / "transition_models",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from medical_scope.transition_model import HierarchicalMoE, RegressionWrapper  # noqa: E402
from blocks import MixtureDensityNetwork  # noqa: E402
from regression_wrapper import RegressionWrapper as MDNRegressionWrapper  # noqa: E402

TRANSITIONS = {
    "human_llm": (0, 1),
    "llm_human": (1, 1),
}

DEFAULT_PATHS = {
    "medical_accumulative": REPO
    / "scope_saved/embeddings/scale_qwen3_4b_branch_d3_hightemp_conf5_train4k_no_reasoning_scope_reward_hf_qwen3_2560_full",
    "medical_turn_independent": REPO
    / "scope_saved/embeddings/scale_qwen3_4b_branch_d3_hightemp_conf5_train4k_no_reasoning_scope_reward_hf_qwen3_2560_turn_independent",
    "code": REPO / "data/med_data/data/embeddings",
}

OUTPUT_ROOT = REPO / "scope_saved/transition_models/new"


class ConversationBackend(ABC):
    @abstractmethod
    def num_conversations(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_matrix(self, idx: int) -> np.ndarray:
        raise NotImplementedError


class HFMatrixBackend(ConversationBackend):
    def __init__(self, path: Path) -> None:
        self.ds = load_from_disk(str(path))

    def num_conversations(self) -> int:
        return len(self.ds)

    def get_matrix(self, idx: int) -> np.ndarray:
        return np.asarray(self.ds[int(idx)]["embeddings"], dtype=np.float32)


class HFTurnIndependentBackend(ConversationBackend):
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        cache = self.path / "_conversation_ranges.json"
        if cache.exists():
            self.ranges = [tuple(pair) for pair in json.loads(cache.read_text())]
        else:
            print(f"Building conversation ranges for {path}...", flush=True)
            ds = load_from_disk(str(path)).sort(["patient_id", "branch_id", "turn_idx"])
            patients = np.asarray(ds["patient_id"])
            branches = np.asarray(ds["branch_id"])
            change = np.ones(len(patients), dtype=bool)
            if len(patients) > 1:
                change[1:] = (patients[1:] != patients[:-1]) | (branches[1:] != branches[:-1])
            starts = np.where(change)[0]
            ends = np.append(starts[1:], len(patients))
            ranges = list(zip(starts.tolist(), ends.tolist()))
            cache.write_text(json.dumps(ranges) + "\n")
            self.ranges = ranges
            print(f"Cached {len(ranges)} conversation ranges -> {cache}", flush=True)
        self.ds = load_from_disk(str(path))

    def num_conversations(self) -> int:
        return len(self.ranges)

    def get_matrix(self, idx: int) -> np.ndarray:
        start, end = self.ranges[int(idx)]
        embs = self.ds[start:end]["embedding"]
        return np.stack([np.asarray(row, dtype=np.float32) for row in embs], axis=0)


class NPZBackend(ConversationBackend):
    def __init__(self, path: Path) -> None:
        self.files = sorted(Path(path).glob("conv_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No conv_*.npz files found under {path}")

    def num_conversations(self) -> int:
        return len(self.files)

    def get_matrix(self, idx: int) -> np.ndarray:
        data = np.load(self.files[int(idx)])
        return np.asarray(data["embeddings"], dtype=np.float32)


class TransitionPairs(IterableDataset):
    def __init__(
        self,
        backend: ConversationBackend,
        conv_indices: set[int] | list[int],
        start: int,
        step: int,
        stats: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.backend = backend
        self.conv_indices = sorted(conv_indices)
        self.start = start
        self.step = step
        self.stats = stats

    def __iter__(self):
        worker = get_worker_info()
        indices = self.conv_indices
        if worker is not None:
            indices = indices[worker.id :: worker.num_workers]
        for conv_idx in indices:
            matrix = self.backend.get_matrix(conv_idx)
            for output_idx in range(self.start + self.step, len(matrix), 2):
                input_idx = output_idx - self.step
                inputs = matrix[input_idx]
                outputs = matrix[output_idx] - matrix[input_idx]
                if self.stats is not None:
                    input_mean, input_std, output_mean, output_std = self.stats
                    inputs = (inputs - input_mean) / input_std
                    outputs = (outputs - output_mean) / output_std
                yield {
                    "inputs": torch.from_numpy(np.asarray(inputs, dtype=np.float32)),
                    "outputs": torch.from_numpy(np.asarray(outputs, dtype=np.float32)),
                }


def build_backend(scope: str, embedding_mode: str, dataset: Path) -> ConversationBackend:
    if scope == "code":
        return NPZBackend(dataset)
    if embedding_mode == "accumulative":
        return HFMatrixBackend(dataset)
    if embedding_mode == "turn_independent":
        return HFTurnIndependentBackend(dataset)
    raise ValueError(f"Unknown embedding mode: {embedding_mode}")


def default_output_dir(scope: str, embedding_mode: str, model_type: str, seed: int, batch_size: int) -> Path:
    if scope == "medical":
        prefix = f"medical_{embedding_mode}"
    else:
        prefix = "code_feedback"
    return OUTPUT_ROOT / f"{prefix}_{model_type}_seed_{seed}_batch_{batch_size}"


def count_pairs(
    backend: ConversationBackend,
    conv_indices: set[int],
    start: int,
    step: int,
    desc: str,
) -> int:
    total = 0
    for conv_idx in tqdm(sorted(conv_indices), desc=desc, unit="conversation"):
        matrix = backend.get_matrix(conv_idx)
        total += len(range(start + step, len(matrix), 2))
    return total


def feature_stats(dataset: TransitionPairs, total_pairs: int, batch_size: int = 4096):
    loader = DataLoader(dataset, batch_size=batch_size)
    count = 0
    input_sum = output_sum = input_sq_sum = output_sq_sum = None
    for batch in tqdm(
        loader,
        total=max(1, (total_pairs + batch_size - 1) // batch_size),
        desc="Calculating normalization stats",
        unit="batch",
    ):
        inputs = batch["inputs"].float()
        outputs = batch["outputs"].float()
        count += len(inputs)
        values = (
            inputs.sum(0),
            outputs.sum(0),
            inputs.square().sum(0),
            outputs.square().sum(0),
        )
        if input_sum is None:
            input_sum, output_sum, input_sq_sum, output_sq_sum = values
        else:
            input_sum += values[0]
            output_sum += values[1]
            input_sq_sum += values[2]
            output_sq_sum += values[3]
    input_mean = input_sum / count
    output_mean = output_sum / count
    input_std = (input_sq_sum / count - input_mean.square()).clamp_min(1e-12).sqrt()
    output_std = (output_sq_sum / count - output_mean.square()).clamp_min(1e-12).sqrt()
    return input_mean, input_std, output_mean, output_std


def save_moe_checkpoint(path: Path, model, stats, optimizer, epoch: int) -> None:
    dim = stats[0].numel()
    wrapper = RegressionWrapper(model, embedding_size=dim)
    wrapper.input_mean = nn.Parameter(stats[0].cpu(), requires_grad=False)
    wrapper.input_std = nn.Parameter(stats[1].cpu(), requires_grad=False)
    wrapper.output_mean = nn.Parameter(stats[2].cpu(), requires_grad=False)
    wrapper.output_std = nn.Parameter(stats[3].cpu(), requires_grad=False)
    wrapper.use_residuals = nn.Parameter(torch.tensor(True), requires_grad=False)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": wrapper.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def save_mdn_checkpoint(path: Path, model, stats, optimizer, epoch: int) -> None:
    dim = stats[0].numel()
    wrapper = MDNRegressionWrapper(model, embedding_size=dim)
    wrapper.set_parameters(stats[0], stats[1], stats[2], stats[3], use_residuals=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": wrapper.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def evaluate_moe(model, loader, criterion, device, total_batches: int, desc: str) -> float:
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
    return total / max(count, 1)


def evaluate_mdn(model, loader, device, total_batches: int, desc: str) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for batch in tqdm(loader, total=total_batches, desc=desc, unit="batch", leave=False):
            inputs = batch["inputs"].to(device).float()
            targets = batch["outputs"].to(device).float()
            loss = model.loss(inputs, targets).mean()
            total += float(loss.detach().cpu()) * len(inputs)
            count += len(inputs)
    return total / max(count, 1)


def train_moe(args, backend, train_indices, val_indices, transition_type: str) -> None:
    start, step = TRANSITIONS[transition_type]
    train_pairs = count_pairs(backend, train_indices, start, step, f"{transition_type}: counting train pairs")
    val_pairs = count_pairs(backend, val_indices, start, step, f"{transition_type}: counting validation pairs")
    stats = feature_stats(TransitionPairs(backend, train_indices, start, step), train_pairs, args.batch_size)
    np_stats = tuple(value.numpy() for value in stats)
    train_ds = TransitionPairs(backend, train_indices, start, step, np_stats)
    val_ds = TransitionPairs(backend, val_indices, start, step, np_stats)
    print(
        f"{transition_type}: train_conversations={len(train_indices)} "
        f"validation_conversations={len(val_indices)} "
        f"train_pairs={train_pairs} validation_pairs={val_pairs}",
        flush=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    dim = int(stats[0].numel())
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
        train_batches = max(1, (train_pairs + args.batch_size - 1) // args.batch_size)
        for batch in tqdm(
            train_loader,
            total=train_batches,
            desc=f"{transition_type} epoch {epoch}/{args.epochs} train",
            unit="batch",
        ):
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
        train_loss = total / max(count, 1)
        val_batches = max(1, (val_pairs + args.batch_size * 2 - 1) // (args.batch_size * 2))
        val_loss = evaluate_moe(
            model,
            val_loader,
            criterion,
            device,
            val_batches,
            f"{transition_type} epoch {epoch}/{args.epochs} validation",
        )
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})
        print(f"{transition_type} epoch={epoch} train_mse={train_loss:.6f} val_mse={val_loss:.6f}", flush=True)
        if train_loss < best_train:
            best_train = train_loss
            save_moe_checkpoint(output_dir / "model_min_train.pth", model, stats, optimizer, epoch)
        if val_loss < best_val:
            best_val = val_loss
            save_moe_checkpoint(output_dir / "model_min_val.pth", model, stats, optimizer, epoch)

    _write_results(
        output_dir,
        transition_type,
        dim,
        train_pairs,
        val_pairs,
        best_train,
        best_val,
        history,
        loss_name="mse",
    )


def train_mdn(args, backend, train_indices, val_indices, transition_type: str) -> None:
    start, step = TRANSITIONS[transition_type]
    train_pairs = count_pairs(backend, train_indices, start, step, f"{transition_type}: counting train pairs")
    val_pairs = count_pairs(backend, val_indices, start, step, f"{transition_type}: counting validation pairs")
    stats = feature_stats(TransitionPairs(backend, train_indices, start, step), train_pairs, args.batch_size)
    np_stats = tuple(value.numpy() for value in stats)
    train_ds = TransitionPairs(backend, train_indices, start, step, np_stats)
    val_ds = TransitionPairs(backend, val_indices, start, step, np_stats)
    print(
        f"{transition_type}: train_conversations={len(train_indices)} "
        f"validation_conversations={len(val_indices)} "
        f"train_pairs={train_pairs} validation_pairs={val_pairs}",
        flush=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    dim = int(stats[0].numel())
    device = torch.device(args.device)
    model = MixtureDensityNetwork(
        dim,
        dim,
        args.mdn_components,
        args.mdn_hidden,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir = args.output_dir / transition_type
    output_dir.mkdir(parents=True, exist_ok=True)
    best_train = float("inf")
    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        train_batches = max(1, (train_pairs + args.batch_size - 1) // args.batch_size)
        for batch in tqdm(
            train_loader,
            total=train_batches,
            desc=f"{transition_type} epoch {epoch}/{args.epochs} train",
            unit="batch",
        ):
            inputs = batch["inputs"].to(device).float()
            targets = batch["outputs"].to(device).float()
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(inputs, targets).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(inputs)
            count += len(inputs)
        train_loss = total / max(count, 1)
        val_batches = max(1, (val_pairs + args.batch_size * 2 - 1) // (args.batch_size * 2))
        val_loss = evaluate_mdn(
            model,
            val_loader,
            device,
            val_batches,
            f"{transition_type} epoch {epoch}/{args.epochs} validation",
        )
        history.append({"epoch": epoch, "train_nll": train_loss, "val_nll": val_loss})
        print(f"{transition_type} epoch={epoch} train_nll={train_loss:.6f} val_nll={val_loss:.6f}", flush=True)
        if train_loss < best_train:
            best_train = train_loss
            save_mdn_checkpoint(output_dir / "model_min_train.pth", model, stats, optimizer, epoch)
        if val_loss < best_val:
            best_val = val_loss
            save_mdn_checkpoint(output_dir / "model_min_val.pth", model, stats, optimizer, epoch)

    _write_results(
        output_dir,
        transition_type,
        dim,
        train_pairs,
        val_pairs,
        best_train,
        best_val,
        history,
        loss_name="nll",
    )


def _write_results(
    output_dir: Path,
    transition_type: str,
    dim: int,
    train_pairs: int,
    val_pairs: int,
    best_train: float,
    best_val: float,
    history: list[dict],
    loss_name: str,
) -> None:
    payload = {
        "transition_type": transition_type,
        "embedding_dim": dim,
        "train_pairs": train_pairs,
        "validation_pairs": val_pairs,
        f"best_train_{loss_name}": best_train,
        f"best_validation_{loss_name}": best_val,
        "history": history,
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["medical", "code"], required=True)
    parser.add_argument(
        "--embedding-mode",
        choices=["accumulative", "turn_independent"],
        default="accumulative",
        help="Medical only. Code always uses NPZ turn embeddings.",
    )
    parser.add_argument("--model-type", choices=["moe", "mdn"], required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--transition-type", choices=["both", *TRANSITIONS], default="both")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outer-experts", type=int, default=4)
    parser.add_argument("--inner-experts", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=10240)
    parser.add_argument("--mdn-components", type=int, default=64)
    parser.add_argument("--mdn-hidden", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_dataset(args: argparse.Namespace) -> Path:
    if args.dataset is not None:
        return args.dataset
    if args.scope == "code":
        return DEFAULT_PATHS["code"]
    key = "medical_accumulative" if args.embedding_mode == "accumulative" else "medical_turn_independent"
    return DEFAULT_PATHS[key]


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = resolve_dataset(args)
    if args.output_dir is None:
        args.output_dir = default_output_dir(
            args.scope,
            args.embedding_mode if args.scope == "medical" else "npz",
            args.model_type,
            args.seed,
            args.batch_size,
        )

    backend = build_backend(args.scope, args.embedding_mode, dataset)
    all_indices = list(range(backend.num_conversations()))
    rng = random.Random(args.seed)
    rng.shuffle(all_indices)
    val_size = max(1, int(len(all_indices) * args.val_frac))
    val_indices = set(all_indices[:val_size])
    train_indices = set(all_indices[val_size:])

    sample_matrix = backend.get_matrix(all_indices[0])
    print(
        json.dumps(
            {
                "scope": args.scope,
                "embedding_mode": args.embedding_mode if args.scope == "medical" else "npz",
                "model_type": args.model_type,
                "dataset": str(dataset),
                "conversations": backend.num_conversations(),
                "embedding_dim": int(sample_matrix.shape[-1]),
                "train_conversations": len(train_indices),
                "validation_conversations": len(val_indices),
                "output_dir": str(args.output_dir),
                "seed": args.seed,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    types = list(TRANSITIONS) if args.transition_type == "both" else [args.transition_type]
    started = time.time()
    train_fn = train_moe if args.model_type == "moe" else train_mdn
    for transition_type in types:
        train_fn(args, backend, train_indices, val_indices, transition_type)
    print(f"Training completed in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
