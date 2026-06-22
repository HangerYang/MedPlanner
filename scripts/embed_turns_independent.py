#!/usr/bin/env python3
"""Embed each conversation turn independently (no accumulated history).

For each branch in the scope_reward_hf dataset, embeds every message
in isolation using Qwen3-4B's last hidden state of the last token:

  turn_idx=0  -> patient initial info  (role: "user")
  turn_idx=1  -> doctor question 1     (role: "assistant")
  turn_idx=2  -> patient answer 1      (role: "user")
  ...

Output is a flat HF dataset, one row per turn:
  patient_id, branch_id, turn_idx, role, text, embedding (2560-d float16)

Uses 2 GPUs via multiprocessing (spawn). Each GPU handles half the rows.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import datasets

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

class Qwen3EmbedWorker:
    def __init__(self, model_name: str, device: str) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[GPU {device}] Loading {model_name}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.device = device

    def embed(self, text: str) -> np.ndarray:
        messages = [{"role": "user", "content": text}]
        with torch.no_grad():
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
            ).to(self.device)
            base = getattr(self.model, "model", self.model)
            outputs = base(input_ids=input_ids)
            emb = outputs.last_hidden_state[:, -1, :].detach().float().cpu()[0]
        return emb.numpy().astype(np.float16)


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

def worker_fn(
    gpu_id: int,
    row_indices: list[int],
    dataset_path: str,
    model_name: str,
    shard_path: str,
) -> None:
    device = f"cuda:{gpu_id}"
    embedder = Qwen3EmbedWorker(model_name, device)

    ds = datasets.load_from_disk(dataset_path)

    patient_ids, branch_ids, turn_idxs, roles, texts, embeddings = [], [], [], [], [], []

    t0 = time.time()
    for n, row_idx in enumerate(row_indices):
        row = ds[row_idx]
        conversation = row["conversation"]

        for turn_idx, msg in enumerate(conversation):
            role = msg["role"]
            text = msg["content"] or ""
            emb = embedder.embed(text)
            patient_ids.append(row["patient_id"])
            branch_ids.append(row["branch_id"])
            turn_idxs.append(turn_idx)
            roles.append(role)
            texts.append(text)
            embeddings.append(emb.tolist())

        if (n + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(
                f"[GPU {gpu_id}] {n + 1}/{len(row_indices)} rows | "
                f"{elapsed:.0f}s elapsed",
                flush=True,
            )

    shard_ds = datasets.Dataset.from_dict(
        {
            "patient_id": patient_ids,
            "branch_id": branch_ids,
            "turn_idx": turn_idxs,
            "role": roles,
            "text": texts,
            "embedding": embeddings,
        },
        features=datasets.Features(
            {
                "patient_id": datasets.Value("int64"),
                "branch_id": datasets.Value("string"),
                "turn_idx": datasets.Value("int32"),
                "role": datasets.Value("string"),
                "text": datasets.Value("string"),
                "embedding": datasets.Sequence(datasets.Value("float32")),
            }
        ),
    )
    Path(shard_path).parent.mkdir(parents=True, exist_ok=True)
    shard_ds.save_to_disk(shard_path)
    print(f"[GPU {gpu_id}] Saved {len(shard_ds)} turns -> {shard_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO
        / "new_outputs/train_hightemp_no_reasoning"
        / "scale_qwen3_4b_branch_d3_hightemp_conf5_train4k_no_reasoning_scope_reward_hf",
        help="Path to scope_reward_hf HF dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO
        / "scope_saved/embeddings"
        / "scale_qwen3_4b_branch_d3_hightemp_conf5_train4k_no_reasoning_scope_reward_hf_qwen3_2560_turn_independent",
        help="Output path for the flat turn-level embedding dataset.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B",
        help="HuggingFace model name for embedding.",
    )
    parser.add_argument(
        "--gpus",
        default="0,1",
        help="Comma-separated GPU IDs to use (e.g. 0,1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    num_gpus = len(gpu_ids)

    print(f"Loading dataset from {args.input}...", flush=True)
    ds = datasets.load_from_disk(str(args.input))
    total_rows = len(ds)
    print(f"Total branches: {total_rows}", flush=True)

    # Split row indices across GPUs
    all_indices = list(range(total_rows))
    shards = [all_indices[i::num_gpus] for i in range(num_gpus)]

    shard_paths = [str(args.output) + f"_shard{i}" for i in range(num_gpus)]

    ctx = mp.get_context("spawn")
    procs = []
    for i, (gpu_id, indices, shard_path) in enumerate(
        zip(gpu_ids, shards, shard_paths)
    ):
        p = ctx.Process(
            target=worker_fn,
            args=(gpu_id, indices, str(args.input), args.model, shard_path),
            daemon=False,
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker process exited with code {p.exitcode}")

    # Merge shards
    print("Merging shards...", flush=True)
    shard_datasets = [datasets.load_from_disk(sp) for sp in shard_paths]
    merged = datasets.concatenate_datasets(shard_datasets)
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_to_disk(str(args.output))
    print(f"Saved {len(merged)} total turns -> {args.output}", flush=True)

    # Clean up shards
    import shutil
    for sp in shard_paths:
        shutil.rmtree(sp, ignore_errors=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
