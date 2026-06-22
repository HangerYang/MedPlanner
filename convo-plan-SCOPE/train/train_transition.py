import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import json
import numpy as np
import datasets as hf_datasets
from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from pathlib import Path
import wandb
import argparse

from mixture_of_experts import HeirarchicalMoE
from transition_models.regression_wrapper import RegressionWrapper

EMB_DIM_QWEN3_4B = 2560  # Qwen3-4B last hidden-state dim


def prepare_npz_splits(npz_dir: Path, cache_dir: Path, test_size: int = 1000, seed: int = 42):
    """
    Conversation-level train/test split of conv_*.npz files.
    Caches two HF datasets (trainval, test) and a split_ids.json to cache_dir.
    Returns (trainval_ds, test_ds).
    """
    trainval_cache = cache_dir / "trainval"
    test_cache = cache_dir / "test"
    split_file = cache_dir / "split_ids.json"

    if trainval_cache.exists() and test_cache.exists() and split_file.exists():
        print(f"Loading cached splits from {cache_dir}")
        return (
            hf_datasets.load_from_disk(str(trainval_cache)),
            hf_datasets.load_from_disk(str(test_cache)),
        )

    all_files = sorted(Path(npz_dir).glob("conv_*.npz"))
    n = len(all_files)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    test_idx = sorted(perm[:test_size].tolist())
    trainval_idx = sorted(perm[test_size:].tolist())

    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(split_file, "w") as f:
        json.dump({"test": test_idx, "trainval": trainval_idx}, f)
    print(f"Split: {len(trainval_idx):,} trainval + {len(test_idx):,} test conversations saved to {split_file}")

    def build_hf_dataset(indices, desc):
        embeddings = []
        for idx in tqdm(indices, desc=desc):
            d = np.load(all_files[idx])
            embeddings.append(d["embeddings"].astype(np.float32))
        return hf_datasets.Dataset.from_dict(
            {"embeddings": embeddings},
            features=hf_datasets.Features(
                {"embeddings": hf_datasets.Array2D(shape=(None, EMB_DIM_QWEN3_4B), dtype="float32")}
            ),
        )

    print("Building trainval HF dataset (one-time, cached)...")
    trainval_ds = build_hf_dataset(trainval_idx, "trainval")
    trainval_ds.save_to_disk(str(trainval_cache))

    print("Building test HF dataset (one-time, cached)...")
    test_ds = build_hf_dataset(test_idx, "test")
    test_ds.save_to_disk(str(test_cache))

    return trainval_ds, test_ds


# Define the transformation function
def transform_samples_wrapper(start, step, use_residuals=False):
    def transform_samples(batch):
        embeddings = batch['embeddings']
        batched = isinstance(embeddings, list) or (len(embeddings.shape) == 3)
        if not batched:
            embeddings = [embeddings]
        inputs = []
        outputs = []
        for d in embeddings:
            inputs += [d[i-step] for i in range(start + step, len(d), 2)]
            if use_residuals:
                outputs += [d[i] - d[i-step] for i in range(start + step, len(d), 2)]
            else:
                outputs += [d[i] for i in range(start + step, len(d), 2)]
        transformed_samples = {
            'inputs': inputs,
            'outputs': outputs
        }

        return transformed_samples
    return transform_samples

def calculate_mean(batch):
    input_sum = batch['inputs'].sum(dim=0)
    output_sum = batch['outputs'].sum(dim=0)
    return {'input_sum': [input_sum], 'output_sum': [output_sum]}

def sum_of_squared_diff(batch, input_mean, output_mean):
    input_squared_diff_sum = (batch['inputs'] - input_mean).square().sum(dim=0)
    output_squared_diff_sum = (batch['outputs'] - output_mean).square().sum(dim=0)
    return {"input_squared_diff_sum": [input_squared_diff_sum], "output_squared_diff_sum": [output_squared_diff_sum]}

def normalize_dataset(batch, input_mean, input_std, output_mean, output_std):
    inputs = (batch['inputs'] - input_mean) / input_std
    outputs = (batch['outputs'] - output_mean) / output_std
    return {'inputs': inputs, 'outputs': outputs}

def load_dataset(**kwargs) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Load dataset — NPZ path (Code-Feedback) or pre-built HF dataset (lmsys legacy)
    if kwargs.get("npz_dir"):
        npz_dir = Path(kwargs["npz_dir"])
        cache_dir = Path(kwargs["hf_cache"]) if kwargs.get("hf_cache") else npz_dir.parent / "hf_cache"
        trainval_ds, _ = prepare_npz_splits(
            npz_dir, cache_dir,
            test_size=kwargs.get("test_size", 1000),
            seed=kwargs["seed"],
        )
        hf_dataset = trainval_ds.with_format("torch")
    else:
        hf_dataset = load_from_disk(kwargs["dataset"]).with_format("torch")
    # Cache the pair dataset keyed by (transition_type, residuals) so the map only runs
    # once per type. Without caching, the second transition type triggers map with
    # num_proc=32 after CUDA is already initialised — fork + CUDA = deadlock on Linux.
    use_residuals = not kwargs['not_residuals']
    pairs_tag = f"pairs_{kwargs['transition_type']}_res{int(use_residuals)}_seed{kwargs['seed']}"
    if kwargs.get("npz_dir"):
        pairs_cache = Path(kwargs["hf_cache"]) / pairs_tag if kwargs.get("hf_cache") \
            else Path(kwargs["npz_dir"]).parent / "hf_cache" / pairs_tag
    else:
        pairs_cache = Path(kwargs["dataset"]).parent / pairs_tag

    if pairs_cache.exists():
        print(f"Loading cached pairs dataset from {pairs_cache}")
        hf_dataset = hf_datasets.load_from_disk(str(pairs_cache))
    else:
        hf_dataset = hf_dataset.map(
            transform_samples_wrapper(kwargs['start'], kwargs['step'], use_residuals=use_residuals),
            remove_columns=hf_dataset.column_names, batched=True, batch_size=2000,
            num_proc=32,
        )
        hf_dataset.save_to_disk(str(pairs_cache))

    hf_dataset = hf_dataset.train_test_split(test_size=0.1, seed=kwargs['seed'], shuffle=True)
    print(f"length of dataset {len(hf_dataset['train']) + len(hf_dataset['test'])}, with train length {len(hf_dataset['train'])} and test length {len(hf_dataset['test'])}")

    # Normalize input and output embeddings for training stability
    print("Calculating mean and std of inputs and outputs...")
    sums_dataset = hf_dataset["train"].map(
        calculate_mean, 
        remove_columns=hf_dataset["train"].column_names, batched=True, batch_size=1000, 
        )
    input_mean = sum(sums_dataset["input_sum"]) / len(hf_dataset["train"])
    output_mean = sum(sums_dataset["output_sum"]) / len(hf_dataset["train"])
    squared_diff = hf_dataset["train"].map(
        sum_of_squared_diff, 
        remove_columns=hf_dataset["train"].column_names, batched=True, batch_size=1000, 
        fn_kwargs={'input_mean': input_mean, 'output_mean': output_mean})
    input_std = torch.sqrt(sum(squared_diff["input_squared_diff_sum"]) / len(hf_dataset["train"]))
    output_std = torch.sqrt(sum(squared_diff["output_squared_diff_sum"]) / len(hf_dataset["train"]))

    normalized_train = hf_dataset["train"].map(normalize_dataset, fn_kwargs={
        'input_mean': input_mean, 'input_std': input_std, 'output_mean': output_mean, 'output_std': output_std
    }, batched=True, batch_size=10000)
    normalized_test = hf_dataset["test"].map(normalize_dataset, fn_kwargs={
        'input_mean': input_mean, 'input_std': input_std, 'output_mean': output_mean, 'output_std': output_std
    }, batched=True, batch_size=10000)

    print("Mean and std calculated.")

    # Convert custom dataset to DataLoader for batching
    train_dataset = DataLoader(
        normalized_train, 
        batch_size=kwargs["batch_size"],
    )
    val_dataset = DataLoader(
        normalized_test, 
        batch_size=8192,
    )

    return train_dataset, val_dataset, input_mean, input_std, output_mean, output_std

def initialize_model(device, **kwargs):
    print(f"Initializing model... on device {device}")

    torch.manual_seed(kwargs["seed"])
    model = HeirarchicalMoE(dim=kwargs.get("emb_dim", 1024))

    model.to(device)

    print(model)
    print("Model initialized.")
    return model

def train_transition_model(**kwargs):
    seed=kwargs["seed"]
    epochs=kwargs["epochs"]
    lr=kwargs["lr"]
    gamma=kwargs["gamma"]
    batch_size=kwargs["batch_size"]
    transition_type=kwargs["transition_type"]
    use_wandb = kwargs.get("use_wandb", False)

    device = 0

    outdir = f"{kwargs['out_dir']}/seed_{seed}_batch_{batch_size}/{transition_type}"
    Path(outdir).mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, input_mean, input_std, output_mean, output_std = load_dataset(**kwargs)

    model = initialize_model(device = device, **kwargs)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Initialize wandb
    if use_wandb:
        run = wandb.init(project=kwargs["wandb_proj"], name=f"seed_{seed}_{transition_type}", config=kwargs)
        run.save("train_transition_distributed.py")
        run.watch(model, log="all", log_graph=True, criterion=criterion)

    if kwargs["continue_from"] is not None:
        checkpoint = torch.load(kwargs["continue_from"])
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict']["model"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        start_epoch = 0

    # Training loop
    min_val_loss = float('inf')
    min_val_loss_epoch = 0
    min_train_loss = float('inf')
    min_train_loss_epoch = 0
    regression_model = RegressionWrapper(model, embedding_size = input_mean.size(0))
    regression_model.set_parameters(input_mean, input_std, output_mean, output_std, use_residuals = not kwargs['not_residuals'])
    for epoch in tqdm(range(start_epoch, epochs), leave=True):
        train_loss = 0.0
        aux_loss = 0.0

        # Training loop
        model.train()
        for batch_no, batch in enumerate(tqdm(train_dataset, leave=True, mininterval=10.0)):
            inputs = batch["inputs"].to(device)
            targets = batch["outputs"].to(device)

            optimizer.zero_grad()   # Zero the gradient buffers

            outputs, curr_aux_loss = model(inputs[:, None]) # Forward pass
            loss = criterion(outputs[:,0,:], targets) # Compute the loss

            (loss + curr_aux_loss).backward() # Backward pass

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step() # Update the weights

            train_loss += loss.item()
            aux_loss += curr_aux_loss.item()

            fractional_epoch = epoch + batch_no / len(train_dataset)
            if batch_no % 100 == 0 and use_wandb and fractional_epoch > 0.1:
                run.log({
                    "Epoch": fractional_epoch, 
                    "Intermediate Training Loss": loss.item(), 
                    "Intermediate Aux Loss": curr_aux_loss.item(), 
                    "Intermediate Total Loss": loss.item() + curr_aux_loss.item()
                    })
        train_loss /= len(train_dataset)
        aux_loss /= len(train_dataset)


        # Validation loop
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in val_dataset:
                inputs = batch['inputs'].to(device)
                targets = batch['outputs'].to(device)

                outputs, aux_loss = model(inputs[:, None])
                val_loss += criterion(outputs[:,0,:], targets).item()
            val_loss /= len(val_dataset)
            tqdm.write(f'{transition_type.ljust(12)}Epoch {epoch + 1}/{epochs}, lr: {lr_scheduler.get_last_lr()[0]:.3e}, Training loss: {train_loss:.5e}, Validation loss: {val_loss:.5e}')

        if use_wandb:
            # Log metrics to wandb
            run.log({
                f"Epoch": epoch+1, 
                f"Training Loss": train_loss, 
                f"Validation Loss": val_loss, 
                f"Aux Loss": aux_loss,
                "lr": lr_scheduler.get_last_lr()[0]
                })
        if True or epoch > 10:
            if val_loss < min_val_loss:
                min_val_loss_epoch = epoch
                min_val_loss = val_loss
                regression_model.model.load_state_dict(model.state_dict())
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': regression_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(checkpoint, f'{outdir}/model_min_val.pth')
            if train_loss < min_train_loss:
                min_train_loss_epoch = epoch
                min_train_loss = train_loss
                regression_model.model.load_state_dict(model.state_dict())
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': regression_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict()
                }
                torch.save(checkpoint, f'{outdir}/model_min_train.pth')
        lr_scheduler.step()

    print("Training complete.")
    print(f"Minimum validation loss: {min_val_loss:.5e} at epoch {min_val_loss_epoch}")
    print(f"Minimum training loss: {min_train_loss:.5e} at epoch {min_train_loss_epoch}")

    # Write to a txt file
    with open(f"{outdir}/results.txt", "w") as file:
        file.write("Training complete.\n")
        file.write(f"Minimum validation loss: {min_val_loss} at epoch {min_val_loss_epoch}\n")
        file.write(f"Minimum training loss: {min_train_loss} at epoch {min_train_loss_epoch}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Seed for random number generation", default=0)
    parser.add_argument("--epochs", type=int, help="Number of epochs for training", default=100)
    parser.add_argument("--lr", type=float, help="Learning rate for optimizer", default=0.001)
    parser.add_argument("--gamma", type=float, help="Exponential decay gamma for learning rate scheduler", default=0.9)
    parser.add_argument("--batch_size", type=int, help="Batch size for training", default=2048)
    parser.add_argument("--type_index", type=int, help="Index of the transition type to train", default=-1)
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--wandb_proj", type=str, help="Wandb project name", default="lm-sys_transition_moe")
    parser.add_argument("--dataset", type=str, help="dataset location", default="embeddings/lmsys-chat-1m_embeddings_1024")
    parser.add_argument("--not_residuals", action="store_true", help="Train on absolute embeddings instead of residuals")
    parser.add_argument("--out_dir", type=str, help="Output directory for models", default="transition_models/deterministic")
    parser.add_argument("--continue_from", type=str, help="Continue training from a checkpoint", default=None)
    parser.add_argument("--npz_dir", type=str, default=None, help="Directory of conv_*.npz files; overrides --dataset")
    parser.add_argument("--hf_cache", type=str, default=None, help="Cache dir for HF dataset conversion (default: <npz_dir>/../hf_cache)")
    parser.add_argument("--test_size", type=int, default=1000, help="Conversations held out as test set (conversation-level split)")
    parser.add_argument("--emb_dim", type=int, default=1024, help="Embedding dimension (1024 for lmsys, 2560 for Qwen3-4B/Code-Feedback)")
    args = vars(parser.parse_args())

    print(args)

    # Conversations start with human
    start_steps = [
        (1,1),
        (0,1),
        (1,2),
        (0,2)
    ]
    types = [
        "llm_human", 
        "human_llm", 
        # "llm_llm", 
        # "human_human"
    ]
    if args["type_index"] >= 0:
        start_steps = [start_steps[args["type_index"]]]
        types = [types[args["type_index"]]]

    if args["use_wandb"]:
        wandb.setup()
    for start_step, transition_type in zip(start_steps, types):
        args['transition_type'] = transition_type
        args['start'] = start_step[0]
        args['step'] = start_step[1]
        train_transition_model(**args)
    if args["use_wandb"]:
        wandb.finish()