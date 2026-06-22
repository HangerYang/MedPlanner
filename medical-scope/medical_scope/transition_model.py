from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn


def _as_float_tuple(value):
    return tuple(np.asarray(value, dtype=np.float32).reshape(-1).tolist())


class HierarchicalMoE(nn.Module):
    """Minimal loader-compatible implementation for saved medical SCOPE MoE weights."""

    def __init__(self, dim=2560, outer_experts=4, inner_experts=4, hidden=10240) -> None:
        super().__init__()
        self.gate_outer = nn.Module()
        self.gate_outer.w_gating = nn.Parameter(torch.empty(dim, outer_experts))
        self.gate_inner = nn.Module()
        self.gate_inner.w_gating = nn.Parameter(torch.empty(outer_experts, dim, inner_experts))
        self.experts = nn.Module()
        self.experts.w1 = nn.Parameter(torch.empty(outer_experts, inner_experts, dim, hidden))
        self.experts.w2 = nn.Parameter(torch.empty(outer_experts, inner_experts, hidden, dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for param in self.parameters():
            nn.init.normal_(param, std=0.02)

    def forward(self, x):
        # x: batch x tokens x dim. RegressionWrapper calls with tokens=1.
        outer = torch.softmax(torch.einsum("btd,do->bto", x, self.gate_outer.w_gating), dim=-1)
        inner = torch.softmax(torch.einsum("btd,odi->btoi", x, self.gate_inner.w_gating), dim=-1)
        hidden = torch.relu(torch.einsum("btd,oidh->btoih", x, self.experts.w1))
        expert_out = torch.einsum("btoih,oihd->btoid", hidden, self.experts.w2)
        mixed_inner = (expert_out * inner.unsqueeze(-1)).sum(dim=3)
        mixed_outer = (mixed_inner * outer.unsqueeze(-1)).sum(dim=2)
        return mixed_outer, None


class MixtureDensityNetwork(nn.Module):
    """Loader-compatible MDN for saved code-feedback transition checkpoints."""

    def __init__(self, dim_in, dim_out, n_components, hidden_dim) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_components = n_components
        self.pi_network = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, n_components),
        )
        self.normal_network = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 2 * dim_out * n_components),
        )

    def forward(self, x, eps=1e-6):
        log_pi = torch.log_softmax(self.pi_network(x), dim=-1)
        normal_params = self.normal_network(x)
        mu = normal_params[..., : self.dim_out * self.n_components]
        sigma = normal_params[..., self.dim_out * self.n_components :]
        mu = mu.reshape(-1, self.n_components, self.dim_out)
        sigma = torch.exp(sigma + eps).reshape(-1, self.n_components, self.dim_out)
        return log_pi, mu, sigma

    def sample(self, x, samples_per_input=1):
        log_pi, mu, sigma = self.forward(x)
        cum_pi = torch.cumsum(torch.exp(log_pi), dim=-1)
        rvs = torch.rand([*x.shape[:-1], samples_per_input], device=x.device)
        rand_pi = torch.searchsorted(cum_pi, rvs).unsqueeze(-1)
        rand_pi = torch.clamp(rand_pi, 0, self.n_components - 1)
        rand_mu = torch.take_along_dim(mu, indices=rand_pi, dim=1)
        rand_sigma = torch.take_along_dim(sigma, indices=rand_pi, dim=1)
        samples = rand_mu + rand_sigma * torch.randn_like(rand_mu)
        return samples.permute(-2, *tuple(range(len(samples.shape) - 2)), -1)


class RegressionWrapper(nn.Module):
    def __init__(self, model, embedding_size=2560):
        super().__init__()
        self.add_module("model", model)
        self.input_mean = nn.Parameter(torch.zeros(embedding_size), requires_grad=False)
        self.input_std = nn.Parameter(torch.ones(embedding_size), requires_grad=False)
        self.output_mean = nn.Parameter(torch.zeros(embedding_size), requires_grad=False)
        self.output_std = nn.Parameter(torch.ones(embedding_size), requires_grad=False)
        self.use_residuals = nn.Parameter(torch.tensor(True, dtype=torch.bool), requires_grad=False)

    def forward(self, x):
        scaled_x = (x - self.input_mean) / self.input_std
        y = self.model(scaled_x[:, None])[0][:, 0, :]
        y = y * self.output_std + self.output_mean
        if bool(self.use_residuals.detach().cpu().item()):
            y = x + y
        return y


class MDNRegressionWrapper(RegressionWrapper):
    def sample(self, x, samples_per_input=1):
        scaled_x = (x - self.input_mean) / self.input_std
        y = self.model.sample(scaled_x, samples_per_input=samples_per_input)
        y = y * self.output_std + self.output_mean
        if bool(self.use_residuals.detach().cpu().item()):
            y = x + y
        return y


class TransitionModelMOE:
    def __init__(self, samples=4, noise=0.005, cuda="cpu", transition_model_dir="scope_saved/transition_models") -> None:
        self.samples = int(samples)
        self.std = float(noise)
        self.cuda = torch.device(cuda if torch.cuda.is_available() or str(cuda) == "cpu" else "cpu")
        self.llm_models = []
        self.human_models = []
        print(f"Loading transition models on device {self.cuda}...")
        self._load_models(Path(transition_model_dir))
        if not self.llm_models or not self.human_models:
            raise FileNotFoundError(f"No transition models loaded from {transition_model_dir}")
        print(f"Loaded {len(self.llm_models)} LLM models and {len(self.human_models)} human models on device {self.cuda}.")

    def _model_from_checkpoint(self, path: Path):
        checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        dim = int(state["input_mean"].numel())
        outer = int(state["model.gate_outer.w_gating"].shape[1])
        inner = int(state["model.gate_inner.w_gating"].shape[2])
        hidden = int(state["model.experts.w1"].shape[-1])
        model = RegressionWrapper(HierarchicalMoE(dim, outer, inner, hidden), embedding_size=dim).float()
        model.load_state_dict(state, strict=True)
        model.to(self.cuda)
        model.eval()
        return model

    def _load_models(self, root: Path) -> None:
        roots = [root]
        roots.extend([p for p in root.iterdir() if p.is_dir()] if root.exists() else [])
        seen = set()
        for base in roots:
            for kind, target in (("human_llm", self.llm_models), ("llm_human", self.human_models)):
                path = base / kind / "model_min_train.pth"
                if not path.exists():
                    path = base / kind / "model_min_val.pth"
                if path.exists() and path not in seen:
                    target.append(self._model_from_checkpoint(path))
                    seen.add(path)

    def forward(self, input_tensor, models):
        next_states = []
        for model in models:
            with torch.no_grad():
                next_states.append(model(input_tensor.to(self.cuda)).cpu())
        next_states = torch.stack(next_states)
        if len(next_states) == 1:
            noise = torch.randn(self.samples, *next_states.shape) * self.std
            perturbed = next_states.repeat(self.samples + 1, *([1] * len(next_states.shape)))
            perturbed[1:] += noise
            return perturbed
        return next_states

    def transit(self, state, action):
        state_t = torch.tensor(state, dtype=torch.float32)
        action_t = torch.tensor(action, dtype=torch.float32)
        intermediate = state_t + action_t
        perturbed = self.forward(intermediate.unsqueeze(0), self.human_models)[:, 0, :]
        return [_as_float_tuple(row.numpy()) for row in perturbed]

    def sample_actions(self, state):
        state_t = torch.tensor(state, dtype=torch.float32)
        perturbed = self.forward(state_t.unsqueeze(0), self.llm_models)[:, 0, :]
        actions = perturbed - state_t
        return [_as_float_tuple(row.numpy()) for row in actions]


class TransitionModelMDN:
    def __init__(self, samples=4, noise=0.005, cuda="cpu", transition_model_dir="scope_saved/transition_models") -> None:
        self.samples = int(samples)
        self.std = float(noise)
        self.cuda = torch.device(cuda if torch.cuda.is_available() or str(cuda) == "cpu" else "cpu")
        root = Path(transition_model_dir)
        print(f"Loading MDN transition models on device {self.cuda}...")
        self.llm_model = self._load_direction(root, "human_llm")
        self.human_model = self._load_direction(root, "llm_human")
        print(f"Loaded 1 MDN LLM model and 1 MDN human model on device {self.cuda}.")

    def _model_from_checkpoint(self, path: Path):
        checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        dim = int(state["input_mean"].numel())
        hidden = int(state["model.pi_network.0.weight"].shape[0])
        n_components = int(state["model.pi_network.6.weight"].shape[0])
        model = MDNRegressionWrapper(
            MixtureDensityNetwork(dim, dim, n_components, hidden),
            embedding_size=dim,
        ).float()
        model.load_state_dict(state, strict=True)
        model.to(self.cuda)
        model.eval()
        return model

    def _load_direction(self, root: Path, kind: str):
        path = root / kind / "model_min_train.pth"
        if not path.exists():
            path = root / kind / "model_min_val.pth"
        if not path.exists():
            raise FileNotFoundError(f"No MDN transition checkpoint found under {root / kind}")
        return self._model_from_checkpoint(path)

    def forward(self, input_tensor, model):
        with torch.no_grad():
            return model.sample(input_tensor.to(self.cuda), samples_per_input=self.samples).cpu()

    def transit(self, state, action):
        state_t = torch.tensor(state, dtype=torch.float32)
        action_t = torch.tensor(action, dtype=torch.float32)
        intermediate = state_t + action_t
        perturbed = self.forward(intermediate.unsqueeze(0), self.human_model)[:, 0, :]
        return [_as_float_tuple(row.numpy()) for row in perturbed]

    def sample_actions(self, state):
        state_t = torch.tensor(state, dtype=torch.float32)
        perturbed = self.forward(state_t.unsqueeze(0), self.llm_model)[:, 0, :]
        actions = perturbed - state_t
        return [_as_float_tuple(row.numpy()) for row in actions]
