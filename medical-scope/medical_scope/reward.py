from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from conversation_feature import IncrementalFeatureState  # noqa: E402


class FeatureScopeReward:
    """Reward model that operates on 26 conversation features instead of raw 2560-dim embeddings."""

    def __init__(self, path_to_model: str, device_map="cuda:0") -> None:
        print(f"[init] Loading feature reward MLP from {path_to_model} on {device_map}...")
        self.device = torch.device(device_map if torch.cuda.is_available() or str(device_map) == "cpu" else "cpu")
        checkpoint = torch.load(path_to_model, map_location=self.device, weights_only=False)
        self.feature_keys: list[str] = checkpoint["feature_keys"]
        self.norm_mean = torch.tensor(checkpoint["norm_mean"], dtype=torch.float32, device=self.device)
        self.norm_std = torch.tensor(checkpoint["norm_std"], dtype=torch.float32, device=self.device)
        n = len(self.feature_keys)
        self.net = nn.Sequential(
            nn.Linear(n, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        ).to(self.device)
        self.net.load_state_dict(checkpoint["net"])
        self.net.eval()

    def _features_from_history(self, embedding_history: tuple) -> dict[str, float]:
        state = IncrementalFeatureState()
        for i, emb in enumerate(embedding_history):
            role = "user" if i % 2 == 0 else "agent"
            t = torch.tensor(np.asarray(emb, dtype=np.float32), dtype=torch.float32)
            state.add_turn(t, role)
        return state._f

    def value(self, embedding_history: tuple) -> float:
        if len(embedding_history) < 2:
            return 0.0
        feats = self._features_from_history(embedding_history)
        x = torch.tensor([feats[k] for k in self.feature_keys], dtype=torch.float32, device=self.device)
        x = (x - self.norm_mean) / self.norm_std
        with torch.no_grad():
            return float(self.net(x.unsqueeze(0)).item())

    def get_reward(self, prev_state, action, new_state) -> float:
        prev_hist = prev_state.embedding_history if hasattr(prev_state, "embedding_history") else ()
        new_hist = new_state.embedding_history if hasattr(new_state, "embedding_history") else ()
        return self.value(new_hist) - self.value(prev_hist)


class EmbeddingScopeReward:
    def __init__(self, path_to_model: str, device_map="cuda:0") -> None:
        print(f"[init] Loading cumulative reward MLP (EmbeddingScopeReward) from {path_to_model} on {device_map}...")
        self.device = torch.device(device_map if torch.cuda.is_available() or str(device_map) == "cpu" else "cpu")
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
        ).to(self.device)
        state = torch.load(path_to_model, map_location=self.device)
        if any(str(key).startswith("net.") for key in state):
            state = {str(key).removeprefix("net."): value for key, value in state.items()}
        self.net.load_state_dict(state)
        self.net.eval()

    def value(self, embedding) -> float:
        tensor = torch.as_tensor(np.asarray(embedding, dtype=np.float32), dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            out = self.net(tensor).reshape(-1)
        if out.numel() == 1:
            return float(out[0].detach().cpu())
        return out.detach().cpu()

    def get_reward(self, prev_state, action, new_state) -> float:
        prev = prev_state.conversation if hasattr(prev_state, "conversation") else prev_state
        new = new_state.conversation if hasattr(new_state, "conversation") else new_state
        return float(self.value(new) - self.value(prev))
