from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from conversation_feature import IncrementalFeatureState


class CodeFeatureReward:
    """Predict cumulative trajectory reward from independent turn embeddings."""

    def __init__(self, checkpoint_path: str, device: str) -> None:
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.feature_keys = checkpoint["feature_keys"]
        self.mean = torch.tensor(checkpoint["norm_mean"], dtype=torch.float32, device=self.device)
        self.std = torch.tensor(checkpoint["norm_std"], dtype=torch.float32, device=self.device)
        self.net = nn.Sequential(
            nn.Linear(26, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        ).to(self.device)
        state = checkpoint.get("model_state_dict", checkpoint.get("net", checkpoint))
        if any(str(key).startswith("net.") for key in state):
            state = {str(key).removeprefix("net."): value for key, value in state.items()}
        self.net.load_state_dict(state)
        self.net.eval()

    def value(self, embedding_history: tuple) -> float:
        if not embedding_history:
            return 0.0
        feature_state = IncrementalFeatureState()
        for index, embedding in enumerate(embedding_history):
            role = "user" if index % 2 == 0 else "agent"
            feature_state.add_turn(torch.tensor(np.asarray(embedding, dtype=np.float32)), role)
        features = torch.tensor(
            [feature_state._f[key] for key in self.feature_keys],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            return float(self.net(((features - self.mean) / self.std).unsqueeze(0)).item())

    def get_reward(self, previous_state, action, new_state) -> float:
        return self.value(new_state.embedding_history) - self.value(previous_state.embedding_history)
