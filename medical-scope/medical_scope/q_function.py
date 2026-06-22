from __future__ import annotations

from collections import defaultdict
from typing import Iterable


class AccumulatedRewardTable:
    """Tiny Q function used online by MCTS.

    It stores the mean observed rollout reward per semantic (state, action).
    This matches the medical runner's intended behavior: learn only within the
    current mediQ decision, then reset before the next expert turn.
    """

    def __init__(self, default: float = 0.0) -> None:
        self.default = default
        self.reset()

    def reset(self) -> None:
        self.reward_sum = defaultdict(float)
        self.reward_count = defaultdict(int)

    def _key(self, state, action):
        return (tuple(state.conversation), tuple(action))

    def update(self, state, action, delta, visits, reward) -> None:
        key = self._key(state, action)
        self.reward_sum[key] += float(reward)
        self.reward_count[key] += 1
        print("updating Q-function with reward: ", reward)

    def get_q_value(self, state, action) -> float:
        key = self._key(state, action)
        count = self.reward_count.get(key, 0)
        if count == 0:
            return self.default
        return self.reward_sum[key] / count

    def get_qs(self, state, actions) -> list[float]:
        qs = [self.get_q_value(state, action) for action in actions]
        print("q values estimate for actions are: ", qs)
        return qs
