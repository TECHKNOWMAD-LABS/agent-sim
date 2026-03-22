from __future__ import annotations

import random
from typing import Any

import numpy as np

from .base import BaseAgent


class LearningAgent(BaseAgent):
    """Q-learning agent with epsilon-greedy exploration.

    State is derived from the agent's (row, col) position encoded as a flat index.
    """

    def __init__(
        self,
        agent_id: str,
        position: tuple[int, int],
        state_size: int = 64,
        learning_rate: float = 0.1,
        discount: float = 0.9,
        epsilon: float = 0.2,
    ) -> None:
        super().__init__(agent_id, position)
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.q_table: np.ndarray = np.zeros((state_size, len(self.ACTIONS)))
        self._state_key: int = 0
        self._last_action_idx: int = 0

    def _obs_to_state(self, obs: dict[str, Any]) -> int:
        pos = obs.get("position", (0, 0))
        x, y = int(pos[0]), int(pos[1])
        grid_size = int(obs.get("grid_size", 8))
        return (x * grid_size + y) % self.q_table.shape[0]

    def perceive(self, observation: dict[str, Any]) -> None:
        self._last_observation = observation
        self._state_key = self._obs_to_state(observation)

    def decide(self) -> str:
        if random.random() < self.epsilon:
            idx = random.randrange(len(self.ACTIONS))
        else:
            idx = int(np.argmax(self.q_table[self._state_key]))
        self._last_action_idx = idx
        return self.ACTIONS[idx]

    def update(self, reward: float, next_obs: dict[str, Any]) -> None:
        """Apply a Q-learning update after receiving a reward."""
        next_state = self._obs_to_state(next_obs)
        old_q = self.q_table[self._state_key, self._last_action_idx]
        max_next = float(np.max(self.q_table[next_state]))
        self.q_table[self._state_key, self._last_action_idx] = (
            old_q + self.lr * (reward + self.gamma * max_next - old_q)
        )
        self.receive_reward(reward)

    def decay_epsilon(self, factor: float = 0.99, minimum: float = 0.01) -> None:
        self.epsilon = max(minimum, self.epsilon * factor)
