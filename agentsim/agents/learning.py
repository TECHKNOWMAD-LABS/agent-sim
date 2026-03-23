from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

from .base import BaseAgent


class LearningAgent(BaseAgent):
    """Q-learning agent with epsilon-greedy exploration.

    State is derived from the agent's (row, col) position encoded as a flat index.

    Args:
        agent_id: Unique identifier string.
        position: Starting (row, col) position.
        state_size: Number of discrete states in the Q-table.  Must be >= 1.
        learning_rate: Alpha for Q-updates.  Must be in (0, 1].
        discount: Gamma discount factor.  Must be in [0, 1].
        epsilon: Initial exploration probability.  Must be in [0, 1].
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
        if not isinstance(state_size, int) or state_size < 1:
            raise ValueError(f"state_size must be >= 1, got {state_size!r}")
        if not isinstance(learning_rate, (int, float)) or not (0 < learning_rate <= 1):
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate!r}")
        if not isinstance(discount, (int, float)) or not (0.0 <= discount <= 1.0):
            raise ValueError(f"discount must be in [0, 1], got {discount!r}")
        if not isinstance(epsilon, (int, float)) or not (0.0 <= epsilon <= 1.0):
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon!r}")
        self.lr = float(learning_rate)
        self.gamma = float(discount)
        self.epsilon = float(epsilon)
        self.q_table: np.ndarray = np.zeros((state_size, len(self.ACTIONS)))
        self._state_key: int = 0
        self._last_action_idx: int = 0

    def _obs_to_state(self, obs: dict[str, Any]) -> int:
        pos = obs.get("position", (0, 0))
        try:
            x, y = int(pos[0]), int(pos[1])
        except (TypeError, IndexError, ValueError):
            x, y = 0, 0
        grid_size = int(obs.get("grid_size", 8))
        return (x * grid_size + y) % self.q_table.shape[0]

    def perceive(self, observation: dict[str, Any]) -> None:
        self._last_observation = observation if observation is not None else {}
        self._state_key = self._obs_to_state(self._last_observation)

    def decide(self) -> str:
        if random.random() < self.epsilon:
            idx = random.randrange(len(self.ACTIONS))
        else:
            idx = int(np.argmax(self.q_table[self._state_key]))
        self._last_action_idx = idx
        return self.ACTIONS[idx]

    def update(self, reward: float, next_obs: dict[str, Any]) -> None:
        """Apply a Q-learning update after receiving a reward.

        Args:
            reward: The reward received.  Non-finite values are treated as 0.
            next_obs: Observation after the action, used to compute max next Q.
        """
        if not isinstance(reward, (int, float)) or not math.isfinite(reward):
            reward = 0.0
        next_obs = next_obs if next_obs is not None else {}
        next_state = self._obs_to_state(next_obs)
        old_q = self.q_table[self._state_key, self._last_action_idx]
        max_next = float(np.max(self.q_table[next_state]))
        self.q_table[self._state_key, self._last_action_idx] = (
            old_q + self.lr * (reward + self.gamma * max_next - old_q)
        )
        self.receive_reward(reward)

    def decay_epsilon(self, factor: float = 0.99, minimum: float = 0.01) -> None:
        """Decay exploration rate by *factor*, clamped to *minimum*.

        Args:
            factor: Multiplicative decay factor.  Must be in (0, 1].
            minimum: Floor value for epsilon.  Must be >= 0.
        """
        if not isinstance(factor, (int, float)) or not (0 < factor <= 1.0):
            raise ValueError(f"factor must be in (0, 1], got {factor!r}")
        if not isinstance(minimum, (int, float)) or minimum < 0:
            raise ValueError(f"minimum must be >= 0, got {minimum!r}")
        self.epsilon = max(float(minimum), self.epsilon * float(factor))
