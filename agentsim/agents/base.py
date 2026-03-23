from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    position: tuple[int, int]
    energy: float = 1.0
    reward: float = 0.0
    step: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> AgentState:
        return AgentState(
            position=self.position,
            energy=self.energy,
            reward=self.reward,
            step=self.step,
            metadata=dict(self.metadata),
        )


class BaseAgent(ABC):
    """Abstract base class for all simulation agents."""

    ACTIONS = ("up", "down", "left", "right", "stay", "collect")

    def __init__(self, agent_id: str, position: tuple[int, int]) -> None:
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError(f"agent_id must be a non-empty string, got {agent_id!r}")
        if (
            not isinstance(position, (tuple, list))
            or len(position) != 2
            or not all(isinstance(v, int) for v in position)
        ):
            raise ValueError(
                f"position must be a 2-tuple of ints, got {position!r}"
            )
        self.id = agent_id
        self.state = AgentState(position=tuple(position))  # type: ignore[arg-type]
        self.history: list[AgentState] = []
        self._last_observation: dict[str, Any] = {}

    @abstractmethod
    def perceive(self, observation: dict[str, Any]) -> None:
        """Process incoming observation and update internal state."""

    @abstractmethod
    def decide(self) -> str:
        """Select an action based on current beliefs/state."""

    def step(self, observation: dict[str, Any]) -> str:
        """Record history, perceive, then decide.

        Args:
            observation: Observation dict from the environment.  May be empty
                but must not be None.

        Returns:
            The chosen action string.
        """
        if observation is None:
            observation = {}
        self.history.append(self.state.copy())
        self.perceive(observation)
        return self.decide()

    def receive_reward(self, reward: float) -> None:
        """Apply a scalar reward and drain a fixed energy cost.

        Args:
            reward: Numeric reward value.  Non-finite values are treated as 0.
        """
        import math

        if not isinstance(reward, (int, float)) or not math.isfinite(reward):
            reward = 0.0
        self.state.reward += float(reward)
        self.state.energy = max(0.0, self.state.energy - 0.01)
        self.state.step += 1

    def is_alive(self) -> bool:
        return self.state.energy > 0
