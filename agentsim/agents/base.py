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
        self.id = agent_id
        self.state = AgentState(position=position)
        self.history: list[AgentState] = []
        self._last_observation: dict[str, Any] = {}

    @abstractmethod
    def perceive(self, observation: dict[str, Any]) -> None:
        """Process incoming observation and update internal state."""

    @abstractmethod
    def decide(self) -> str:
        """Select an action based on current beliefs/state."""

    def step(self, observation: dict[str, Any]) -> str:
        """Record history, perceive, then decide."""
        self.history.append(self.state.copy())
        self.perceive(observation)
        return self.decide()

    def receive_reward(self, reward: float) -> None:
        self.state.reward += reward
        self.state.energy = max(0.0, self.state.energy - 0.01)
        self.state.step += 1

    def is_alive(self) -> bool:
        return self.state.energy > 0
