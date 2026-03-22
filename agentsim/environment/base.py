from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEnvironment(ABC):
    """Abstract interface every simulation environment must implement."""

    @abstractmethod
    def reset(self) -> dict[str, Any]:
        """Reset the environment and return initial observations keyed by agent_id."""

    @abstractmethod
    def step(self, agent_id: str, action: str) -> tuple[dict[str, Any], float, bool]:
        """Execute *action* for *agent_id*.

        Returns
        -------
        observation : dict[str, Any]
            New observation for the agent.
        reward : float
            Scalar reward signal.
        done : bool
            True when the episode has ended.
        """

    @abstractmethod
    def get_observation(self, agent_id: str) -> dict[str, Any]:
        """Return the current observation for *agent_id* without advancing time."""

    @abstractmethod
    def render(self) -> str:
        """Return a human-readable string representation of the environment."""
