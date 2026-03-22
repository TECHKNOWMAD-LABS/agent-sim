from __future__ import annotations

from typing import Any, Callable

from .base import BaseAgent

# A rule is a (condition, action) pair
Rule = tuple[Callable[[dict[str, Any]], bool], str]


class ReactiveAgent(BaseAgent):
    """Stimulus-response agent driven by condition-action rules.

    Rules are evaluated in insertion order; the first matching rule fires.
    Falls back to the default action when no rule matches.
    """

    def __init__(self, agent_id: str, position: tuple[int, int]) -> None:
        super().__init__(agent_id, position)
        self._rules: list[Rule] = []
        self._default_action = "stay"

    def add_rule(self, condition: Callable[[dict[str, Any]], bool], action: str) -> None:
        if action not in self.ACTIONS:
            raise ValueError(f"Unknown action '{action}'. Valid: {self.ACTIONS}")
        self._rules.append((condition, action))

    def set_default(self, action: str) -> None:
        if action not in self.ACTIONS:
            raise ValueError(f"Unknown action '{action}'. Valid: {self.ACTIONS}")
        self._default_action = action

    def perceive(self, observation: dict[str, Any]) -> None:
        self._last_observation = observation

    def decide(self) -> str:
        for condition, action in self._rules:
            if condition(self._last_observation):
                return action
        return self._default_action
