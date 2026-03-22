from __future__ import annotations

from typing import Any

from .base import BaseAgent


class Goal:
    """A named goal with a mutable priority and achieved flag."""

    def __init__(self, name: str, priority: float = 1.0) -> None:
        self.name = name
        self.priority = priority
        self.achieved = False

    def __repr__(self) -> str:
        return f"Goal({self.name!r}, priority={self.priority}, achieved={self.achieved})"


class DeliberativeAgent(BaseAgent):
    """Simplified BDI agent: maintains beliefs, desires (goals), and intentions (plan)."""

    def __init__(self, agent_id: str, position: tuple[int, int]) -> None:
        super().__init__(agent_id, position)
        self.beliefs: dict[str, Any] = {}
        self.goals: list[Goal] = []
        self._plan: list[str] = []

    def add_goal(self, goal: Goal) -> None:
        self.goals.append(goal)
        self.goals.sort(key=lambda g: g.priority, reverse=True)

    def perceive(self, observation: dict[str, Any]) -> None:
        self._last_observation = observation
        self.beliefs.update(observation)
        self._revise_goals()
        if not self._plan:
            self._plan = self._make_plan()

    def _revise_goals(self) -> None:
        for goal in self.goals:
            if goal.name == "find_food" and self.beliefs.get("food_nearby"):
                goal.priority = 2.0
            elif goal.name == "explore":
                goal.priority = 1.0

    def _make_plan(self) -> list[str]:
        active = [g for g in self.goals if not g.achieved]
        if not active:
            return ["stay"]
        top = active[0]
        if top.name == "find_food":
            return self._plan_to_food()
        if top.name == "explore":
            return self._plan_explore()
        return ["stay"]

    def _plan_to_food(self) -> list[str]:
        food_dir = self.beliefs.get("food_direction")
        if food_dir:
            return [food_dir, "collect"]
        return ["stay"]

    def _plan_explore(self) -> list[str]:
        import random

        return [random.choice(("up", "down", "left", "right"))]

    def decide(self) -> str:
        if self._plan:
            return self._plan.pop(0)
        self._plan = self._make_plan()
        return self._plan.pop(0) if self._plan else "stay"
