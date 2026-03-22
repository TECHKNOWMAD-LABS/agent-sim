from __future__ import annotations

from dataclasses import dataclass

from ..agents.reactive import ReactiveAgent
from ..environment.grid import GridEnvironment


@dataclass
class PursuitResult:
    caught: bool
    steps: int
    min_distance: float


def make_predator(agent_id: str, position: tuple[int, int]) -> ReactiveAgent:
    """Reactive predator that always moves toward the prey."""
    agent = ReactiveAgent(agent_id, position)
    agent.add_rule(lambda obs: obs.get("prey_direction") == "up", "up")
    agent.add_rule(lambda obs: obs.get("prey_direction") == "down", "down")
    agent.add_rule(lambda obs: obs.get("prey_direction") == "left", "left")
    agent.add_rule(lambda obs: obs.get("prey_direction") == "right", "right")
    agent.set_default("stay")
    return agent


def make_prey(agent_id: str, position: tuple[int, int]) -> ReactiveAgent:
    """Reactive prey that always flees from the predator."""
    agent = ReactiveAgent(agent_id, position)
    agent.add_rule(lambda obs: obs.get("predator_direction") == "up", "down")
    agent.add_rule(lambda obs: obs.get("predator_direction") == "down", "up")
    agent.add_rule(lambda obs: obs.get("predator_direction") == "left", "right")
    agent.add_rule(lambda obs: obs.get("predator_direction") == "right", "left")
    agent.set_default("stay")
    return agent


def _direction(from_pos: tuple[int, int], to_pos: tuple[int, int]) -> str | None:
    r1, c1 = from_pos
    r2, c2 = to_pos
    if abs(r2 - r1) > abs(c2 - c1):
        return "up" if r2 < r1 else "down"
    if c2 != c1:
        return "left" if c2 < c1 else "right"
    return None


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


class PursuitScenario:
    """Predator-prey pursuit: predator starts at (0,0), prey at opposite corner."""

    def __init__(
        self,
        grid_size: int = 10,
        max_steps: int = 100,
        seed: int = 0,
    ) -> None:
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.env = GridEnvironment(
            rows=grid_size,
            cols=grid_size,
            food_count=0,
            wall_density=0.05,
            seed=seed,
        )
        self.predator: ReactiveAgent | None = None
        self.prey: ReactiveAgent | None = None

    def setup(self) -> None:
        self.env._agents.clear()
        pred_start = (0, 0)
        prey_start = (self.grid_size - 1, self.grid_size - 1)
        self.env.add_agent("predator", pred_start)
        self.env.add_agent("prey", prey_start)
        self.env.reset()
        # Pin agents to their canonical start positions after reset
        self.env._agents["predator"].position = pred_start
        self.env._agents["prey"].position = prey_start
        self.predator = make_predator("predator", pred_start)
        self.prey = make_prey("prey", prey_start)

    def run(self) -> PursuitResult:
        self.setup()
        assert self.predator is not None
        assert self.prey is not None

        min_dist = float("inf")
        for step in range(self.max_steps):
            pred_pos = self.env._agents["predator"].position
            prey_pos = self.env._agents["prey"].position
            dist = _manhattan(pred_pos, prey_pos)
            min_dist = min(min_dist, dist)

            if dist <= 1.0:
                return PursuitResult(caught=True, steps=step, min_distance=min_dist)

            pred_obs = self.env.get_observation("predator")
            pred_obs["prey_direction"] = _direction(pred_pos, prey_pos)
            prey_obs = self.env.get_observation("prey")
            prey_obs["predator_direction"] = _direction(prey_pos, pred_pos)

            pred_action = self.predator.step(pred_obs)
            prey_action = self.prey.step(prey_obs)
            self.env.step("predator", pred_action)
            self.env.step("prey", prey_action)
            self.predator.receive_reward(-0.01)
            self.prey.receive_reward(-0.01)

        return PursuitResult(caught=False, steps=self.max_steps, min_distance=min_dist)
