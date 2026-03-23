from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..agents.reactive import ReactiveAgent
from ..environment.grid import GridEnvironment


@dataclass
class ForagingResult:
    """Result of a completed ForagingScenario run.

    Attributes:
        total_collected: Total food items collected by all agents.
        steps: Number of environment steps taken.
        efficiency: Ratio of items collected to steps taken.
        agent_results: Mapping of agent_id to number of items collected.
    """

    total_collected: int
    steps: int
    efficiency: float
    agent_results: dict[str, int]


def make_forager(agent_id: str, position: tuple[int, int]) -> ReactiveAgent:
    """Create a reactive agent wired for food collection.

    The agent uses condition-action rules in priority order:
    1. Collect food if currently on a food cell.
    2. Move toward the nearest food using the compass direction provided in
       the observation.
    3. Stay if no food signal is present.

    Args:
        agent_id: Unique string identifier for the agent.
        position: Starting (row, col) position.

    Returns:
        A fully configured ReactiveAgent.
    """
    agent = ReactiveAgent(agent_id, position)
    agent.add_rule(lambda obs: obs.get("food_nearby", False), "collect")
    agent.add_rule(lambda obs: obs.get("food_direction") == "up", "up")
    agent.add_rule(lambda obs: obs.get("food_direction") == "down", "down")
    agent.add_rule(lambda obs: obs.get("food_direction") == "left", "left")
    agent.add_rule(lambda obs: obs.get("food_direction") == "right", "right")
    agent.set_default("stay")
    return agent


class ForagingScenario:
    """Multiple reactive agents competing to collect food on a grid.

    Args:
        n_agents: Number of forager agents.
        grid_size: Side length of the square grid.
        food_count: Number of food items placed on reset.
        max_steps: Step budget before the episode is cut off.
        seed: Random seed for reproducible grid layouts.
    """

    def __init__(
        self,
        n_agents: int = 2,
        grid_size: int = 8,
        food_count: int = 10,
        max_steps: int = 200,
        seed: int = 42,
    ) -> None:
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.env = GridEnvironment(
            rows=grid_size,
            cols=grid_size,
            food_count=food_count,
            wall_density=0.1,
            seed=seed,
        )
        self.agents: list[ReactiveAgent] = []

    def setup(self) -> dict[str, Any]:
        """Register agents and reset the environment.

        Returns:
            Initial observation map keyed by agent_id.
        """
        self.env._agents.clear()
        for i in range(self.n_agents):
            self.env.add_agent(f"agent_{i}")
        obs_map = self.env.reset()
        self.agents = [
            make_forager(f"agent_{i}", self.env._agents[f"agent_{i}"].position)
            for i in range(self.n_agents)
        ]
        return obs_map

    def run(self) -> ForagingResult:
        """Execute the scenario and return aggregate results.

        Returns:
            ForagingResult with collection counts and efficiency metrics.
        """
        obs_map = self.setup()
        done = False
        step = 0
        while not done and step < self.max_steps:
            for agent in self.agents:
                obs = obs_map.get(agent.id, {})
                action = agent.step(obs)
                new_obs, reward, done = self.env.step(agent.id, action)
                agent.receive_reward(reward)
                obs_map[agent.id] = new_obs
                if done:
                    break
            step += 1

        total = sum(self.env._agents[a.id].collected for a in self.agents)
        efficiency = total / max(step, 1)
        return ForagingResult(
            total_collected=total,
            steps=step,
            efficiency=efficiency,
            agent_results={a.id: self.env._agents[a.id].collected for a in self.agents},
        )
