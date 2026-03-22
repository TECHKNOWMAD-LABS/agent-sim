from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .agents.base import BaseAgent
from .environment.base import BaseEnvironment


@dataclass
class SimulationConfig:
    max_steps: int = 500
    n_episodes: int = 1
    render_every: int = 0  # 0 = never


@dataclass
class EpisodeResult:
    episode: int
    steps: int
    total_reward: dict[str, float]
    done: bool
    frames: list[str] = field(default_factory=list)


class Simulation:
    """Orchestrates a set of agents in an environment for one or more episodes.

    The environment must already have agents registered (via *add_agent* or
    equivalent) before this class is instantiated.
    """

    def __init__(
        self,
        env: BaseEnvironment,
        agents: list[BaseAgent],
        config: SimulationConfig | None = None,
    ) -> None:
        self.env = env
        self.agents: dict[str, BaseAgent] = {a.id: a for a in agents}
        self.config = config or SimulationConfig()
        self.results: list[EpisodeResult] = []

    def run_episode(self, episode: int = 0) -> EpisodeResult:
        obs_map: dict[str, Any] = self.env.reset()
        total_reward: dict[str, float] = {aid: 0.0 for aid in self.agents}
        done = False
        frames: list[str] = []
        step = 0

        while not done and step < self.config.max_steps:
            if self.config.render_every > 0 and step % self.config.render_every == 0:
                frames.append(self.env.render())

            for aid, agent in self.agents.items():
                obs = obs_map.get(aid, {})
                action = agent.step(obs)
                new_obs, reward, done = self.env.step(aid, action)
                agent.receive_reward(reward)
                total_reward[aid] += reward
                obs_map[aid] = new_obs
                if done:
                    break
            step += 1

        result = EpisodeResult(
            episode=episode,
            steps=step,
            total_reward=total_reward,
            done=done,
            frames=frames,
        )
        self.results.append(result)
        return result

    def run(self) -> list[EpisodeResult]:
        self.results = []
        for ep in range(self.config.n_episodes):
            self.run_episode(ep)
        return self.results

    def summary(self) -> dict[str, Any]:
        if not self.results:
            return {}
        n = len(self.results)
        avg_steps = sum(r.steps for r in self.results) / n
        avg_reward = {
            aid: sum(r.total_reward.get(aid, 0.0) for r in self.results) / n
            for aid in self.agents
        }
        return {
            "episodes": n,
            "avg_steps": avg_steps,
            "avg_reward": avg_reward,
            "completion_rate": sum(1 for r in self.results if r.done) / n,
        }
