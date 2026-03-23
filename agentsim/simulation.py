from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .agents.base import BaseAgent
from .environment.base import BaseEnvironment


@dataclass
class SimulationConfig:
    """Configuration knobs for a Simulation run.

    Attributes:
        max_steps: Hard cap on steps per episode.  Must be >= 1.
        n_episodes: Number of episodes to run.  Must be >= 1.
        render_every: Render the environment every N steps (0 = never).
    """

    max_steps: int = 500
    n_episodes: int = 1
    render_every: int = 0  # 0 = never

    def __post_init__(self) -> None:
        if not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps!r}")
        if not isinstance(self.n_episodes, int) or self.n_episodes < 1:
            raise ValueError(f"n_episodes must be >= 1, got {self.n_episodes!r}")
        if not isinstance(self.render_every, int) or self.render_every < 0:
            raise ValueError(f"render_every must be >= 0, got {self.render_every!r}")


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

    Args:
        env: The simulation environment.
        agents: List of agent objects.  Each agent's id must be unique.
        config: Optional simulation configuration; defaults to SimulationConfig().

    Raises:
        ValueError: If *agents* is empty.
    """

    def __init__(
        self,
        env: BaseEnvironment,
        agents: list[BaseAgent],
        config: SimulationConfig | None = None,
    ) -> None:
        if env is None:
            raise ValueError("env must not be None")
        if not agents:
            raise ValueError("agents list must contain at least one agent")
        self.env = env
        self.agents: dict[str, BaseAgent] = {a.id: a for a in agents}
        self.config = config or SimulationConfig()
        self.results: list[EpisodeResult] = []

    def run_episode(self, episode: int = 0) -> EpisodeResult:
        """Run a single episode and return the result.

        Args:
            episode: Episode index (used in EpisodeResult metadata).

        Returns:
            EpisodeResult for the completed episode.
        """
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
        """Run all configured episodes and return the list of results."""
        self.results = []
        for ep in range(self.config.n_episodes):
            self.run_episode(ep)
        return self.results

    def summary(self) -> dict[str, Any]:
        """Return aggregate statistics across all completed episodes.

        Returns:
            Empty dict if no episodes have been run yet.
        """
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
