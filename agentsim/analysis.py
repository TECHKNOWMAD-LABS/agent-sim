from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .agents.base import BaseAgent
from .simulation import EpisodeResult


@dataclass
class AgentMetrics:
    agent_id: str
    total_steps: int
    total_reward: float
    avg_reward_per_step: float
    max_reward: float
    min_reward: float
    alive: bool


@dataclass
class SimulationMetrics:
    n_episodes: int
    avg_steps: float
    completion_rate: float
    agent_metrics: dict[str, AgentMetrics]
    reward_trend: list[float]


def compute_metrics(
    results: list[EpisodeResult],
    agents: dict[str, BaseAgent],
) -> SimulationMetrics:
    """Aggregate metrics across all episodes."""
    n = len(results)
    if n == 0:
        return SimulationMetrics(
            n_episodes=0,
            avg_steps=0.0,
            completion_rate=0.0,
            agent_metrics={},
            reward_trend=[],
        )

    avg_steps = float(np.mean([r.steps for r in results]))
    completion_rate = sum(1 for r in results if r.done) / n

    agent_metrics: dict[str, AgentMetrics] = {}
    for aid, agent in agents.items():
        rewards = [r.total_reward.get(aid, 0.0) for r in results]
        steps = sum(r.steps for r in results)
        total_reward = sum(rewards)
        agent_metrics[aid] = AgentMetrics(
            agent_id=aid,
            total_steps=steps,
            total_reward=total_reward,
            avg_reward_per_step=total_reward / max(steps, 1),
            max_reward=max(rewards),
            min_reward=min(rewards),
            alive=agent.is_alive(),
        )

    reward_trend = [
        float(np.mean([r.total_reward.get(aid, 0.0) for aid in agents]))
        for r in results
    ]

    return SimulationMetrics(
        n_episodes=n,
        avg_steps=avg_steps,
        completion_rate=completion_rate,
        agent_metrics=agent_metrics,
        reward_trend=reward_trend,
    )


def compute_trajectory_stats(agent: BaseAgent) -> dict[str, Any]:
    """Position and reward statistics over an agent's recorded history."""
    history = agent.history
    if not history:
        return {}

    xs = [h.position[0] for h in history]
    ys = [h.position[1] for h in history]
    rewards = [h.reward for h in history]

    return {
        "n_steps": len(history),
        "mean_x": float(np.mean(xs)),
        "mean_y": float(np.mean(ys)),
        "std_x": float(np.std(xs)),
        "std_y": float(np.std(ys)),
        "total_reward": float(history[-1].reward),
        "reward_variance": float(np.var(rewards)),
        "unique_positions": len(set(zip(xs, ys))),
    }
