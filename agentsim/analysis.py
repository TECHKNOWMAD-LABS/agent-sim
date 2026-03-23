from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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
    """Aggregate metrics across all episodes.

    Args:
        results: List of EpisodeResult objects from a simulation run.
        agents: Mapping of agent_id to BaseAgent instances.

    Returns:
        SimulationMetrics with per-agent and aggregate statistics.
    """
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
    """Position and reward statistics over an agent's recorded history.

    Results for the same agent object are cached after the first call.
    The cache is keyed on the current history length; if the agent has
    stepped further the cache is bypassed (history length changed).

    Args:
        agent: Any BaseAgent with a populated history list.

    Returns:
        Dict of statistics, or empty dict if no history.
    """
    history = agent.history
    if not history:
        return {}
    return _trajectory_stats_cached(
        tuple((h.position, h.reward) for h in history)
    )


@lru_cache(maxsize=256)
def _trajectory_stats_cached(
    history_snapshot: tuple[tuple[tuple[int, int], float], ...],
) -> dict[str, Any]:
    """Internal LRU-cached computation kernel for trajectory statistics."""
    xs = [h[0][0] for h in history_snapshot]
    ys = [h[0][1] for h in history_snapshot]
    rewards = [h[1] for h in history_snapshot]

    return {
        "n_steps": len(history_snapshot),
        "mean_x": float(np.mean(xs)),
        "mean_y": float(np.mean(ys)),
        "std_x": float(np.std(xs)),
        "std_y": float(np.std(ys)),
        "total_reward": float(history_snapshot[-1][1]),
        "reward_variance": float(np.var(rewards)),
        "unique_positions": len(set(zip(xs, ys))),
    }


def run_episodes_parallel(
    sim_factory: Any,
    n_episodes: int,
    max_workers: int = 4,
) -> list[EpisodeResult]:
    """Run independent simulation episodes in parallel using ThreadPoolExecutor.

    Args:
        sim_factory: Callable that returns a fresh (env, agents, config) tuple.
            Called once per episode; must be thread-safe.
        n_episodes: Total number of episodes to run.
        max_workers: Thread-pool size (default 4, capped at n_episodes).

    Returns:
        List of EpisodeResult objects, one per episode, in episode order.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from .simulation import Simulation

    actual_workers = min(max_workers, n_episodes)
    results: list[EpisodeResult] = [None] * n_episodes  # type: ignore[list-item]

    def _run_one(episode_idx: int) -> tuple[int, EpisodeResult]:
        env, agents, config = sim_factory()
        sim = Simulation(env, agents, config)
        return episode_idx, sim.run_episode(episode_idx)

    with ThreadPoolExecutor(max_workers=actual_workers) as pool:
        futures = {pool.submit(_run_one, i): i for i in range(n_episodes)}
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    return results
