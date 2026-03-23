"""Shared fixtures and mock helpers for the agentsim test suite."""

from __future__ import annotations

from typing import Any

import pytest

from agentsim import (
    GridEnvironment,
    ReactiveAgent,
    Simulation,
    SimulationConfig,
)

# ---------------------------------------------------------------------------
# Agent fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def basic_reactive_agent() -> ReactiveAgent:
    """A ReactiveAgent with no rules at (0, 0)."""
    return ReactiveAgent("test_agent", (0, 0))


@pytest.fixture()
def forager_agent() -> ReactiveAgent:
    """A ReactiveAgent wired for food collection at (3, 3)."""
    agent = ReactiveAgent("forager", (3, 3))
    agent.add_rule(lambda obs: obs.get("food_nearby", False), "collect")
    agent.add_rule(lambda obs: obs.get("food_direction") == "up", "up")
    agent.add_rule(lambda obs: obs.get("food_direction") == "down", "down")
    agent.add_rule(lambda obs: obs.get("food_direction") == "left", "left")
    agent.add_rule(lambda obs: obs.get("food_direction") == "right", "right")
    agent.set_default("stay")
    return agent


# ---------------------------------------------------------------------------
# Environment fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_grid() -> GridEnvironment:
    """5x5 grid with no walls and no food — clean slate."""
    env = GridEnvironment(rows=5, cols=5, food_count=0, wall_density=0.0, seed=0)
    return env


@pytest.fixture()
def seeded_grid() -> GridEnvironment:
    """8x8 grid with deterministic seed, 5 food items."""
    env = GridEnvironment(rows=8, cols=8, food_count=5, seed=42)
    return env


# ---------------------------------------------------------------------------
# Simulation fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_sim() -> Simulation:
    """One forager agent on a 6x6 grid, 3 episodes."""
    env = GridEnvironment(rows=6, cols=6, food_count=3, seed=7)
    env.add_agent("a0", (0, 0))
    agent = ReactiveAgent("a0", (0, 0))
    agent.add_rule(lambda obs: obs.get("food_nearby"), "collect")
    agent.add_rule(lambda obs: obs.get("food_direction") == "down", "down")
    agent.add_rule(lambda obs: obs.get("food_direction") == "right", "right")
    agent.set_default("stay")
    cfg = SimulationConfig(max_steps=50, n_episodes=3)
    return Simulation(env, [agent], cfg)


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------


def make_obs(
    position: tuple[int, int] = (0, 0),
    food_nearby: bool = False,
    food_direction: str | None = None,
    grid_size: int = 8,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a minimal observation dict."""
    obs: dict[str, Any] = {
        "position": position,
        "food_nearby": food_nearby,
        "food_direction": food_direction,
        "grid_size": grid_size,
    }
    obs.update(kwargs)
    return obs
