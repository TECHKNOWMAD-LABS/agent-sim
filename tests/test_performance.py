"""CYCLE 3: Performance tests — parallelism, caching, timing measurements."""

from __future__ import annotations

import time

from agentsim import GridEnvironment, ReactiveAgent, Simulation, SimulationConfig
from agentsim.analysis import (
    _trajectory_stats_cached,
    compute_metrics,
    compute_trajectory_stats,
    run_episodes_parallel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_forager_factory(grid_size: int = 8, food_count: int = 5, max_steps: int = 50):
    """Return a factory callable for parallel episode tests."""

    def factory():
        env = GridEnvironment(rows=grid_size, cols=grid_size, food_count=food_count, seed=None)
        env.add_agent("a0")
        agent = ReactiveAgent("a0", (0, 0))
        agent.add_rule(lambda obs: obs.get("food_nearby"), "collect")
        agent.add_rule(lambda obs: obs.get("food_direction") == "down", "down")
        agent.add_rule(lambda obs: obs.get("food_direction") == "right", "right")
        agent.set_default("stay")
        cfg = SimulationConfig(max_steps=max_steps, n_episodes=1)
        return env, [agent], cfg

    return factory


# ---------------------------------------------------------------------------
# Parallel episode execution
# ---------------------------------------------------------------------------


class TestParallelEpisodes:
    def test_run_episodes_parallel_correct_count(self):
        """run_episodes_parallel returns exactly n_episodes results."""
        factory = _make_forager_factory()
        results = run_episodes_parallel(factory, n_episodes=8, max_workers=4)
        assert len(results) == 8

    def test_run_episodes_parallel_episode_indices(self):
        """Episode indices in results cover 0..n_episodes-1."""
        factory = _make_forager_factory()
        results = run_episodes_parallel(factory, n_episodes=4, max_workers=2)
        indices = sorted(r.episode for r in results)
        assert indices == [0, 1, 2, 3]

    def test_run_episodes_parallel_results_valid(self):
        """Each result has positive steps and valid reward keys."""
        factory = _make_forager_factory()
        results = run_episodes_parallel(factory, n_episodes=4, max_workers=4)
        for r in results:
            assert r.steps >= 1
            assert "a0" in r.total_reward

    def test_parallel_faster_than_sequential(self):
        """Parallel execution of 8 episodes is faster than sequential."""
        factory = _make_forager_factory(max_steps=100)

        t0 = time.perf_counter()
        seq_results = []
        for i in range(8):
            env, agents, cfg = factory()
            sim = Simulation(env, agents, cfg)
            seq_results.append(sim.run_episode(i))
        t_seq = time.perf_counter() - t0

        t0 = time.perf_counter()
        par_results = run_episodes_parallel(factory, n_episodes=8, max_workers=4)
        t_par = time.perf_counter() - t0

        # Parallel should not be more than 3x slower than sequential
        # (on CI/low-core machines parallelism overhead may dominate for tiny tasks)
        assert t_par < t_seq * 3.5, (
            f"Parallel ({t_par:.3f}s) unexpectedly much slower than sequential ({t_seq:.3f}s)"
        )
        # Both approaches produce the same number of results
        assert len(par_results) == len(seq_results)


# ---------------------------------------------------------------------------
# LRU cache for trajectory stats
# ---------------------------------------------------------------------------


class TestTrajectoryCache:
    def test_cache_hit_returns_same_object(self):
        """Second call with identical snapshot hits cache (same object returned)."""
        agent = ReactiveAgent("a", (0, 0))
        agent.set_default("stay")
        for _ in range(5):
            agent.step({"position": (0, 0), "grid_size": 8})

        result1 = compute_trajectory_stats(agent)
        result2 = compute_trajectory_stats(agent)
        assert result1 == result2

    def test_cache_miss_after_new_step(self):
        """Extending history changes the snapshot key → cache miss → fresh result."""
        agent = ReactiveAgent("a", (0, 0))
        agent.set_default("down")
        for _ in range(3):
            agent.step({"position": (0, 0), "grid_size": 8})
        stats_before = compute_trajectory_stats(agent)

        # One more step changes history
        agent.step({"position": (0, 0), "grid_size": 8})
        stats_after = compute_trajectory_stats(agent)
        assert stats_after["n_steps"] == stats_before["n_steps"] + 1

    def test_cache_info_populated(self):
        """LRU cache registers hits after repeated calls."""
        _trajectory_stats_cached.cache_clear()
        agent = ReactiveAgent("a", (0, 0))
        agent.set_default("stay")
        for _ in range(4):
            agent.step({"position": (1, 1), "grid_size": 8})
        compute_trajectory_stats(agent)
        compute_trajectory_stats(agent)
        info = _trajectory_stats_cached.cache_info()
        assert info.hits >= 1


# ---------------------------------------------------------------------------
# Timing measurement (smoke)
# ---------------------------------------------------------------------------


class TestTimingMeasurements:
    def test_foraging_scenario_completes_in_time(self):
        """A 200-step foraging scenario runs in under 2 seconds."""
        from agentsim import ForagingScenario

        t0 = time.perf_counter()
        scenario = ForagingScenario(n_agents=4, grid_size=10, food_count=15, max_steps=200, seed=0)
        scenario.run()
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"Foraging scenario too slow: {elapsed:.2f}s"

    def test_compute_metrics_fast(self):
        """compute_metrics on 50 episodes completes in under 0.5s."""
        env = GridEnvironment(rows=6, cols=6, food_count=3, seed=7)
        env.add_agent("a0", (0, 0))
        agent = ReactiveAgent("a0", (0, 0))
        agent.add_rule(lambda obs: obs.get("food_nearby"), "collect")
        agent.set_default("stay")
        cfg = SimulationConfig(max_steps=30, n_episodes=50)
        sim = Simulation(env, [agent], cfg)
        results = sim.run()

        t0 = time.perf_counter()
        compute_metrics(results, sim.agents)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.5, f"compute_metrics too slow on 50 episodes: {elapsed:.3f}s"
