"""Test 14: Simulation orchestration."""

from agentsim import GridEnvironment, ReactiveAgent, Simulation, SimulationConfig


def _make_sim(n_episodes: int = 1) -> Simulation:
    env = GridEnvironment(rows=6, cols=6, food_count=3, seed=7)
    env.add_agent("a0", (0, 0))
    agent = ReactiveAgent("a0", (0, 0))
    agent.add_rule(lambda obs: obs.get("food_nearby"), "collect")
    agent.add_rule(lambda obs: obs.get("food_direction") == "up", "up")
    agent.add_rule(lambda obs: obs.get("food_direction") == "down", "down")
    agent.add_rule(lambda obs: obs.get("food_direction") == "left", "left")
    agent.add_rule(lambda obs: obs.get("food_direction") == "right", "right")
    agent.set_default("stay")
    cfg = SimulationConfig(max_steps=50, n_episodes=n_episodes)
    return Simulation(env, [agent], cfg)


# --- Test 14 ---
def test_simulation_runs_expected_episodes():
    sim = _make_sim(n_episodes=3)
    results = sim.run()
    assert len(results) == 3
    for r in results:
        assert r.steps <= 50
        assert "a0" in r.total_reward

    summary = sim.summary()
    assert summary["episodes"] == 3
    assert 0.0 <= summary["completion_rate"] <= 1.0
