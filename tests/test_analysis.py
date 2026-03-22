"""Test 15: Analysis, trajectory stats, and viz rendering."""

from agentsim import (
    GridEnvironment,
    ReactiveAgent,
    Simulation,
    SimulationConfig,
    compute_metrics,
    compute_trajectory_stats,
)
from agentsim.viz import (
    plot_reward_trend,
    render_agent_heatmap,
    render_episode_summary,
    simulation_report,
)


def _run_small_sim() -> tuple[Simulation, list]:
    env = GridEnvironment(rows=6, cols=6, food_count=3, seed=99)
    env.add_agent("b0", (0, 0))
    agent = ReactiveAgent("b0", (0, 0))
    agent.add_rule(lambda obs: obs.get("food_nearby"), "collect")
    agent.add_rule(lambda obs: obs.get("food_direction") == "right", "right")
    agent.add_rule(lambda obs: obs.get("food_direction") == "down", "down")
    agent.set_default("stay")
    cfg = SimulationConfig(max_steps=30, n_episodes=2)
    sim = Simulation(env, [agent], cfg)
    results = sim.run()
    return sim, results


# --- Test 15 ---
def test_compute_metrics_and_viz():
    sim, results = _run_small_sim()

    metrics = compute_metrics(results, sim.agents)
    assert metrics.n_episodes == 2
    assert "b0" in metrics.agent_metrics
    m = metrics.agent_metrics["b0"]
    assert m.total_steps > 0
    assert isinstance(m.alive, bool)
    assert len(metrics.reward_trend) == 2

    # trajectory stats
    agent = sim.agents["b0"]
    stats = compute_trajectory_stats(agent)
    assert "n_steps" in stats
    assert stats["n_steps"] > 0
    assert "unique_positions" in stats

    # viz: sparkline
    sparkline = plot_reward_trend(metrics.reward_trend)
    assert "Reward Trend" in sparkline

    # viz: heatmap
    positions = [h.position for h in agent.history]
    heatmap = render_agent_heatmap(positions, rows=6, cols=6)
    assert len(heatmap.splitlines()) == 6

    # viz: episode summary
    summary_text = render_episode_summary(results[0])
    assert "Episode 0" in summary_text

    # viz: full report
    report = simulation_report(results, metrics.reward_trend)
    assert "AgentSim" in report
    assert "Episodes" in report
