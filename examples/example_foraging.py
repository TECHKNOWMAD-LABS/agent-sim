"""Example 1: Foraging scenario with reactive agents.

Demonstrates how to run the built-in ForagingScenario and inspect results.
"""

from __future__ import annotations

from agentsim import ForagingScenario
from agentsim.analysis import compute_metrics
from agentsim.viz import simulation_report


def main() -> None:
    """Run a two-agent foraging scenario and print a summary report."""
    print("=== AgentSim — Foraging Example ===\n")

    scenario = ForagingScenario(
        n_agents=3,
        grid_size=10,
        food_count=15,
        max_steps=300,
        seed=42,
    )
    result = scenario.run()

    print(f"Steps taken    : {result.steps}")
    print(f"Total collected: {result.total_collected}")
    print(f"Efficiency     : {result.efficiency:.4f} items/step")
    print(f"Per-agent score: {result.agent_results}")
    print()

    # Build a richer report via the Simulation class
    from agentsim import GridEnvironment, ReactiveAgent, Simulation, SimulationConfig
    from agentsim.scenarios.foraging import make_forager

    env = GridEnvironment(rows=10, cols=10, food_count=15, seed=0)
    agents = []
    for i in range(3):
        env.add_agent(f"agent_{i}")
        agents.append(make_forager(f"agent_{i}", (0, 0)))

    sim = Simulation(
        env,
        agents,
        SimulationConfig(max_steps=200, n_episodes=5),
    )
    sim_results = sim.run()
    metrics = compute_metrics(sim_results, sim.agents)

    report = simulation_report(sim_results, metrics.reward_trend)
    print(report)
    print(f"\nAvg steps / episode: {metrics.avg_steps:.1f}")
    print(f"Completion rate    : {metrics.completion_rate:.1%}")


if __name__ == "__main__":
    main()
