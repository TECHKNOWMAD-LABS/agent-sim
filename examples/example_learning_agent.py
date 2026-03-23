"""Example 3: Q-learning agent training over multiple episodes.

Demonstrates how to train a LearningAgent inside a GridEnvironment,
observe Q-table convergence, and visualise the reward trend.
"""

from __future__ import annotations

import time

from agentsim import GridEnvironment, LearningAgent, SimulationConfig
from agentsim.analysis import compute_metrics
from agentsim.simulation import Simulation
from agentsim.viz import plot_reward_trend, render_agent_heatmap


def _make_learning_sim(seed: int = 0) -> tuple[Simulation, LearningAgent]:
    """Build a simulation with one Q-learning agent."""
    env = GridEnvironment(rows=8, cols=8, food_count=5, wall_density=0.05, seed=seed)
    env.add_agent("learner")
    agent = LearningAgent(
        "learner",
        position=(0, 0),
        state_size=64,
        learning_rate=0.1,
        discount=0.9,
        epsilon=0.3,
    )
    cfg = SimulationConfig(max_steps=200, n_episodes=1)
    sim = Simulation(env, [agent], cfg)
    return sim, agent


def main() -> None:
    """Train a Q-learning agent for 30 episodes and plot the reward trend."""
    print("=== AgentSim — Q-Learning Agent Example ===\n")
    n_episodes = 30
    all_rewards: list[float] = []

    t0 = time.perf_counter()

    # We run each episode manually so we can decay epsilon between episodes
    for ep in range(n_episodes):
        sim, agent = _make_learning_sim(seed=ep % 10)
        result = sim.run_episode(ep)
        total = sum(result.total_reward.values())
        all_rewards.append(total)
        agent.decay_epsilon(factor=0.95, minimum=0.02)

    elapsed = time.perf_counter() - t0
    print(f"Trained {n_episodes} episodes in {elapsed:.2f}s\n")

    # Reward trend sparkline
    print(plot_reward_trend(all_rewards, title="Training reward trend"))
    print()

    # Final episode stats
    sim, agent = _make_learning_sim(seed=42)
    results = sim.run()
    metrics = compute_metrics(results, sim.agents)
    m = metrics.agent_metrics["learner"]
    print(f"Final episode — steps: {results[0].steps}")
    print(f"  Total reward   : {m.total_reward:.4f}")
    print(f"  Reward/step    : {m.avg_reward_per_step:.4f}")
    print(f"  Agent alive    : {m.alive}")
    print()

    # Heatmap of visited positions
    positions = [h.position for h in agent.history]
    if positions:
        print("Visitation heatmap (8x8 grid):")
        print(render_agent_heatmap(positions, rows=8, cols=8))


if __name__ == "__main__":
    main()
