"""Example 2: Predator–prey pursuit scenario.

Demonstrates the PursuitScenario: a reactive predator chases a reactive prey
across a grid until caught or the step budget is exhausted.
"""

from __future__ import annotations

from agentsim import PursuitScenario
from agentsim.scenarios.pursuit import _manhattan


def main() -> None:
    """Run a predator-prey scenario and report the outcome."""
    print("=== AgentSim — Pursuit Example ===\n")

    for seed in range(5):
        scenario = PursuitScenario(grid_size=12, max_steps=150, seed=seed)
        result = scenario.run()

        outcome = "CAUGHT" if result.caught else "ESCAPED"
        print(
            f"seed={seed}  outcome={outcome:<8}  steps={result.steps:3d}"
            f"  min_distance={result.min_distance:.1f}"
        )

    print()
    print("Running large pursuit (20x20 grid, 500 steps) …")
    scenario = PursuitScenario(grid_size=20, max_steps=500, seed=7)
    result = scenario.run()
    print(f"  Caught : {result.caught}")
    print(f"  Steps  : {result.steps}")
    print(f"  Closest approach (Manhattan): {result.min_distance:.1f} cells")

    # Show start/end positions
    if scenario.predator and scenario.prey:
        pred_pos = scenario.env._agents["predator"].position
        prey_pos = scenario.env._agents["prey"].position
        final_dist = _manhattan(pred_pos, prey_pos)
        print(f"  Final predator position: {pred_pos}")
        print(f"  Final prey position    : {prey_pos}")
        print(f"  Final distance         : {final_dist:.1f}")


if __name__ == "__main__":
    main()
