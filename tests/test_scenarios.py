"""Tests 12-13: Scenario setup and execution."""

from agentsim import ForagingScenario, PursuitScenario


# --- Test 12 ---
def test_foraging_scenario_runs_and_collects():
    scenario = ForagingScenario(n_agents=2, grid_size=8, food_count=5, max_steps=100, seed=7)
    result = scenario.run()
    assert result.steps > 0
    assert result.steps <= 100
    assert result.total_collected >= 0
    assert len(result.agent_results) == 2
    # efficiency is total / steps, should be ≥ 0
    assert result.efficiency >= 0.0


# --- Test 13 ---
def test_pursuit_scenario_setup_and_terminates():
    scenario = PursuitScenario(grid_size=8, max_steps=50, seed=42)
    result = scenario.run()
    # Either predator catches prey or time runs out
    assert result.steps <= 50
    assert result.min_distance >= 0.0
    assert isinstance(result.caught, bool)
