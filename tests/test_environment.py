"""Tests 6-9: GridEnvironment correctness."""

from agentsim import GridEnvironment
from agentsim.environment.grid import FOOD, WALL


# --- Test 6 ---
def test_grid_creation():
    env = GridEnvironment(rows=8, cols=8, food_count=5, seed=42)
    assert env.rows == 8
    assert env.cols == 8
    assert env.food_remaining == 0  # no food until reset


# --- Test 7 ---
def test_grid_reset_populates_food_and_agents():
    env = GridEnvironment(rows=8, cols=8, food_count=5, seed=42)
    env.add_agent("a0")
    obs_map = env.reset()
    assert "a0" in obs_map
    assert env.food_remaining == 5
    assert obs_map["a0"]["position"] is not None


# --- Test 8 ---
def test_grid_movement_into_free_cell():
    env = GridEnvironment(rows=8, cols=8, food_count=0, wall_density=0.0, seed=1)
    env.add_agent("a0", (4, 4))
    env.reset()
    # Force known position after reset
    env._agents["a0"].position = (4, 4)
    obs, _, _ = env.step("a0", "right")
    assert obs["position"] == (4, 5)


# --- Test 9 ---
def test_grid_out_of_bounds_blocked():
    env = GridEnvironment(rows=5, cols=5, food_count=0, wall_density=0.0, seed=0)
    env.add_agent("a0", (0, 0))
    env.reset()
    env._agents["a0"].position = (0, 0)
    obs, _, _ = env.step("a0", "up")  # (-1, 0) is invalid
    assert obs["position"] == (0, 0)


# --- Test 10 ---
def test_grid_food_collection_grants_reward():
    env = GridEnvironment(rows=5, cols=5, food_count=0, wall_density=0.0, seed=0)
    env.add_agent("a0", (2, 2))
    env.reset()
    # Manually place food and set agent there
    env._agents["a0"].position = (2, 2)
    env._grid[2, 2] = FOOD
    obs, reward, _ = env.step("a0", "collect")
    assert reward == 1.0
    assert obs["collected"] == 1
    assert env._grid[2, 2] != FOOD


# --- Test 11 ---
def test_grid_wall_blocks_movement():
    env = GridEnvironment(rows=5, cols=5, food_count=0, wall_density=0.0, seed=0)
    env.add_agent("a0", (2, 2))
    env.reset()
    env._agents["a0"].position = (2, 2)
    env._grid[2, 3] = WALL
    obs, _, _ = env.step("a0", "right")
    assert obs["position"] == (2, 2)
