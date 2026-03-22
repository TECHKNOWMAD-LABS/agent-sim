"""Tests 1-4: Agent creation and behaviour."""

import pytest

from agentsim import DeliberativeAgent, Goal, LearningAgent, ReactiveAgent


# --- Test 1 ---
def test_reactive_agent_creation():
    agent = ReactiveAgent("r1", (0, 0))
    assert agent.id == "r1"
    assert agent.state.position == (0, 0)
    assert agent.is_alive()
    assert agent.state.energy == 1.0


# --- Test 2 ---
def test_reactive_agent_rules_fire_in_order():
    agent = ReactiveAgent("r2", (2, 3))
    agent.add_rule(lambda obs: obs.get("food_nearby"), "collect")
    agent.add_rule(lambda obs: obs.get("food_direction") == "up", "up")
    agent.set_default("stay")

    obs_food = {"food_nearby": True, "food_direction": None, "position": (2, 3), "grid_size": 8}
    assert agent.step(obs_food) == "collect"

    obs_up = {"food_nearby": False, "food_direction": "up", "position": (2, 3), "grid_size": 8}
    assert agent.step(obs_up) == "up"

    obs_none = {"food_nearby": False, "food_direction": None, "position": (2, 3), "grid_size": 8}
    assert agent.step(obs_none) == "stay"


# --- Test 3 ---
def test_deliberative_agent_goals():
    agent = DeliberativeAgent("d1", (1, 1))
    goal = Goal("find_food", priority=1.5)
    agent.add_goal(goal)
    assert len(agent.goals) == 1
    assert agent.goals[0].name == "find_food"

    obs = {
        "food_nearby": False,
        "food_direction": "right",
        "position": (1, 1),
        "grid_size": 8,
        "step": 0,
    }
    action = agent.step(obs)
    assert action in agent.ACTIONS
    # With food_direction="right" and no food_nearby, plan is ["right","collect"]
    assert action == "right"


# --- Test 4 ---
def test_learning_agent_update():
    agent = LearningAgent("l1", (0, 0), state_size=64, epsilon=0.0)
    obs = {"position": (0, 0), "grid_size": 8}
    action = agent.step(obs)
    assert action in agent.ACTIONS

    before = agent.q_table[agent._state_key, agent._last_action_idx]
    next_obs = {"position": (0, 1), "grid_size": 8}
    agent.update(1.0, next_obs)

    after = agent.q_table[agent._state_key, agent._last_action_idx]
    assert after != before, "Q-value should change after update with non-zero reward"
    assert agent.state.reward == pytest.approx(1.0)


# --- Test 4b: invalid action raises ---
def test_reactive_agent_invalid_action_raises():
    agent = ReactiveAgent("r3", (0, 0))
    with pytest.raises(ValueError, match="Unknown action"):
        agent.add_rule(lambda obs: True, "fly")
