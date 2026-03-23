"""CYCLE 2: Tests for input validation and error hardening added to all public APIs."""

from __future__ import annotations

import math

import pytest

from agentsim import GridEnvironment, LearningAgent, ReactiveAgent, Simulation, SimulationConfig
from agentsim.simulation import EpisodeResult


# ===========================================================================
# BaseAgent / ReactiveAgent — constructor validation
# ===========================================================================


class TestBaseAgentValidation:
    def test_empty_agent_id_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            ReactiveAgent("", (0, 0))

    def test_whitespace_agent_id_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            ReactiveAgent("   ", (0, 0))

    def test_invalid_position_type_raises(self):
        with pytest.raises(ValueError, match="2-tuple of ints"):
            ReactiveAgent("a", "bad_position")  # type: ignore[arg-type]

    def test_position_wrong_length_raises(self):
        with pytest.raises(ValueError, match="2-tuple of ints"):
            ReactiveAgent("a", (1,))  # type: ignore[arg-type]

    def test_position_floats_raises(self):
        with pytest.raises(ValueError, match="2-tuple of ints"):
            ReactiveAgent("a", (1.0, 2.0))  # type: ignore[arg-type]

    def test_receive_reward_nan_treated_as_zero(self):
        agent = ReactiveAgent("a", (0, 0))
        agent.receive_reward(float("nan"))
        assert agent.state.reward == pytest.approx(0.0)

    def test_receive_reward_inf_treated_as_zero(self):
        agent = ReactiveAgent("a", (0, 0))
        agent.receive_reward(float("inf"))
        assert agent.state.reward == pytest.approx(0.0)

    def test_step_with_none_obs_does_not_raise(self):
        agent = ReactiveAgent("a", (0, 0))
        agent.set_default("stay")
        # None observation should be treated as empty dict
        action = agent.step(None)  # type: ignore[arg-type]
        assert action == "stay"


# ===========================================================================
# GridEnvironment — constructor and method validation
# ===========================================================================


class TestGridEnvironmentValidation:
    def test_zero_rows_raises(self):
        with pytest.raises(ValueError, match="rows"):
            GridEnvironment(rows=0, cols=5)

    def test_negative_cols_raises(self):
        with pytest.raises(ValueError, match="cols"):
            GridEnvironment(rows=5, cols=-1)

    def test_negative_food_count_raises(self):
        with pytest.raises(ValueError, match="food_count"):
            GridEnvironment(rows=5, cols=5, food_count=-1)

    def test_wall_density_out_of_range_raises(self):
        with pytest.raises(ValueError, match="wall_density"):
            GridEnvironment(rows=5, cols=5, wall_density=1.5)

    def test_add_agent_empty_id_raises(self):
        env = GridEnvironment(rows=5, cols=5)
        with pytest.raises(ValueError, match="non-empty string"):
            env.add_agent("")

    def test_add_agent_out_of_bounds_raises(self):
        env = GridEnvironment(rows=5, cols=5)
        with pytest.raises(ValueError, match="out of bounds"):
            env.add_agent("a", (10, 10))

    def test_step_unknown_agent_raises(self):
        env = GridEnvironment(rows=5, cols=5)
        with pytest.raises(KeyError, match="Unknown agent"):
            env.step("ghost", "up")

    def test_get_observation_unknown_agent_raises(self):
        env = GridEnvironment(rows=5, cols=5)
        with pytest.raises(KeyError, match="Unknown agent"):
            env.get_observation("ghost")

    def test_step_non_string_action_treated_as_stay(self):
        env = GridEnvironment(rows=5, cols=5, food_count=0, wall_density=0.0, seed=0)
        env.add_agent("a", (2, 2))
        env.reset()
        env._agents["a"].position = (2, 2)
        obs, reward, done = env.step("a", 999)  # type: ignore[arg-type]
        # Non-string action → stay → position unchanged
        assert obs["position"] == (2, 2)


# ===========================================================================
# SimulationConfig — validation
# ===========================================================================


class TestSimulationConfigValidation:
    def test_zero_max_steps_raises(self):
        with pytest.raises(ValueError, match="max_steps"):
            SimulationConfig(max_steps=0)

    def test_negative_n_episodes_raises(self):
        with pytest.raises(ValueError, match="n_episodes"):
            SimulationConfig(n_episodes=-1)

    def test_negative_render_every_raises(self):
        with pytest.raises(ValueError, match="render_every"):
            SimulationConfig(render_every=-5)


# ===========================================================================
# Simulation — constructor validation
# ===========================================================================


class TestSimulationValidation:
    def test_empty_agents_raises(self):
        env = GridEnvironment(rows=5, cols=5, seed=0)
        with pytest.raises(ValueError, match="at least one agent"):
            Simulation(env, [])

    def test_none_env_raises(self):
        agent = ReactiveAgent("a", (0, 0))
        with pytest.raises(ValueError, match="env must not be None"):
            Simulation(None, [agent])  # type: ignore[arg-type]


# ===========================================================================
# LearningAgent — constructor validation
# ===========================================================================


class TestLearningAgentValidation:
    def test_zero_state_size_raises(self):
        with pytest.raises(ValueError, match="state_size"):
            LearningAgent("l", (0, 0), state_size=0)

    def test_invalid_learning_rate_raises(self):
        with pytest.raises(ValueError, match="learning_rate"):
            LearningAgent("l", (0, 0), learning_rate=0.0)

    def test_invalid_discount_raises(self):
        with pytest.raises(ValueError, match="discount"):
            LearningAgent("l", (0, 0), discount=1.5)

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            LearningAgent("l", (0, 0), epsilon=-0.1)

    def test_update_with_nan_reward_safe(self):
        agent = LearningAgent("l", (0, 0), state_size=64, epsilon=0.0)
        obs = {"position": (0, 0), "grid_size": 8}
        agent.step(obs)
        # Should not raise; NaN treated as 0
        agent.update(float("nan"), {"position": (0, 1), "grid_size": 8})

    def test_decay_epsilon_invalid_factor_raises(self):
        agent = LearningAgent("l", (0, 0))
        with pytest.raises(ValueError, match="factor"):
            agent.decay_epsilon(factor=0.0)

    def test_decay_epsilon_negative_minimum_raises(self):
        agent = LearningAgent("l", (0, 0))
        with pytest.raises(ValueError, match="minimum"):
            agent.decay_epsilon(minimum=-0.1)

    def test_perceive_none_obs_safe(self):
        agent = LearningAgent("l", (0, 0))
        agent.perceive(None)  # type: ignore[arg-type]
        assert agent._state_key == 0
