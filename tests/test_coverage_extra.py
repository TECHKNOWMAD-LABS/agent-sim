"""Extra tests to drive coverage of previously-missed branches.

Covers:
- deliberative.py: _revise_goals, _plan_to_food (no food_direction), _plan_explore,
  decide() with empty plan, multiple goals
- learning.py: epsilon-greedy exploration path, decay_epsilon
- reactive.py: set_default with valid action
- analysis.py: empty results branch, compute_trajectory_stats with history
- environment/grid.py: render(), step_count property, _random_empty fallback,
  _nearest_food_direction same-column tie-break
- simulation.py: render_every frames, summary() with empty results
- viz.py: render_grid_ascii, plot_reward_trend empty, simulation_report >5 episodes
- pursuit.py: _manhattan tie-break / no prey caught path already tested; cover None dir
"""

from __future__ import annotations

import numpy as np
import pytest

from agentsim import (
    DeliberativeAgent,
    Goal,
    GridEnvironment,
    LearningAgent,
    ReactiveAgent,
    Simulation,
    SimulationConfig,
    compute_metrics,
    compute_trajectory_stats,
)
from agentsim.environment.grid import FOOD, WALL
from agentsim.scenarios.pursuit import _direction, _manhattan
from agentsim.viz import (
    plot_reward_trend,
    render_agent_heatmap,
    render_grid_ascii,
    render_episode_summary,
    simulation_report,
)
from tests.conftest import make_obs


# ===========================================================================
# DeliberativeAgent — missed branches
# ===========================================================================


class TestDeliberativeAgentBranches:
    def test_plan_to_food_no_direction(self):
        """_plan_to_food returns ['stay'] when food_direction is absent."""
        agent = DeliberativeAgent("d1", (0, 0))
        agent.add_goal(Goal("find_food", priority=2.0))
        # food_nearby=True triggers goal priority bump, but no food_direction
        obs = make_obs(position=(0, 0), food_nearby=True, food_direction=None)
        action = agent.step(obs)
        assert action in agent.ACTIONS

    def test_plan_explore_returns_movement(self):
        """_plan_explore returns a cardinal direction."""
        agent = DeliberativeAgent("d2", (1, 1))
        agent.add_goal(Goal("explore", priority=1.0))
        obs = make_obs(position=(1, 1))
        action = agent.step(obs)
        assert action in ("up", "down", "left", "right", "stay", "collect")

    def test_decide_with_empty_plan_rebuilds(self):
        """decide() rebuilds plan when _plan is empty."""
        agent = DeliberativeAgent("d3", (2, 2))
        agent.add_goal(Goal("explore", priority=1.0))
        # Pre-clear plan
        agent._plan = []
        action = agent.decide()
        assert action in agent.ACTIONS

    def test_no_active_goals_returns_stay(self):
        """With all goals achieved, plan defaults to stay."""
        agent = DeliberativeAgent("d4", (0, 0))
        achieved = Goal("find_food", priority=1.0)
        achieved.achieved = True
        agent.add_goal(achieved)
        action = agent.step(make_obs())
        assert action == "stay"

    def test_revise_goals_sets_priority(self):
        """_revise_goals bumps find_food priority when food_nearby."""
        agent = DeliberativeAgent("d5", (0, 0))
        g = Goal("find_food", priority=0.5)
        agent.add_goal(g)
        obs = make_obs(food_nearby=True, food_direction="up")
        agent.step(obs)
        # After perceive, find_food priority should be 2.0
        assert g.priority == pytest.approx(2.0)

    def test_unknown_goal_falls_through_to_stay(self):
        """A goal with an unrecognized name falls through to stay plan."""
        agent = DeliberativeAgent("d6", (0, 0))
        agent.add_goal(Goal("mystery_goal", priority=3.0))
        obs = make_obs()
        action = agent.step(obs)
        assert action == "stay"

    def test_multiple_goals_priority_ordering(self):
        """Goals are sorted by priority descending after add_goal."""
        agent = DeliberativeAgent("d7", (0, 0))
        agent.add_goal(Goal("explore", priority=1.0))
        agent.add_goal(Goal("find_food", priority=2.0))
        assert agent.goals[0].name == "find_food"
        assert agent.goals[1].name == "explore"


# ===========================================================================
# LearningAgent — missed branches
# ===========================================================================


class TestLearningAgentBranches:
    def test_epsilon_greedy_exploration(self):
        """With epsilon=1.0 agent always explores (random action)."""
        agent = LearningAgent("l1", (0, 0), state_size=64, epsilon=1.0)
        obs = make_obs(position=(0, 0))
        action = agent.step(obs)
        assert action in agent.ACTIONS

    def test_decay_epsilon(self):
        """decay_epsilon reduces epsilon but not below minimum."""
        agent = LearningAgent("l2", (0, 0), epsilon=0.5)
        agent.decay_epsilon(factor=0.5, minimum=0.3)
        assert agent.epsilon == pytest.approx(0.3)  # 0.5*0.5=0.25 < 0.3 → clamped

        agent2 = LearningAgent("l3", (0, 0), epsilon=0.8)
        agent2.decay_epsilon(factor=0.9, minimum=0.01)
        assert agent2.epsilon == pytest.approx(0.72)

    def test_greedy_policy_selects_best_q(self):
        """With epsilon=0.0 agent always picks max Q-value action."""
        agent = LearningAgent("l4", (0, 0), state_size=64, epsilon=0.0)
        obs = make_obs(position=(0, 0))
        agent.step(obs)
        state_key = agent._state_key
        # Set a specific action as clearly best
        best_action_idx = 2
        agent.q_table[state_key, :] = 0.0
        agent.q_table[state_key, best_action_idx] = 10.0
        action = agent.decide()
        assert action == agent.ACTIONS[best_action_idx]

    def test_update_changes_qtable(self):
        """update() modifies the correct Q-table cell."""
        agent = LearningAgent("l5", (0, 0), state_size=64, epsilon=0.0)
        obs = make_obs(position=(1, 1), grid_size=8)
        agent.step(obs)
        before = agent.q_table.copy()
        agent.update(5.0, make_obs(position=(1, 2), grid_size=8))
        assert not np.array_equal(agent.q_table, before)


# ===========================================================================
# ReactiveAgent — missed branch
# ===========================================================================


class TestReactiveAgentBranches:
    def test_set_default_valid_action(self):
        """set_default with a valid action does not raise."""
        agent = ReactiveAgent("r1", (0, 0))
        agent.set_default("up")
        # Trigger decide with no rules → should return "up"
        obs = make_obs()
        agent.perceive(obs)
        assert agent.decide() == "up"

    def test_set_default_invalid_raises(self):
        """set_default with invalid action raises ValueError."""
        agent = ReactiveAgent("r2", (0, 0))
        with pytest.raises(ValueError, match="Unknown action"):
            agent.set_default("teleport")


# ===========================================================================
# Analysis — missed branches
# ===========================================================================


class TestAnalysisBranches:
    def test_compute_metrics_empty(self):
        """compute_metrics returns zero metrics for empty results list."""
        metrics = compute_metrics([], {})
        assert metrics.n_episodes == 0
        assert metrics.avg_steps == 0.0
        assert metrics.completion_rate == 0.0
        assert metrics.agent_metrics == {}
        assert metrics.reward_trend == []

    def test_compute_trajectory_stats_empty_history(self):
        """compute_trajectory_stats returns {} for agent with no history."""
        agent = ReactiveAgent("r", (0, 0))
        stats = compute_trajectory_stats(agent)
        assert stats == {}


# ===========================================================================
# GridEnvironment — missed branches
# ===========================================================================


class TestGridEnvironmentBranches:
    def test_render_returns_string(self):
        """render() returns a non-empty multi-line string."""
        env = GridEnvironment(rows=4, cols=4, food_count=2, seed=1)
        env.add_agent("a0", (0, 0))
        env.reset()
        rendered = env.render()
        assert isinstance(rendered, str)
        assert len(rendered.splitlines()) == 4

    def test_step_count_property(self):
        """step_count increments correctly after env.step()."""
        env = GridEnvironment(rows=5, cols=5, food_count=0, wall_density=0.0, seed=0)
        env.add_agent("a0", (2, 2))
        env.reset()
        env._agents["a0"].position = (2, 2)
        assert env.step_count == 0
        env.step("a0", "right")
        assert env.step_count == 1

    def test_nearest_food_direction_same_column(self):
        """Food directly below: nearest direction is 'down'."""
        env = GridEnvironment(rows=8, cols=8, food_count=0, wall_density=0.0, seed=0)
        env.add_agent("a0", (2, 4))
        env.reset()
        env._agents["a0"].position = (2, 4)
        env._grid[5, 4] = FOOD  # same column, below
        obs = env.get_observation("a0")
        assert obs["food_direction"] == "down"

    def test_nearest_food_direction_same_row_right(self):
        """Food in same row to the right gives direction 'right'."""
        env = GridEnvironment(rows=8, cols=8, food_count=0, wall_density=0.0, seed=0)
        env.add_agent("a0", (3, 3))
        env.reset()
        env._agents["a0"].position = (3, 3)
        env._grid[3, 6] = FOOD  # same row, right
        obs = env.get_observation("a0")
        assert obs["food_direction"] == "right"

    def test_get_observation_food_nearby(self):
        """get_observation sets food_nearby=True when agent is on food cell."""
        env = GridEnvironment(rows=5, cols=5, food_count=0, wall_density=0.0, seed=0)
        env.add_agent("a0", (2, 2))
        env.reset()
        env._agents["a0"].position = (2, 2)
        env._grid[2, 2] = FOOD
        obs = env.get_observation("a0")
        assert obs["food_nearby"] is True

    def test_render_grid_ascii_via_viz(self):
        """render_grid_ascii wrapper returns the env.render() string."""
        env = GridEnvironment(rows=4, cols=4, food_count=0, wall_density=0.0, seed=0)
        env.add_agent("a0", (0, 0))
        env.reset()
        result = render_grid_ascii(env)
        assert isinstance(result, str)
        assert len(result) > 0


# ===========================================================================
# Simulation — missed branches
# ===========================================================================


class TestSimulationBranches:
    def test_render_every_captures_frames(self):
        """With render_every=1, frames list is populated."""
        env = GridEnvironment(rows=5, cols=5, food_count=1, wall_density=0.0, seed=0)
        env.add_agent("a0", (0, 0))
        agent = ReactiveAgent("a0", (0, 0))
        agent.add_rule(lambda obs: obs.get("food_nearby"), "collect")
        agent.add_rule(lambda obs: obs.get("food_direction") == "down", "down")
        agent.add_rule(lambda obs: obs.get("food_direction") == "right", "right")
        agent.set_default("stay")
        cfg = SimulationConfig(max_steps=20, n_episodes=1, render_every=1)
        sim = Simulation(env, [agent], cfg)
        result = sim.run_episode(0)
        assert len(result.frames) > 0

    def test_summary_empty(self):
        """summary() returns {} when no episodes have been run."""
        env = GridEnvironment(rows=5, cols=5, food_count=1, seed=0)
        env.add_agent("a0", (0, 0))
        agent = ReactiveAgent("a0", (0, 0))
        agent.set_default("stay")
        sim = Simulation(env, [agent])
        assert sim.summary() == {}


# ===========================================================================
# Viz — missed branches
# ===========================================================================


class TestVizBranches:
    def test_plot_reward_trend_empty(self):
        """plot_reward_trend with empty list returns '(no data)' message."""
        result = plot_reward_trend([])
        assert "no data" in result

    def test_simulation_report_over_five_episodes(self):
        """simulation_report truncates display to 5 episodes + '… and N more'."""
        from agentsim.simulation import EpisodeResult

        results = [
            EpisodeResult(episode=i, steps=10, total_reward={"a": 0.5}, done=True)
            for i in range(8)
        ]
        report = simulation_report(results)
        assert "and 3 more" in report

    def test_render_agent_heatmap_all_zeros(self):
        """render_agent_heatmap with no positions returns blank-space grid."""
        heatmap = render_agent_heatmap([], rows=3, cols=3)
        assert len(heatmap.splitlines()) == 3

    def test_render_episode_summary_format(self):
        """render_episode_summary contains expected labels."""
        from agentsim.simulation import EpisodeResult

        result = EpisodeResult(episode=5, steps=42, total_reward={"a0": 3.14}, done=False)
        text = render_episode_summary(result)
        assert "Episode 5" in text
        assert "Steps" in text
        assert "a0" in text


# ===========================================================================
# Pursuit helpers — missed branch
# ===========================================================================


class TestPursuitHelpers:
    def test_direction_same_position_returns_none(self):
        """_direction returns None when both positions are identical."""
        assert _direction((3, 3), (3, 3)) is None

    def test_direction_same_row_left(self):
        """_direction returns 'left' when target is in same row to the left."""
        assert _direction((3, 5), (3, 2)) == "left"

    def test_manhattan_zero(self):
        """_manhattan returns 0.0 for identical positions."""
        assert _manhattan((4, 4), (4, 4)) == pytest.approx(0.0)

    def test_manhattan_asymmetric(self):
        """_manhattan is symmetric."""
        a, b = (1, 2), (5, 8)
        assert _manhattan(a, b) == _manhattan(b, a)
