"""CYCLE 6: Property-based tests using Hypothesis.

Core invariants verified:
1. Agent actions are always a member of BaseAgent.ACTIONS
2. Q-table updates do not produce NaN or Inf values
3. Simulation steps always return reward as a finite float
4. GridEnvironment positions after any step remain in bounds
5. epsilon decay always stays in [minimum, 1.0]
6. Trajectory stats are consistent with actual history length
7. Serialization round-trip: AgentState.copy() is deep-equal to original
"""

from __future__ import annotations

import math

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from agentsim import DeliberativeAgent, Goal, GridEnvironment, LearningAgent, ReactiveAgent
from agentsim.agents.base import AgentState

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

valid_positions = st.tuples(
    st.integers(min_value=0, max_value=7),
    st.integers(min_value=0, max_value=7),
)

valid_obs = st.fixed_dictionaries(
    {
        "position": valid_positions,
        "food_nearby": st.booleans(),
        "food_direction": st.one_of(st.none(), st.sampled_from(["up", "down", "left", "right"])),
        "grid_size": st.just(8),
        "step": st.integers(min_value=0, max_value=500),
    }
)

valid_agent_ids = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
    min_size=1,
    max_size=16,
)

finite_floats = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)


# ===========================================================================
# Property 1: ReactiveAgent always returns a valid action
# ===========================================================================


@given(obs=valid_obs)
@settings(max_examples=200)
def test_reactive_agent_always_valid_action(obs):
    """ReactiveAgent.step() always returns a member of ACTIONS."""
    agent = ReactiveAgent("a", (0, 0))
    agent.set_default("stay")
    action = agent.step(obs)
    assert action in agent.ACTIONS


@given(obs=valid_obs)
@settings(max_examples=100)
def test_reactive_agent_with_food_rule_always_valid(obs):
    """Forager reactive agent always produces a valid action."""
    agent = ReactiveAgent("a", (0, 0))
    agent.add_rule(lambda o: o.get("food_nearby", False), "collect")
    agent.add_rule(lambda o: o.get("food_direction") == "up", "up")
    agent.set_default("stay")
    action = agent.step(obs)
    assert action in agent.ACTIONS


# ===========================================================================
# Property 2: LearningAgent Q-table never contains NaN/Inf after updates
# ===========================================================================


@given(
    reward=finite_floats,
    pos1=valid_positions,
    pos2=valid_positions,
)
@settings(max_examples=200)
def test_learning_agent_qtable_finite_after_update(reward, pos1, pos2):
    """Q-table entries are always finite after any valid update."""
    agent = LearningAgent("l", (0, 0), state_size=64, epsilon=0.0)
    obs1 = {"position": pos1, "grid_size": 8}
    obs2 = {"position": pos2, "grid_size": 8}
    agent.step(obs1)
    agent.update(reward, obs2)
    assert not any(math.isnan(v) or math.isinf(v) for v in agent.q_table.flatten())


# ===========================================================================
# Property 3: Simulation rewards are always finite floats
# ===========================================================================


@given(
    rows=st.integers(min_value=3, max_value=10),
    cols=st.integers(min_value=3, max_value=10),
    food=st.integers(min_value=0, max_value=5),
    seed=st.integers(min_value=0, max_value=9999),
    steps=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=50, deadline=5000)
def test_grid_env_step_returns_finite_reward(rows, cols, food, seed, steps):
    """env.step() always returns a finite reward."""
    env = GridEnvironment(rows=rows, cols=cols, food_count=food, wall_density=0.0, seed=seed)
    env.add_agent("a", (0, 0))
    env.reset()
    env._agents["a"].position = (0, 0)
    actions = ["right", "down", "up", "left", "stay", "collect"]
    for i in range(steps):
        action = actions[i % len(actions)]
        _, reward, _ = env.step("a", action)
        assert math.isfinite(reward)


# ===========================================================================
# Property 4: Agent position after any step remains in-bounds
# ===========================================================================


@given(
    rows=st.integers(min_value=2, max_value=10),
    cols=st.integers(min_value=2, max_value=10),
    seed=st.integers(min_value=0, max_value=9999),
    n_steps=st.integers(min_value=1, max_value=30),
)
@settings(max_examples=50, deadline=5000)
def test_agent_position_always_in_bounds(rows, cols, seed, n_steps):
    """After any sequence of steps the agent position stays within the grid."""
    env = GridEnvironment(rows=rows, cols=cols, food_count=0, wall_density=0.0, seed=seed)
    env.add_agent("a", (0, 0))
    env.reset()
    env._agents["a"].position = (0, 0)
    actions = ["right", "down", "left", "up", "stay"]
    for i in range(n_steps):
        obs, _, done = env.step("a", actions[i % len(actions)])
        r, c = obs["position"]
        assert 0 <= r < rows
        assert 0 <= c < cols
        if done:
            break


# ===========================================================================
# Property 5: epsilon decay stays in [minimum, 1.0]
# ===========================================================================


@given(
    initial_eps=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    factor=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    minimum=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
    n_decays=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=200)
def test_epsilon_decay_stays_bounded(initial_eps, factor, minimum, n_decays):
    """After any number of decay steps, epsilon stays in [minimum, 1.0]."""
    assume(factor > 0)
    assume(minimum >= 0)
    agent = LearningAgent("l", (0, 0), epsilon=initial_eps)
    for _ in range(n_decays):
        agent.decay_epsilon(factor=factor, minimum=minimum)
    assert agent.epsilon >= minimum - 1e-9  # small tolerance for float precision
    assert agent.epsilon <= 1.0 + 1e-9


# ===========================================================================
# Property 6: trajectory stats n_steps matches actual history length
# ===========================================================================


@given(n=st.integers(min_value=1, max_value=50))
@settings(max_examples=100)
def test_trajectory_stats_n_steps_consistent(n):
    """compute_trajectory_stats n_steps equals actual history length."""
    from agentsim.analysis import compute_trajectory_stats

    agent = ReactiveAgent("a", (0, 0))
    agent.set_default("stay")
    for _ in range(n):
        agent.step({"position": (0, 0), "grid_size": 8})
    stats = compute_trajectory_stats(agent)
    assert stats["n_steps"] == n


# ===========================================================================
# Property 7: AgentState.copy() is deep-equal to original
# ===========================================================================


@given(
    pos=valid_positions,
    energy=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    reward=finite_floats,
    step=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=200)
def test_agent_state_copy_deep_equal(pos, energy, reward, step):
    """AgentState.copy() produces an object equal to but independent of the original."""
    state = AgentState(position=pos, energy=energy, reward=reward, step=step)
    state.metadata["key"] = "value"
    copy = state.copy()

    # Values are equal
    assert copy.position == state.position
    assert copy.energy == state.energy
    assert copy.reward == state.reward
    assert copy.step == state.step
    assert copy.metadata == state.metadata

    # Mutation of copy does not affect original
    copy.metadata["key"] = "mutated"
    assert state.metadata["key"] == "value"


# ===========================================================================
# Property 8: DeliberativeAgent always returns valid action on any obs
# ===========================================================================


@given(obs=valid_obs)
@settings(max_examples=100)
def test_deliberative_agent_always_valid_action(obs):
    """DeliberativeAgent never crashes and always returns a valid action."""
    agent = DeliberativeAgent("d", (0, 0))
    agent.add_goal(Goal("find_food", priority=1.0))
    agent.add_goal(Goal("explore", priority=0.5))
    action = agent.step(obs)
    assert action in agent.ACTIONS
