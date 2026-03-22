# AgentSim

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

Multi-agent simulation framework for studying agent behaviors in grid-based environments. Supports reactive, deliberative (BDI), and reinforcement-learning agents with built-in scenarios, metrics, and ASCII visualization.

---

## Features

- **Three agent architectures** — Reactive (condition-action rules), Deliberative (BDI goals/beliefs/plans), and Q-learning agents in a single unified interface.
- **Grid environment** — 2D grid with configurable walls, food resources, and multi-agent support; extend via `BaseEnvironment`.
- **Turnkey scenarios** — `ForagingScenario` (multi-agent food collection) and `PursuitScenario` (predator-prey) runnable in one call.
- **Episode orchestration** — `Simulation` drives agent-environment loops across multiple episodes, tracking per-agent rewards and step counts.
- **Metrics and analysis** — `compute_metrics` and `compute_trajectory_stats` aggregate episode results into structured summaries.
- **ASCII visualization** — `render_grid_ascii` and `simulation_report` produce plain-text grid snapshots and run reports with no GUI dependency.

---

## Quick Start

```bash
pip install agentsim
```

```python
from agentsim import (
    GridEnvironment,
    LearningAgent,
    Simulation,
    SimulationConfig,
    compute_metrics,
    simulation_report,
)

# Build environment and agent
env = GridEnvironment(width=10, height=10, n_food=15, n_walls=8)
agent = LearningAgent("learner", position=(0, 0))
env.add_agent(agent)

# Run 20 episodes
cfg = SimulationConfig(max_steps=200, n_episodes=20)
sim = Simulation(env, [agent], config=cfg)
results = sim.run()

# Analyse and display
metrics = compute_metrics(results, [agent])
print(simulation_report(results, [agent]))
print(sim.summary())
```

Use a built-in scenario instead:

```python
from agentsim import ForagingScenario, make_forager

scenario = ForagingScenario(grid_size=12, n_food=20)
agents = [make_forager(f"agent_{i}", position=(i, 0)) for i in range(3)]
result = scenario.run(agents)
print(f"Collected {result.total_collected} food in {result.steps} steps "
      f"(efficiency {result.efficiency:.2f})")
```

---

## Architecture

```
agentsim/
├── agents/
│   ├── base.py          # BaseAgent, AgentState — abstract interface
│   ├── reactive.py      # ReactiveAgent — condition-action rules
│   ├── deliberative.py  # DeliberativeAgent — BDI (goals, beliefs, plans)
│   └── learning.py      # LearningAgent — Q-learning, epsilon-greedy
├── environment/
│   ├── base.py          # BaseEnvironment — reset/step/render contract
│   └── grid.py          # GridEnvironment — 2D grid, walls, food
├── scenarios/
│   ├── foraging.py      # ForagingScenario, make_forager()
│   └── pursuit.py       # PursuitScenario, make_predator(), make_prey()
├── simulation.py        # Simulation, SimulationConfig, EpisodeResult
├── analysis.py          # compute_metrics, compute_trajectory_stats
└── viz.py               # render_grid_ascii, simulation_report
```

**Data flow per episode:**

1. `Simulation.run_episode()` calls `env.reset()` → returns initial observations per agent.
2. Each step: `agent.step(obs)` → action → `env.step(agent_id, action)` → `(new_obs, reward, done)`.
3. `agent.receive_reward(reward)` updates internal state; loop continues until `done` or `max_steps`.
4. `EpisodeResult` collected; `compute_metrics()` aggregates across episodes.

**Extension points:**

- New agent type: subclass `BaseAgent`, implement `perceive()` and `decide()`.
- New environment: subclass `BaseEnvironment`, implement `reset()`, `step()`, `render()`.
- New scenario: compose agents + environment setup and delegate to `Simulation`.

---

## Development

```bash
git clone https://github.com/techknowmad/agent-sim.git
cd agent-sim
pip install -e ".[dev]"
pytest -v
ruff check .
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch, test, and PR conventions.

---

## License

MIT — see [LICENSE](LICENSE).

---

Built by [TechKnowMad Labs](https://techknowmad.ai)
