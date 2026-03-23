# Changelog

All notable changes to **agentsim** are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.1.0] — 2026-03-23

Initial release produced by **8 autonomous Edgecraft development cycles**.

### Added — Cycle 1: Test Coverage
- `tests/conftest.py`: shared fixtures (`basic_reactive_agent`, `forager_agent`,
  `small_grid`, `seeded_grid`, `simple_sim`) and `make_obs()` helper.
- `tests/test_coverage_extra.py`: 31 tests covering previously-missed branches in
  `deliberative.py`, `learning.py`, `reactive.py`, `analysis.py`, `grid.py`,
  `simulation.py`, `viz.py`, `pursuit.py`.
- Coverage raised from **94% → 99%** (3 unreachable lines remain).

### Added — Cycle 2: Error Hardening
- `agentsim/agents/base.py`: `ValueError` for empty/invalid `agent_id` and
  `position`; non-finite rewards silently treated as 0; `None` observation
  handled gracefully.
- `agentsim/agents/learning.py`: validation for `state_size`, `learning_rate`,
  `discount`, `epsilon`, `factor`, `minimum`; `None` observation safe in
  `perceive()`; non-finite reward in `update()` treated as 0.
- `agentsim/environment/grid.py`: validation for `rows`, `cols`, `food_count`,
  `wall_density`; `add_agent()` validates id and bounds; `step()` and
  `get_observation()` raise `KeyError` for unknown agents; non-string action
  treated as `stay`; exhaustive fallback in `_random_empty()`.
- `agentsim/simulation.py`: `SimulationConfig.__post_init__` validates
  `max_steps`, `n_episodes`, `render_every`; `Simulation.__init__` rejects
  `None` env and empty agent list.
- `tests/test_error_hardening.py`: 30 validation tests.

### Added — Cycle 3: Performance
- `agentsim/analysis.py`: `run_episodes_parallel()` using `ThreadPoolExecutor`
  for parallel episode execution; `lru_cache(maxsize=256)` on trajectory stats
  computation kernel `_trajectory_stats_cached()`.
- `tests/test_performance.py`: 9 tests covering parallelism correctness, cache
  hits/misses, and timing benchmarks (foraging < 2s, metrics < 0.5s).

### Added — Cycle 4: Security
- `tests/test_security.py`: 4 live AST-based security tests that fail CI if
  `eval`, `exec`, `pickle`, `subprocess`, or hardcoded secrets are introduced.
- Scan result: **0 findings**, 2 false positives filtered (stdlib `random` for
  agent exploration — not cryptographic use).

### Added — Cycle 5: CI/CD
- `.github/workflows/ci.yml`: GitHub Actions pipeline running ruff + pytest
  with 95% coverage gate on Python 3.10, 3.11, and 3.12.
- `.pre-commit-config.yaml`: ruff (lint + format) and mypy hooks.
- Fixed 12 ruff lint issues across 5 test files (import ordering, unused
  imports, f-string without placeholders).

### Added — Cycle 6: Property-Based Testing
- `tests/test_property_based.py`: 9 Hypothesis property tests with strategies
  covering 50–200 examples each:
  - ReactiveAgent always returns a valid action
  - Q-table never produces NaN/Inf
  - Step reward always finite
  - Agent position always in bounds
  - Epsilon decay bounded in `[minimum, 1.0]`
  - Trajectory stats consistent with history length
  - `AgentState.copy()` deep-equal and independent
  - DeliberativeAgent never crashes

### Added — Cycle 7: Examples + Docs
- `examples/example_foraging.py`: 3-agent foraging with full metrics report.
- `examples/example_pursuit.py`: predator-prey across 5 seeds + 20×20 grid.
- `examples/example_learning_agent.py`: 30-episode Q-learning training with
  sparkline trend and visitation heatmap.
- Extended Google-style docstrings on `viz.py` and `scenarios/foraging.py`.

### Changed — Cycle 8: Release Engineering
- `pyproject.toml`: added `authors`, `readme`, `keywords`, `classifiers`, and
  full dev dependency group (`pytest-cov`, `hypothesis`, `mypy`, `pre-commit`).
- `Makefile`: `test`, `lint`, `format`, `security`, `coverage`, `clean` targets.
- `AGENTS.md`: Edgecraft autonomous development protocol documentation.
- `EVOLUTION.md`: per-cycle timestamps and findings log.

---

[0.1.0]: https://github.com/TECHKNOWMAD-LABS/agent-sim/releases/tag/v0.1.0
