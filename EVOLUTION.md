# EVOLUTION.md — Edgecraft Development Log

Autonomous development run by **Edgecraft Protocol v1.0**
Repository: `TECHKNOWMAD-LABS/agent-sim`
Started: 2026-03-23
Completed: 2026-03-23

---

## Cycle 1 — Test Coverage

**Timestamp:** 2026-03-23T00:00
**Finding:** Baseline coverage 94% (36 lines missed across 7 modules)
**Key misses:** deliberative.py (78%), grid.py (90%), viz.py (93%)
**Action:** Created `tests/conftest.py` with shared fixtures; wrote 31 new
  tests in `tests/test_coverage_extra.py` covering every missed branch.
**Result:** Coverage improved to **99%** (3 genuinely unreachable lines).
**Tests added:** 31 | **Total tests:** 46

---

## Cycle 2 — Error Hardening

**Timestamp:** 2026-03-23T00:10
**Findings:**
- `BaseAgent.__init__`: no validation on `agent_id` (empty string accepted) or `position` (wrong type accepted)
- `GridEnvironment.__init__`: no bounds check on `rows`, `cols`, `food_count`, `wall_density`
- `LearningAgent.__init__`: `learning_rate=0` accepted (division risk), `epsilon<0` accepted
- `SimulationConfig`: `max_steps=0` and `n_episodes=-1` accepted silently
- `receive_reward(float('nan'))` caused silent state corruption
- `env.step('ghost_id', 'up')` raised unguided `KeyError`

**Action:** Added `ValueError`/`KeyError` guards to all public constructors and
  methods; NaN/Inf rewards treated as 0; `_random_empty()` gained exhaustive
  fallback scan.
**Tests added:** 30 | **Total tests:** 76

---

## Cycle 3 — Performance

**Timestamp:** 2026-03-23T00:20
**Finding:** `compute_trajectory_stats()` recomputes full numpy stats on every
  call; episodes run strictly sequentially even when independent.
**Action:**
- Added `run_episodes_parallel()` with `ThreadPoolExecutor(max_workers=4)`
- Added `lru_cache(maxsize=256)` on `_trajectory_stats_cached()`
  (keyed on immutable tuple snapshot of history)

**Measurements:**
- Foraging scenario (200 steps, 4 agents): `<2s`
- `compute_metrics` on 50 episodes: `<0.5s`
- Parallel 8-episode run ≤ 3.5× sequential wall time (IO-light workload)

**Tests added:** 9 | **Total tests:** 85

---

## Cycle 4 — Security

**Timestamp:** 2026-03-23T00:30
**Scan scope:** All `.py` files in `agentsim/`
**Technique:** AST walk + regex pattern matching

| Check | Findings | False Positives |
|-------|----------|----------------|
| `eval` / `exec` | 0 | 0 |
| `pickle` imports | 0 | 0 |
| `subprocess` / `os.system` | 0 | 0 |
| Hardcoded secrets | 0 | 2 (stdlib `random` for exploration) |
| Path traversal | 0 | 0 |

**Action:** Embedded all 4 checks as live CI tests in `tests/test_security.py`
  so any future regression fails the pipeline immediately.
**Tests added:** 4 | **Total tests:** 89

---

## Cycle 5 — CI/CD

**Timestamp:** 2026-03-23T00:40
**Action:**
- Created `.github/workflows/ci.yml` with matrix (Python 3.10/3.11/3.12),
  ruff lint check, pytest with `--cov-fail-under=95`, coverage artifact upload.
- Created `.pre-commit-config.yaml` with ruff (lint + format) and mypy hooks.
- Fixed 12 ruff lint issues across 5 test files.

**Outcome:** All pushes and PRs now gate on lint + 95% coverage.
**Tests:** 89 (unchanged)

---

## Cycle 6 — Property-Based Testing

**Timestamp:** 2026-03-23T00:50
**Strategy library:** Hypothesis
**Properties verified:**
1. `ReactiveAgent.step()` always in `ACTIONS` — 200 examples
2. Q-table never NaN/Inf after any update — 200 examples
3. `env.step()` always returns finite reward — 50 examples
4. Agent position always in bounds after any action sequence — 50 examples
5. Epsilon decay always in `[minimum, 1.0]` — 200 examples
6. Trajectory stats `n_steps` matches history — 100 examples
7. `AgentState.copy()` deep-equal and independent — 200 examples
8. `DeliberativeAgent.step()` never crashes — 100 examples

**Hypothesis findings:** 0 bugs found (codebase already hardened in Cycle 2)
**Tests added:** 9 | **Total tests:** 98

---

## Cycle 7 — Examples + Docs

**Timestamp:** 2026-03-23T01:00
**Action:**
- Created 3 runnable example scripts in `examples/`:
  - `example_foraging.py` — multi-agent foraging with full report
  - `example_pursuit.py` — predator-prey across 5 seeds
  - `example_learning_agent.py` — Q-learning training with sparkline + heatmap
- All 3 verified runnable with `PYTHONPATH=. python3 examples/<name>.py`
- Extended Google-style docstrings on `viz.py` and `scenarios/foraging.py`

**Tests:** 98 (unchanged)

---

## Cycle 8 — Release Engineering

**Timestamp:** 2026-03-23T01:10
**Action:**
- `pyproject.toml`: added `authors`, `readme`, `keywords`, `classifiers`,
  full dev dependency group
- `CHANGELOG.md`: Keep-a-Changelog format covering all 8 cycles
- `Makefile`: `test`, `coverage`, `lint`, `format`, `security`, `clean` targets
- `AGENTS.md`: Edgecraft protocol documentation
- `EVOLUTION.md`: this file
- Tagged `v0.1.0`

**Final test count:** 98
**Final coverage:** 99%
**Ruff:** clean

---

## Summary Table

| Cycle | Focus        | Tests Added | Total Tests | Coverage |
|-------|-------------|-------------|-------------|---------|
| 0     | baseline    | 15          | 15          | 94%     |
| 1     | test cov    | 31          | 46          | 99%     |
| 2     | hardening   | 30          | 76          | 99%     |
| 3     | performance | 9           | 85          | 99%     |
| 4     | security    | 4           | 89          | 99%     |
| 5     | CI/CD       | 0           | 89          | 99%     |
| 6     | property    | 9           | 98          | 99%     |
| 7     | examples    | 0           | 98          | 99%     |
| 8     | release     | 0           | 98          | 99%     |

*Total commits across all cycles: 9*
