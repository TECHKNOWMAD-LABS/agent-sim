# AGENTS.md — Edgecraft Autonomous Development Protocol

This repository was developed autonomously using the **Edgecraft Protocol v1.0**,
a structured 8-cycle autonomous software engineering methodology developed by
[TechKnowMad Labs](https://techknowmad.ai).

---

## What is Edgecraft?

Edgecraft is a self-directed development loop in which an AI agent iterates
through well-defined engineering cycles — test coverage, error hardening,
performance, security, CI/CD, property testing, documentation, and release —
without human intervention between cycles.

Each cycle follows the **L-notation protocol**:

| Level | Label        | Meaning                                          |
|-------|-------------|--------------------------------------------------|
| L1    | detection    | Identify a gap or deficiency                     |
| L2    | noise        | Filter false positives / irrelevant signals      |
| L3    | sub-noise    | Surface subtle edge cases                        |
| L4    | conjecture   | Form a hypothesis about a fix or improvement     |
| L5    | action       | Implement the change                             |
| L6    | grounding    | Verify with tests or measurements                |
| L7    | flywheel     | Generalise the lesson to other modules/repos     |

---

## The 8 Edgecraft Cycles

### Cycle 1 — Test Coverage
- Run baseline coverage analysis
- Identify modules at 0% or low coverage
- Write tests for every missed branch
- Commit with coverage delta

### Cycle 2 — Error Hardening
- Audit all public APIs for missing validation
- Handle None, NaN, Inf, empty, and out-of-bounds inputs
- Add graceful error messages (no bare `except`)
- Write tests confirming each guard

### Cycle 3 — Performance
- Identify sequential operations suitable for parallelism
- Add `ThreadPoolExecutor` or `asyncio.gather` + semaphore patterns
- Add `functools.lru_cache` for pure, expensive computations
- Measure and document speedups in commit messages

### Cycle 4 — Security
- AST scan: eval, exec, pickle, subprocess, os.system
- Pattern scan: hardcoded secrets, API keys, tokens
- Document findings and false positives
- Embed scan as a live CI test (regression guard)

### Cycle 5 — CI/CD
- Create `.github/workflows/ci.yml` (multi-Python, lint + test + coverage gate)
- Create `.pre-commit-config.yaml` (ruff + mypy)
- Fix all lint issues before committing
- Ensure every push triggers automated checks

### Cycle 6 — Property-Based Testing
- Write Hypothesis strategies for core data types
- Test invariants: range bounds, round-trips, no-crash guarantees
- Run with ≥50 examples per property
- Fix any Hypothesis-discovered edge cases before committing

### Cycle 7 — Examples + Docs
- Write 2-3 runnable scripts in `examples/`
- Verify each example runs without errors
- Add Google-style docstrings to every public function
- Ensure examples import cleanly via `PYTHONPATH=.`

### Cycle 8 — Release Engineering
- Validate `pyproject.toml` has author, license, classifiers
- Write `CHANGELOG.md` following Keep a Changelog format
- Create `Makefile` with `test`, `lint`, `format`, `security`, `clean`
- Tag `v0.1.0` and push with tags

---

## Commit Convention

All Edgecraft commits are prefixed with L-notation to make the development
rationale machine-readable and auditable:

```
L5/action: add input validation and error handling
L6/grounding: 30 tests passing, coverage 99%
L3/sub-noise: hypothesis found edge case — NaN epsilon
```

---

## Running the Protocol Yourself

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests with coverage
make coverage

# Lint
make lint

# Security scan
make security

# Run examples
PYTHONPATH=. python3 examples/example_foraging.py
PYTHONPATH=. python3 examples/example_pursuit.py
PYTHONPATH=. python3 examples/example_learning_agent.py
```

---

*Edgecraft Protocol v1.0 — TechKnowMad Labs, 2026*
