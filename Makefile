.PHONY: test lint format security coverage clean help

PYTHON ?= python3
SRC     = agentsim
TESTS   = tests

help:
	@echo "Available targets:"
	@echo "  test      — run pytest (fast, no coverage)"
	@echo "  coverage  — run pytest with coverage report"
	@echo "  lint      — check code with ruff"
	@echo "  format    — auto-format code with ruff"
	@echo "  security  — run security scan tests only"
	@echo "  clean     — remove build/cache artefacts"

test:
	$(PYTHON) -m pytest -v --tb=short $(TESTS)

coverage:
	$(PYTHON) -m pytest -v --tb=short \
		--cov=$(SRC) --cov-report=term-missing --cov-fail-under=95 \
		$(TESTS)

lint:
	$(PYTHON) -m ruff check $(SRC) $(TESTS)

format:
	$(PYTHON) -m ruff check $(SRC) $(TESTS) --fix
	$(PYTHON) -m ruff format $(SRC) $(TESTS)

security:
	$(PYTHON) -m pytest -v --tb=short -k "security" $(TESTS)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".hypothesis" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	@echo "Clean complete."
