# ─── AquaIA — developer shortcuts ────────────────────────────────────────────
#
# Prerequisites (macOS/Homebrew — use a venv):
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install -r requirements-dev.txt
#
# Usage:
#   make lint          lint only the files changed vs origin/development (= CI)
#   make lint-fix      same, but auto-fix what ruff can
#   make lint-all      lint every Python file in the repo
#   make format        auto-format every Python file with ruff
#   make test          run the pytest suite
#   make ci            run lint + test together (full local CI simulation)
#
# All lint targets exit with the same code ruff would, so you can use them
# in git hooks or scripts.

PYTHON     ?= python3
BASE_BRANCH ?= origin/development

# Compute the list of .py files changed since the base branch (same logic as
# the CI workflow — git diff --diff-filter=ACMR excludes deleted files).
CHANGED_PY := $(shell git diff --name-only --diff-filter=ACMR $(BASE_BRANCH)...HEAD -- '*.py' 2>/dev/null | tr '\n' ' ')

.PHONY: lint lint-fix lint-all format test ci _check-ruff

# Guard: print a helpful message if ruff is not installed.
_check-ruff:
	@command -v ruff >/dev/null 2>&1 || \
		{ echo "ruff not found — activate your venv then: pip install -r requirements-dev.txt"; exit 1; }

lint: _check-ruff
	@if [ -z "$(CHANGED_PY)" ]; then \
		echo "No Python files changed vs $(BASE_BRANCH) — nothing to lint."; \
	else \
		echo "==> ruff check (changed files vs $(BASE_BRANCH))"; \
		ruff check $(CHANGED_PY); \
		echo "==> ruff format --check (changed files vs $(BASE_BRANCH))"; \
		ruff format --check $(CHANGED_PY); \
	fi

lint-fix: _check-ruff
	@if [ -z "$(CHANGED_PY)" ]; then \
		echo "No Python files changed vs $(BASE_BRANCH) — nothing to fix."; \
	else \
		echo "==> ruff check --fix (changed files vs $(BASE_BRANCH))"; \
		ruff check --fix $(CHANGED_PY); \
	fi

lint-all: _check-ruff
	@echo "==> ruff check (all Python files)"
	ruff check .
	@echo "==> ruff format --check (all Python files)"
	ruff format --check .

format: _check-ruff
	@echo "==> ruff format (all Python files)"
	ruff format .

test:
	@echo "==> pytest"
	pytest -q

ci: lint test
