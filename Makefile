# Makefile for the "RAG against the machine" project.
# Written to run identically on Linux (GNU make, dash) and macOS (BSD).
# Recipes stay POSIX sh: no bashisms, so /bin/sh -> dash is safe.

SHELL := /bin/sh

UV := uv
MODULE := src

# Mandatory mypy flags, as defined by the Common Instructions.
MYPY_FLAGS := --warn-return-any --warn-unused-ignores --ignore-missing-imports \
              --disallow-untyped-defs --check-untyped-defs

QUESTION ?= How is the scheduler configured?
K ?= 5

.PHONY: all check-uv install reinstall run debug test lint lint-strict clean re \
        index search evaluate

all: run

# Fail early and explain, instead of letting every rule die on "uv: not found".
check-uv:
	@command -v $(UV) >/dev/null 2>&1 || { \
		echo "Error: 'uv' was not found in PATH."; \
		echo "  Install it with:"; \
		echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		echo "  then open a new shell, or run:"; \
		echo '    export PATH="$$HOME/.local/bin:$$PATH"'; \
		exit 1; \
	}

# `uv sync` creates .venv and installs every dependency, including the
# transformers/torch stack used by the answer commands.
install: check-uv
	$(UV) sync

# Rebuild the environment from scratch. Use this when the sources were copied
# between machines: a .venv built on another operating system holds binaries
# that cannot run here, and removing it is faster than diagnosing it.
reinstall: check-uv
	rm -rf .venv
	$(UV) sync

run: check-uv
	$(UV) run python -m $(MODULE)

debug: check-uv
	$(UV) run python -m pdb -m $(MODULE)

test: check-uv
	$(UV) run pytest

lint: check-uv
	$(UV) run flake8 .
	$(UV) run mypy . $(MYPY_FLAGS)

lint-strict: check-uv
	$(UV) run flake8 .
	$(UV) run mypy . --strict

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.mypy_cache' -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.pytest_cache' -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

re: clean run

index: check-uv
	$(UV) run python -m $(MODULE) index

search: check-uv
	$(UV) run python -m $(MODULE) search "$(QUESTION)" --k $(K)

evaluate: check-uv
	$(UV) run python -m $(MODULE) evaluate \
		--student_results_path data/output/search_results/dataset_docs_public.json \
		--dataset_path datasets_public/public/AnsweredQuestions/dataset_docs_public.json \
		--k 10
