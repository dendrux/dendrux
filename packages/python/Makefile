.PHONY: test test-model-check test-model-install bench lint typecheck arch crap api-check format package-smoke ci clean

# Benchmarks measure the server, not the code, so they are not part of the gates.
test: test-model-check
	pytest tests/ -v -n auto -m "not benchmark" --cov=src/dendrux --cov-report=term-missing --cov-report=json:coverage.json --cov-fail-under=80
	python tools/crap_ratchet.py stamp --coverage coverage.json

# Installed explicitly before xdist workers start; tests never fetch model data.
test-model-check:
	python -m tests._helpers.nlp_models

test-model-install:
	python -m tests._helpers.nlp_models --install

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy src/dendrux/

# CRAP ratchet. Needs coverage.json, which `make test` writes.
crap:
	python tools/crap_ratchet.py check --coverage coverage.json

# Release step: run on a quiet machine (nothing else using Postgres) and record the numbers
# in the load-test results doc. Needs DENDRUX_TEST_PG_URL.
bench:
	pytest tests/ -v -rs -n 0 -m benchmark -p no:randomly

# Public API breaks against the last release tag (needs `git fetch --tags`).
# Fails on any break without a migration-note B-item.
api-check:
	python tools/api_breaks.py

# Layer and dependency rules from .importlinter.
arch:
	lint-imports

format:
	ruff format .
	ruff check --fix .

package-smoke:
	python scripts/mcp_distribution_smoke.py

ci: lint typecheck arch test crap api-check

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.json coverage.json.sources.json
