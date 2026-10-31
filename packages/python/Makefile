.PHONY: test test-model-check test-model-install bench load load-quick lint typecheck arch crap api-check format package-smoke ci clean

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
# Report only: the hotspot list feeds refactor work; it never fails CI.
crap:
	python tools/crap_ratchet.py check --report-only --coverage coverage.json

# Release step: run on a quiet machine (nothing else using Postgres) and record the numbers
# in the load-test results doc. Needs DENDRUX_TEST_PG_URL.
bench:
	pytest tests/ -v -rs -n 0 -m benchmark -p no:randomly

# Scale evidence. Needs a private Postgres database whose name ends in _test (the script drops
# its tables), given as DENDRUX_LOAD_PG_URL. `load` is the full run (tens of minutes, run it on a
# quiet machine); `load-quick` is the CI subset and fails on a >20% statement-count regression
# against tools/load_baseline.json. Both also enforce fixed 0.6 Scenario A budgets;
# the v0.5 reference is informational. Both write their full results to $(LOAD_JSON).
LOAD_JSON ?= $(or $(TMPDIR),/tmp)/dendrux-load-results.json

load:
	python tools/load.py --scale full --workers 2 --json $(LOAD_JSON)

load-quick:
	python tools/load.py --scale quick --workers 2 --baseline tools/load_baseline.json --json $(LOAD_JSON)

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
