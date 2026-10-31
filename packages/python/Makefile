.PHONY: test test-model-check test-model-install test-dbs-up test-dbs-down test-mysql bench load load-quick load-pg load-mysql load-mariadb lint typecheck arch crap api-check format package-smoke ci clean

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

# MySQL 8.4 and MariaDB 10.11 test servers (docker/test-dbs.compose.yml): only the
# dendrux-test-mysql and dendrux-test-mariadb containers, on 127.0.0.1:53306 / 53307.
# Start them for a test run and remove them (with their data) afterwards.
TEST_DBS = docker compose -f docker/test-dbs.compose.yml

test-dbs-up:
	$(TEST_DBS) up -d --wait

test-dbs-down:
	$(TEST_DBS) down -v

# Schema and maintenance suites on MySQL/MariaDB. Needs DENDRUX_TEST_MYSQL_URL and/or
# DENDRUX_TEST_MARIADB_URL (a database name ending in _test; each test makes its own copy).
test-mysql:
	pytest tests/db -v -rs -n auto -m "not benchmark" --no-cov

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

# The 1,000-run scenario on each database family, with resource monitoring and four workload cycles.
# load-pg needs DENDRUX_LOAD_PG_URL (a private Postgres database ending in _test). load-mysql and
# load-mariadb use their own private databases on the test servers (make test-dbs-up first); the
# harness refuses a MySQL-family name that lacks "_load_" or does not end in "_test". MySQL-family
# runs have no stored baseline: declare the statement ceilings in a JSON file first and pass it as
# LOAD_STATEMENT_BUDGETS=path (see docs/guides/scaling.mdx, "Load testing your own deployment").
LOAD_MYSQL_URL ?= mysql+asyncmy://root:dendrux_test@127.0.0.1:53306/dendrux_load_mysql_test
LOAD_MARIADB_URL ?= mysql+asyncmy://root:dendrux_test@127.0.0.1:53307/dendrux_load_mariadb_test
LOAD_RUNS = --scale full --scenarios runs --workers 2 --leak-cycles 4

load-pg:
	python tools/load.py $(LOAD_RUNS) --json $(LOAD_JSON)

load-mysql:
	@test -n "$(LOAD_STATEMENT_BUDGETS)" || { echo "LOAD_STATEMENT_BUDGETS is required: a JSON file of declared statement ceilings"; exit 2; }
	python tools/load.py --url $(LOAD_MYSQL_URL) $(LOAD_RUNS) --statement-budgets $(LOAD_STATEMENT_BUDGETS) --json $(LOAD_JSON)

load-mariadb:
	@test -n "$(LOAD_STATEMENT_BUDGETS)" || { echo "LOAD_STATEMENT_BUDGETS is required: a JSON file of declared statement ceilings"; exit 2; }
	python tools/load.py --url $(LOAD_MARIADB_URL) $(LOAD_RUNS) --statement-budgets $(LOAD_STATEMENT_BUDGETS) --json $(LOAD_JSON)

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
