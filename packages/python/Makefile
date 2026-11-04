.PHONY: dev-sync cov-mysql test test-fast test-area test-pg test-kit test-merge test-lists test-shard-proof test-model-check test-model-install test-dbs-up test-dbs-down test-redis-up test-redis-down test-redis test-mysql bench load load-quick load-pg load-mysql load-mariadb lint typecheck arch crap cognitive duplicates deadcode cycles semgrep quality quality-stamp api-check format package-smoke ci clean
.DEFAULT_GOAL := test

# Match a fresh CI install: exact dev tools and the newest allowed runtime dependencies.
DEV_EXTRAS ?= dev,all
dev-sync:
	python -m pip install --upgrade --upgrade-strategy eager -e ".[$(DEV_EXTRAS)]"

# Opt-in: both throwaway containers must already be running; never manage their lifecycle.
cov-mysql:
	python -m tools.mysql_coverage

# Parallel workers for local runs: the core count, never more than 6 (owner decision 2026-11-01;
# the machine also runs Docker and other builders). TEST_WORKERS=<n> (command line or environment)
# and DENDRUX_TEST_WORKERS=<n> can lower it; a larger number or `auto` ends at the cap.
# GitHub CI runs pytest itself with `-n auto`.
override TEST_WORKERS := $(shell python tools/test_tiers.py workers "$(TEST_WORKERS)")

# Benchmarks measure the server, not the code, so they are not part of the gates.
#
# The last line is not a baseline stamp: it records the digests of the source this coverage.json
# measured (coverage.json.sources.json), which `make crap` requires. Baselines are changed only
# by `make quality-stamp`, run by the coordinator.
test: test-model-check
	pytest tests/ -v -n $(TEST_WORKERS) -m "not benchmark" --cov=src/dendrux --cov-report=term-missing --cov-report=json:coverage.json --cov-fail-under=80
	python tools/crap_ratchet.py stamp --coverage coverage.json

# Installed explicitly before xdist workers start; tests never fetch model data.
test-model-check:
	python -m tests._helpers.nlp_models

test-model-install:
	python -m tests._helpers.nlp_models --install

# ---- Fast loop (tools/QUALITY.md, "Test tiers"). The merge gate below is the full check. ----
# test-fast: only the tests affected by your change (pytest-testmon), SQLite, no coverage, ruff on
# the changed files. The first run (no testmon data yet) runs every non-slow test and records
# what each one touches; later runs are small. The fast loop never runs the tests marked slow
# (a deliberate choice: `make ci` at hand-back runs them); SLOW=1 includes them.
SLOW ?=
FAST_MARKS = not benchmark$(if $(SLOW),, and not slow)
TESTMON_DATAFILE ?= .pytest_cache/testmondata
export TESTMON_DATAFILE

test-fast: test-model-check
	python tools/test_tiers.py ruff-changed
	python tools/test_tiers.py prepare-testmon
	env -u DENDRUX_TEST_PG_URL pytest tests/ --testmon --testmon-forceselect -n $(TEST_WORKERS) -m "$(FAST_MARKS)" --no-cov -q

# test-area: one area (python tools/test_tiers.py areas), SQLite, no coverage.
# Example: make test-area AREA=runtime
test-area: test-model-check
	@python tools/test_tiers.py check-area "$(AREA)"
	env -u DENDRUX_TEST_PG_URL pytest tests/ -m "area_$(AREA) and $(FAST_MARKS)" -n $(TEST_WORKERS) --no-cov -q

# test-pg: the Postgres integration matrix in parallel, one database per worker (needs
# DENDRUX_TEST_PG_URL naming a private database that ends in _test). No coverage.
test-pg: test-model-check
	@test -n "$$DENDRUX_TEST_PG_URL" || { echo "DENDRUX_TEST_PG_URL is required"; exit 2; }
	pytest tests/integration -n $(TEST_WORKERS) -m "not benchmark" --no-cov -q -rs

# test-kit: the store kit and conformance suite on SQLite and Postgres. Postgres cells run serial
# (-n 0) on exactly one authorised database (tests/conformance/pg_guard.py); this target authorises
# the database named by DENDRUX_TEST_PG_URL. MySQL and MariaDB cells skip unless their URLs are set.
test-kit: test-model-check
	@test -n "$$DENDRUX_TEST_PG_URL" || { echo "DENDRUX_TEST_PG_URL is required"; exit 2; }
	DENDRUX_CONFORMANCE_PG_DB="$$(python tools/test_tiers.py dbname "$$DENDRUX_TEST_PG_URL")" \
		pytest tests/conformance -n 0 -m "not benchmark" --no-cov -q -rs

# test-merge: the merge gate, run by the coordinator. Local scope is SQLite and Postgres only
# (owner decision 2026-11-01); MySQL and MariaDB run in GitHub CI. Needs DENDRUX_TEST_PG_URL, a
# private Postgres database ending in _test. It runs: make ci (lint, types, architecture, the full
# suite on SQLite and the Postgres matrix in parallel, CRAP and the quality gates, API check), then
# the store kit serially on SQLite and Postgres, then the benchmarks serially (they measure the
# server and fail under parallel load). The live provider smoke is separate (coordinator, .env).
test-merge:
	@test -n "$$DENDRUX_TEST_PG_URL" || { echo "DENDRUX_TEST_PG_URL is required"; exit 2; }
	@test -z "$$DENDRUX_TEST_MYSQL_URL$$DENDRUX_TEST_MARIADB_URL" || { echo "unset DENDRUX_TEST_MYSQL_URL and DENDRUX_TEST_MARIADB_URL: local gates use SQLite and Postgres only"; exit 2; }
	$(MAKE) ci
	$(MAKE) test-kit
	$(MAKE) bench

# Check the slow/flaky lists name real tests; print the CI shard split and prove it is a cover.
test-lists:
	python tools/test_tiers.py check-lists

test-shard-proof:
	python tools/test_tiers.py shard-proof --count $(or $(SHARDS),4)

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy src/dendrux/

# CRAP ratchet. Needs coverage.json, which `make test` writes. Enforcing: a new function over
# the threshold, or an existing one whose score rises, fails (tools/QUALITY.md).
# Scores are only comparable with CI-grade coverage: set DENDRUX_TEST_PG_URL for `make test`,
# or the skipped Postgres tests lower coverage and the ratchet reports false regressions.
crap:
	python tools/crap_ratchet.py check --coverage coverage.json

# Quality gates (tools/QUALITY.md). Each one is a ratchet against a committed baseline:
# a new problem, or an existing one getting worse, fails. `make quality-stamp` rewrites the
# baselines after an improvement; it refuses to record a new problem.
cognitive:
	python tools/cognitive_ratchet.py check

duplicates:
	python tools/duplicate_ratchet.py check

deadcode:
	python tools/vulture_ratchet.py check

cycles:
	python tools/import_cycles.py check

# Needs semgrep (tools/semgrep-requirements.txt) in its own environment; point to it with
# `make semgrep SEMGREP=/path/to/semgrep` when it is not on PATH.
semgrep:
	python tools/semgrep_ratchet.py test
	python tools/semgrep_ratchet.py check

quality: cognitive duplicates deadcode cycles semgrep crap

quality-stamp:
	python tools/cognitive_ratchet.py stamp
	python tools/duplicate_ratchet.py stamp
	python tools/vulture_ratchet.py stamp
	python tools/import_cycles.py stamp
	python tools/semgrep_ratchet.py stamp
	python tools/crap_ratchet.py update --coverage coverage.json

# MySQL 8.4 and MariaDB 10.11 test servers (docker/test-dbs.compose.yml): only the
# dendrux-test-mysql and dendrux-test-mariadb containers, on 127.0.0.1:53306 / 53307.
# Start them for a test run and remove them (with their data) afterwards.
TEST_DBS = docker compose -f docker/test-dbs.compose.yml

test-dbs-up:
	$(TEST_DBS) up -d --wait

test-dbs-down:
	$(TEST_DBS) down -v

# Redis for the RedisUpdates tests (docker/test-dbs.compose.yml, service redis, 127.0.0.1:56379).
# The tests skip cleanly when DENDRUX_TEST_REDIS_URL is unset; GitHub CI runs them on a service.
REDIS_TEST_URL ?= redis://127.0.0.1:56379/0

test-redis-up:
	$(TEST_DBS) up -d --wait redis

test-redis-down:
	$(TEST_DBS) rm -sfv redis

test-redis: test-model-check
	DENDRUX_TEST_REDIS_URL=$(REDIS_TEST_URL) pytest tests/unit/test_redis_updates*.py tests/integration/test_redis_updates*.py -n $(TEST_WORKERS) -rs --no-cov -q

# GitHub CI only (owner decision 2026-11-01: local runs use SQLite and Postgres only).
# Schema and maintenance suites on MySQL/MariaDB. Needs DENDRUX_TEST_MYSQL_URL and/or
# DENDRUX_TEST_MARIADB_URL (a database name ending in _test; each test makes its own copy).
test-mysql:
	pytest tests/db -v -rs -n $(TEST_WORKERS) -m "not benchmark" --no-cov

# Always serial (-n 0): benchmarks measure the server and fail under parallel load. Release step:
# run on a quiet machine (nothing else using Postgres) and record the numbers in the load-test
# results doc. Needs DENDRUX_TEST_PG_URL.
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
# runs have no stored baseline: the declared ceilings are in tools/load_budgets_mysql.json (set before
# measuring); override with LOAD_STATEMENT_BUDGETS=path (see docs/guides/scaling.mdx, "Load testing").
LOAD_MYSQL_URL ?= mysql+asyncmy://root:dendrux_test@127.0.0.1:53306/dendrux_load_mysql_test
LOAD_MARIADB_URL ?= mysql+asyncmy://root:dendrux_test@127.0.0.1:53307/dendrux_load_mariadb_test
LOAD_STATEMENT_BUDGETS ?= tools/load_budgets_mysql.json
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

ci: lint typecheck arch test crap quality api-check

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.json coverage.json.sources.json
