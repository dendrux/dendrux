.PHONY: test lint typecheck crap format package-smoke ci clean

test:
	pytest tests/ -v -n auto --cov=src/dendrux --cov-report=term-missing --cov-report=json:coverage.json --cov-fail-under=80

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy src/dendrux/

# CRAP ratchet. Needs coverage.json, which `make test` writes.
crap:
	python tools/crap_ratchet.py check --coverage coverage.json

format:
	ruff format .
	ruff check --fix .

package-smoke:
	python scripts/mcp_distribution_smoke.py

ci: lint typecheck test crap

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.json
