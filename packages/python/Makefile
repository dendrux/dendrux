.PHONY: test lint typecheck format package-smoke ci clean

test:
	pytest tests/ -v --cov=src/dendrux --cov-report=term-missing --cov-fail-under=80

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy src/dendrux/

format:
	ruff format .
	ruff check --fix .

package-smoke:
	python scripts/mcp_distribution_smoke.py

ci: lint typecheck test

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
