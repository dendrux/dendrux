# Dendrite

> The runtime for agents that act in the real world.

Client tool execution, human-in-the-loop, resumability, full observability. From prototype to production.

## Status

Under active development. Not yet published.

## Packages

| Package | Language | Status |
|---------|----------|--------|
| [dendrite](packages/python/) | Python | In development |

## Development

### Prerequisites

- Python 3.11+

### Setup

```bash
cd packages/python
pip install -e ".[dev]"    # Install in editable mode with dev tools
pre-commit install         # Set up git hooks (auto-lint on commit)
```

### Commands

All commands can be run from the repo root. They delegate to the appropriate package.

| Command | What it does |
|---------|-------------|
| `make ci` | Run all checks — lint + typecheck + tests. **Run this before every commit.** |
| `make test` | Run tests only (`pytest tests/ -v`) |
| `make lint` | Check code style and formatting (`ruff check .` + `ruff format --check .`) |
| `make typecheck` | Check type annotations (`mypy src/dendrite/`) |
| `make format` | Auto-fix formatting and lint issues (`ruff format .` + `ruff check --fix .`) |
| `make clean` | Remove build artifacts, caches |

### Typical workflow

```bash
# 1. Write code
# 2. Auto-fix formatting
make format

# 3. Run all checks
make ci

# 4. If green, commit
git commit -m "your message"
```

If `make ci` fails, `make format` often fixes lint/format issues automatically. Type errors and test failures need manual fixing.

### What `make ci` checks

1. **`ruff check .`** — Lints for code quality issues (unused imports, bad patterns, import sorting)
2. **`ruff format --check .`** — Verifies code formatting (consistent style, line length ≤100)
3. **`mypy src/dendrite/`** — Static type checking (catches type mismatches, missing annotations)
4. **`pytest tests/ -v`** — Runs all tests with verbose output

All four must pass. Same checks run on GitHub Actions CI for every push.

Built with AI pair programming using [Claude Code](https://claude.ai/code).

## License

[Apache 2.0](LICENSE)
