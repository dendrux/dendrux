"""Every Postgres-touching integration test creates its worker database first (no connection).

Under ``pytest -n`` each worker owns ``<name>_gwN_test``, which does not exist on a fresh
server. ``pg_database`` creates it, and it must sit in the fixture closure of every test
that opens a Postgres connection, including the migration and schema tests that manage
their own schema instead of using the ``engine`` fixture. Collection only; nothing
connects to a database.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

INTEGRATION = Path(__file__).resolve().parents[1] / "integration"
HELPER_FILES = {"conftest.py", "_migration_schema.py", "__init__.py"}


class _Collected:
    def __init__(self) -> None:
        self.items: list[pytest.Item] = []

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.items = list(session.items)


@pytest.fixture(scope="module")
def integration_items() -> list[pytest.Item]:
    collected = _Collected()
    # The outer run may pass options (a random seed, say) through PYTEST_ADDOPTS. This nested
    # run switches plugins off itself and must not inherit them. The setenv stands in for
    # such a run, so a regression shows up here too.
    patch = pytest.MonkeyPatch()
    patch.setenv("PYTEST_ADDOPTS", "--randomly-seed=7")
    patch.delenv("PYTEST_ADDOPTS")
    code = pytest.main(
        [
            str(INTEGRATION),
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:randomly",
            "-p",
            "no:xdist",
            "-p",
            "no:cov",
            "-o",
            "addopts=",
        ],
        plugins=[collected],
    )
    patch.undo()
    assert code == 0 and collected.items, "could not collect tests/integration"
    return collected.items


def _imports_migration_schema(path: Path) -> bool:
    tree = ast.parse(path.read_text())
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "tests.integration._migration_schema"
        and any(alias.name == "migration_schema" for alias in node.names)
        for node in ast.walk(tree)
    )


def test_engine_fixture_requires_the_database_bootstrap(
    integration_items: list[pytest.Item],
) -> None:
    users = [item for item in integration_items if "engine" in item.fixturenames]  # type: ignore[attr-defined]
    assert users
    for item in users:
        names = item.fixturenames  # type: ignore[attr-defined]
        assert "pg_database" in names and "_pg_schema_setup" in names, item.nodeid


def test_schema_and_migration_tests_require_the_database_bootstrap(
    integration_items: list[pytest.Item],
) -> None:
    migration_modules = {
        p.name for p in INTEGRATION.glob("test_*.py") if _imports_migration_schema(p)
    }
    assert {
        "test_schema_check.py",
        "test_050_t1_r1_migration.py",
        "test_migration_image_blobs.py",
        "test_migration_run_control.py",
    } <= migration_modules
    checked = 0
    for item in integration_items:
        if Path(str(item.path)).name not in migration_modules:
            continue
        # SQLite-only tests in these modules never open Postgres; the Postgres ones are the
        # ones that name the fixture or use migration_schema, found by the marker below.
        if "engine" in item.fixturenames or not _opens_postgres(item):  # type: ignore[attr-defined]
            continue
        checked += 1
        assert "pg_database" in item.fixturenames, item.nodeid  # type: ignore[attr-defined]
    assert checked >= 4


def _opens_postgres(item: pytest.Item) -> bool:
    """True when the test function (not just its module) goes through migration_schema."""
    names = set(item.fixturenames)  # type: ignore[attr-defined]
    if names & {"pg", "pg_url", "pg_test_url"}:
        return True
    fn = getattr(item, "function", None)
    return fn is not None and "migration_schema" in fn.__code__.co_names


def test_no_test_module_reads_the_database_url_around_the_bootstrap() -> None:
    offenders = []
    for path in sorted(INTEGRATION.glob("test_*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and node.value == "DENDRUX_TEST_PG_URL"
                and not _is_docstring_or_skip_text(tree, node)
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], f"use the pg_test_url fixture instead of the env var: {offenders}"


def _is_docstring_or_skip_text(tree: ast.Module, node: ast.Constant) -> bool:
    doc = tree.body[0] if tree.body else None
    return isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant) and doc.value is node
