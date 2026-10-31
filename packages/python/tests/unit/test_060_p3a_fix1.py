"""Public contract regressions from the phase 3a review."""

import inspect

from dendrux import Agent


def test_cancel_docs_define_unscoped_cascade_and_shared_bound():
    doc = inspect.getdoc(Agent.cancel_run)
    assert "root's tenant" in doc
    assert "Budget.max_subtree_nodes" in doc
    assert "shared" in doc and "10,000" in doc


def test_legacy_migration_lock_timeout_keeps_revision(caplog):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import Mock

    import pytest
    from sqlalchemy.exc import OperationalError

    from dendrux.db._migration_ddl import bounded_step

    revision = "d7e8f9a0b1c2"
    driver_error = RuntimeError("private driver message")
    driver_error.sqlstate = "55P03"
    connection = Mock(dialect=SimpleNamespace(name="postgresql"))
    connection.begin_nested.side_effect = lambda: nullcontext()

    def execute(statement, *args):
        if str(statement) == "blocked DDL":
            raise OperationalError("blocked DDL", {}, driver_error)
        return Mock(scalar_one=lambda: "0")

    connection.execute.side_effect = execute
    with pytest.raises(RuntimeError, match="migration lock"):
        bounded_step(
            connection, "blocked DDL", revision=revision, step="add_claim_epoch", transactional=True
        )
    record = next(
        record
        for record in caplog.records
        if getattr(record, "code", None) == "migration_lock_timeout"
    )
    assert record.revision == revision
    assert "private driver message" not in record.getMessage()


def test_bounded_migration_revisions_are_in_log_enum():
    import ast
    from pathlib import Path

    from dendrux import _codes
    from dendrux.db import _migration_ddl

    revisions = set()
    for path in (Path(_migration_ddl.__file__).parent / "migrations/versions").glob("*.py"):
        tree = ast.parse(path.read_text())
        if not any(
            isinstance(node, ast.Name) and node.id == "bounded_step" for node in ast.walk(tree)
        ):
            continue
        for node in tree.body:
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "revision"
                or isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "revision" for t in node.targets)
            ):
                revisions.add(ast.literal_eval(node.value))
    assert revisions == {"d7e8f9a0b1c2", "e8f9a0b1c2d3", "f9a0b1c2d3e4"}
    assert revisions <= _codes._ENUMS["revision"]


def test_children_page_docs_state_cap():
    from dendrux.runtime.stores.sql.base import SQLStateStore

    assert "500" in inspect.getdoc(SQLStateStore.get_children_page)


def test_mcp_transport_detail_docs_state_code_summary():
    from dendrux.mcp._errors import MCPConnectionError

    doc = inspect.getdoc(MCPConnectionError)
    assert "code-and-count summary" in doc and "None" in doc


def test_source_exception_helper_docs_distinguish_transport_detail():
    from dendrux.mcp._source import safe_source_exception_detail

    doc = inspect.getdoc(safe_source_exception_detail)
    assert "transport_detail" in doc
    assert "code-and-count summary" in doc and "None" in doc
