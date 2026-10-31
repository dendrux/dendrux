"""Adapter and retry contracts corrected in the first review round."""

from types import SimpleNamespace

import pytest
from sqlalchemy import case
from sqlalchemy.exc import DBAPIError, OperationalError, TimeoutError

from dendrux.db.models import AgentRun
from dendrux.runtime import durability
from dendrux.runtime.stores.sql.base import SQLStateStore


@pytest.mark.parametrize(
    "code", [None, "08006", "08003", "53300", "57014", "57P01", "57P02", "57P03"]
)
def test_before_commit_transient_failures_keep_retryability(code):
    class DriverError(Exception):
        sqlstate = code

    error = (
        OperationalError(None, None, DriverError("PRIVATE"))
        if code is None
        else DBAPIError(None, None, DriverError("PRIVATE"))
    )
    assert durability.is_transient_db_error(error)


def test_pool_timeout_keeps_retryability():
    assert durability.is_transient_db_error(TimeoutError())


@pytest.mark.parametrize("site", ["mark", "sweep"])
def test_assignment_dependencies_preserve_old_cancel_value(site):
    from sqlalchemy.dialects import mysql

    table = AgentRun.__table__
    status = case((table.c.cancel_requested.is_(True), "cancelled"), else_="error")
    values = {"status": status, "cancel_requested": False}
    if site == "mark":
        values["status_changed_at"] = case(
            (table.c.status != status, table.c.updated_at), else_=table.c.status_changed_at
        )
    stmt = table.update().ordered_values(
        *SQLStateStore._ordered(object.__new__(SQLStateStore), table, values)
    )
    sql = str(stmt.compile(dialect=mysql.dialect()))
    assert sql.index("status=CASE") < sql.index("cancel_requested=")
    if site == "mark":
        assert sql.index("status_changed_at=CASE") < sql.index("status=CASE")


def test_cyclic_assignment_dependencies_fail_before_execute():
    table = AgentRun.__table__
    with pytest.raises(ValueError, match="[Cc]ycl"):
        SQLStateStore._ordered(
            object.__new__(SQLStateStore),
            table,
            {
                "claim_epoch": table.c.reclaim_attempts + 1,
                "reclaim_attempts": table.c.claim_epoch + 1,
            },
        )


async def test_concrete_store_refuses_wrong_engine():
    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux.runtime.stores.sql.postgres import PostgresStateStore

    engine = create_async_engine("sqlite+aiosqlite://")
    try:
        with pytest.raises(ValueError, match="dialect"):
            PostgresStateStore(engine)
    finally:
        await engine.dispose()


def test_public_store_exports():
    import dendrux.store as public

    for name in (
        "StateStore",
        "StoreCapabilities",
        "SQLStateStore",
        "PostgresStateStore",
        "SQLiteStateStore",
        "store_for_engine",
    ):
        assert name in public.__all__ and getattr(public, name) is not None


def test_unknown_outcome_has_own_public_code():
    from dendrux import _codes, errors

    assert durability.StoreOutcomeUnknown().code == "store_outcome_unknown"
    assert not durability.StoreOutcomeUnknown().retryable
    assert "store_outcome_unknown" in _codes.REGISTRY
    error_type = getattr(errors, "StoreOutcomeUnknown", None)
    assert error_type is durability.StoreOutcomeUnknown


def test_deployment_warning_points_to_application_under_default_filters():
    import warnings

    from dendrux import Agent
    from dendrux.llm.mock import MockLLM
    from dendrux.runtime.state import StoreCapabilities

    with warnings.catch_warnings(record=True) as caught:
        warnings.resetwarnings()
        warnings.filterwarnings("default", category=DeprecationWarning, module=__name__)
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"dendrux\..*")
        Agent(
            prompt="p",
            provider=MockLLM([]),
            state_store=SimpleNamespace(capabilities=StoreCapabilities()),
        )
    assert len(caught) == 1
    assert caught[0].filename == __file__
