"""Default SQL hooks preserve inserts and tenant-scoped row locking."""

from unittest.mock import AsyncMock

import pytest
from sqlalchemy.dialects import postgresql

from dendrux.db.models import AgentRun
from dendrux.runtime.state import UNSCOPED
from dendrux.runtime.stores.sql._types import RunLockScope
from dendrux.runtime.stores.sql.base import SQLStateStore


def test_base_insert_keeps_target_and_values():
    store = object.__new__(SQLStateStore)
    stmt = store._insert(AgentRun.__table__).values(id="run", agent_name="agent")
    compiled = stmt.compile(dialect=postgresql.dialect())
    assert stmt.table is AgentRun.__table__
    assert str(compiled).startswith("INSERT INTO agent_runs ")
    assert compiled.params["id"] == "run"
    assert compiled.params["agent_name"] == "agent"


@pytest.mark.parametrize("tenant", [UNSCOPED, None, "tenant"])
async def test_base_serialization_locks_only_owner_scope(tenant):
    store = object.__new__(SQLStateStore)
    session = AsyncMock()
    await store.serialize_in_tx(session, owner=RunLockScope("run", tenant))
    session.execute.assert_awaited_once()
    stmt = session.execute.await_args.args[0]
    compiled = stmt.compile(dialect=postgresql.dialect())
    sql = str(compiled)
    assert sql.endswith(" FOR UPDATE")
    assert "agent_runs.id = " in sql
    assert compiled.params["id_1"] == "run"
    if tenant is UNSCOPED:
        assert "agent_runs.tenant_id" not in sql
    elif tenant is None:
        assert "agent_runs.tenant_id IS NULL" in sql
    else:
        assert "agent_runs.tenant_id = " in sql
        assert compiled.params["tenant_id_1"] == tenant
    session.commit.assert_not_awaited()
