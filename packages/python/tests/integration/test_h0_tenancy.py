"""The RT adapters refuse foreign and explicitly untenanted access without writes."""

from contextlib import contextmanager

import pytest
from sqlalchemy import event, select

from dendrux.db.models import Base
from dendrux.errors import RunNotFoundError, TenantScopeRequiredError
from dendrux.runtime.context import (
    DelegationContext,
    reset_delegation_context,
    set_delegation_context,
)
from dendrux.types import RunEventType, RunStatus
from tests._fixtures.storage import pin_storage_format
from tests._helpers.paused_runs import _pause
from tests._helpers.runtime_paths import PATHS, ProbeProvider, make_agent, path_case

pytestmark = pytest.mark.parametrize("engine", ["sqlite"], indirect=True)
EXISTING_PATHS = [p for p in PATHS if p not in {"run", "run_stream"}]


async def database_snapshot(engine):
    """Capture every persisted table, including submissions, notes, traces and tool evidence."""
    async with engine.connect() as conn:
        return {
            table.name: (await conn.execute(select(table).order_by(*table.primary_key))).all()
            for table in Base.metadata.sorted_tables
        }


@contextmanager
def refuse_writes(engine):
    """Count SQLite row changes, including rolled-back writes and no-op lock probes."""
    connections = {}

    def record(conn, cursor, statement, parameters, context, executemany):
        raw = conn.connection.driver_connection
        connections.setdefault(id(raw), (raw, raw.total_changes))

    # cursor.rowcount is -1 for SQLite UPDATE ... RETURNING, even for zero rows.
    event.listen(engine.sync_engine, "before_cursor_execute", record)
    try:
        yield
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", record)
    assert all(raw.total_changes == before for raw, before in connections.values()), (
        "tenant refusal must precede every row write"
    )


@pytest.mark.parametrize(
    "path,storage_format",
    [(p, fmt) for p in EXISTING_PATHS for fmt in (1, 2) if p != "recovery" or fmt == 2],
)
@pytest.mark.parametrize("tenant", ["foreign", None])
async def test_foreign_tenant_refuses_rt_entry_before_any_write(
    store, engine, path, tenant, storage_format
):
    await pin_storage_format(engine, storage_format)
    provider = ProbeProvider()
    case = await path_case(store, engine, path, provider)
    before = await database_snapshot(engine)
    with refuse_writes(engine), pytest.raises(RunNotFoundError):
        await case.invoke(tenant=tenant)
    assert await database_snapshot(engine) == before
    assert provider.calls == 0 and case.events == []


@pytest.mark.parametrize("path", ["cancel_run", "request_pause", "submit_note", "get_result"])
@pytest.mark.parametrize("tenant", ["foreign", None])
@pytest.mark.parametrize("storage_format", [1, 2])
async def test_foreign_tenant_refuses_control_and_read_wrappers(
    store, engine, path, tenant, storage_format
):
    await pin_storage_format(engine, storage_format)
    paused = await _pause("client", store, "tenant")
    before = await database_snapshot(engine)
    calls = paused.provider.calls
    method = getattr(paused.agent, path)
    args = (paused.run_id, "note") if path == "submit_note" else (paused.run_id,)
    with refuse_writes(engine), pytest.raises(RunNotFoundError):
        await method(*args, tenant_id=tenant)
    assert await database_snapshot(engine) == before
    assert paused.provider.calls == calls


@pytest.mark.parametrize("path", ["run", "stream"])
@pytest.mark.parametrize("storage_format", [1, 2])
async def test_new_run_requires_explicit_scope_before_any_write(
    store, engine, path, storage_format
):
    await pin_storage_format(engine, storage_format)
    provider = ProbeProvider()
    agent = make_agent(store, provider, strict_tenancy=True)
    before = await database_snapshot(engine)
    with refuse_writes(engine), pytest.raises(TenantScopeRequiredError) as caught:
        if path == "run":
            await agent.run("go")
        else:
            agent.stream("go")
    assert caught.value.operation == path
    assert await database_snapshot(engine) == before
    assert provider.calls == 0


@pytest.mark.parametrize("path", ["run", "stream"])
@pytest.mark.parametrize("storage_format", [1, 2])
async def test_child_run_refuses_foreign_scope_before_any_write(
    store, engine, path, storage_format
):
    await pin_storage_format(engine, storage_format)
    provider = ProbeProvider()
    agent = make_agent(store, provider)
    before = await database_snapshot(engine)
    token = set_delegation_context(DelegationContext(run_id="parent", tenant_id="tenant"))
    try:
        with refuse_writes(engine):
            if path == "run":
                with pytest.raises(
                    ValueError, match="^Delegated run cannot use tenant_id="
                ) as caught:
                    await agent.run("go", tenant_id="foreign")
                assert type(caught.value) is ValueError
            else:
                async with agent.stream("go", tenant_id="foreign") as stream:
                    events = [event async for event in stream]
                assert [event.type for event in events] == [RunEventType.RUN_ERROR]
                assert stream.result.status == RunStatus.ERROR
                assert stream.result.meta["error_type"] == "ValueError"
                assert events[0].error.startswith("Delegated run cannot use tenant_id=")
    finally:
        reset_delegation_context(token)
    assert await database_snapshot(engine) == before
    assert provider.calls == 0
