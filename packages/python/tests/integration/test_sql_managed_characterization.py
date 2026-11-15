"""Managed binding preserves lock order, one transaction, replay and event rollback."""

import json
from contextlib import contextmanager
from unittest.mock import Mock

import pytest
from sqlalchemy import event

import dendrux.runtime.state as state
from dendrux.errors import ManagedStateError
from dendrux.runtime.stores.sql import _sql_managed
from dendrux.types import generate_ulid
from tests.integration.test_managed_bind import bind_request, managed_parent, snapshot


@contextmanager
def _transaction_trace(engine):
    statements, boundaries, connections = [], [], set()

    def statement(conn, cursor, sql, parameters, context, many):
        connections.add(id(conn))
        statements.append(" ".join(sql.split()))

    def boundary(kind):
        def observe(conn):
            connections.add(id(conn))
            boundaries.append(kind)

        return observe

    listeners = [("before_cursor_execute", statement)] + [
        (kind, boundary(kind)) for kind in ("begin", "commit", "rollback")
    ]
    for kind, listener in listeners:
        event.listen(engine.sync_engine, kind, listener)
    try:
        yield statements, boundaries, connections
    finally:
        for kind, listener in listeners:
            event.remove(engine.sync_engine, kind, listener)


async def _nested_request(store):
    root, call = await managed_parent(store)
    first = await store.bind_managed_child(bind_request(root, call))
    parent, call = first.child_run_id, generate_ulid()
    loop = dict(first.child_claim.loop_state)
    loop.update(phase="dispatch", pending=[{"call_id": call, "target": "agent"}])
    await store.checkpoint_step(
        parent, claim_epoch=0, loop_state=loop, pii_mapping=None, trace=None
    )
    return bind_request(parent, call, root=root, path=("reviewer", "reviewer"))


@pytest.mark.parametrize("replay", [False, True], ids=["created", "existing"])
async def test_bind_locks_root_before_parent_and_keeps_one_transaction(
    store, engine, monkeypatch, record_property, replay
):
    request = await _nested_request(store)
    if replay:
        await store.bind_managed_child(request)
    phases, sessions, locks = [], [], []

    def observe(name):
        original = getattr(store, name)

        async def call(session, *args, **kwargs):
            phases.append(name)
            sessions.append(session)
            return await original(session, *args, **kwargs)

        monkeypatch.setattr(store, name, call)

    for name in (
        "_lock_managed_root",
        "_bind_parent",
        "_bind_existing",
        "_admit_managed_limit",
        "_bind_park_parent",
        "_bind_intent",
        "_bind_child",
        "_bind_seeds",
        "_bind_event",
        "_write_tx",
    ):
        observe(name)
    lock = store.lock_clause

    def lock_clause(stmt, intent):
        locks.append((intent, list(stmt.compile().params.values())))
        return lock(stmt, intent)

    monkeypatch.setattr(store, "lock_clause", lock_clause)
    with _transaction_trace(engine) as (statements, boundaries, connections):
        result = await store.bind_managed_child(request)
    assert result.outcome == ("existing" if replay else "created")
    assert (result.child_claim is None) is replay
    assert all(session is sessions[0] for session in sessions)
    assert len(connections) == 1
    assert boundaries == ["begin", "rollback" if replay else "commit"]
    assert phases == (
        ["_lock_managed_root", "_bind_parent", "_bind_existing"]
        if replay
        else [
            "_lock_managed_root",
            "_bind_parent",
            "_bind_existing",
            "_admit_managed_limit",
            "_bind_park_parent",
            "_bind_intent",
            "_bind_child",
            "_bind_seeds",
            "_bind_event",
            "_write_tx",
        ]
    )
    assert locks[0][0] == locks[1][0] == "update"
    assert request.root_run_id in locks[0][1]
    assert request.parent_run_id in locks[1][1]
    assert request.root_run_id != request.parent_run_id
    writes = [
        (sql.split()[0], sql.split()[2 if sql.startswith("INSERT") else 1])
        for sql in statements
        if sql.startswith(("INSERT", "UPDATE"))
    ]
    reservation = [("UPDATE", "agent_runs")] if engine.dialect.name == "sqlite" else []
    assert len(statements) == (5 if replay else 13) - (engine.dialect.name != "sqlite")
    selects = [sql for sql in statements if sql.startswith("SELECT")]
    assert selects[0].startswith("SELECT agent_runs.id FROM agent_runs WHERE")
    assert selects[1].startswith("SELECT agent_runs.tenant_id, agent_runs.status,")
    assert ["FOR UPDATE" in sql for sql in selects[:2]] == [
        engine.dialect.name == "postgresql",
        engine.dialect.name == "postgresql",
    ]
    assert writes == reservation + (
        []
        if replay
        else [
            ("UPDATE", "agent_runs"),
            ("INSERT", "tool_calls"),
            ("INSERT", "agent_runs"),
            ("INSERT", "react_traces"),
            ("INSERT", "run_events"),
        ]
    )
    record_property("sql", json.dumps(statements))
    record_property("boundaries", json.dumps(boundaries))


@pytest.mark.parametrize("fault", ["seed", "event"])
async def test_bind_fault_after_seed_or_event_rolls_back_all_rows(
    store, engine, monkeypatch, record_property, fault
):
    parent, call = await managed_parent(store)
    request = bind_request(parent, call)
    before = await snapshot(engine, parent)
    calls = []
    original = store._insert_trace if fault == "seed" else state._write_event

    async def fail_after(session, run_id, value):
        await original(session, run_id, value)
        calls.append(run_id)
        raise ValueError("forced managed boundary")

    with monkeypatch.context() as patch:
        if fault == "seed":
            patch.setattr(store, "_insert_trace", fail_after)
        else:
            patch.setattr(state, "_write_event", fail_after)
        with (
            _transaction_trace(engine) as (statements, boundaries, connections),
            pytest.raises(ValueError, match="forced managed boundary"),
        ):
            await store.bind_managed_child(request)
    assert calls == [request.child_run_id if fault == "seed" else parent]
    assert boundaries == ["begin", "rollback"]
    assert len(statements) == (10 if fault == "seed" else 11) - (engine.dialect.name != "sqlite")
    assert len(connections) == 1
    assert any(sql.startswith("INSERT INTO react_traces") for sql in statements)
    assert any(sql.startswith("INSERT INTO run_events") for sql in statements) is (fault == "event")
    assert await snapshot(engine, parent) == before
    assert await store.get_run(request.child_run_id) is None
    record_property("sql", json.dumps(statements))
    record_property("boundaries", json.dumps(boundaries))
    retried = await store.bind_managed_child(request)
    assert retried.outcome == "created" and retried.child_claim is not None


async def test_replay_consults_the_live_binding_helper_and_grants_no_new_claim(store, monkeypatch):
    parent, call = await managed_parent(store)
    request = bind_request(parent, call)
    await store.bind_managed_child(request)
    helper = Mock(wraps=_sql_managed._bound_child_id)
    with monkeypatch.context() as patch:
        patch.setattr(_sql_managed, "_bound_child_id", helper)
        result = await store.bind_managed_child(request)
    helper.assert_called_once()
    assert len(helper.call_args.args[0]) == 1
    assert result.outcome == "existing" and result.child_claim is None
    conflict = Mock(return_value=None)
    with monkeypatch.context() as patch:
        patch.setattr(_sql_managed, "_bound_child_id", conflict)
        with pytest.raises(ManagedStateError) as caught:
            await store.bind_managed_child(request)
    conflict.assert_called_once()
    assert caught.value.reason == "binding_conflict"
