"""Managed subagents (0.7, T2a): the one atomic bind operation on the SQL store.

``bind_managed_child`` binds one pending parent invocation to a fresh running child
in a single transaction: unique managed tool intent, child row (running, seeded,
with its managed header), parent v3 managed wait and a retired parent epoch. It
runs on SQLite and on Postgres (``DENDRUX_TEST_PG_URL``, a private ``*_test``
database); the contender tests use independent sessions and engines on Postgres.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from dendrux._lifecycle import StoredEventType
from dendrux.db.models import AgentRun, ReactTrace, RunEvent, ToolCallRecord
from dendrux.errors import (
    ManagedStateError,
    PersistenceError,
    RunCancelledError,
    RunFencedError,
    RunNotFoundError,
    StoreCapabilityError,
)
from dendrux.runtime.context import DelegationDepthExceededError
from dendrux.runtime.persistence import message_trace
from dendrux.runtime.state import (
    MANAGED_BINDING_KEY,
    MANAGED_HEADER_KEY,
    BindManagedChildRequest,
    EventWrite,
    LoopStatePatch,
    ManagedHeader,
)
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.types import (
    ManagedWait,
    Message,
    PauseState,
    Role,
    RunStatus,
    ToolCall,
    UsageStats,
    generate_ulid,
)

BRIEF = "SENTINEL-managed-brief-41c9"


# ---------------------------------------------------------------------------
# Helpers (also used by the store-kit conformance cases)
# ---------------------------------------------------------------------------


async def managed_parent(
    store: Any,
    *,
    tenant: str | None = "t1",
    header: bool = True,
    pending: bool = True,
    call_id: str | None = None,
) -> tuple[str, str]:
    """A running managed root at claim epoch 0 with one pending managed call."""
    root = generate_ulid()
    call_id = call_id or generate_ulid()
    await store.create_run(root, "lead", tenant_id=tenant, input_data={"input": "go"})
    state: dict[str, Any] = {
        "v": 1,
        "loop": "react",
        "phase": "dispatch",
        "submission": None,
        "pending": [
            {
                "call_id": call_id,
                "tool": "reviewer",
                "phase": "dispatchable",
                "order_index": 3,
                "target": "agent",
            }
        ]
        if pending
        else [],
    }
    if header:
        state[MANAGED_HEADER_KEY] = ManagedHeader(root_run_id=root, path=()).to_dict()
    await store.checkpoint_step(root, claim_epoch=0, loop_state=state, pii_mapping=None, trace=None)
    return root, call_id


def bind_request(
    parent: str,
    call_id: str,
    *,
    root: str | None = None,
    tenant: str | None = "t1",
    epoch: int = 0,
    child_id: str | None = None,
    path: tuple[str, ...] = ("reviewer",),
    **overrides: Any,
) -> BindManagedChildRequest:
    child_id = child_id or generate_ulid()
    wait = PauseState(
        agent_name="lead",
        pending_tool_calls=[ToolCall(name="reviewer", params={"task": BRIEF}, id=call_id)],
        history=[],
        steps=[],
        iteration=1,
        trace_order_offset=4,
        usage=UsageStats(input_tokens=3, output_tokens=2, total_tokens=5),
        pending_targets={call_id: "agent"},
        pause_reason="managed_wait",
        pause_id=generate_ulid(),
        schema_version=3,
        loop_state_ref=True,
        provider_kwargs={"model": "m"},
        managed_wait=ManagedWait(tool_call_id=call_id, child_run_id=child_id),
    )
    fields: dict[str, Any] = dict(
        tenant_id=tenant,
        root_run_id=root or parent,
        parent_run_id=parent,
        parent_claim_epoch=epoch,
        tool_call_id=call_id,
        row_id=generate_ulid(),
        provider_tool_call_id="toolu_1",
        tool_name="reviewer",
        params={"task": BRIEF},
        iteration_index=1,
        child_run_id=child_id,
        child_agent_name="reviewer",
        child_path=path,
        child_input_data={"input": BRIEF},
        child_loop_state={"v": 1, "loop": "react"},
        parent_wait=wait,
        parent_patch=LoopStatePatch(set_fields={"iteration": 1}),
        intent_event=EventWrite(
            generate_ulid(),
            StoredEventType.TOOL_STARTED.value,
            1,
            {"tool_name": "reviewer", "call_id": call_id, "idempotent": False},
            call_id,
            None,
        ),
        child_seed_traces=(message_trace(Message(role=Role.USER, content=BRIEF), 0, kind="seed"),),
        child_model="child-model",
        child_strategy="NativeToolCalling",
    )
    fields.update(overrides)
    return BindManagedChildRequest(**fields)


def _row(record: Any) -> SimpleNamespace:
    return SimpleNamespace(**{c.name: getattr(record, c.name) for c in record.__table__.columns})


async def snapshot(engine: Any, parent: str) -> dict[str, Any]:
    """Every row the bind can write, as plain values, read with an independent session."""
    async with AsyncSession(engine) as session:
        run = (await session.execute(select(AgentRun).where(AgentRun.id == parent))).scalar_one()
        children = (
            (
                await session.execute(
                    select(AgentRun).where(AgentRun.parent_run_id == parent).order_by(AgentRun.id)
                )
            )
            .scalars()
            .all()
        )
        tools = (
            (
                await session.execute(
                    select(ToolCallRecord)
                    .where(ToolCallRecord.agent_run_id == parent)
                    .order_by(ToolCallRecord.id)
                )
            )
            .scalars()
            .all()
        )
        child_ids = [c.id for c in children]
        traces = await session.scalar(
            select(func.count())
            .select_from(ReactTrace)
            .where(ReactTrace.agent_run_id.in_(child_ids or [""]))
        )
        events = (
            (
                await session.execute(
                    select(RunEvent.event_type).where(RunEvent.agent_run_id == parent)
                )
            )
            .scalars()
            .all()
        )
        return {
            "status": run.status,
            "epoch": run.claim_epoch,
            "pause_data": run.pause_data,
            "loop_state": run.loop_state,
            "updated_at": run.updated_at,
            "children": [_row(c) for c in children],
            "tools": [_row(t) for t in tools],
            "child_traces": traces,
            "events": list(events),
        }


# ---------------------------------------------------------------------------
# Spec test 2: atomic before any child effect
# ---------------------------------------------------------------------------


async def test_bind_is_atomic_before_child_effect(store, engine):
    parent, call_id = await managed_parent(store)
    before = await snapshot(engine, parent)
    request = bind_request(parent, call_id)

    result = await store.bind_managed_child(request)

    assert result.outcome == "created" and result.child_run_id == request.child_run_id
    assert result.parent_wait_id == request.parent_wait.pause_id
    claim = result.child_claim
    assert claim is not None and claim.claim_epoch == 0 and claim.storage_format == 2
    assert claim.submission is None
    header = ManagedHeader.from_dict(
        claim.loop_state[MANAGED_HEADER_KEY], run_id=result.child_run_id
    )
    assert header == ManagedHeader(
        parent, ("reviewer",), parent_run_id=parent, tool_call_id=call_id
    )

    after = await snapshot(engine, parent)
    # Parent: v3 managed wait, retired epoch, binding kept beside its pending entry.
    assert after["status"] == RunStatus.PAUSED.value
    assert after["epoch"] == before["epoch"] + 1
    wait = PauseState.from_managed_dict(after["pause_data"])
    assert wait.managed_wait == ManagedWait(call_id, request.child_run_id)
    assert wait.pause_id == request.parent_wait.pause_id
    assert after["loop_state"][MANAGED_BINDING_KEY] == {
        "v": 1,
        "tool_call_id": call_id,
        "child_run_id": request.child_run_id,
        "pause_id": request.parent_wait.pause_id,
    }
    assert [p["call_id"] for p in after["loop_state"]["pending"]] == [call_id]
    assert after["loop_state"]["iteration"] == 1
    assert after["loop_state"][MANAGED_HEADER_KEY] == before["loop_state"][MANAGED_HEADER_KEY]
    # Intent: the unique managed tool row, started, with the binding.
    (tool,) = after["tools"]
    assert (tool.tool_call_id, tool.target, tool.outcome) == (call_id, "agent", "started")
    managed = tool.meta["managed"]
    assert set(tool.meta) == {"managed"} and set(managed) == {"v", "child_run_id", "bind"}
    assert (managed["v"], managed["child_run_id"]) == (1, request.child_run_id)
    assert isinstance(managed["bind"], str) and len(managed["bind"]) == 26
    assert tool.params == {"task": BRIEF} and tool.id == request.row_id
    assert after["events"] == [StoredEventType.TOOL_STARTED.value]
    # Child: running, unowned beyond epoch 0, seeded, linked and tenant-pinned.
    (child,) = after["children"]
    assert child.id == request.child_run_id
    assert (child.status, child.claim_epoch, child.storage_format) == ("running", 0, 2)
    assert (child.tenant_id, child.delegation_level, child.agent_name) == ("t1", 1, "reviewer")
    assert child.input_data == {"input": BRIEF} and child.model == "child-model"
    assert child.loop_state == claim.loop_state
    assert child.pause_data is None
    assert after["child_traces"] == 1

    # The retired parent epoch cannot write evidence or checkpoints any more.
    with pytest.raises(RunFencedError):
        await store.save_trace(parent, "assistant", "late", order_index=9, claim_epoch=0)
    with pytest.raises(RunFencedError):
        await store.checkpoint_step(
            parent, claim_epoch=0, loop_state={"v": 1}, pii_mapping=None, trace=None
        )
    # Ordinary readers refuse the private wait; only the managed decoder reads it.
    with pytest.raises(PersistenceError):
        PauseState.from_dict(await store.get_pause_state(parent))


WRITE_STEPS = ("_bind_park_parent", "_bind_intent", "_bind_child", "_bind_seeds", "_bind_event")


@pytest.mark.parametrize("step", [*WRITE_STEPS, "commit"])
async def test_bind_failure_after_any_write_leaves_nothing(store, engine, monkeypatch, step):
    parent, call_id = await managed_parent(store)
    before = await snapshot(engine, parent)
    request = bind_request(parent, call_id)

    class InjectedError(RuntimeError):
        pass

    if step == "commit":

        async def fail_commit(session, **kwargs):
            await session.flush()
            raise InjectedError()

        monkeypatch.setattr(store, "_write_tx", fail_commit)
    else:
        original = getattr(store, step)

        async def fail_after(*args, **kwargs):
            await original(*args, **kwargs)
            raise InjectedError()

        monkeypatch.setattr(store, step, fail_after)

    with pytest.raises(InjectedError):
        await store.bind_managed_child(request)
    assert await snapshot(engine, parent) == before
    async with AsyncSession(engine) as session:
        assert await session.get(AgentRun, request.child_run_id) is None

    monkeypatch.undo()
    retried = await store.bind_managed_child(request)
    assert retried.outcome == "created" and retried.child_claim is not None
    after = await snapshot(engine, parent)
    assert len(after["children"]) == 1 and len(after["tools"]) == 1


# ---------------------------------------------------------------------------
# Spec test 3: duplicates and lost acknowledgements never grant a second owner
# ---------------------------------------------------------------------------


async def test_duplicate_bind_never_grants_second_child_ownership(store, engine):
    parent, call_id = await managed_parent(store)
    first_request = bind_request(parent, call_id)
    first = await store.bind_managed_child(first_request)
    assert first.outcome == "created" and first.child_claim is not None

    # The same request again: read-back of the one binding, no execution token.
    again = await store.bind_managed_child(first_request)
    assert again.outcome == "existing" and again.child_claim is None
    assert again.child_run_id == first.child_run_id
    assert again.parent_wait_id == first.parent_wait_id

    # A different generated candidate for the same invocation cannot create another.
    other = bind_request(parent, call_id, child_id=generate_ulid())
    duplicate = await store.bind_managed_child(other)
    assert duplicate.outcome == "existing" and duplicate.child_claim is None
    assert duplicate.child_run_id == first.child_run_id
    # Even with the newer epoch the binding is reported, never recreated.
    newer = await store.bind_managed_child(bind_request(parent, call_id, epoch=1))
    assert newer.outcome == "existing" and newer.child_run_id == first.child_run_id

    state = await snapshot(engine, parent)
    assert [c.id for c in state["children"]] == [first.child_run_id]
    assert len(state["tools"]) == 1
    async with AsyncSession(engine) as session:
        assert await session.get(AgentRun, other.child_run_id) is None


def _lose_commit_ack(monkeypatch: pytest.MonkeyPatch, *, applied: bool) -> list[int]:
    """Make the first COMMIT raise a connection-style error, after or instead of applying."""
    calls: list[int] = []
    original = AsyncSession.commit

    async def commit(self: AsyncSession) -> None:
        calls.append(1)
        if len(calls) == 1:
            if applied:
                await original(self)
            raise OSError("connection lost during COMMIT")
        await original(self)

    monkeypatch.setattr(AsyncSession, "commit", commit)
    return calls


async def test_lost_ack_after_commit_proves_the_creator(store, engine, monkeypatch):
    parent, call_id = await managed_parent(store)
    request = bind_request(parent, call_id)
    calls = _lose_commit_ack(monkeypatch, applied=True)
    result = await store.bind_managed_child(request)
    monkeypatch.undo()
    assert calls == [1]
    # The readback proved this caller's own rows: it is the confirmed creator.
    assert result.outcome == "created" and result.child_claim is not None
    state = await snapshot(engine, parent)
    assert [c.id for c in state["children"]] == [request.child_run_id]
    # Any later repeat reconnects without ownership.
    repeat = await store.bind_managed_child(request)
    assert repeat.outcome == "existing" and repeat.child_claim is None


async def test_lost_ack_before_commit_retries_once(store, engine, monkeypatch):
    parent, call_id = await managed_parent(store)
    request = bind_request(parent, call_id)
    calls = _lose_commit_ack(monkeypatch, applied=False)
    result = await store.bind_managed_child(request)
    monkeypatch.undo()
    # Readback proved nothing was applied (parent unchanged, child absent): retry once.
    assert len(calls) == 2
    assert result.outcome == "created" and result.child_claim is not None
    state = await snapshot(engine, parent)
    assert [c.id for c in state["children"]] == [request.child_run_id]
    assert len(state["tools"]) == 1 and state["epoch"] == 1


async def test_unprovable_commit_is_unknown_and_grants_nothing(store, engine, monkeypatch):
    from dendrux.errors import StoreOutcomeUnknown
    from dendrux.runtime.stores.sql import _commit

    parent, call_id = await managed_parent(store)
    request = bind_request(parent, call_id)
    _lose_commit_ack(monkeypatch, applied=True)

    async def readback_fails(self, table, predicate, expected):
        raise OSError("readback lost too")

    monkeypatch.setattr(_commit._CommitReadback, "_check_proof", readback_fails)
    with pytest.raises(StoreOutcomeUnknown):
        await store.bind_managed_child(request)
    monkeypatch.undo()
    # The bind did commit; the uncertain caller learns it only as a read-back.
    repeat = await store.bind_managed_child(request)
    assert repeat.outcome == "existing" and repeat.child_claim is None
    assert repeat.child_run_id == request.child_run_id


# ---------------------------------------------------------------------------
# Spec test 4: refusals, invalid state, and Postgres contenders
# ---------------------------------------------------------------------------


async def _assert_unchanged(engine: Any, parent: str, before: dict[str, Any]) -> None:
    assert await snapshot(engine, parent) == before


async def test_tenant_mismatch_is_not_found(store, engine):
    parent, call_id = await managed_parent(store, tenant="t1")
    before = await snapshot(engine, parent)
    for tenant in ("t2", None):
        with pytest.raises(RunNotFoundError):
            await store.bind_managed_child(bind_request(parent, call_id, tenant=tenant))
    with pytest.raises(RunNotFoundError):
        await store.bind_managed_child(bind_request(generate_ulid(), call_id))
    await _assert_unchanged(engine, parent, before)


async def test_untenanted_parent_binds_untenanted_child(store, engine):
    parent, call_id = await managed_parent(store, tenant=None)
    with pytest.raises(RunNotFoundError):
        await store.bind_managed_child(bind_request(parent, call_id, tenant="t1"))
    result = await store.bind_managed_child(bind_request(parent, call_id, tenant=None))
    (child,) = (await snapshot(engine, parent))["children"]
    assert result.outcome == "created" and child.tenant_id is None


async def test_stale_epoch_is_fenced(store, engine, caplog):
    parent, call_id = await managed_parent(store)
    before = await snapshot(engine, parent)
    with pytest.raises(RunFencedError):
        await store.bind_managed_child(bind_request(parent, call_id, epoch=5))
    await _assert_unchanged(engine, parent, before)
    assert any(getattr(r, "code", None) == "run_fenced" for r in caplog.records)


async def test_cancelled_or_terminal_parent_refuses(store, engine):
    parent, call_id = await managed_parent(store)
    await store.request_cancel(parent)
    before = await snapshot(engine, parent)
    with pytest.raises(RunCancelledError):
        await store.bind_managed_child(bind_request(parent, call_id))
    await _assert_unchanged(engine, parent, before)

    finished, call = await managed_parent(store)
    await store.finalize_run(finished, status=RunStatus.SUCCESS.value, answer="done")
    with pytest.raises(RunCancelledError):
        await store.bind_managed_child(bind_request(finished, call))


async def test_parent_already_paused_is_fenced(store, engine):
    parent, call_id = await managed_parent(store)
    paused = await store.try_pause_run(
        parent, status=RunStatus.PAUSED.value, pause_data={"agent_name": "lead"}, claim_epoch=0
    )
    assert paused
    with pytest.raises(RunFencedError):
        await store.bind_managed_child(bind_request(parent, call_id))


@pytest.mark.parametrize(
    "case,reason",
    [
        ("no_header", "header_invalid"),
        ("other_root", "header_invalid"),
        ("wrong_path", "header_invalid"),
        ("no_pending", "pending_missing"),
        ("outstanding", "binding_outstanding"),
        ("format_1", "storage_format"),
    ],
)
async def test_unmanaged_or_inconsistent_parent_is_invalid_state(store, engine, case, reason):
    parent, call_id = await managed_parent(
        store, header=case != "no_header", pending=case != "no_pending"
    )
    request = bind_request(parent, call_id)
    if case == "other_root":
        request = bind_request(parent, call_id, root=generate_ulid())
    elif case == "wrong_path":
        request = bind_request(parent, call_id, path=("a", "reviewer"))
    elif case in ("outstanding", "format_1"):
        async with AsyncSession(engine) as session:
            run = await session.get(AgentRun, parent)
            if case == "format_1":
                run.storage_format = 1
            else:
                run.loop_state = {
                    **run.loop_state,
                    MANAGED_BINDING_KEY: {"tool_call_id": "other", "child_run_id": generate_ulid()},
                }
            await session.commit()
    before = await snapshot(engine, parent)
    with pytest.raises(ManagedStateError) as refused:
        await store.bind_managed_child(request)
    assert refused.value.reason == reason and refused.value.code == "managed_state_invalid"
    assert BRIEF not in str(refused.value)
    await _assert_unchanged(engine, parent, before)


async def test_missing_or_mismatched_bound_child_is_never_recreated(store, engine):
    parent, call_id = await managed_parent(store)
    request = bind_request(parent, call_id)
    await store.bind_managed_child(request)

    async with AsyncSession(engine) as session:
        await session.execute(
            update(AgentRun)
            .where(AgentRun.id == request.child_run_id)
            .values(loop_state={"v": 1, "loop": "react"})
        )
        await session.commit()
    with pytest.raises(ManagedStateError) as mismatch:
        await store.bind_managed_child(request)
    assert mismatch.value.reason == "child_mismatch"

    async with AsyncSession(engine) as session:
        await session.execute(delete(AgentRun).where(AgentRun.id == request.child_run_id))
        await session.commit()
    for attempt in (request, bind_request(parent, call_id, child_id=generate_ulid())):
        with pytest.raises(ManagedStateError) as missing:
            await store.bind_managed_child(attempt)
        assert missing.value.reason == "child_missing"
    assert (await snapshot(engine, parent))["children"] == []


async def test_conflicting_saved_intent_is_invalid_state(store, engine):
    parent, call_id = await managed_parent(store)
    await store.start_tool_call(
        parent,
        claim_epoch=0,
        row_id=generate_ulid(),
        tool_call_id=call_id,
        provider_tool_call_id=None,
        tool_name="reviewer",
        target="server",
        params=None,
        iteration_index=1,
        idempotent=False,
        event=EventWrite(generate_ulid(), "tool.started", 1, None, call_id, None),
    )
    before = await snapshot(engine, parent)
    with pytest.raises(ManagedStateError) as refused:
        await store.bind_managed_child(bind_request(parent, call_id))
    assert refused.value.reason == "binding_conflict"
    await _assert_unchanged(engine, parent, before)


async def test_checkpoint_binding_without_intent_is_invalid_state(store, engine):
    parent, call_id = await managed_parent(store)
    async with AsyncSession(engine) as session:
        run = await session.get(AgentRun, parent)
        run.loop_state = {
            **run.loop_state,
            MANAGED_BINDING_KEY: {"tool_call_id": call_id, "child_run_id": generate_ulid()},
        }
        await session.commit()
    with pytest.raises(ManagedStateError) as refused:
        await store.bind_managed_child(bind_request(parent, call_id))
    assert refused.value.reason == "binding_missing"


async def test_depth_ceiling_is_checked_under_the_parent_lock(store, engine):
    parent, call_id = await managed_parent(store)
    async with AsyncSession(engine) as session:
        await session.execute(
            update(AgentRun).where(AgentRun.id == parent).values(delegation_level=2)
        )
        await session.commit()
    before = await snapshot(engine, parent)
    with pytest.raises(DelegationDepthExceededError):
        await store.bind_managed_child(bind_request(parent, call_id, max_delegation_depth=2))
    await _assert_unchanged(engine, parent, before)
    result = await store.bind_managed_child(bind_request(parent, call_id, max_delegation_depth=3))
    (child,) = (await snapshot(engine, parent))["children"]
    assert result.outcome == "created" and child.delegation_level == 3


async def test_nested_child_binds_its_own_child(store, engine):
    parent, call_id = await managed_parent(store)
    first = await store.bind_managed_child(bind_request(parent, call_id))
    child = first.child_run_id
    grand_call = generate_ulid()
    pending = {
        "call_id": grand_call,
        "tool": "auditor",
        "phase": "dispatchable",
        "order_index": 2,
        "target": "agent",
    }
    state = {**first.child_claim.loop_state, "pending": [pending]}
    await store.checkpoint_step(
        child, claim_epoch=0, loop_state=state, pii_mapping=None, trace=None
    )
    # The registered path must extend the child's own path.
    with pytest.raises(ManagedStateError) as wrong:
        await store.bind_managed_child(
            bind_request(child, grand_call, root=parent, path=("auditor",))
        )
    assert wrong.value.reason == "header_invalid"
    nested = bind_request(child, grand_call, root=parent, path=("reviewer", "auditor"))
    result = await store.bind_managed_child(nested)
    assert result.outcome == "created"
    header = ManagedHeader.from_dict(
        result.child_claim.loop_state[MANAGED_HEADER_KEY], run_id=result.child_run_id
    )
    assert header.root_run_id == parent and header.path == ("reviewer", "auditor")
    assert header.parent_run_id == child and header.tool_call_id == grand_call
    (grandchild,) = (await snapshot(engine, child))["children"]
    assert (grandchild.delegation_level, grandchild.tenant_id) == (2, "t1")


def test_sql_stores_declare_the_managed_capability(store):
    from dendrux.runtime._capabilities import validate_capabilities
    from dendrux.runtime.stores.sql.base import SQLStateStore

    assert SQLStateStore.capabilities.managed_subagents is True
    assert validate_capabilities(store).managed_subagents is True


async def test_store_without_declared_capability_refuses_bind(store, engine):
    from dataclasses import replace

    parent, call_id = await managed_parent(store)
    before = await snapshot(engine, parent)
    store.capabilities = replace(store.capabilities, managed_subagents=False)
    with pytest.raises(StoreCapabilityError):
        await store.bind_managed_child(bind_request(parent, call_id))
    await _assert_unchanged(engine, parent, before)


def _engine_url(engine: Any) -> str:
    return engine.url.render_as_string(hide_password=False)


async def test_bind_race_stale_epoch_cancel_and_tenant(store, engine):
    """N independent contenders for one parent: exactly one child and intent, no orphan.

    Half the contenders bind the same invocation with different candidate child ids,
    the rest bind other invocations of the same parent epoch. On Postgres they use two
    engines (separate pools stand in for separate processes) and real row locks.
    """
    contenders = 8
    engines = [engine]
    if engine.dialect.name == "postgresql":
        engines.append(create_async_engine(_engine_url(engine)))
    try:
        stores = [store_for_engine(e) for e in engines]
        for _ in range(3):
            parent, call_id = await managed_parent(store)
            requests = [
                bind_request(parent, call_id, child_id=generate_ulid()) for _ in range(contenders)
            ] + [bind_request(parent, generate_ulid()) for _ in range(contenders // 2)]
            results = await asyncio.gather(
                *(stores[i % len(stores)].bind_managed_child(r) for i, r in enumerate(requests)),
                return_exceptions=True,
            )
            await _assert_one_binding(engine, parent, requests, results)
    finally:
        for extra in engines[1:]:
            await extra.dispose()


async def _assert_one_binding(
    engine: Any, parent: str, requests: list[Any], results: list[Any]
) -> None:
    created = [
        (q, r)
        for q, r in zip(requests, results, strict=True)
        if not isinstance(r, BaseException) and r.outcome == "created"
    ]
    assert len(created) == 1, results
    won, result = created[0]
    for request, outcome in zip(requests, results, strict=True):
        if outcome is result:
            continue
        if isinstance(outcome, BaseException):
            # Another invocation found the parent parked with its epoch retired.
            assert isinstance(outcome, RunFencedError), outcome
            assert request.tool_call_id != won.tool_call_id
        else:
            # The same invocation reads the binding back, never with ownership.
            assert outcome.outcome == "existing" and outcome.child_claim is None
            assert outcome.child_run_id == won.child_run_id
            assert request.tool_call_id == won.tool_call_id
    state = await snapshot(engine, parent)
    assert [c.id for c in state["children"]] == [won.child_run_id]
    assert [(t.tool_call_id, t.target) for t in state["tools"]] == [(won.tool_call_id, "agent")]
    assert state["epoch"] == 1 and state["status"] == RunStatus.PAUSED.value
    async with AsyncSession(engine) as session:
        orphans = await session.scalar(
            select(func.count())
            .select_from(AgentRun)
            .where(AgentRun.id.in_([r.child_run_id for r in requests if r is not won]))
        )
    assert orphans == 0


async def test_bind_contends_with_cancel_and_stale_writer(store, engine):
    """Cancellation and a stale parent writer race the bind: one consistent outcome."""
    for _ in range(4):
        parent, call_id = await managed_parent(store)
        request = bind_request(parent, call_id)
        outcomes = await asyncio.gather(
            store.bind_managed_child(request),
            store.request_cancel(parent),
            store.save_trace(parent, "assistant", "late", order_index=7, claim_epoch=0),
            return_exceptions=True,
        )
        bound, _cancel, late = outcomes
        state = await snapshot(engine, parent)
        if isinstance(bound, BaseException):
            assert isinstance(bound, RunCancelledError)
            assert state["children"] == [] and state["tools"] == []
            assert state["status"] == RunStatus.RUNNING.value and state["epoch"] == 0
        else:
            assert bound.outcome == "created"
            assert [c.id for c in state["children"]] == [request.child_run_id]
            # A stale write either landed before the bind or was fenced after it.
            assert late is None or isinstance(late, (RunFencedError, str))


async def test_lost_ack_with_changed_rows_reads_back_without_a_token(store, engine, monkeypatch):
    """A lost acknowledgement whose readback cannot prove this invocation's own rows
    re-enters under the lock: the saved binding is read back, never with a token."""
    parent, call_id = await managed_parent(store)
    request = bind_request(parent, call_id)
    original = AsyncSession.commit
    calls: list[int] = []

    async def commit(self: AsyncSession) -> None:
        calls.append(1)
        if len(calls) == 1:
            await original(self)
            async with AsyncSession(engine) as other:
                await other.execute(
                    update(AgentRun).where(AgentRun.id == parent).values(claim_epoch=7)
                )
                await other.commit()
            raise OSError("connection lost during COMMIT")
        await original(self)

    monkeypatch.setattr(AsyncSession, "commit", commit)
    result = await store.bind_managed_child(request)
    monkeypatch.undo()
    assert result.outcome == "existing" and result.child_claim is None
    assert result.child_run_id == request.child_run_id
    state = await snapshot(engine, parent)
    assert [c.id for c in state["children"]] == [request.child_run_id]
    assert len(state["tools"]) == 1


async def test_generic_continuations_refuse_a_managed_wait_before_claiming(store, engine):
    """Until the managed driver exists (T2b), every public continuation of a parked
    parent refuses with the stable pause_format_unsupported code and claims nothing."""
    from dendrux import Agent
    from dendrux.llm.mock import MockLLM
    from dendrux.types import ToolResult

    parent, call_id = await managed_parent(store)
    request = bind_request(parent, call_id)
    bound = await store.bind_managed_child(request)
    agent = Agent(name="lead", prompt="p", provider=MockLLM([]), state_store=store)
    before = await snapshot(engine, parent)
    attempts = [
        agent.submit_resume(parent, tenant_id="t1", expected_pause_id=bound.parent_wait_id),
        agent.resume(parent, tenant_id="t1", expected_pause_id=bound.parent_wait_id),
        agent.submit_tool_results(
            parent,
            [ToolResult(name="reviewer", call_id=call_id, payload='"forged"')],
            tenant_id="t1",
            expected_pause_id=bound.parent_wait_id,
        ),
    ]
    for attempt in attempts:
        with pytest.raises(PersistenceError) as refused:
            await attempt
        assert refused.value.code == "pause_format_unsupported"
    await _assert_unchanged(engine, parent, before)


# ---------------------------------------------------------------------------
# Fix round 1
# ---------------------------------------------------------------------------


async def _two_pools(engine: Any, tmp_path: Any) -> tuple[Any, Any, list[Any]]:
    """Two independent engines on one database (two pools stand in for two processes)."""
    if engine.dialect.name == "postgresql":
        other = create_async_engine(_engine_url(engine))
        return engine, other, [other]
    from dendrux.db.models import Base

    url = f"sqlite+aiosqlite:///{tmp_path / 'two_pools.db'}"
    first = create_async_engine(url, connect_args={"check_same_thread": False})
    second = create_async_engine(url, connect_args={"check_same_thread": False})
    async with first.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return first, second, [first, second]


async def test_r1_competing_identical_request_during_lost_ack_gets_no_second_token(
    store, engine, monkeypatch, tmp_path
):
    """Caller A's COMMIT fails unapplied; before A's readback, caller B (another pool)
    commits the identical request. Only B may receive the child's execution token."""
    engine_a, engine_b, owned = await _two_pools(engine, tmp_path)
    try:
        store_a, store_b = store_for_engine(engine_a), store_for_engine(engine_b)
        parent, call_id = await managed_parent(store_a)
        request = bind_request(parent, call_id)
        original_commit = AsyncSession.commit
        commits: list[int] = []

        async def commit(self: AsyncSession) -> None:
            commits.append(1)
            if len(commits) == 1:
                raise OSError("connection lost before COMMIT applied")
            await original_commit(self)

        competitor: list[Any] = []
        original_prepare = store_a._prepare_commit_readback

        async def prepare(session: Any) -> None:
            if not competitor:
                competitor.append(await store_b.bind_managed_child(request))
            await original_prepare(session)

        monkeypatch.setattr(AsyncSession, "commit", commit)
        monkeypatch.setattr(store_a, "_prepare_commit_readback", prepare)
        result_a = await store_a.bind_managed_child(request)
        monkeypatch.undo()
        result_b = competitor[0]

        tokens = [r for r in (result_a, result_b) if r.child_claim is not None]
        assert len(tokens) == 1
        assert result_b.outcome == "created" and result_b.child_claim is not None
        assert result_a.outcome == "existing" and result_a.child_claim is None
        assert result_a.child_run_id == result_b.child_run_id == request.child_run_id
        state = await snapshot(engine_a, parent)
        assert len(state["children"]) == 1 and len(state["tools"]) == 1
    finally:
        for extra in owned:
            await extra.dispose()


async def test_r2_caller_mutation_during_bind_cannot_split_the_binding(store, engine, monkeypatch):
    """The bind works from a detached snapshot taken before its first await."""
    parent, call_id = await managed_parent(store)
    request = bind_request(parent, call_id)
    original_child, original_wait = request.child_run_id, request.parent_wait.pause_id
    entered, release = asyncio.Event(), asyncio.Event()
    original = store._bind_parent

    async def gated(*args: Any, **kwargs: Any) -> Any:
        row = await original(*args, **kwargs)
        entered.set()
        await release.wait()
        return row

    monkeypatch.setattr(store, "_bind_parent", gated)
    task = asyncio.create_task(store.bind_managed_child(request))
    await entered.wait()
    request.parent_wait.managed_wait = ManagedWait(call_id, generate_ulid())
    request.parent_wait.pause_id = generate_ulid()
    request.parent_patch.set_fields[MANAGED_BINDING_KEY] = {"tool_call_id": "x"}
    request.child_loop_state[MANAGED_HEADER_KEY] = {"forged": True}
    request.params["task"] = "forged"
    request.child_seed_traces[0].meta["forged"] = True
    release.set()
    result = await task
    monkeypatch.undo()

    assert result.outcome == "created" and result.child_run_id == original_child
    assert result.parent_wait_id == original_wait
    state = await snapshot(engine, parent)
    wait = PauseState.from_managed_dict(state["pause_data"])
    assert wait.managed_wait == ManagedWait(call_id, original_child)
    assert wait.pause_id == original_wait
    assert state["loop_state"][MANAGED_BINDING_KEY] == {
        "v": 1,
        "tool_call_id": call_id,
        "child_run_id": original_child,
        "pause_id": original_wait,
    }
    (tool,) = state["tools"]
    assert tool.params == {"task": BRIEF}
    (child,) = state["children"]
    assert child.loop_state[MANAGED_HEADER_KEY]["invocation"]["tool_call_id"] == call_id
    async with AsyncSession(engine) as session:
        seeds = (
            (await session.execute(select(ReactTrace).where(ReactTrace.agent_run_id == child.id)))
            .scalars()
            .all()
        )
    assert all("forged" not in (trace.meta or {}) for trace in seeds)


async def _edit_run(engine: Any, run_id: str, **changes: Any) -> None:
    async with AsyncSession(engine) as session:
        run = await session.get(AgentRun, run_id)
        for key, change in changes.items():
            setattr(run, key, change(getattr(run, key)))
        await session.commit()


@pytest.mark.parametrize("case", ["checkpoint_child", "generation", "pause_binding"])
async def test_r3_active_binding_disagreement_is_invalid_state(store, engine, case):
    parent, call_id = await managed_parent(store)
    request = bind_request(parent, call_id)
    await store.bind_managed_child(request)
    other = generate_ulid()
    if case == "checkpoint_child":
        await _edit_run(
            engine,
            parent,
            loop_state=lambda s: {
                **s,
                MANAGED_BINDING_KEY: {**s[MANAGED_BINDING_KEY], "child_run_id": other},
            },
        )
    elif case == "generation":
        await _edit_run(engine, parent, pause_data=lambda p: {**p, "pause_id": other})
    else:
        await _edit_run(
            engine,
            parent,
            pause_data=lambda p: {
                **p,
                "managed_wait": {**p["managed_wait"], "child_run_id": other},
            },
        )
    with pytest.raises(ManagedStateError) as refused:
        await store.bind_managed_child(request)
    assert refused.value.reason == "binding_conflict"


async def test_r3_wrong_registered_path_is_child_mismatch(store, engine):
    parent, call_id = await managed_parent(store)
    request = bind_request(parent, call_id)
    await store.bind_managed_child(request)
    # A repeat naming another registration path is not this binding.
    with pytest.raises(ManagedStateError) as other_path:
        await store.bind_managed_child(bind_request(parent, call_id, path=("auditor",)))
    assert other_path.value.reason == "child_mismatch"
    # A stored child header with another path (same root and invocation) is refused.
    await _edit_run(
        engine,
        request.child_run_id,
        loop_state=lambda s: {
            **s,
            MANAGED_HEADER_KEY: {**s[MANAGED_HEADER_KEY], "path": ["auditor"]},
        },
    )
    with pytest.raises(ManagedStateError) as stored_path:
        await store.bind_managed_child(request)
    assert stored_path.value.reason == "child_mismatch"


async def test_r3_foreign_child_is_refused_before_its_header_is_read(store, engine, caplog):
    parent, call_id = await managed_parent(store)
    request = bind_request(parent, call_id)
    await store.bind_managed_child(request)
    await _edit_run(
        engine,
        request.child_run_id,
        tenant_id=lambda _t: "t2",
        loop_state=lambda s: {**s, MANAGED_HEADER_KEY: {"garbage": BRIEF}},
    )
    caplog.clear()
    with pytest.raises(ManagedStateError) as refused:
        await store.bind_managed_child(request)
    assert refused.value.reason == "child_mismatch"
    reasons = [getattr(r, "managed_reason", None) for r in caplog.records]
    assert "header_invalid" not in reasons and BRIEF not in caplog.text


async def _adopt_first_and_queue(store: Any, parent: str, first_call: str, child: str) -> str:
    """Complete a bound invocation through real store transitions (what T2b adoption does):
    the child finishes, the parent is claimed, the started intent is completed with its
    observation while the outgoing binding is cleared, and a new managed call is queued."""
    from dendrux.runtime.state import TraceWrite

    assert await store.finalize_run(child, status=RunStatus.SUCCESS.value, answer="done")
    claim = await store.claim_paused_run(parent, expected_status=RunStatus.PAUSED.value)
    assert claim and claim.claim_epoch == 2
    await store.complete_tool_call(
        parent,
        claim_epoch=2,
        tool_call_id=first_call,
        outcome="completed",
        result_payload='"done"',
        success=True,
        duration_ms=1,
        error_message=None,
        replayed=False,
        observation=TraceWrite(
            generate_ulid(), "tool", '"done"', 5, meta={"tool_name": "reviewer"}
        ),
        event=EventWrite(
            generate_ulid(), StoredEventType.TOOL_COMPLETED.value, 1, None, first_call, None
        ),
        loop_state_patch=LoopStatePatch(
            remove_pending=(first_call,), set_fields={MANAGED_BINDING_KEY: None}
        ),
    )
    second_call = generate_ulid()
    state = await store.get_loop_state(parent)
    pending = {
        "call_id": second_call,
        "tool": "reviewer",
        "phase": "dispatchable",
        "order_index": 9,
        "target": "agent",
    }
    await store.checkpoint_step(
        parent,
        claim_epoch=2,
        loop_state={**state, "pending": [pending]},
        pii_mapping=None,
        trace=None,
    )
    return second_call


async def test_r3_completed_invocation_still_reads_back(store, engine):
    """A genuinely completed older invocation (intent completed, child terminal, binding
    cleared by adoption) reads back without a token, also while a newer call is active."""
    parent, first_call = await managed_parent(store)
    first = await store.bind_managed_child(bind_request(parent, first_call))
    second_call = await _adopt_first_and_queue(store, parent, first_call, first.child_run_id)
    (intent,) = (await snapshot(engine, parent))["tools"]
    assert intent.outcome == "completed"

    old = await store.bind_managed_child(bind_request(parent, first_call, epoch=2))
    assert old.outcome == "existing" and old.child_claim is None
    assert old.child_run_id == first.child_run_id and old.parent_wait_id is None
    new = await store.bind_managed_child(bind_request(parent, second_call, epoch=2))
    assert new.outcome == "created" and new.child_claim is not None
    # The older invocation stays readable while the new one is active.
    again = await store.bind_managed_child(bind_request(parent, first_call, epoch=3))
    assert again.outcome == "existing" and again.child_run_id == first.child_run_id
    assert again.child_claim is None and again.parent_wait_id is None


# ---------------------------------------------------------------------------
# Fix round 2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case,reason",
    [("other_call", "binding_conflict"), ("cleared", "binding_missing")],
)
async def test_v1_active_invocation_with_a_rewritten_checkpoint_is_invalid(
    store, engine, case, reason
):
    """Pending call, v3 pause and started intent still say this call is active: a
    checkpoint naming another call, or none, is contradictory, not 'historical'."""
    parent, call_id = await managed_parent(store)
    request = bind_request(parent, call_id)
    await store.bind_managed_child(request)
    if case == "other_call":
        change = {"tool_call_id": generate_ulid()}
        await _edit_run(
            engine,
            parent,
            loop_state=lambda s: {
                **s,
                MANAGED_BINDING_KEY: {**s[MANAGED_BINDING_KEY], **change},
            },
        )
    else:
        await _edit_run(engine, parent, loop_state=lambda s: {**s, MANAGED_BINDING_KEY: None})
    with pytest.raises(ManagedStateError) as refused:
        await store.bind_managed_child(request)
    assert refused.value.reason == reason


async def test_v1_active_invocation_without_its_pending_call_is_invalid(store, engine):
    parent, call_id = await managed_parent(store)
    request = bind_request(parent, call_id)
    await store.bind_managed_child(request)
    await _edit_run(engine, parent, loop_state=lambda s: {**s, "pending": []})
    with pytest.raises(ManagedStateError) as refused:
        await store.bind_managed_child(request)
    assert refused.value.reason == "pending_missing"


async def test_v1_cancelled_parent_reads_the_binding_back(store, engine):
    """Finalizing the paused parent clears its pause; the permanent binding is not
    corrupt: a repeat reads it back without a token."""
    parent, call_id = await managed_parent(store)
    request = bind_request(parent, call_id)
    created = await store.bind_managed_child(request)
    assert await store.finalize_run(
        parent,
        status=RunStatus.CANCELLED.value,
        expected_current_status=RunStatus.PAUSED.value,
        claim_epoch=1,
    )
    again = await store.bind_managed_child(request)
    assert again.outcome == "existing" and again.child_claim is None
    assert again.child_run_id == created.child_run_id and again.parent_wait_id is None


async def test_v1_completed_intent_still_named_by_the_checkpoint_is_invalid(store, engine):
    parent, call_id = await managed_parent(store)
    request = bind_request(parent, call_id)
    await store.bind_managed_child(request)
    async with AsyncSession(engine) as session:
        await session.execute(
            update(ToolCallRecord)
            .where(ToolCallRecord.agent_run_id == parent)
            .values(outcome="completed")
        )
        await session.commit()
    with pytest.raises(ManagedStateError) as refused:
        await store.bind_managed_child(request)
    assert refused.value.reason == "binding_conflict"


@pytest.mark.parametrize(
    "envelope",
    [BRIEF, 42, '{"v": 1, "tool_call_id": "x", "child_run_id": "y"} ' + BRIEF],
    ids=["text", "number", "encoded_object"],
)
async def test_v2_malformed_pause_envelope_is_a_safe_refusal(store, engine, caplog, envelope):
    parent, call_id = await managed_parent(store)
    request = bind_request(parent, call_id)
    await store.bind_managed_child(request)
    await _edit_run(engine, parent, pause_data=lambda p: {**p, "managed_wait": envelope})
    caplog.clear()
    with pytest.raises(ManagedStateError) as refused:
        await store.bind_managed_child(request)
    error = refused.value
    assert error.reason == "binding_conflict"
    assert error.__cause__ is None and error.__context__ is None
    assert BRIEF not in repr(error.__dict__) and BRIEF not in str(error)
    assert BRIEF not in caplog.text
