"""A refused controller sees only the workflow its tenant admitted."""

from copy import copy

import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.db.models import AgentRun
from dendrux.errors import GovernanceMismatchError, RunNotFoundError, RunNotReclaimableError
from dendrux.runtime import _managed, _managed_pending, _pending, _run_resume_stream
from dendrux.runtime.state import UNSCOPED
from dendrux.types import Limits, RunEventType, RunStatus
from tests._helpers.scripted_provider import tool_calls
from tests.integration.test_continuation_governance import with_skill
from tests.integration.test_fanout_routing import approver
from tests.integration.test_h0_tenancy import database_snapshot
from tests.integration.test_managed_routing import TENANT, ContextLLM, build, done, lead


def workflow(store, shape):
    if shape == "chain":
        return build(store, "approval")
    children = [approver("left"), approver("right")]
    return lead(
        store,
        ContextLLM(
            [tool_calls(*[(c.name, {"task": "go"}, c.name) for c in children]), done("joined")]
        ),
        *children,
    )


async def change_owner(engine, run_id):
    writer = create_async_engine(engine.url)
    try:
        async with writer.begin() as conn:
            await conn.execute(
                update(AgentRun)
                .where(AgentRun.id == run_id)
                .values(tenant_id="other", output_data={"answer": "foreign-refusal-answer"})
            )
    finally:
        await writer.dispose()


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
@pytest.mark.parametrize(
    "shape,tenant,scope",
    [("chain", TENANT, TENANT), ("chain", None, None), ("fanout", TENANT, UNSCOPED)],
)
@pytest.mark.parametrize("policy", ["governance", "skills"])
@pytest.mark.parametrize("replace_root", [False, True], ids=["same-owner", "changed-owner"])
async def test_refusal_uses_admitted_tenant(
    store, engine, monkeypatch, shape, tenant, scope, policy, replace_root
):
    root = workflow(store, shape)
    paused = await root.run("go", tenant_id=tenant)
    controller = copy(root)
    if policy == "governance":
        controller._limits = Limits(max_parallel_subagents=1)
    else:
        controller = with_skill(root, "added")
    refused = False
    after_change = None
    reads = []

    def observe(method, name):
        async def read(*args, tenant_id=UNSCOPED, **kwargs):
            if refused:
                reads.append((name, tenant_id))
            return await method(*args, tenant_id=tenant_id, **kwargs)

        return read

    for name in ("get_run", "get_pii_mapping", "get_pending_pause_summary", "get_managed_children"):
        monkeypatch.setattr(store, name, observe(getattr(store, name), name))
    read_hop = _pending._HopReader.read

    async def observed_hop(reader, run_id, tenant_id=UNSCOPED):
        if refused:
            reads.append(("hop", tenant_id))
        return await read_hop(reader, run_id, tenant_id)

    monkeypatch.setattr(_pending._HopReader, "read", observed_hop)
    original = _managed.route

    async def at_refusal(*args, **kwargs):
        nonlocal refused, after_change
        try:
            return await original(*args, **kwargs)
        except (GovernanceMismatchError, RunNotReclaimableError):
            refused = True
            if replace_root:
                await change_owner(engine, paused.run_id)
                after_change = await database_snapshot(engine)
            raise

    monkeypatch.setattr(_managed, "route", at_refusal)
    before = await database_snapshot(engine)
    async with controller.resume_stream(
        paused.run_id, expected_pause_id=paused.pause_id, tenant_id=scope
    ) as stream:
        if replace_root:
            with pytest.raises(RunNotFoundError) as missing:
                await anext(stream.__aiter__())
            assert "foreign-refusal-answer" not in str(missing.value)
        else:
            events = [event async for event in stream]
            assert [event.type for event in events] == [RunEventType.RUN_ERROR]
            result = events[0].run_result
            assert result.run_id == paused.run_id and result.status == paused.status
            assert [(p.origin_run_id, p.pause_id) for p in result.pendings] == [
                (p.origin_run_id, p.pause_id) for p in paused.pendings
            ]
            assert result.meta["error_type"] == (
                "GovernanceMismatchError" if policy == "governance" else "RunNotReclaimableError"
            )
            assert await database_snapshot(engine) == before
    if replace_root:
        assert await database_snapshot(engine) == after_change
    assert refused
    assert reads and all(stored_scope == tenant for _, stored_scope in reads)
    if not replace_root:
        assert {"get_run", "get_pii_mapping", "get_pending_pause_summary", "hop"} <= {
            name for name, _ in reads
        }
        for pending in reversed(paused.pendings):
            retried = await root.submit_approval(
                paused.run_id, approved=True, tenant_id=tenant, expected_pause_id=pending.pause_id
            )
        assert retried.status == RunStatus.SUCCESS


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
@pytest.mark.parametrize("shape", ["chain", "fanout"])
@pytest.mark.parametrize("policy", ["governance", "skills"])
@pytest.mark.parametrize("replace_root", [False, True], ids=["same-owner", "changed-owner"])
async def test_leaf_stream_refusal_uses_admitted_root(
    store, engine, monkeypatch, shape, policy, replace_root
):
    root = workflow(store, shape)
    child = with_skill(root.subagents[0], "original")
    siblings = root.subagents[1:]
    root._subagents = [child, *siblings]
    paused = await root.run("go", tenant_id=TENANT)
    changed = copy(child)
    if policy == "governance":
        changed._limits = Limits(max_parallel_subagents=1)
    else:
        changed = with_skill(child, "revised")
    controller = copy(root)
    controller._subagents = [changed, *siblings]
    fail = _run_resume_stream._stream_resume_failure
    refusals = []

    async def after_leaf_refusal(shared, notifier, run_id, exc):
        status = await fail(shared, notifier, run_id, exc)
        refusals.append(type(exc).__name__)
        if replace_root:
            await change_owner(engine, paused.run_id)
        return status

    monkeypatch.setattr(_run_resume_stream, "_stream_resume_failure", after_leaf_refusal)
    before = await database_snapshot(engine)
    async with controller.resume_stream(
        paused.run_id, tenant_id=TENANT, expected_pause_id=paused.pause_id
    ) as stream:
        if replace_root:
            with pytest.raises(RunNotFoundError):
                await anext(stream.__aiter__())
        else:
            events = [event async for event in stream]
            assert [event.type for event in events] == [RunEventType.RUN_ERROR]
            result = events[0].run_result
            assert result.run_id == paused.run_id and result.status == paused.status
            assert [p.pause_id for p in result.pendings] == [p.pause_id for p in paused.pendings]
            assert await database_snapshot(engine) == before
    assert refusals == [
        "GovernanceMismatchError" if policy == "governance" else "SkillMismatchError"
    ]


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
@pytest.mark.parametrize("transition", ["replacement", "completed"])
async def test_root_change_during_fanout_refusal(store, engine, monkeypatch, transition):
    root = workflow(store, "fanout")
    paused = await root.run("go", tenant_id=TENANT)
    changed = with_skill(root, "added")
    scoped_pause = _managed_pending._fanout_pause
    switched = False

    async def move_root(state, hop):
        nonlocal switched
        run_id, tenant_id = hop.run_id, hop.tenant_id
        if run_id == paused.run_id and tenant_id == TENANT and not switched:
            switched = True
            if transition == "replacement":
                await change_owner(engine, run_id)
            else:
                # Independently committed terminal outcome between the hop and envelope reads.
                async with engine.begin() as conn:
                    await conn.execute(
                        update(AgentRun)
                        .where(AgentRun.id == run_id)
                        .values(
                            status="success", pause_data=None, output_data={"answer": "completed"}
                        )
                    )
        return await scoped_pause(state, hop)

    monkeypatch.setattr(_managed_pending, "_fanout_pause", move_root)
    async with changed.resume_stream(
        paused.run_id, tenant_id=TENANT, expected_pause_id=paused.pause_id
    ) as stream:
        if transition == "replacement":
            with pytest.raises(RunNotFoundError):
                await anext(stream.__aiter__())
        else:
            events = [event async for event in stream]
            assert [event.type for event in events] == [RunEventType.RUN_ERROR]
            result = events[0].run_result
            assert (result.status, result.answer, result.pendings) == (
                RunStatus.SUCCESS,
                "completed",
                (),
            )
    assert switched


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
async def test_recovered_root_changes_during_refusal_evidence(store, engine, monkeypatch):
    from datetime import UTC, datetime

    root = workflow(store, "chain")
    paused = await root.run("go", tenant_id=TENANT)
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == paused.run_id)
            .values(last_reclaimed_at=datetime.now(UTC), reclaim_attempts=1)
        )
    read_calls = store.get_replayed_tool_calls
    scopes = []

    async def move_before_evidence(run_id, *, limit, tenant_id):
        scopes.append(tenant_id)
        await change_owner(engine, run_id)
        return await read_calls(run_id, limit=limit, tenant_id=tenant_id)

    monkeypatch.setattr(store, "get_replayed_tool_calls", move_before_evidence)
    changed = with_skill(root, "added")
    async with changed.resume_stream(
        paused.run_id, tenant_id=TENANT, expected_pause_id=paused.pause_id
    ) as stream:
        with pytest.raises(RunNotFoundError):
            await anext(stream.__aiter__())
    assert scopes == [TENANT]
