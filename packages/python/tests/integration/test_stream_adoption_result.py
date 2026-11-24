"""A losing stream follows the outcome committed by the controller that adopted."""

import asyncio

import pytest

from dendrux.db.models import AgentRun
from dendrux.errors import GovernanceMismatchError, PersistenceError
from dendrux.runtime import _managed
from dendrux.runtime.state import UNSCOPED
from dendrux.types import LLMResponse, RunEventType, ToolResult
from tests.integration.test_managed_stream import (
    BRIEF,
    CHILD_TEXT,
    ROOT_DONE,
    TENANT,
    CrashError,
    TaskLLM,
    agent,
    call,
    collect,
    read_sheet,
    rows,
)


class PublicStore:
    def __init__(self, store):
        for name in dir(store):
            if not name.startswith("_") and name not in {"json_get", "pause_projection"}:
                setattr(self, name, getattr(store, name))


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
@pytest.mark.parametrize("outcome", ["completed", "new-pause", "unscoped-reader", "read-error"])
async def test_losing_adoption_reads_the_winners_result(store, engine, monkeypatch, outcome):
    adapter = PublicStore(store)
    child = agent("reviewer", TaskLLM([LLMResponse(text=CHILD_TEXT)]), tools=[])
    next_step = (
        LLMResponse(text=ROOT_DONE)
        if outcome != "new-pause"
        else call("read_sheet", {"sheet": "orders"}, "next")
    )
    root = agent(
        "lead",
        TaskLLM([call("reviewer", {"task": BRIEF}, "r1"), next_step, LLMResponse(text=ROOT_DONE)]),
        tools=[read_sheet],
        subagents=[child],
        state_store=adapter,
    )
    claim = _managed.claim_adoption

    async def crash(*args):
        raise CrashError

    with monkeypatch.context() as setup:
        setup.setattr(_managed, "claim_adoption", crash)
        with pytest.raises(CrashError):
            await root.run("go", tenant_id=TENANT)
    (record,) = await rows(engine, AgentRun, AgentRun.parent_run_id.is_(None))
    ready, release = asyncio.Event(), asyncio.Event()
    claims, reads = [], []

    async def held_claim(*args):
        if not ready.is_set():
            ready.set()
            await release.wait()
        result = await claim(*args)
        claims.append(result is not None)
        return result

    def track_read(name):
        method = getattr(store, name)

        async def read(run_id, *, tenant_id=UNSCOPED):
            if claims == [True, False]:
                reads.append((name, tenant_id))
                if outcome == "read-error" and name == "get_run":
                    raise GovernanceMismatchError(run_id)
            return await method(run_id, tenant_id=tenant_id)

        return read

    monkeypatch.setattr(_managed, "claim_adoption", held_claim)
    for name in ("get_run", "get_pii_mapping", "get_pending_pause_summary"):
        monkeypatch.setattr(adapter, name, track_read(name))

    async def losing_stream():
        stream = root.resume_stream(record.id, tenant_id=TENANT)
        if outcome == "read-error":
            with pytest.raises(GovernanceMismatchError):
                await collect(stream)
            return []
        if outcome == "unscoped-reader":
            with pytest.raises(PersistenceError) as failure:
                await collect(stream)
            assert failure.value.retryable
            return []
        return await collect(stream)

    async with asyncio.timeout(10), asyncio.TaskGroup() as group:
        loser = group.create_task(losing_stream())
        await ready.wait()
        try:
            winner = (await collect(root.resume_stream(record.id, tenant_id=TENANT)))[-1]
            if outcome == "unscoped-reader":

                async def unsafe(run_id):
                    pytest.fail("the losing read must keep the admitted tenant")

                monkeypatch.setattr(adapter, "get_run", unsafe)
        finally:
            release.set()
    assert claims == [True, False]
    if outcome in {"unscoped-reader", "read-error"}:
        assert not loser.result() and winner.run_result.answer == ROOT_DONE
        return
    (event,) = loser.result()
    assert {scope for _, scope in reads} == {TENANT}
    expected_reads = {"get_run", "get_pii_mapping"}
    if outcome == "new-pause":
        expected_reads.add("get_pending_pause_summary")
    assert {name for name, _ in reads} == expected_reads
    assert event.run_id == record.id and event.type == winner.type
    result = event.run_result
    if outcome == "completed":
        assert event.type == RunEventType.RUN_COMPLETED and result.answer == ROOT_DONE
    else:
        assert event.type == RunEventType.RUN_PAUSED
        assert result.pause_id == winner.run_result.pause_id and result.pause_id
        (pending,) = result.pendings
        (tool_call,) = pending.tool_calls
        assert tool_call == winner.run_result.pending.tool_calls[0]
        final = await root.submit_tool_results(
            record.id,
            [ToolResult(tool_call.name, tool_call.id, '"cells"')],
            tenant_id=TENANT,
            expected_pause_id=result.pause_id,
        )
        assert final.answer == ROOT_DONE


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
async def test_leaf_manifest_refusal_keeps_the_custom_stream_error(store, engine):
    from dendrux.skills._session import META_KEY
    from tests.integration.test_managed_routing import build

    root = build(PublicStore(store), "approval")
    paused = await root.run("go", tenant_id=TENANT)
    await store.update_run_meta(paused.pending.origin_run_id, {META_KEY: {"version": 2}})
    (event,) = await collect(
        root.resume_stream(paused.run_id, tenant_id=TENANT, expected_pause_id=paused.pause_id)
    )
    assert event.type == RunEventType.RUN_ERROR and event.run_result.run_id == paused.run_id
    assert "Invalid skill manifest" in event.error
