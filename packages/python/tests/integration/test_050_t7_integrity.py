"""Execution ownership and pause submissions, on SQLite and real Postgres."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from dataclasses import replace

import anyio
import pytest
from sqlalchemy import select, update

from dendrux import Agent, Limits, tool
from dendrux.db.models import AgentRun
from dendrux.errors import (
    GovernanceMismatchError,
    InvalidToolResultError,
    PauseStatusMismatchError,
    RunNotFoundError,
)
from dendrux.guardrails import PII
from dendrux.runtime.runner import resume_claimed
from dendrux.runtime.state import claimed_store
from dendrux.runtime.submit import GOVERNANCE_KEY, governance_snapshot
from dendrux.types import Budget, LLMResponse, RunStatus, ToolCall, ToolResult
from tests._helpers.scripted_provider import ScriptedLLM, tool_call


@tool(target="client")
async def client_read() -> str:
    """Read on the client."""
    raise AssertionError("client tool executed on server")


def agent(store, responses, **kwargs):
    return Agent(
        name="integrity",
        prompt="p",
        tools=[client_read],
        state_store=store,
        provider=responses if isinstance(responses, ScriptedLLM) else ScriptedLLM(responses),
        strict_tenancy=False,
        on_governance_change="warn",
        **kwargs,
    )


def result_for(paused, payload='"ok"'):
    call = paused.meta["pause_state"].pending_tool_calls[0]
    return ToolResult(name=call.name, call_id=call.id, payload=payload)


async def consume(stream):
    async with stream:
        events = [event async for event in stream]
    return stream.result, events


async def epoch_of(store, run_id):
    async with store._session_factory() as session:
        return (
            await session.execute(select(AgentRun.claim_epoch).where(AgentRun.id == run_id))
        ).scalar_one()


async def test_duplicate_resume_stream_cannot_finalize_owner(store):
    llm = ScriptedLLM(
        [tool_call("client_read", pid="a"), LLMResponse(text="owned answer")], gate_at=1
    )
    a = agent(store, llm)
    paused = await a.run("go")
    results = [result_for(paused)]
    owner = asyncio.create_task(consume(a.resume_stream(paused.run_id, tool_results=results)))
    try:
        await asyncio.wait_for(llm.entered.wait(), 5)
        losers = await asyncio.gather(
            *[consume(a.resume_stream(paused.run_id, tool_results=results)) for _ in range(8)]
        )
        assert all(result.status == RunStatus.ERROR for result, _ in losers)
        assert (await store.get_run(paused.run_id)).status == "running"
        events = await store.get_run_events(paused.run_id)
        assert not any(e.event_type in {"run.error", "run.cancelled"} for e in events)
    finally:
        llm.gate.set()
        done, _ = await owner
    assert done.status == RunStatus.SUCCESS
    assert (await store.get_run(paused.run_id)).answer == "owned answer"
    events = await store.get_run_events(paused.run_id)
    assert len({e.sequence_index for e in events}) == len(events)


@pytest.mark.parametrize("entry", ["run", "stream", "resume", "resume_stream"])
async def test_swept_worker_stops_before_next_llm_or_tool(store, entry):
    entered, gate = asyncio.Event(), asyncio.Event()
    effects = []

    @tool()
    async def slow() -> str:
        entered.set()
        await gate.wait()
        effects.append("slow")
        return "ok"

    @tool()
    async def later() -> str:
        effects.append("later")
        return "bad"

    responses = [
        tool_call("slow", pid="slow"),
        tool_call("later", pid="later"),
        LLMResponse(text="done"),
    ]
    if entry.startswith("resume"):
        responses.insert(0, tool_call("client_read", pid="client"))
    llm = ScriptedLLM(responses)
    a = Agent(
        name="integrity",
        prompt="p",
        tools=[client_read, slow, later],
        state_store=store,
        provider=llm,
        strict_tenancy=False,
    )
    if entry.startswith("resume"):
        paused = await a.run("go")
        rid = paused.run_id
        action = (
            a.resume(rid, tool_results=[result_for(paused)])
            if entry == "resume"
            else consume(a.resume_stream(rid, tool_results=[result_for(paused)]))
        )
    else:
        action = a.run("go") if entry == "run" else consume(a.stream("go"))
    task = asyncio.create_task(action)
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if not entry.startswith("resume"):
            rid = (await store.list_runs())[0].id
        before = await epoch_of(store, rid)
        async with store._session_factory() as session:
            await session.execute(
                update(AgentRun)
                .where(AgentRun.id == rid)
                .values(last_progress_at=dt.datetime.now(dt.UTC) - dt.timedelta(days=1))
            )
            await session.commit()
        assert len(await store.sweep_stale_runs(dt.timedelta(hours=1))) == 1
        assert await epoch_of(store, rid) == before + 1
    finally:
        gate.set()
        outcome = await asyncio.wait_for(task, 5)
    assert effects == ["slow"]
    assert llm.calls == (2 if entry.startswith("resume") else 1)
    assert (await store.get_run(rid)).status == "error"
    result = outcome[0] if isinstance(outcome, tuple) else outcome
    assert result.status == RunStatus.ERROR


async def test_fresh_pause_id_wins_over_stale_submissions(store):
    llm = ScriptedLLM(
        [
            tool_call("client_read", pid="a"),
            tool_call("client_read", pid="b"),
            LLMResponse(text="done"),
        ],
        gate_at=2,
    )
    a = agent(store, llm)
    first = await a.run("go")
    second = await a.submit_tool_results(
        first.run_id, [result_for(first)], expected_pause_id=first.pause_id
    )
    assert first.pause_id and second.pause_id and first.pause_id != second.pause_id
    assert len(second.pause_id) == 26
    saved = await store.get_pause_state(first.run_id)
    assert saved["pause_id"] == second.pause_id
    # Fresh call IDs with an OLD token isolate the token check from ID checking.
    with pytest.raises(PauseStatusMismatchError) as caught:
        await a.submit_tool_results(
            first.run_id, [result_for(second)], expected_pause_id=first.pause_id
        )
    assert caught.value.reason == "pause_id"
    fresh = asyncio.create_task(
        a.submit_tool_results(first.run_id, [result_for(second)], expected_pause_id=second.pause_id)
    )
    try:
        await asyncio.wait_for(llm.entered.wait(), 5)
        stale = await asyncio.gather(
            *[
                a.submit_tool_results(
                    first.run_id, [result_for(second)], expected_pause_id=first.pause_id
                )
                for _ in range(10)
            ],
            return_exceptions=True,
        )
        assert all(isinstance(item, Exception) for item in stale)
        assert (await store.get_run(first.run_id)).status == "running"
    finally:
        llm.gate.set()
        assert (await fresh).status == RunStatus.SUCCESS
    from dendrux.store import RunStore

    pairs = await RunStore(state_store=store).get_pauses(first.run_id, tenant_id=None)
    assert [p.pause_id for p in pairs] == [first.pause_id, second.pause_id]


@pytest.mark.parametrize("plain", [False, True])
async def test_pause_validation_repeats_inside_claim_transaction(store, plain):
    a = agent(store, [tool_call("client_read", pid="a"), LLMResponse(text="done")])
    paused = await a.run("go")
    name = "claim_paused_run" if plain else "submit_and_claim"
    original = getattr(store, name)

    async def replace_then_claim(run_id, **kwargs):
        raw = await store.get_pause_state(run_id)
        raw["pause_id"] = "new-pause"
        await store.pause_run(run_id, status="waiting_client_tool", pause_data=raw)
        return await original(run_id, **kwargs)

    setattr(store, name, replace_then_claim)
    try:
        with pytest.raises(PauseStatusMismatchError):
            if plain:
                await a.resume(
                    paused.run_id,
                    tool_results=[result_for(paused)],
                    expected_pause_id=paused.pause_id,
                )
            else:
                await a.submit_tool_results(
                    paused.run_id, [result_for(paused)], expected_pause_id=paused.pause_id
                )
    finally:
        setattr(store, name, original)
    assert (await store.get_run(paused.run_id)).status == "waiting_client_tool"
    assert await epoch_of(store, paused.run_id) == 0


async def test_failed_post_claim_validation_releases_owned_claim(store):
    a = agent(store, [tool_call("client_read", pid="a")])
    paused = await a.run("go")
    claim = await store.submit_and_claim(
        paused.run_id,
        expected_status="waiting_client_tool",
        submitted_data={
            "submitted_tool_results": [{"name": "wrong", "call_id": "wrong", "payload": '"ok"'}]
        },
    )
    with pytest.raises(InvalidToolResultError):
        await resume_claimed(
            paused.run_id,
            state_store=claimed_store(store, paused.run_id, claim),
            agent=a,
            provider=a._require_provider(),
            pre_claim_status="waiting_client_tool",
        )
    assert (await store.get_run(paused.run_id)).status == "waiting_client_tool"
    assert await store.get_pause_state(paused.run_id)


@pytest.mark.parametrize("entry", ["submit", "resume", "resume_stream"])
@pytest.mark.parametrize("bad", ["name", "json", "nan", "duplicate"])
async def test_client_results_validate_identity_and_normalize_text_before_claim(store, entry, bad):
    a = agent(store, [tool_call("client_read", pid="a"), LLMResponse(text="done")])
    paused = await a.run("go")
    good = result_for(paused)
    results = [
        replace(good, name="forged")
        if bad == "name"
        else replace(good, payload="{" if bad == "json" else "NaN")
    ]
    if bad == "duplicate":
        results = [good, good]
    if bad in {"json", "nan"}:
        if entry == "submit":
            result = await a.submit_tool_results(paused.run_id, results)
        elif entry == "resume":
            result = await a.resume(paused.run_id, tool_results=results)
        else:
            result, _ = await consume(a.resume_stream(paused.run_id, tool_results=results))
        assert result.status == RunStatus.SUCCESS
        message = next(m for m in a.provider.messages[-1] if m.role.value == "tool")
        assert json.loads(message.content) == ("{" if bad == "json" else "NaN")
        return
    if entry == "resume_stream":
        result, _ = await consume(a.resume_stream(paused.run_id, tool_results=results))
        assert result.status == RunStatus.ERROR
    else:
        with pytest.raises(InvalidToolResultError):
            if entry == "submit":
                await a.submit_tool_results(paused.run_id, results)
            else:
                await a.resume(paused.run_id, tool_results=results)
    assert (await store.get_run(paused.run_id)).status == "waiting_client_tool"
    assert await epoch_of(store, paused.run_id) == 0


@pytest.mark.parametrize("entry", ["submit", "resume", "resume_stream"])
async def test_client_result_cap_then_transform(store, entry):
    seen = []

    async def transform(ctx):
        seen.append(ctx)
        return "transformed"

    llm = ScriptedLLM([tool_call("client_read", pid="a"), LLMResponse(text="done")])
    a = agent(store, llm, limits=Limits(max_tool_result_bytes=200), tool_result_transform=transform)
    paused = await a.run("go")
    results = [result_for(paused, json.dumps("x" * 2000))]
    if entry == "submit":
        result = await a.submit_tool_results(paused.run_id, results)
    elif entry == "resume":
        result = await a.resume(paused.run_id, tool_results=results)
    else:
        result, _ = await consume(a.resume_stream(paused.run_id, tool_results=results))
    assert result.status == RunStatus.SUCCESS
    assert len(seen) == 1 and len(seen[0].result) < 2000
    assert seen[0].tool_name == "client_read"
    messages = [m for m in llm.messages[-1] if m.role.value == "tool"]
    assert messages[-1].content == '"transformed"'


@pytest.mark.parametrize("mode", ["warn", "raise"])
@pytest.mark.parametrize("entry", ["submit", "resume", "resume_stream"])
async def test_governance_change_is_visible_or_preserves_pause(store, mode, entry, caplog):
    a = agent(store, [tool_call("client_read", pid="a")], guardrails=[PII(action="redact")])
    paused = await a.run("contact alice@example.com")
    llm = ScriptedLLM([LLMResponse(text="done")])
    b = Agent(
        name="integrity",
        prompt="p",
        tools=[client_read],
        provider=llm,
        state_store=store,
        strict_tenancy=False,
        on_governance_change=mode,
    )
    results = [result_for(paused)]
    if entry == "resume_stream":
        result, _ = await consume(b.resume_stream(paused.run_id, tool_results=results))
    elif mode == "raise":
        with pytest.raises(GovernanceMismatchError):
            if entry == "submit":
                await b.submit_tool_results(paused.run_id, results)
            else:
                await b.resume(paused.run_id, tool_results=results)
        result = None
    else:
        result = await (
            b.submit_tool_results(paused.run_id, results)
            if entry == "submit"
            else b.resume(paused.run_id, tool_results=results)
        )
    if mode == "raise":
        assert (await store.get_run(paused.run_id)).status == "waiting_client_tool"
        assert llm.calls == 0
    else:
        assert result.status == RunStatus.SUCCESS
        events = [
            e
            for e in await store.get_run_events(paused.run_id)
            if e.event_type == "governance.changed"
        ]
        assert len(events) == 1 and events[0].data["changed"] == ["guardrails"]
        assert "alice@example.com" not in json.dumps(events[0].data)
        assert "code=governance_changed" in caplog.text


async def test_legacy_governance_is_skipped_and_logged(store, caplog):
    a = agent(store, [tool_call("client_read", pid="a"), LLMResponse(text="done")])
    paused = await a.run("go")
    raw = await store.get_pause_state(paused.run_id)
    raw.pop(GOVERNANCE_KEY)
    await store.pause_run(paused.run_id, status="waiting_client_tool", pause_data=raw)
    caplog.set_level("INFO")
    assert (
        await a.resume(paused.run_id, tool_results=[result_for(paused)])
    ).status == RunStatus.SUCCESS
    assert "code=governance_legacy" in caplog.text


async def test_budget_amount_changes_do_not_change_governance_digest(store):
    a = agent(store, [], budget=Budget(max_tokens=100, count_cache_tokens=True))
    b = agent(store, [], budget=Budget(max_tokens=1000, count_cache_tokens=True))
    assert governance_snapshot(a) == governance_snapshot(b)


@pytest.mark.parametrize("entry", ["submit", "resume", "resume_stream"])
@pytest.mark.parametrize("policy", ["deny", "cap"])
async def test_approved_tools_recheck_current_deny_and_call_cap(store, entry, policy):
    effects = []

    def limited(limit, label):
        @tool(max_calls_per_run=limit)
        async def effect() -> str:
            effects.append(label)
            return "ok"

        return effect

    initial = limited(3, "initial")
    current = limited(1, "current")

    a = Agent(
        name="integrity",
        prompt="p",
        tools=[initial],
        require_approval=["effect"],
        provider=ScriptedLLM([LLMResponse(tool_calls=[ToolCall("effect", {}) for _ in range(3)])]),
        state_store=store,
        strict_tenancy=False,
        on_governance_change="warn",
    )
    paused = await a.run("go")
    b = Agent(
        name="integrity",
        prompt="p",
        tools=[current],
        deny=["effect"] if policy == "deny" else [],
        provider=ScriptedLLM([LLMResponse(text="done")]),
        state_store=store,
        strict_tenancy=False,
        on_governance_change="warn",
    )
    if entry == "submit":
        result = await b.submit_approval(
            paused.run_id, approved=True, expected_pause_id=paused.pause_id
        )
    elif entry == "resume":
        result = await b.resume(paused.run_id)
    else:
        result, _ = await consume(b.resume_stream(paused.run_id))
    assert result.status == RunStatus.SUCCESS
    assert effects == ([] if policy == "deny" else ["current"])


@pytest.mark.parametrize("entry", ["stream", "resume_stream"])
async def test_anyio_cancelled_stream_persists_cancelled(store, entry):
    llm = ScriptedLLM(
        ([tool_call("client_read", pid="a")] if entry == "resume_stream" else [])
        + [LLMResponse(text="done")],
        gate_at=1 if entry == "resume_stream" else 0,
    )
    a = agent(store, llm)
    if entry == "resume_stream":
        paused = await a.run("go")
        stream = a.resume_stream(paused.run_id, tool_results=[result_for(paused)])
    else:
        stream = a.stream("go")
    with anyio.CancelScope() as scope:

        async def cancel():
            await llm.entered.wait()
            scope.cancel()

        async with anyio.create_task_group() as group:
            group.start_soon(cancel)
            await consume(stream)
    record = await store.get_run(stream.run_id)
    assert record.status == "cancelled"
    assert (
        sum(e.event_type == "run.cancelled" for e in await store.get_run_events(stream.run_id)) == 1
    )


@pytest.mark.parametrize("entry", ["tool", "input", "approval", "resume", "retry"])
async def test_foreign_and_missing_runs_have_one_not_found_error(store, entry):
    a = agent(store, [tool_call("client_read", pid="a")])
    paused = await a.run("go", tenant_id="owner")
    for rid in [paused.run_id, "missing"]:
        with pytest.raises(RunNotFoundError):
            if entry == "tool":
                await a.submit_tool_results(rid, [result_for(paused)], tenant_id="other")
            elif entry == "input":
                await a.submit_input(rid, "answer", tenant_id="other")
            elif entry == "approval":
                await a.submit_approval(rid, approved=True, tenant_id="other")
            elif entry == "resume":
                await a.submit_resume(rid, tenant_id="other")
            else:
                await a.retry(rid, tenant_id="other")
