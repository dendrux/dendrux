"""Root start callbacks at the owned, committed run boundary."""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from dendrux import Agent, cancel_run
from dendrux.db.models import AgentRun, RunEvent
from dendrux.llm.mock import MockLLM
from dendrux.loops.base import BaseNotifier
from dendrux.loops.react import ReActLoop
from dendrux.loops.single import SingleCall
from dendrux.runtime import _run_tools, runner
from dendrux.runtime.context import (
    DelegationContext,
    reset_delegation_context,
    set_delegation_context,
)
from dendrux.types import LLMResponse, RunAlreadyActiveError, RunStatus
from tests._helpers.gates import wait_for_gate
from tests.integration.test_managed_inline import ContextLLM, call, child_agent, root_agent


def make_agent(store=None, **options):
    provider = MockLLM([LLMResponse(text="done"), LLMResponse(text="retried")])
    return Agent(provider=provider, prompt="answer", state_store=store, **options), provider


@pytest.mark.parametrize("entry", [Agent.run, Agent.retry, runner.run, runner.retry])
def test_explicit_keyword_only(entry):
    param = inspect.signature(entry).parameters.get("on_started")
    assert param is not None and param.kind == inspect.Parameter.KEYWORD_ONLY
    assert param.default is None


@pytest.mark.parametrize("entry", ["run", "retry"])
@pytest.mark.parametrize("direct", [False, True], ids=["agent", "runner"])
async def test_noncallable_on_started_leaves_no_run_or_event(
    store, engine, entry, direct, monkeypatch
):
    agent, provider = make_agent(store)
    source = await agent.run("source", tenant_id=None) if entry == "retry" else None

    async def persisted_ids():
        async with engine.connect() as conn:
            runs = set(await conn.scalars(select(AgentRun.id)))
            events = set(await conn.scalars(select(RunEvent.id)))
        return runs, events

    before = await persisted_ids()
    resolve_store = AsyncMock(wraps=agent._resolve_state_store)
    monkeypatch.setattr(agent, "_resolve_state_store", resolve_store)
    options = dict(tenant_id=None, on_started="not callable")
    if direct:
        options.update(agent=agent, provider=provider, state_store=store)
    entry_point = runner if direct else agent
    with pytest.raises(ValueError, match="on_started"):
        if source:
            await entry_point.retry(source.run_id, **options)
        else:
            await entry_point.run(user_input="go", **options)
    assert await persisted_ids() == before
    resolve_store.assert_not_awaited()
    assert provider.calls_made == int(source is not None)


@pytest.mark.parametrize("entry", ["run", "retry"])
async def test_async_callback_observes_claim_in_caller_before_model(store, entry):
    agent, provider = make_agent(store)
    source = await agent.run("hello", tenant_id=None) if entry == "retry" else None
    caller = asyncio.current_task()
    seen = []

    async def started(run_id):
        row = await store.get_run(run_id)
        assert row.status == "running" and row.claim_epoch == 0
        assert provider.calls_made == int(source is not None)
        assert agent._task_manager._owned[run_id] is caller is asyncio.current_task()
        seen.append(run_id)

    if source:
        result = await agent.retry(source.run_id, tenant_id=None, on_started=started)
        assert result.run_id != source.run_id
    else:
        result = await agent.run("hello", tenant_id=None, on_started=started)
    assert result.status == RunStatus.SUCCESS
    assert seen == [result.run_id]
    assert result.run_id not in agent._task_manager._owned


@pytest.mark.parametrize("entry", ["run_stream", "resume_stream"])
@pytest.mark.parametrize("callback", [None, lambda run_id: None], ids=["none", "callable"])
async def test_runner_stream_refuses_on_started_at_construction(store, engine, entry, callback):
    agent, provider = make_agent(store)
    resolver = AsyncMock(return_value=store)
    options = dict(
        agent=agent, provider=provider, state_store_resolver=resolver, on_started=callback
    )
    with pytest.raises(ValueError) as refused:
        if entry == "run_stream":
            runner.run_stream(user_input="go", tenant_id=None, **options)
        else:
            runner.resume_stream("01ARZ3NDEKTSV4RRFFQ69G5FAV", **options)
    assert str(refused.value) == "on_started is only supported on run()/retry()."
    resolver.assert_not_awaited()
    assert provider.call_history == []
    async with engine.connect() as conn:
        assert (await conn.scalars(select(AgentRun.id))).first() is None
        assert (await conn.scalars(select(RunEvent.id))).first() is None


@pytest.mark.parametrize("entry", ["run", "retry"])
async def test_sync_callback_and_provider_options(store, entry):
    agent, provider = make_agent(store)
    seen = []
    options = dict(agent=agent, provider=provider, state_store=store, tenant_id=None)
    callback_options = dict(on_started=seen.append, temperature=0.5)
    if entry == "retry":
        source = await runner.run(user_input="hello", **options)
        result = await runner.retry(source.run_id, **options, **callback_options)
    else:
        result = await runner.run(user_input="hello", **options, **callback_options)
    kwargs = provider.call_history[-1]["kwargs"]
    assert "on_started" not in kwargs and kwargs["temperature"] == 0.5
    assert seen == [result.run_id]


async def test_active_and_terminal_idempotency_hits(store):
    agent, provider = make_agent(store)
    seen, duplicates = [], []

    async def started(run_id):
        seen.append(run_id)
        with pytest.raises(RunAlreadyActiveError) as active:
            await agent.run(
                "hello", tenant_id=None, idempotency_key="same", on_started=duplicates.append
            )
        assert active.value.run_id == run_id

    options = dict(tenant_id=None, idempotency_key="same")
    first = await agent.run("hello", on_started=started, **options)
    cached = await agent.run("hello", on_started=duplicates.append, **options)
    assert seen == [first.run_id] and duplicates == []
    assert cached.run_id == first.run_id and provider.calls_made == 1


async def test_ephemeral_run_suppresses_callback():
    agent, provider = make_agent()
    seen = []
    result = await agent.run("hello", on_started=seen.append)
    assert result.status == RunStatus.SUCCESS and seen == []
    assert "on_started" not in provider.call_history[0]["kwargs"]


@pytest.mark.parametrize("entry", ["run", "retry"])
async def test_child_with_no_durable_parent_link(store, entry):
    agent, provider = make_agent(store)
    source = await agent.run("hello", tenant_id=None) if entry == "retry" else None
    seen = []
    token = set_delegation_context(DelegationContext("ephemeral-parent", tenant_explicit=True))
    try:
        if source:
            result = await agent.retry(source.run_id, tenant_id=None, on_started=seen.append)
        else:
            result = await agent.run("hello", tenant_id=None, on_started=seen.append)
    finally:
        reset_delegation_context(token)
    row = await store.get_run(result.run_id)
    assert row.delegation_level == 1 and row.parent_run_id is None
    assert seen == [] and "on_started" not in provider.call_history[-1]["kwargs"]


async def test_managed_child_does_not_inherit_callback(store):
    child_llm = ContextLLM([LLMResponse(text="reviewed")])
    root_llm = ContextLLM([call("reviewer", {"task": "review"}, "c"), LLMResponse(text="done")])
    root = root_agent(store, root_llm, child_agent(store, child_llm, tools=[]))
    seen = []
    result = await root.run("hello", tenant_id="t1", on_started=seen.append)
    assert result.status == RunStatus.SUCCESS
    assert len(child_llm.served) == 1 and child_llm.served[0][0] != result.run_id
    assert seen == [result.run_id]


async def retry_source(store, entry):
    if entry == "run":
        return None
    seeder, _ = make_agent(store)
    return await seeder.run("hello", tenant_id=None)


@pytest.mark.parametrize("engine", ["sqlite_file"], indirect=True)
@pytest.mark.parametrize("entry", ["run", "retry"])
@pytest.mark.parametrize("loop_type", [ReActLoop, SingleCall])
@pytest.mark.parametrize("when", ["before", "inside", "after"])
@pytest.mark.parametrize("cancel", ["agent", "module", "task"])
async def test_cancellation_at_callback_boundary(
    store, monkeypatch, entry, loop_type, when, cancel
):
    loop = loop_type()
    agent, provider = make_agent(store, loop=loop)
    source = await retry_source(store, entry)
    loop_run = AsyncMock(wraps=loop.run)
    monkeypatch.setattr(loop, "run", loop_run)
    entered, release = asyncio.Event(), asyncio.Event()
    seen = []
    ids = []
    returned = False

    async def block():
        entered.set()
        await release.wait()

    class Notifier(BaseNotifier):
        async def on_run_started(self, run_id, **kwargs):
            ids.append(run_id)
            if when == "before":
                await block()

    async def started(run_id):
        nonlocal returned
        seen.append(run_id)
        if when == "inside":
            await block()
        returned = True

    read_control = store.get_run_control

    async def control(run_id):
        if when == "after" and returned and asyncio.current_task() is task:
            await block()
        return await read_control(run_id)

    monkeypatch.setattr(store, "get_run_control", control)
    options = dict(tenant_id=None, notifier=Notifier(), on_started=started)
    coro = agent.retry(source.run_id, **options) if source else agent.run("hello", **options)
    task = asyncio.create_task(coro)
    try:
        await wait_for_gate(entered, task=task)
        (run_id,) = ids
        if cancel == "task":
            task.cancel("caller stopped")
        elif cancel == "agent":
            await agent.cancel_run(run_id, tenant_id=None)
        else:
            await cancel_run(run_id, tenant_id=None, state_store=store)
        release.set()
        if cancel == "task":
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            assert caught.value.args == ("caller stopped",)
        else:
            result = await task
            assert result.status == RunStatus.CANCELLED
            assert bool(result.meta.get("interrupted")) == (cancel == "agent")
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    # A local task interrupt can unwind startup before the callback boundary.
    interrupted_before_entry = when == "before" and cancel in {"agent", "task"}
    assert seen == ([] if interrupted_before_entry else [run_id])
    assert (await store.get_run(run_id)).status == "cancelled"
    events = await store.get_run_events(run_id)
    (event,) = [event for event in events if event.event_type == "run.cancelled"]
    assert event.data["reason"] == ("task_cancelled" if cancel == "task" else "cancel_requested")
    assert run_id not in agent._task_manager._owned
    assert provider.calls_made == 0
    loop_run.assert_not_called()


@pytest.mark.parametrize("entry", ["run", "retry"])
@pytest.mark.parametrize("loop_type", [ReActLoop, SingleCall])
@pytest.mark.parametrize("changed", ["status", "epoch"])
@pytest.mark.parametrize("partial_control", [False, True], ids=["full-control", "legacy-control"])
async def test_changed_claim_stops_before_loop(
    store, engine, monkeypatch, entry, loop_type, changed, partial_control
):
    from sqlalchemy import update

    from dendrux.db.models import AgentRun
    from dendrux.runtime.state import RunControl

    loop = loop_type()
    agent, provider = make_agent(store, loop=loop)
    source = await retry_source(store, entry)
    loop_run = AsyncMock(wraps=loop.run)
    monkeypatch.setattr(loop, "run", loop_run)
    discovery = AsyncMock(wraps=_run_tools._discover_runtime_tools)
    monkeypatch.setattr(_run_tools, "_discover_runtime_tools", discovery)
    if partial_control:
        # A legacy custom store exposes the remaining ownership fields on its run record.
        monkeypatch.setattr(store, "get_run_control", AsyncMock(return_value=RunControl()))
        monkeypatch.setattr(store, "_complete_run_control", None)
        monkeypatch.setattr(store, "_requires_persisted_claim", False)

    async def started(run_id):
        if changed == "status":
            assert await store.finalize_run(
                run_id, status="success", answer="winner", expected_current_status="running"
            )
        else:
            async with engine.begin() as conn:
                await conn.execute(
                    update(AgentRun).where(AgentRun.id == run_id).values(claim_epoch=1)
                )

    if source:
        result = await agent.retry(source.run_id, tenant_id=None, on_started=started)
    else:
        result = await agent.run("hello", tenant_id=None, on_started=started)
    row = await store.get_run(result.run_id)
    expected = ("success", "winner", 0) if changed == "status" else ("running", None, 1)
    assert (row.status, row.answer, row.claim_epoch) == expected
    assert (result.status.value, result.answer) == expected[:2]
    assert provider.calls_made == 0
    discovery.assert_not_called()
    loop_run.assert_not_called()


@pytest.mark.parametrize("entry", ["run", "retry"])
async def test_sync_callback_return_value_is_ignored(store, entry):
    source = await retry_source(store, entry)
    agent, provider = make_agent(store)
    seen = []

    def save_id(run_id):
        seen.append(run_id)
        return run_id

    if source:
        result = await agent.retry(source.run_id, tenant_id=None, on_started=save_id)
    else:
        result = await agent.run("hello", tenant_id=None, on_started=save_id)
    assert seen == [result.run_id]
    assert result.status == RunStatus.SUCCESS and provider.calls_made == 1


@pytest.mark.parametrize("engine", ["sqlite_file"], indirect=True)
@pytest.mark.parametrize("entry", ["run", "retry"])
@pytest.mark.parametrize("change", ["cancel", "reclaim"])
@pytest.mark.parametrize("wrap_recorder", [False, True], ids=["native", "wrapped"])
async def test_single_call_reads_claim_at_provider_boundary(
    store, engine, entry, change, monkeypatch, wrap_recorder
):
    from functools import wraps

    from sqlalchemy import update

    from dendrux.db.models import AgentRun
    from dendrux.loops import _llm_call, single

    source = await retry_source(store, entry)
    agent, provider = make_agent(store, loop=SingleCall())
    started, announced = [], []
    ready, release = asyncio.Event(), asyncio.Event()
    prelude = single._single_prelude

    class PublicRecorder:
        def __init__(self, recorder):
            self.recorder = recorder

        def __getattr__(self, name):
            if name.startswith("_"):
                raise AttributeError(name)
            return getattr(self.recorder, name)

    @wraps(prelude)
    async def wrap_loop_recorder(*args, **kwargs):
        pre = await prelude(*args, **kwargs)
        pre.calls.recorder = PublicRecorder(pre.calls.recorder)
        return pre

    if wrap_recorder:
        monkeypatch.setattr(single, "_single_prelude", wrap_loop_recorder)
    prepare = single.ImageCallSite.prepare

    @wraps(prepare)
    async def before_admission(*args, **kwargs):
        request = await prepare(*args, **kwargs)
        ready.set()
        await release.wait()
        return request

    monkeypatch.setattr(single.ImageCallSite, "prepare", before_admission)
    record_start = AsyncMock(wraps=_llm_call.record_llm_started)
    monkeypatch.setattr(_llm_call, "record_llm_started", record_start)

    class Notifier(BaseNotifier):
        async def on_llm_call_started(self, run_id, iteration, **kwargs):
            announced.append(run_id)

    options = dict(tenant_id=None, on_started=started.append, notifier=Notifier())
    task = asyncio.create_task(
        agent.retry(source.run_id, **options) if source else agent.run("hello", **options)
    )
    try:
        await wait_for_gate(ready, task=task)
        (run_id,) = started
        if change == "cancel":
            await cancel_run(run_id, tenant_id=None, state_store=store)
        else:
            async with engine.begin() as conn:
                await conn.execute(
                    update(AgentRun).where(AgentRun.id == run_id).values(claim_epoch=1)
                )
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    result = await task
    assert provider.calls_made == 0
    expected = RunStatus.CANCELLED if change == "cancel" else RunStatus.RUNNING
    row = await store.get_run(run_id)
    assert result.status == expected and row.status == expected.value
    assert row.claim_epoch == int(change == "reclaim")
    record_start.assert_not_awaited()
    assert announced == []
    assert all(e.event_type != "llm.started" for e in await store.get_run_events(run_id))
