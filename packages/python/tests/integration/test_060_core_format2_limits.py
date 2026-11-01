"""Format-2-only entry bounds, rejected effects, and compact checkpoint bytes."""

import json

import pytest

from dendrux import Agent, tool
from dendrux.errors import PersistenceError
from dendrux.llm.mock import MockLLM
from dendrux.runtime.state import LoopStatePatch
from dendrux.types import LLMResponse, RunStatus, ToolCall, generate_ulid
from tests._fixtures.storage import pin_storage_format


@pytest.mark.parametrize("fmt", [0, 1, 2])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("option", ["large", "object"])
async def test_provider_limits_only_for_checkpoints(store, engine, fmt, streaming, option):
    await pin_storage_format(engine, fmt or 1)
    seen = []

    class Model(MockLLM):
        async def complete(self, *args, **kwargs):
            seen.append(kwargs["extra_body"])
            return await super().complete(*args, **kwargs)

        async def complete_stream(self, *args, **kwargs):
            seen.append(kwargs["extra_body"])
            async for chunk in super().complete_stream(*args, **kwargs):
                yield chunk

    value = "x" * 3000 if option == "large" else object()
    agent = Agent(
        prompt="p",
        provider=Model([LLMResponse(text="done")]),
        state_store=store if fmt else None,
        strict_tenancy=False,
    )
    if streaming:
        async with agent.stream("go", extra_body=value) as stream:
            events = [event async for event in stream]
        result = events[-1].run_result
        assert result.status == (RunStatus.ERROR if fmt == 2 else RunStatus.SUCCESS)
    elif fmt == 2:
        with pytest.raises(ValueError):
            await agent.run("go", extra_body=value)
    else:
        assert (await agent.run("go", extra_body=value)).status == RunStatus.SUCCESS
    assert (not seen) if fmt == 2 else all(item is value for item in seen) and bool(seen)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("fmt", [1, 2])
async def test_rejected_calls_are_failed_only_in_format2(store, engine, streaming, fmt):
    await pin_storage_format(engine, fmt)
    executed = []

    @tool(max_calls_per_run=1)
    async def work(value: int) -> int:
        """Return the value."""
        executed.append(value)
        return value

    calls = [
        ToolCall("missing"),
        ToolCall("work", params={}),
        ToolCall("work", params={"value": 1}),
        ToolCall("work", params={"value": 2}),
    ]
    agent = Agent(
        prompt="p",
        provider=MockLLM([LLMResponse(tool_calls=calls), LLMResponse(text="done")]),
        tools=[work],
        state_store=store,
        strict_tenancy=False,
    )
    if streaming:
        async with agent.stream("go") as stream:
            events = [event async for event in stream]
        result = events[-1].run_result
    else:
        result = await agent.run("go")
    assert executed == [1]
    rows = {row.tool_call_id: row for row in await store.get_tool_calls(result.run_id)}
    assert rows[calls[2].id].outcome == "completed"
    assert [rows[calls[i].id].outcome for i in (0, 1, 3)] == [
        "failed" if fmt == 2 else "completed"
    ] * 3


@pytest.mark.parametrize("overflow", [False, True])
async def test_compact_checkpoint_patch_margin_and_rollback(store, engine, overflow):
    run_id = generate_ulid()
    await store.create_run(run_id, "compact")
    # Many short keys make PostgreSQL's spaced JSONB text exceed the compact bound.
    payload = {str(i): "" for i in range(2500)}
    payload["quoted"] = 'é " \\ , : spaced string'

    def compact(value):
        return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())

    payload["padding"] = "a" * (31 * 1024 - compact({**payload, "padding": ""}))
    assert compact(payload) == 31 * 1024
    await store.checkpoint_step(
        run_id, claim_epoch=0, loop_state=payload, pii_mapping=None, trace=None
    )
    patch = LoopStatePatch(set_fields={"new": "b" * (2048 if overflow else 1)})

    async def apply():
        async with store._session_factory() as session:
            await store._patch_fence(session, run_id, 0, patch)
            await session.commit()

    if overflow:
        with pytest.raises(PersistenceError) as caught:
            await apply()
        assert caught.value.code == "checkpoint_too_large"
        assert await store.get_loop_state(run_id) == payload
    else:
        await apply()
        assert await store.get_loop_state(run_id) == {**payload, **patch.set_fields}


@pytest.mark.parametrize("streaming", [False, True])
async def test_successful_hook_substitution_is_not_an_executed_effect(store, streaming):
    from dendrux import ToolDecision

    @tool()
    async def work() -> str:
        """Must not run."""
        raise AssertionError("substituted effect ran")

    agent = Agent(
        prompt="p",
        provider=MockLLM([LLMResponse(tool_calls=[ToolCall("work")]), LLMResponse(text="done")]),
        tools=[work],
        before_tool=lambda call: ToolDecision.substitute({"answer": "cached"}),
        state_store=store,
        strict_tenancy=False,
    )
    if streaming:
        async with agent.stream("go") as stream:
            events = [event async for event in stream]
        result = events[-1].run_result
    else:
        result = await agent.run("go")
    rows = await store.get_tool_calls(result.run_id)
    assert len(rows) == 1 and rows[0].success
    assert rows[0].outcome == "failed"


@pytest.mark.parametrize("entry", ["run", "stream", "retry"])
@pytest.mark.parametrize("option", ["large", "object"])
async def test_format2_refuses_options_before_creation(store, engine, monkeypatch, entry, option):
    from functools import wraps

    from sqlalchemy import func, select

    from dendrux.db.models import AgentRun

    agent = Agent(
        prompt="p",
        provider=MockLLM([LLMResponse(text="done")]),
        state_store=store,
        strict_tenancy=False,
    )
    original_run = await agent.run("original") if entry == "retry" else None
    created = []
    create = store.create_run

    @wraps(create)
    async def capture(*args, **kwargs):
        created.append(args[0])
        return await create(*args, **kwargs)

    monkeypatch.setattr(store, "create_run", capture)
    value = "x" * 3000 if option == "large" else object()
    if entry == "stream":
        async with agent.stream("go", extra_body=value) as stream:
            events = [event async for event in stream]
        assert events[-1].run_result.status == RunStatus.ERROR
    else:
        with pytest.raises(ValueError):
            if entry == "retry":
                await agent.retry(original_run.run_id, extra_body=value)
            else:
                await agent.run("go", extra_body=value)
    assert not created, "refusal must precede create_run, not merely roll it back"
    async with engine.connect() as connection:
        assert await connection.scalar(select(func.count()).select_from(AgentRun)) == (
            1 if original_run else 0
        )


async def test_entry_settings_failure_is_a_safe_persistence_error(store, monkeypatch):
    from contextlib import asynccontextmanager

    from sqlalchemy.exc import OperationalError

    @asynccontextmanager
    async def unavailable():
        raise OperationalError("settings", {}, Exception("PRIVATE_DATABASE_DETAIL"))
        yield  # pragma: no cover

    monkeypatch.setattr(store, "_session_factory", unavailable)
    agent = Agent(prompt="p", provider=MockLLM([]), state_store=store, strict_tenancy=False)
    with pytest.raises(PersistenceError) as caught:
        await agent.run("go")
    assert "PRIVATE_DATABASE_DETAIL" not in str(caught.value)
