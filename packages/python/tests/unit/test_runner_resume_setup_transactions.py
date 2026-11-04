"""Setup SQL and transaction boundaries captured before continuation refactoring."""

import inspect
import json
from collections import Counter
from functools import wraps
from pathlib import Path

import pytest
from sqlalchemy import event, update

from dendrux import Agent
from dendrux.db.models import AgentRun
from dendrux.errors import RunNotReclaimableError
from dendrux.llm.mock import MockLLM
from dendrux.types import LLMResponse, RunStatus, ToolCall, ToolResult
from tests._fixtures.storage import pin_storage_format
from tests._helpers.recovery_checkpoint import checkpoint
from tests.unit.test_runner_failure_save import client_read
from tests.unit.test_runner_failure_save import store as store  # noqa: F401


@pytest.mark.parametrize("entry", ["resume", "resume_stream", "submit_tool_results", "reclaim"])
@pytest.mark.parametrize("storage_format", [1, 2])
async def test_resume_setup_transaction_bytes(store, monkeypatch, entry, storage_format):
    await pin_storage_format(store._engine, storage_format)
    provider = MockLLM(
        [
            LLMResponse(tool_calls=[ToolCall("client_read", id="client-call")]),
            LLMResponse(text="done"),
        ]
    )
    agent = Agent(
        name="setup",
        prompt="p",
        tools=[client_read],
        provider=provider,
        state_store=store,
        strict_tenancy=False,
    )
    if entry == "reclaim":
        # Reclaim cannot continue legacy rows; snapshot its existing refusal too.
        await pin_storage_format(store._engine, 2)
        item = await checkpoint(agent, store, store._engine)
        run_id = item.run_id
        if storage_format == 1:
            async with store._engine.begin() as conn:
                await conn.execute(
                    update(AgentRun).where(AgentRun.id == run_id).values(storage_format=1)
                )
        provider._responses = [LLMResponse(text="done")]
    else:
        paused = await agent.run("go", tenant_id=None)
        run_id = paused.run_id
    active = True
    reached_provider = False
    statements = []
    counts = Counter()

    def instrument(name, method):
        @wraps(method)
        async def wrapped(*args, **kwargs):
            if active:
                counts[name] += 1
            return await method(*args, **kwargs)

        return wrapped

    for name in dir(store):
        method = getattr(store, name)
        if not name.startswith("_") and inspect.iscoroutinefunction(method):
            monkeypatch.setattr(store, name, instrument(name, method))

    original = provider.complete

    async def complete(*args, **kwargs):
        nonlocal active, reached_provider
        active = False
        reached_provider = True
        return await original(*args, **kwargs)

    monkeypatch.setattr(provider, "complete", complete)

    def boundary(name):
        def capture(conn):
            if active:
                statements.append(name)

        return capture

    def execute(conn, cursor, statement, parameters, context, executemany):
        if active:
            statements.append(" ".join(statement.split()))

    engine = store._engine.sync_engine
    listeners = [(name, boundary(name.upper())) for name in ("begin", "commit", "rollback")]
    listeners.append(("before_cursor_execute", execute))
    for name, listener in listeners:
        event.listen(engine, name, listener)
    try:
        if entry == "reclaim":
            if storage_format == 1:
                with pytest.raises(RunNotReclaimableError) as caught:
                    await agent.reclaim(run_id, tenant_id="tenant", item=item)
                assert caught.value.reason == "legacy_run"
            else:
                result = await agent.reclaim(run_id, tenant_id="tenant", item=item)
                assert result.status == RunStatus.SUCCESS
        else:
            results = [ToolResult("client_read", "client-call", '"ok"')]
            kwargs = dict(tenant_id=None, expected_pause_id=paused.pause_id)
            if entry == "submit_tool_results":
                result = await agent.submit_tool_results(run_id, results, **kwargs)
            elif entry == "resume":
                result = await agent.resume(run_id, tool_results=results, **kwargs)
            else:
                stream = agent.resume_stream(run_id, tool_results=results, **kwargs)
                async for _ in stream:
                    pass
                result = stream.result
            assert result.status == RunStatus.SUCCESS
    finally:
        for name, listener in listeners:
            event.remove(engine, name, listener)
        await agent.close()
    assert reached_provider == (entry != "reclaim" or storage_format == 2)
    wire = (
        json.dumps({"calls": dict(sorted(counts.items())), "transactions": statements}, indent=2)
        + "\n"
    )
    expected = Path(__file__).with_name("fixtures") / f"resume_setup_{entry}_{storage_format}.json"
    assert wire == expected.read_text()
