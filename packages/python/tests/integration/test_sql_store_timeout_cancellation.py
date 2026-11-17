"""Timeouts and previously handled cancellation retain ordinary store outcomes."""

import asyncio
from contextlib import suppress
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from dendrux import Agent
from dendrux.errors import PersistenceError
from dendrux.llm.mock import MockLLM
from dendrux.runtime import durability
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.types import LLMResponse, generate_ulid


@pytest.mark.parametrize("commit_number", [2, 3, 4])
async def test_commit_timeout_does_not_cancel_run(store, monkeypatch, commit_number):
    real_commit = AsyncSession.commit
    calls = 0
    monkeypatch.setattr(durability, "BASE_BACKOFF_S", 0)

    async def commit(session):
        nonlocal calls
        calls += 1
        if calls == commit_number:
            async with asyncio.timeout(0):
                await asyncio.sleep(0)
        await real_commit(session)

    agent = Agent(
        provider=MockLLM([LLMResponse(text="done")]),
        prompt="p",
        state_store=store,
        strict_tenancy=False,
    )
    try:
        with monkeypatch.context() as patch:
            patch.setattr(AsyncSession, "commit", commit)
            result = await agent.run("go")
        assert result.status.value == "success"
        assert (await store.get_run(result.run_id)).status == "success"
        assert calls > commit_number
    finally:
        await agent.close()


@pytest.mark.parametrize("site", ["claim", "submit"])
@pytest.mark.parametrize("active_timeout", [False, True])
async def test_prior_handled_cancel_does_not_replace_validation_error(store, site, active_timeout):
    run_id = generate_ulid()
    await store.create_run(run_id, "stale-cancel")
    await store.pause_run(run_id, status="waiting_client_tool", pause_data={"ready": True})
    failure = ValueError("invalid submission")

    def validate(_):
        raise failure

    async def call():
        with pytest.raises(ValueError) as caught:
            if site == "claim":
                await store.claim_paused_run(
                    run_id, expected_status="waiting_client_tool", validate=validate
                )
            else:
                await store.submit_and_claim(
                    run_id,
                    expected_status="waiting_client_tool",
                    submitted_data={},
                    validate=validate,
                )
        assert caught.value is failure

    async def worker():
        task = asyncio.current_task()
        task.cancel()
        # Simulate a previously handled cancellation without uncancel().
        with suppress(asyncio.CancelledError):
            await asyncio.sleep(0)
        assert task.cancelling() == 1
        if active_timeout:
            try:
                async with asyncio.timeout(0):
                    await asyncio.sleep(0)
            except TimeoutError:
                await call()
        else:
            await call()
        assert task.cancelling() == 1

    await asyncio.create_task(worker())
    run = await store.get_run(run_id)
    assert (run.status, run.claim_epoch) == ("waiting_client_tool", 0)


@pytest.mark.parametrize("method", ["create", "finalize", "claim"])
async def test_asyncpg_connect_timeout_remains_typed_persistence_error(monkeypatch, method):
    handlers = set()
    writers = set()

    async def silent(reader, writer):
        task = asyncio.current_task()
        handlers.add(task)
        writers.add(writer)
        try:
            await reader.read()
        finally:
            writers.discard(writer)
            writer.close()
            await writer.wait_closed()
            handlers.discard(task)

    server = await asyncio.start_server(silent, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    engine = create_async_engine(
        f"postgresql+asyncpg://postgres:root@127.0.0.1:{port}/dendrux_070_r2_test",
        connect_args={"timeout": 0.02},
    )
    store = store_for_engine(engine)
    run_id = generate_ulid()
    operations = {
        "create": lambda: store.create_run(run_id, "timeout"),
        "finalize": lambda: store.finalize_run(run_id, status="success"),
        "claim": lambda: store.claim_paused_run(run_id, expected_status="waiting_client_tool"),
    }
    monkeypatch.setattr(durability, "_backoff", AsyncMock())
    try:
        before = asyncio.current_task().cancelling()
        with pytest.raises(PersistenceError):
            await operations[method]()
        assert asyncio.current_task().cancelling() == before
    finally:
        await store.close()
        await engine.dispose()
        server.close()
        await server.wait_closed()
        for writer in tuple(writers):
            writer.close()
        await asyncio.gather(*tuple(handlers))
