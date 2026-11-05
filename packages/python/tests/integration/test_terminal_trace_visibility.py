"""Finished runs expose every committed row even when reservations remain."""

import asyncio
from datetime import timedelta
from functools import wraps

import pytest

from dendrux import Agent, sweep, tool
from dendrux.llm.mock import MockLLM
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.store import RunStore
from dendrux.types import LLMResponse, ToolCall
from tests._fixtures.storage import pin_storage_format
from tests._helpers.public_trace_store import KeysetStore


def _agent(store, gate, *, max_iterations=10):
    @tool(parallel=True, idempotent=True)
    async def slow() -> str:
        await gate.wait()
        return "SLOW"

    @tool(parallel=True, idempotent=True)
    async def fast() -> str:
        return "FAST"

    return Agent(
        prompt="p",
        state_store=store,
        tools=[slow, fast],
        max_iterations=max_iterations,
        provider=MockLLM(
            [
                LLMResponse(tool_calls=[ToolCall("slow", id="slow"), ToolCall("fast", id="fast")]),
                LLMResponse(text="done"),
            ]
        ),
    )


async def _assert_visible(store, run_id, status):
    assert (await store.get_run(run_id)).status == status
    for reader in (store, KeysetStore(store)):
        rows = await RunStore(reader).get_traces(run_id, tenant_id="t", limit=50)
        assert '"FAST"' in [r.content for r in rows]
        assert [r.order_index for r in rows] == list(range(len(rows)))


@pytest.mark.parametrize("terminal", ["success", "error", "cancelled", "max_iterations"])
async def test_terminal_paths_expose_completed_parallel_rows(store, engine, monkeypatch, terminal):
    await pin_storage_format(engine, 2)
    gate, saved = asyncio.Event(), asyncio.Event()
    identity = []
    original = store.complete_tool_call

    @wraps(original)
    async def complete(run_id, **kwargs):
        if kwargs["tool_call_id"] == "slow" and terminal == "error":
            raise RuntimeError("completion unavailable")
        await original(run_id, **kwargs)
        if kwargs["tool_call_id"] == "fast":
            identity.append(run_id)
            saved.set()

    monkeypatch.setattr(store, "complete_tool_call", complete)
    task = asyncio.create_task(
        _agent(store, gate, max_iterations=1 if terminal == "max_iterations" else 10).run(
            "go", tenant_id="t"
        )
    )
    try:
        await saved.wait()
        if terminal == "cancelled":
            task.cancel()
        else:
            gate.set()
        try:
            await task
        except asyncio.CancelledError:
            assert terminal == "cancelled"
        except RuntimeError as exc:
            assert terminal == "error" and str(exc) == "completion unavailable"
        await _assert_visible(store, identity[0], terminal)
    finally:
        gate.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def _crash_worker(url, pipe):
    from sqlalchemy.ext.asyncio import create_async_engine

    async def main():
        engine = create_async_engine(url)
        base = type(store_for_engine(engine))

        class Store(base):
            @wraps(base.complete_tool_call)
            async def complete_tool_call(self, run_id, **kwargs):
                await super().complete_tool_call(run_id, **kwargs)
                if kwargs["tool_call_id"] == "fast":
                    pipe.send(run_id)
                    pipe.recv()
                    raise AssertionError("kill handshake returned")

        await _agent(Store(engine), asyncio.Event()).run("go", tenant_id="t")

    asyncio.run(main())


async def test_crashed_parallel_run_sweep_exposes_completed_rows(store, engine):
    from tests.integration.test_060_reclaim_crashes import kill_process

    if engine.dialect.name != "postgresql":
        pytest.skip("process crash requires Postgres")
    await pin_storage_format(engine, 2)
    run_id = await kill_process(_crash_worker, (engine.url.render_as_string(hide_password=False),))
    assert (await store.get_loop_state(run_id))["pending"]
    assert '"FAST"' not in [
        r.content for r in await RunStore(store).get_traces(run_id, tenant_id="t", limit=50)
    ]
    await sweep(state_store=store, stale_running=timedelta(0), on_stale="error")
    await _assert_visible(store, run_id, "error")
