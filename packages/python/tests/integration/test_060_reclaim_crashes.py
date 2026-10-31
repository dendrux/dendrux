"""Public recovery after real SIGKILL cuts, including a second recovery crash."""

import asyncio
import multiprocessing
import signal
from datetime import UTC, datetime, timedelta
from functools import wraps

import pytest
from sqlalchemy import update

from dendrux import Agent, sweep, tool
from dendrux.db.models import AgentRun
from dendrux.llm.mock import MockLLM
from dendrux.types import LLMResponse, ToolCall, UsageStats
from tests.integration.test_060_core_crash_cuts import CUTS, _kill_at_cut


def recovery_agent(store, cut, effect_path, *, provider=None):
    @tool(idempotent=cut != "approval_replay")
    async def change() -> str:
        """Record a visible side effect."""
        with open(effect_path, "a") as output:
            output.write("effect\n")
        return "ok"

    @tool(target="client")
    async def client() -> str:
        """A client-owned call."""
        raise AssertionError("client executed")

    @tool(idempotent=True)
    async def slow() -> str:
        """Finish first call after restart."""
        return "slow"

    @tool(idempotent=True)
    async def middle() -> str:
        """Finish second call after restart."""
        return "middle"

    @tool(idempotent=True)
    async def fast() -> str:
        """Already completed in initial process."""
        raise AssertionError("completed sibling executed twice")

    tools = (
        [slow, middle, fast]
        if cut == "parallel"
        else [client]
        if cut == "client_submit"
        else [change]
    )
    return Agent(
        prompt="system",
        provider=provider or MockLLM([LLMResponse(text="final answer")]),
        tools=tools,
        state_store=store,
        require_approval=["change"]
        if cut in ("gate", "approval_submit", "approval_consume", "rejected")
        else [],
        guardrails=[],
        strict_tenancy=False,
        on_governance_change="raise",
    )


async def expire(store, engine, run):
    async with engine.begin() as conn:
        await conn.execute(
            update(AgentRun)
            .where(AgentRun.id == run)
            .values(last_progress_at=datetime.now(UTC) - timedelta(minutes=10))
        )
    return next(
        r
        for r in (
            await sweep(state_store=store, stale_running=timedelta(minutes=1), on_stale="reclaim")
        ).reclaimable
        if r.run_id == run
    )


def second_worker(url, run, cut, effects, pipe):
    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux.runtime.stores._factory import store_for_engine

    class StopBeforeModel(MockLLM):
        async def complete(self, *args, **kwargs):
            pipe.send("resolved")
            pipe.recv()
            raise AssertionError("kill handshake returned")

    async def main():
        engine = create_async_engine(url)
        agent = recovery_agent(store_for_engine(engine), cut, effects, provider=StopBeforeModel([]))
        await agent.reclaim(run, tenant_id=None)

    try:
        asyncio.run(main())
    except BaseException:
        import traceback

        pipe.send({"error": traceback.format_exc()})
        raise


async def kill_process(target, args):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=target, args=(*args, child))
    process.start()
    child.close()
    try:
        assert await asyncio.to_thread(parent.poll, 30), "worker missed crash boundary"
        message = parent.recv()
        assert not isinstance(message, dict) or "error" not in message, message
        process.kill()
        await asyncio.to_thread(process.join, 10)
        assert process.exitcode == -signal.SIGKILL
        return message
    finally:
        if process.is_alive():
            process.kill()
            await asyncio.to_thread(process.join, 10)
        parent.close()


def sweep_worker(url, pipe):
    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux.runtime.stores._factory import store_for_engine

    async def main():
        engine = create_async_engine(url)
        page = await sweep(
            state_store=store_for_engine(engine),
            stale_running=timedelta(minutes=1),
            on_stale="reclaim",
        )
        pipe.send(page.reclaimable)
        pipe.recv()

    try:
        asyncio.run(main())
    except BaseException:
        import traceback

        pipe.send({"error": traceback.format_exc()})
        raise


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
async def test_sweeper_sigkill_after_mark_is_rediscovered(store, engine):
    from tests.integration.test_060_reclaim_sweep import candidate

    run = await candidate(store, engine)
    marked = await kill_process(sweep_worker, (engine.url.render_as_string(hide_password=False),))
    page = await sweep(state_store=store, stale_running=timedelta(minutes=1), on_stale="reclaim")
    assert page.reclaimable == marked
    assert [r.run_id for r in marked] == [run]
    events = await store.get_run_events(run)
    assert len([e for e in events if e.event_type == "run.lease_expired"]) == 1


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
@pytest.mark.parametrize("cut", CUTS)
async def test_resume_phase1_sigkill_cut(engine, store, tmp_path, cut):
    effects = tmp_path / "effects"
    url = engine.url.render_as_string(hide_password=False)
    first = await _kill_at_cut(url, cut, effects)
    run = first["run"]
    item = await expire(store, engine, run)
    # A second SIGKILL after actual pending resolution, before the model, proves
    # completed effects and consumed submissions survive another process loss.
    second = cut in (
        "assistant",
        "intent",
        "returned",
        "parallel",
        "approval_submit",
        "approval_consume",
        "rejected",
        "input_submit",
        "input_consume",
        "client_submit",
    )
    if second:
        await kill_process(second_worker, (url, run, cut, str(effects)))
        item = await expire(store, engine, run)
    agent = recovery_agent(store, cut, effects)
    result = await agent.reclaim(run, tenant_id=None, item=item)
    if cut in ("gate", "approval_replay"):
        assert result.status.value == "waiting_approval"
        if cut == "approval_replay":
            assert effects.read_text() == "effect\n"
        result = await agent.submit_approval(
            run,
            tenant_id=None,
            approved=cut != "approval_replay",
            expected_pause_id=result.pause_id,
        )
    elif cut == "clarification":
        assert result.status.value == "waiting_human_input"
        result = await agent.submit_input(
            run, "chosen", tenant_id=None, expected_pause_id=result.pause_id
        )
    assert result.status.value == "success"
    assert result.recovery.reclaim_attempts == (2 if second else 1)
    rows = await store.get_tool_calls(run)
    if cut == "approval_replay":
        assert effects.read_text() == "effect\n"
        assert rows[0].outcome == "unknown"
    if cut == "parallel":
        assert [r.outcome for r in rows] == ["completed"] * 3
    state = await store.get_loop_state(run)
    assert not state["pending"] and state["submission"] is None
    traces = await store.get_traces(run)
    indexes = [t.order_index for t in traces]
    assert len(indexes) == len(set(indexes))


def billed_worker(url, cut, bills, pipe):
    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux.runtime.context import get_delegation_context
    from dendrux.runtime.stores._factory import store_for_engine
    from dendrux.runtime.stores.sql.base import SQLStateStore

    def halt(run):
        pipe.send(run)
        pipe.recv()
        raise AssertionError("kill handshake returned")

    class Store(type(store_for_engine(create_async_engine(url)))):
        @wraps(SQLStateStore.checkpoint_step)
        async def checkpoint_step(self, run, **kwargs):
            await super().checkpoint_step(run, **kwargs)
            trace = kwargs.get("trace")
            if (
                cut == "checkpoint"
                and kwargs["loop_state"].get("iteration") == 3
                and trace
                and (trace.meta or {}).get("tool_calls")
            ):
                halt(run)

    class Provider(MockLLM):
        async def complete(self, messages, **kwargs):
            with open(bills, "a") as f:
                f.write("billed\n")
            if cut == "inflight" and self.calls_made == 2:
                halt(get_delegation_context().run_id)
            return await super().complete(messages, **kwargs)

    async def main():
        engine = create_async_engine(url)
        agent = bill_agent(
            Store(engine),
            Provider(
                [
                    LLMResponse(
                        tool_calls=[ToolCall("work")],
                        usage=UsageStats(input_tokens=1, output_tokens=1, total_tokens=2),
                    )
                    for _ in range(4)
                ]
                + [
                    LLMResponse(
                        text="done",
                        usage=UsageStats(input_tokens=1, output_tokens=1, total_tokens=2),
                    )
                ]
            ),
        )
        await agent.run("go", tenant_id="tenant")

    try:
        asyncio.run(main())
    except BaseException:
        import traceback

        pipe.send({"error": traceback.format_exc()})
        raise


def bill_agent(store, provider):
    @tool(idempotent=True)
    async def work() -> str:
        """Idempotent work."""
        return "ok"

    return Agent(
        name="billing",
        prompt="system",
        provider=provider,
        tools=[work],
        state_store=store,
        strict_tenancy=True,
        on_governance_change="raise",
    )


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
@pytest.mark.parametrize("cut,expected", [("checkpoint", 5), ("inflight", 6)])
async def test_a1_a1b_billed_calls(engine, store, tmp_path, cut, expected):
    bills = tmp_path / "bills"
    run = await kill_process(
        billed_worker, (engine.url.render_as_string(hide_password=False), cut, str(bills))
    )
    item = await expire(store, engine, run)
    state = await store.get_loop_state(run)
    remaining = 4 - state["iteration"]

    class Provider(MockLLM):
        async def complete(self, *args, **kwargs):
            with bills.open("a") as f:
                f.write("billed\n")
            return await super().complete(*args, **kwargs)

    model = Provider(
        [
            LLMResponse(
                tool_calls=[ToolCall("work")],
                usage=UsageStats(input_tokens=1, output_tokens=1, total_tokens=2),
            )
            for _ in range(remaining)
        ]
        + [
            LLMResponse(
                text="done", usage=UsageStats(input_tokens=1, output_tokens=1, total_tokens=2)
            )
        ]
    )
    result = await bill_agent(store, model).reclaim(run, tenant_id="tenant", item=item)
    assert result.status.value == "success"
    assert len(bills.read_text().splitlines()) == expected
    assert result.recovery.reclaim_attempts == 1
    assert result.usage.total_tokens == 10
