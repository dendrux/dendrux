"""Cancellation hardening (PR 6).

Covers:
  - StateStore: request_cancel + is_cancel_requested round-trip + idempotency
  - StateStore: finalize_run_if_status_in atomic CAS over multiple statuses
  - StateStore: finalize_run clears cancel_requested on terminal
  - Runner: top-of-iteration checkpoint exits as CANCELLED
  - Runner: pre-pause checkpoint finalizes cancelled instead of pausing
  - Agent.submit_*: rejects when cancel_requested=True
  - Agent.cancel_run: emits run.cancelled event (not run.completed)
  - Agent.cancel_run: idempotent under concurrency
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.agent import Agent
from dendrux.db.models import AgentRun, Base, RunEvent
from dendrux.errors import RunAlreadyTerminalError
from dendrux.llm.mock import MockLLM
from dendrux.runtime.state import SQLAlchemyStateStore
from dendrux.tool import tool
from dendrux.types import LLMResponse, RunStatus, ToolCall, ToolResult, UsageStats


@tool()
async def server_echo(text: str) -> str:
    return text


@tool(target="client")
async def client_read(sheet: str) -> str:
    return ""


@tool()
async def refund_tool(order_id: int) -> str:
    return f"Refunded {order_id}"


@pytest.fixture
async def db_store():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    store = SQLAlchemyStateStore(engine)
    try:
        yield store
    finally:
        await engine.dispose()


def _client_agent(llm: MockLLM, store: SQLAlchemyStateStore) -> Agent:
    return Agent(
        provider=llm,
        prompt="Test agent.",
        tools=[server_echo, client_read],
        state_store=store,
    )


def _approval_agent(llm: MockLLM, store: SQLAlchemyStateStore) -> Agent:
    return Agent(
        provider=llm,
        prompt="Test agent.",
        tools=[refund_tool],
        require_approval=["refund_tool"],
        state_store=store,
    )


# ---------------------------------------------------------------------------
# StateStore primitives
# ---------------------------------------------------------------------------


class TestStateStoreCancelFlag:
    async def test_round_trip_and_idempotent(self, db_store) -> None:
        await db_store.create_run("r1", "agent")
        assert await db_store.is_cancel_requested("r1") is False

        won = await db_store.request_cancel("r1")
        assert won is True
        assert await db_store.is_cancel_requested("r1") is True

        # Idempotent — setting again on a non-terminal row is still True.
        won = await db_store.request_cancel("r1")
        assert won is True
        assert await db_store.is_cancel_requested("r1") is True

    async def test_missing_run_returns_false(self, db_store) -> None:
        assert await db_store.is_cancel_requested("nope") is False

    async def test_request_cancel_refuses_terminal_rows(self, db_store) -> None:
        """Race guard: a finalize that lands between a caller's preflight
        and the request_cancel UPDATE must NOT leave cancel_requested=True
        on the terminal row."""
        await db_store.create_run("r1", "agent")
        await db_store.finalize_run("r1", status=RunStatus.SUCCESS.value)

        won = await db_store.request_cancel("r1")
        assert won is False
        # The flag must remain False on the terminal row.
        assert await db_store.is_cancel_requested("r1") is False


class TestFinalizeRunIfStatusIn:
    async def test_matches_one_of_allowed(self, db_store) -> None:
        await db_store.create_run("r1", "agent")
        # Move to running so we can pause.
        from sqlalchemy import update

        async with db_store._session_factory() as session:
            await session.execute(
                update(AgentRun).where(AgentRun.id == "r1").values(status="running")
            )
            await session.commit()
        await db_store.pause_run("r1", status="waiting_client_tool", pause_data={"x": 1})

        won = await db_store.finalize_run_if_status_in(
            "r1",
            status=RunStatus.CANCELLED.value,
            allowed_current_statuses=[
                RunStatus.RUNNING.value,
                RunStatus.WAITING_CLIENT_TOOL.value,
            ],
            total_usage=UsageStats(input_tokens=10, output_tokens=5),
        )
        assert won is True
        record = await db_store.get_run("r1")
        assert record.status == RunStatus.CANCELLED.value

        # pause_data and cancel_requested are cleared on terminal.
        async with db_store._session_factory() as session:
            row = (await session.execute(select(AgentRun).where(AgentRun.id == "r1"))).scalar_one()
            assert row.pause_data is None
            assert row.cancel_requested is False

    async def test_no_match_returns_false(self, db_store) -> None:
        await db_store.create_run("r1", "agent")
        # Force terminal so the CAS misses.
        await db_store.finalize_run("r1", status=RunStatus.SUCCESS.value)

        won = await db_store.finalize_run_if_status_in(
            "r1",
            status=RunStatus.CANCELLED.value,
            allowed_current_statuses=[
                RunStatus.RUNNING.value,
                RunStatus.WAITING_CLIENT_TOOL.value,
            ],
        )
        assert won is False
        record = await db_store.get_run("r1")
        assert record.status == RunStatus.SUCCESS.value


class TestFinalizeRunClearsCancelFlag:
    async def test_cleared_on_success(self, db_store) -> None:
        from sqlalchemy import update

        await db_store.create_run("r1", "agent")
        await db_store.request_cancel("r1")
        async with db_store._session_factory() as session:
            await session.execute(
                update(AgentRun).where(AgentRun.id == "r1").values(status="running")
            )
            await session.commit()

        await db_store.finalize_run(
            "r1", status=RunStatus.SUCCESS.value, expected_current_status="running"
        )
        record = await db_store.get_run("r1")
        assert record.status == RunStatus.SUCCESS.value
        assert record.cancel_requested is False


# ---------------------------------------------------------------------------
# Runner cooperative cancel
# ---------------------------------------------------------------------------


class TestRunnerCheckpoint:
    async def test_top_of_iteration_observes_flag(self, db_store) -> None:
        """Set cancel_requested before resume; runner must finalize cancelled
        on its first iteration check rather than completing the work."""
        tc = ToolCall(name="client_read", params={"sheet": "S1"}, provider_tool_call_id="t1")
        # Mock has a 2nd response in case we wrongly proceed past the checkpoint.
        llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="should not be reached")])
        agent = _client_agent(llm, db_store)

        paused = await agent.run("read S1")
        assert paused.status == RunStatus.WAITING_CLIENT_TOOL

        # Caller flips the flag before resume gets a chance to iterate.
        await db_store.request_cancel(paused.run_id)

        pause = await db_store.get_pause_state(paused.run_id)
        call_id = pause["pending_tool_calls"][0]["id"]

        # Submit must reject — the run is on the path to CANCELLED.
        with pytest.raises(RunAlreadyTerminalError):
            await agent.submit_tool_results(
                paused.run_id,
                [ToolResult(name="client_read", call_id=call_id, payload='"rows"')],
            )


class TestPrePauseCheckpoint:
    async def test_cancel_during_iteration_finalizes_cancelled_not_paused(self, db_store) -> None:
        """A cancel that lands while the iteration is mid-run must produce
        a CANCELLED finalize, not leave the run paused holding stale
        pause_data."""

        # MockLLM provider that flips the cancel flag inline before returning
        # a tool-call response. This simulates a cancel arriving during the
        # iteration that ultimately produces a WAITING_CLIENT_TOOL pause.
        class _CancellingProvider:
            model = "mock"

            def __init__(self, store: SQLAlchemyStateStore) -> None:
                self.store = store
                self.run_id_holder: dict[str, str] = {}
                self._call_count = 0

            async def complete(self, messages, tools, **_):
                self._call_count += 1
                # Find run_id by reading the most recent created run
                # (single-run integration test).
                async with self.store._session_factory() as session:
                    row = (
                        await session.execute(
                            select(AgentRun).order_by(AgentRun.created_at.desc()).limit(1)
                        )
                    ).scalar_one()
                    await self.store.request_cancel(row.id)
                tc = ToolCall(
                    name="client_read",
                    params={"sheet": "S1"},
                    provider_tool_call_id=f"t{self._call_count}",
                )
                return LLMResponse(tool_calls=[tc])

            async def complete_stream(self, *_, **__):
                raise NotImplementedError

        provider = _CancellingProvider(db_store)
        agent = Agent(
            provider=provider,
            prompt="Test agent.",
            tools=[client_read],
            state_store=db_store,
        )

        result = await agent.run("read")
        # The loop returned WAITING_CLIENT_TOOL but cancel_requested is set,
        # so the runner's pre-pause checkpoint flips to CANCELLED.
        assert result.status == RunStatus.CANCELLED


# ---------------------------------------------------------------------------
# Agent.cancel_run
# ---------------------------------------------------------------------------


class TestCancelRunEvent:
    async def test_emits_run_cancelled_not_run_completed(self, db_store) -> None:
        tc = ToolCall(name="client_read", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc])])
        agent = _client_agent(llm, db_store)
        paused = await agent.run("read")

        await agent.cancel_run(paused.run_id)

        # Verify the event emitted is run.cancelled, not run.completed.
        async with db_store._session_factory() as session:
            rows = (
                (
                    await session.execute(
                        select(RunEvent).where(RunEvent.agent_run_id == paused.run_id)
                    )
                )
                .scalars()
                .all()
            )
        event_types = [r.event_type for r in rows]
        assert "run.cancelled" in event_types
        # No spurious run.completed for a cancelled run.
        assert not any(t == "run.completed" for t in event_types)

    async def test_cancel_event_appended_after_existing_events(self, db_store) -> None:
        """run.cancelled must sort AFTER run.started/run.paused so SSE
        clients reading by after_sequence_index see it."""
        tc = ToolCall(name="client_read", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc])])
        agent = _client_agent(llm, db_store)
        paused = await agent.run("read")

        await agent.cancel_run(paused.run_id)

        async with db_store._session_factory() as session:
            rows = (
                (
                    await session.execute(
                        select(RunEvent)
                        .where(RunEvent.agent_run_id == paused.run_id)
                        .order_by(RunEvent.sequence_index)
                    )
                )
                .scalars()
                .all()
            )

        assert len(rows) >= 2
        cancelled = next(r for r in rows if r.event_type == "run.cancelled")
        max_other = max(r.sequence_index for r in rows if r.event_type != "run.cancelled")
        assert cancelled.sequence_index > max_other


class TestCancelRunIdempotent:
    async def test_concurrent_cancel_both_succeed(self, db_store) -> None:
        tc = ToolCall(name="client_read", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc])])
        agent = _client_agent(llm, db_store)
        paused = await agent.run("read")

        a, b = await asyncio.gather(
            agent.cancel_run(paused.run_id),
            agent.cancel_run(paused.run_id),
        )
        assert a.status == RunStatus.CANCELLED
        assert b.status == RunStatus.CANCELLED


# ---------------------------------------------------------------------------
# Submit rejection when cancel pending
# ---------------------------------------------------------------------------


class TestSubmitRejectedAfterCancel:
    async def test_submit_tool_results_rejected(self, db_store) -> None:
        tc = ToolCall(name="client_read", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc])])
        agent = _client_agent(llm, db_store)
        paused = await agent.run("read")
        await db_store.request_cancel(paused.run_id)

        pause = await db_store.get_pause_state(paused.run_id)
        call_id = pause["pending_tool_calls"][0]["id"]
        with pytest.raises(RunAlreadyTerminalError):
            await agent.submit_tool_results(
                paused.run_id,
                [ToolResult(name="client_read", call_id=call_id, payload='"x"')],
            )

    async def test_submit_approval_rejected(self, db_store) -> None:
        tc = ToolCall(name="refund_tool", params={"order_id": 1}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc])])
        agent = _approval_agent(llm, db_store)
        paused = await agent.run("refund 1")
        await db_store.request_cancel(paused.run_id)

        with pytest.raises(RunAlreadyTerminalError):
            await agent.submit_approval(paused.run_id, approved=True)
