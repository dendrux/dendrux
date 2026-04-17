"""End-to-end tests for Agent submit/cancel methods (PR 4).

Covers race-safety, precondition contracts, and public exception
shapes for ``submit_tool_results``, ``submit_input``, ``submit_approval``,
and ``cancel_run`` — the methods that replace the bridge's HTTP surface.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.agent import Agent
from dendrux.db.models import Base
from dendrux.errors import (
    InvalidToolResultError,
    PauseStatusMismatchError,
    PersistenceNotConfiguredError,
    RunAlreadyClaimedError,
    RunAlreadyTerminalError,
    RunNotFoundError,
    RunNotPausedError,
)
from dendrux.llm.mock import MockLLM
from dendrux.runtime.state import SQLAlchemyStateStore
from dendrux.tool import tool
from dendrux.types import LLMResponse, RunStatus, ToolCall, ToolResult


@tool()
async def server_add(a: int, b: int) -> int:
    return a + b


@tool()
async def refund(order_id: int) -> str:
    return f"Refunded {order_id}"


@tool(target="client")
async def read_range(sheet: str) -> str:
    return ""


@tool(target="client")
async def ask_user(question: str) -> str:
    """Client-tool clarifier — pauses for human input when invoked."""
    return ""


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


def _client_tool_agent(llm: MockLLM, store: SQLAlchemyStateStore) -> Agent:
    return Agent(
        provider=llm,
        prompt="Test agent.",
        tools=[server_add, read_range],
        state_store=store,
    )


def _approval_agent(llm: MockLLM, store: SQLAlchemyStateStore) -> Agent:
    return Agent(
        provider=llm,
        prompt="Support agent.",
        tools=[refund],
        require_approval=["refund"],
        state_store=store,
    )


async def _force_running(store: SQLAlchemyStateStore, run_id: str) -> None:
    """Set a run's status to RUNNING directly (test helper — skips state machine)."""
    from sqlalchemy import update

    from dendrux.db.models import AgentRun

    async with store._session_factory() as session:
        await session.execute(
            update(AgentRun).where(AgentRun.id == run_id).values(status="running")
        )
        await session.commit()


# ---------------------------------------------------------------------------
# submit_tool_results
# ---------------------------------------------------------------------------


class TestSubmitToolResults:
    async def test_happy_path(self, db_store) -> None:
        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="done")])
        agent = _client_tool_agent(llm, db_store)

        paused = await agent.run("Read the sheet")
        assert paused.status == RunStatus.WAITING_CLIENT_TOOL

        pause = await db_store.get_pause_state(paused.run_id)
        call_id = pause["pending_tool_calls"][0]["id"]

        result = await agent.submit_tool_results(
            paused.run_id,
            [ToolResult(name="read_range", call_id=call_id, payload='"rows"')],
        )
        assert result.status == RunStatus.SUCCESS
        assert result.answer == "done"

    async def test_invalid_tool_result_ids_raises(self, db_store) -> None:
        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc])])
        agent = _client_tool_agent(llm, db_store)

        paused = await agent.run("Read the sheet")
        with pytest.raises(InvalidToolResultError):
            await agent.submit_tool_results(
                paused.run_id,
                [ToolResult(name="read_range", call_id="wrong-id", payload='"x"')],
            )

    async def test_missing_run_raises_not_found(self, db_store) -> None:
        llm = MockLLM([])
        agent = _client_tool_agent(llm, db_store)
        with pytest.raises(RunNotFoundError):
            await agent.submit_tool_results(
                "does-not-exist",
                [ToolResult(name="read_range", call_id="x", payload='""')],
            )

    async def test_not_paused_raises(self, db_store) -> None:
        # Agent completes without pausing — subsequent submit sees terminal.
        llm = MockLLM([LLMResponse(text="immediate answer")])
        agent = _client_tool_agent(llm, db_store)
        completed = await agent.run("hello")
        assert completed.status == RunStatus.SUCCESS

        with pytest.raises(RunAlreadyTerminalError):
            await agent.submit_tool_results(
                completed.run_id,
                [ToolResult(name="read_range", call_id="x", payload='""')],
            )

    async def test_pause_status_mismatch(self, db_store) -> None:
        # Pause for approval, then try to submit tool results.
        tc = ToolCall(name="refund", params={"order_id": 1}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc])])
        agent = _approval_agent(llm, db_store)
        paused = await agent.run("refund order 1")
        assert paused.status == RunStatus.WAITING_APPROVAL

        pause = await db_store.get_pause_state(paused.run_id)
        call_id = pause["pending_tool_calls"][0]["id"]
        with pytest.raises(PauseStatusMismatchError):
            await agent.submit_tool_results(
                paused.run_id,
                [ToolResult(name="refund", call_id=call_id, payload='"ok"')],
            )

    async def test_no_persistence_raises(self) -> None:
        llm = MockLLM([])
        agent = Agent(provider=llm, prompt="x", tools=[read_range])
        with pytest.raises(PersistenceNotConfiguredError):
            await agent.submit_tool_results(
                "run-id", [ToolResult(name="read_range", call_id="x", payload='""')]
            )

    async def test_duplicate_call_ids_rejected(self, db_store) -> None:
        """Two results for one pending call is rejected, even if the id matches."""
        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc])])
        agent = _client_tool_agent(llm, db_store)
        paused = await agent.run("read")
        pause = await db_store.get_pause_state(paused.run_id)
        call_id = pause["pending_tool_calls"][0]["id"]

        dup = [
            ToolResult(name="read_range", call_id=call_id, payload='"x"'),
            ToolResult(name="read_range", call_id=call_id, payload='"y"'),
        ]
        with pytest.raises(InvalidToolResultError, match="duplicate"):
            await agent.submit_tool_results(paused.run_id, dup)


# ---------------------------------------------------------------------------
# submit_input
# ---------------------------------------------------------------------------


class TestSubmitInput:
    async def test_pause_status_mismatch(self, db_store) -> None:
        # Client-tool pause, then submit_input → PauseStatusMismatchError.
        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc])])
        agent = _client_tool_agent(llm, db_store)
        paused = await agent.run("read")

        with pytest.raises(PauseStatusMismatchError):
            await agent.submit_input(paused.run_id, "answer")

    async def test_missing_run_raises_not_found(self, db_store) -> None:
        llm = MockLLM([])
        agent = _client_tool_agent(llm, db_store)
        with pytest.raises(RunNotFoundError):
            await agent.submit_input("nope", "text")

    async def test_running_run_raises_not_paused(self, db_store) -> None:
        """Submit on a running (never-paused) run → RunNotPausedError, not Claimed."""
        await db_store.create_run(
            "r-running",
            "TestAgent",
            input_data={"input": "x"},
            model="mock",
            strategy="NativeToolCalling",
        )
        await _force_running(db_store, "r-running")

        llm = MockLLM([])
        agent = _client_tool_agent(llm, db_store)
        with pytest.raises(RunNotPausedError):
            await agent.submit_input("r-running", "text")


# ---------------------------------------------------------------------------
# submit_approval
# ---------------------------------------------------------------------------


class TestSubmitApproval:
    async def test_approve_path_executes_tool(self, db_store) -> None:
        tc = ToolCall(name="refund", params={"order_id": 7}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="processed")])
        agent = _approval_agent(llm, db_store)
        paused = await agent.run("refund 7")
        assert paused.status == RunStatus.WAITING_APPROVAL

        result = await agent.submit_approval(paused.run_id, approved=True)
        assert result.status == RunStatus.SUCCESS
        assert result.answer == "processed"

    async def test_reject_path_feeds_rejection(self, db_store) -> None:
        tc = ToolCall(name="refund", params={"order_id": 7}, provider_tool_call_id="t1")
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[tc]),
                LLMResponse(text="acknowledged rejection"),
            ]
        )
        agent = _approval_agent(llm, db_store)
        paused = await agent.run("refund 7")
        assert paused.status == RunStatus.WAITING_APPROVAL

        result = await agent.submit_approval(
            paused.run_id, approved=False, rejection_reason="Not authorized."
        )
        assert result.status == RunStatus.SUCCESS
        # The LLM's second call should have received a failed tool result.
        second_call = llm.call_history[1]
        tool_msgs = [m for m in second_call["messages"] if m.role.value == "tool"]
        assert tool_msgs and "Not authorized." in tool_msgs[0].content

    async def test_pause_status_mismatch(self, db_store) -> None:
        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc])])
        agent = _client_tool_agent(llm, db_store)
        paused = await agent.run("read")
        with pytest.raises(PauseStatusMismatchError):
            await agent.submit_approval(paused.run_id, approved=True)

    async def test_missing_run_raises_not_found(self, db_store) -> None:
        llm = MockLLM([])
        agent = _approval_agent(llm, db_store)
        with pytest.raises(RunNotFoundError):
            await agent.submit_approval("nope", approved=True)

    async def test_running_run_approved_raises_not_paused(self, db_store) -> None:
        """submit_approval(approved=True) on a running run → RunNotPausedError."""
        await db_store.create_run(
            "r-running-approve",
            "TestAgent",
            input_data={"input": "x"},
            model="mock",
            strategy="NativeToolCalling",
        )
        await _force_running(db_store, "r-running-approve")

        llm = MockLLM([])
        agent = _approval_agent(llm, db_store)
        with pytest.raises(RunNotPausedError):
            await agent.submit_approval("r-running-approve", approved=True)


# ---------------------------------------------------------------------------
# cancel_run
# ---------------------------------------------------------------------------


class TestCancelRun:
    async def test_cancel_paused_run(self, db_store) -> None:
        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc])])
        agent = _client_tool_agent(llm, db_store)
        paused = await agent.run("read")

        result = await agent.cancel_run(paused.run_id)
        assert result.status == RunStatus.CANCELLED

    async def test_cancel_terminal_is_noop(self, db_store) -> None:
        llm = MockLLM([LLMResponse(text="done")])
        agent = _client_tool_agent(llm, db_store)
        completed = await agent.run("hi")
        assert completed.status == RunStatus.SUCCESS

        result = await agent.cancel_run(completed.run_id)
        assert result.status == RunStatus.SUCCESS  # unchanged

    async def test_cancel_missing_raises(self, db_store) -> None:
        llm = MockLLM([])
        agent = _client_tool_agent(llm, db_store)
        with pytest.raises(RunNotFoundError):
            await agent.cancel_run("nope")

    async def test_cancel_no_persistence_raises(self) -> None:
        llm = MockLLM([])
        agent = Agent(provider=llm, prompt="x", tools=[read_range])
        with pytest.raises(PersistenceNotConfiguredError):
            await agent.cancel_run("run-id")

    async def test_cancel_cancels_in_process_task(self, db_store) -> None:
        """A tracked in-process task gets cancelled and the run is CAS-finalized."""
        import asyncio

        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc])])
        agent = _client_tool_agent(llm, db_store)
        paused = await agent.run("read")
        run_id = paused.run_id

        # Simulate a long-running resume by parking a sleeper task into the
        # agent's task manager at the same key cancel_run() will check.
        cancel_observed = asyncio.Event()

        async def fake_resume() -> None:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancel_observed.set()
                raise

        task = agent._task_manager.spawn(run_id, fake_resume())
        await asyncio.sleep(0)  # let the sleep park the task

        result = await agent.cancel_run(run_id)
        assert result.status == RunStatus.CANCELLED
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert cancel_observed.is_set()
        assert not agent._task_manager.is_running(run_id)


# ---------------------------------------------------------------------------
# Race-safety
# ---------------------------------------------------------------------------


class TestConcurrentSubmits:
    async def test_concurrent_submits_one_wins(self, db_store) -> None:
        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="done")])
        agent = _client_tool_agent(llm, db_store)
        paused = await agent.run("read")
        pause = await db_store.get_pause_state(paused.run_id)
        call_id = pause["pending_tool_calls"][0]["id"]

        results = [ToolResult(name="read_range", call_id=call_id, payload='"x"')]
        a, b = await asyncio.gather(
            agent.submit_tool_results(paused.run_id, results),
            agent.submit_tool_results(paused.run_id, results),
            return_exceptions=True,
        )
        successes = [r for r in (a, b) if not isinstance(r, Exception)]
        failures = [r for r in (a, b) if isinstance(r, Exception)]
        assert len(successes) == 1
        assert len(failures) == 1
        assert isinstance(failures[0], (RunAlreadyClaimedError, RunAlreadyTerminalError))


# ---------------------------------------------------------------------------
# RunNotPausedError sanity
# ---------------------------------------------------------------------------


class TestRunNotPausedError:
    def test_fields_preserved(self) -> None:
        err = RunNotPausedError("r1", RunStatus.RUNNING)
        assert err.run_id == "r1"
        assert err.current_status == RunStatus.RUNNING
