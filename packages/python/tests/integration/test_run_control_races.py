"""Contention between run owners, finalizers and the sweeper (Track E fix round).

Deterministic hook tests, run on ``sqlite_file`` and Postgres (independent
connections). Covers:

- cancel committed right after a pause: one event each, distinct ordered
  sequence indexes, and the runner returns the persisted (cancelled) state;
- a runner whose terminal CAS lost returns the persisted winner;
- cancel requested after the last checkpoint: a finished run completes
  (documented cooperative semantics);
- the stale sweeper honors a pending cancel (CANCELLED, never ERROR);
- concurrent finalizers / sweeper on one paused row: one transition event,
  unique sequence indexes;
- two concurrent resume owners: one wins, notes delivered once;
- cancel_run results list undelivered notes on every terminal path;
- RunStore.get_notes tenant scoping;
- pre-0.3 custom stores keep working without run-control APIs.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import update

from dendrux.agent import Agent
from dendrux.db.models import AgentRun
from dendrux.errors import (
    PauseStatusMismatchError,
    RunAlreadyClaimedError,
    RunNotPausedError,
)
from dendrux.runtime.context import get_delegation_context
from dendrux.store import RunNotFoundError as StoreRunNotFoundError
from dendrux.store import RunStore
from dendrux.tool import tool
from dendrux.types import LLMResponse, RunStatus, ToolResult
from tests._helpers.scripted_provider import ScriptedLLM, tool_call

if TYPE_CHECKING:
    from dendrux.runtime.state import SQLAlchemyStateStore

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True),
]


def _hook_after_pause(store: Any, after: Any) -> None:
    """Run ``after(run_id)`` right after whichever pause primitive the runner uses commits."""
    for name in ("pause_with_event", "try_pause_run"):
        original = getattr(store, name, None)
        if original is None:
            continue

        async def wrapped(run_id: str, _original: Any = original, **kwargs: Any) -> Any:
            won = await _original(run_id, **kwargs)
            if won is not None and won is not False:
                await after(run_id)
            return won

        setattr(store, name, wrapped)


def _pausing_agent(store: Any) -> Agent:
    holder: dict[str, Agent] = {}

    @tool()
    async def step() -> str:
        ctx = get_delegation_context()
        assert ctx is not None
        await holder["agent"].request_pause(ctx.run_id)
        return "ok"

    agent = Agent(
        provider=ScriptedLLM([tool_call("step", pid="t1"), LLMResponse(text="done")]),
        prompt="P.",
        tools=[step],
        state_store=store,
    )
    holder["agent"] = agent
    return agent


def _other(store: Any) -> Agent:
    return Agent(provider=ScriptedLLM([]), prompt="Other.", state_store=store)


async def _paused_run(store: Any) -> str:
    result = await _pausing_agent(store).run("go")
    assert result.status == RunStatus.PAUSED
    return result.run_id


async def _set_row(store: Any, run_id: str, **values: Any) -> None:
    async with store._session_factory() as session:
        await session.execute(update(AgentRun).where(AgentRun.id == run_id).values(**values))
        await session.commit()


def _assert_unique_sequences(events: list[Any]) -> None:
    seqs = [e.sequence_index for e in events]
    assert len(seqs) == len(set(seqs)), [(e.sequence_index, e.event_type) for e in events]


class TestPausePublication:
    async def test_cancel_right_after_pause_commit(self, store) -> None:
        agent = _pausing_agent(store)
        other = _other(store)
        _hook_after_pause(store, other.cancel_run)
        result = await agent.run("go")

        record = await store.get_run(result.run_id)
        assert record.status == "cancelled"
        # The runner returns the persisted winner, not the PAUSED it wrote.
        assert result.status == RunStatus.CANCELLED
        events = await store.get_run_events(result.run_id)
        _assert_unique_sequences(events)
        by_type = {e.event_type: e.sequence_index for e in events}
        assert by_type["run.paused"] < by_type["run.cancelled"]

    async def test_cancel_before_pause_cas_wins(self, store) -> None:
        agent = _pausing_agent(store)
        original = store.pause_with_event

        async def cancel_first(run_id: str, **kwargs: Any) -> Any:
            await store.request_cancel(run_id)
            return await original(run_id, **kwargs)

        store.pause_with_event = cancel_first
        result = await agent.run("go")
        assert result.status == RunStatus.CANCELLED
        events = await store.get_run_events(result.run_id)
        types = [e.event_type for e in events]
        assert "run.paused" not in types and types.count("run.cancelled") == 1
        _assert_unique_sequences(events)

    async def test_waiting_pause_is_published_atomically_too(self, store) -> None:
        @tool(target="client")
        async def client_read() -> str:
            return ""

        agent = Agent(
            provider=ScriptedLLM([tool_call("client_read", pid="t1")]),
            prompt="P.",
            tools=[client_read],
            state_store=store,
        )
        _hook_after_pause(store, _other(store).cancel_run)
        result = await agent.run("go")
        assert result.status == RunStatus.CANCELLED
        events = await store.get_run_events(result.run_id)
        _assert_unique_sequences(events)
        by_type = {e.event_type: e.sequence_index for e in events}
        assert by_type["run.paused"] < by_type["run.cancelled"]


class TestTerminalPrecedence:
    async def test_cancel_after_last_checkpoint_completes_normally(self, store) -> None:
        """Documented: cooperative cancel does not undo a finished run."""
        agent = Agent(
            provider=ScriptedLLM([LLMResponse(text="done")]), prompt="P.", state_store=store
        )
        original = store.finalize_run

        async def cancel_then_finalize(run_id: str, **kwargs: Any) -> bool:
            if kwargs.get("status") == "success":
                await store.request_cancel(run_id)
            return await original(run_id, **kwargs)

        store.finalize_run = cancel_then_finalize
        result = await agent.run("go")
        record = await store.get_run(result.run_id)
        assert result.status == RunStatus.SUCCESS and record.status == "success"
        assert record.cancel_requested is False

    async def test_lost_terminal_cas_returns_persisted_state(self, store) -> None:
        agent = Agent(
            provider=ScriptedLLM([LLMResponse(text="done")]), prompt="P.", state_store=store
        )
        original = store.finalize_run

        async def swept_first(run_id: str, **kwargs: Any) -> bool:
            if kwargs.get("status") == "success":
                await _set_row(store, run_id, status="error", failure_reason="stale_running")
            return await original(run_id, **kwargs)

        store.finalize_run = swept_first
        result = await agent.run("go")
        assert result.status == RunStatus.ERROR
        assert result.meta["_finalize_won"] is False
        events = [e.event_type for e in await store.get_run_events(result.run_id)]
        assert "run.completed" not in events

    async def test_sweeper_honors_pending_cancel(self, store) -> None:
        await store.create_run("stale-c", "a")
        await store.create_run("stale-e", "a")
        old = dt.datetime(2000, 1, 1, tzinfo=dt.UTC)
        for rid in ("stale-c", "stale-e"):
            await _set_row(store, rid, last_progress_at=old)
        await store.request_cancel("stale-c")

        swept = await store.sweep_stale_runs(dt.timedelta(minutes=1))
        by_id = {s.run_id: s for s in swept}
        assert by_id["stale-c"].status == "cancelled"
        assert by_id["stale-e"].status == "error"
        cancelled = await store.get_run("stale-c")
        assert cancelled.status == "cancelled"
        assert cancelled.failure_reason == "never_started"
        assert cancelled.error is None
        assert cancelled.cancel_requested is False
        events = await store.get_run_events("stale-c")
        assert [(e.event_type, e.data) for e in events] == [
            ("run.cancelled", {"reason": "cancel_requested", "failure_reason": "never_started"})
        ]
        assert (await store.get_run("stale-e")).status == "error"


class TestFinalizerContention:
    async def test_concurrent_cancels_one_event_unique_sequences(self, store) -> None:
        run_id = await _paused_run(store)
        results = await asyncio.gather(*(_other(store).cancel_run(run_id) for _ in range(6)))
        assert all(r.status == RunStatus.CANCELLED for r in results)
        events = await store.get_run_events(run_id)
        assert [e.event_type for e in events].count("run.cancelled") == 1
        _assert_unique_sequences(events)

    async def test_cancel_vs_abandoned_sweep_one_transition(self, store) -> None:
        run_id = await _paused_run(store)
        await _set_row(store, run_id, updated_at=dt.datetime(2000, 1, 1, tzinfo=dt.UTC))
        await asyncio.gather(
            _other(store).cancel_run(run_id),
            store.sweep_abandoned_runs(older_than=dt.timedelta(minutes=1)),
        )
        events = await store.get_run_events(run_id)
        terminal = [e for e in events if e.event_type in ("run.cancelled", "run.abandoned")]
        assert len(terminal) == 1
        _assert_unique_sequences(events)
        status = (await store.get_run(run_id)).status
        assert status == ("cancelled" if terminal[0].event_type == "run.cancelled" else "error")

    async def test_two_concurrent_resume_owners_deliver_notes_once(self, store) -> None:
        holder: dict[str, Agent] = {}

        @tool()
        async def step() -> str:
            ctx = get_delegation_context()
            assert ctx is not None
            await holder["agent"].request_pause(ctx.run_id)
            return "ok"

        llms = [
            ScriptedLLM([tool_call("step", pid="t1"), LLMResponse(text="done")]),
            ScriptedLLM([LLMResponse(text="done")]),
            ScriptedLLM([LLMResponse(text="done")]),
        ]
        first = Agent(provider=llms[0], prompt="P.", tools=[step], state_store=store)
        holder["agent"] = first
        paused = await first.run("go")
        await first.submit_note(paused.run_id, "one")
        await first.submit_note(paused.run_id, "two")

        owners = [
            Agent(provider=llm, prompt="P.", tools=[step], state_store=store) for llm in llms[1:]
        ]
        outcomes = await asyncio.gather(
            *(o.submit_resume(paused.run_id) for o in owners), return_exceptions=True
        )
        wins = [o for o in outcomes if not isinstance(o, BaseException)]
        losses = [o for o in outcomes if isinstance(o, BaseException)]
        assert len(wins) == 1 and wins[0].status == RunStatus.SUCCESS
        assert len(losses) == 1
        assert isinstance(
            losses[0], RunAlreadyClaimedError | RunNotPausedError | PauseStatusMismatchError
        )
        delivered = [
            m.content
            for llm in llms[1:]
            for call in llm.messages
            for m in call
            if m.kind == "inbox_note"
        ]
        assert delivered == ["NOTE FROM USER:\none", "NOTE FROM USER:\ntwo"]
        events = await store.get_run_events(paused.run_id)
        assert [e.event_type for e in events].count("inbox.delivered") == 2
        _assert_unique_sequences(events)


class TestCancelRunNotes:
    @pytest.mark.parametrize(
        "status", ["paused", "waiting_client_tool", "waiting_human_input", "waiting_approval"]
    )
    async def test_cancel_lists_undelivered_notes(self, store, status) -> None:
        await store.create_run("r1", "a")
        await store.try_pause_run("r1", status=status, pause_data={"agent_name": "a"})
        await store.add_note("r1", note_id="note-1", text="never seen")
        agent = _other(store)
        result = await agent.cancel_run("r1")
        assert result.status == RunStatus.CANCELLED
        assert result.meta["undelivered_notes"] == ["note-1"]
        # The no-op path on the now-terminal run reports them too.
        again = await agent.cancel_run("r1")
        assert again.meta["undelivered_notes"] == ["note-1"]

    async def test_running_cancel_result_has_no_note_list(self, store) -> None:
        await store.create_run("r1", "a")
        await store.add_note("r1", note_id="note-1", text="pending")
        result = await _other(store).cancel_run("r1")
        assert result.status == RunStatus.RUNNING
        assert "undelivered_notes" not in result.meta


class TestGetNotesTenancy:
    async def test_scoped_reads(self, store) -> None:
        await store.create_run("t", "a", tenant_id="acme")
        await store.create_run("u", "a")
        await store.add_note("t", note_id="n-t", text="tenant note", metadata={"k": 1})
        await store.add_note("u", note_id="n-u", text="untenanted note")
        rs = RunStore(store)

        assert [n.note_id for n in await rs.get_notes("t")] == ["n-t"]  # unscoped
        assert [n.note_id for n in await rs.get_notes("t", tenant_id="acme")] == ["n-t"]
        with pytest.raises(StoreRunNotFoundError):
            await rs.get_notes("t", tenant_id="other")
        with pytest.raises(StoreRunNotFoundError):
            await rs.get_notes("t", tenant_id=None)
        assert [n.note_id for n in await rs.get_notes("u", tenant_id=None)] == ["n-u"]
        with pytest.raises(StoreRunNotFoundError):
            await rs.get_notes("u", tenant_id="acme")
        with pytest.raises(StoreRunNotFoundError):
            await rs.get_notes("missing", tenant_id="acme")
        # The store query applies the same predicate.
        assert await store.get_notes("t", tenant_id="other") == []
        assert await store.get_notes("t", tenant_id=None) == []


# ---------------------------------------------------------------------------
# Pre-0.3 custom store compatibility
# ---------------------------------------------------------------------------

_PRE_03_METHODS = frozenset(
    {
        "store_identity",
        "create_run",
        "save_trace",
        "save_tool_call",
        "save_usage",
        "save_llm_interaction",
        "get_llm_interactions",
        "finalize_run",
        "finalize_run_if_status_in",
        "request_cancel",
        "is_cancel_requested",
        "get_next_event_sequence",
        "pause_run",
        "get_pause_state",
        "get_pii_mapping",
        "claim_paused_run",
        "submit_and_claim",
        "get_run",
        "get_traces",
        "get_tool_calls",
        "save_run_event",
        "get_run_events",
        "list_runs",
        "count_runs",
        "touch_progress",
        "sweep_stale_runs",
        "sweep_abandoned_runs",
        "get_delegation_info",
    }
)


class Pre03Store:
    """A custom store exposing exactly the 0.2 StateStore protocol."""

    def __init__(self, inner: SQLAlchemyStateStore) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        if name in _PRE_03_METHODS:
            return getattr(self._inner, name)
        raise AttributeError(name)


class TestPre03Store:
    async def test_existing_flows_work(self, store) -> None:
        legacy = Pre03Store(store)

        @tool()
        async def add(a: int, b: int) -> int:
            return a + b

        @tool(target="client")
        async def client_read() -> str:
            return ""

        @tool()
        async def refund() -> str:
            return "refunded"

        llm = ScriptedLLM(
            [
                tool_call("add", {"a": 1, "b": 2}, pid="t1"),
                LLMResponse(text="3"),
                tool_call("client_read", pid="t2"),
                LLMResponse(text="read"),
                tool_call("refund", pid="t3"),
                LLMResponse(text="refunded"),
                tool_call("client_read", pid="t4"),
            ]
        )
        agent = Agent(
            provider=llm,
            prompt="P.",
            tools=[add, client_read, refund],
            require_approval=["refund"],
            state_store=legacy,  # type: ignore[arg-type]
        )
        assert (await agent.run("add")).answer == "3"

        waiting = await agent.run("read")
        assert waiting.status == RunStatus.WAITING_CLIENT_TOOL
        call_id = waiting.meta["pause_state"].pending_tool_calls[0].id
        done = await agent.submit_tool_results(
            waiting.run_id, [ToolResult(name="client_read", call_id=call_id, payload='"x"')]
        )
        assert done.status == RunStatus.SUCCESS

        approval = await agent.run("refund")
        assert approval.status == RunStatus.WAITING_APPROVAL
        assert (await agent.submit_approval(approval.run_id, approved=True)).answer == "refunded"

        to_cancel = await agent.run("read again")
        cancelled = await agent.cancel_run(to_cancel.run_id)
        assert cancelled.status == RunStatus.CANCELLED
        events = await store.get_run_events(to_cancel.run_id)
        assert [e.event_type for e in events][-1] == "run.cancelled"
        _assert_unique_sequences(events)

    async def test_cancel_between_legacy_check_and_pause_is_honored(self, store) -> None:
        """Legacy path reads the cancel flag, then pauses unconditionally: a
        cancel landing in between must still end the run CANCELLED."""
        legacy = Pre03Store(store)

        @tool(target="client")
        async def client_read() -> str:
            return ""

        agent = Agent(
            provider=ScriptedLLM([tool_call("client_read", pid="t1")]),
            prompt="P.",
            tools=[client_read],
            state_store=legacy,  # type: ignore[arg-type]
        )
        original = store.pause_run

        async def cancel_then_pause(run_id: str, **kwargs: Any) -> None:
            await store.request_cancel(run_id)
            await original(run_id, **kwargs)

        store.pause_run = cancel_then_pause  # type: ignore[method-assign]
        result = await agent.run("read")

        record = await store.get_run(result.run_id)
        assert result.status == RunStatus.CANCELLED
        assert record.status == "cancelled"
        assert record.cancel_requested is False
        events = await store.get_run_events(result.run_id)
        types = [e.event_type for e in events]
        assert types.index("run.paused") < types.index("run.cancelled")
        _assert_unique_sequences(events)

    async def test_run_control_apis_name_the_missing_method(self, store) -> None:
        legacy = Pre03Store(store)
        agent = Agent(provider=ScriptedLLM([]), prompt="P.", state_store=legacy)  # type: ignore[arg-type]
        await store.create_run("r1", "a")
        with pytest.raises(NotImplementedError, match="request_pause"):
            await agent.request_pause("r1", cascade=False)
        with pytest.raises(NotImplementedError, match="add_note"):
            await agent.submit_note("r1", "hi")
        with pytest.raises(NotImplementedError, match="get_notes"):
            await RunStore(legacy).get_notes("r1")  # type: ignore[arg-type]
        with pytest.raises(NotImplementedError, match="get_subtree"):
            await agent.cancel_run("r1", cascade=True)
