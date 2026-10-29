"""Integration tests for ``agent.run(runs=...)`` across the backend matrix."""

from __future__ import annotations

from typing import Any

import pytest

from dendrux import Agent, ReplayRunError, tool
from dendrux.context_blocks import ContextBlock
from dendrux.guardrails import PII
from dendrux.llm.mock import MockLLM
from dendrux.loops.single import SingleCall
from dendrux.runtime.replay import PREFIX_META_KEY, REPLAY_META_KEY
from dendrux.store import RunStore
from dendrux.types import (
    IdempotencyConflictError,
    LLMResponse,
    Role,
    RunEventType,
    RunStatus,
    ToolCall,
)

THINK = [{"type": "thinking", "thinking": "plan", "signature": "sig"}]


@tool()
async def lookup(q: str) -> str:
    """Look something up."""
    return f"result for {q}"


def _tool_turn(q: str, answer: str, *, reasoning: bool = True) -> list[LLMResponse]:
    blocks = THINK if reasoning else None
    return [
        LLMResponse(
            text=None,
            tool_calls=[ToolCall(name="lookup", params={"q": q}, provider_tool_call_id=f"p-{q}")],
            reasoning_blocks=blocks,
        ),
        LLMResponse(text=answer, reasoning_blocks=blocks),
    ]


def _agent(store: Any, mock: MockLLM, **kw: Any) -> Agent:
    kw.setdefault("tools", [lookup])
    return Agent(provider=mock, prompt="You are helpful.", state_store=store, **kw)


async def _chat(store: Any, n: int, replay_last: Any = "all") -> list[str]:
    """Run an n-turn conversation, each turn replaying all earlier runs."""
    run_ids: list[str] = []
    for i in range(n):
        mock = MockLLM(_tool_turn(f"q{i}", f"a{i}"))
        result = await _agent(store, mock).run(
            f"question {i}", runs=list(run_ids), replay_last=replay_last
        )
        assert result.status == RunStatus.SUCCESS
        run_ids.append(result.run_id)
    return run_ids


class TestFullReplay:
    async def test_fourth_turn_sees_stored_turns(self, store):
        run_ids = await _chat(store, 3)
        mock = MockLLM([LLMResponse(text="final")])
        result = await _agent(store, mock).run("question 3", runs=run_ids, replay_last="all")

        sent = mock.call_history[0]["messages"]
        user_msgs = [m.content for m in sent if m.role == Role.USER]
        assert user_msgs == ["question 0", "question 1", "question 2", "question 3"]

        tool_msgs = [m for m in sent if m.role == Role.TOOL]
        assert [m.content for m in tool_msgs] == [
            '"result for q0"',
            '"result for q1"',
            '"result for q2"',
        ]
        for rid, tm in zip(run_ids, tool_msgs, strict=True):
            stored = [
                t
                for t in await store.get_traces(rid)
                if t.role == "tool" and t.trace_kind == "turn"
            ][0]
            assert tm.call_id == stored.meta["call_id"]
            assert tm.name == stored.meta["tool_name"]

        assert sum(1 for m in sent if m.reasoning_blocks) == 6
        assert result.replay is not None
        assert result.replay.runs_full == run_ids
        assert result.replay.reasoning_kept == run_ids
        assert result.replay.warnings == []

        record = await store.get_run(result.run_id)
        assert record.meta[REPLAY_META_KEY]["runs"] == run_ids
        assert record.meta[PREFIX_META_KEY]["seed"]["expansion"][-1]["mode"] == "full"
        # Format 2 retains the replay prefix as seeds and appends its own turn.
        traces = await store.get_traces(result.run_id)
        assert any(t.trace_kind == "seed" and t.role == "tool" for t in traces)
        assert [t.role for t in traces if t.trace_kind == "turn"] == ["assistant"]

    async def test_every_run_records_prefix(self, store):
        mock = MockLLM([LLMResponse(text="hi")])
        result = await _agent(store, mock).run("hello")
        prefix = (await store.get_run(result.run_id)).meta[PREFIX_META_KEY]
        assert prefix["seed"] == {}
        assert prefix["model"] == "mock"
        assert prefix["provider"] == "MockLLM"
        assert set(prefix["tools"]) == {"lookup"}

    async def test_sliding_window_drops_reasoning_with_warning(self, store):
        run_ids = await _chat(store, 3)
        mock = MockLLM([LLMResponse(text="final")])
        result = await _agent(store, mock).run("q", runs=run_ids, replay_last=1)
        assert result.status == RunStatus.SUCCESS
        assert result.replay.runs_text == run_ids[:2]
        assert result.replay.reasoning_dropped == [run_ids[2]]
        assert "seed" in result.replay.warnings[0]
        sent = mock.call_history[0]["messages"]
        assert not any(m.reasoning_blocks for m in sent)

    async def test_history_appended_after_runs(self, store):
        from dendrux.chat import ChatMessage

        run_ids = await _chat(store, 1)
        mock = MockLLM([LLMResponse(text="ok")])
        await _agent(store, mock).run(
            "now",
            runs=run_ids,
            replay_last=0,
            history=[ChatMessage.user("marker"), ChatMessage.assistant("noted")],
        )
        contents = [m.content for m in mock.call_history[0]["messages"] if m.role != Role.SYSTEM]
        assert contents == ["question 0", "a0", "marker", "noted", "now"]

    async def test_dynamic_context_replays_as_sent(self, store):
        mock = MockLLM([LLMResponse(text="a0")])
        r0 = await _agent(store, mock).run("q0", context=[ContextBlock("DOCS")])
        traces = await store.get_traces(r0.run_id)
        assert next(t for t in traces if t.role == "user").content == "q0"

        mock2 = MockLLM([LLMResponse(text="a1")])
        await _agent(store, mock2).run("q1", runs=[r0.run_id], replay_last="all")
        first = mock2.call_history[0]["messages"][1]
        assert first.content == mock.call_history[0]["messages"][1].content
        assert "DOCS" in first.content

    async def test_stable_context_survives_text_replay(self, store):
        mock = MockLLM([LLMResponse(text="a0")])
        r0 = await _agent(store, mock).run(
            "q0", context=[ContextBlock("RULES", placement="stable")]
        )
        mock2 = MockLLM([LLMResponse(text="a1")])
        await _agent(store, mock2).run("q1", runs=[r0.run_id], replay_last=0)
        first = mock2.call_history[0]["messages"][1]
        assert "RULES" in first.content and first.placement == "stable"


class TestValidation:
    async def test_unknown_run(self, store):
        agent = _agent(store, MockLLM([]))
        with pytest.raises(ReplayRunError) as exc:
            await agent.run("q", runs=["nope"])
        assert exc.value.reason == "not_found"
        assert await store.list_runs() == []

    async def test_tenant_mismatch(self, store):
        mock = MockLLM([LLMResponse(text="a")])
        r0 = await _agent(store, mock).run("q", tenant_id="acme")
        with pytest.raises(ReplayRunError) as exc:
            await _agent(store, MockLLM([])).run("q", runs=[r0.run_id], tenant_id="other")
        assert exc.value.reason == "not_found"
        ok = await _agent(store, MockLLM([LLMResponse(text="b")])).run(
            "q", runs=[r0.run_id], tenant_id="acme"
        )
        assert ok.status == RunStatus.SUCCESS

    async def test_not_terminal(self, store):
        await store.create_run("live", "a", input_data={"input": "x"})
        with pytest.raises(ReplayRunError) as exc:
            await _agent(store, MockLLM([])).run("q", runs=["live"])
        assert exc.value.reason == "not_terminal"

    async def test_requires_persistence(self):
        agent = Agent(provider=MockLLM([]), prompt="p")
        with pytest.raises(ValueError, match="requires persistence"):
            await agent.run("q", runs=["r"])

    async def test_singlecall(self, store):
        mock = MockLLM([LLMResponse(text="a")])
        r0 = await Agent(provider=mock, prompt="p", state_store=store, loop=SingleCall()).run("q")
        single = Agent(
            provider=MockLLM([LLMResponse(text="b")]),
            prompt="p",
            state_store=store,
            loop=SingleCall(),
        )
        with pytest.raises(ValueError, match="SingleCall"):
            await single.run("q2", runs=[r0.run_id], replay_last="all")
        ok = await single.run("q2", runs=[r0.run_id])
        assert ok.status == RunStatus.SUCCESS

    async def test_idempotency_includes_knobs(self, store):
        run_ids = await _chat(store, 1)
        agent = _agent(store, MockLLM([LLMResponse(text="x")]))
        await agent.run("q", runs=run_ids, idempotency_key="k")
        with pytest.raises(IdempotencyConflictError):
            await agent.run("q", runs=run_ids, replay_last="all", idempotency_key="k")
        cached = await agent.run("q", runs=run_ids, idempotency_key="k")
        assert cached.replay is not None and cached.replay.runs_text == run_ids


class TestGuardrailsAndRetry:
    async def test_guardrails_force_reasoning_off(self, store):
        run_ids = await _chat(store, 1)
        mock = MockLLM([LLMResponse(text="final")])
        result = await _agent(store, mock, guardrails=[PII()]).run(
            "q", runs=run_ids, replay_last="all"
        )
        assert "forced off" in result.replay.warnings[0]
        assert not any(m.reasoning_blocks for m in mock.call_history[0]["messages"])

    async def test_retry_re_expands_same_runs(self, store):
        run_ids = await _chat(store, 2)
        mock = MockLLM([LLMResponse(text="first try")])
        agent = _agent(store, mock)
        r = await agent.run("q", runs=run_ids, replay_last=1)
        mock2 = MockLLM([LLMResponse(text="retried")])
        retried = await _agent(store, mock2).retry(r.run_id)
        assert retried.replay is not None
        assert retried.replay.runs_text == run_ids[:1]
        assert retried.replay.runs_full == run_ids[1:]
        contents = [m.content for m in mock2.call_history[0]["messages"] if m.role == Role.USER]
        assert contents == ["question 0", "question 1", "q"]


class TestStreamAndStore:
    async def test_stream_path(self, store):
        run_ids = await _chat(store, 1)
        agent = _agent(store, MockLLM([LLMResponse(text="streamed")]))
        events = [e async for e in agent.stream("q", runs=run_ids, replay_last="all")]
        done = events[-1]
        assert done.type == RunEventType.RUN_COMPLETED
        assert done.run_result.replay.runs_full == run_ids

    async def test_stream_bad_run_yields_error(self, store):
        agent = _agent(store, MockLLM([]))
        events = [e async for e in agent.stream("q", runs=["missing"])]
        assert events[-1].type == RunEventType.RUN_ERROR
        assert events[-1].run_result.meta["error_type"] == "ReplayRunError"
        assert await store.list_runs() == []

    async def test_stream_requires_persistence(self, monkeypatch):
        monkeypatch.delenv("DENDRUX_DATABASE_URL", raising=False)
        agent = Agent(provider=MockLLM([]), prompt="p")
        events = [e async for e in agent.stream("q", runs=["r"])]
        assert events[-1].type == RunEventType.RUN_ERROR
        assert "requires persistence" in events[-1].error

    async def test_stream_sync_validation(self, store):
        agent = _agent(store, MockLLM([]))
        with pytest.raises(ValueError, match="stable"):
            agent.stream("q", runs=["r"], context=[ContextBlock("x", placement="stable")])

    async def test_run_store_surfaces_replay(self, store, engine):
        run_ids = await _chat(store, 2)
        rs = RunStore.from_engine(engine)
        summaries = await rs.list_runs()
        by_id = {s.run_id: s for s in summaries}
        assert by_id[run_ids[0]].replay is None
        assert by_id[run_ids[1]].replay.runs_full == run_ids[:1]
        detail = await rs.get_run(run_ids[1])
        assert detail.replay.reasoning_kept == run_ids[:1]
