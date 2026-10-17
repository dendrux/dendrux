"""Run inbox: submit_note acceptance, delivery, ordering and races.

SQLite + Postgres via the ``store`` matrix fixture; tests that open
concurrent sessions use ``sqlite_file`` (independent connections) instead of
in-memory SQLite.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from dendrux.agent import Agent
from dendrux.errors import RunAlreadyTerminalError, RunNotFoundError
from dendrux.runtime.context import get_delegation_context
from dendrux.store import RunStore
from dendrux.tool import tool
from dendrux.types import LLMResponse, RunStatus, ToolResult
from tests._helpers.scripted_provider import ScriptedLLM, tool_call

pytestmark = pytest.mark.asyncio

_CONCURRENT = pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)


def _notes_in(messages: list[Any]) -> list[str]:
    return [m.content for m in messages if getattr(m, "kind", None) == "inbox_note"]


@tool()
async def step() -> str:
    return "stepped"


@tool(target="client")
async def client_read() -> str:
    return ""


class TestDelivery:
    async def test_note_delivered_at_next_iteration(self, store) -> None:
        llm = ScriptedLLM([tool_call("step", pid="t1"), LLMResponse(text="done")], gate_at=0)
        agent = Agent(provider=llm, prompt="Test.", tools=[step], state_store=store)
        task = asyncio.create_task(agent.run("go"))
        await asyncio.wait_for(llm.entered.wait(), 5)
        run_id = (await store.list_runs())[0].id

        note_id = await agent.submit_note(run_id, "use the EU region", metadata={"by": "ops"})
        llm.gate.set()
        result = await asyncio.wait_for(task, 5)

        assert result.status == RunStatus.SUCCESS
        assert "undelivered_notes" not in result.meta
        # Not visible to the in-flight call; delivered before the next one.
        assert _notes_in(llm.messages[0]) == []
        assert _notes_in(llm.messages[1]) == ["NOTE FROM USER:\nuse the EU region"]
        note_msg = next(m for m in llm.messages[1] if m.kind == "inbox_note")
        assert note_msg.role.value == "user"
        assert note_msg.source == "run_inbox"
        # Metadata is never rendered into model content.
        assert all("ops" not in (m.content or "") for m in llm.messages[1])

        notes = await RunStore(store).get_notes(run_id)
        assert [(n.note_id, n.delivered_iteration) for n in notes] == [(note_id, 2)]
        assert notes[0].metadata == {"by": "ops"}
        assert notes[0].delivered_at is not None

        traces = await store.get_traces(run_id)
        note_traces = [t for t in traces if t.meta and t.meta.get("note_id") == note_id]
        assert len(note_traces) == 1
        assert note_traces[0].role == "user"

        delivered = [
            e for e in await store.get_run_events(run_id) if e.event_type == "inbox.delivered"
        ]
        assert len(delivered) == 1
        assert delivered[0].data == {"note_id": note_id}  # no note text in events
        assert delivered[0].correlation_id == note_id

    @pytest.mark.parametrize(
        "status",
        ["waiting_client_tool", "waiting_human_input", "waiting_approval", "paused"],
    )
    async def test_note_on_every_pause_status_waits_until_resume(self, store, status) -> None:
        holder: dict[str, Agent] = {}

        @tool()
        async def guarded() -> str:
            return "ok"

        @tool()
        async def pauser() -> str:
            ctx = get_delegation_context()
            assert ctx is not None
            await holder["agent"].request_pause(ctx.run_id)
            return "ok"

        first = {
            "waiting_client_tool": tool_call("client_read", pid="t1"),
            "waiting_human_input": LLMResponse(text="Which region?", tool_calls=None),
            "waiting_approval": tool_call("guarded", pid="t1"),
            "paused": tool_call("pauser", pid="t1"),
        }[status]
        llm = ScriptedLLM([first, LLMResponse(text="done")])
        agent = Agent(
            provider=llm,
            prompt="Test.",
            tools=[client_read, guarded, pauser],
            require_approval=["guarded"],
            state_store=store,
        )
        holder["agent"] = agent
        if status == "waiting_human_input":
            from dendrux.strategies.native import NativeToolCalling

            # Force a clarification turn from the scripted text.
            original = NativeToolCalling.parse_response

            def parse(self, response):  # type: ignore[no-untyped-def]
                from dendrux.types import AgentStep, Clarification

                if response.text == "Which region?":
                    return AgentStep(reasoning=None, action=Clarification(question=response.text))
                return original(self, response)

            NativeToolCalling.parse_response = parse  # type: ignore[method-assign]
            try:
                paused = await agent.run("go")
            finally:
                NativeToolCalling.parse_response = original  # type: ignore[method-assign]
        else:
            paused = await agent.run("go")
        assert paused.status.value == status

        note_id = await agent.submit_note(paused.run_id, "heads up")
        assert (await store.get_notes(paused.run_id))[0].delivered_at is None

        if status == "waiting_client_tool":
            call_id = paused.meta["pause_state"].pending_tool_calls[0].id
            result = await agent.submit_tool_results(
                paused.run_id, [ToolResult(name="client_read", call_id=call_id, payload='"x"')]
            )
        elif status == "waiting_human_input":
            result = await agent.submit_input(paused.run_id, "EU")
        elif status == "waiting_approval":
            result = await agent.submit_approval(paused.run_id, approved=True)
        else:
            result = await agent.submit_resume(paused.run_id)

        assert result.status == RunStatus.SUCCESS
        assert _notes_in(llm.messages[1]) == ["NOTE FROM USER:\nheads up"]
        notes = await store.get_notes(paused.run_id)
        assert [(n.id, n.delivered_iteration) for n in notes] == [(note_id, 2)]

    async def test_note_during_final_call_is_reported_undelivered(self, store) -> None:
        llm = ScriptedLLM([LLMResponse(text="done")], gate_at=0)
        agent = Agent(provider=llm, prompt="Test.", state_store=store)
        task = asyncio.create_task(agent.run("go"))
        await asyncio.wait_for(llm.entered.wait(), 5)
        run_id = (await store.list_runs())[0].id
        note_id = await agent.submit_note(run_id, "too late")
        llm.gate.set()
        result = await asyncio.wait_for(task, 5)

        assert result.status == RunStatus.SUCCESS
        assert result.meta["undelivered_notes"] == [note_id]
        notes = await RunStore(store).get_notes(run_id, undelivered_only=True)
        assert [n.note_id for n in notes] == [note_id]
        with pytest.raises(RunAlreadyTerminalError):
            await agent.submit_note(run_id, "after the end")

    async def test_guardrails_apply_to_note_text(self, store) -> None:
        from dendrux.guardrails import PII

        llm = ScriptedLLM([tool_call("step", pid="t1"), LLMResponse(text="done")], gate_at=0)
        agent = Agent(
            provider=llm,
            prompt="Test.",
            tools=[step],
            state_store=store,
            guardrails=[PII(action="redact")],
        )
        task = asyncio.create_task(agent.run("go"))
        await asyncio.wait_for(llm.entered.wait(), 5)
        run_id = (await store.list_runs())[0].id
        await agent.submit_note(run_id, "email me at jane.doe@example.com")
        llm.gate.set()
        await asyncio.wait_for(task, 5)
        sent = " ".join(llm.contents(1))
        assert "jane.doe@example.com" not in sent
        assert "NOTE FROM USER:" in sent


class TestValidation:
    async def test_rejections(self, store) -> None:
        agent = Agent(provider=ScriptedLLM([]), prompt="Test.", state_store=store)
        await store.create_run("r1", "a", tenant_id="t1")
        with pytest.raises(ValueError):
            await agent.submit_note("r1", "")
        with pytest.raises(ValueError):
            await agent.submit_note("r1", "é" * (16 * 1024 + 1))  # > 32 KiB as UTF-8
        with pytest.raises(ValueError):
            await agent.submit_note("r1", "x", metadata={"bad": object()})
        with pytest.raises(ValueError):
            await agent.submit_note("r1", "x", metadata={"big": "y" * 9000})
        with pytest.raises(RunNotFoundError):
            await agent.submit_note("missing", "x")
        with pytest.raises(RunNotFoundError):
            await agent.submit_note("r1", "x", tenant_id="t2")
        with pytest.raises(RunNotFoundError):
            await agent.submit_note("r1", "x", tenant_id=None)
        assert await store.get_notes("r1") == []

        assert await agent.submit_note("r1", "x" * (32 * 1024), tenant_id="t1")
        await store.finalize_run("r1", status="success")
        with pytest.raises(RunAlreadyTerminalError):
            await agent.submit_note("r1", "x", tenant_id="t1")


@_CONCURRENT
class TestConcurrency:
    async def test_concurrent_submitters_each_delivered_once_in_order(self, store) -> None:
        llm = ScriptedLLM([tool_call("client_read", pid="t1"), LLMResponse(text="done")])
        agent = Agent(provider=llm, prompt="Test.", tools=[client_read], state_store=store)
        other = Agent(provider=ScriptedLLM([]), prompt="Other.", state_store=store)
        waiting = await agent.run("go")

        ids = await asyncio.gather(
            *(
                (agent if i % 2 else other).submit_note(waiting.run_id, f"note {i}")
                for i in range(10)
            )
        )
        assert len(set(ids)) == 10
        stored = await store.get_notes(waiting.run_id)
        assert sorted(n.id for n in stored) == sorted(ids)
        order = [n.text for n in stored]

        call_id = waiting.meta["pause_state"].pending_tool_calls[0].id
        result = await agent.submit_tool_results(
            waiting.run_id, [ToolResult(name="client_read", call_id=call_id, payload='"x"')]
        )
        assert result.status == RunStatus.SUCCESS
        delivered = _notes_in(llm.messages[1])
        assert delivered == [f"NOTE FROM USER:\n{t}" for t in order]
        assert all(n.delivered_iteration == 2 for n in await store.get_notes(waiting.run_id))
        events = [
            e
            for e in await store.get_run_events(waiting.run_id)
            if e.event_type == "inbox.delivered"
        ]
        assert sorted(e.data["note_id"] for e in events) == sorted(ids)

    async def test_note_vs_terminal_race_never_loses_an_accepted_note(self, store) -> None:
        agent = Agent(provider=ScriptedLLM([]), prompt="Test.", state_store=store)
        for i in range(15):
            run_id = f"race-{i:02d}"
            await store.create_run(run_id, "a")

            async def note(rid: str = run_id) -> str | None:
                try:
                    return await agent.submit_note(rid, "late")
                except RunAlreadyTerminalError:
                    return None

            note_id, _ = await asyncio.gather(
                note(),
                store.finalize_run(run_id, status="success", expected_current_status="running"),
            )
            undelivered = [n.id for n in await store.get_notes(run_id, undelivered_only=True)]
            # Accepted ⇒ visible as undelivered after the terminal write;
            # rejected ⇒ nothing stored.
            assert undelivered == ([note_id] if note_id else [])
