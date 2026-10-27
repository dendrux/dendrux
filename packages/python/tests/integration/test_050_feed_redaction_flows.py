"""Redacted feeds over real agent runs (SQLite + Postgres).

A finished run's ``RunStore.follow(include_payloads=False)`` and
``get_event_view`` never show a past approval's arguments, a person's answer
to a question, or a reviewer's denial reason. The trusted view
(``include_payloads=True``) still has them.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from dendrux.agent import Agent
from dendrux.guardrails import PII
from dendrux.store import FollowEnd, FollowHeartbeat, RunStore, StoredEvent
from dendrux.tool import tool
from dendrux.types import ApprovalDecision, LLMResponse
from tests._helpers.scripted_provider import ScriptedLLM, tool_call

pytestmark = pytest.mark.asyncio

SQL = "SECRET-SQL: select * from salaries where ceo='x'"
ANSWER = "use bob.private@example.org please, and password hunter2"
REASON = "Denied: patient Jane Roe SSN 123-45-6789 is not authorised"


@tool()
async def run_sql(q: str) -> str:
    return "rows"


@tool()
async def add(a: int, b: int) -> int:
    return a + b


async def _follow(run_store: RunStore, run_id: str, **kw: Any) -> list[StoredEvent]:
    out: list[StoredEvent] = []
    async with run_store.follow(run_id, tenant_id="A", max_duration_s=10.0, **kw) as f:
        async for item in f:
            if isinstance(item, FollowEnd):
                break
            if not isinstance(item, FollowHeartbeat):
                out.append(item)
    return out


def _text(events: list[StoredEvent]) -> str:
    return json.dumps([e.data for e in events], default=str)


async def test_past_approval_args_hidden_after_rejection(store: Any, engine: Any) -> None:
    run_store = RunStore.from_engine(engine)
    llm = ScriptedLLM([tool_call("run_sql", {"q": SQL}, pid="a"), LLMResponse(text="done")])
    agent = Agent(
        provider=llm,
        prompt="t",
        state_store=store,
        tools=[run_sql],
        require_approval=["run_sql"],
        strict_tenancy=False,
    )
    paused = await agent.run("go", tenant_id="A")
    done = await agent.submit_approval(
        paused.run_id,
        approved=False,
        rejection_reason="no",
        tenant_id="A",
        expected_pause_id=paused.pause_id,
    )
    assert done.status.value == "success"
    events = await _follow(run_store, done.run_id)
    assert "approval.requested" in [e.event_type for e in events]
    assert SQL not in _text(events)
    views = [
        await run_store.get_event_view(done.run_id, e, tenant_id="A")
        for e in await run_store.get_events(done.run_id, tenant_id="A")
    ]
    assert SQL not in _text(views)
    assert SQL in _text(await _follow(run_store, done.run_id, include_payloads=True))


async def test_denial_reason_hidden(store: Any, engine: Any) -> None:
    run_store = RunStore.from_engine(engine)
    llm = ScriptedLLM([tool_call("run_sql", {"q": "x"}, pid="a"), LLMResponse(text="done")])
    agent = Agent(
        provider=llm,
        prompt="t",
        state_store=store,
        tools=[run_sql],
        require_approval=["run_sql"],
        strict_tenancy=False,
    )
    paused = await agent.run("go", tenant_id="A")
    assert paused.pending is not None
    call_id = paused.pending.tool_calls[0].id
    await agent.submit_approval(
        paused.run_id,
        decisions={call_id: ApprovalDecision.deny(REASON)},
        tenant_id="A",
        expected_pause_id=paused.pause_id,
    )
    events = await _follow(run_store, paused.run_id)
    decided = [e for e in events if e.event_type == "approval.decided"]
    assert decided and decided[0].data["decisions"][0]["decision"] == "denied"
    assert REASON not in _text(events)
    assert REASON in _text(await _follow(run_store, paused.run_id, include_payloads=True))


async def test_answer_to_a_question_hidden(
    store: Any, engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dendrux.strategies.native import NativeToolCalling
    from dendrux.types import AgentStep, Clarification

    original = NativeToolCalling.parse_response

    def parse(self: Any, response: Any) -> Any:
        if response.text == "Which email?":
            return AgentStep(reasoning=None, action=Clarification(question=response.text))
        return original(self, response)

    monkeypatch.setattr(NativeToolCalling, "parse_response", parse)
    run_store = RunStore.from_engine(engine)
    llm = ScriptedLLM([LLMResponse(text="Which email?"), LLMResponse(text="done")])
    agent = Agent(
        provider=llm,
        prompt="t",
        state_store=store,
        tools=[add],
        guardrails=[PII()],
        strict_tenancy=False,
    )
    paused = await agent.run("hi", tenant_id="A")
    await agent.submit_input(
        paused.run_id, ANSWER, tenant_id="A", expected_pause_id=paused.pause_id
    )
    events = await _follow(run_store, paused.run_id)
    resumed = [e for e in events if e.event_type == "run.resumed"]
    assert resumed and resumed[0].data["resumed_from"] == "waiting_human_input"
    assert "bob.private" not in _text(events) and "hunter2" not in _text(events)
    pairs = await run_store.get_pauses(paused.run_id, tenant_id="A")
    assert pairs[0].user_input is None
