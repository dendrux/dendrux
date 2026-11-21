"""Oversized batches refuse every call with actionable, payload-free diagnostics."""

from __future__ import annotations

import json

import pytest

from dendrux import tool
from dendrux.db.models import ToolCallRecord
from dendrux.loops import react
from dendrux.runtime.context import get_delegation_context
from dendrux.types import Limits, LLMResponse, RunEventType, RunStatus, ToolCall, ToolResult
from tests._helpers.fanout_effects import FanoutEffects
from tests._helpers.recovery_checkpoint import expire
from tests._helpers.scripted_provider import ScriptedLLM
from tests.integration.test_fanout_stream import collect, lead, worker
from tests.integration.test_managed_adoption import ProcessDeathError, root_id_of, rows
from tests.integration.test_managed_routing import patch_questions

pytestmark = pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)

BRIEF = "SENTINEL-private-fanout-task-7128"
ENTRIES = [
    "run",
    "stream",
    "resume",
    "resume_stream",
    "submit_tool_results",
    "submit_input",
    "submit_approval",
    "submit_resume",
]


def refusal_text(count, limit):
    return (
        f"Refused: {count} subagent calls in one response; the limit is {limit}. "
        f"Call at most {limit} subagents at once."
    )


async def continue_at(store, root, entry, paused):
    kwargs = {"tenant_id": "t1", "expected_pause_id": paused.pause_id}
    if entry == "submit_input":
        return await root.submit_input(paused.run_id, "answer", **kwargs), []
    if entry == "submit_approval":
        return await root.submit_approval(paused.run_id, approved=True, **kwargs), []
    if entry == "submit_resume":
        return await root.submit_resume(paused.run_id, **kwargs), []
    raw = await store.get_pause_state(paused.run_id)
    replies = [ToolResult("read", raw["pending_tool_calls"][0]["id"], '"answer"')]
    if entry == "resume_stream":
        events = await collect(root.resume_stream(paused.run_id, tool_results=replies, **kwargs))
        return events[-1].run_result, events
    if entry == "resume":
        return await root.resume(paused.run_id, tool_results=replies, **kwargs), []
    return await root.submit_tool_results(paused.run_id, replies, **kwargs), []


async def assert_refusal(store, engine, result, llm, calls, limit, caplog, events):
    ids = {c.id for c in calls}
    records = await rows(engine, ToolCallRecord, ToolCallRecord.agent_run_id == result.run_id)
    records = [r for r in records if r.tool_call_id in ids]
    messages = [m for m in llm.messages[-1] if m.role.value == "tool" and m.call_id in ids]
    stored_events = [
        e
        for e in await store.get_run_events(result.run_id)
        if e.correlation_id in ids and e.event_type == "tool.completed"
    ]
    logs = [
        r
        for r in caplog.records
        if getattr(r, "run_id", None) == result.run_id
        and getattr(r, "code", "") in ("managed_batch_unsupported", "managed_fanout_limit")
    ]
    expected_text = refusal_text(len(calls), limit)
    assert len(records) == len(messages) == len(stored_events) == len(calls)
    assert {
        "stored_codes": {r.meta.get("code") for r in records},
        "stored_errors": {r.error_message for r in records},
        "model_errors": {json.loads(m.content)["error"] for m in messages},
        "event_codes": {e.data.get("code") for e in stored_events},
        "log_codes": {r.code for r in logs},
    } == {
        "stored_codes": {"managed_fanout_limit"},
        "stored_errors": {expected_text},
        "model_errors": {expected_text},
        "event_codes": {"managed_fanout_limit"},
        "log_codes": {"managed_fanout_limit"},
    }
    assert logs and all(r.run_id == result.run_id for r in logs)
    for message in messages:
        assert json.loads(message.content) == {
            "error": expected_text,
            "reason": "fanout_limit",
            "limit": limit,
            "requested": len(calls),
        }
        assert BRIEF not in message.content
    assert BRIEF not in caplog.text
    streamed = [e.tool_result for e in events if e.type == RunEventType.TOOL_RESULT]
    if events:
        assert len(streamed) == len(calls)
        assert all(r.meta["code"] == "managed_fanout_limit" for r in streamed)
        assert all(r.payload == "" and r.error is None for r in streamed)
    assert await store.list_runs(parent_run_id=result.run_id, limit=20) == []


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("limit", [2, None], ids=["limit2", "default4"])
@pytest.mark.parametrize("over", [True, False], ids=["over", "exact"])
async def test_fanout_limit_on_every_entry(store, engine, monkeypatch, caplog, entry, limit, over):
    effective = limit or 4
    count = effective + int(over)
    effects = FanoutEffects()
    children = [
        worker(f"worker{i}", effects.provider(str(i), ScriptedLLM([LLMResponse(text="ok")])))
        for i in range(count)
    ]
    calls = [ToolCall(c.name, {"task": BRIEF}) for c in children]
    holder = {}

    @tool(target="client")
    async def read() -> str:
        return ""

    @tool()
    async def approve() -> str:
        return "approved"

    @tool()
    async def pause() -> str:
        await holder["root"].request_pause(get_delegation_context().run_id)
        return "paused"

    first_name = {"submit_approval": "approve", "submit_resume": "pause"}.get(entry, "read")
    first = (
        [] if entry in ("run", "stream") else [LLMResponse(tool_calls=[ToolCall(first_name, {})])]
    )
    if entry == "submit_input":
        patch_questions(monkeypatch)
        first = [LLMResponse(text="Which?")]
    llm = ScriptedLLM([*first, LLMResponse(tool_calls=calls), LLMResponse(text="done")])
    root = lead(
        store,
        llm,
        children,
        tools=[read, approve, pause],
        require_approval=["approve"],
        limits=None if limit is None else Limits(max_parallel_subagents=limit),
    )
    holder["root"] = root
    try:
        events = []
        if entry == "stream":
            events = await collect(root.stream("go", tenant_id="t1"))
            result = events[-1].run_result
        else:
            result = await root.run("go", tenant_id="t1")
            if entry != "run":
                assert result.status in (
                    RunStatus.WAITING_CLIENT_TOOL,
                    RunStatus.WAITING_HUMAN_INPUT,
                    RunStatus.WAITING_APPROVAL,
                    RunStatus.PAUSED,
                )
                result, events = await continue_at(store, root, entry, result)
        assert result.status == RunStatus.SUCCESS and result.answer == "done"
        if over:
            await assert_refusal(store, engine, result, llm, calls, effective, caplog, events)
            assert all(effects.counts().provider_calls(str(i)) == 0 for i in range(count))
        else:
            assert all(effects.counts().provider_calls(str(i)) == 1 for i in range(count))
            rows = await store.get_tool_calls(result.run_id, limit=20)
            managed = [r for r in rows if r.tool_call_id in {c.id for c in calls}]
            assert len(managed) == count and all(r.success for r in managed)
    finally:
        await root.close()
        for child in children:
            await child.close()


@pytest.mark.parametrize("completed", [0, 1], ids=["before_refusal", "partial_refusal"])
@pytest.mark.parametrize("limit", [2, 4])
async def test_recovered_over_limit_batch(store, engine, monkeypatch, caplog, completed, limit):
    child_llm = ScriptedLLM([])
    child = worker("reviewer", child_llm)
    calls = [ToolCall(child.name, {"task": BRIEF}) for _ in range(limit + 1)]
    llm = ScriptedLLM([LLMResponse(tool_calls=calls), LLMResponse(text="done")])
    root = lead(store, llm, [child], limits=Limits(max_parallel_subagents=limit))
    original = react._ToolBatch.reject

    async def die(*args, **kwargs):
        raise ProcessDeathError

    async def partial(batch, call, result):
        await original(batch, call, result)
        raise ProcessDeathError

    try:
        with monkeypatch.context() as changes:
            if completed:
                changes.setattr(react._ToolBatch, "reject", partial)
            else:
                changes.setattr(react, "_refuse_managed_batch", die)
            with pytest.raises(ProcessDeathError):
                await root.run("go", tenant_id="t1")
        run_id = await root_id_of(engine)
        before = await store.get_tool_calls(run_id, limit=20)
        assert len(before) == completed
        caplog.clear()
        result = await root.reclaim(
            run_id, tenant_id="t1", item=await expire(store, engine, run_id)
        )
        assert result.status == RunStatus.SUCCESS and result.answer == "done"
        await assert_refusal(store, engine, result, llm, calls, limit, caplog, [])
        assert child_llm.calls == 0
    finally:
        await root.close()
        await child.close()
