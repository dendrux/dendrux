"""Persistence regressions from Skills v2 review, on both SQL backends."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from dendrux import Agent
from dendrux.llm.mock import MockLLM
from dendrux.runtime.context import get_delegation_context
from dendrux.skills._session import current
from dendrux.types import LLMResponse, RunStatus, ToolCall, ToolResult
from tests.integration.test_skills_v2 import agent, client
from tests.unit.test_skill_runtime_fix1 import Reads


async def test_script_requested_once_across_approval(store):
    a = agent(
        store,
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        name="run_skill_script",
                        params={"skill": "demo", "path": "scripts/a.py", "args": []},
                    )
                ]
            )
        ],
        require_approval=["run_skill_script"],
    )
    paused = await a.run("input", activate_skills=["demo"])
    b = agent(store, [LLMResponse(text="done")])
    await b.submit_approval(paused.run_id, approved=True, expected_pause_id=paused.pause_id)
    events = await store.get_run_events(paused.run_id)
    requested = [e for e in events if e.event_type == "skill.script_requested"]
    assert len(requested) == 1
    completed = next(e for e in events if e.event_type == "skill.script_completed")
    assert requested[0].correlation_id == completed.correlation_id


async def test_retry_replay_records_own_wire_input(store):
    a = agent(store, [LLMResponse(text="first"), LLMResponse(text="second")])
    first = await a.run("first question")
    second = await a.run("second question", runs=[first.run_id], activate_skills=["demo"])
    b = agent(store, [LLMResponse(text="retry")])
    retried = await b.retry(second.run_id)
    trace = next(
        t
        for t in await store.get_traces(retried.run_id)
        if t.role == "user" and not (t.meta or {}).get("dendrux.seed_prefix")
    )
    assert trace.content == "second question"
    assert "second question" in trace.meta["wire_content"]
    assert "first question" not in trace.meta["wire_content"]
    assert trace.meta["wire_content"].count('<skill_content name="demo"') == 1


@pytest.mark.parametrize("path", ["resume", "submit"])
async def test_non_skill_cancel_during_resume_preparation_sequences_event(store, path):
    p = MockLLM([LLMResponse(tool_calls=[ToolCall(name="client", params={})])])
    a = Agent(prompt="test", provider=p, tools=[client], state_store=store)
    paused = await a.run("input")
    state = await store.get_pause_state(paused.run_id)
    results = [
        ToolResult(name="client", call_id=state["pending_tool_calls"][0]["id"], payload='"ok"')
    ]
    previous = max(e.sequence_index for e in await store.get_run_events(paused.run_id))
    entered = asyncio.Event()

    async def preparing(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    with patch("dendrux.runtime.runner._prepare_resume", preparing):
        task = asyncio.create_task(
            a.resume(paused.run_id, tool_results=results)
            if path == "resume"
            else a.submit_tool_results(paused.run_id, results)
        )
        await asyncio.wait_for(entered.wait(), 5)
        if path == "submit":
            await a.cancel_run(paused.run_id)
        else:
            task.cancel()
        if path == "submit":
            assert (await task).status == RunStatus.CANCELLED
        else:
            with pytest.raises(asyncio.CancelledError):
                await task
    assert (await store.get_run(paused.run_id)).status == "cancelled"
    events = await store.get_run_events(paused.run_id)
    cancelled = [e for e in events if e.event_type == "run.cancelled"]
    assert len(cancelled) == 1 and cancelled[0].sequence_index > previous
    assert len({e.sequence_index for e in events}) == len(events)


@pytest.mark.parametrize("different_agents", [False, True])
async def test_interleaved_resume_streams_own_session(store, different_agents):
    a = agent(
        store,
        [
            LLMResponse(tool_calls=[ToolCall(name="client", params={})]),
            LLMResponse(tool_calls=[ToolCall(name="client", params={})]),
        ],
        tools=[client],
    )
    runs = [await a.run("A", activate_skills=["demo"]), await a.run("B")]
    provider = Reads()
    provider.path = "ref.txt"
    b = agent(store, [], tools=[client])
    b._provider = provider
    c = agent(store, [], tools=[client]) if different_agents else b
    c._provider = provider
    streams = []
    for owner, run in zip([b, c], runs, strict=True):
        state = await store.get_pause_state(run.run_id)
        results = [
            ToolResult(name="client", call_id=state["pending_tool_calls"][0]["id"], payload='"ok"')
        ]
        streams.append(owner.resume_stream(run.run_id, tool_results=results))
    iterators = [s.__aiter__() for s in streams]
    stale = []
    done = set()
    while len(done) != 2:
        for i in [1, 0]:
            if i in done:
                continue
            try:
                await anext(iterators[i])
                stale.append((current(), get_delegation_context()))
            except StopAsyncIteration:
                done.add(i)
    assert dict(provider.seen) == {runs[0].run_id: False, runs[1].run_id: True}
    assert all(s is None and d is None for s, d in stale)
    assert current() is None and get_delegation_context() is None


@pytest.mark.parametrize("claimed", [False, True])
async def test_malformed_manifest_fails_without_stuck_claim(store, claimed):
    # This fault injection / pause blob exercises the format-1 write path.
    from tests._fixtures.storage import pin_storage_format

    await pin_storage_format(store._engine, 1)
    from dendrux.runtime.runner import resume_claimed

    a = agent(store, [LLMResponse(tool_calls=[ToolCall(name="client", params={})])], tools=[client])
    paused = await a.run("input")
    await store.update_run_meta(paused.run_id, {"dendrux.skills": {"version": 1}})
    b = agent(store, [LLMResponse(text="done")], tools=[client])
    state = await store.get_pause_state(paused.run_id)
    result = ToolResult(name="client", call_id=state["pending_tool_calls"][0]["id"], payload='"ok"')
    if claimed:
        from dendrux.runtime.submit import serialize_tool_results

        await store.submit_and_claim(
            paused.run_id,
            expected_status="waiting_client_tool",
            submitted_data={"submitted_tool_results": serialize_tool_results([result])},
        )
        action = resume_claimed(
            paused.run_id,
            state_store=store,
            agent=b,
            provider=b._provider,
            pre_claim_status="waiting_client_tool",
        )
    else:
        action = b.submit_tool_results(paused.run_id, [result])
    with pytest.raises(ValueError, match="skill manifest"):
        await action
    assert (await store.get_run(paused.run_id)).status == (
        "error" if claimed else "waiting_client_tool"
    )
