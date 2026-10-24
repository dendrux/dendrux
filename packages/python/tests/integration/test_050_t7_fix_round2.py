"""Disconnect cleanup and byte-for-byte client payload compatibility."""

import asyncio
import json

import anyio
import pytest

from dendrux import Limits
from dendrux.runtime import runner
from dendrux.types import LLMResponse, RunStatus
from tests._helpers.scripted_provider import ScriptedLLM, tool_call
from tests.integration.test_050_t7_integrity import agent, consume, result_for


@pytest.mark.parametrize("entry", ["stream", "resume_stream"])
async def test_disconnect_between_events_closes_generator_without_gc(store, monkeypatch, entry):
    resumed = entry == "resume_stream"
    llm = ScriptedLLM(
        ([tool_call("client_read", pid="a")] if resumed else []) + [LLMResponse(text="done")]
    )
    a = agent(store, llm)
    if resumed:
        paused = await a.run("go")
        stream = a.resume_stream(paused.run_id, tool_results=[result_for(paused)])
    else:
        stream = a.stream("go")
    released = []
    release_edges = runner._release_memory_edges

    def tracked_release(runtime, state_store, run_id):
        released.append(run_id)
        release_edges(runtime, state_store, run_id)

    monkeypatch.setattr(runner, "_release_memory_edges", tracked_release)
    cleanup = stream._cleanup
    cleaned = []

    async def tracked_cleanup():
        try:
            return await cleanup()
        finally:
            cleaned.append(asyncio.current_task())

    stream._cleanup = tracked_cleanup
    owner = asyncio.current_task()
    with anyio.CancelScope() as disconnect:
        async with stream:
            events = stream.__aiter__()
            while not a._task_manager.has_interrupt(stream.run_id):
                event = await anext(events)
                assert event.run_result is None
            # The consumer is sending an event, not requesting another one.
            # Keep both generators referenced through all assertions: no GC.
            disconnect.cancel()
            await anyio.sleep_forever()
    assert disconnect.cancelled_caught
    assert cleaned == [owner]
    assert stream._gen.ag_frame is None
    assert not a._task_manager.has_interrupt(stream.run_id)
    assert resumed or released == [stream.run_id]
    assert (await store.get_run(stream.run_id)).status == "cancelled"
    assert (
        sum(
            event.event_type == "run.cancelled"
            for event in await store.get_run_events(stream.run_id)
        )
        == 1
    )
    await events.aclose()


@pytest.mark.parametrize("entry", ["submit", "resume", "resume_stream"])
@pytest.mark.parametrize(
    "payload",
    [
        "Revenue: $100M",
        'Revenue:\n"quoted"\tC:\\sheets\\é',
        ' { "rows" : [1, 2], "label": "\\u00e9" }\n',
        '"Revenue:\\n\\u00e9"',
        "NaN",
    ],
)
async def test_client_payload_reaches_model_byte_for_byte_as_041(store, entry, payload):
    llm = ScriptedLLM([tool_call("client_read", pid="a"), LLMResponse(text="done")])
    # JSON encoding of the plain-text cases would exceed this exact wire cap.
    a = agent(store, llm, limits=Limits(max_tool_result_bytes=len(payload.encode("utf-8"))))
    paused = await a.run("go")
    results = [result_for(paused, payload)]
    if entry == "submit":
        result = await a.submit_tool_results(paused.run_id, results)
    elif entry == "resume":
        result = await a.resume(paused.run_id, tool_results=results)
    else:
        result, _ = await consume(a.resume_stream(paused.run_id, tool_results=results))
    assert result.status == RunStatus.SUCCESS
    message = next(m for m in llm.messages[-1] if m.role.value == "tool")
    assert message.content.encode("utf-8") == payload.encode("utf-8")
    stored = (await store.get_tool_calls(paused.run_id))[-1].result
    assert stored == (payload if payload == "NaN" else _stored_value(payload))


def _stored_value(payload):
    try:
        return json.loads(payload)
    except ValueError:
        return payload
