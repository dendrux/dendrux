"""The persisted v2 submission is one closed envelope, independent of parsing views."""

from functools import wraps

import pytest

from dendrux import Agent, tool
from dendrux.llm.mock import MockLLM
from dendrux.runtime.submit import submitted_view
from dendrux.types import LLMResponse, ToolCall, ToolResult


@pytest.mark.parametrize("streaming", [False, True])
async def test_submission_envelope_is_durable_before_consumption(store, monkeypatch, streaming):
    @tool(target="client")
    async def lookup() -> str:
        """Return a client value."""
        return ""

    agent = Agent(
        prompt="p",
        provider=MockLLM([LLMResponse(tool_calls=[ToolCall("lookup")]), LLMResponse(text="done")]),
        tools=[lookup],
        state_store=store,
        strict_tenancy=False,
    )
    paused = await agent.run("go")
    checked = []
    consume = store.consume_submission

    @wraps(consume)
    async def inspect(run_id, **kwargs):
        raw = await store.get_pause_state(run_id)
        descriptor = (await store.get_loop_state(run_id))["submission"]
        assert "submitted" not in raw
        envelope = raw["submission"]
        assert set(envelope) == {"submission_id", "kind", "pause_id", "call_ids", "payload"}
        assert {k: envelope[k] for k in envelope if k != "payload"} == {
            k: descriptor[k] for k in envelope if k != "payload"
        }
        assert descriptor["payload_ref"] == {"pause_data_key": "submission"}
        assert envelope["kind"] == "tool_results"
        assert envelope["payload"]["submitted_tool_results"][0]["payload"] == '"answer"'
        assert envelope["call_ids"] == [paused.pending.tool_calls[0].id]
        checked.append(True)
        return await consume(run_id, **kwargs)

    monkeypatch.setattr(store, "consume_submission", inspect)
    kwargs = dict(
        expected_pause_id=paused.pause_id,
        tool_results=[
            ToolResult(name="lookup", call_id=paused.pending.tool_calls[0].id, payload='"answer"')
        ],
    )
    if streaming:
        async with agent.resume_stream(paused.run_id, **kwargs) as stream:
            events = [event async for event in stream]
        result = events[-1].run_result
    else:
        result = await agent.resume(paused.run_id, **kwargs)
    assert result.status.value == "success"
    assert checked == [True]
    assert "submission" not in ((await store.get_pause_state(paused.run_id)) or {})


@pytest.mark.parametrize(
    "bad", ["old", "missing_field", "extra_field", "bad_kind", "bad_payload", "bad_call_ids"]
)
def test_v2_submission_reader_rejects_noncontract_shapes(bad):
    envelope = {
        "submission_id": "s",
        "kind": "input",
        "pause_id": "p",
        "call_ids": [],
        "payload": {"submitted_user_input": "answer"},
    }
    raw = {"schema_version": 2, "submission": envelope}
    if bad == "old":
        raw = {"schema_version": 2, "submitted": envelope["payload"]}
    elif bad == "missing_field":
        envelope.pop("pause_id")
    elif bad == "extra_field":
        envelope["reserved_order_index"] = 1
    elif bad == "bad_kind":
        envelope["kind"] = "unknown"
    elif bad == "bad_payload":
        envelope["payload"] = "answer"
    else:
        envelope["call_ids"] = "not-a-list"
    with pytest.raises(ValueError):
        submitted_view(raw)


def test_submission_reader_preserves_legacy_and_projects_valid_v2():
    legacy = {"schema_version": 1, "submitted_user_input": "old"}
    assert submitted_view(legacy) is legacy
    raw = {
        "schema_version": 2,
        "submission": {
            "submission_id": "s",
            "kind": "input",
            "pause_id": "p",
            "call_ids": [],
            "payload": {"submitted_user_input": "answer"},
        },
    }
    assert submitted_view(raw)["submitted_user_input"] == "answer"
    assert "submitted_user_input" not in raw
