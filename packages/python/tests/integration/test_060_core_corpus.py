"""A7: format-2 continuation preserves every recorded provider conversation."""

import json
from pathlib import Path

import pytest
from sqlalchemy import update

from dendrux.db.models import DendruxSetting
from tests.conformance import harness as h
from tests.conformance.scenarios import CELLS


@pytest.mark.parametrize("cell", CELLS, ids=[c.id for c in CELLS])
async def test_a7_format2_provider_corpus(cell, store, engine, monkeypatch):
    async with engine.begin() as conn:
        await conn.execute(
            update(DendruxSetting).where(DendruxSetting.key == "storage_format").values(value="2")
        )
    from dendrux.runtime.replay import rebuild_history

    def provider_view(message):
        # The only projections allowed by contracts §12.28.
        view = h.message_view(message)
        for tc, value in zip(message.tool_calls or [], view["tool_calls"] or [], strict=True):
            raw = value.pop("raw_arguments")
            value["raw_length"] = len(raw) if raw is not None else getattr(tc, "raw_length", None)
        for value in view["parts"] or []:
            if value["type"] == "image":
                ref = value["ref"]
                value.clear()
                value.update(ref=ref, media_type=ref["media_type"], byte_length=ref["bytes"])
        return view

    original = h.RecordingProvider.complete
    comparisons = []
    rewrites = []

    async def checked(provider, messages, tools=None, **kwargs):
        run_id = kwargs.get("run_id")
        record = await store.get_run(run_id) if run_id else None
        if record is not None and record.storage_format == 2:
            rewrites.extend(
                await store.get_traces_after(run_id, after_order_index=None, kinds=("rewrite",))
            )
            rebuilt = await rebuild_history(store, run_id, reasoning_for=provider.model)
            comparisons.append(
                h.diff(
                    [provider_view(m) for m in messages],
                    [provider_view(m) for m in rebuilt.messages],
                )
            )
        return await original(provider, messages, tools, **kwargs)

    monkeypatch.setattr(h.RecordingProvider, "complete", checked)
    recording = await h.run_cell(cell, engine, store)
    assert all(d is None for d in comparisons), comparisons
    # §12.28 uses the live format-2 provider boundary as the oracle. The
    # unchanged format-1 corpus/goldens remain checked by conformance.
    assert len(comparisons) == sum(call["run_id"] is not None for call in recording["calls"])
    assert len(recording["calls"]) == len(h.read_json(h.corpus_path(cell.id))["calls"])

    if "compaction" not in cell.id and "history_edit" not in cell.id:
        assert not rewrites, [r.order_index for r in rewrites]


_RESUME_FLOWS = json.loads(
    (Path(__file__).parents[1] / "_fixtures/corpus_060_resume.json").read_text()
)


@pytest.mark.parametrize("flow", _RESUME_FLOWS, ids=lambda flow: flow["id"])
@pytest.mark.parametrize(
    "entry,approved",
    [("resume", True), ("resume_stream", True), ("submit", True), ("submit", False)],
)
@pytest.mark.parametrize("large", [False, True])
async def test_a7_mixed_resume_corpus(store, flow, entry, approved, large):
    from dendrux import Agent, tool
    from dendrux.llm.mock import MockLLM
    from dendrux.runtime.replay import rebuild_history
    from dendrux.types import Limits, LLMResponse, Role, ToolCall, ToolResult

    @tool(target=flow["target"])
    async def ask() -> str:
        """Resolve the pending operation."""
        return "ok"

    @tool()
    async def change(value: int) -> int:
        """Require a valid integer."""
        raise AssertionError("invalid call executed")

    calls = [ToolCall("ask"), ToolCall(**flow["rejected"])]
    observed = []

    class Provider(MockLLM):
        async def complete(self, messages, tools=None, **kwargs):
            tools_seen = [m.call_id for m in messages if m.role == Role.TOOL]
            if tools_seen:
                assert tools_seen == [call.id for call in calls]
                rebuilt = await rebuild_history(store, kwargs["run_id"])
                assert [h.message_view(m) for m in messages] == [
                    h.message_view(m) for m in rebuilt.messages
                ]
                observed.append(tools_seen)
            return await super().complete(messages, tools, **kwargs)

    agent = Agent(
        prompt="p",
        provider=Provider([LLMResponse(tool_calls=calls), LLMResponse(text="done")]),
        tools=[ask, change],
        require_approval=["ask"] if flow["target"] == "server" else [],
        state_store=store,
        strict_tenancy=False,
        limits=Limits(max_input_bytes=2 * 1024 * 1024),
    )
    paused = await agent.run("x" * 1_050_000 if large else "go")
    kwargs = {"expected_pause_id": paused.pause_id}
    if flow["target"] == "client":
        kwargs["tool_results"] = [ToolResult(name="ask", call_id=calls[0].id, payload='"ok"')]
    elif entry == "submit":
        kwargs["approved"] = approved
    if entry == "resume_stream":
        async with agent.resume_stream(paused.run_id, **kwargs) as stream:
            events = [event async for event in stream]
        result = events[-1].run_result
    elif entry == "submit":
        method = agent.submit_tool_results if flow["target"] == "client" else agent.submit_approval
        if flow["target"] == "client":
            kwargs["results"] = kwargs.pop("tool_results")
        result = await method(paused.run_id, **kwargs)
    else:
        result = await agent.resume(paused.run_id, **kwargs)
    assert result.status.value == "success", result.error
    assert observed == [[call.id for call in calls]]
    assert not await store.get_traces_after(
        result.run_id, after_order_index=None, kinds=("rewrite",)
    )
