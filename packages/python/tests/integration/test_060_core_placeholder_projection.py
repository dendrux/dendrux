import asyncio

import pytest

from dendrux import Agent, tool
from dendrux.guardrails import PII, GuardrailEngine, Pattern
from dendrux.llm.mock import MockLLM
from dendrux.runtime.replay import rebuild_history
from dendrux.types import LLMResponse, Role, ToolCall, ToolResult
from tests._fixtures.storage import pin_storage_format


@pytest.mark.parametrize("entry", ["resume", "submit_tool_results"])
async def test_issued_digit_projection_survives_resume(store, entry):
    @tool(target="client")
    async def lookup() -> str:
        return "ok"

    views = []

    class Provider(MockLLM):
        async def complete(self, messages, tools=None, **kwargs):
            view = [m.content for m in messages]
            rebuilt = await rebuild_history(store, kwargs["run_id"])
            assert [m.content for m in rebuilt.messages] == view
            assert next(m.content for m in messages if m.role == Role.USER) == "my pin <<NUM_1>>"
            views.append(view)
            return await super().complete(messages, tools, **kwargs)

    provider = Provider([LLMResponse(tool_calls=[ToolCall("lookup")]), LLMResponse(text="done")])
    agent = Agent(
        prompt="p",
        tools=[lookup],
        provider=provider,
        state_store=store,
        strict_tenancy=False,
        guardrails=[PII(extra_patterns=[Pattern("NUM", r"\d")], restore="all")],
    )
    paused = await agent.run("my pin 7")
    assert paused.status.value == "waiting_client_tool"
    results = [ToolResult(name="lookup", call_id=paused.pending.tool_calls[0].id, payload='"ok"')]
    result = await getattr(agent, entry)(
        paused.run_id,
        expected_pause_id=paused.pause_id,
        **{"tool_results" if entry == "resume" else "results": results},
    )
    assert result.status.value == "success"
    assert len(views) == 2
    assert views[1][: len(views[0])] == views[0]
    assert await store.get_pii_mapping(result.run_id) == {"<<NUM_1>>": "7"}
    assert not await store.get_traces_after(
        result.run_id, after_order_index=None, kinds=("rewrite",)
    )


@pytest.mark.parametrize("fmt", [0, 1, 2], ids=["storeless", "format1", "format2"])
async def test_values_in_unissued_text_are_redacted(store, engine, fmt, caplog):
    await pin_storage_format(engine, fmt or 1)
    text = "card <<X_ 4111111111111111>> phone <<Y_5551234567>> SSN <<Z_ 123-45-6789>>"
    values = {"4111111111111111", "5551234567", "123-45-6789"}
    pii = PII(restore="all")
    assert {f.text for f in await pii.scan(text)} == values
    provider = MockLLM([LLMResponse(text="done")])
    result = await Agent(
        prompt="p",
        provider=provider,
        state_store=store if fmt else None,
        strict_tenancy=False,
        guardrails=[pii],
    ).run(text)
    assert result.status.value == "success"
    received = next(m.content for m in provider.wire_messages[0] if m.role == Role.USER)
    assert all(value not in received for value in values)
    assert all(value not in caplog.text for value in values)
    assert all(
        f"<<{entity}_1>>" in received for entity in ("CREDIT_CARD", "PHONE_NUMBER", "US_SSN")
    )


@pytest.mark.parametrize("fmt", [0, 1, 2], ids=["storeless", "format1", "format2"])
async def test_unissued_index_with_issued_prefix_is_redacted(store, engine, fmt):
    await pin_storage_format(engine, fmt or 1)

    @tool()
    async def lookup() -> str:
        return "<<NUM_9>>"

    provider = MockLLM([LLMResponse(tool_calls=[ToolCall("lookup")]), LLMResponse(text="done")])
    result = await Agent(
        prompt="p",
        provider=provider,
        tools=[lookup],
        state_store=store if fmt else None,
        strict_tenancy=False,
        guardrails=[PII(extra_patterns=[Pattern("NUM", r"\d")], restore="all")],
    ).run("my pin 7")
    assert result.status.value == "success"
    messages = provider.wire_messages[1]
    assert next(m.content for m in messages if m.role == Role.USER) == "my pin <<NUM_1>>"
    content = next(m.content for m in messages if m.role == Role.TOOL)
    assert "<<NUM_9>>" not in content
    assert "<<NUM_2>>" in content


async def test_issued_projection_is_local_to_each_engine():
    engines = [
        GuardrailEngine([PII(extra_patterns=[Pattern("NUM", r"\d")], restore="all")])
        for _ in range(2)
    ]
    issued, _, _ = await engines[0].scan_incoming("7")
    assert issued == "<<NUM_1>>"
    own, other = await asyncio.gather(*(engine.scan_incoming(issued) for engine in engines))
    assert own[0] == issued
    assert other[0] != issued
    assert engines[0].get_pii_mapping() == {"<<NUM_1>>": "7"}
    assert engines[1].get_pii_mapping() == {"<<NUM_1>>": "1"}
