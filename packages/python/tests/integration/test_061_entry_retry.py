"""Successful retries regenerate answers without changing conversation replay."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import asdict
from typing import Any

import pytest
from pydantic import BaseModel

from dendrux import Agent, tool
from dendrux.errors import ProviderResponseError
from dendrux.llm.mock import MockLLM
from dendrux.loops.single import SingleCall
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.types import LLMResponse, Message, Role, RunStatus, ToolCall, UsageStats
from tests._fixtures.storage import pin_storage_format

BLOCKS = [{"type": "thinking", "thinking": "plan", "signature": "sig-retry"}]


class Answer(BaseModel):
    """A structured final answer."""

    value: str


@tool()
async def lookup(key: str) -> str:
    """Return a deterministic observation."""
    return f"found {key}"


@pytest.fixture(params=[1, 2], ids=["format1", "format2"])
async def retry_store(engine: Any, request: pytest.FixtureRequest) -> Any:
    await pin_storage_format(engine, request.param)
    return store_for_engine(engine)


def _agent(store: Any, provider: MockLLM, **kwargs: Any) -> Agent:
    return Agent(
        name="retry-test",
        prompt="Answer using the observations.",
        provider=provider,
        state_store=store,
        strict_tenancy=False,
        tools=kwargs.pop("tools", [lookup]),
        **kwargs,
    )


def _conversation(messages: list[Message]) -> list[dict[str, Any]]:
    return [
        {
            "role": m.role.value,
            "content": m.content,
            "calls": [asdict(c) for c in m.tool_calls or []],
            "call_id": m.call_id,
            "name": m.name,
            "reasoning_blocks": m.reasoning_blocks,
        }
        for m in messages
        if m.role != Role.SYSTEM
    ]


def _tool_response() -> LLMResponse:
    return LLMResponse(
        text="Checking both records.",
        tool_calls=[
            ToolCall(name="lookup", params={"key": "a"}),
            ToolCall(name="lookup", params={"key": "b"}),
        ],
        reasoning_blocks=BLOCKS,
        usage=UsageStats(input_tokens=10, output_tokens=5, total_tokens=15),
    )


@pytest.mark.parametrize("kind", ["plain", "tools", "structured"])
async def test_061_success_retry_omits_final_answer_only(retry_store: Any, kind: str) -> None:
    old = '{"value":"source answer"}' if kind == "structured" else "source answer"
    fresh = '{"value":"fresh answer"}' if kind == "structured" else "fresh answer"
    source_provider = MockLLM(
        [*([_tool_response()] if kind != "plain" else []), LLMResponse(text=old)]
    )
    source = await _agent(retry_store, source_provider).run("Look up a and b.")
    assert source.status == RunStatus.SUCCESS
    before = deepcopy(await retry_store.get_run(source.run_id))
    traces_before = deepcopy(await retry_store.get_traces(source.run_id))
    expected = _conversation(source_provider.wire_messages[-1])
    provider = MockLLM(
        [
            LLMResponse(
                text=fresh,
                usage=UsageStats(input_tokens=3, output_tokens=2, total_tokens=5),
            )
        ]
    )
    retried = await _agent(retry_store, provider).retry(source.run_id)
    sent = _conversation(provider.wire_messages[0])
    assert sent[-1]["content"] != old
    assert sent == expected
    assert retried.status == RunStatus.SUCCESS
    assert retried.answer == fresh
    if kind == "structured":
        assert Answer.model_validate_json(retried.answer) == Answer(value="fresh answer")
    if kind != "plain":
        assert sent[1]["reasoning_blocks"] == BLOCKS
        assert [c["id"] for c in sent[1]["calls"]] == [m["call_id"] for m in sent[2:]]
    assert retried.run_id != source.run_id
    assert retried.iteration_count == 1
    assert retried.usage.total_tokens == 5
    row = await retry_store.get_run(retried.run_id)
    assert row.retry_of_run_id == source.run_id
    assert row.iteration_count == 1
    assert (row.total_input_tokens, row.total_output_tokens) == (3, 2)
    assert await retry_store.get_run(source.run_id) == before
    assert await retry_store.get_traces(source.run_id) == traces_before


@pytest.mark.parametrize("replay_last", [0, 1])
async def test_success_replay_keeps_final_answer(retry_store: Any, replay_last: int) -> None:
    source = await _agent(retry_store, MockLLM([LLMResponse(text="source answer")])).run("Hi")
    provider = MockLLM([LLMResponse(text="next answer")])
    result = await _agent(retry_store, provider).run(
        "Continue", runs=[source.run_id], replay_last=replay_last
    )
    assert result.status == RunStatus.SUCCESS
    assert [(m["role"], m["content"]) for m in _conversation(provider.wire_messages[0])] == [
        ("user", "Hi"),
        ("assistant", "source answer"),
        ("user", "Continue"),
    ]


async def test_retry_genuine_empty_completion_still_errors(retry_store: Any) -> None:
    source = await _agent(retry_store, MockLLM([LLMResponse(text="source answer")])).run("Hi")
    provider = MockLLM([LLMResponse()])
    agent = _agent(retry_store, provider)
    with pytest.raises(ProviderResponseError) as caught:
        await agent.retry(source.run_id)
    assert caught.value.provider_code == "empty_completion"
    assert caught.value.retryable is False
    result = await agent.get_result(caught.value.run_id, tenant_id=None)
    assert result.status == RunStatus.ERROR
    assert result.error_info is not None
    assert result.error_info.provider_code == "empty_completion"
    assert result.error_info.retryable is False
    assert provider.calls_made == 1


class CancelAfterTools(MockLLM):
    """Cancel when the model is asked to answer from tool observations."""

    async def complete(self, *args: Any, **kwargs: Any) -> LLMResponse:
        response = await super().complete(*args, **kwargs)
        if self.calls_made == 2:
            raise asyncio.CancelledError
        return response


@pytest.mark.parametrize("status", ["error", "cancelled"])
async def test_non_success_retry_keeps_history(retry_store: Any, status: str) -> None:
    provider_type = CancelAfterTools if status == "cancelled" else MockLLM
    source_provider = provider_type([_tool_response(), LLMResponse()])
    agent = _agent(retry_store, source_provider)
    if status == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await agent.run("Look up a and b.")
        source_id = source_provider.call_history[0]["run_id"]
    else:
        with pytest.raises(ProviderResponseError) as caught:
            await agent.run("Look up a and b.")
        source_id = caught.value.run_id
    before = deepcopy(await retry_store.get_run(source_id))
    assert before.status == status
    retry_provider = MockLLM([LLMResponse(text="fresh answer")])
    result = await _agent(retry_store, retry_provider).retry(source_id)
    assert result.status == RunStatus.SUCCESS
    assert _conversation(retry_provider.wire_messages[0]) == _conversation(
        source_provider.wire_messages[-1]
    )
    assert await retry_store.get_run(source_id) == before


async def test_success_single_call_retry_stays_unsupported(retry_store: Any) -> None:
    provider = MockLLM([LLMResponse(text="source answer")])
    source = await _agent(retry_store, provider, loop=SingleCall(), tools=[]).run("Hi")
    assert source.status == RunStatus.SUCCESS
    with pytest.raises(ValueError, match="SingleCall"):
        await _agent(retry_store, provider).retry(source.run_id)
    assert provider.calls_made == 1
