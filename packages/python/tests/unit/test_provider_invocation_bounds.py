"""The invocation boundary keeps diagnostics typed and evidence JSON bounded."""

import math
from dataclasses import replace

import pytest

from dendrux.llm._calls import complete_call, stream_call
from dendrux.llm._evidence import persistable_response
from dendrux.llm.mock import MockLLM
from dendrux.types import LLMResponse, StreamEvent, StreamEventType, ToolCall, _validate_persistable


@pytest.mark.parametrize("stream", [False, True])
async def test_nonprovider_exception_groups_remain_intact(stream):
    original = ExceptionGroup("app", [ValueError("app failure")])
    provider = MockLLM([])

    async def complete(*args, **kwargs):
        raise original

    async def events(*args, **kwargs):
        raise original
        yield

    provider.complete = complete
    provider.complete_stream = events
    with pytest.raises(ExceptionGroup) as caught:
        if stream:
            async for _ in stream_call(provider, []):
                pass
        else:
            await complete_call(provider, [])
    assert caught.value is original


def test_model_evidence_preserves_json_types_and_replaces_only_invalid_characters():
    class Opaque:
        def __repr__(self):
            pytest.fail("opaque provider evidence must not invoke repr")

    response = LLMResponse(
        text="valid \U0001f600\x00\ud800",
        reasoning="reason\x00",
        raw_stop_reason="end\x00",
        provider_response={"key\x00": [True, 17, None, 1.25, math.inf, Opaque(), ("v\x00",)]},
        reasoning_blocks=[{"text": "think\udfff"}],
        tool_calls=[ToolCall("lookup\x00", {"valid": "unchanged"}, id="id\x00")],
    )
    clean = persistable_response(response)
    assert clean.text == "valid \U0001f600\ufffd\ufffd"
    assert clean.reasoning == "reason\ufffd" and clean.raw_stop_reason == "end\ufffd"
    assert clean.provider_response == {"key\ufffd": [True, 17, None, 1.25, None, None, ["v\ufffd"]]}
    assert clean.tool_calls[0].name == "lookup\ufffd"
    assert clean.tool_calls[0].id == "id\ufffd"
    _validate_persistable(clean.reasoning_blocks, "evidence")
    assert response.text.endswith("\x00\ud800"), "normalization must not mutate a shared response"


@pytest.mark.parametrize("overflow", ["depth", "nodes", "bytes", "utf8", "keys"])
def test_oversized_opaque_evidence_is_omitted_without_losing_the_answer(overflow):
    value = [0] * 10001
    if overflow == "depth":
        value = {}
        for _ in range(40):
            value = {"nested": value}
    if overflow == "bytes":
        value = "x" * 1048577
    elif overflow == "utf8":
        value = "é" * 600000
    elif overflow == "keys":
        value = {str(i): i for i in range(5001)}
    response = persistable_response(LLMResponse(text="answer", provider_response={"value": value}))
    assert response.text == "answer" and response.provider_response is None


async def test_streamed_text_and_tool_identity_are_safe_before_delivery():
    provider = MockLLM([])
    original = StreamEvent(StreamEventType.TEXT_DELTA, text="v\x00\ud800")

    async def events(*args, **kwargs):
        yield original
        yield replace(original, type=StreamEventType.TOOL_USE_START, tool_name="tool\x00")

    provider.complete_stream = events
    delivered = [event async for event in stream_call(provider, [])]
    assert delivered[0].text == "v\ufffd\ufffd"
    assert delivered[1].tool_name == "tool\ufffd"
