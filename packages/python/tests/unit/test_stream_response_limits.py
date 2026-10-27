"""Provider streams stop accumulating when their local resource budget is exhausted."""

import pytest

from dendrux.errors import ProviderError
from tests.unit.test_050_providers import (
    USER,
    a_end,
    a_start,
    anthropic_provider,
    c_chunk,
    chat_provider,
    drain,
    r_response,
    responses_provider,
    sse,
    stream_handler,
    wire,
)


@pytest.mark.parametrize("kind", ["text", "reasoning", "arguments", "refusal"])
@pytest.mark.parametrize("adapter", ["chat", "responses", "anthropic"])
async def test_active_stream_byte_bound_closes_upstream(adapter, kind):
    if adapter == "anthropic" and kind == "refusal":
        pytest.skip("Anthropic has no refusal text delta")
    fragment = "é" * (512 * 1024)
    parts = []
    if adapter == "chat":
        provider = chat_provider()
        for _ in range(18):
            delta = (
                {"content": fragment}
                if kind == "text"
                else {"reasoning": fragment}
                if kind == "reasoning"
                else {"refusal": fragment}
                if kind == "refusal"
                else {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "c",
                            "type": "function",
                            "function": {"name": "f", "arguments": fragment},
                        }
                    ]
                }
            )
            parts.append(sse((None, c_chunk(**delta))))
        parts.append(sse((None, c_chunk(finish="stop"))))
    elif adapter == "responses":
        provider = responses_provider()
        event_type = {
            "text": "response.output_text.delta",
            "reasoning": "response.reasoning_summary_text.delta",
            "arguments": "response.function_call_arguments.delta",
            "refusal": "response.refusal.delta",
        }[kind]
        if kind == "arguments":
            parts.append(
                sse(
                    (
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": 0,
                            "item": {
                                "type": "function_call",
                                "id": "c",
                                "call_id": "c",
                                "name": "f",
                                "arguments": "",
                            },
                        },
                    )
                )
            )
        parts += [
            sse(
                (
                    event_type,
                    {
                        "type": event_type,
                        "delta": fragment,
                        "item_id": "c",
                        "output_index": 0,
                        "content_index": 0,
                        "summary_index": 0,
                    },
                )
            )
            for _ in range(18)
        ]
        parts.append(
            sse(("response.completed", {"type": "response.completed", "response": r_response()}))
        )
    else:
        provider = anthropic_provider()
        block_type = {"text": "text", "reasoning": "thinking", "arguments": "tool_use"}[kind]
        block = {
            "type": block_type,
            "text": "",
            "thinking": "",
            "signature": "",
            "id": "c",
            "name": "f",
            "input": {},
        }
        delta_type, field = {
            "text": ("text_delta", "text"),
            "reasoning": ("thinking_delta", "thinking"),
            "arguments": ("input_json_delta", "partial_json"),
        }[kind]
        parts = [
            a_start(),
            sse(
                (
                    "content_block_start",
                    {"type": "content_block_start", "index": 0, "content_block": block},
                )
            ),
        ]
        if kind == "arguments":
            parts.append(
                sse(
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "input_json_delta", "partial_json": '{"x":"'},
                        },
                    )
                )
            )
        parts += [
            sse(
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": delta_type, field: fragment},
                    },
                )
            )
            for _ in range(18)
        ]
        parts += [sse(("content_block_stop", {"type": "content_block_stop", "index": 0})), a_end()]
    bodies = []
    wire(provider, stream_handler(parts, record=bodies))
    try:
        with pytest.raises(ProviderError) as caught:
            await drain(provider.complete_stream(USER, run_id="bounded"))
        assert caught.value.category == "rejected"
        assert caught.value.provider_code == "stream_size_limit"
        assert bodies and all(body.closed for body in bodies)
        assert len(bodies) == 1
    finally:
        await provider.close()


async def test_tool_slot_bound_and_configurable_stream_limits():
    provider = chat_provider(max_stream_bytes=65536, max_stream_tool_calls=2)
    parts = [
        sse(
            (
                None,
                c_chunk(
                    tool_calls=[
                        {
                            "index": i,
                            "id": f"c{i}",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ]
                ),
            )
        )
        for i in range(3)
    ]
    bodies = []
    wire(provider, stream_handler(parts, record=bodies))
    try:
        with pytest.raises(ProviderError) as caught:
            await drain(provider.complete_stream(USER))
        assert caught.value.provider_code == "stream_tool_call_limit"
        assert all(body.closed for body in bodies)
    finally:
        await provider.close()


@pytest.mark.parametrize("adapter", ["chat", "responses", "anthropic"])
async def test_completed_calls_also_obey_slot_bound(adapter):
    if adapter == "chat":
        provider = chat_provider(max_stream_tool_calls=2)
        parts = [
            sse(
                (
                    None,
                    c_chunk(
                        finish="tool_calls",
                        tool_calls=[
                            {
                                "index": 0,
                                "id": f"c{i}",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    ),
                )
            )
            for i in range(3)
        ]
    elif adapter == "responses":
        provider = responses_provider(max_stream_tool_calls=2)
        parts = [
            sse(
                (
                    "response.function_call_arguments.done",
                    {
                        "type": "response.function_call_arguments.done",
                        "item_id": "c",
                        "output_index": i,
                        "arguments": "{}",
                    },
                )
            )
            for i in range(3)
        ]
    else:
        provider = anthropic_provider(max_stream_tool_calls=2)
        parts = [a_start()]
        for i in range(3):
            parts.append(
                sse(
                    (
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": i,
                            "content_block": {
                                "type": "tool_use",
                                "id": f"c{i}",
                                "name": "f",
                                "input": {},
                            },
                        },
                    ),
                    ("content_block_stop", {"type": "content_block_stop", "index": i}),
                )
            )
    bodies = []
    wire(provider, stream_handler(parts, record=bodies))
    try:
        with pytest.raises(ProviderError) as caught:
            await drain(provider.complete_stream(USER))
        assert caught.value.provider_code == "stream_tool_call_limit"
        assert all(body.closed for body in bodies)
    finally:
        await provider.close()
