"""Long token streams fit the response budget without charging repeated SDK snapshots."""

import pytest

from dendrux.errors import ProviderError
from tests.unit.test_050_providers import (
    USER,
    a_end,
    a_start,
    anthropic_provider,
    c_chunk,
    chat_provider,
    r_response,
    responses_provider,
    sse,
    stream_handler,
    wire,
)


def anthropic_block(kind, text, count):
    delta_type, field = (
        ("thinking_delta", "thinking") if kind == "thinking" else ("text_delta", "text")
    )
    block = {"type": kind, field: ""}
    if kind == "thinking":
        block["signature"] = ""
    parts = [
        a_start(),
        sse(
            (
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": block},
            )
        ),
    ]
    parts += [
        sse(
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": delta_type, field: text},
                },
            )
        )
    ] * count
    parts += [sse(("content_block_stop", {"type": "content_block_stop", "index": 0})), a_end()]
    return parts


def response_output(text):
    return [
        {
            "type": "message",
            "id": "m",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
    ]


@pytest.mark.parametrize("adapter", ["chat", "responses", "anthropic", "anthropic_thinking"])
async def test_sixty_four_thousand_token_deltas_fit_default_budget(adapter):
    token = "word"
    count = 65536
    if adapter == "chat":
        provider = chat_provider()
        parts = [sse((None, c_chunk(token)))] * count + [sse((None, c_chunk(finish="stop")))]
    elif adapter == "responses":
        provider = responses_provider()
        parts = [
            sse(
                (
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": "m",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": token,
                    },
                )
            )
        ] * count
        parts += [
            sse(
                (
                    "response.completed",
                    {
                        "type": "response.completed",
                        "response": r_response(output=response_output(token * count)),
                    },
                )
            )
        ]
    else:
        provider = anthropic_provider()
        parts = anthropic_block(
            "thinking" if adapter.endswith("thinking") else "text", token, count
        )
    bodies = []
    wire(provider, stream_handler(parts, record=bodies))
    seen = 0
    response = None
    try:
        async for event in provider.complete_stream(USER, run_id="long-response"):
            if event.text:
                assert event.text == token
                seen += 1
            if event.type.value == "done":
                response = event.raw
        assert seen == count
        assert (
            response.reasoning if adapter.endswith("thinking") else response.text
        ) == token * count
        assert len(bodies) == 1 and bodies[0].closed
    finally:
        await provider.close()


@pytest.mark.parametrize("adapter", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize(
    "limit,repetitions",
    [(256 * 1024, 1), (None, 33)],
    ids=["single_configured", "cumulative_default"],
)
async def test_large_fragments_exceed_response_budget(adapter, limit, repetitions):
    fragment = "x" * (512 * 1024)
    options = {"max_stream_bytes": limit} if limit is not None else {}
    if adapter == "chat":
        provider = chat_provider(**options)
        parts = [sse((None, c_chunk(fragment)))] * repetitions
    elif adapter == "responses":
        provider = responses_provider(**options)
        parts = [
            sse(
                (
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": "m",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": fragment,
                    },
                )
            )
        ] * repetitions
    else:
        provider = anthropic_provider(**options)
        parts = anthropic_block("text", fragment, repetitions)
    bodies = []
    wire(provider, stream_handler(parts, record=bodies))
    try:
        with pytest.raises(ProviderError) as caught:
            async for _ in provider.complete_stream(USER):
                pass
        assert caught.value.provider_code == "stream_size_limit"
        assert not caught.value.retryable
        assert len(bodies) == 1 and bodies[0].closed
    finally:
        await provider.close()


async def test_responses_completed_snapshot_is_not_charged_twice():
    provider = responses_provider(max_stream_bytes=80000)
    text = "x" * 64
    parts = [
        sse(
            (
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": "m",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": text,
                },
            )
        )
    ] * 1024
    parts += [
        sse(
            (
                "response.completed",
                {
                    "type": "response.completed",
                    "response": r_response(output=response_output(text * 1024)),
                },
            )
        )
    ]
    wire(provider, stream_handler(parts))
    try:
        events = [event async for event in provider.complete_stream(USER)]
        assert events[-1].raw.text == text * 1024
    finally:
        await provider.close()


async def test_responses_argument_completion_credits_utf8_deltas():
    provider = responses_provider(max_stream_bytes=80000)
    prefix, fragment, suffix = '{"x":"', "é" * 32, '"}'
    arguments = prefix + fragment * 1024 + suffix
    parts = [
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
    ]
    for delta in [prefix, *([fragment] * 1024), suffix]:
        parts.append(
            sse(
                (
                    "response.function_call_arguments.delta",
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": "c",
                        "output_index": 0,
                        "delta": delta,
                    },
                )
            )
        )
    parts += [
        sse(
            (
                "response.function_call_arguments.done",
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": "c",
                    "output_index": 0,
                    "arguments": arguments,
                },
            )
        ),
        sse(("response.completed", {"type": "response.completed", "response": r_response()})),
    ]
    wire(provider, stream_handler(parts))
    try:
        events = [event async for event in provider.complete_stream(USER)]
        assert events[-1].raw.tool_calls[0].params == {"x": fragment * 1024}
    finally:
        await provider.close()


async def test_responses_argument_without_deltas_still_has_a_byte_bound():
    provider = responses_provider(max_stream_bytes=256 * 1024)
    bodies = []
    parts = [
        sse(
            (
                "response.function_call_arguments.done",
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": "c",
                    "output_index": 0,
                    "arguments": '{"x":"' + "x" * (512 * 1024) + '"}',
                },
            )
        )
    ]
    wire(provider, stream_handler(parts, record=bodies))
    try:
        with pytest.raises(ProviderError) as caught:
            async for _ in provider.complete_stream(USER):
                pass
        assert caught.value.provider_code == "stream_size_limit"
        assert bodies[0].closed
    finally:
        await provider.close()
