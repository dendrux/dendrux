"""Anthropic strict-mode fallback (0.4.1 T2 fix round).

A request that carried a strict entry and got a 400 pointing at the strict
schema is retried once with every strict entry reverted to its 0.4.0 payload.
Served through the real SDK over a fake httpx transport (no network). Error
messages are the ones recorded in live probes on 2026-10-18.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import anthropic
import httpx
import pytest

from dendrux.errors import ImageRejectedError, ProviderError, ProviderUnavailableError
from dendrux.llm import anthropic as anthropic_mod
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.types import Message, Role, StreamEventType, ToolDef

if TYPE_CHECKING:
    from collections.abc import Iterator

from tests.unit.test_provider_images_errors import (
    USER_IMAGE,
    _anthropic_head,
    assert_clean,
    assert_frames_clean,
    drain,
    sse,
)

GRAMMAR_TOO_LARGE = (
    "The compiled grammar is too large, which would cause performance issues. "
    "Simplify your tool schemas or reduce the number of strict tools."
)
TOO_COMPLEX = (
    "Schema is too complex for compilation. Try reducing the number of tools or "
    "simplifying tool schemas."
)
REGEX_BUDGET = (
    "tools.{i}.custom: Unsupported regex feature in pattern field: pattern is too complex "
    "for structured output: reduce the {{n,m}} upper bound, narrow the character class "
    "range, or avoid nesting quantified groups"
)
UNRELATED = "messages.0.content: Input should be a valid list"
THINKING_FORCED = "Thinking may not be enabled when tool_choice forces tool use."

STRICT_PARAMS = {
    "type": "object",
    "properties": {"code": {"type": "string", "pattern": "^[a-z]{1,100000}$"}},
    "required": ["code"],
    "additionalProperties": False,
}
PLAIN_PARAMS = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
STRICT_TOOL = ToolDef(
    name="lookup_code", description="Look up.", parameters=STRICT_PARAMS, strict=True
)
PLAIN_TOOL = ToolDef(name="search", description="Search.", parameters=PLAIN_PARAMS)
TEXT = [Message(role=Role.USER, content="hello")]


def _error(message: str, kind: str = "invalid_request_error") -> dict[str, Any]:
    return {"type": "error", "error": {"type": kind, "message": message}}


def _message_body(content: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": content or [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }


def _stream_ok() -> bytes:
    return sse(
        *_anthropic_head(),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 3},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    )


class Wire:
    """Scripted fake transport: one (status, body) per request; records request bodies."""

    def __init__(self, provider: AnthropicProvider, *replies: tuple[int, Any]) -> None:
        self.replies = list(replies)
        self.bodies: list[dict[str, Any]] = []
        provider._client = anthropic.AsyncAnthropic(
            api_key="test-key",
            base_url="http://anthropic.test",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self)),
            max_retries=0,
        )

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        status, body = self.replies.pop(0)
        if isinstance(body, bytes):
            return httpx.Response(
                status, content=body, headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(status, json=body)

    def tools(self, n: int) -> list[dict[str, Any]]:
        return self.bodies[n]["tools"]


@pytest.fixture(autouse=True)
def _fresh_strict_state() -> Iterator[None]:
    anthropic_mod._STRICT_WARNED.clear()
    anthropic_mod._STRICT_REJECTED.clear()
    yield
    anthropic_mod._STRICT_WARNED.clear()
    anthropic_mod._STRICT_REJECTED.clear()


@pytest.fixture
def claude() -> AnthropicProvider:
    return AnthropicProvider(model="claude-sonnet-4-6", api_key="test-key")


def _legacy(td: ToolDef) -> dict[str, Any]:
    return {"name": td.name, "description": td.description, "input_schema": td.parameters}


def _chained(exc: BaseException) -> str:
    """Text of every exception chained under ``exc`` (cause or context)."""
    out = []
    seen: set[int] = set()
    current: BaseException | None = exc.__cause__ or exc.__context__
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        out.append(str(current))
        current = current.__cause__ or current.__context__
    return "\n".join(out)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "dendrux.llm.anthropic"
    ]


class TestBatchFallback:
    @pytest.mark.parametrize("message", [REGEX_BUDGET.format(i=1), GRAMMAR_TOO_LARGE, TOO_COMPLEX])
    async def test_strict_rejection_retries_without_strict(
        self, claude: AnthropicProvider, message: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        wire = Wire(claude, (400, _error(message)), (200, _message_body()))
        with caplog.at_level(logging.WARNING, logger="dendrux.llm.anthropic"):
            result = await claude.complete(TEXT, tools=[PLAIN_TOOL, STRICT_TOOL])
        assert result.text == "ok"
        assert len(wire.bodies) == 2
        assert wire.tools(0)[1]["strict"] is True
        assert wire.tools(1) == [_legacy(PLAIN_TOOL), _legacy(STRICT_TOOL)]
        assert json.dumps(result.provider_request["tools"]) == json.dumps(wire.tools(1))
        warned = _warnings(caplog)
        assert len(warned) == 1 and "lookup_code" in warned[0]

    @pytest.mark.parametrize(
        "message",
        [UNRELATED, THINKING_FORCED, REGEX_BUDGET.format(i=0)],  # tools.0 is not strict
    )
    async def test_unmatched_400_retries_once_without_remembering(
        self, claude: AnthropicProvider, message: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        wire = Wire(
            claude,
            (400, _error(message)),
            (200, _message_body()),
            (200, _message_body()),
        )
        with caplog.at_level(logging.WARNING, logger="dendrux.llm.anthropic"):
            result = await claude.complete(TEXT, tools=[PLAIN_TOOL, STRICT_TOOL])
            await claude.complete(TEXT, tools=[PLAIN_TOOL, STRICT_TOOL])
        assert result.text == "ok"
        assert wire.tools(1) == [_legacy(PLAIN_TOOL), _legacy(STRICT_TOOL)]
        assert result.provider_request["tools"] == wire.tools(1)
        assert wire.tools(2)[1]["strict"] is True  # not remembered
        assert not anthropic_mod._STRICT_REJECTED
        warned = _warnings(caplog)
        assert len(warned) == 1 and message[:40] in warned[0] and "lookup_code" in warned[0]

    async def test_unmatched_400_second_failure_raises_second(
        self, claude: AnthropicProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        wire = Wire(claude, (400, _error(UNRELATED)), (400, _error(THINKING_FORCED)))
        with (
            caplog.at_level(logging.WARNING, logger="dendrux.llm.anthropic"),
            pytest.raises(ProviderError) as info,
        ):
            await claude.complete(TEXT, tools=[STRICT_TOOL])
        assert info.value.category == "rejected"
        assert isinstance(info.value.__cause__, anthropic.BadRequestError)
        assert THINKING_FORCED in str(info.value.__cause__)
        assert THINKING_FORCED not in str(info.value)
        assert UNRELATED not in _chained(info.value)
        assert len(wire.bodies) == 2
        assert not anthropic_mod._STRICT_REJECTED
        assert not _warnings(caplog)

    async def test_gateway_shaped_400_retries(self, claude: AnthropicProvider) -> None:
        wire = Wire(
            claude,
            (400, {"detail": "Extra inputs are not permitted: tools.0.strict"}),
            (200, _message_body()),
        )
        result = await claude.complete(TEXT, tools=[STRICT_TOOL])
        assert result.text == "ok"
        assert wire.tools(1) == [_legacy(STRICT_TOOL)]

    async def test_non_400_errors_raise_unchanged(self, claude: AnthropicProvider) -> None:
        wire = Wire(claude, (529, _error(GRAMMAR_TOO_LARGE, "overloaded_error")))
        with pytest.raises(ProviderUnavailableError) as info:
            await claude.complete(TEXT, tools=[PLAIN_TOOL, STRICT_TOOL])
        assert info.value.status_code == 529
        assert GRAMMAR_TOO_LARGE in str(info.value.__cause__)
        assert len(wire.bodies) == 1

    async def test_no_strict_entries_never_retry(self, claude: AnthropicProvider) -> None:
        wire = Wire(claude, (400, _error(GRAMMAR_TOO_LARGE)))
        with pytest.raises(ProviderError) as info:
            await claude.complete(TEXT, tools=[PLAIN_TOOL])
        assert isinstance(info.value.__cause__, anthropic.BadRequestError)
        assert len(wire.bodies) == 1

    async def test_retries_once_then_raises_second_error(self, claude: AnthropicProvider) -> None:
        wire = Wire(claude, (400, _error(GRAMMAR_TOO_LARGE)), (400, _error(UNRELATED)))
        with pytest.raises(ProviderError) as info:
            await claude.complete(TEXT, tools=[STRICT_TOOL])
        assert UNRELATED in str(info.value.__cause__)
        assert "grammar" not in _chained(info.value.__cause__)
        assert len(wire.bodies) == 2

    async def test_second_strict_like_error_is_not_retried_again(
        self, claude: AnthropicProvider
    ) -> None:
        wire = Wire(claude, (400, _error(GRAMMAR_TOO_LARGE)), (400, _error(GRAMMAR_TOO_LARGE)))
        with pytest.raises(ProviderError, match="rejected the request"):
            await claude.complete(TEXT, tools=[STRICT_TOOL])
        assert len(wire.bodies) == 2

    async def test_rejected_entry_sent_without_strict_afterwards(
        self, claude: AnthropicProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        wire = Wire(
            claude,
            (400, _error(GRAMMAR_TOO_LARGE)),
            (200, _message_body()),
            (200, _message_body()),
        )
        with caplog.at_level(logging.WARNING, logger="dendrux.llm.anthropic"):
            await claude.complete(TEXT, tools=[STRICT_TOOL])
            result = await claude.complete(TEXT, tools=[STRICT_TOOL])
        assert len(wire.bodies) == 3
        assert wire.tools(2) == [_legacy(STRICT_TOOL)]
        assert result.provider_request["tools"] == [_legacy(STRICT_TOOL)]
        assert len(_warnings(caplog)) == 1

    async def test_structured_output_falls_back(
        self, claude: AnthropicProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        schema = {
            "type": "object",
            "properties": {"n": {"type": "integer", "minimum": 0}},
            "required": ["n"],
        }
        tool_use = [
            {"type": "tool_use", "id": "t1", "name": "structured_output", "input": {"n": 1}}
        ]
        wire = Wire(claude, (400, _error(TOO_COMPLEX)), (200, _message_body(tool_use)))
        with caplog.at_level(logging.WARNING, logger="dendrux.llm.anthropic"):
            result = await claude.complete(TEXT, output_schema=schema)
        assert json.loads(result.text or "") == {"n": 1}
        assert wire.tools(0)[0]["strict"] is True
        legacy = {
            "name": "structured_output",
            "description": "Return your response in this exact format.",
            "input_schema": schema,
        }
        assert wire.tools(1) == [legacy]
        assert result.provider_request["tools"] == [legacy]
        assert result.provider_request["tool_choice"] == {
            "type": "tool",
            "name": "structured_output",
        }
        warned = _warnings(caplog)
        assert len(warned) == 1 and "structured_output" in warned[0]

    async def test_image_request_retry_succeeds(self, claude: AnthropicProvider) -> None:
        wire = Wire(claude, (400, _error(GRAMMAR_TOO_LARGE)), (200, _message_body()))
        result = await claude.complete(USER_IMAGE, tools=[STRICT_TOOL])
        assert result.text == "ok"
        assert wire.tools(1) == [_legacy(STRICT_TOOL)]
        assert "data:image" not in json.dumps(result.provider_request)

    async def test_image_request_second_failure_is_typed_and_clean(
        self, claude: AnthropicProvider
    ) -> None:
        Wire(claude, (400, _error(GRAMMAR_TOO_LARGE)), (400, _error("Could not process image")))
        with pytest.raises((ImageRejectedError, ProviderError)) as info:
            await claude.complete(USER_IMAGE, tools=[STRICT_TOOL])
        assert_clean(info.value)
        assert_frames_clean(info.value)


class TestStreamFallback:
    async def test_strict_rejection_retries_before_any_event(
        self, claude: AnthropicProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        wire = Wire(claude, (400, _error(GRAMMAR_TOO_LARGE)), (200, _stream_ok()))
        with caplog.at_level(logging.WARNING, logger="dendrux.llm.anthropic"):
            events = await drain(claude.complete_stream(TEXT, tools=[PLAIN_TOOL, STRICT_TOOL]))
        assert [e.type for e in events] == [StreamEventType.TEXT_DELTA, StreamEventType.DONE]
        assert wire.tools(0)[1]["strict"] is True
        assert wire.tools(1) == [_legacy(PLAIN_TOOL), _legacy(STRICT_TOOL)]
        assert events[-1].raw.provider_request["tools"] == wire.tools(1)
        assert len(_warnings(caplog)) == 1

    async def test_unmatched_400_retries_in_stream(
        self, claude: AnthropicProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        wire = Wire(claude, (400, _error(UNRELATED)), (200, _stream_ok()))
        with caplog.at_level(logging.WARNING, logger="dendrux.llm.anthropic"):
            events = await drain(claude.complete_stream(TEXT, tools=[STRICT_TOOL]))
        assert events[-1].type is StreamEventType.DONE
        assert wire.tools(1) == [_legacy(STRICT_TOOL)]
        assert not anthropic_mod._STRICT_REJECTED
        assert len(_warnings(caplog)) == 1

    async def test_no_strict_stream_error_raises(self, claude: AnthropicProvider) -> None:
        wire = Wire(claude, (400, _error(UNRELATED)))
        with pytest.raises(ProviderError) as info:
            await drain(claude.complete_stream(TEXT, tools=[PLAIN_TOOL]))
        assert isinstance(info.value.__cause__, anthropic.BadRequestError)
        assert len(wire.bodies) == 1

    async def test_retries_once_then_raises(self, claude: AnthropicProvider) -> None:
        wire = Wire(claude, (400, _error(GRAMMAR_TOO_LARGE)), (400, _error(GRAMMAR_TOO_LARGE)))
        with pytest.raises(ProviderError) as info:
            await drain(claude.complete_stream(TEXT, tools=[STRICT_TOOL]))
        assert isinstance(info.value.__cause__, anthropic.BadRequestError)
        assert "grammar" not in _chained(info.value.__cause__)
        assert len(wire.bodies) == 2

    async def test_mid_stream_error_is_not_retried(self, claude: AnthropicProvider) -> None:
        body = sse(*_anthropic_head(), ("error", _error(GRAMMAR_TOO_LARGE)))
        wire = Wire(claude, (200, body))
        with pytest.raises(ProviderError) as info:
            await drain(claude.complete_stream(TEXT, tools=[STRICT_TOOL]))
        assert info.value.category == "rejected"
        assert isinstance(info.value.__cause__, anthropic.APIError)
        assert len(wire.bodies) == 1

    async def test_image_stream_second_failure_is_typed_and_clean(
        self, claude: AnthropicProvider
    ) -> None:
        Wire(claude, (400, _error(GRAMMAR_TOO_LARGE)), (400, _error("Could not process image")))
        with pytest.raises((ImageRejectedError, ProviderError)) as info:
            await drain(claude.complete_stream(USER_IMAGE, tools=[STRICT_TOOL]))
        assert_clean(info.value)
        assert_frames_clean(info.value)


class TestPreSendDowngradeWarning:
    def test_strict_tool_downgrade_warns_once(
        self, claude: AnthropicProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        recursive = ToolDef(
            name="tree",
            description="T",
            parameters={
                "type": "object",
                "properties": {"n": {"$ref": "#/$defs/N"}},
                "required": ["n"],
                "additionalProperties": False,
                "$defs": {
                    "N": {
                        "type": "object",
                        "properties": {"kids": {"type": "array", "items": {"$ref": "#/$defs/N"}}},
                        "required": ["kids"],
                        "additionalProperties": False,
                    }
                },
            },
            strict=True,
        )
        with caplog.at_level(logging.DEBUG, logger="dendrux.llm.anthropic"):
            claude._convert_tools([recursive])
            claude._convert_tools([recursive])
        warned = _warnings(caplog)
        assert len(warned) == 1 and "tree" in warned[0] and "recursive" in warned[0]
        debug = [
            r for r in caplog.records if r.levelno == logging.DEBUG and "tree" in r.getMessage()
        ]
        assert len(debug) == 1

    def test_budget_downgrade_warns(
        self, claude: AnthropicProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        params = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
            "additionalProperties": False,
        }
        tools = [
            ToolDef(name=f"t{i}", description="T", parameters=params, strict=True)
            for i in range(21)
        ]
        with caplog.at_level(logging.WARNING, logger="dendrux.llm.anthropic"):
            claude._convert_tools(tools)
        warned = _warnings(caplog)
        assert len(warned) == 1 and "t20" in warned[0]

    def test_structured_downgrade_stays_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        schema = {
            "type": "object",
            "properties": {"d": {"type": "object", "additionalProperties": {"type": "string"}}},
            "required": ["d"],
        }
        with caplog.at_level(logging.DEBUG, logger="dendrux.llm.anthropic"):
            AnthropicProvider(model="m", api_key="k")._structured_output_tool(schema, model="m")
        assert not _warnings(caplog)
        assert any(r.levelno == logging.DEBUG for r in caplog.records)


class TestStrictOptOut:
    async def test_strict_false_sends_0_4_0_payloads(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        claude = AnthropicProvider(model="claude-sonnet-4-6", api_key="test-key", strict=False)
        schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
        tool_use = [
            {"type": "tool_use", "id": "t1", "name": "structured_output", "input": {"n": 1}}
        ]
        wire = Wire(claude, (200, _message_body()), (200, _message_body(tool_use)))
        with caplog.at_level(logging.DEBUG, logger="dendrux.llm.anthropic"):
            await claude.complete(TEXT, tools=[PLAIN_TOOL, STRICT_TOOL])
            await claude.complete(TEXT, output_schema=schema)
        assert wire.tools(0) == [_legacy(PLAIN_TOOL), _legacy(STRICT_TOOL)]
        assert wire.tools(1) == [
            {
                "name": "structured_output",
                "description": "Return your response in this exact format.",
                "input_schema": schema,
            }
        ]
        assert not caplog.records

    async def test_strict_false_never_retries(self) -> None:
        claude = AnthropicProvider(model="claude-sonnet-4-6", api_key="test-key", strict=False)
        wire = Wire(claude, (400, _error(GRAMMAR_TOO_LARGE)))
        with pytest.raises(ProviderError) as info:
            await claude.complete(TEXT, tools=[STRICT_TOOL])
        assert isinstance(info.value.__cause__, anthropic.BadRequestError)
        assert len(wire.bodies) == 1


class TestRejectedMemory:
    async def test_keyed_by_model(self, claude: AnthropicProvider) -> None:
        wire = Wire(
            claude,
            (400, _error(GRAMMAR_TOO_LARGE)),
            (200, _message_body()),
            (200, _message_body()),
            (200, _message_body()),
        )
        await claude.complete(TEXT, tools=[STRICT_TOOL])
        await claude.complete(TEXT, tools=[STRICT_TOOL], model="claude-haiku-4-5")
        await claude.complete(TEXT, tools=[STRICT_TOOL])
        assert wire.tools(2)[0]["strict"] is True  # other model still strict
        assert wire.tools(3) == [_legacy(STRICT_TOOL)]  # same model remembered

    async def test_bounded_and_evicts_oldest(
        self, claude: AnthropicProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(anthropic_mod, "_STRICT_REJECTED_MAX", 2)
        tools = [
            ToolDef(name=f"t{i}", description="T", parameters=STRICT_PARAMS, strict=True)
            for i in range(3)
        ]
        replies: list[tuple[int, Any]] = []
        for _ in tools:
            replies += [(400, _error(GRAMMAR_TOO_LARGE)), (200, _message_body())]
        replies.append((200, _message_body()))
        wire = Wire(claude, *replies)
        for td in tools:
            await claude.complete(TEXT, tools=[td])
        assert len(anthropic_mod._STRICT_REJECTED) == 2
        await claude.complete(TEXT, tools=tools)
        sent = wire.tools(len(wire.bodies) - 1)
        assert [t.get("strict") for t in sent] == [True, None, None]


async def test_fallback_warning_redacts_echoed_image_data(
    claude: AnthropicProvider, caplog: pytest.LogCaptureFixture
) -> None:
    from tests.unit.test_provider_images_errors import ECHO, LEAK_MARKERS

    Wire(claude, (400, {"detail": ECHO}), (200, _message_body()))
    with caplog.at_level(logging.DEBUG, logger="dendrux.llm.anthropic"):
        await claude.complete(USER_IMAGE, tools=[STRICT_TOOL])
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "[redacted]" in logged
    for marker in LEAK_MARKERS:
        assert marker not in logged
