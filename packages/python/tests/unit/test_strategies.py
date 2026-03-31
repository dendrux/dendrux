"""Tests for Strategy protocol and NativeToolCalling."""

from __future__ import annotations

import pytest

from dendrux.strategies.base import Strategy
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import (
    Finish,
    LLMResponse,
    Message,
    Role,
    ToolCall,
    ToolDef,
    ToolResult,
)


@pytest.fixture
def strategy() -> NativeToolCalling:
    return NativeToolCalling()


@pytest.fixture
def sample_tools() -> list[ToolDef]:
    return [
        ToolDef(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
            },
        ),
        ToolDef(
            name="search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {"q": {"type": "string"}},
            },
        ),
    ]


# ------------------------------------------------------------------
# Strategy ABC
# ------------------------------------------------------------------


class TestStrategyABC:
    def test_cannot_instantiate_without_methods(self) -> None:
        with pytest.raises(TypeError):
            Strategy()  # type: ignore[abstract]

    def test_native_tool_calling_is_a_strategy(self, strategy: NativeToolCalling) -> None:
        assert isinstance(strategy, Strategy)


# ------------------------------------------------------------------
# build_messages
# ------------------------------------------------------------------


class TestBuildMessages:
    def test_prepends_system_prompt(
        self, strategy: NativeToolCalling, sample_tools: list[ToolDef]
    ) -> None:
        history = [Message(role=Role.USER, content="Hi")]
        messages, _ = strategy.build_messages(
            system_prompt="You are helpful",
            history=history,
            tool_defs=sample_tools,
        )

        assert len(messages) == 2
        assert messages[0].role == Role.SYSTEM
        assert messages[0].content == "You are helpful"
        assert messages[1].role == Role.USER
        assert messages[1].content == "Hi"

    def test_passes_tools_through(
        self, strategy: NativeToolCalling, sample_tools: list[ToolDef]
    ) -> None:
        history = [Message(role=Role.USER, content="Hi")]
        _, tools = strategy.build_messages(
            system_prompt="prompt",
            history=history,
            tool_defs=sample_tools,
        )

        assert tools is not None
        assert len(tools) == 2
        assert tools[0].name == "add"
        assert tools[1].name == "search"

    def test_empty_tools_returns_none(self, strategy: NativeToolCalling) -> None:
        history = [Message(role=Role.USER, content="Hi")]
        _, tools = strategy.build_messages(
            system_prompt="prompt",
            history=history,
            tool_defs=[],
        )

        assert tools is None

    def test_preserves_full_history(
        self, strategy: NativeToolCalling, sample_tools: list[ToolDef]
    ) -> None:
        tc = ToolCall(name="add", params={"a": 1}, provider_tool_call_id="toolu_1")
        history = [
            Message(role=Role.USER, content="add 1+2"),
            Message(role=Role.ASSISTANT, content="", tool_calls=[tc]),
            Message(
                role=Role.TOOL,
                content='{"result": 3}',
                name="add",
                call_id=tc.id,
            ),
        ]
        messages, _ = strategy.build_messages(
            system_prompt="You are a calculator",
            history=history,
            tool_defs=sample_tools,
        )

        assert len(messages) == 4  # system + 3 history
        assert messages[0].role == Role.SYSTEM
        assert messages[1].role == Role.USER
        assert messages[2].role == Role.ASSISTANT
        assert messages[3].role == Role.TOOL

    def test_does_not_mutate_history(
        self, strategy: NativeToolCalling, sample_tools: list[ToolDef]
    ) -> None:
        history = [Message(role=Role.USER, content="Hi")]
        original_len = len(history)
        strategy.build_messages(
            system_prompt="prompt",
            history=history,
            tool_defs=sample_tools,
        )

        assert len(history) == original_len


# ------------------------------------------------------------------
# parse_response
# ------------------------------------------------------------------


class TestParseResponse:
    def test_text_response_becomes_finish(self, strategy: NativeToolCalling) -> None:
        response = LLMResponse(text="The answer is 42")
        step = strategy.parse_response(response)

        assert isinstance(step.action, Finish)
        assert step.action.answer == "The answer is 42"
        assert step.reasoning is None

    def test_tool_call_response_becomes_tool_call(self, strategy: NativeToolCalling) -> None:
        tc = ToolCall(
            name="add",
            params={"a": 1, "b": 2},
            provider_tool_call_id="toolu_abc",
        )
        response = LLMResponse(
            text="Let me calculate",
            tool_calls=[tc],
        )
        step = strategy.parse_response(response)

        assert isinstance(step.action, ToolCall)
        assert step.action.name == "add"
        assert step.action.params == {"a": 1, "b": 2}
        assert step.action.provider_tool_call_id == "toolu_abc"
        assert step.reasoning == "Let me calculate"

    def test_multiple_tool_calls_returns_first(self, strategy: NativeToolCalling) -> None:
        tc1 = ToolCall(name="add", params={"a": 1})
        tc2 = ToolCall(name="mul", params={"b": 2})
        response = LLMResponse(tool_calls=[tc1, tc2])
        step = strategy.parse_response(response)

        assert isinstance(step.action, ToolCall)
        assert step.action.name == "add"

    def test_multiple_tool_calls_stored_in_meta(self, strategy: NativeToolCalling) -> None:
        tc1 = ToolCall(name="add", params={})
        tc2 = ToolCall(name="mul", params={})
        response = LLMResponse(tool_calls=[tc1, tc2])
        step = strategy.parse_response(response)

        assert step.meta["all_tool_calls"] == [tc1, tc2]

    def test_empty_text_response_becomes_finish(self, strategy: NativeToolCalling) -> None:
        response = LLMResponse(text=None)
        step = strategy.parse_response(response)

        assert isinstance(step.action, Finish)
        assert step.action.answer == ""

    def test_raw_response_preserved(self, strategy: NativeToolCalling) -> None:
        response = LLMResponse(text="answer")
        step = strategy.parse_response(response)

        assert step.raw_response == "answer"


# ------------------------------------------------------------------
# format_tool_result
# ------------------------------------------------------------------


class TestFormatToolResult:
    def test_success_result_becomes_tool_message(self, strategy: NativeToolCalling) -> None:
        tc = ToolCall(name="add", params={"a": 1}, provider_tool_call_id="toolu_1")
        result = ToolResult(
            name="add",
            call_id=tc.id,
            payload='{"result": 3}',
            success=True,
        )
        msg = strategy.format_tool_result(result)

        assert msg.role == Role.TOOL
        assert msg.content == '{"result": 3}'
        assert msg.name == "add"
        assert msg.call_id == tc.id
        assert msg.meta == {}

    def test_error_result_sets_is_error_meta(self, strategy: NativeToolCalling) -> None:
        result = ToolResult(
            name="div",
            call_id="01ABC",
            payload="Division by zero",
            success=False,
            error="Division by zero",
        )
        msg = strategy.format_tool_result(result)

        assert msg.role == Role.TOOL
        assert msg.meta["is_error"] is True

    def test_result_message_has_correct_call_id(self, strategy: NativeToolCalling) -> None:
        result = ToolResult(
            name="search",
            call_id="01DENDRITE_ID",
            payload='{"results": []}',
        )
        msg = strategy.format_tool_result(result)

        assert msg.call_id == "01DENDRITE_ID"
        assert msg.name == "search"
