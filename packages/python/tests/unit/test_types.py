"""Tests for core types."""

import pytest

from dendrite.types import (
    Action,
    AgentStep,
    Clarification,
    Finish,
    LLMResponse,
    Message,
    ProviderCapabilities,
    Role,
    RunResult,
    RunStatus,
    ToolCall,
    ToolDef,
    ToolResult,
    ToolTarget,
    UsageStats,
)


class TestMessage:
    def test_create_user_message(self) -> None:
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"
        assert msg.name is None
        assert msg.tool_calls is None
        assert msg.call_id is None
        assert msg.meta == {}

    def test_create_system_message(self) -> None:
        msg = Message(role=Role.SYSTEM, content="You are helpful")
        assert msg.role == Role.SYSTEM

    def test_create_assistant_message(self) -> None:
        msg = Message(role=Role.ASSISTANT, content="Hello!")
        assert msg.role == Role.ASSISTANT
        assert msg.tool_calls is None

    def test_create_assistant_message_with_tool_calls(self) -> None:
        tc = ToolCall(name="add", params={"a": 1, "b": 2})
        msg = Message(role=Role.ASSISTANT, content="Let me calculate", tool_calls=[tc])
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "add"

    def test_create_tool_message(self) -> None:
        msg = Message(role=Role.TOOL, content='{"result": 42}', name="add", call_id="01ABC")
        assert msg.name == "add"
        assert msg.call_id == "01ABC"

    def test_message_is_frozen(self) -> None:
        msg = Message(role=Role.USER, content="Hello")
        with pytest.raises(AttributeError):
            msg.content = "Changed"  # type: ignore[misc]


class TestMessageInvariants:
    """Role-dependent invariants enforced via __post_init__."""

    def test_tool_message_requires_name(self) -> None:
        with pytest.raises(ValueError, match="TOOL messages require name"):
            Message(role=Role.TOOL, content="result", call_id="01ABC")

    def test_tool_message_requires_call_id(self) -> None:
        with pytest.raises(ValueError, match="TOOL messages require call_id"):
            Message(role=Role.TOOL, content="result", name="add")

    def test_tool_message_cannot_have_tool_calls(self) -> None:
        tc = ToolCall(name="add", params={})
        with pytest.raises(ValueError, match="TOOL messages cannot have tool_calls"):
            Message(
                role=Role.TOOL,
                content="result",
                name="add",
                call_id="01ABC",
                tool_calls=[tc],
            )

    def test_assistant_cannot_have_call_id(self) -> None:
        with pytest.raises(ValueError, match="ASSISTANT messages cannot have call_id"):
            Message(role=Role.ASSISTANT, content="Hi", call_id="01ABC")

    def test_assistant_cannot_have_name(self) -> None:
        with pytest.raises(ValueError, match="ASSISTANT messages cannot have name"):
            Message(role=Role.ASSISTANT, content="Hi", name="bot")

    def test_user_cannot_have_tool_calls(self) -> None:
        tc = ToolCall(name="add", params={})
        with pytest.raises(ValueError, match="USER messages cannot have tool_calls"):
            Message(role=Role.USER, content="Hi", tool_calls=[tc])

    def test_user_cannot_have_call_id(self) -> None:
        with pytest.raises(ValueError, match="USER messages cannot have call_id"):
            Message(role=Role.USER, content="Hi", call_id="01ABC")

    def test_user_cannot_have_name(self) -> None:
        with pytest.raises(ValueError, match="USER messages cannot have name"):
            Message(role=Role.USER, content="Hi", name="alice")

    def test_system_cannot_have_tool_calls(self) -> None:
        tc = ToolCall(name="add", params={})
        with pytest.raises(ValueError, match="SYSTEM messages cannot have tool_calls"):
            Message(role=Role.SYSTEM, content="prompt", tool_calls=[tc])

    def test_system_cannot_have_name(self) -> None:
        with pytest.raises(ValueError, match="SYSTEM messages cannot have name"):
            Message(role=Role.SYSTEM, content="prompt", name="sys")


class TestToolCall:
    def test_tool_call(self) -> None:
        tc = ToolCall(name="search", params={"q": "test"})
        assert tc.name == "search"
        assert tc.params == {"q": "test"}

    def test_tool_call_default_params(self) -> None:
        tc = ToolCall(name="get_metadata")
        assert tc.params == {}

    def test_tool_call_has_dendrite_id(self) -> None:
        tc = ToolCall(name="add", params={"a": 1})
        assert tc.id is not None
        assert len(tc.id) > 0

    def test_tool_call_ids_are_unique(self) -> None:
        tc1 = ToolCall(name="add")
        tc2 = ToolCall(name="add")
        assert tc1.id != tc2.id

    def test_tool_call_provider_id_defaults_none(self) -> None:
        tc = ToolCall(name="add")
        assert tc.provider_tool_call_id is None

    def test_tool_call_with_provider_id(self) -> None:
        tc = ToolCall(
            name="add",
            params={"a": 1},
            provider_tool_call_id="toolu_abc123",
        )
        assert tc.provider_tool_call_id == "toolu_abc123"


class TestActions:
    def test_finish(self) -> None:
        f = Finish(answer="The answer is 42")
        assert f.answer == "The answer is 42"
        assert f.meta == {}

    def test_clarification(self) -> None:
        c = Clarification(question="Which sheet?", options=["Sheet1", "Sheet2"])
        assert c.question == "Which sheet?"
        assert len(c.options) == 2

    def test_action_union(self) -> None:
        """Action type accepts all three variants."""
        actions: list[Action] = [
            ToolCall(name="search", params={"q": "x"}),
            Finish(answer="done"),
            Clarification(question="which?"),
        ]
        assert len(actions) == 3


class TestAgentStep:
    def test_step_with_tool_call(self) -> None:
        step = AgentStep(
            reasoning="I need to search for this",
            action=ToolCall(name="search", params={"q": "test"}),
        )
        assert step.reasoning == "I need to search for this"
        assert isinstance(step.action, ToolCall)

    def test_step_with_finish(self) -> None:
        step = AgentStep(
            reasoning="I have the answer",
            action=Finish(answer="42"),
        )
        assert isinstance(step.action, Finish)

    def test_step_with_no_reasoning(self) -> None:
        step = AgentStep(
            reasoning=None,
            action=Finish(answer="42"),
        )
        assert step.reasoning is None

    def test_step_meta_captures_extra_fields(self) -> None:
        step = AgentStep(
            reasoning="thinking",
            action=Finish(answer="done"),
            meta={"confidence": 0.85, "category": "dcf"},
        )
        assert step.meta["confidence"] == 0.85
        assert step.meta["category"] == "dcf"


class TestToolDef:
    def test_tool_def_defaults(self) -> None:
        td = ToolDef(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        assert td.target == ToolTarget.SERVER
        assert td.parallel is True
        assert td.priority == 0
        assert td.timeout_seconds == 120.0

    def test_client_tool(self) -> None:
        td = ToolDef(
            name="read_range",
            description="Read cells from Excel",
            parameters={},
            target=ToolTarget.CLIENT,
        )
        assert td.target == ToolTarget.CLIENT


class TestToolResult:
    def test_success_result(self) -> None:
        tr = ToolResult(name="add", call_id="01ABC", payload='{"result": 42}')
        assert tr.success is True
        assert tr.error is None
        assert tr.call_id == "01ABC"
        assert tr.payload == '{"result": 42}'

    def test_error_result(self) -> None:
        tr = ToolResult(
            name="add",
            call_id="01ABC",
            payload="null",
            success=False,
            error="Division by zero",
        )
        assert tr.success is False
        assert tr.error == "Division by zero"


class TestProviderCapabilities:
    def test_defaults_are_conservative(self) -> None:
        caps = ProviderCapabilities()
        assert caps.supports_native_tools is False
        assert caps.supports_streaming is False
        assert caps.supports_system_prompt is True  # Most providers support this
        assert caps.max_context_tokens is None

    def test_custom_capabilities(self) -> None:
        caps = ProviderCapabilities(
            supports_native_tools=True,
            supports_tool_call_ids=True,
            supports_streaming=True,
            supports_streaming_tool_deltas=True,
            supports_parallel_tool_calls=True,
            max_context_tokens=200_000,
        )
        assert caps.supports_native_tools is True
        assert caps.max_context_tokens == 200_000


class TestLLMResponse:
    def test_text_response(self) -> None:
        resp = LLMResponse(text="Hello world")
        assert resp.text == "Hello world"
        assert resp.tool_calls is None

    def test_tool_call_response(self) -> None:
        resp = LLMResponse(
            tool_calls=[ToolCall(name="search", params={"q": "test"})],
        )
        assert resp.text is None
        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1

    def test_tool_call_response_has_ids(self) -> None:
        tc = ToolCall(
            name="search",
            params={"q": "test"},
            provider_tool_call_id="toolu_xyz",
        )
        resp = LLMResponse(tool_calls=[tc])
        assert resp.tool_calls is not None
        assert resp.tool_calls[0].id is not None
        assert resp.tool_calls[0].provider_tool_call_id == "toolu_xyz"

    def test_usage_stats(self) -> None:
        usage = UsageStats(input_tokens=100, output_tokens=50, total_tokens=150)
        resp = LLMResponse(text="Hi", usage=usage)
        assert resp.usage.input_tokens == 100
        assert resp.usage.total_tokens == 150


class TestRunResult:
    def test_successful_run(self) -> None:
        result = RunResult(
            run_id="01JQX123",
            status=RunStatus.SUCCESS,
            answer="The answer is 42",
            iteration_count=3,
        )
        assert result.status == RunStatus.SUCCESS
        assert result.answer == "The answer is 42"

    def test_error_run(self) -> None:
        result = RunResult(
            run_id="01JQX456",
            status=RunStatus.ERROR,
            error="LLM call failed",
        )
        assert result.status == RunStatus.ERROR
        assert result.answer is None
        assert result.error == "LLM call failed"

    def test_max_iterations_run(self) -> None:
        result = RunResult(
            run_id="01JQX789",
            status=RunStatus.MAX_ITERATIONS,
            iteration_count=15,
        )
        assert result.status == RunStatus.MAX_ITERATIONS
