"""Tests for core types."""

from dendrite.types import (
    Action,
    AgentStep,
    Clarification,
    Finish,
    LLMResponse,
    Message,
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
        assert msg.meta == {}

    def test_create_tool_message(self) -> None:
        msg = Message(role=Role.TOOL, content='{"result": 42}', name="add")
        assert msg.name == "add"

    def test_message_is_frozen(self) -> None:
        msg = Message(role=Role.USER, content="Hello")
        try:
            msg.content = "Changed"  # type: ignore[misc]
            raise AssertionError("Should have raised FrozenInstanceError")
        except AttributeError:
            pass


class TestActions:
    def test_tool_call(self) -> None:
        tc = ToolCall(name="search", params={"q": "test"})
        assert tc.name == "search"
        assert tc.params == {"q": "test"}

    def test_tool_call_default_params(self) -> None:
        tc = ToolCall(name="get_metadata")
        assert tc.params == {}

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
        assert td.timeout_seconds == 30.0

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
        tr = ToolResult(name="add", result=42)
        assert tr.success is True
        assert tr.error is None

    def test_error_result(self) -> None:
        tr = ToolResult(name="add", result=None, success=False, error="Division by zero")
        assert tr.success is False
        assert tr.error == "Division by zero"


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
