"""Tests for pause signal and PauseState serialization (Sprint 3, Group 1)."""

from __future__ import annotations

from dendrite.agent import Agent
from dendrite.llm.mock import MockLLM
from dendrite.loops.react import ReActLoop
from dendrite.strategies.native import NativeToolCalling
from dendrite.tool import tool
from dendrite.types import (
    AgentStep,
    Clarification,
    Finish,
    LLMResponse,
    Message,
    PauseState,
    Role,
    RunStatus,
    ToolCall,
    UsageStats,
)

# ------------------------------------------------------------------
# Test tools
# ------------------------------------------------------------------


@tool()
async def server_add(a: int, b: int) -> int:
    """Server-side add."""
    return a + b


@tool(target="client")
async def read_range(sheet: str) -> str:
    """Client-side tool — reads from Excel."""
    return "should never run"


@tool(target="human")
async def ask_user(question: str) -> str:
    """Human tool — asks a clarifying question."""
    return "should never run"


def _make_agent(**overrides) -> Agent:
    defaults = {
        "prompt": "You are a test agent.",
        "tools": [server_add, read_range],
        "max_iterations": 10,
    }
    defaults.update(overrides)
    return Agent(**defaults)


# ------------------------------------------------------------------
# Pause behavior
# ------------------------------------------------------------------


class TestPauseBehavior:
    async def test_client_tool_pauses_loop(self) -> None:
        """LLM calls a client tool → loop returns WAITING_CLIENT_TOOL."""
        tc = ToolCall(name="read_range", params={"sheet": "Sheet1"}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc])])
        agent = _make_agent()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Read the sheet",
        )

        assert result.status == RunStatus.WAITING_CLIENT_TOOL
        pause_state = result.meta["pause_state"]
        assert isinstance(pause_state, PauseState)
        assert len(pause_state.pending_tool_calls) == 1
        assert pause_state.pending_tool_calls[0].name == "read_range"

    async def test_human_tool_pauses_loop(self) -> None:
        """target=human also → WAITING_CLIENT_TOOL (D3)."""
        tc = ToolCall(
            name="ask_user", params={"question": "Which sheet?"}, provider_tool_call_id="t1"
        )
        llm = MockLLM([LLMResponse(tool_calls=[tc])])
        agent = _make_agent(tools=[server_add, ask_user])

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Help me",
        )

        assert result.status == RunStatus.WAITING_CLIENT_TOOL
        assert result.meta["pause_state"].pending_tool_calls[0].name == "ask_user"

    async def test_mixed_tools_executes_server_pauses_client(self) -> None:
        """LLM calls [server_add, read_range] → server executes, client pending."""
        tc_server = ToolCall(name="server_add", params={"a": 1, "b": 2}, provider_tool_call_id="t1")
        tc_client = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t2")
        llm = MockLLM([LLMResponse(tool_calls=[tc_server, tc_client])])
        agent = _make_agent()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Do both",
        )

        assert result.status == RunStatus.WAITING_CLIENT_TOOL
        pause_state = result.meta["pause_state"]
        # Only client tool is pending
        assert len(pause_state.pending_tool_calls) == 1
        assert pause_state.pending_tool_calls[0].name == "read_range"
        # Server tool result should be in history
        tool_msgs = [m for m in pause_state.history if m.role == Role.TOOL]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].name == "server_add"

    async def test_all_server_tools_no_pause(self) -> None:
        """All-server tool calls execute normally — no pause."""
        tc = ToolCall(name="server_add", params={"a": 5, "b": 3}, provider_tool_call_id="t1")
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[tc]),
                LLMResponse(text="The answer is 8"),
            ]
        )
        agent = _make_agent()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Add 5 + 3",
        )

        assert result.status == RunStatus.SUCCESS
        assert result.answer == "The answer is 8"

    async def test_pause_state_has_full_history(self) -> None:
        """PauseState contains the complete conversation so far."""
        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(text="Let me read", tool_calls=[tc])])
        agent = _make_agent()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Read it",
        )

        ps = result.meta["pause_state"]
        # History: user message + assistant message (with tool calls)
        assert len(ps.history) >= 2
        assert ps.history[0].role == Role.USER
        assert ps.history[0].content == "Read it"
        assert ps.history[1].role == Role.ASSISTANT

    async def test_pause_state_has_steps(self) -> None:
        """Steps captured up to the pause point."""
        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc])])
        agent = _make_agent()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Read",
        )

        ps = result.meta["pause_state"]
        assert len(ps.steps) == 1
        assert isinstance(ps.steps[0].action, ToolCall)

    async def test_pause_state_has_usage(self) -> None:
        """Cumulative token usage is captured."""
        tc = ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="t1")
        usage = UsageStats(input_tokens=100, output_tokens=50, total_tokens=150)
        llm = MockLLM([LLMResponse(tool_calls=[tc], usage=usage)])
        agent = _make_agent()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Read",
        )

        ps = result.meta["pause_state"]
        assert ps.usage.input_tokens == 100
        assert ps.usage.output_tokens == 50
        assert ps.usage.total_tokens == 150

    async def test_clarification_produces_pause_state(self) -> None:
        """Clarification action builds a PauseState for resume_with_input."""
        from dendrite.strategies.base import Strategy

        class ClarifyStrategy(Strategy):
            """Strategy that always returns Clarification."""

            def build_messages(self, *, system_prompt, history, tool_defs):
                msgs = [Message(role=Role.SYSTEM, content=system_prompt), *history]
                return msgs, tool_defs or None

            def parse_response(self, response):
                return AgentStep(
                    reasoning=response.text,
                    action=Clarification(question="Which file should I analyze?"),
                )

            def format_tool_result(self, result):
                return Message(
                    role=Role.TOOL, content=result.payload, name=result.name, call_id=result.call_id
                )

        llm = MockLLM([LLMResponse(text="I need to ask a question")])
        agent = _make_agent(tools=[server_add])

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=ClarifyStrategy(),
            user_input="Analyze something",
        )

        assert result.status == RunStatus.WAITING_HUMAN_INPUT
        assert result.answer == "Which file should I analyze?"
        # Must have pause_state for resume_with_input
        assert "pause_state" in result.meta
        ps = result.meta["pause_state"]
        assert isinstance(ps, PauseState)
        assert ps.pending_tool_calls == []  # No tools — resume is a user message
        assert len(ps.history) >= 2  # user + assistant
        assert ps.agent_name == agent.name
        assert ps.iteration == 1


# ------------------------------------------------------------------
# PauseState serialization (D7 — core infrastructure)
# ------------------------------------------------------------------


class TestPauseStateSerialization:
    def test_roundtrip_simple(self) -> None:
        """Minimal PauseState round-trips through to_dict/from_dict."""
        ps = PauseState(
            agent_name="TestAgent",
            pending_tool_calls=[ToolCall(name="read", params={"x": 1}, id="tc1")],
            history=[Message(role=Role.USER, content="hello")],
            steps=[],
            iteration=1,
            trace_order_offset=2,
            usage=UsageStats(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        restored = PauseState.from_dict(ps.to_dict())
        assert restored.agent_name == "TestAgent"
        assert restored.iteration == 1
        assert restored.trace_order_offset == 2
        assert len(restored.pending_tool_calls) == 1
        assert restored.pending_tool_calls[0].name == "read"
        assert restored.pending_tool_calls[0].id == "tc1"
        assert restored.history[0].content == "hello"

    def test_roundtrip_with_tool_calls(self) -> None:
        """Message with tool_calls list survives round-trip."""
        tc = ToolCall(name="add", params={"a": 1, "b": 2}, id="tc1", provider_tool_call_id="p1")
        msg = Message(role=Role.ASSISTANT, content="calling", tool_calls=[tc])
        ps = PauseState(
            agent_name="A",
            pending_tool_calls=[],
            history=[msg],
            steps=[],
            iteration=1,
            trace_order_offset=0,
            usage=UsageStats(),
        )
        restored = PauseState.from_dict(ps.to_dict())
        assert restored.history[0].tool_calls is not None
        assert len(restored.history[0].tool_calls) == 1
        assert restored.history[0].tool_calls[0].provider_tool_call_id == "p1"

    def test_roundtrip_with_mixed_actions(self) -> None:
        """Steps containing ToolCall and Finish actions survive round-trip."""
        steps = [
            AgentStep(reasoning="thinking", action=ToolCall(name="add", params={}, id="tc1")),
            AgentStep(reasoning="done", action=Finish(answer="42")),
        ]
        ps = PauseState(
            agent_name="A",
            pending_tool_calls=[],
            history=[Message(role=Role.USER, content="hi")],
            steps=steps,
            iteration=2,
            trace_order_offset=5,
            usage=UsageStats(),
        )
        restored = PauseState.from_dict(ps.to_dict())
        assert isinstance(restored.steps[0].action, ToolCall)
        assert restored.steps[0].action.name == "add"
        assert isinstance(restored.steps[1].action, Finish)
        assert restored.steps[1].action.answer == "42"

    def test_roundtrip_with_clarification(self) -> None:
        """Clarification action in steps survives round-trip."""
        steps = [
            AgentStep(
                reasoning="need info",
                action=Clarification(question="Which file?", options=["a.csv", "b.csv"]),
            ),
        ]
        ps = PauseState(
            agent_name="A",
            pending_tool_calls=[],
            history=[Message(role=Role.USER, content="analyze")],
            steps=steps,
            iteration=1,
            trace_order_offset=0,
            usage=UsageStats(),
        )
        restored = PauseState.from_dict(ps.to_dict())
        action = restored.steps[0].action
        assert isinstance(action, Clarification)
        assert action.question == "Which file?"
        assert action.options == ["a.csv", "b.csv"]

    def test_roundtrip_after_mixed_turn(self) -> None:
        """Server results in history + client calls pending."""
        # Simulate: assistant called [server_add, read_range]
        # server_add result already in history, read_range is pending
        tc_pending = ToolCall(name="read_range", params={"sheet": "S1"}, id="tc2")
        history = [
            Message(role=Role.USER, content="do both"),
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(name="server_add", params={"a": 1, "b": 2}, id="tc1"),
                    tc_pending,
                ],
            ),
            Message(role=Role.TOOL, content="3", name="server_add", call_id="tc1"),
        ]
        ps = PauseState(
            agent_name="A",
            pending_tool_calls=[tc_pending],
            history=history,
            steps=[
                AgentStep(reasoning="", action=ToolCall(name="server_add", params={}, id="tc1"))
            ],
            iteration=1,
            trace_order_offset=3,
            usage=UsageStats(input_tokens=50, output_tokens=20, total_tokens=70),
        )
        restored = PauseState.from_dict(ps.to_dict())
        assert len(restored.history) == 3
        assert restored.history[2].role == Role.TOOL
        assert restored.history[2].name == "server_add"
        assert len(restored.pending_tool_calls) == 1
        assert restored.pending_tool_calls[0].name == "read_range"

    def test_roundtrip_preserves_provider_ids(self) -> None:
        """provider_tool_call_id on ToolCall survives serialization."""
        tc = ToolCall(name="add", params={}, id="tc1", provider_tool_call_id="toolu_abc123")
        ps = PauseState(
            agent_name="A",
            pending_tool_calls=[tc],
            history=[],
            steps=[],
            iteration=1,
            trace_order_offset=0,
            usage=UsageStats(),
        )
        restored = PauseState.from_dict(ps.to_dict())
        assert restored.pending_tool_calls[0].provider_tool_call_id == "toolu_abc123"

    def test_roundtrip_with_none_fields(self) -> None:
        """Optional fields (cost_usd=None, tool_calls=None) survive."""
        ps = PauseState(
            agent_name="A",
            pending_tool_calls=[],
            history=[
                Message(role=Role.USER, content="hi"),
                Message(role=Role.ASSISTANT, content="hello"),  # no tool_calls
            ],
            steps=[AgentStep(reasoning=None, action=Finish(answer="done"), raw_response=None)],
            iteration=1,
            trace_order_offset=2,
            usage=UsageStats(input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=None),
        )
        restored = PauseState.from_dict(ps.to_dict())
        assert restored.history[1].tool_calls is None
        assert restored.steps[0].reasoning is None
        assert restored.steps[0].raw_response is None
        assert restored.usage.cost_usd is None

    def test_roundtrip_step_meta_with_all_tool_calls(self) -> None:
        """AgentStep.meta['all_tool_calls'] (list of ToolCall) round-trips."""
        tc1 = ToolCall(name="add", params={"a": 1}, id="tc1")
        tc2 = ToolCall(name="read", params={"s": "A"}, id="tc2")
        step = AgentStep(
            reasoning="multi",
            action=tc1,
            meta={"all_tool_calls": [tc1, tc2]},
        )
        ps = PauseState(
            agent_name="A",
            pending_tool_calls=[tc2],
            history=[],
            steps=[step],
            iteration=1,
            trace_order_offset=0,
            usage=UsageStats(),
        )
        restored = PauseState.from_dict(ps.to_dict())
        all_tc = restored.steps[0].meta["all_tool_calls"]
        assert len(all_tc) == 2
        assert isinstance(all_tc[0], ToolCall)
        assert all_tc[1].name == "read"

    def test_to_dict_is_json_serializable(self) -> None:
        """to_dict() output must be json.dumps-safe — validates on call."""
        import json

        ps = PauseState(
            agent_name="A",
            pending_tool_calls=[ToolCall(name="read", params={"x": 1}, id="tc1")],
            history=[
                Message(role=Role.USER, content="hi"),
                Message(
                    role=Role.ASSISTANT,
                    content="calling",
                    tool_calls=[
                        ToolCall(name="read", params={"x": 1}, id="tc1", provider_tool_call_id="p1")
                    ],
                ),
            ],
            steps=[AgentStep(reasoning="think", action=Finish(answer="done"))],
            iteration=1,
            trace_order_offset=2,
            usage=UsageStats(input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.01),
        )
        d = ps.to_dict()
        # Must not raise
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_non_serializable_step_meta_coerced_to_string(self) -> None:
        """Non-JSON-serializable values in step.meta are coerced to str."""
        from datetime import datetime

        dt = datetime(2026, 3, 15, 12, 0, 0)
        step = AgentStep(
            reasoning="x",
            action=Finish(answer="done"),
            meta={"timestamp": dt, "count": 42},
        )
        ps = PauseState(
            agent_name="A",
            pending_tool_calls=[],
            history=[],
            steps=[step],
            iteration=1,
            trace_order_offset=0,
            usage=UsageStats(),
        )
        d = ps.to_dict()
        # datetime should be coerced to string, int should pass through
        assert d["steps"][0]["meta"]["timestamp"] == str(dt)
        assert d["steps"][0]["meta"]["count"] == 42

    def test_non_serializable_message_meta_coerced_to_string(self) -> None:
        """Non-JSON-serializable values in message.meta are coerced to str."""
        msg = Message(
            role=Role.USER,
            content="hi",
            meta={"obj": object()},
        )
        ps = PauseState(
            agent_name="A",
            pending_tool_calls=[],
            history=[msg],
            steps=[],
            iteration=1,
            trace_order_offset=0,
            usage=UsageStats(),
        )
        d = ps.to_dict()
        # object() should be coerced to its str representation
        assert isinstance(d["history"][0]["meta"]["obj"], str)
