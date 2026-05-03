"""Tests for OpenTelemetryNotifier.

Deterministic span assertions using ``InMemorySpanExporter``. We never
hit a real LLM or a real OTel backend.

Coverage matches the reviewer-required matrix:
  - success
  - LLM failure / run failure / tool failure
  - safe-defaults vs include_messages / include_tool_params opt-ins
  - concurrent runs sharing one notifier
  - parent-context inheritance
  - fail-open on broken tracer
  - governance events as span events
"""

from __future__ import annotations

from typing import Any

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from dendrux import Agent, run, tool
from dendrux.llm.base import LLMProvider
from dendrux.llm.mock import MockLLM
from dendrux.loops.react import ReActLoop
from dendrux.notifiers.otel import OpenTelemetryNotifier
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import LLMResponse, RunStatus, ToolCall, UsageStats


@tool()
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool()
async def boom(_: str = "") -> str:
    """Always errors."""
    raise RuntimeError("tool exploded")


@pytest.fixture(scope="session")
def _provider_and_exporter() -> tuple[TracerProvider, InMemorySpanExporter]:
    """Process-global TracerProvider — OTel disallows override after first set."""
    provider = TracerProvider()
    exp = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    otel_trace.set_tracer_provider(provider)
    return provider, exp


@pytest.fixture
def exporter(
    _provider_and_exporter: tuple[TracerProvider, InMemorySpanExporter],
) -> InMemorySpanExporter:
    """Per-test view of the session-scoped exporter — cleared on entry."""
    _, exp = _provider_and_exporter
    exp.clear()
    yield exp
    exp.clear()


def _spans_by_name(exp: InMemorySpanExporter, prefix: str) -> list[Any]:
    return [s for s in exp.get_finished_spans() if s.name.startswith(prefix)]


def _attr(span: Any, key: str) -> Any:
    return span.attributes.get(key)


# ------------------------------------------------------------------
# Happy path
# ------------------------------------------------------------------


class TestSuccessfulRun:
    async def test_emits_invoke_agent_span_with_namespaced_attrs(
        self, exporter: InMemorySpanExporter
    ) -> None:
        notifier = OpenTelemetryNotifier()
        llm = MockLLM(
            [LLMResponse(text="done", usage=UsageStats(input_tokens=12, output_tokens=3))]
        )
        agent = Agent(name="alice", prompt="Test.", tools=[add])

        await run(agent, provider=llm, user_input="Hi", extra_notifier=notifier)

        invoke_spans = _spans_by_name(exporter, "invoke_agent")
        assert len(invoke_spans) == 1
        s = invoke_spans[0]
        assert _attr(s, "gen_ai.operation.name") == "invoke_agent"
        assert _attr(s, "gen_ai.agent.name") == "alice"
        assert _attr(s, "dendrux.framework") == "dendrux"
        assert _attr(s, "dendrux.run.id")
        assert _attr(s, "dendrux.run.status") == RunStatus.SUCCESS.value
        assert s.status.status_code == StatusCode.OK

    async def test_chat_span_carries_model_and_token_usage(
        self, exporter: InMemorySpanExporter
    ) -> None:
        notifier = OpenTelemetryNotifier()
        llm = MockLLM(
            [LLMResponse(text="ok", usage=UsageStats(input_tokens=42, output_tokens=7))],
            model="gpt-fake",
        )
        agent = Agent(prompt="Test.", tools=[add])

        await run(agent, provider=llm, user_input="Hi", extra_notifier=notifier)

        chat_spans = _spans_by_name(exporter, "chat")
        assert len(chat_spans) == 1
        c = chat_spans[0]
        assert _attr(c, "gen_ai.operation.name") == "chat"
        assert _attr(c, "gen_ai.request.model") == "gpt-fake"
        assert _attr(c, "gen_ai.usage.input_tokens") == 42
        assert _attr(c, "gen_ai.usage.output_tokens") == 7

    async def test_chat_span_is_child_of_invoke_agent_span(
        self, exporter: InMemorySpanExporter
    ) -> None:
        notifier = OpenTelemetryNotifier()
        llm = MockLLM([LLMResponse(text="done")])
        agent = Agent(prompt="Test.", tools=[add])

        await run(agent, provider=llm, user_input="Hi", extra_notifier=notifier)

        invoke = _spans_by_name(exporter, "invoke_agent")[0]
        chat = _spans_by_name(exporter, "chat")[0]
        assert chat.parent is not None
        assert chat.parent.span_id == invoke.context.span_id
        assert chat.context.trace_id == invoke.context.trace_id

    async def test_execute_tool_span_carries_tool_attrs(
        self, exporter: InMemorySpanExporter
    ) -> None:
        notifier = OpenTelemetryNotifier()
        tc = ToolCall(name="add", params={"a": 1, "b": 2}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="3")])
        agent = Agent(prompt="Test.", tools=[add])

        await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="1+2?",
            run_id="run-tool-1",
            notifier=notifier,
        )

        tool_spans = _spans_by_name(exporter, "execute_tool")
        assert len(tool_spans) == 1
        t = tool_spans[0]
        assert _attr(t, "gen_ai.operation.name") == "execute_tool"
        assert _attr(t, "dendrux.tool.name") == "add"
        assert _attr(t, "dendrux.tool.call_id") == tc.id
        assert _attr(t, "dendrux.tool.success") is True
        assert t.status.status_code == StatusCode.OK


# ------------------------------------------------------------------
# Failure paths
# ------------------------------------------------------------------


class TestFailures:
    async def test_llm_failure_marks_chat_span_error_and_records_exception(
        self, exporter: InMemorySpanExporter
    ) -> None:
        class BoomLLM(LLMProvider):
            @property
            def model(self) -> str:
                return "boom-model"

            async def complete(self, *a: Any, **kw: Any) -> Any:
                raise RuntimeError("provider crashed")

            async def complete_stream(self, *a: Any, **kw: Any) -> Any:
                raise RuntimeError("provider crashed")

        notifier = OpenTelemetryNotifier()
        agent = Agent(prompt="Test.", tools=[add])

        with pytest.raises(RuntimeError, match="provider crashed"):
            await run(agent, provider=BoomLLM(), user_input="Hi", extra_notifier=notifier)

        chat = _spans_by_name(exporter, "chat")
        assert len(chat) == 1
        assert chat[0].status.status_code == StatusCode.ERROR
        assert any(e.name == "exception" for e in chat[0].events)

    async def test_run_failure_marks_invoke_agent_span_error(
        self, exporter: InMemorySpanExporter
    ) -> None:
        class BoomLLM(LLMProvider):
            @property
            def model(self) -> str:
                return "boom-model"

            async def complete(self, *a: Any, **kw: Any) -> Any:
                raise RuntimeError("provider crashed")

            async def complete_stream(self, *a: Any, **kw: Any) -> Any:
                raise RuntimeError("provider crashed")

        notifier = OpenTelemetryNotifier()
        agent = Agent(prompt="Test.", tools=[add])

        with pytest.raises(RuntimeError):
            await run(agent, provider=BoomLLM(), user_input="Hi", extra_notifier=notifier)

        invoke = _spans_by_name(exporter, "invoke_agent")
        assert len(invoke) == 1
        assert invoke[0].status.status_code == StatusCode.ERROR
        assert any(e.name == "exception" for e in invoke[0].events)

    async def test_tool_failure_marks_execute_tool_span_with_success_false(
        self, exporter: InMemorySpanExporter
    ) -> None:
        notifier = OpenTelemetryNotifier()
        tc = ToolCall(name="boom", params={}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="recovered")])
        agent = Agent(prompt="Test.", tools=[boom])

        await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="boom",
            run_id="run-tool-fail",
            notifier=notifier,
        )

        tool_spans = _spans_by_name(exporter, "execute_tool")
        assert len(tool_spans) == 1
        assert _attr(tool_spans[0], "dendrux.tool.success") is False
        assert tool_spans[0].status.status_code == StatusCode.ERROR


# ------------------------------------------------------------------
# Safe defaults & opt-in flags
# ------------------------------------------------------------------


class TestSafeDefaults:
    async def test_no_completion_or_tool_params_attrs_by_default(
        self, exporter: InMemorySpanExporter
    ) -> None:
        notifier = OpenTelemetryNotifier()
        tc = ToolCall(name="add", params={"a": 1, "b": 2}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="secret answer")])
        agent = Agent(prompt="Test.", tools=[add])

        await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="1+2?",
            run_id="run-safe",
            notifier=notifier,
        )

        for c in _spans_by_name(exporter, "chat"):
            assert "gen_ai.completion" not in c.attributes
        for t in _spans_by_name(exporter, "execute_tool"):
            assert "dendrux.tool.params" not in t.attributes

    async def test_include_messages_attaches_completion_text(
        self, exporter: InMemorySpanExporter
    ) -> None:
        notifier = OpenTelemetryNotifier(include_messages=True)
        llm = MockLLM([LLMResponse(text="completion-payload")])
        agent = Agent(prompt="Test.", tools=[add])

        await run(agent, provider=llm, user_input="Hi", extra_notifier=notifier)

        chat = _spans_by_name(exporter, "chat")[0]
        assert _attr(chat, "gen_ai.completion") == "completion-payload"

    async def test_include_tool_params_serializes_params_as_json(
        self, exporter: InMemorySpanExporter
    ) -> None:
        notifier = OpenTelemetryNotifier(include_tool_params=True)
        tc = ToolCall(name="add", params={"a": 1, "b": 2}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="3")])
        agent = Agent(prompt="Test.", tools=[add])

        await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="1+2?",
            run_id="run-params",
            notifier=notifier,
        )

        t = _spans_by_name(exporter, "execute_tool")[0]
        assert _attr(t, "dendrux.tool.params") == '{"a": 1, "b": 2}'


# ------------------------------------------------------------------
# Concurrency & shared notifier
# ------------------------------------------------------------------


class TestConcurrentRuns:
    async def test_two_runs_sharing_notifier_dont_cross_contaminate_spans(
        self, exporter: InMemorySpanExporter
    ) -> None:
        import asyncio

        notifier = OpenTelemetryNotifier()
        agent_a = Agent(name="A", prompt="A", tools=[add])
        agent_b = Agent(name="B", prompt="B", tools=[add])
        llm_a = MockLLM([LLMResponse(text="a")], model="model-a")
        llm_b = MockLLM([LLMResponse(text="b")], model="model-b")

        await asyncio.gather(
            run(agent_a, provider=llm_a, user_input="hi-a", extra_notifier=notifier),
            run(agent_b, provider=llm_b, user_input="hi-b", extra_notifier=notifier),
        )

        invoke = _spans_by_name(exporter, "invoke_agent")
        assert len(invoke) == 2
        names = {_attr(s, "gen_ai.agent.name") for s in invoke}
        assert names == {"A", "B"}
        # Each chat span's parent must be its own invoke span.
        chats = _spans_by_name(exporter, "chat")
        assert len(chats) == 2
        invoke_by_id = {s.context.span_id: s for s in invoke}
        for c in chats:
            parent = invoke_by_id[c.parent.span_id]
            assert _attr(c, "gen_ai.request.model") == _attr(parent, "gen_ai.request.model")


# ------------------------------------------------------------------
# Parent context inheritance
# ------------------------------------------------------------------


class TestParentContext:
    async def test_invoke_agent_span_inherits_active_parent(
        self, exporter: InMemorySpanExporter
    ) -> None:
        notifier = OpenTelemetryNotifier()
        llm = MockLLM([LLMResponse(text="done")])
        agent = Agent(prompt="Test.", tools=[add])

        tracer = otel_trace.get_tracer("test-host")
        with tracer.start_as_current_span("http.request") as outer:
            await run(agent, provider=llm, user_input="Hi", extra_notifier=notifier)
            outer_ctx = outer.context

        invoke = _spans_by_name(exporter, "invoke_agent")[0]
        assert invoke.parent is not None
        assert invoke.parent.span_id == outer_ctx.span_id
        assert invoke.context.trace_id == outer_ctx.trace_id


# ------------------------------------------------------------------
# Fail-open contract
# ------------------------------------------------------------------


class TestFailOpen:
    async def test_broken_tracer_does_not_kill_run(self, exporter: InMemorySpanExporter) -> None:
        class BrokenTracer:
            def start_span(self, *a: Any, **kw: Any) -> Any:
                raise RuntimeError("tracer exploded")

        notifier = OpenTelemetryNotifier(tracer=BrokenTracer())
        llm = MockLLM([LLMResponse(text="done")])
        agent = Agent(prompt="Test.", tools=[add])

        # Run completes successfully despite tracer failures.
        result = await run(agent, provider=llm, user_input="Hi", extra_notifier=notifier)
        assert result.status == RunStatus.SUCCESS


# ------------------------------------------------------------------
# Governance events as span events
# ------------------------------------------------------------------


class TestGovernanceEvents:
    async def test_governance_event_attached_to_invoke_agent_span(
        self, exporter: InMemorySpanExporter
    ) -> None:
        notifier = OpenTelemetryNotifier()

        @tool()
        async def secret() -> str:
            """Forbidden tool."""
            return "shh"

        tc = ToolCall(name="secret", params={}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="done")])
        agent = Agent(prompt="Test.", tools=[secret], deny=["secret"])

        await run(agent, provider=llm, user_input="hi", extra_notifier=notifier)

        invoke = _spans_by_name(exporter, "invoke_agent")[0]
        gov_events = [e for e in invoke.events if e.name.startswith("governance.")]
        assert any(e.name == "governance.policy.denied" for e in gov_events)


# ------------------------------------------------------------------
# Import guard
# ------------------------------------------------------------------


class TestImportGuard:
    async def test_constructor_errors_when_otel_unavailable(self) -> None:
        import dendrux.notifiers.otel as otel_mod

        original = otel_mod._OTEL_AVAILABLE
        otel_mod._OTEL_AVAILABLE = False
        try:
            with pytest.raises(ImportError, match="dendrux\\[otel\\]"):
                OpenTelemetryNotifier()
        finally:
            otel_mod._OTEL_AVAILABLE = original
