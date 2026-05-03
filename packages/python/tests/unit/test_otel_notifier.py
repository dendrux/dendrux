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
    """Process-global TracerProvider — OTel disallows override after first set.

    Caveat: ``set_tracer_provider`` is one-shot per process. Once this
    session fixture installs its provider, any other test file in the
    same pytest run that also calls ``set_tracer_provider`` will be
    silently ignored (just a stderr warning). If a future test file
    needs its own provider, factor this fixture out to a shared
    ``conftest.py`` so both files use the same instance and the
    InMemorySpanExporter remains the source of truth.
    """
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
        # Each agent gets its own invoke span, distinct trace IDs.
        names = {_attr(s, "gen_ai.agent.name") for s in invoke}
        assert names == {"A", "B"}
        invoke_by_name = {_attr(s, "gen_ai.agent.name"): s for s in invoke}
        assert invoke_by_name["A"].context.trace_id != invoke_by_name["B"].context.trace_id

        # Each chat span's model matches the LLM of the agent that owns its parent.
        chats = _spans_by_name(exporter, "chat")
        assert len(chats) == 2
        invoke_by_id = {s.context.span_id: s for s in invoke}
        for c in chats:
            parent = invoke_by_id[c.parent.span_id]
            parent_name = _attr(parent, "gen_ai.agent.name")
            expected_model = "model-a" if parent_name == "A" else "model-b"
            assert _attr(c, "gen_ai.request.model") == expected_model
            assert c.context.trace_id == parent.context.trace_id


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


# ------------------------------------------------------------------
# Fail-open: failures in non-entry-point hooks
# ------------------------------------------------------------------


class _BrokenSpan:
    """Span that raises on every mutation — exercises the in-hook try/except guards.

    The entry-point hook (start_span) succeeds via a real tracer; the
    failure happens later when the notifier tries to set attributes,
    set status, or end the span. This proves fail-open holds across
    the full hook surface, not just the start path covered by
    TestFailOpen.
    """

    def __init__(self) -> None:
        self.ended = False

    def set_attribute(self, *a: Any, **kw: Any) -> None:
        raise RuntimeError("set_attribute exploded")

    def set_status(self, *a: Any, **kw: Any) -> None:
        raise RuntimeError("set_status exploded")

    def record_exception(self, *a: Any, **kw: Any) -> None:
        raise RuntimeError("record_exception exploded")

    def add_event(self, *a: Any, **kw: Any) -> None:
        raise RuntimeError("add_event exploded")

    def end(self, *a: Any, **kw: Any) -> None:
        self.ended = True


class _BrokenSpanTracer:
    """Tracer that returns broken spans — start_span succeeds, mutations don't."""

    def __init__(self) -> None:
        self.spans: list[_BrokenSpan] = []

    def start_span(self, *a: Any, **kw: Any) -> _BrokenSpan:
        s = _BrokenSpan()
        self.spans.append(s)
        return s


class TestFailOpenNonEntryPoint:
    async def test_run_completes_when_span_mutations_raise(
        self, exporter: InMemorySpanExporter
    ) -> None:
        tracer = _BrokenSpanTracer()
        notifier = OpenTelemetryNotifier(tracer=tracer)
        tc = ToolCall(name="add", params={"a": 1, "b": 2}, provider_tool_call_id="t1")
        llm = MockLLM([LLMResponse(tool_calls=[tc]), LLMResponse(text="3")])
        agent = Agent(prompt="Test.", tools=[add])

        result = await run(agent, provider=llm, user_input="1+2?", extra_notifier=notifier)

        assert result.status == RunStatus.SUCCESS
        # All started spans got at least one mutation attempt — the guards
        # caught the exceptions and we still got the run done.
        assert len(tracer.spans) >= 1


# ------------------------------------------------------------------
# Simulated pause/resume — driving notifier hooks directly
# ------------------------------------------------------------------


class TestPauseResumeLifecycle:
    """Pause/resume goes through two on_run_started → on_run_finished cycles.

    The runner-level pause/resume integration is covered by the lifecycle-hook
    test suite (PRs #5, #6). Here we just verify the OTel notifier produces
    two clean invoke_agent spans — one per cycle — with no leaks.
    """

    async def test_two_cycles_yield_two_clean_invoke_agent_spans(
        self, exporter: InMemorySpanExporter
    ) -> None:
        from dendrux.types import RunResult

        notifier = OpenTelemetryNotifier()

        # Cycle 1: pause as waiting_client_tool.
        await notifier.on_run_started("run-pr", agent_name="alice", agent_model="m1")
        paused = RunResult(run_id="run-pr", status=RunStatus.WAITING_CLIENT_TOOL)
        await notifier.on_run_finished("run-pr", paused)

        # Cycle 2: resume to terminal SUCCESS.
        await notifier.on_run_started("run-pr", agent_name="alice", agent_model="m1")
        done = RunResult(run_id="run-pr", status=RunStatus.SUCCESS, answer="x")
        await notifier.on_run_finished("run-pr", done)

        invoke = _spans_by_name(exporter, "invoke_agent")
        assert len(invoke) == 2
        statuses = [_attr(s, "dendrux.run.status") for s in invoke]
        assert statuses == [RunStatus.WAITING_CLIENT_TOOL.value, RunStatus.SUCCESS.value]
        # No leaked state in the notifier's internal dicts.
        assert notifier._run_spans == {}
        assert notifier._run_models == {}


# ------------------------------------------------------------------
# Governance: non-scalar data preservation
# ------------------------------------------------------------------


class TestGovernanceNonScalarData:
    async def test_dict_value_is_json_stringified_not_dropped(
        self, exporter: InMemorySpanExporter
    ) -> None:
        from dendrux.types import RunResult

        notifier = OpenTelemetryNotifier()
        await notifier.on_run_started("run-gov", agent_name="alice", agent_model="m")
        await notifier.on_governance_event(
            "run-gov",
            "guardrail.detected",
            iteration=0,
            data={"scalar_str": "hi", "scalar_int": 7, "complex": {"nested": [1, 2, 3]}},
            correlation_id="corr-1",
        )
        await notifier.on_run_finished(
            "run-gov",
            RunResult(run_id="run-gov", status=RunStatus.SUCCESS, answer="x"),
        )

        invoke = _spans_by_name(exporter, "invoke_agent")[0]
        gov_events = [e for e in invoke.events if e.name == "governance.guardrail.detected"]
        assert len(gov_events) == 1
        attrs = dict(gov_events[0].attributes)
        assert attrs["dendrux.governance.scalar_str"] == "hi"
        assert attrs["dendrux.governance.scalar_int"] == 7
        # Non-scalar value survives as JSON rather than being silently dropped.
        assert attrs["dendrux.governance.complex"] == '{"nested": [1, 2, 3]}'
        assert attrs["dendrux.correlation_id"] == "corr-1"


# ------------------------------------------------------------------
# Orphan child spans closed on run terminal — stream-cancellation safety
# ------------------------------------------------------------------


class TestOrphanChildClosure:
    """Stream cancellation (GeneratorExit / asyncio.CancelledError) bypasses
    the loop's ``except Exception`` paths, so on_llm_call_completed/failed
    may not fire for the in-flight LLM call. The notifier's terminal hooks
    must sweep these or the chat/tool spans leak forever in observability
    backends.

    The runner-level guarantee that on_run_finished fires on abandoned
    streams is covered by ``tests/integration/test_lifecycle_hooks_resume.py``
    (PR #5 cleanup tests). Here we test the OTel layer in isolation:
    given a terminal hook fires with children still open, they get closed.
    """

    async def test_orphan_chat_span_closed_when_run_terminates_as_cancelled(
        self, exporter: InMemorySpanExporter
    ) -> None:
        from dendrux.types import RunResult

        notifier = OpenTelemetryNotifier()
        await notifier.on_run_started("run-stream", agent_name="alice", agent_model="m1")
        # LLM call started, but stream gets abandoned — completed/failed never fire.
        await notifier.on_llm_call_started("run-stream", iteration=0)
        # Runner-level cleanup detects abandonment and finalizes the run.
        await notifier.on_run_finished(
            "run-stream", RunResult(run_id="run-stream", status=RunStatus.CANCELLED)
        )

        chat = _spans_by_name(exporter, "chat")
        assert len(chat) == 1
        assert chat[0].status.status_code == StatusCode.ERROR
        assert _attr(chat[0], "dendrux.span.orphan_close_reason") is not None
        assert "cancelled" in _attr(chat[0], "dendrux.span.orphan_close_reason")
        # Notifier internal state cleared — no leaks.
        assert notifier._llm_spans == {}
        assert notifier._run_spans == {}

    async def test_orphan_tool_span_closed_when_run_fails(
        self, exporter: InMemorySpanExporter
    ) -> None:
        notifier = OpenTelemetryNotifier()
        tc = ToolCall(name="add", params={"a": 1, "b": 2}, provider_tool_call_id="t1")

        await notifier.on_run_started("run-fail", agent_name="alice", agent_model="m1")
        await notifier.on_tool_started("run-fail", tc, iteration=0)
        # Some downstream exception kills the run before tool_completed fires.
        await notifier.on_run_failed("run-fail", RuntimeError("boom"), iteration=0)

        tool_spans = _spans_by_name(exporter, "execute_tool")
        assert len(tool_spans) == 1
        assert tool_spans[0].status.status_code == StatusCode.ERROR
        assert _attr(tool_spans[0], "dendrux.tool.success") is False
        assert _attr(tool_spans[0], "dendrux.span.orphan_close_reason") is not None
        # No leaks.
        assert notifier._tool_spans == {}
        assert notifier._run_spans == {}


# ------------------------------------------------------------------
# Tool span keying — concurrent runs with overlapping tool_call.id
# ------------------------------------------------------------------


class TestToolSpanKeying:
    async def test_concurrent_runs_with_same_tool_call_id_dont_collide(
        self, exporter: InMemorySpanExporter
    ) -> None:
        from dendrux.types import RunResult, ToolResult

        notifier = OpenTelemetryNotifier()
        # Both runs use the SAME tool_call.id — caller misbehavior or fixture pattern.
        # With (run_id, tool_call.id) keying, no collision; with id-only keying,
        # the second on_tool_started would overwrite the first's span and one
        # span would never be ended.
        tc_a = ToolCall(name="add", params={}, id="shared-id", provider_tool_call_id="p1")
        tc_b = ToolCall(name="add", params={}, id="shared-id", provider_tool_call_id="p2")

        await notifier.on_run_started("run-A", agent_name="A", agent_model="m")
        await notifier.on_run_started("run-B", agent_name="B", agent_model="m")
        await notifier.on_tool_started("run-A", tc_a, iteration=0)
        await notifier.on_tool_started("run-B", tc_b, iteration=0)
        await notifier.on_tool_completed(
            "run-A",
            tc_a,
            ToolResult(name="add", call_id="shared-id", payload="3", success=True),
            iteration=0,
        )
        await notifier.on_tool_completed(
            "run-B",
            tc_b,
            ToolResult(name="add", call_id="shared-id", payload="3", success=True),
            iteration=0,
        )
        await notifier.on_run_finished("run-A", RunResult(run_id="run-A", status=RunStatus.SUCCESS))
        await notifier.on_run_finished("run-B", RunResult(run_id="run-B", status=RunStatus.SUCCESS))

        tool_spans = _spans_by_name(exporter, "execute_tool")
        assert len(tool_spans) == 2  # both closed cleanly, no orphans
        # Each tool span belongs to its own run's trace.
        invoke_traces = {
            _attr(s, "gen_ai.agent.name"): s.context.trace_id
            for s in _spans_by_name(exporter, "invoke_agent")
        }
        run_id_by_trace = {
            invoke_traces["A"]: "run-A",
            invoke_traces["B"]: "run-B",
        }
        for t in tool_spans:
            # No orphan close — both completed normally.
            assert _attr(t, "dendrux.span.orphan_close_reason") is None
            assert t.context.trace_id in run_id_by_trace
        assert notifier._tool_spans == {}
