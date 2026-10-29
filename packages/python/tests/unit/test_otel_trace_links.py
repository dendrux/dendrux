"""OTel links are validated W3C contexts and run status is authoritative."""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from dendrux.notifiers.otel import OpenTelemetryNotifier
from dendrux.types import RunResult, RunStatus, ToolCall, ToolResult


@pytest.fixture
def tracing():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield OpenTelemetryNotifier(provider.get_tracer("p2b")), exporter
    provider.shutdown()


async def test_run_and_current_tool_span_links_have_exact_context(tracing):
    notifier, exporter = tracing
    await notifier.on_run_started("run")
    link = notifier.trace_link("run")
    assert link is not None, "run span must expose its durable traceparent"
    tool = ToolCall("work")
    await notifier.on_tool_started("run", tool, 1)
    tool_link = notifier.current_span_link("run", tool.id)
    assert tool_link is not None
    assert link.split("-")[1] == tool_link.split("-")[1]
    assert link.split("-")[2] != tool_link.split("-")[2]
    assert notifier.current_span_link("other", tool.id) is None
    assert notifier.current_span_link("run", "missing") is None
    await notifier.on_tool_completed("run", tool, ToolResult("work", tool.id, "ok"), 1)
    assert notifier.current_span_link("run", tool.id) is None
    await notifier.on_run_finished("run", RunResult("run", RunStatus.SUCCESS))
    assert notifier.trace_link("run") is None
    assert len(exporter.get_finished_spans()) == 2


@pytest.mark.parametrize("status", list(RunStatus))
async def test_status_without_provider_error_info_is_authoritative(tracing, status):
    notifier, exporter = tracing
    await notifier.on_run_started("run")
    await notifier.on_run_finished("run", RunResult("run", status))
    (span,) = exporter.get_finished_spans()
    expected = (
        StatusCode.ERROR
        if status.value in {"error", "max_iterations", "cancelled"}
        else StatusCode.OK
    )
    assert span.status.status_code == expected


@pytest.mark.parametrize(
    "link", ["PRIVATE_LINK_SENTINEL", "00-" + "0" * 32 + "-" + "2" * 16 + "-01", None]
)
async def test_invalid_parent_is_discarded_without_echo(tracing, link, caplog):
    notifier, exporter = tracing
    await notifier.on_run_started("run", trace_link=link)
    await notifier.on_run_finished("run", RunResult("run", RunStatus.SUCCESS))
    (span,) = exporter.get_finished_spans()
    assert span.parent is None
    assert "PRIVATE_LINK_SENTINEL" not in repr(span.attributes) + caplog.text


async def test_explicit_link_parents_the_resumed_span(tracing):
    notifier, exporter = tracing
    link = "00-" + "1" * 32 + "-" + "2" * 16 + "-00"
    await notifier.on_run_started("run", trace_link=link)
    await notifier.on_run_finished("run", RunResult("run", RunStatus.SUCCESS))
    # The default sampler preserves the unsampled parent, so inspect a sampled link too.
    assert not exporter.get_finished_spans()
    await notifier.on_run_started("run", trace_link=link[:-2] + "01")
    await notifier.on_run_finished("run", RunResult("run", RunStatus.SUCCESS))
    (span,) = exporter.get_finished_spans()
    assert span.context.trace_id == int("1" * 32, 16)
    assert span.parent.span_id == int("2" * 16, 16)


async def test_link_hooks_fail_open_for_broken_span(tracing):
    notifier, _ = tracing

    class Broken:
        def get_span_context(self):
            raise RuntimeError("PRIVATE_SPAN_SENTINEL")

    notifier._run_spans["run"] = Broken()
    notifier._tool_spans[("run", "tool")] = Broken()
    assert notifier.trace_link("run") is None
    assert notifier.current_span_link("run", "tool") is None


@pytest.mark.parametrize("nested", [False, True])
async def test_span_failures_never_copy_exception_messages(tracing, nested):
    from dendrux.errors import ProviderError

    notifier, exporter = tracing
    error = RuntimeError("PRIVATE_MESSAGE_SENTINEL")
    if nested:
        error.__cause__ = ProviderError(provider="openai", provider_code="PRIVATE_CODE_SENTINEL")
    await notifier.on_run_started("run")
    await notifier.on_run_failed("run", error)
    (span,) = exporter.get_finished_spans()
    projected = repr(span.attributes) + repr(span.events) + repr(span.status.description)
    assert "PRIVATE_MESSAGE_SENTINEL" not in projected
    assert "PRIVATE_CODE_SENTINEL" not in projected


async def test_tool_failure_span_never_copies_tool_error_payload(tracing):
    notifier, exporter = tracing
    await notifier.on_run_started("run")
    tool = ToolCall("work")
    await notifier.on_tool_started("run", tool, 1)
    await notifier.on_tool_completed(
        "run",
        tool,
        ToolResult("work", tool.id, "", success=False, error="PRIVATE_TOOL_SENTINEL"),
        1,
    )
    (span,) = exporter.get_finished_spans()
    assert "PRIVATE_TOOL_SENTINEL" not in repr(span.status.description)


async def test_orphan_children_never_copy_failure_messages(tracing):
    notifier, exporter = tracing
    await notifier.on_run_started("run")
    await notifier.on_llm_call_started("run", 1)
    await notifier.on_tool_started("run", ToolCall("work"), 1)
    await notifier.on_run_failed("run", RuntimeError("PRIVATE_ORPHAN_SENTINEL"))
    assert len(exporter.get_finished_spans()) == 3
    for span in exporter.get_finished_spans():
        assert span.status.status_code == StatusCode.ERROR
        assert "PRIVATE_ORPHAN_SENTINEL" not in repr(span.events) + repr(span.status.description)


async def test_error_result_keeps_mapped_provider_description(tracing):
    from dendrux.errors import ProviderError
    from dendrux.types import ErrorInfo

    notifier, exporter = tracing
    error = ProviderError(provider="openai", provider_code="PRIVATE_CODE_SENTINEL")
    await notifier.on_run_started("run")
    await notifier.on_run_finished(
        "run", RunResult("run", RunStatus.ERROR, error_info=ErrorInfo.from_error(error))
    )
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == error.safe_message
    assert "PRIVATE_CODE_SENTINEL" not in span.status.description


async def test_single_composite_forwards_active_run_and_tool_links(tracing):
    from dendrux.notifiers.composite import CompositeNotifier

    notifier, _ = tracing
    composite = CompositeNotifier([notifier])
    await composite.on_run_started("run")
    assert composite.trace_link("run") == notifier.trace_link("run")
    call = ToolCall("work")
    await composite.on_tool_started("run", call, 1)
    assert composite.current_span_link("run", call.id) == notifier.current_span_link("run", call.id)
    assert composite.current_span_link("other", call.id) is None
    await composite.on_run_finished("run", RunResult("run", RunStatus.SUCCESS))
    assert composite.trace_link("run") is None


async def test_composite_parents_otel_and_signature_gates_legacy_and_kwargs(tracing, caplog):
    from dendrux.loops.base import BaseNotifier
    from dendrux.notifiers.composite import CompositeNotifier

    notifier, exporter = tracing
    seen = []

    class Legacy(BaseNotifier):
        async def on_run_started(self, run_id, *, agent_name=None, agent_model=None):
            seen.append((run_id, agent_name))

    class Flexible(BaseNotifier):
        async def on_run_started(self, run_id, **kwargs):
            seen.append(kwargs["trace_link"])

    class Broken(BaseNotifier):
        async def on_run_started(self, *args, **kwargs):
            raise RuntimeError("PRIVATE_COMPOSITE_SENTINEL")

    link = "00-" + "1" * 32 + "-" + "2" * 16 + "-01"
    composite = CompositeNotifier([Broken(), notifier, Legacy(), Flexible()])
    await composite.on_run_started("run", agent_name="agent", trace_link=link)
    assert seen == [("run", "agent"), link]
    await composite.on_run_finished("run", RunResult("run", RunStatus.SUCCESS))
    (span,) = exporter.get_finished_spans()
    assert span.context.trace_id == int("1" * 32, 16)
    assert span.parent.span_id == int("2" * 16, 16)
    assert "PRIVATE_COMPOSITE_SENTINEL" not in caplog.text


async def test_composite_link_hooks_skip_invalid_and_failed_children(tracing, caplog):
    from dendrux.loops.base import BaseNotifier
    from dendrux.notifiers.composite import CompositeNotifier

    notifier, _ = tracing

    class Broken(BaseNotifier):
        def trace_link(self, run_id):
            raise RuntimeError("PRIVATE_COMPOSITE_SENTINEL")

        def current_span_link(self, run_id, call_id):
            raise RuntimeError("PRIVATE_COMPOSITE_SENTINEL")

    class Invalid(BaseNotifier):
        def trace_link(self, run_id):
            return "PRIVATE_INVALID_LINK_SENTINEL"

    await notifier.on_run_started("run")
    mixed = CompositeNotifier([Broken(), Invalid(), notifier])
    assert mixed.trace_link("run") == notifier.trace_link("run")
    assert mixed.current_span_link("run", "call") is None
    assert CompositeNotifier([Broken()]).current_span_link("run", "call") is None
    assert CompositeNotifier([]).trace_link("run") is None
    assert "PRIVATE_COMPOSITE_SENTINEL" not in caplog.text
    assert "PRIVATE_INVALID_LINK_SENTINEL" not in caplog.text
    await notifier.on_run_finished("run", RunResult("run", RunStatus.SUCCESS))


async def test_non_provider_failure_keeps_only_own_frame_diagnostics(tracing):
    notifier, exporter = tracing
    try:
        raise RuntimeError("PRIVATE_FRAME_SENTINEL")
    except RuntimeError as error:
        await notifier.on_run_started("run")
        await notifier.on_run_failed("run", error)
    (span,) = exporter.get_finished_spans()
    (event,) = span.events
    assert event.attributes["exception.type"] == "RuntimeError"
    stack = event.attributes["exception.stacktrace"]
    assert "test_non_provider_failure_keeps_only_own_frame_diagnostics" in stack
    assert __file__ in stack and "line " in stack
    assert "raise RuntimeError" not in stack
    assert "PRIVATE_FRAME_SENTINEL" not in repr(span.events)
    assert span.status.description is None


async def test_failed_tool_keeps_safe_completion_frames(tracing):
    notifier, exporter = tracing
    await notifier.on_run_started("run")
    tool = ToolCall("work", params={"private": "PRIVATE_TOOL_FRAME_SENTINEL"})
    await notifier.on_tool_started("run", tool, 1)
    await notifier.on_tool_completed(
        "run",
        tool,
        ToolResult(
            "work",
            tool.id,
            "PRIVATE_TOOL_FRAME_SENTINEL",
            success=False,
            error="PRIVATE_TOOL_FRAME_SENTINEL",
        ),
        1,
    )
    (span,) = exporter.get_finished_spans()
    (event,) = span.events
    # ToolResult carries no original exception/traceback. Diagnose the completion site.
    assert event.attributes["exception.type"] == "ToolResult"
    assert "on_tool_completed" in event.attributes["exception.stacktrace"]
    assert "PRIVATE_TOOL_FRAME_SENTINEL" not in repr(span.events) + repr(span.attributes)


@pytest.mark.parametrize("stored", [False, True])
async def test_stored_raw_provider_code_is_mapped_at_span_boundary(tracing, stored):
    from dendrux.types import ErrorInfo

    notifier, exporter = tracing
    raw = "PRIVATE_STORED_CODE_SENTINEL"
    data = dict(
        category="timeout",
        retryable=True,
        provider="anthropic",
        provider_code=raw,
        safe_message=f"anthropic timeout (HTTP 504, code {raw})",
    )
    info = ErrorInfo.from_dict(data) if stored else ErrorInfo(**data)
    await notifier.on_run_started("run")
    await notifier.on_run_finished("run", RunResult("run", RunStatus.ERROR, error_info=info))
    (span,) = exporter.get_finished_spans()
    assert span.attributes["dendrux.error.provider_code"] == "provider_unknown"
    assert span.events[0].attributes["dendrux.error.provider_code"] == "provider_unknown"
    assert raw not in repr(span.attributes) + repr(span.events) + repr(span.status.description)
    assert info.provider_code == raw  # Export projection must not rewrite stored input.


async def test_composite_start_failure_retains_safe_exception_diagnostics(caplog):
    from dendrux.loops.base import BaseNotifier
    from dendrux.notifiers.composite import CompositeNotifier

    class Broken(BaseNotifier):
        async def on_run_started(self, run_id, **kwargs):
            raise RuntimeError("PRIVATE_NOTIFIER_SENTINEL")

    await CompositeNotifier([Broken()]).on_run_started("run")
    (record,) = caplog.records
    assert record.exc_info is not None
    assert "RuntimeError" in caplog.text and "on_run_started" in caplog.text
    assert "line " in caplog.text  # frame diagnostics, without source or locals
    assert "PRIVATE_NOTIFIER_SENTINEL" not in caplog.text
