"""Real Agent boundaries preserve OTel contexts and safe provider projections."""

import asyncio
import json
import os
import sys

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from dendrux import Agent, tool
from dendrux.errors import ProviderError
from dendrux.guardrails import SecretDetection
from dendrux.llm._resilience import logs_provider_errors, logs_provider_stream_errors
from dendrux.llm.mock import MockLLM
from dendrux.loops.base import BaseNotifier
from dendrux.notifiers.composite import CompositeNotifier
from dendrux.notifiers.otel import OpenTelemetryNotifier
from dendrux.types import LLMResponse, RunStatus, ToolCall

SENTINEL = "PRIVATE_VENDOR_SENTINEL_7921"
LINK = "00-" + "1" * 32 + "-" + "2" * 16 + "-01"


@pytest.fixture
def tracing():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider, exporter
    provider.shutdown()


def spans(exporter, operation):
    return [
        s
        for s in exporter.get_finished_spans()
        if s.attributes.get("gen_ai.operation.name") == operation
    ]


@tool()
async def work() -> str:
    """Complete an approval-gated action."""
    return "ok"


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
@pytest.mark.parametrize("streaming", [False, True])
async def test_d2_subprocess_resume_exports_same_trace_and_saved_parent(
    store, engine, tracing, streaming
):
    provider, exporter = tracing
    agent = Agent(
        prompt="p",
        provider=MockLLM([LLMResponse(tool_calls=[ToolCall("work")])]),
        tools=[work],
        require_approval=["work"],
        state_store=store,
        strict_tenancy=False,
    )
    paused = await agent.run("go", notifier=OpenTelemetryNotifier(provider.get_tracer("A")))
    assert paused.status == RunStatus.WAITING_APPROVAL
    (original,) = spans(exporter, "invoke_agent")
    saved = (await store.get_run(paused.run_id)).trace_link
    assert saved is not None, "process A must persist the OTel run context"
    assert saved.split("-")[2] == f"{original.context.span_id:016x}"
    code = """
import asyncio, json, sys
from sqlalchemy.ext.asyncio import create_async_engine
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from dendrux import Agent, tool
from dendrux.llm.mock import MockLLM
from dendrux.notifiers.otel import OpenTelemetryNotifier
from dendrux.runtime.stores._factory import store_for_engine
from dendrux.types import LLMResponse
@tool()
async def work() -> str:
    return "ok"
async def main():
    engine = create_async_engine(sys.argv[1])
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        store = store_for_engine(engine)
        agent = Agent(prompt="p", provider=MockLLM([LLMResponse(text="done")]),
                      tools=[work], require_approval=["work"], state_store=store,
                      strict_tenancy=False)
        notifier = OpenTelemetryNotifier(provider.get_tracer("B"))
        if sys.argv[4] == "True":
            async with agent.resume_stream(sys.argv[2], expected_pause_id=sys.argv[3],
                                           notifier=notifier) as stream:
                events = [e async for e in stream]
            result = events[-1].run_result
        else:
            result = await agent.submit_approval(sys.argv[2], approved=True,
                expected_pause_id=sys.argv[3], notifier=notifier)
        spans = exporter.get_finished_spans()
        run, = [s for s in spans if s.attributes.get("gen_ai.operation.name") == "invoke_agent"]
        print(json.dumps({"status": result.status.value, "trace": run.context.trace_id,
                          "parent": run.parent.span_id if run.parent else None,
                          "children": [s.context.trace_id for s in spans],
                          "stored": (await store.get_run(sys.argv[2])).trace_link}))
    finally:
        provider.shutdown()
        await engine.dispose()
asyncio.run(main())
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        code,
        engine.url.render_as_string(hide_password=False),
        paused.run_id,
        paused.pause_id,
        str(streaming),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": os.path.abspath("src")},
    )
    try:
        output, error = await asyncio.wait_for(process.communicate(), 30)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == 0, error.decode()
    resumed = json.loads(output)
    assert resumed["status"] == "success"
    assert resumed["trace"] == original.context.trace_id
    assert resumed["parent"] == original.context.span_id
    assert set(resumed["children"]) == {original.context.trace_id}
    assert resumed["stored"] == saved


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
async def test_d3_delegated_child_parents_on_active_tool_span(store, tracing, shared, streaming):
    provider, exporter = tracing
    parent_notifier = OpenTelemetryNotifier(provider.get_tracer("parent"))
    child_notifier = (
        parent_notifier if shared else OpenTelemetryNotifier(provider.get_tracer("child"))
    )

    @tool(parallel=True)
    async def child() -> str:
        """Run a delegated child under this tool span."""
        result = await Agent(
            name="child",
            prompt="p",
            provider=MockLLM([LLMResponse(text="ok")]),
            state_store=store,
            strict_tenancy=False,
        ).run("go", notifier=child_notifier)
        assert result.status == RunStatus.SUCCESS
        return "ok"

    agent = Agent(
        name="parent",
        prompt="p",
        state_store=store,
        strict_tenancy=False,
        tools=[child],
        provider=MockLLM(
            [
                LLMResponse(tool_calls=[ToolCall("child"), ToolCall("child")]),
                LLMResponse(text="done"),
            ]
        ),
    )
    if streaming:
        async with agent.stream("go", notifier=parent_notifier) as stream:
            events = [event async for event in stream]
        result = events[-1].run_result
    else:
        result = await agent.run("go", notifier=parent_notifier)
    assert result.status == RunStatus.SUCCESS
    runs = spans(exporter, "invoke_agent")
    (parent,) = [s for s in runs if s.attributes["gen_ai.agent.name"] == "parent"]
    children = [s for s in runs if s.attributes["gen_ai.agent.name"] == "child"]
    tools = spans(exporter, "execute_tool")
    assert len(children) == len(tools) == 2
    assert {s.parent.span_id for s in children} == {s.context.span_id for s in tools}
    assert {s.context.trace_id for s in runs + tools} == {parent.context.trace_id}
    assert not parent_notifier._run_spans and not child_notifier._tool_spans


class CustomNotifier(BaseNotifier):
    def __init__(self):
        self.starts = []

    def trace_link(self, run_id):
        return LINK

    async def on_run_started(self, run_id, *, agent_name=None, agent_model=None, trace_link=None):
        self.starts.append(trace_link)


async def test_d3_custom_parent_without_tool_link_gives_otel_child_run_link(tracing):
    provider, exporter = tracing

    @tool()
    async def child() -> str:
        """Delegate with a custom parent notifier."""
        await Agent(
            prompt="p", provider=MockLLM([LLMResponse(text="ok")]), strict_tenancy=False
        ).run("go", notifier=OpenTelemetryNotifier(provider.get_tracer("child")))
        return "ok"

    await Agent(
        prompt="p",
        provider=MockLLM([LLMResponse(tool_calls=[ToolCall("child")]), LLMResponse(text="done")]),
        tools=[child],
        strict_tenancy=False,
    ).run("go", notifier=CustomNotifier())
    (child_span,) = spans(exporter, "invoke_agent")
    assert child_span.context.trace_id == int("1" * 32, 16)
    assert child_span.parent.span_id == int("2" * 16, 16)


async def test_d3_custom_child_receives_only_parent_run_link(tracing):
    provider, _ = tracing
    notifier = OpenTelemetryNotifier(provider.get_tracer("parent"))
    custom = CustomNotifier()
    expected = []

    @tool()
    async def child() -> str:
        """Delegate to a custom child notifier."""
        from dendrux.runtime.context import get_delegation_context

        expected.append(get_delegation_context().trace_link)
        await Agent(
            prompt="p", provider=MockLLM([LLMResponse(text="ok")]), strict_tenancy=False
        ).run("go", notifier=custom)
        return "ok"

    await Agent(
        prompt="p",
        provider=MockLLM([LLMResponse(tool_calls=[ToolCall("child")]), LLMResponse(text="done")]),
        tools=[child],
        strict_tenancy=False,
    ).run("go", notifier=notifier)
    assert custom.starts == expected


async def test_d3_mixed_notifier_exposes_run_link_only(tracing):
    provider, _ = tracing
    otel = OpenTelemetryNotifier(provider.get_tracer("parent"))
    mixed = CompositeNotifier([otel, CustomNotifier()])
    await mixed.on_run_started("run")
    try:
        assert mixed.trace_link("run") == otel.trace_link("run")
        call = ToolCall("work")
        await mixed.on_tool_started("run", call, 1)
        assert mixed.current_span_link("run", call.id) is None
    finally:
        from dendrux.types import RunResult

        await mixed.on_run_finished("run", RunResult("run", RunStatus.CANCELLED))


@pytest.mark.parametrize("direction", ["incoming", "outgoing"])
async def test_d4_guardrail_block_exports_error_without_provider_info(store, tracing, direction):
    provider, exporter = tracing
    secret = "api_key=sk_test_abcdefghijklmnopqrstuvwxyz123456"
    agent = Agent(
        prompt="p",
        state_store=store,
        strict_tenancy=False,
        provider=MockLLM([LLMResponse(text=secret)]),
        guardrails=[SecretDetection(action="block")],
    )
    result = await agent.run(
        secret if direction == "incoming" else "go",
        notifier=OpenTelemetryNotifier(provider.get_tracer("guardrail")),
    )
    assert result.status == RunStatus.ERROR and result.error_info is None
    (span,) = spans(exporter, "invoke_agent")
    assert span.status.status_code == StatusCode.ERROR
    assert secret not in repr(span.attributes) + repr(span.events) + repr(span.status.description)


@pytest.mark.parametrize("entry", ["run", "stream", "submit_approval", "resume_stream"])
async def test_d6_unknown_provider_code_is_safe_in_logs_result_event_span_and_wire(
    store, tracing, caplog, entry
):
    provider, exporter = tracing
    error = ProviderError(
        provider="openai", provider_code=SENTINEL, message="The provider reported an error."
    )

    class BrokenProvider(MockLLM):
        @logs_provider_errors
        async def complete(self, *args, **kwargs):
            raise error

        @logs_provider_stream_errors
        async def complete_stream(self, *args, **kwargs):
            raise error
            yield  # Make this an async generator, matching the provider protocol.

    agent = Agent(
        prompt="p",
        provider=BrokenProvider([]),
        tools=[work],
        require_approval=["work"],
        state_store=store,
        strict_tenancy=False,
    )
    notifier = OpenTelemetryNotifier(provider.get_tracer("safe"))
    if entry in {"submit_approval", "resume_stream"}:
        pausing = Agent(
            prompt="p",
            provider=MockLLM([LLMResponse(tool_calls=[ToolCall("work")])]),
            tools=[work],
            require_approval=["work"],
            state_store=store,
            strict_tenancy=False,
        )
        paused = await pausing.run("go", notifier=notifier)
        exporter.clear()
        caplog.clear()
    if entry in {"stream", "resume_stream"}:
        stream = (
            agent.stream("go", notifier=notifier)
            if entry == "stream"
            else agent.resume_stream(
                paused.run_id, expected_pause_id=paused.pause_id, notifier=notifier
            )
        )
        async with stream:
            events = [event async for event in stream]
        result = events[-1].run_result
        event_projection = repr(events)
    else:
        with pytest.raises(ProviderError):
            if entry == "run":
                await agent.run("go", notifier=notifier)
            else:
                await agent.submit_approval(
                    paused.run_id,
                    approved=True,
                    expected_pause_id=paused.pause_id,
                    notifier=notifier,
                )
        run_id = paused.run_id if entry == "submit_approval" else (await store.list_runs())[0].id
        result = await agent.get_result(run_id, tenant_id=None)
        event_projection = ""
    assert result.error_info.provider_code == "provider_unknown"
    log = [r for r in caplog.records if getattr(r, "kind", None) == "provider"]
    assert log and all(r.code == "provider_unknown" for r in log)
    span_events = [
        event
        for span in exporter.get_finished_spans()
        for event in span.events
        if event.name == "provider.error"
    ]
    assert span_events and all(
        event.attributes["dendrux.error.provider_code"] == "provider_unknown"
        for event in span_events
    )
    wire = result.to_dict()
    assert wire["error_info"]["provider_code"] == "provider_unknown"
    row = await store.get_run(result.run_id)
    durable_events = await store.get_run_events(result.run_id)
    interactions = await store.get_llm_interactions(result.run_id)
    projections = [
        str(error),
        repr(error),
        error.safe_message,
        result.error_info.to_dict(),
        wire,
        row.error,
        row.meta,
        [e.data for e in durable_events],
        event_projection,
        repr([s.attributes for s in exporter.get_finished_spans()]),
        repr([s.events for s in exporter.get_finished_spans()]),
        caplog.text,
        repr([r.__dict__ for r in caplog.records]),
        repr(interactions),
    ]
    assert SENTINEL not in repr(projections)
    assert error.provider_code_raw == SENTINEL


async def test_reclaim_reuses_durable_trace_through_phase1_start_seam(store, engine, tracing):
    from datetime import UTC, datetime, timedelta

    from sqlalchemy import update

    from dendrux.db.models import AgentRun
    from dendrux.runtime.runner import _start_trace
    from dendrux.types import RunResult, generate_ulid

    provider, exporter = tracing
    run_id = generate_ulid()
    await store.create_run(run_id, "reclaim-trace", strategy="NativeToolCalling")
    first = OpenTelemetryNotifier(provider.get_tracer("before-crash"))
    saved, _ = await _start_trace(
        first, store, run_id, agent_name="reclaim-trace", agent_model=None
    )
    assert saved is not None, "the original worker must persist its run span"
    await first.on_run_finished(run_id, RunResult(run_id, RunStatus.PAUSED))
    (original,) = spans(exporter, "invoke_agent")
    async with engine.begin() as connection:
        await connection.execute(
            update(AgentRun)
            .where(AgentRun.id == run_id)
            .values(
                last_progress_at=datetime.now(UTC) - timedelta(hours=1),
                loop_state={"loop": "react"},
            )
        )
    marked = await store.mark_lease_expired(
        cutoff=datetime.now(UTC) - timedelta(minutes=1),
        batch_size=10,
        max_runs=10,
        max_returned=10,
        reclaim_limit=3,
        cursor=None,
    )
    claim = await store.reclaim_run(
        run_id,
        tenant_id=None,
        agent_name="reclaim-trace",
        expected_claim_epoch=marked.reclaimable[0].claim_epoch,
        governance={},
        governance_policy="raise",
        skill_pins={},
        loop_type="react",
    )
    assert claim
    row = await store.get_run(run_id)
    assert row.trace_link == saved and row.reclaim_attempts == 1
    resumed = OpenTelemetryNotifier(provider.get_tracer("reclaimed"))
    returned, _ = await _start_trace(
        resumed, store, run_id, agent_name="reclaim-trace", agent_model=None, saved=row.trace_link
    )
    await resumed.on_run_finished(run_id, RunResult(run_id, RunStatus.SUCCESS))
    continued = spans(exporter, "invoke_agent")[-1]
    assert returned == saved
    assert continued.context.trace_id == original.context.trace_id
    assert continued.parent.span_id == original.context.span_id


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
async def test_public_reclaim_after_process_crash_keeps_durable_parent(store, engine, tracing):
    import signal
    from datetime import UTC, datetime, timedelta

    from sqlalchemy import update

    from dendrux import sweep
    from dendrux.db.models import AgentRun

    code = """
import asyncio, json, os, signal, sys
from sqlalchemy.ext.asyncio import create_async_engine
from opentelemetry.sdk.trace import TracerProvider
from dendrux import Agent
from dendrux.llm.mock import MockLLM
from dendrux.notifiers.otel import OpenTelemetryNotifier
from dendrux.runtime.stores._factory import store_for_engine
async def main():
    engine = create_async_engine(sys.argv[1])
    store = store_for_engine(engine)
    notifier = OpenTelemetryNotifier(TracerProvider().get_tracer("crashed-worker"))
    class CrashBeforeCompletion(MockLLM):
        async def complete(self, *args, **kwargs):
            run_id, = notifier._run_spans
            row = await store.get_run(run_id)
            assert row.trace_link
            print(json.dumps({"run_id": run_id, "link": row.trace_link}), flush=True)
            os.kill(os.getpid(), signal.SIGKILL)
    await Agent(name="crash-trace", prompt="p", state_store=store,
                provider=CrashBeforeCompletion([]), strict_tenancy=False).run(
                    "go", notifier=notifier)
asyncio.run(main())
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        code,
        engine.url.render_as_string(hide_password=False),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": os.path.abspath("src")},
    )
    try:
        output, error = await asyncio.wait_for(process.communicate(), 30)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode == -signal.SIGKILL, error.decode()
    crashed = json.loads(output)
    run_id, saved = crashed["run_id"], crashed["link"]
    row = await store.get_run(run_id)
    assert row.status == "running" and row.trace_link == saved
    async with engine.begin() as connection:
        await connection.execute(
            update(AgentRun)
            .where(AgentRun.id == run_id)
            .values(last_progress_at=datetime.now(UTC) - timedelta(minutes=10))
        )
    page = await sweep(state_store=store, stale_running=timedelta(minutes=1), on_stale="reclaim")
    (item,) = page.reclaimable
    assert item.run_id == run_id
    provider, exporter = tracing
    seen = []

    class RecordingOTel(OpenTelemetryNotifier):
        async def on_run_started(
            self, run_id, *, agent_name=None, agent_model=None, trace_link=None
        ):
            seen.append(trace_link)
            await super().on_run_started(
                run_id, agent_name=agent_name, agent_model=agent_model, trace_link=trace_link
            )

    agent = Agent(
        name="crash-trace",
        prompt="p",
        state_store=store,
        provider=MockLLM([LLMResponse(text="done")]),
        strict_tenancy=False,
    )
    result = await agent.reclaim(
        run_id,
        tenant_id=None,
        item=item,
        notifier=RecordingOTel(provider.get_tracer("reclaimed-worker")),
    )
    assert result.status == RunStatus.SUCCESS
    assert seen == [saved]
    (continued,) = spans(exporter, "invoke_agent")
    assert continued.context.trace_id == int(saved.split("-")[1], 16)
    assert continued.parent.span_id == int(saved.split("-")[2], 16)
    assert all(
        s.context.trace_id == continued.context.trace_id for s in exporter.get_finished_spans()
    )
    row = await store.get_run(run_id)
    assert row.trace_link == saved and row.reclaim_attempts == 1


@pytest.mark.parametrize(
    "container", ["custom", "custom_composite", "otel_composite", "nested_otel"]
)
@pytest.mark.parametrize("streaming", [False, True])
async def test_child_link_selection_matches_notifier_kind(tracing, container, streaming):
    provider, exporter = tracing
    custom = CustomNotifier()
    child_otel = OpenTelemetryNotifier(provider.get_tracer("child"))
    child_notifier = {
        "custom": custom,
        "custom_composite": CompositeNotifier([custom]),
        "otel_composite": CompositeNotifier([child_otel, custom]),
        "nested_otel": CompositeNotifier([CompositeNotifier([child_otel]), custom]),
    }[container]
    expected = []
    otel_child = container in {"otel_composite", "nested_otel"}

    @tool()
    async def child() -> str:
        """Delegate under an active parent tool span."""
        from dendrux.runtime.context import get_delegation_context

        context = get_delegation_context()
        assert context.tool_span_link is not None
        expected.append(context.tool_span_link if otel_child else context.trace_link)
        agent = Agent(
            name="child",
            prompt="p",
            provider=MockLLM([LLMResponse(text="done")]),
            strict_tenancy=False,
        )
        if streaming:
            async with agent.stream("go", notifier=child_notifier) as stream:
                events = [event async for event in stream]
            assert events[-1].run_result.status == RunStatus.SUCCESS
        else:
            assert (await agent.run("go", notifier=child_notifier)).status == RunStatus.SUCCESS
        return "ok"

    await Agent(
        name="parent",
        prompt="p",
        tools=[child],
        strict_tenancy=False,
        provider=MockLLM([LLMResponse(tool_calls=[ToolCall("child")]), LLMResponse(text="done")]),
    ).run("go", notifier=OpenTelemetryNotifier(provider.get_tracer("parent")))
    assert custom.starts == expected
    children = [
        s for s in spans(exporter, "invoke_agent") if s.attributes["gen_ai.agent.name"] == "child"
    ]
    assert len(children) == int(otel_child)
    if otel_child:
        (tool_span,) = spans(exporter, "execute_tool")
        assert children[0].parent.span_id == tool_span.context.span_id
