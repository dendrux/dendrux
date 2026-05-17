"""Example 22: Observability complete stack — recorder + notifier + OTel + RunStore in one run.

Demonstrates how dendrux's four observability surfaces compose. One
agent run, all four layers active simultaneously, then a Rich report
showing exactly what each layer captured.

The stack:
    1. Recorder (PersistenceRecorder)  — durable audit truth in DB
    2. Notifier: ConsoleNotifier       — live terminal output
    3. Notifier: OpenTelemetryNotifier — host's tracing backend (here InMemorySpanExporter)
    4. Notifier: AlertNotifier         — custom subclass counting governance events
    5. RunStore (read facade)          — replay the recorder's output post-run

See docs/recipes/observability.mdx for the umbrella story this example
illustrates.

Run:
    ANTHROPIC_API_KEY=sk-... python examples/22_observability_complete_stack.py
"""

from __future__ import annotations

import asyncio
import tempfile
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from dendrux import Agent, Budget, tool
from dendrux.guardrails import PII
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.loops.base import BaseNotifier
from dendrux.notifiers import CompositeNotifier, ConsoleNotifier
from dendrux.notifiers.otel import OpenTelemetryNotifier
from dendrux.store import RunStore

if TYPE_CHECKING:
    from dendrux.types import LLMResponse, Message, RunResult, ToolCall, ToolResult

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

console = Console()


# ---------------------------------------------------------------------------
# Layer 2c — custom notifier counting governance events
# ---------------------------------------------------------------------------


class AlertNotifier(BaseNotifier):
    """Custom notifier — counts governance events by type.

    Real apps would do something with these (PagerDuty, Slack webhook,
    Prometheus counters, etc.). Here we just collect them so we can
    print a summary at the end and prove the live wire ran.
    """

    def __init__(self) -> None:
        self.governance_events: Counter[str] = Counter()
        self.llm_calls: int = 0
        self.tool_calls: int = 0

    async def on_governance_event(
        self,
        run_id: str,
        event_type: str,
        iteration: int,
        data: dict[str, Any],
        *,
        correlation_id: str | None = None,
    ) -> None:
        self.governance_events[event_type] += 1

    async def on_llm_call_completed(
        self,
        run_id: str,
        response: LLMResponse,
        iteration: int,
        *,
        semantic_messages: list[Message] | None = None,
        semantic_tools: list[Any] | None = None,
        duration_ms: int | None = None,
        guardrail_findings: dict[str, Any] | None = None,
    ) -> None:
        self.llm_calls += 1

    async def on_tool_completed(
        self,
        run_id: str,
        tool_call: ToolCall,
        tool_result: ToolResult,
        iteration: int,
    ) -> None:
        self.tool_calls += 1


# ---------------------------------------------------------------------------
# Tools — small but trigger every observability surface
# ---------------------------------------------------------------------------


@tool()
async def lookup_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"Sunny, 72F in {city}. Wind 8mph SW."


@tool()
async def lookup_customer(customer_id: str) -> dict[str, Any]:
    """Look up a customer's profile and contact information."""
    return {
        "id": customer_id,
        "name": "Jane Doe",
        "email": "jane@example.com",
        "phone": "+1-555-123-4567",
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _print_recorder_summary(detail: Any, events: list[Any], llm_calls: list[Any],
                            tools: list[Any], traces: list[Any]) -> None:
    table = Table(title="Layer 1: Recorder (durable audit in DB)")
    table.add_column("Table", style="cyan")
    table.add_column("Rows for this run", justify="right")
    table.add_column("Notes")

    table.add_row("agent_runs", "1", f"status={detail.status}, model={detail.model}")
    table.add_row("react_traces", str(len(traces)), "user / assistant / tool messages")
    table.add_row("tool_calls", str(len(tools)), "every tool invocation persisted")
    table.add_row("llm_interactions", str(len(llm_calls)),
                  "semantic + provider payloads per LLM call")
    table.add_row("run_events", str(len(events)),
                  "lifecycle + governance event log, monotonic sequence")

    console.print(table)
    console.print()


def _print_notifier_summary(alert: AlertNotifier) -> None:
    table = Table(title="Layer 2: Notifier (live event dispatch)")
    table.add_column("Notifier", style="cyan")
    table.add_column("What it did during the run")

    table.add_row(
        "ConsoleNotifier",
        "Printed live progress to your terminal above this report.",
    )
    table.add_row(
        "OpenTelemetryNotifier",
        "Emitted GenAI-semconv spans (see span tree below).",
    )
    table.add_row(
        "AlertNotifier (custom)",
        f"Saw {alert.llm_calls} LLM call(s), {alert.tool_calls} tool call(s), "
        f"{sum(alert.governance_events.values())} governance event(s).",
    )

    console.print(table)
    if alert.governance_events:
        gov = Table(title="Governance events the custom notifier counted")
        gov.add_column("Event type", style="yellow")
        gov.add_column("Count", justify="right")
        for event_type, count in sorted(alert.governance_events.items()):
            gov.add_row(event_type, str(count))
        console.print(gov)
    console.print()


def _print_otel_spans(exporter: InMemorySpanExporter) -> None:
    spans = sorted(exporter.get_finished_spans(), key=lambda s: s.start_time or 0)
    if not spans:
        console.print("[dim]No OTel spans captured.[/dim]\n")
        return

    span_by_id: dict[int, Any] = {s.context.span_id: s for s in spans}
    children: dict[int | None, list[Any]] = {}
    for s in spans:
        parent_id = s.parent.span_id if s.parent else None
        children.setdefault(parent_id, []).append(s)

    def _label(span: Any) -> str:
        dur_ms = ((span.end_time or 0) - (span.start_time or 0)) // 1_000_000
        status = span.status.status_code.name
        suffix = ""
        if span.name.startswith("chat") and "gen_ai.usage.input_tokens" in span.attributes:
            suffix = f" · tokens={span.attributes['gen_ai.usage.input_tokens']}" \
                     f"/{span.attributes.get('gen_ai.usage.output_tokens', 0)}"
        gov_events = [e.name for e in span.events
                      if e.name.startswith(("policy.", "approval.", "budget.",
                                            "guardrail.", "skill.", "mcp."))]
        gov_suffix = f" · gov: {', '.join(gov_events)}" if gov_events else ""
        return f"[{status}] {span.name}  {dur_ms}ms{suffix}{gov_suffix}"

    tree = Tree("[bold]Layer 3: OpenTelemetryNotifier (host's TracerProvider)[/bold]")
    roots = children.get(None, []) + [
        s for parent_id, sibs in children.items()
        if parent_id is not None and parent_id not in span_by_id
        for s in sibs
    ]
    for root in sorted(roots, key=lambda s: s.start_time or 0):
        _attach_children(tree.add(_label(root)), root, children)
    console.print(tree)
    console.print()


def _attach_children(node: Any, span: Any, children: dict[int | None, list[Any]]) -> None:
    for child in sorted(children.get(span.context.span_id, []), key=lambda s: s.start_time or 0):
        _attach_children(node.add(_label_short(child)), child, children)


def _label_short(span: Any) -> str:
    dur_ms = ((span.end_time or 0) - (span.start_time or 0)) // 1_000_000
    status = span.status.status_code.name
    return f"[{status}] {span.name}  {dur_ms}ms"


def _print_runstore_replay(detail: Any, events: list[Any], pauses: list[Any]) -> None:
    table = Table(title="Layer 4: RunStore (programmatic read facade)")
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    table.add_row("run_id", detail.run_id)
    table.add_row("status", detail.status)
    table.add_row("agent_name", detail.agent_name)
    table.add_row("model", detail.model or "")
    table.add_row("iteration_count", str(detail.iteration_count))
    table.add_row(
        "tokens (in / out / cache_read)",
        f"{detail.total_input_tokens} / "
        f"{detail.total_output_tokens} / "
        f"{detail.total_cache_read_tokens}",
    )
    table.add_row("event count", str(len(events)))
    table.add_row("pause cycles", str(len(pauses)))
    table.add_row("answer", (detail.answer or "")[:80] + ("..." if detail.answer and len(detail.answer) > 80 else ""))

    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    # OTel SDK setup: in production this is the host application's
    # TracerProvider with whatever exporter the host already uses
    # (OTLP -> Datadog / Honeycomb / Jaeger / Grafana / etc.). Here we
    # use InMemorySpanExporter so the demo is self-contained.
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # One temp DB so the recorder has somewhere to write and RunStore
    # can read back from the same place.
    db_dir = Path(tempfile.mkdtemp(prefix="dendrux_observability_"))
    db_url = f"sqlite+aiosqlite:///{db_dir}/observability.db"

    alert_notifier = AlertNotifier()

    # ---- One agent run with all four observability surfaces active ----
    console.print(
        Panel(
            "[bold]Running one agent with the complete observability stack[/bold]\n\n"
            "Layer 1: PersistenceRecorder (configured by database_url)\n"
            "Layer 2: CompositeNotifier composing 3 notifiers\n"
            "Layer 3: OpenTelemetryNotifier emits GenAI semconv spans\n"
            "Layer 4: RunStore reads back the recorder's output after the run",
            title="Observability Stack",
        )
    )

    async with Agent(
        provider=AnthropicProvider(model="claude-sonnet-4-6"),
        prompt=(
            "You are a helpful assistant. When asked for weather, use the lookup_weather tool. "
            "When asked about customers, use lookup_customer."
        ),
        tools=[lookup_weather, lookup_customer],
        database_url=db_url,
        guardrails=[PII()],
        budget=Budget(max_tokens=20_000, warn_at=(0.5, 0.75, 0.9)),
    ) as agent:
        notifier = CompositeNotifier([
            ConsoleNotifier(),
            OpenTelemetryNotifier(),
            alert_notifier,
        ])

        result = await agent.run(
            "Look up customer C-7782, then check the weather in their city — assume Paris.",
            notifier=notifier,
        )

    console.print()
    console.print(Panel(f"[bold]Run complete: {result.status.value}[/bold]\n\n{result.answer or ''}",
                        title="Result"))
    console.print()

    # ---- Read back via RunStore — same source as the dashboard would use ----
    async with RunStore.from_database_url(db_url) as store:
        detail = await store.get_run(result.run_id)
        assert detail is not None
        events = await store.get_events(result.run_id)
        llm_calls = await store.get_llm_calls(result.run_id)
        tool_invocations = await store.get_tool_invocations(result.run_id)
        traces = await store.get_traces(result.run_id)
        pauses = await store.get_pauses(result.run_id)

    # ---- Report ----
    console.print(Panel("[bold]What each observability layer captured[/bold]",
                        title="Stack report"))
    console.print()
    _print_recorder_summary(detail, events, llm_calls, tool_invocations, traces)
    _print_notifier_summary(alert_notifier)
    _print_otel_spans(exporter)
    _print_runstore_replay(detail, events, pauses)

    console.print(
        Panel(
            "All four surfaces saw the same agent run.\n\n"
            "Same hook events. Different consumers, different time horizons, different failure policies:\n"
            "  • Recorder: durable, fail-closed, queryable forever via RunStore\n"
            "  • Notifier (live): fail-open, dispatched as the run unfolds\n"
            "  • OTel: spans plug into the host's existing observability stack\n"
            "  • Custom: any BaseNotifier subclass — Slack / PagerDuty / Prometheus / etc.\n\n"
            "You configured all four with one Agent + one notifier= kwarg.",
            title="Why the stack composes",
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
