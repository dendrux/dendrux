"""Example 21: Notifier complete cycle — with OpenTelemetry.

Mirrors example 20 stage-for-stage but adds ``OpenTelemetryNotifier``
to the composite. Spans are captured in-process via
``InMemorySpanExporter`` and printed as a tidy per-stage tree plus
a final invariants check.

This is the canonical visual smoke test for the OTel notifier:
re-run after any change to the notifier or runtime lifecycle and
eyeball the span tree shape against expectations.

Stages mirror example 20:
    1. Streaming + server tools
    2. Streaming + deny policy
    3. Streaming approve via resume_stream
    4. Sync reject via submit_approval
    5. Streaming client-tool resume
    6. Streaming + budget cap
    7. Sync + PII guardrails

Run:
    ANTHROPIC_API_KEY=sk-... python examples/21_otel_complete_cycle.py
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from dendrux import Agent, tool
from dendrux.guardrails import PII
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.loops.base import BaseNotifier
from dendrux.notifiers.composite import CompositeNotifier
from dendrux.notifiers.console import ConsoleNotifier
from dendrux.notifiers.otel import OpenTelemetryNotifier
from dendrux.types import Budget, GovernanceEventType, RunEventType, RunStatus, ToolResult

if TYPE_CHECKING:
    from dendrux.types import LLMResponse, Message, ToolCall

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

_console = Console()
MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# HookTracker — same as example 20.
# ---------------------------------------------------------------------------


class HookTracker(BaseNotifier):
    def __init__(self) -> None:
        self.hook_counts: Counter[str] = Counter()
        self.governance_events: Counter[str] = Counter()

    async def on_run_started(self, run_id, *, agent_name=None, agent_model=None) -> None:
        self.hook_counts["on_run_started"] += 1

    async def on_run_finished(self, run_id, result) -> None:
        self.hook_counts["on_run_finished"] += 1

    async def on_run_failed(self, run_id, error, *, iteration=None) -> None:
        self.hook_counts["on_run_failed"] += 1

    async def on_message_appended(self, run_id, message: Message, iteration: int) -> None:
        self.hook_counts["on_message_appended"] += 1

    async def on_llm_call_started(
        self, run_id, iteration, *, semantic_messages=None, semantic_tools=None
    ) -> None:
        self.hook_counts["on_llm_call_started"] += 1

    async def on_llm_call_completed(
        self, run_id, response: LLMResponse, iteration, **kwargs: Any
    ) -> None:
        self.hook_counts["on_llm_call_completed"] += 1

    async def on_llm_call_failed(self, run_id, iteration, error, *, duration_ms=None) -> None:
        self.hook_counts["on_llm_call_failed"] += 1

    async def on_tool_started(self, run_id, tool_call: ToolCall, iteration) -> None:
        self.hook_counts["on_tool_started"] += 1

    async def on_tool_completed(self, run_id, tool_call, tool_result, iteration) -> None:
        self.hook_counts["on_tool_completed"] += 1

    async def on_governance_event(
        self, run_id, event_type: str, iteration, data, *, correlation_id=None
    ) -> None:
        self.hook_counts["on_governance_event"] += 1
        self.governance_events[event_type] += 1


# ---------------------------------------------------------------------------
# Tools (same as example 20).
# ---------------------------------------------------------------------------


@tool()
async def lookup_customer(customer_id: str) -> dict:
    """Look up a customer record by ID. Returns name, email, phone."""
    return {
        "id": customer_id,
        "name": "Jane Doe",
        "email": "jane@example.com",
        "phone": "+1-555-444-3333",
    }


@tool()
async def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to the given address."""
    return f"sent email to {to}: {subject}"


@tool()
async def refund(order_id: int) -> str:
    """Issue a refund for the given order. Requires manager approval."""
    return f"refunded ${order_id * 10:.2f} for order {order_id}"


@tool()
async def delete_account(user_id: int) -> str:
    """Permanently delete a user account."""
    return f"deleted user {user_id}"


@tool(target="client")
async def read_browser_state(field: str) -> str:
    """Read state from the user's browser. Pauses for client execution."""
    return ""


# ---------------------------------------------------------------------------
# Stage helpers (same as example 20).
# ---------------------------------------------------------------------------


async def _drain_stream(stream) -> Any:
    terminal = None
    async with stream:
        async for event in stream:
            if event.type in (
                RunEventType.RUN_COMPLETED,
                RunEventType.RUN_PAUSED,
                RunEventType.RUN_ERROR,
                RunEventType.RUN_CANCELLED,
            ):
                terminal = event.run_result
    return terminal


def _banner(title: str, subtitle: str) -> None:
    _console.print()
    _console.print(
        Panel(
            f"[bold white]{title}[/bold white]\n[dim]{subtitle}[/dim]",
            border_style="bright_magenta",
            width=min(76, _console.width),
            padding=(0, 1),
        )
    )


# ---------------------------------------------------------------------------
# Stages (same as example 20, parameterized on db_url + notifier).
# ---------------------------------------------------------------------------


async def stage_1_streaming_server_tools(db_url, notifier) -> None:
    _banner("Stage 1 — streaming + server tools", "Baseline lifecycle pair.")
    async with Agent(
        provider=AnthropicProvider(model=MODEL),
        prompt="You look up customers and email them. Be concise.",
        tools=[lookup_customer, send_email],
        database_url=db_url,
    ) as agent:
        await _drain_stream(
            agent.stream(
                "Look up customer C-1001 and send them a brief hello email.",
                notifier=notifier,
            )
        )


async def stage_2_streaming_deny(db_url, notifier) -> None:
    _banner("Stage 2 — streaming + deny policy", "delete_account is denied.")
    async with Agent(
        provider=AnthropicProvider(model=MODEL),
        prompt=(
            "You are a customer support agent. Use tools when asked. "
            "delete_account is the only way to delete users."
        ),
        tools=[lookup_customer, delete_account],
        deny=["delete_account"],
        database_url=db_url,
    ) as agent:
        await _drain_stream(
            agent.stream(
                "Delete user 42's account, then look up customer C-2002.",
                notifier=notifier,
            )
        )


async def stage_3_streaming_approve(db_url, notifier) -> None:
    _banner("Stage 3 — streaming approval", "Approve via resume_stream.")
    async with Agent(
        provider=AnthropicProvider(model=MODEL),
        prompt="You are a support agent. Call refund when asked.",
        tools=[refund],
        require_approval=["refund"],
        database_url=db_url,
    ) as agent:
        first = await _drain_stream(agent.stream("Refund order 7.", notifier=notifier))
        assert first.status == RunStatus.WAITING_APPROVAL, first.status
        await _drain_stream(agent.resume_stream(first.run_id, notifier=notifier))


async def stage_4_sync_reject(db_url, notifier) -> None:
    _banner("Stage 4 — sync rejection", "submit_approval(approved=False).")
    async with Agent(
        provider=AnthropicProvider(model=MODEL),
        prompt="You are a support agent. Call refund when asked.",
        tools=[refund],
        require_approval=["refund"],
        database_url=db_url,
    ) as agent:
        first = await _drain_stream(agent.stream("Refund order 99.", notifier=notifier))
        assert first.status == RunStatus.WAITING_APPROVAL, first.status
        await agent.submit_approval(
            first.run_id,
            approved=False,
            rejection_reason="Manager declined — order too old.",
            notifier=notifier,
        )


async def stage_5_streaming_client_tool(db_url, notifier) -> None:
    _banner("Stage 5 — streaming client tool", "resume_stream replays result.")
    async with Agent(
        provider=AnthropicProvider(model=MODEL),
        prompt=(
            "You read state from the browser using read_browser_state. "
            "Then summarize what you got. Be brief."
        ),
        tools=[read_browser_state],
        database_url=db_url,
    ) as agent:
        first = await _drain_stream(
            agent.stream("What does the 'theme' field say?", notifier=notifier)
        )
        assert first.status == RunStatus.WAITING_CLIENT_TOOL, first.status

        from dendrux.store import RunStore

        async with RunStore.from_database_url(db_url) as run_store:
            pauses = await run_store.get_pauses(first.run_id)
            active = next(p for p in pauses if p.resume_sequence_index is None)
            call_id = active.pending_tool_calls[0]["id"]

        await _drain_stream(
            agent.resume_stream(
                first.run_id,
                tool_results=[
                    ToolResult(
                        name="read_browser_state",
                        call_id=call_id,
                        payload=json.dumps("dark"),
                    )
                ],
                notifier=notifier,
            )
        )


async def stage_6_streaming_budget(db_url, notifier) -> None:
    _banner("Stage 6 — streaming + budget cap", "Tight max_tokens triggers exceeded.")
    async with Agent(
        provider=AnthropicProvider(model=MODEL),
        prompt=(
            "You are a research assistant. When asked to research, "
            "make multiple tool calls and write a long report."
        ),
        tools=[lookup_customer],
        budget=Budget(max_tokens=500),
        database_url=db_url,
    ) as agent:
        await _drain_stream(
            agent.stream(
                "Research customer C-3003 thoroughly. Look them up "
                "multiple times if needed and write a detailed profile.",
                notifier=notifier,
            )
        )


async def stage_7_sync_guardrails(db_url, notifier) -> None:
    _banner("Stage 7 — sync + PII guardrails", "PII detected + redacted.")
    async with Agent(
        provider=AnthropicProvider(model=MODEL),
        prompt=(
            "You look up customers and email them. The lookup tool "
            "returns PII; pass that PII through to send_email."
        ),
        tools=[lookup_customer, send_email],
        guardrails=[PII()],
        database_url=db_url,
    ) as agent:
        await agent.run(
            "Look up customer C-4004 and email them a short hello.",
            notifier=notifier,
        )


# ---------------------------------------------------------------------------
# OTel summary helpers.
# ---------------------------------------------------------------------------


def _print_stage_span_tree(exporter: InMemorySpanExporter, stage_name: str) -> None:
    """Print a Rich tree of every invoke_agent span produced by the latest stage.

    We diff against the cumulative-spans-so-far tracked on the exporter
    object (we stash a high-water mark on each call).
    """
    all_spans = list(exporter.get_finished_spans())
    seen = getattr(exporter, "_seen_count", 0)
    new_spans = all_spans[seen:]
    exporter._seen_count = len(all_spans)  # type: ignore[attr-defined]

    if not new_spans:
        _console.print(f"[dim]({stage_name}: no new spans)[/dim]")
        return

    invokes = [s for s in new_spans if s.name.startswith("invoke_agent")]
    if not invokes:
        _console.print(f"[dim]({stage_name}: no invoke_agent spans this stage)[/dim]")
        return

    by_id = {s.context.span_id: s for s in new_spans}
    children: dict[int, list[Any]] = {}
    for s in new_spans:
        if s.parent is not None and s.parent.span_id in by_id:
            children.setdefault(s.parent.span_id, []).append(s)

    _console.print(f"[bold cyan]{stage_name} — OTel span tree:[/bold cyan]")
    for inv in invokes:
        tree = Tree(_format_span_label(inv))
        _add_children(tree, inv, children)
        _console.print(tree)


def _format_span_label(span: Any) -> str:
    name = span.name
    status = "OK" if span.status.status_code == StatusCode.OK else "ERR"
    color = "green" if status == "OK" else "red"
    duration_ms = (span.end_time - span.start_time) // 1_000_000

    extras = []
    if t_in := span.attributes.get("gen_ai.usage.input_tokens"):
        t_out = span.attributes.get("gen_ai.usage.output_tokens", 0)
        extras.append(f"tokens={t_in}/{t_out}")
    if model := span.attributes.get("gen_ai.request.model"):
        extras.append(f"model={model}")
    if tool_name := span.attributes.get("dendrux.tool.name"):
        success = span.attributes.get("dendrux.tool.success")
        extras.append(f"tool={tool_name} ok={success}")
    if status_attr := span.attributes.get("dendrux.run.status"):
        extras.append(f"status={status_attr}")
    if orphan := span.attributes.get("dendrux.span.orphan_close_reason"):
        extras.append(f"[yellow]ORPHAN: {orphan}[/yellow]")

    extra_s = " · ".join(extras)
    extra_s = f" [dim]({extra_s})[/dim]" if extra_s else ""

    gov_events = [e for e in span.events if e.name.startswith("governance.")]
    gov_s = ""
    if gov_events:
        evs = ", ".join(e.name.replace("governance.", "") for e in gov_events)
        gov_s = f" [magenta]· gov: {evs}[/magenta]"

    return f"[{color}]{status}[/{color}] {name}  [dim]{duration_ms}ms[/dim]{extra_s}{gov_s}"


def _add_children(tree: Tree, parent: Any, children_map: dict[int, list[Any]]) -> None:
    for child in children_map.get(parent.context.span_id, []):
        sub = tree.add(_format_span_label(child))
        _add_children(sub, child, children_map)


def _print_otel_summary(exporter: InMemorySpanExporter) -> None:
    spans = list(exporter.get_finished_spans())

    by_op: Counter[str] = Counter()
    for s in spans:
        op = s.attributes.get("gen_ai.operation.name") or s.name.split()[0]
        by_op[str(op)] += 1

    error_count = sum(1 for s in spans if s.status.status_code == StatusCode.ERROR)
    orphan_count = sum(
        1 for s in spans if s.attributes.get("dendrux.span.orphan_close_reason") is not None
    )
    governance_event_count = sum(
        1 for s in spans for e in s.events if e.name.startswith("governance.")
    )

    parented_count = sum(1 for s in spans if s.parent is not None)
    root_count = len(spans) - parented_count

    table = Table(title="OTel surface summary", show_lines=False, box=None)
    table.add_column("metric", style="bold")
    table.add_column("value", justify="right")
    table.add_row("total spans", str(len(spans)))
    table.add_row("invoke_agent spans (root-of-run)", str(by_op.get("invoke_agent", 0)))
    table.add_row("chat spans", str(by_op.get("chat", 0)))
    table.add_row("execute_tool spans", str(by_op.get("execute_tool", 0)))
    table.add_row("ERROR-status spans", str(error_count))
    table.add_row("orphan-closed spans", str(orphan_count))
    table.add_row("governance span events", str(governance_event_count))
    table.add_row("spans with a parent", str(parented_count))
    table.add_row("root spans", str(root_count))
    _console.print(table)


def _print_invariants_check(exporter: InMemorySpanExporter) -> None:
    spans = list(exporter.get_finished_spans())
    by_id = {s.context.span_id: s for s in spans}

    # 1. Every invoke_agent span has dendrux.framework=dendrux + dendrux.run.id.
    bad_invoke = [
        s
        for s in spans
        if s.name.startswith("invoke_agent")
        and (
            s.attributes.get("dendrux.framework") != "dendrux"
            or not s.attributes.get("dendrux.run.id")
        )
    ]

    # 2. Every chat / execute_tool span has a parent in the same trace.
    orphan_children = [
        s
        for s in spans
        if (s.name.startswith("chat") or s.name.startswith("execute_tool"))
        and (s.parent is None or s.parent.span_id not in by_id)
    ]

    # 3. Every chat span carries either both or neither of the usage attrs.
    inconsistent_usage = [
        s
        for s in spans
        if s.name.startswith("chat")
        and (
            ("gen_ai.usage.input_tokens" in s.attributes)
            != ("gen_ai.usage.output_tokens" in s.attributes)
        )
    ]

    # 4. No leaked orphans (we close them deliberately in cleanup, but there
    #    should still be a reason attribute on every one we emit).
    orphan_no_reason = [
        s
        for s in spans
        if s.attributes.get("dendrux.span.orphan_close_reason") is not None
        and s.status.status_code != StatusCode.ERROR
    ]

    table = Table(title="Invariants", show_lines=False, box=None)
    table.add_column("invariant", style="bold")
    table.add_column("violations", justify="right")
    table.add_column("status")
    rows = [
        ("invoke_agent has framework + run.id", len(bad_invoke)),
        ("chat / execute_tool has live parent", len(orphan_children)),
        ("chat has both usage attrs (or neither)", len(inconsistent_usage)),
        ("orphan-closed spans are ERROR", len(orphan_no_reason)),
    ]
    for label, n in rows:
        status = "[green]✓[/green]" if n == 0 else f"[red]× {n} fail[/red]"
        table.add_row(label, str(n), status)
    _console.print(table)


# ---------------------------------------------------------------------------
# Coverage report (same as example 20).
# ---------------------------------------------------------------------------


_EXPECTED_HOOKS = (
    "on_run_started",
    "on_run_finished",
    "on_run_failed",
    "on_message_appended",
    "on_llm_call_started",
    "on_llm_call_completed",
    "on_llm_call_failed",
    "on_tool_started",
    "on_tool_completed",
    "on_governance_event",
)

_EXPECTED_GOV_EVENTS = (
    GovernanceEventType.POLICY_DENIED,
    GovernanceEventType.APPROVAL_REQUESTED,
    GovernanceEventType.APPROVAL_DECIDED,
    GovernanceEventType.BUDGET_THRESHOLD,
    GovernanceEventType.BUDGET_EXCEEDED,
    GovernanceEventType.GUARDRAIL_DETECTED,
    GovernanceEventType.GUARDRAIL_REDACTED,
)


def _print_hook_coverage(tracker: HookTracker) -> None:
    _console.print()
    _console.print(
        Panel(
            "[bold]Notifier surface coverage[/bold]",
            border_style="bright_green",
            width=min(76, _console.width),
            padding=(0, 1),
        )
    )

    hooks_table = Table(title="Lifecycle hooks", show_lines=False, box=None)
    hooks_table.add_column("hook", style="bold")
    hooks_table.add_column("count", justify="right")
    hooks_table.add_column("status")
    for hook in _EXPECTED_HOOKS:
        count = tracker.hook_counts.get(hook, 0)
        status = "[green]✓[/green]" if count > 0 else "[dim]–[/dim]"
        hooks_table.add_row(hook, str(count), status)
    _console.print(hooks_table)

    gov_table = Table(title="Governance event types", show_lines=False, box=None)
    gov_table.add_column("event", style="bold")
    gov_table.add_column("count", justify="right")
    gov_table.add_column("status")
    for event in _EXPECTED_GOV_EVENTS:
        count = tracker.governance_events.get(event.value, 0)
        status = "[green]✓[/green]" if count > 0 else "[red]×[/red]"
        gov_table.add_row(event.value, str(count), status)
    _console.print(gov_table)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


async def main() -> None:
    # OTel SDK: install a TracerProvider with InMemorySpanExporter so we
    # can both see spans during stages and summarize at the end. In a real
    # app this would be the host's TracerProvider with OTLP/Jaeger/etc.
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)

    tmpdir = tempfile.mkdtemp(prefix="dendrux_otel_tour_")
    db_path = Path(tmpdir) / "tour.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"

    tracker = HookTracker()
    console_notifier = ConsoleNotifier()
    otel_notifier = OpenTelemetryNotifier()
    notifier = CompositeNotifier([console_notifier, tracker, otel_notifier])

    _console.print(
        Panel(
            f"[bold]OTel notifier complete cycle[/bold]\n"
            f"[dim]model={MODEL} · db={db_path.name}[/dim]\n"
            f"[dim]OTel: in-memory exporter; production would be OTLP/Jaeger/etc.[/dim]",
            border_style="bright_blue",
            padding=(0, 1),
        )
    )

    stages = [
        ("stage 1", stage_1_streaming_server_tools),
        ("stage 2", stage_2_streaming_deny),
        ("stage 3", stage_3_streaming_approve),
        ("stage 4", stage_4_sync_reject),
        ("stage 5", stage_5_streaming_client_tool),
        ("stage 6", stage_6_streaming_budget),
        ("stage 7", stage_7_sync_guardrails),
    ]
    for label, stage in stages:
        try:
            await stage(db_url, notifier)
        except Exception as exc:
            _console.print(f"[red]{label} raised:[/red] {type(exc).__name__}: {exc}")
        _print_stage_span_tree(exporter, label)

    _console.print()
    _console.print(
        Panel(
            "[bold]Final OTel surface report[/bold]",
            border_style="bright_yellow",
            width=min(76, _console.width),
            padding=(0, 1),
        )
    )
    _print_otel_summary(exporter)
    _console.print()
    _print_invariants_check(exporter)

    _print_hook_coverage(tracker)

    db_path.unlink(missing_ok=True)
    Path(tmpdir).rmdir()


if __name__ == "__main__":
    asyncio.run(main())
