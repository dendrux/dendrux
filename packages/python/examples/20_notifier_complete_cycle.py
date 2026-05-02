"""Example 20: Notifier complete cycle — every hook, every flow.

Canonical reference exercise for the LoopNotifier surface. Each stage
is a self-contained vignette that mixes one runner feature with
streaming so we can watch the lifecycle pair open and close. At the
end, ``HookTracker`` prints a coverage table summarizing which hooks
and which governance event types fired across the whole tour.

When the notifier or governance surface changes, run this and compare
the coverage table against the expected baseline (commented at the
bottom of this file).

Stages:
    1. Streaming + server tools          (baseline lifecycle pair)
    2. Streaming + deny policy           (POLICY_DENIED)
    3. Streaming approve via resume_stream (APPROVAL_REQUESTED + DECIDED)
    4. Sync reject via submit_approval   (APPROVAL_REQUESTED + DECIDED)
    5. Streaming client-tool resume      (resume_stream + submit_tool_results)
    6. Streaming + budget cap            (BUDGET_THRESHOLD / BUDGET_EXCEEDED)
    7. Sync + PII guardrails             (GUARDRAIL_DETECTED + REDACTED)

Run:
    ANTHROPIC_API_KEY=sk-... python examples/20_notifier_complete_cycle.py
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from dendrux import Agent, tool
from dendrux.guardrails import PII
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.loops.base import BaseNotifier
from dendrux.notifiers.composite import CompositeNotifier
from dendrux.notifiers.console import ConsoleNotifier
from dendrux.types import Budget, GovernanceEventType, RunEventType, RunStatus, ToolResult

if TYPE_CHECKING:
    from dendrux.types import LLMResponse, Message, ToolCall

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

_console = Console()
MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# HookTracker — records every notifier callback for the final coverage table.
# ---------------------------------------------------------------------------


class HookTracker(BaseNotifier):
    """Records every notifier callback. Doubles as the regression check."""

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
# Tools shared across stages.
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
    return ""  # body never executes server-side


# ---------------------------------------------------------------------------
# Stage helpers.
# ---------------------------------------------------------------------------


async def _drain_stream(stream) -> Any:
    """Iterate a RunStream to completion, returning the terminal RunResult.

    All side-effects we care about flow through the notifier — we just
    need to consume the wire so the run fully terminates.
    """
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
# Stages.
# ---------------------------------------------------------------------------


async def stage_1_streaming_server_tools(db_url: str, notifier) -> None:
    _banner(
        "Stage 1 — streaming + server tools",
        "Baseline lifecycle pair on a clean run with one server tool.",
    )
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


async def stage_2_streaming_deny(db_url: str, notifier) -> None:
    _banner(
        "Stage 2 — streaming + deny policy",
        "delete_account is denied; the LLM sees a tool error and recovers.",
    )
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


async def stage_3_streaming_approve(db_url: str, notifier) -> None:
    _banner(
        "Stage 3 — streaming approval (approve via resume_stream)",
        "Initial run pauses; resume_stream(no-args) executes the approved tool.",
    )
    async with Agent(
        provider=AnthropicProvider(model=MODEL),
        prompt="You are a support agent. Call refund when asked.",
        tools=[refund],
        require_approval=["refund"],
        database_url=db_url,
    ) as agent:
        first = await _drain_stream(agent.stream("Refund order 7.", notifier=notifier))
        assert first.status == RunStatus.WAITING_APPROVAL, first.status
        # Approve via streaming resume — fires a fresh lifecycle pair.
        await _drain_stream(agent.resume_stream(first.run_id, notifier=notifier))


async def stage_4_sync_reject(db_url: str, notifier) -> None:
    _banner(
        "Stage 4 — sync rejection (submit_approval approved=False)",
        "Mirrors stage 3 but the operator declines; rejection fires a sync"
        " resume that surfaces the rejection on the LLM's next turn.",
    )
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


async def stage_5_streaming_client_tool(db_url: str, notifier) -> None:
    _banner(
        "Stage 5 — streaming client tool (resume_stream)",
        "Run pauses on a client tool; resume_stream replays the result.",
    )
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

        # Discover the call_id via the public RunStore facade.
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


async def stage_6_streaming_budget(db_url: str, notifier) -> None:
    _banner(
        "Stage 6 — streaming + budget cap",
        "Low max_tokens triggers BUDGET_THRESHOLD / BUDGET_EXCEEDED.",
    )
    async with Agent(
        provider=AnthropicProvider(model=MODEL),
        prompt=(
            "You are a research assistant. When asked to research, "
            "make multiple tool calls and write a long report."
        ),
        tools=[lookup_customer],
        budget=Budget(max_tokens=500),  # tight cap, fires fast
        database_url=db_url,
    ) as agent:
        await _drain_stream(
            agent.stream(
                "Research customer C-3003 thoroughly. Look them up "
                "multiple times if needed and write a detailed profile.",
                notifier=notifier,
            )
        )


async def stage_7_sync_guardrails(db_url: str, notifier) -> None:
    _banner(
        "Stage 7 — sync + PII guardrails",
        "PII guardrail (default action=redact) fires DETECTED + REDACTED."
        " resume_stream rejects guardrails, so this stage is sync.",
    )
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
# Coverage report.
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

# Governance event types this tour should cover. on_run_failed and
# on_llm_call_failed are intentionally not exercised — they need a
# forced failure path that distorts the rest of the demo. Add them
# to a separate stage if you want full coverage.
_EXPECTED_GOV_EVENTS = (
    GovernanceEventType.POLICY_DENIED,
    GovernanceEventType.APPROVAL_REQUESTED,
    GovernanceEventType.APPROVAL_DECIDED,
    GovernanceEventType.BUDGET_THRESHOLD,
    GovernanceEventType.BUDGET_EXCEEDED,
    GovernanceEventType.GUARDRAIL_DETECTED,
    GovernanceEventType.GUARDRAIL_REDACTED,
)


def _print_coverage(tracker: HookTracker) -> None:
    _console.print()
    _console.print(
        Panel(
            "[bold]Notifier surface coverage[/bold]\n"
            "[dim]Compare counts to the baseline at the bottom of this file.[/dim]",
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
    # Surface anything unexpected too.
    extras = sorted(set(tracker.governance_events) - {e.value for e in _EXPECTED_GOV_EVENTS})
    for event_type in extras:
        gov_table.add_row(
            event_type, str(tracker.governance_events[event_type]), "[yellow]?[/yellow]"
        )
    _console.print(gov_table)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


async def main() -> None:
    tmpdir = tempfile.mkdtemp(prefix="dendrux_notifier_tour_")
    db_path = Path(tmpdir) / "tour.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"

    tracker = HookTracker()
    console_notifier = ConsoleNotifier()
    notifier = CompositeNotifier([console_notifier, tracker])

    _console.print(
        Panel(
            f"[bold]Notifier complete cycle[/bold]\n[dim]model={MODEL} · db={db_path.name}[/dim]",
            border_style="bright_blue",
            padding=(0, 1),
        )
    )

    stages = [
        stage_1_streaming_server_tools,
        stage_2_streaming_deny,
        stage_3_streaming_approve,
        stage_4_sync_reject,
        stage_5_streaming_client_tool,
        stage_6_streaming_budget,
        stage_7_sync_guardrails,
    ]
    for stage in stages:
        try:
            await stage(db_url, notifier)
        except Exception as exc:
            _console.print(f"[red]stage {stage.__name__} raised:[/red] {type(exc).__name__}: {exc}")

    _print_coverage(tracker)

    # Cleanup
    db_path.unlink(missing_ok=True)
    Path(tmpdir).rmdir()


# ---------------------------------------------------------------------------
# Expected baseline (update when intentional surface changes land).
# ---------------------------------------------------------------------------
#
# Lifecycle hooks
#   on_run_started        ≥ 10  (one per stream/sync entry; pause+resume = 2)
#   on_run_finished       == on_run_started   (lifecycle pair contract)
#   on_run_failed         == 0  (no stage forces a failure)
#   on_message_appended   ≥ 30  (varies with LLM tool-call count)
#   on_llm_call_started   ≥ 12
#   on_llm_call_completed ≥ 12
#   on_llm_call_failed    == 0
#   on_tool_started       == on_tool_completed (started/completed pair contract)
#   on_governance_event   ≥ 12
#
# Governance event types that must fire on every run
#   policy.denied         (stage 2)
#   approval.requested    (stages 3, 4)
#   approval.decided      (stages 3, 4 — fires twice, once approved + once rejected)
#   budget.threshold      (stage 6 — typically multiple at 50/75/90 %)
#   budget.exceeded       (stage 6)
#   guardrail.detected    (stage 7 — fires per turn that touches PII)
#   guardrail.redacted    (stage 7)
#
# Known asymmetries surfaced by this example (track in dendrux issues)
#   - on_tool_started < on_tool_completed:
#         The approval-approve path executes via _execute_approved_tools,
#         which historically fired only tool_completed. Closed by the
#         lifecycle-hook-gaps PR (firing tool_started before _execute_tool).
#         Until that PR lands, expect a +1 tool_completed delta per
#         approval-approve resume.
#   - approval.decided fires only once (stages 3 + 4 should each emit it):
#         Sync submit_approval(approved=False) flows through resume_claimed
#         which calls _resume_core with expected_status="running", failing
#         the rejection-emit gate at runner.py:2002. Streaming rejection
#         (resume_stream with rejection ToolResults) does not have this
#         issue because resume_stream re-derives expected_status from
#         the actual pause status. Tracked separately.
#   - Deny path emits neither tool_started nor tool_completed (deliberate,
#         per src/dendrux/loops/react.py:370 — "denied tools are not
#         executions"). For OTel parity an opt-in paired emit may be useful,
#         but this is a design call, not a bug.


if __name__ == "__main__":
    asyncio.run(main())
