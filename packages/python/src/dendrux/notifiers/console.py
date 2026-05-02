"""Console notifier — rich terminal output for agent runs.

Opt-in notifier that prints agent lifecycle events to the terminal
using rich formatting. Plugs into the standard LoopNotifier protocol.

Usage:
    from dendrux.notifiers.console import ConsoleNotifier

    result = await agent.run("do the thing", notifier=ConsoleNotifier())
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from dendrux.loops.base import BaseNotifier
from dendrux.types import GovernanceEventType as _GovType
from dendrux.types import Message, Role, ToolCall, ToolResult

if TYPE_CHECKING:
    from dendrux.types import LLMResponse, RunResult, ToolDef

# Run statuses for which on_run_finished prints a "paused" inline marker.
# The pause states are still terminal-from-runner-perspective, but the
# final SUCCESS/ERROR/CANCELLED flavors get the rich print_summary panel
# instead so we don't double-paint when callers also call print_summary.
_PAUSE_STATUSES = {"waiting_client_tool", "waiting_human_input", "waiting_approval"}

_console = Console()


class ConsoleNotifier(BaseNotifier):
    """Rich terminal notifier for agent runs.

    Args:
        show_llm_text: Show the LLM's final text content (default False).
        max_text_length: Truncate displayed text to this length (default 120).
        show_params: Show tool call parameters (default True).
    """

    def __init__(
        self,
        *,
        show_llm_text: bool = False,
        max_text_length: int = 120,
        show_params: bool = True,
    ) -> None:
        self._show_llm_text = show_llm_text
        self._max_text_length = max_text_length
        self._show_params = show_params
        self._iteration = 0
        self._tool_starts: dict[str, float] = {}
        self._started = False
        self._total_tokens = 0
        self._total_tools = 0
        self._total_tool_time = 0.0
        self._total_cache_read = 0
        self._total_cache_creation = 0
        self._run_start = 0.0

    async def on_run_started(
        self,
        run_id: str,
        *,
        agent_name: str | None = None,
        agent_model: str | None = None,
    ) -> None:
        """Called once per run-or-resume entry, before any other event."""
        # Track only the first run_started — pause/resume cycles re-fire
        # the lifecycle pair, and we don't want to reset _run_start on each.
        if not self._started:
            self._started = True
            self._run_start = time.monotonic()
        meta = " · ".join(part for part in (agent_name, agent_model) if part is not None)
        meta_str = f" [dim]{meta}[/dim]" if meta else ""
        _console.print(f"\n[bright_blue]▶ run[/bright_blue] [dim]{run_id[-12:]}[/dim]{meta_str}")

    async def on_run_finished(self, run_id: str, result: RunResult) -> None:
        """Called when a run reaches a terminal state — including pause.

        For pause states (the run handed control back), prints an inline
        marker so the trace is bookended cleanly across resume cycles.
        For final terminals (SUCCESS / ERROR / CANCELLED) we stay quiet
        so callers using ``print_summary(result)`` don't double-paint.
        """
        if result.status.value in _PAUSE_STATUSES:
            _console.print(
                f"  [bright_yellow]‖ paused[/bright_yellow] [dim]{result.status.value}[/dim]"
            )

    async def on_run_failed(
        self,
        run_id: str,
        error: BaseException,
        *,
        iteration: int | None = None,
    ) -> None:
        """Called when a run terminates with an unhandled exception."""
        _console.print()
        _console.print(
            Panel(
                f"[bold red]{type(error).__name__}[/bold red]: "
                f"[white]{_truncate(str(error), 240)}[/white]",
                title="[bold]run failed[/bold]",
                border_style="red",
                width=min(76, _console.width),
                padding=(0, 1),
            )
        )

    async def on_message_appended(self, run_id: str, message: Message, iteration: int) -> None:
        """Called when a message is appended to history."""
        if iteration != self._iteration:
            self._iteration = iteration
            _console.print()
            _console.print(f"  [bold bright_cyan]Step {iteration}[/bold bright_cyan]")

        if message.role == Role.USER and iteration == 0 and message.content:
            # Inline-render the user input panel once, on the first turn,
            # the first time we see a user message. Resume turns reuse the
            # existing history so this won't re-fire.
            _console.print()
            _console.print(
                Panel(
                    f"[white]{_truncate(message.content, 200)}[/white]",
                    border_style="bright_blue",
                    width=min(76, _console.width),
                    padding=(0, 1),
                )
            )

    async def on_llm_call_started(
        self,
        run_id: str,
        iteration: int,
        *,
        semantic_messages: list[Message] | None = None,
        semantic_tools: list[ToolDef] | None = None,
    ) -> None:
        """Called when an LLM call begins. We don't print here to keep the
        trace tight — the matching on_llm_call_completed prints duration
        and tokens. Subclasses can override for live-latency displays."""
        return None

    async def on_llm_call_failed(
        self,
        run_id: str,
        iteration: int,
        error: BaseException,
        *,
        duration_ms: int | None = None,
    ) -> None:
        """Called when an LLM call raises (e.g. provider error, stream cut)."""
        duration_str = f" [dim]after {duration_ms / 1000:.1f}s[/dim]" if duration_ms else ""
        _console.print(
            f"  [red]  llm[/red]    [bold]{type(error).__name__}[/bold]{duration_str} "
            f"[dim]{_truncate(str(error), 100)}[/dim]"
        )

    async def on_tool_started(self, run_id: str, tool_call: ToolCall, iteration: int) -> None:
        """Called immediately before a tool dispatches.

        Replaces the prior on_message_appended-based "calling" display so
        the marker also fires on resume paths (client-tool replay,
        approval-approved server tools), not just fresh assistant turns.
        """
        self._tool_starts[tool_call.id] = time.monotonic()
        params_str = ""
        if self._show_params and tool_call.params:
            params_str = f" [dim]{_format_params(tool_call.params)}[/dim]"
        _console.print(f"  [yellow]  calling[/yellow] [bold]{tool_call.name}[/bold]{params_str}")

    async def on_llm_call_completed(
        self,
        run_id: str,
        response: LLMResponse,
        iteration: int,
        *,
        semantic_messages: list[Message] | None = None,
        semantic_tools: list[ToolDef] | None = None,
        duration_ms: int | None = None,
        guardrail_findings: dict[str, Any] | None = None,
    ) -> None:
        """Called after an LLM call completes."""
        tokens = response.usage.total_tokens if response.usage else 0
        self._total_tokens += tokens
        duration_str = f" in {duration_ms / 1000:.1f}s" if duration_ms else ""

        cache_str = ""
        if response.usage:
            cache_read = response.usage.cache_read_input_tokens or 0
            cache_create = response.usage.cache_creation_input_tokens or 0
            self._total_cache_read += cache_read
            self._total_cache_creation += cache_create
            parts = []
            if cache_read > 0:
                parts.append(f"[bright_green]{cache_read:,} cached[/bright_green]")
            if cache_create > 0:
                parts.append(f"[yellow]{cache_create:,} cache-write[/yellow]")
            if parts:
                cache_str = " · " + " · ".join(parts)

        _console.print(
            f"  [dim]  llm [bright_white]{tokens:,}[/bright_white] tokens{duration_str}[/dim]"
            f"{cache_str}"
        )

    async def on_tool_completed(
        self,
        run_id: str,
        tool_call: ToolCall,
        tool_result: ToolResult,
        iteration: int,
    ) -> None:
        """Called after a tool execution completes."""
        elapsed = 0.0
        if tool_call.id in self._tool_starts:
            elapsed = time.monotonic() - self._tool_starts.pop(tool_call.id)
        elif tool_result.duration_ms:
            elapsed = tool_result.duration_ms / 1000

        self._total_tools += 1
        self._total_tool_time += elapsed
        duration_str = f" [dim]{elapsed:.1f}s[/dim]" if elapsed else ""

        if tool_result.success:
            _console.print(
                f"  [green]  done[/green]    [bold]{tool_call.name}[/bold]{duration_str}"
            )
        else:
            error = tool_result.error or "unknown error"
            if "reached its maximum" in error:
                _console.print(
                    f"  [bright_red]  limit[/bright_red]   [bold]{tool_call.name}[/bold]"
                    f" [dim]max calls reached[/dim]"
                )
            else:
                _console.print(
                    f"  [red]  fail[/red]    [bold]{tool_call.name}[/bold]{duration_str}"
                )

    async def on_governance_event(
        self,
        run_id: str,
        event_type: str,
        iteration: int,
        data: dict[str, Any],
        *,
        correlation_id: str | None = None,
    ) -> None:
        """Called when a governance action fires."""
        if event_type == _GovType.BUDGET_THRESHOLD:
            frac = data.get("fraction", 0)
            used = data.get("used", 0)
            max_t = data.get("max", 0)
            _console.print(
                f"  [bright_yellow]  budget[/bright_yellow] "
                f"[bold]{frac:.0%}[/bold] used "
                f"[dim]({used:,} / {max_t:,} tokens)[/dim]"
            )
        elif event_type == _GovType.BUDGET_EXCEEDED:
            used = data.get("used", 0)
            max_t = data.get("max", 0)
            _console.print(
                f"  [bright_red]  budget[/bright_red] "
                f"[bold]exceeded[/bold] "
                f"[dim]({used:,} / {max_t:,} tokens)[/dim]"
            )
        elif event_type == _GovType.GUARDRAIL_DETECTED:
            direction = data.get("direction", "")
            count = data.get("findings_count", 0)
            entities = data.get("entities", [])
            _console.print(
                f"  [bright_cyan]  guard[/bright_cyan]  "
                f"[bold]{count} finding(s)[/bold] "
                f"[dim]{direction} ({', '.join(entities)})[/dim]"
            )
        elif event_type == _GovType.GUARDRAIL_REDACTED:
            direction = data.get("direction", "")
            entities = data.get("entities", [])
            _console.print(
                f"  [bright_magenta]  guard[/bright_magenta]  "
                f"[bold]redacted[/bold] "
                f"[dim]{direction} ({', '.join(entities)})[/dim]"
            )
        elif event_type == _GovType.GUARDRAIL_BLOCKED:
            error = data.get("error", "blocked")
            _console.print(
                f"  [bright_red]  guard[/bright_red]  [bold]blocked[/bold] [dim]{error}[/dim]"
            )
        elif event_type == _GovType.SKILL_REGISTERED:
            skill_name = data.get("skill_name", "")
            _console.print(
                f"  [bright_green]  skill[/bright_green]  "
                f"[bold]{skill_name}[/bold] [dim]registered[/dim]"
            )
        elif event_type == _GovType.SKILL_DENIED:
            skill_name = data.get("skill_name", "")
            _console.print(
                f"  [bright_yellow]  skill[/bright_yellow]  "
                f"[bold]{skill_name}[/bold] [dim]denied[/dim]"
            )
        elif event_type == _GovType.SKILL_INVOKED:
            skill_name = data.get("skill_name", "")
            _console.print(
                f"  [bright_cyan]  skill[/bright_cyan]  "
                f"[bold]{skill_name}[/bold] [dim]invoked[/dim]"
            )
        elif event_type == _GovType.MCP_CONNECTED:
            source = data.get("source_name", "")
            count = data.get("tool_count", 0)
            _console.print(
                f"  [bright_green]  mcp[/bright_green]    "
                f"[bold]{source}[/bold] [dim]{count} tool(s)[/dim]"
            )
        elif event_type == _GovType.MCP_ERROR:
            error = data.get("error", "unknown")
            _console.print(
                f"  [bright_red]  mcp[/bright_red]    [bold]error[/bold] [dim]{error}[/dim]"
            )
        elif event_type == _GovType.POLICY_DENIED:
            tool_name = data.get("tool_name", "")
            reason = data.get("reason", "denied by policy")
            _console.print(
                f"  [bright_red]  deny[/bright_red]   [bold]{tool_name}[/bold] [dim]{reason}[/dim]"
            )
        elif event_type == _GovType.APPROVAL_REQUESTED:
            tool_name = data.get("tool_name", "")
            _console.print(
                f"  [bright_yellow]  approve[/bright_yellow] "
                f"[bold]{tool_name}[/bold] [dim]waiting for human sign-off[/dim]"
            )
        elif event_type == _GovType.APPROVAL_DECIDED:
            tool_name = data.get("tool_name", "")
            decision = data.get("decision", "unknown")
            color = "bright_green" if decision == "approved" else "bright_red"
            _console.print(
                f"  [{color}]  approve[/{color}] [bold]{tool_name}[/bold] [dim]{decision}[/dim]"
            )
        elif event_type == _GovType.PROVIDER_RETRY:
            attempt = data.get("attempt", "?")
            reason = data.get("reason", "")
            _console.print(
                f"  [bright_yellow]  retry[/bright_yellow]  "
                f"[bold]attempt {attempt}[/bold] [dim]{reason}[/dim]"
            )
        elif event_type == _GovType.GUARDRAIL_UNMAPPED_PLACEHOLDER:
            placeholders = data.get("placeholders", [])
            _console.print(
                f"  [bright_red]  guard[/bright_red]  "
                f"[bold]unmapped placeholder[/bold] [dim]{placeholders}[/dim]"
            )
        else:
            tool_name = data.get("tool_name", "")
            reason = data.get("reason", event_type)
            _console.print(
                f"  [bright_magenta]  policy[/bright_magenta] [bold]{tool_name}[/bold]"
                f" [dim]{reason}[/dim]"
            )

    def print_summary(self, result: RunResult) -> None:
        """Print a final summary panel. Call after agent.run() completes."""
        total_time = time.monotonic() - self._run_start if self._run_start else 0

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="dim")
        table.add_column(style="bold")

        status_style = "green" if result.status.value == "success" else "yellow"
        table.add_row("status", f"[{status_style}]{result.status.value}[/{status_style}]")
        table.add_row("iterations", str(result.iteration_count))
        table.add_row("tokens", f"{result.usage.total_tokens:,}")

        cache_read = result.usage.cache_read_input_tokens or self._total_cache_read
        cache_create = result.usage.cache_creation_input_tokens or self._total_cache_creation
        if cache_read or cache_create:
            denom = (result.usage.input_tokens or 0) + (cache_read or 0)
            ratio = (cache_read / denom) if denom else 0.0
            table.add_row(
                "cache read",
                f"[bright_green]{cache_read:,}[/bright_green] [dim]({ratio:.0%} hit ratio)[/dim]",
            )
            if cache_create:
                table.add_row("cache writes", f"[yellow]{cache_create:,}[/yellow]")

        table.add_row("tools called", str(self._total_tools))
        if total_time:
            table.add_row("total time", f"{total_time:.1f}s")

        _console.print()
        _console.print(
            Panel(
                table,
                title="[bold]Run Complete[/bold]",
                border_style="bright_green" if result.status.value == "success" else "yellow",
                width=min(50, _console.width),
                padding=(0, 1),
            )
        )


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len, adding ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _format_params(params: dict[str, Any]) -> str:
    """Format tool params for display — short, no large values."""
    if not params:
        return ""
    parts = []
    for k, v in params.items():
        val = repr(v)
        if len(val) > 30:
            val = val[:27] + "..."
        parts.append(f"{k}={val}")
    result = ", ".join(parts)
    if len(result) > 60:
        result = result[:57] + "..."
    return result
