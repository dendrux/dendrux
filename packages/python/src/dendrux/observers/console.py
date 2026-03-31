"""Console observer — rich terminal output for agent runs.

Opt-in observer that prints agent lifecycle events to the terminal
using rich formatting. Plugs into the standard LoopObserver protocol.

Usage:
    from dendrux.observers.console import ConsoleObserver

    result = await agent.run("do the thing", observer=ConsoleObserver())
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from dendrux.types import Message, Role, ToolCall, ToolResult

_console = Console()


class ConsoleObserver:
    """Rich terminal observer for agent runs.

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
        self._run_start = 0.0

    async def on_message_appended(self, message: Message, iteration: int) -> None:
        """Called when a message is appended to history."""
        # Iteration header on change
        if iteration != self._iteration:
            self._iteration = iteration
            _console.print()
            _console.print(
                f"  [bold bright_cyan]Step {iteration}[/bold bright_cyan]"
            )

        if message.role == Role.USER and iteration == 0:
            if not self._started:
                self._started = True
                self._run_start = time.monotonic()
                _console.print()
                _console.print(
                    Panel(
                        f"[white]{_truncate(message.content, 200)}[/white]",
                        border_style="bright_blue",
                        width=min(76, _console.width),
                        padding=(0, 1),
                    )
                )

        elif message.role == Role.ASSISTANT:
            if message.tool_calls:
                for tc in message.tool_calls:
                    self._tool_starts[tc.id] = time.monotonic()
                    params_str = ""
                    if self._show_params and tc.params:
                        params_str = f" [dim]{_format_params(tc.params)}[/dim]"
                    _console.print(
                        f"  [yellow]  calling[/yellow] [bold]{tc.name}[/bold]{params_str}"
                    )

    async def on_llm_call_completed(
        self,
        response: Any,
        iteration: int,
        **kwargs: Any,
    ) -> None:
        """Called after an LLM call completes."""
        tokens = response.usage.total_tokens if response.usage else 0
        self._total_tokens += tokens
        duration_ms = kwargs.get("duration_ms")
        duration_str = f" in {duration_ms / 1000:.1f}s" if duration_ms else ""

        _console.print(
            f"  [dim]  llm [bright_white]{tokens:,}[/bright_white] tokens{duration_str}[/dim]"
        )

    async def on_tool_completed(
        self, tool_call: ToolCall, tool_result: ToolResult, iteration: int
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

    def print_summary(self, result: Any) -> None:
        """Print a final summary panel. Call after agent.run() completes."""
        total_time = time.monotonic() - self._run_start if self._run_start else 0

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="dim")
        table.add_column(style="bold")

        status_style = "green" if result.status.value == "success" else "yellow"
        table.add_row("status", f"[{status_style}]{result.status.value}[/{status_style}]")
        table.add_row("iterations", str(result.iteration_count))
        table.add_row("tokens", f"{result.usage.total_tokens:,}")
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
