"""Attach the executing task to its notifier trace."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import dendrux.runtime.context as _context
from dendrux.loops._helpers import (
    notifier_span_link,
    notifier_trace_link,
    notify_run_started,
)
from dendrux.runtime._tracelink import validate_trace_link
from dendrux.runtime.context import (
    DelegationContext,
    reset_delegation_context,
    set_delegation_context,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from dendrux.loops.base import LoopNotifier
    from dendrux.runtime.state import StateStore
    from dendrux.types import RunResult, ToolCall, ToolResult


class _TraceNotifier:
    """Forward hooks while attaching a tool span only to its executing task."""

    def __init__(self, inner: LoopNotifier) -> None:
        self.inner = inner
        self.tokens: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    async def on_tool_started(self, run_id: str, call: ToolCall, iteration: int) -> None:
        await self.inner.on_tool_started(run_id, call, iteration)
        ctx = _context.get_delegation_context()
        link = validate_trace_link(notifier_span_link(self.inner, run_id, call.id))
        if ctx is not None and ctx.run_id == run_id and link is not None:
            self.tokens[call.id] = set_delegation_context(replace(ctx, tool_span_link=link))

    async def on_tool_completed(
        self, run_id: str, call: ToolCall, result: ToolResult, iteration: int
    ) -> None:
        try:
            await self.inner.on_tool_completed(run_id, call, result, iteration)
        finally:
            token = self.tokens.pop(call.id, None)
            if token is not None:
                reset_delegation_context(token)


async def _start_trace(
    notifier: LoopNotifier | None,
    store: StateStore | None,
    run_id: str,
    *,
    agent_name: str | None,
    agent_model: str | None,
    parent: DelegationContext | None = None,
    saved: str | None = None,
    on_started: Callable[[], None] = lambda: None,
) -> tuple[str | None, Any]:
    """Start the notifier, persist its valid run link, and retain resume identity."""
    from dendrux.runtime._capabilities import capabilities

    saved = validate_trace_link(saved)
    incoming = saved
    if incoming is None and parent is not None:
        try:
            otel_child = getattr(notifier, "_dendrux_otel", False) is True
        except Exception:
            otel_child = False
        incoming = (
            validate_trace_link(parent.tool_span_link) if otel_child else None
        ) or validate_trace_link(parent.trace_link)
    await notify_run_started(
        notifier, run_id, agent_name=agent_name, agent_model=agent_model, trace_link=incoming
    )
    on_started()
    link = saved or validate_trace_link(notifier_trace_link(notifier, run_id))
    if (
        link is not None
        and saved is None
        and store is not None
        and capabilities(store).fenced_writes
    ):
        await store.set_trace_link(
            run_id, claim_epoch=getattr(store, "claim_epoch", 0), trace_link=link
        )
    return link, _TraceNotifier(notifier) if notifier is not None else None


def _cause_type(error: BaseException) -> str | None:
    """Class name of the wrapped discovery failure, for governance payloads."""
    cause = error.__cause__
    return type(cause).__name__ if cause is not None else None


def _attach_mcp_meta(result: RunResult, mcp_meta: dict[str, Any]) -> None:
    result.meta.update(deepcopy(mcp_meta))


def _attach_run_error_meta(exc: Exception, run_id: str, mcp_meta: dict[str, Any]) -> None:
    # Preserve provider exception identity and catch compatibility, including
    # exceptions whose __setattr__ disallows assigning new attributes.
    exc.__dict__.update(run_id=run_id, run_meta=deepcopy(mcp_meta))


_TraceNotifier.__module__ = "dendrux.runtime.runner"
