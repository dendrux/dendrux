"""Managed subagent batch validation, brief admission and loop handoff."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dendrux import _codes
from dendrux.loops import _pause_shared
from dendrux.types import RunResult, RunStatus, ToolCall, ToolResult, ToolTarget

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dendrux.agent import Agent
    from dendrux.loops.react import _RunContext, _ToolBatch, _ToolCallOutcome
    from dendrux.tools._lookups import ToolLookups


MANAGED_HANDOFF_KEY = "managed_handoff"


MANAGED_BATCH_UNSUPPORTED = "managed_batch_unsupported"


_MANAGED_BATCH_MESSAGE = (
    "Call one subagent per turn, and not together with client tools. "
    "No tool in this batch ran; call them again in separate turns."
)


class SubagentTool:
    """The lookup entry of a generated subagent call: a marker, never executed.

    Only these entries make an ``agent``-target call managed; a call is never
    inferred to be managed from its target or name alone.
    """

    __slots__ = ("agent",)

    def __init__(self, agent: Agent) -> None:
        self.agent = agent

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("A subagent call is never executed as a tool")


SubagentTool.__module__ = "dendrux.loops.react"


@dataclass(frozen=True)
class ManagedHandoff:
    """In-memory loop outcome: one admitted subagent call for the runner to bind."""

    call: ToolCall
    child: Agent
    parent: Agent


ManagedHandoff.__module__ = "dendrux.loops.react"


def _is_managed(name: str, lookups: ToolLookups) -> bool:
    return isinstance(lookups.fn.get(name), SubagentTool)


def _managed_batch_unsupported(calls: Sequence[ToolCall], lookups: ToolLookups) -> bool:
    """Two managed calls, or one with another non-server pending target, in one batch."""
    managed = sum(1 for tc in calls if _is_managed(tc.name, lookups))
    if managed == 0:
        return False
    return managed > 1 or any(
        not _is_managed(tc.name, lookups)
        and lookups.target.get(tc.name, ToolTarget.SERVER) != ToolTarget.SERVER
        for tc in calls
    )


def _managed_batch_result(tc: ToolCall) -> ToolResult:
    return ToolResult(
        name=tc.name,
        call_id=tc.id,
        payload=json.dumps({"error": _MANAGED_BATCH_MESSAGE}),
        success=False,
        error=_MANAGED_BATCH_MESSAGE,
        meta={"code": MANAGED_BATCH_UNSUPPORTED},
    )


MANAGED_BRIEF_TOO_LARGE = "managed_brief_too_large"


async def _admit_managed_briefs(
    pending: list[ToolCall], lookups: ToolLookups, batch: _ToolBatch
) -> list[ToolCall]:
    """Refuse a brief over the child's ``Limits.max_input_bytes`` before any bind."""
    kept: list[ToolCall] = []
    for tc in pending:
        entry = lookups.fn.get(tc.name)
        if isinstance(entry, SubagentTool):
            size = len(str((tc.params or {}).get("task", "")).encode("utf-8"))
            limit = entry.agent._limits.max_input_bytes
            if size > limit:
                _codes.log_code(
                    _pause_shared._facade_binding("logger", _pause_shared.logger),
                    _codes.INPUT_SIZE_LIMIT,
                    run_id=batch.run_id,
                    bytes=size,
                    limit=limit,
                )
                await batch.reject(tc, _brief_too_large(tc, size, limit))
                continue
        kept.append(tc)
    return kept


def _brief_too_large(tc: ToolCall, size: int, limit: int) -> ToolResult:
    message = "The task for this subagent is larger than its input limit."
    return ToolResult(
        name=tc.name,
        call_id=tc.id,
        payload=json.dumps({"error": message, "bytes": size, "limit": limit}),
        success=False,
        error=message,
        meta={"code": MANAGED_BRIEF_TOO_LARGE},
    )


def _managed_handoff(
    ctx: _RunContext, outcome: _ToolCallOutcome, iteration: int
) -> RunResult | None:
    """The single admitted managed call, handed to the runner instead of a client pause.

    Server calls of the batch have already finished and are durable. The result
    stays RUNNING: only the runner's atomic bind parks this run.
    """
    pending = outcome.pending_calls
    if outcome.pending_approval or len(pending) != 1:
        return None
    entry = ctx.lookups.fn.get(pending[0].name)
    if not isinstance(entry, SubagentTool):
        return None
    result = _pause_shared._facade_binding("_pause_result", _pause_shared._pause_result)(
        ctx, RunStatus.RUNNING, iteration, pending_calls=pending
    )
    result.meta[MANAGED_HANDOFF_KEY] = ManagedHandoff(
        call=pending[0], child=entry.agent, parent=ctx.agent
    )
    return result
