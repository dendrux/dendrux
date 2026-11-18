"""Managed subagent batch validation, brief admission and loop handoff."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from dendrux import _codes
from dendrux.loops import _pause_shared
from dendrux.types import RunResult, RunStatus, ToolCall, ToolResult, ToolTarget

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from dendrux.agent import Agent
    from dendrux.loops.react import _RunContext, _ToolBatch, _ToolCallOutcome
    from dendrux.tools._lookups import ToolLookups
    from dendrux.types import Limits


logger = logging.getLogger("dendrux.loops.react")


MANAGED_HANDOFF_KEY = "managed_handoff"
"""``RunResult.meta`` key of an in-memory managed handoff (never persisted)."""


MANAGED_BATCH_UNSUPPORTED = "managed_batch_unsupported"


_MANAGED_BATCH_MESSAGE = (
    "Call one subagent per turn, and not together with client tools. "
    "No tool in this batch ran; call them again in separate turns."
)

_FANOUT_LIMIT_MESSAGE = (
    "Too many subagent calls in one turn (at most {limit}). "
    "No tool in this batch ran; call fewer subagents per turn."
)

_EXCLUSIVE_MESSAGE = (
    "A subagent in this batch must be the only subagent call of its turn. "
    "No tool in this batch ran; call it again on its own."
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


# Preserve the class identity required by the cumulative move manifest.
SubagentTool.__module__ = "dendrux.loops.react"


@dataclass(frozen=True, init=False)
class ManagedHandoff:
    """Admitted managed calls in order; ``call`` is the first call.

    ``child`` is that first call's child, also when two or more calls are admitted.
    """

    call: ToolCall
    child: Agent
    parent: Agent
    calls: tuple[ToolCall, ...]

    def __init__(
        self,
        call: ToolCall | None = None,
        child: Agent | None = None,
        parent: Agent | None = None,
        *,
        calls: Sequence[ToolCall] = (),
    ) -> None:
        ordered = tuple(calls) or ((call,) if call is not None else ())
        if not ordered or child is None or parent is None:
            raise ValueError("A managed handoff needs calls, a child and a parent")
        object.__setattr__(self, "call", ordered[0])
        object.__setattr__(self, "child", child)
        object.__setattr__(self, "parent", parent)
        object.__setattr__(self, "calls", ordered)


# Preserve the class identity required by the cumulative move manifest.
ManagedHandoff.__module__ = "dendrux.loops.react"


def _is_managed(name: str, lookups: ToolLookups) -> bool:
    return isinstance(lookups.fn.get(name), SubagentTool)


@dataclass(frozen=True)
class BatchDecision:
    """The verdict on one emitted batch.

    ``action`` is ``none`` (no managed call), ``single`` (the 0.7 path),
    ``fanout`` (``count`` calls bind together) or ``refuse`` (``reason`` is
    ``client_call``, ``fanout_limit`` or ``exclusive``). ``count`` is the number
    of managed calls in the emitted batch.
    """

    action: str
    count: int = 0
    reason: str | None = None


def decide_batch(
    managed: Sequence[str], *, client_calls: int, exclusive: Collection[str], limit: int
) -> BatchDecision:
    """Apply the batch rules to the emitted batch (before hooks and approvals).

    ``managed`` lists the subagent name of each managed call; ``client_calls``
    counts the non-server, non-managed calls. Server calls never matter. When
    several rules apply the refusal reason is, in order: ``client_call`` (the one
    rule no limit can lift), ``fanout_limit`` (a plain count to fix), then
    ``exclusive`` (only reachable when the limit allows two or more calls).
    """
    count = len(managed)
    if count == 0:
        return BatchDecision("none")
    if client_calls:
        return BatchDecision("refuse", count, "client_call")
    if count > limit:
        return BatchDecision("refuse", count, "fanout_limit")
    if count == 1:
        return BatchDecision("single", 1)
    if any(name in exclusive for name in managed):
        return BatchDecision("refuse", count, "exclusive")
    return BatchDecision("fanout", count)


def effective_limit(limits: Limits, *, fanout_capable: bool) -> int:
    """Subagent calls one turn may make: the configured limit, or 1 without the capability."""
    return limits.max_parallel_subagents if fanout_capable else 1


@dataclass(frozen=True)
class BatchPolicy:
    """What a parent agent contributes to the batch decision."""

    exclusive: frozenset[str] = frozenset()
    limit: int = 1
    fanout_capable: bool = False


DEFAULT_POLICY = BatchPolicy()
"""The 0.7 rules: no exclusive subagents, one managed call per turn."""


def batch_policy(agent: Agent, *, fanout_capable: bool) -> BatchPolicy:
    return BatchPolicy(
        exclusive=agent._exclusive_subagents,
        limit=effective_limit(agent._limits, fanout_capable=fanout_capable),
        fanout_capable=fanout_capable,
    )


def classify_batch(
    calls: Sequence[ToolCall], lookups: ToolLookups, policy: BatchPolicy = DEFAULT_POLICY
) -> BatchDecision:
    """The decision for an emitted batch under ``policy`` (default: the 0.7 rules)."""
    managed = [tc.name for tc in calls if _is_managed(tc.name, lookups)]
    client = sum(
        1
        for tc in calls
        if not _is_managed(tc.name, lookups)
        and lookups.target.get(tc.name, ToolTarget.SERVER) != ToolTarget.SERVER
    )
    return decide_batch(
        managed, client_calls=client, exclusive=policy.exclusive, limit=policy.limit
    )


def _managed_batch_unsupported(calls: Sequence[ToolCall], lookups: ToolLookups) -> bool:
    """True when the 0.7 rules (one managed call per turn) refuse this batch."""
    return classify_batch(calls, lookups).action == "refuse"


def _refusal_message(decision: BatchDecision, limit: int) -> str:
    if decision.reason == "exclusive":
        return _EXCLUSIVE_MESSAGE
    if decision.reason == "fanout_limit" and limit > 1:
        return _FANOUT_LIMIT_MESSAGE.format(limit=limit)
    return _MANAGED_BATCH_MESSAGE


def _managed_batch_result(
    tc: ToolCall, decision: BatchDecision | None = None, *, limit: int = 1
) -> ToolResult:
    """One refused call: ``error`` plus ``reason`` (and counts), mirrored in ``meta``."""
    message = _MANAGED_BATCH_MESSAGE if decision is None else _refusal_message(decision, limit)
    body: dict[str, Any] = {"error": message}
    if decision is not None and decision.reason is not None:
        body["reason"] = decision.reason
        if decision.reason == "fanout_limit":
            body["limit"], body["requested"] = limit, decision.count
    meta = {"code": MANAGED_BATCH_UNSUPPORTED, **{k: v for k, v in body.items() if k != "error"}}
    return ToolResult(
        name=tc.name,
        call_id=tc.id,
        payload=json.dumps(body),
        success=False,
        error=message,
        meta=meta,
    )


async def refuse_managed_batch(
    calls: Sequence[ToolCall], batch: _ToolBatch, decision: BatchDecision, policy: BatchPolicy
) -> None:
    """Answer every call of a refused batch before any hook, approval or effect."""
    _codes.log_code(
        logger,
        _codes.MANAGED_BATCH_UNSUPPORTED,
        run_id=batch.run_id,
        count=len(calls),
        reason=decision.reason,
        limit=policy.limit,
        requested=decision.count,
    )
    if decision.reason == "fanout_limit":
        _codes.log_code(
            logger,
            _codes.MANAGED_FANOUT_LIMIT,
            run_id=batch.run_id,
            requested=decision.count,
            limit=policy.limit,
            store_capability=policy.fanout_capable,
        )
    mark = getattr(batch.recorder, "mark_managed_refusal", None)
    if mark is not None:
        await mark([tc.id for tc in calls])  # durable for the whole batch, before any refusal
    for tc in calls:
        await batch.reject(tc, _managed_batch_result(tc, decision, limit=policy.limit))


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
                    logger,
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
    """Admitted managed calls, handed to the runner instead of a client pause.

    Server calls of the batch have already finished and are durable. The result
    stays RUNNING: only the runner's atomic bind parks this run.
    """
    pending = outcome.pending_calls
    if outcome.pending_approval or not pending:
        return None
    entries = [ctx.lookups.fn.get(call.name) for call in pending]
    if not all(isinstance(entry, SubagentTool) for entry in entries):
        return None
    entry = cast("SubagentTool", entries[0])  # Every entry was narrowed by the batch check.
    result = _pause_shared._pause_result(ctx, RunStatus.RUNNING, iteration, pending_calls=pending)
    result.meta[MANAGED_HANDOFF_KEY] = ManagedHandoff(
        calls=pending, child=entry.agent, parent=ctx.agent
    )
    return result


def loop_fanout_capable(recorder: object) -> bool:
    """Read only this frame's explicit flag; absent flags always disable fan-out."""
    return getattr(recorder, "_managed_fanout_capable", False) is True
