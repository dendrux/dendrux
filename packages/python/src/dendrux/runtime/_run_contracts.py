"""Define invocation contracts shared by runner entry modes."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from dendrux._sentinel import _UnsetType
from dendrux.errors import StoreOutcomeUnknown

if TYPE_CHECKING:
    from dendrux.loops.base import Loop, LoopNotifier, LoopRecorder
    from dendrux.runtime.state import RunRecord, TraceRecord
    from dendrux.strategies.base import Strategy
    from dendrux.types import Message, PauseState, RunStatus, ToolCall, ToolResult


def _signature_params(fn: Any) -> Any:
    try:
        return inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return None


def _accepts_kwarg(fn: Any, name: str) -> bool:
    """Whether ``fn`` takes keyword ``name`` (explicitly or via ``**kwargs``)."""
    params = _signature_params(fn)
    if params is None:
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


_UNSET_DEPTH = _UnsetType()


class _ResumeContext:
    """Prepared state for re-entering the loop after a resume.

    Built by _prepare_resume(), consumed by both _resume_core() (batch)
    and resume_stream() (streaming). Does NOT own claiming — that happens
    before prepare is called.
    """

    __slots__ = (
        "history",
        "recorder",
        "notifier",
        "pause_state",
        "resolved_loop",
        "resolved_strategy",
        "recorder_started",
        "notifier_started",
        "notifier_warnings",
    )

    def __init__(
        self,
        *,
        history: list[Message],
        recorder: LoopRecorder,
        notifier: LoopNotifier | None,
        pause_state: PauseState,
        resolved_loop: Loop,
        resolved_strategy: Strategy,
    ) -> None:
        self.history = history
        self.recorder = recorder
        self.notifier = notifier
        self.pause_state = pause_state
        self.resolved_loop = resolved_loop
        self.resolved_strategy = resolved_strategy
        self.recorder_started = False
        self.notifier_started = False
        self.notifier_warnings: list[str] = []

    def _mark_notifier_started(self) -> None:
        self.notifier_started = True


@dataclass
class _PauseReads:
    record: RunRecord | None
    checkpoint: dict[str, Any] | None = None
    history_records: list[TraceRecord] | None = None


class _ApprovalResume(NamedTuple):
    """Outcome of executing an approval batch on resume."""

    # (ToolCall, ToolResult) pairs appended to history, in pending order.
    results: list[tuple[ToolCall, ToolResult]]
    # Client-target calls held in the batch; they never run on the server.
    client_pending: list[ToolCall]
    # ``approval.decided`` payload (scalar decision + per-call outcomes).
    decided: dict[str, Any]
    # Set when a resume barrier stopped execution before a side effect.
    barrier: RunStatus | None
    # Calls not yet resolved when the barrier hit (client calls excluded),
    # as (call, outcome, denial reason) in pending order.
    remaining: tuple[tuple[ToolCall, str, str | None], ...] = ()
    # The admitted generated subagent call, handed to the bind (0.7).
    handoff: Any = None


@dataclass
class _ResumeProgress:
    submission: Any = None
    started: bool = False

    def _start(self) -> None:
        self.started = True

    def _can_release(self, ctx: _ResumeContext | None, exc: Exception) -> bool:
        if self.started or isinstance(exc, StoreOutcomeUnknown):
            return False
        return self.submission is None or (
            ctx is not None
            and getattr(ctx.recorder, "_loop_state", {}).get("submission") is not None
        )


_ResumeContext.__module__ = "dendrux.runtime.runner"
_PauseReads.__module__ = "dendrux.runtime.runner"
_ApprovalResume.__module__ = "dendrux.runtime.runner"
_ResumeProgress.__module__ = "dendrux.runtime.runner"
