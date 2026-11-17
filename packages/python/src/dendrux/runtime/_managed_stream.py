"""Managed subagents: the pieces of the root-facing workflow stream.

``stream`` and ``resume_stream`` on a managed root drive the same transitions as
``run`` while advancing each run's generator in the caller's task (the drive
loop stays in :mod:`dendrux.runtime._managed`). This module holds the stream's
state and legs, the structural view of a subagent's tool events, the terminal
events, the adopting parent's continuation stream and the close-time cleanup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dendrux.runtime import _managed_route
from dendrux.runtime._managed_route import _Waiting  # noqa: TC001 - move-proof binding
from dendrux.types import RunResult, RunStatus

if TYPE_CHECKING:
    from dendrux.agent import Agent


@dataclass
class _StreamState:
    """What the workflow stream's cleanup needs (mutable, one stream)."""

    root_agent: Agent
    store: Any = None
    root_run_id: str | None = None
    tenant_id: Any = None
    current: Any = None
    waiting: list[_Waiting] | None = None
    root_active: bool = True
    """The active leg is the root's own: closing it cancels the root by itself."""
    unstarted: tuple[str, Any] | None = None
    """``(run_id, claim)`` of a claimed continuation whose stream has not started yet."""


def _structural(event: Any, run_id: str | None) -> Any:
    """A tool event with ids, name, iteration, phase and outcome code only."""
    from dendrux.types import RunEvent, ToolCall, ToolResult

    call = event.tool_call
    result = event.tool_result
    code = (result.meta or {}).get("code") if result is not None else None
    return RunEvent(
        type=event.type,
        run_id=run_id,
        tool_name=event.tool_name or (call.name if call is not None else None),
        tool_call_id=event.tool_call_id or (call.id if call is not None else None),
        iteration=event.iteration,
        phase=event.phase,
        tool_call=None
        if call is None
        else ToolCall(name=call.name, params={}, id=call.id, provider_tool_call_id=None),
        tool_result=None
        if result is None
        else ToolResult(
            name=result.name,
            call_id=result.call_id,
            payload="",
            success=result.success,
            duration_ms=result.duration_ms,
            meta={"code": code} if isinstance(code, str) else {},
        ),
    )


def _continuation_stream(waiting: _Waiting, store: Any, claim: Any, notifier: Any) -> Any:
    """The adopting parent's continuation, streamed from its checkpoint under ``claim``."""
    from dendrux.runtime.runner import resume_stream

    parent = waiting.parent_agent
    return resume_stream(
        waiting.parent_run_id,
        agent=parent,
        provider=parent._require_provider(),
        state_store=store,
        extra_notifier=notifier,
        _claimed=claim,
    )


def _event(result: RunResult) -> Any:
    """The workflow stream's terminal event for a root-facing result."""
    from dendrux.types import RunEvent, RunEventType

    kind = {
        RunStatus.SUCCESS: RunEventType.RUN_COMPLETED,
        RunStatus.MAX_ITERATIONS: RunEventType.RUN_COMPLETED,
        RunStatus.ERROR: RunEventType.RUN_ERROR,
        RunStatus.CANCELLED: RunEventType.RUN_CANCELLED,
    }.get(result.status, RunEventType.RUN_PAUSED)
    return RunEvent(type=kind, run_result=result, run_id=result.run_id, error=result.error)


@dataclass(frozen=True)
class _Terminal:
    event: Any


async def _paused_or_failed(
    state: _StreamState, waiting: list[_Waiting], result: RunResult, terminal: Any
) -> RunResult:
    """A leaf that stopped short of terminal: its pause through the root, or a failed attempt.

    A continuation attempt that failed and released its claim reports the
    root-facing state with the attempt's error (the leaf keeps waiting).
    """
    root_run_id = str(state.root_run_id)
    if getattr(getattr(terminal, "type", None), "value", None) == "run_error":
        view = await _managed_route.root_view(state.store, root_run_id)
        view.error = result.error
        view.error_info = result.error_info
        return view
    return await _managed_route._projected_pause(state.store, root_run_id, waiting, result)


async def _stream_failure(state: _StreamState, exc: Exception, run_id: str) -> Any:
    """A failure while driving ends the workflow stream with a root-facing RUN_ERROR."""
    from dendrux.runtime.runner import _safe_error
    from dendrux.types import RunEvent, RunEventType

    error = _safe_error(exc)
    if state.root_run_id is None or state.store is None:
        result = RunResult(run_id=run_id, status=RunStatus.ERROR, error=error)
    else:
        result = await _managed_route.root_view(state.store, state.root_run_id)
        result.error = error
    result.meta["error_type"] = type(exc).__name__
    state.root_active = True  # nothing to cancel: the failure is reported, not abandoned
    return RunEvent(
        type=RunEventType.RUN_ERROR, run_result=result, run_id=result.run_id, error=error
    )


async def _close_workflow(state: _StreamState) -> RunResult | None:
    """Abandoned stream: release a claimed but unstarted continuation, close the active
    run's stream, then cancel a waiting root with its descendants."""
    result = None
    if state.unstarted is not None:
        await _cancel_unstarted(state)
    if state.current is not None:
        await state.current.aclose()
        result = state.current.result
    if not state.root_active and state.root_run_id is not None:
        await state.root_agent.cancel_run(
            state.root_run_id, cascade=True, tenant_id=state.tenant_id
        )
        result = await _managed_route.root_view(state.store, state.root_run_id)
    return result


async def _cancel_unstarted(state: _StreamState) -> None:
    """Finalize a continuation this stream claimed but never started, under its epoch.

    Its own stream never ran, so its ordinary close-time cleanup has nothing to
    finalize; without this the claimed run would stay RUNNING with no owner.
    """
    from dendrux._lifecycle import CancelReason
    from dendrux.runtime.runner import _persist_run_cancellation
    from dendrux.runtime.state import claimed_store

    assert state.unstarted is not None
    run_id, claim = state.unstarted
    state.unstarted = None
    owned = claimed_store(state.store, run_id, claim)
    await _persist_run_cancellation(
        owned,
        run_id,
        {"reason": CancelReason.STREAM_CLOSED.value, "partial_output": False},
        on_won=lambda: None,
        emit_event=True,
    )


def _event_with_error(result: RunResult, error: str | None) -> Any:
    from dendrux.types import RunEvent, RunEventType

    return RunEvent(
        type=RunEventType.RUN_ERROR, run_result=result, run_id=result.run_id, error=error
    )


class _Started:
    """A run's stream whose first event was already taken in this task (its claim made)."""

    def __init__(self, stream: Any, events: Any, head: Any) -> None:
        self._stream = stream
        self._events = events
        self._head = head
        self.run_id = stream.run_id

    @property
    def result(self) -> Any:
        return self._stream.result

    async def __aiter__(self) -> Any:
        yield self._head
        async for event in self._events:
            yield event

    async def aclose(self) -> None:
        await self._events.aclose()  # runs the run's own close-time cleanup


class _Single:
    """A one-event stream: the root-facing state when nothing can be continued."""

    def __init__(self, event: Any) -> None:
        self._event = event
        self.run_id = event.run_id
        self.result = event.run_result

    async def __aiter__(self) -> Any:
        yield self._event

    async def aclose(self) -> None:
        return None


_StreamState.__module__ = "dendrux.runtime._managed"
_Terminal.__module__ = "dendrux.runtime._managed"
_Started.__module__ = "dendrux.runtime._managed"
_Single.__module__ = "dendrux.runtime._managed"
