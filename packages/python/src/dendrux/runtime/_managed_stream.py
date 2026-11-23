"""Managed subagents: the pieces of the root-facing workflow stream.

``stream`` and ``resume_stream`` on a managed root drive the same transitions as
``run`` while advancing each run's generator in the caller's task (the drive
loop stays in :mod:`dendrux.runtime._managed`). This module holds the stream's
state and legs, the structural view of a subagent's tool events, the terminal
events, the adopting parent's continuation stream, the consumer of a fan-out's
owner tasks and the close-time cleanup.
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from dendrux._lifecycle import PAUSE_STATUSES, TERMINAL_STATUSES, CancelReason, PauseReason
from dendrux.errors import RunCancelledError, RunFencedError
from dendrux.loops import _helpers as _hooks
from dendrux.loops.base import BaseNotifier
from dendrux.runtime import _managed_definitions, _managed_route
from dendrux.runtime._entry_checks import entry_checks
from dendrux.runtime._managed_route import _Waiting
from dendrux.runtime._managed_tree import FanoutInvocation
from dendrux.runtime._pending import PendingPause, PendingToolCall
from dendrux.runtime._subagents import _tighten
from dendrux.runtime.state import UNSCOPED, claimed_store
from dendrux.runtime.tasks import _drain_cleanup
from dendrux.types import RunEvent, RunEventType, RunResult, RunStatus, ToolCall, ToolResult

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from dendrux.agent import Agent
    from dendrux.runtime._managed_fanout import FanoutRun
    from dendrux.runtime._managed_route import Route
    from dendrux.runtime._managed_tree import WaitNode

QUEUE_EVENTS = 256
"""Bound of one fan-out's stream queue; a full queue blocks the producing owner task."""


class _NoFanout:
    """No fan-out is live: closing has no owner tasks to stop."""

    async def close(self) -> asyncio.CancelledError | None:
        return None


_NO_FANOUT = _NoFanout()


@dataclass
class _StreamState:
    """What the workflow stream's cleanup needs (mutable, one stream)."""

    root_agent: Agent
    store: Any = None
    root_run_id: str | None = None
    tenant_id: Any = None
    current: Any = None
    waiting: list[_Waiting | WaitNode] | None = None
    """Parked parents, root first: a single-child wait, or a fan-out's wait node."""
    root_active: bool = True
    """The active leg is the root's own: closing it cancels the root by itself."""
    unstarted: tuple[str, Any] | None = None
    """``(run_id, claim)`` of a claimed continuation whose stream has not started yet."""
    fanout: FanoutStream | _NoFanout = field(default_factory=lambda: _NO_FANOUT)
    """The fan-out whose owner tasks are live: closing the stream runs its exit path first."""


@dataclass
class _Leg:
    """One run's stream inside the workflow (or one fan-out): its outcome once drained."""

    stream: Any
    invocation: Any = None
    terminal: Any = None
    result: Any = None
    bound: bool = False


def _structural(event: Any, run_id: str | None) -> Any:
    """A tool event with ids, name, iteration, phase and outcome code only."""
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


def _continuation_stream(
    parent: Agent, parent_run_id: str, store: Any, claim: Any, notifier: Any
) -> Any:
    """The adopting parent's continuation, streamed from its checkpoint under ``claim``."""
    from dendrux.runtime.runner import resume_stream

    return resume_stream(
        parent_run_id,
        agent=parent,
        provider=parent._require_provider(),
        state_store=store,
        extra_notifier=notifier,
        _claimed=claim,
    )


def _descend_leg(invocation: Any, state: _StreamState, notifier: Any) -> _Leg:
    """The leg after a handoff: one bound child's own stream, or a fan-out's owner tasks."""
    state.tenant_id = invocation.tenant_id
    if isinstance(invocation, FanoutInvocation):
        return _Leg(FanoutStream(invocation, notifier))
    return _Leg(_child_stream(invocation, notifier), invocation)


def _child_stream(invocation: Any, notifier: Any) -> Any:
    """The bound child's own stream (created by the atomic bind on first step)."""
    from dendrux.runtime.runner import run_stream

    child = invocation.child
    return run_stream(
        child,
        provider=child._require_provider(),
        user_input=str((invocation.call.params or {}).get("task", "")),
        state_store=invocation.store,
        tenant_id=invocation.tenant_id,
        extra_notifier=notifier,
        max_delegation_depth=_tighten(invocation.max_depth, child.max_delegation_depth),
        _managed_invocation=invocation,
    )


def _waiting_continuation(parent: _Waiting, store: Any, claim: Any, notifier: Any) -> Any:
    """A single-child wait's parent continuation, after it adopted its child under ``claim``."""
    return _continuation_stream(parent.parent_agent, parent.parent_run_id, store, claim, notifier)


def _stream_result(result: RunResult) -> RunResult:
    """Keep parallel child execution and display data on the scoped read path."""
    if not any(getattr(p, "_parallel", False) for p in result.pendings):
        return result
    pauses = tuple(
        PendingPause(
            run_id=p.run_id,
            status=p.status,
            kind=p.kind,
            pause_id=p.pause_id,
            origin_run_id=p.origin_run_id,
            reason=p.reason if p.reason in (PauseReason.REQUESTED, PauseReason.BUDGET) else None,
            agent_name=p.agent_name,
            agent_path=p.agent_path,
            tool_calls=[
                PendingToolCall(c.id, "", None, None, args_hidden=True) for c in p.tool_calls
            ],
        )
        for p in result.pendings
    )
    meta = {k: v for k, v in result.meta.items() if k != "pii_mapping"}
    return replace(result, pending=pauses[0], pendings=pauses, meta=meta)


def _event(result: RunResult) -> Any:
    """The workflow stream's terminal event for a root-facing result."""
    kind = {
        RunStatus.SUCCESS: RunEventType.RUN_COMPLETED,
        RunStatus.MAX_ITERATIONS: RunEventType.RUN_COMPLETED,
        RunStatus.ERROR: RunEventType.RUN_ERROR,
        RunStatus.CANCELLED: RunEventType.RUN_CANCELLED,
    }.get(result.status, RunEventType.RUN_PAUSED)
    return RunEvent(
        type=kind, run_result=_stream_result(result), run_id=result.run_id, error=result.error
    )


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


async def _paused_view(
    state: _StreamState, waiting: list[_Waiting | WaitNode], result: RunResult, terminal: Any
) -> RunResult:
    """``_paused_or_failed`` for any chain: under a fan-out wait the view is read again
    from the root, so every sibling's pause is in it (and a failed attempt's error)."""
    chain = [w for w in waiting if isinstance(w, _Waiting)]
    if len(chain) == len(waiting):
        return await _paused_or_failed(state, chain, result, terminal)
    view = await _managed_route.root_view(state.store, str(state.root_run_id))
    if getattr(getattr(terminal, "type", None), "value", None) == "run_error":
        view.error, view.error_info = result.error, result.error_info
    return view


async def _stream_failure(state: _StreamState, exc: Exception, run_id: str) -> Any:
    """A failure while driving ends the workflow stream with a root-facing RUN_ERROR."""
    from dendrux.runtime.runner import _safe_error

    error = _safe_error(exc)
    if state.root_run_id is None or state.store is None:
        result = RunResult(run_id=run_id, status=RunStatus.ERROR, error=error)
    else:
        result = await _managed_route.root_view(state.store, state.root_run_id)
        result.error = error
    result.meta["error_type"] = type(exc).__name__
    state.root_active = True  # nothing to cancel: the failure is reported, not abandoned
    return RunEvent(
        type=RunEventType.RUN_ERROR,
        run_result=_stream_result(result),
        run_id=result.run_id,
        error=error,
    )


async def _close_workflow(state: _StreamState) -> RunResult | None:
    """Abandoned stream: run a live fan-out's exit path, close the active run's stream in
    this task, then finalize the root and its descendants in one retained store-only
    cleanup. A cancellation arriving anywhere on the way is kept; the first one is
    re-raised once the root is final."""
    first = await state.fanout.close()
    closed, late = await _close_current(state)
    result, kept = await _drain_cleanup(_finalize_root(state, closed), run_id=state.root_run_id)
    cancellation = _first_of(first, late, kept)
    if cancellation is not None:
        raise cancellation
    return result


async def _close_current(state: _StreamState) -> tuple[RunResult | None, Any]:
    """Close the active run's stream here (its generator never leaves this task).

    Returns its result and a cancellation that interrupted the close, if any.
    """
    if state.current is None:
        return None, None
    try:
        await state.current.aclose()
    except asyncio.CancelledError as cancellation:
        return state.current.result, cancellation
    return state.current.result, None


async def _finalize_root(state: _StreamState, result: RunResult | None) -> RunResult | None:
    """Store-only: release a claimed but unstarted continuation, then cancel a waiting
    root with its descendants and read its view."""
    if state.unstarted is not None:
        await _cancel_unstarted(state)
    if not state.root_active and state.root_run_id is not None:
        await state.root_agent.cancel_run(
            state.root_run_id, cascade=True, tenant_id=state.tenant_id
        )
        result = await _managed_route.root_view(state.store, state.root_run_id)
    return result


def _first_of(*cancellations: Any) -> Any:
    """The earliest cancellation seen, if any."""
    return next((c for c in cancellations if c is not None), None)


async def _cancel_unstarted(state: _StreamState) -> None:
    """Finalize a continuation this stream claimed but never started, under its epoch.

    Its own stream never ran, so its ordinary close-time cleanup has nothing to
    finalize; without this the claimed run would stay RUNNING with no owner.
    """
    from dendrux.runtime.runner import _persist_run_cancellation

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
    return RunEvent(
        type=RunEventType.RUN_ERROR,
        run_result=_stream_result(result),
        run_id=result.run_id,
        error=error,
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


async def _owned_leaf_stream(
    state: _StreamState,
    root_run_id: str,
    waiting: list[_Waiting | WaitNode],
    first: Any,
    resumed: Any,
) -> Any:
    """Validate and claim the leaf in this task before the workflow exposes anything.

    The leaf stream's first step runs its generation check and claim. Only a
    stream that now owns the leaf arms workflow cancellation and opens with the
    root's RUN_RESUMED; a refused or lost claim ends with a root-facing
    RUN_ERROR and changes nothing.
    """
    events = first.__aiter__()
    head = await anext(events, None)
    if head is None or getattr(head.type, "value", head.type) in _TERMINAL_EVENTS:
        await events.aclose()
        state.waiting = []
        view = await _managed_route.root_view(state.store, root_run_id)
        error = getattr(head, "error", None) if head is not None else None
        view.error = error
        return _Single(_event_with_error(view, error)), None
    started = _Started(first, events, head)
    state.waiting = waiting
    state.current, state.root_active = started, False
    return started, resumed


_TERMINAL_EVENTS = frozenset({"run_completed", "run_paused", "run_error", "run_cancelled"})


async def open_fanout_leaf(
    state: _StreamState,
    root_agent: Agent,
    found: Route,
    *,
    tool_results: Any,
    user_input: Any,
    expected_pause_id: str | None,
    notifier: Any,
    tenant_explicit: bool,
) -> Any:
    """``resume_stream`` on a workflow with a fan-out below: stream the leaf its token names.

    The whole tree is walked from the root again; the leaf meets its own loop and entry
    rules, is claimed by its own stream in this task, and the waits above it (single-child
    or fan-out) unwind leaf-first as each run below them ends.
    """
    from dendrux.runtime.runner import resume_stream

    tree = await _managed_route.route_tree(
        root_agent, state.store, found.root_run_id, tenant_id=found.tenant_id
    )
    leaf = _managed_route._required_leaf(tree, expected_pause_id)
    ancestors = tree.ancestors(leaf.run_id)
    _managed_definitions._require_supported_loop(ancestors[0].agent, leaf.agent, paused=True)
    entry_checks(
        leaf.agent,
        "resume_stream",
        tree.tenant_id,
        default_tenant=UNSCOPED,
        run_id=leaf.run_id,
        tool_results=tool_results,
        user_input=user_input,
    )
    first = resume_stream(
        leaf.run_id,
        agent=leaf.agent,
        provider=leaf.agent._require_provider(),
        state_store=state.store,
        tool_results=tool_results,
        user_input=user_input,
        extra_notifier=notifier,
        expected_pause_id=expected_pause_id,
        _tenant_explicit=tenant_explicit,
    )
    waiting: list[_Waiting | WaitNode] = [
        wait if wait.version == 4 else _managed_route._waiting_of(wait)
        for wait in reversed(ancestors)
    ]
    resumed = RunEvent(type=RunEventType.RUN_RESUMED, run_id=tree.root_run_id)
    return await _owned_leaf_stream(state, tree.root_run_id, waiting, first, resumed)


# ---------------------------------------------------------------------------
# Fan-out: owner tasks produce, the root stream consumes (one bounded queue)
# ---------------------------------------------------------------------------


class FanoutStream:
    """One live fan-out inside the root stream: owner progress, the join, then the fan-in.

    Only an owner task's structured subtree puts on the bounded queue: the owner and the
    parallel-tool tasks it creates, awaits, and cancels and joins with itself. Each put
    waits for room (backpressure) and is followed by ``wake.set()``; each owner's done
    callback sets ``wake``. The consumer, the root stream's generator in the caller's
    task, clears ``wake``, drains the queue, leaves once every owner is done, else waits
    on ``wake``. No task waits on the queue, and no cleanup path touches it: closing
    only stops later puts, and producers blocked on a full queue are cancelled and
    joined with their owner. Every run of a branch reports through one notifier, so
    nested events reach this queue tagged with their own run ids.
    """

    def __init__(self, invocation: FanoutInvocation, notifier: Any) -> None:
        self.invocation = invocation
        self.run_id = invocation.parent_run_id
        self.queue: asyncio.Queue[RunEvent] = asyncio.Queue(maxsize=QUEUE_EVENTS)
        self.wake = asyncio.Event()
        self.result: RunResult | None = None
        self._notifier = notifier
        self._run: FanoutRun | None = None
        self._closed = False

    async def events(self, state: _StreamState) -> AsyncGenerator[RunEvent, None]:
        """Forward owner progress until every owner is done; join; settle on ``state``."""
        # cycle: _managed_fanout imports _managed, which imports this module at load.
        from dendrux.runtime import _managed, _managed_fanout

        state.current, state.root_active = None, False
        run = await self._start()
        if run is None:
            return
        state.fanout = self
        try:
            async with aclosing(self._pending(run)) as pending:
                async for event in pending:
                    yield event
        except BaseException as stop:
            await self._abandon(stop)
            raise
        finally:
            state.fanout = _NO_FANOUT
        await run.aexit()
        while not self.queue.empty():
            yield self.queue.get_nowait()
        run.outcomes()  # the first collected branch error, after every event
        if run.bound.outcome != "created":
            self.result = _managed.not_owner_result(self.run_id)
            return
        state.waiting = [*(state.waiting or ()), _managed_fanout.wait_node(self.invocation)]
        self.result = RunResult(run_id=self.run_id, status=RunStatus.RUNNING)

    async def _pending(self, run: FanoutRun) -> AsyncGenerator[RunEvent, None]:
        """The owner wait: completion is observed from the owner tasks, never set here."""
        while True:
            self.wake.clear()
            while not self.queue.empty():
                yield self.queue.get_nowait()
            if run.owners_done():
                return
            await self.wake.wait()

    async def _start(self) -> FanoutRun | None:
        """Bind and hand every claim to its owner; a refused bind leaves ``result`` set."""
        # cycle: _managed_fanout imports _managed, which imports this module at load.
        from dendrux.runtime import _managed, _managed_fanout

        notifier = _managed_fanout._BranchEvents(_RunEvents(self._notifier, self._put), self._put)
        try:
            run = await _managed_fanout.start_fanout(self.invocation, notifier=notifier)
        except (RunCancelledError, RunFencedError) as stop:
            branch = _managed_fanout.branch_invocations(self.invocation)[0]
            self.result, _ = await _managed._after_child_error(branch, stop)
            return None
        for owner in run.owners:
            owner.add_done_callback(self._owner_done)
        self._run = run
        return run

    def _owner_done(self, _: asyncio.Task[Any]) -> None:
        self.wake.set()

    async def _put(self, event: RunEvent) -> None:
        """Owner subtrees only: wait for room (backpressure), then wake the consumer."""
        if self._closed:
            return
        await self.queue.put(event)
        self.wake.set()

    async def leg_events(self, state: _StreamState, leg: _Leg) -> AsyncGenerator[RunEvent, None]:
        """This fan-out as one leg of the workflow stream: its events, then its outcome."""
        async with aclosing(self.events(state)) as fanned:
            async for event in fanned:
                yield event
        leg.result = self.result

    async def close(self) -> asyncio.CancelledError | None:
        """Stop later puts, then run the driver's exit path (idempotent).

        Returns a cancellation that arrived during the retained cleanup, for the
        caller to re-raise once the rest of the workflow is cancelled.
        """
        self._closed = True
        return await _exit_retaining(self._run)

    async def _abandon(self, stop: BaseException) -> None:
        """The consumer stopped early (``stop`` propagates): run the exit path.

        The first cancellation wins: a later one that lands in the retained cleanup is
        dropped when ``stop`` is itself a cancellation, and replaces any other ``stop``.
        """
        cancellation = await self.close()
        if cancellation is not None and not isinstance(stop, asyncio.CancelledError):
            raise cancellation


async def _exit_retaining(run: FanoutRun | None) -> asyncio.CancelledError | None:
    """Run a fan-out's exit path; return (not raise) a cancellation it retained."""
    if run is None:
        return None
    try:
        await run.aexit()
    except asyncio.CancelledError as cancellation:
        return cancellation
    return None


@dataclass(frozen=True, kw_only=True)
class SubagentPausedEvent(RunEvent):
    """A parallel child's pause signal. Execution arguments are read through RunStore."""

    pause_id: str | None
    call_ids: tuple[str, ...]


class _RunEvents:
    """Forwards every hook; a run that stops at an external pause also tells the stream.

    The dynamic notifier hook boundary is the same forwarding seam as the branch events.
    """

    def __init__(self, inner: Any, sink: Callable[[RunEvent], Awaitable[None]]) -> None:
        self.inner = inner or BaseNotifier()
        self._sink = sink

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    async def on_run_finished(self, run_id: str, result: RunResult) -> None:
        await _hooks.notify_run_finished(self.inner, run_id, result)
        if result.status in PAUSE_STATUSES:
            pending = result.pending
            await self._sink(
                SubagentPausedEvent(
                    type=RunEventType.SUBAGENT_PAUSED,
                    run_id=(pending.origin_run_id if pending else None) or run_id,
                    pause_id=result.pause_id,
                    call_ids=tuple(c.id for c in pending.tool_calls) if pending else (),
                )
            )


async def settle_wait(state: _StreamState, leg: _Leg, wait: WaitNode, notifier: Any) -> Any:
    """The run below the fan-out ``wait`` stopped: settle it, or end with the root view.

    After a live fan-out's join, or once the leaf under the wait is terminal, the
    wait is settled by a fresh read of every child, then one claim and the
    completions in call order. The winner's continuation streams next; otherwise
    the stream ends with the root view (every pause below the root in it).
    """
    # cycle: _managed_fanout imports _managed, which imports this module at load.
    from dendrux.runtime import _managed_fanout

    waiting = state.waiting or []
    joined = isinstance(leg.stream, FanoutStream)
    if not joined and leg.result.status not in TERMINAL_STATUSES:
        return _event(await _paused_view(state, waiting, leg.result, leg.terminal))
    claim = await _managed_fanout.settle_parent(state.store, wait)
    if claim is None:
        return _event(await _managed_route.root_view(state.store, str(state.root_run_id)))
    waiting.pop()
    state.unstarted = (wait.run_id, claim)
    return _Leg(_continuation_stream(wait.agent, wait.run_id, state.store, claim, notifier))


_StreamState.__module__ = "dendrux.runtime._managed"
_Leg.__module__ = "dendrux.runtime._managed"
_Terminal.__module__ = "dendrux.runtime._managed"
_Started.__module__ = "dendrux.runtime._managed"
_Single.__module__ = "dendrux.runtime._managed"
