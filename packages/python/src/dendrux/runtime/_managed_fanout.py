"""Prepare a fan-out, transfer claims to branch owners, and join before fan-in."""

from __future__ import annotations

import asyncio
import logging
from contextvars import Context
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, cast

from dendrux import _codes
from dendrux._lifecycle import STORAGE_FORMAT_CHECKPOINTED, TERMINAL_STATUSES
from dendrux.errors import ManagedStateError, PersistenceError, ProviderError, RunFencedError
from dendrux.loops import _helpers as _hooks
from dendrux.loops.base import BaseNotifier, LoopNotifier
from dendrux.runtime import (
    _managed,
    _managed_budget,
    _managed_cancel,
    _reclaim,
    _run_entry,
    _run_events,
    _run_outcome,
    _run_start,
    _run_tools,
)
from dendrux.runtime import context as _context
from dendrux.runtime._capabilities import capabilities, fanout_support
from dendrux.runtime._fanout_ledger import FanoutLedger
from dendrux.runtime._managed import ManagedInvocation
from dendrux.runtime._managed_fanout_store import BindManagedChildrenRequest, ManagedFanoutStore
from dendrux.runtime._managed_tree import LeafNode, WaitNode
from dendrux.runtime._run_entry import PreparedChild, prepare_child_creation
from dendrux.runtime._subagents import _tighten
from dendrux.runtime.context import (
    DelegationContext,
)
from dendrux.runtime.durability import _fanout_ownership_handoff
from dendrux.runtime.state import (
    ClaimResult,
    LoopStatePatch,
    ManagedHeader,
    StoreCapabilities,
    claimed_store,
)
from dendrux.runtime.tasks import _drain_cleanup
from dendrux.skills._session import SkillSession
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import (
    PauseState,
    RunAlreadyActiveError,
    RunEvent,
    RunEventType,
    RunResult,
    RunStatus,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from dendrux.agent import Agent
    from dendrux.runtime._managed_fanout_store import BindManagedChildrenResult
    from dendrux.runtime._managed_tree import AdoptFanout, FanoutInvocation
    from dendrux.runtime.state import StateStore
    from dendrux.types import ToolCall, ToolResult

    EventSink = Callable[[RunEvent], Awaitable[None]]
    ClaimParent = Callable[[StateStore, WaitNode], Awaitable[ClaimResult | None]]

logger = logging.getLogger(__name__)
_TERMINAL = frozenset(s.value for s in TERMINAL_STATUSES)


@dataclass(frozen=True)
class BranchOutcome:
    """One branch's settled state; errors are retained only until its caller joins."""

    kind: Literal["terminal", "paused", "not_owner", "error"]
    result: RunResult | None = None
    error: Exception | None = None


def branch_invocations(invocation: FanoutInvocation) -> tuple[ManagedInvocation, ...]:
    """Adapt the batch's immutable parent fields to the existing child preparation API."""

    return tuple(
        ManagedInvocation(
            store=invocation.store,
            tenant_id=invocation.tenant_id,
            root_run_id=invocation.root_run_id,
            parent_run_id=invocation.parent_run_id,
            parent_epoch=invocation.parent_epoch,
            parent_level=invocation.parent_level,
            parent_agent=invocation.parent_agent,
            call=b.call,
            child=_execution_child(b.child, invocation.parent_agent),
            child_path=b.child_path,
            wait=invocation.wait,
            iteration=invocation.iteration,
            parent_usage=invocation.parent_usage,
            max_depth=invocation.max_depth,
            child_run_id=b.child_run_id,
            unpriced_accepted=invocation.unpriced_accepted,
        )
        for b in invocation.branches
    )


async def start_fanout(
    invocation: FanoutInvocation,
    *,
    notifier: LoopNotifier | None = None,
    event_sink: EventSink | None = None,
    streaming: bool = False,
) -> FanoutRun:
    """Prepare and bind in the caller, then synchronously create every branch owner.

    The returned handle must be closed with ``aexit`` even if never waited on.
    No provider or execution context is created until a branch accepts its claim.
    """
    if not fanout_support(invocation.store).capable:
        raise ManagedStateError(reason="fanout_capability", run_id=invocation.parent_run_id)
    branches = branch_invocations(invocation)
    try:
        prepared = await _prepare_batch(invocation.store, branches)
        request = BindManagedChildrenRequest(
            tenant_id=invocation.tenant_id,
            root_run_id=invocation.root_run_id,
            parent_run_id=invocation.parent_run_id,
            parent_claim_epoch=invocation.parent_epoch,
            iteration_index=invocation.iteration,
            parent_wait=invocation.wait,
            parent_patch=LoopStatePatch(set_phase="dispatch"),
            children=tuple(p.spec for p in prepared),
            max_parallel=invocation.parent_agent._limits.max_parallel_subagents,
            parent_iteration_count=invocation.iteration,
            parent_total_usage=invocation.parent_usage,
            max_delegation_depth=invocation.max_depth,
        )
        bound = await _fanout_ownership_handoff(
            invocation.store,
            lambda: cast("ManagedFanoutStore", invocation.store).bind_managed_children(request),
            run_id=invocation.parent_run_id,
        )
    except Exception as error:  # noqa: BLE001, RUF100 - pre-bind isolation preserves the single-child refusal policy.
        await _managed._after_child_error(
            branches[0],
            error,
            failure_for_storage=_storage_error(error, invocation.parent_run_id),
        )
        raise
    run = FanoutRun(
        invocation, bound, branches, prepared, notifier, event_sink, streaming=streaming
    )
    try:
        run.start()
    except BaseException as failure:
        try:
            await run.aexit()
        except asyncio.CancelledError:
            if isinstance(failure, asyncio.CancelledError):
                raise failure from None
            raise
        raise
    return run


class FanoutRun:
    """Structured ownership for one live bind; T7 drains ``queue`` using ``wake``.

    ``owners``, ``wake`` and optional bounded ``queue`` are per-call objects. Only
    owners touch child generators or contexts. ``aexit`` closes custody before
    cancelling owners once and retaining one store-only finalization and join.
    """

    def __init__(
        self,
        invocation: FanoutInvocation,
        bound: BindManagedChildrenResult,
        branches: Sequence[ManagedInvocation],
        prepared: Sequence[PreparedChild],
        notifier: LoopNotifier | None,
        event_sink: EventSink | None,
        *,
        streaming: bool = False,
    ) -> None:
        self.invocation = invocation
        self.bound = bound
        self.ledger = FanoutLedger(bound)
        self.owners: list[asyncio.Task[BranchOutcome | None]] = []
        self.wake = asyncio.Event()
        self.queue: asyncio.Queue[RunEvent] | None = (
            asyncio.Queue(maxsize=256) if streaming else None
        )
        self._branches = tuple(branches)
        self._prepared = tuple(prepared)
        self._notifier = notifier
        self._event_sink = self._put if streaming else event_sink
        self._exiting = False
        self._exited = False
        self._exit_finished = asyncio.Event()

    def start(self) -> None:
        """Create owners in call order, without yielding custody to another task."""
        if self.bound.outcome != "created":
            return
        for i, branch in enumerate(self._branches):
            claim = self.bound.children[i].child_claim
            if claim is None:
                raise ManagedStateError(reason="binding_conflict", run_id=branch.child_run_id)
            coro = run_branch(
                self.ledger,
                i,
                branch,
                self._prepared[i],
                claim,
                notifier=self._notifier,
                event_sink=self._event_sink,
            )
            try:
                task = asyncio.create_task(
                    coro, name=f"fanout:{branch.child_run_id}", context=Context()
                )
            except BaseException:
                coro.close()
                raise
            self.owners.append(task)
            task.add_done_callback(lambda _: self.wake.set())

    async def _put(self, event: RunEvent) -> None:
        if self.queue is None:
            raise ManagedStateError(reason="binding_conflict", run_id=self.invocation.parent_run_id)
        await self.queue.put(event)
        self.wake.set()

    def owners_done(self) -> bool:
        """Observe completion synchronously; zero-event branches need no queue waiter."""
        return all(task.done() for task in self.owners)

    async def wait(self) -> tuple[BranchOutcome, ...]:
        """Wait without a queue; join and release custody before exposing outcomes."""
        cancellation = None
        try:
            while True:
                self.wake.clear()
                if self.owners_done():
                    break
                await self.wake.wait()
        except asyncio.CancelledError as error:
            cancellation = error
        finally:
            try:
                await self.aexit()
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
        if cancellation is not None:
            raise cancellation
        return self.outcomes()

    def outcomes(self) -> tuple[BranchOutcome, ...]:
        """Return call-order outcomes after exit, raising the first collected error."""
        if not self._exited:
            raise ManagedStateError(reason="binding_conflict", run_id=self.invocation.parent_run_id)
        outcomes = tuple(task.result() for task in self.owners)
        for outcome in outcomes:
            if outcome is not None and outcome.error is not None:
                raise outcome.error
        return tuple(outcome for outcome in outcomes if outcome is not None)

    async def aexit(self) -> None:
        """Close, cancel once, then finalize and join in one retained store-only cleanup."""
        driver_owned = self.ledger.close()
        if self._exiting:
            await self._join_exit()
            return
        self._exiting = True
        for owner in self.owners:
            if not owner.done():
                owner.cancel()
        _, cancellation = await _drain_cleanup(
            self._cleanup(driver_owned), run_id=self.invocation.parent_run_id
        )
        if cancellation is not None:
            raise cancellation

    async def _cleanup(self, driver_owned: Sequence[int]) -> None:
        try:
            try:
                await self.ledger.finalize(self.invocation.store, driver_owned)
            finally:
                if self.owners:
                    await asyncio.wait(self.owners)
                    for owner in self.owners:
                        if not owner.cancelled():
                            owner.exception()
        finally:
            self._exited = True
            self._exit_finished.set()

    async def _join_exit(self) -> None:
        cancellation = None
        while not self._exit_finished.is_set():
            try:
                await self._exit_finished.wait()
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
        if cancellation is not None:
            raise cancellation


async def run_branch(
    ledger: FanoutLedger,
    i: int,
    branch: ManagedInvocation,
    prepared: PreparedChild,
    claim: ClaimResult,
    *,
    notifier: LoopNotifier | None,
    event_sink: EventSink | None = None,
) -> BranchOutcome | None:
    """Own one claim and every nested run it drives, from first step to cleanup."""
    if not ledger.accept(i):
        return None
    cancelled_in_runner = False
    try:
        try:
            try:
                result = await _run_prepared(
                    branch, prepared, claim, notifier=notifier, event_sink=event_sink
                )
            except asyncio.CancelledError:
                cancelled_in_runner = True
                raise
            result = await _managed.drive(result, notifier=notifier)
            if result.meta.get(_managed.NOT_OWNER_KEY) or result.status == RunStatus.RUNNING:
                return BranchOutcome("not_owner", result)
            return BranchOutcome(
                "terminal" if result.status in TERMINAL_STATUSES else "paused", result
            )
        except (RunFencedError, RunAlreadyActiveError):
            return BranchOutcome("not_owner")
        except Exception as error:  # noqa: BLE001, RUF100 - branch isolation collects failures without cancelling siblings.
            return await _branch_error(branch, error)
    except asyncio.CancelledError:
        if not cancelled_in_runner:
            await _run_events._drain_task_cancellation(
                claimed_store(branch.store, branch.child_run_id, claim), branch.child_run_id, False
            )
        raise


async def _branch_error(branch: ManagedInvocation, error: Exception) -> BranchOutcome:
    _codes.log_code(
        logger,
        _codes.MANAGED_FANOUT_BRANCH_ERROR,
        run_id=branch.child_run_id,
        error_type=type(error).__name__,
    )
    try:
        record = await branch.store.get_run(branch.child_run_id)
        if record is not None and record.status in _TERMINAL:
            return BranchOutcome(
                "terminal", await _managed._terminal_child(branch.store, record.id)
            )
    except Exception as diagnostic:  # noqa: BLE001, RUF100 - diagnostic isolation preserves the original branch error.
        _codes.log_code(
            logger,
            _codes.MANAGED_FANOUT_BRANCH_ERROR,
            run_id=branch.child_run_id,
            error_type=type(diagnostic).__name__,
        )
    return BranchOutcome("error", error=error)


def wait_node(invocation: FanoutInvocation) -> WaitNode:
    """The live wait's call-order bindings, never checkpoint mapping iteration order."""
    wait = PauseState.from_managed_dict(invocation.wait.to_dict(), run_id=invocation.parent_run_id)
    fanout = wait.managed_fanout
    if fanout is None or wait.pause_id is None:
        raise ManagedStateError(reason="wait_invalid", run_id=invocation.parent_run_id)
    branches = {(b.call.id, b.child_run_id): b for b in invocation.branches}
    ordered = [(b.tool_call_id, b.child_run_id) for b in fanout.bindings]
    if set(ordered) != set(branches):
        raise ManagedStateError(reason="binding_conflict", run_id=invocation.parent_run_id)
    return WaitNode(
        invocation.parent_run_id,
        invocation.root_run_id,
        invocation.tenant_id,
        invocation.parent_agent,
        invocation.branches[0].child_path[:-1],
        wait.pause_id,
        4,
        invocation.iteration,
        tuple(
            LeafNode(b.child_run_id, b.child, b.child_path, RunStatus.RUNNING.value, call=b.call)
            for b in (branches[key] for key in ordered)
        ),
    )


async def settle_parent(
    store: StateStore,
    node: WaitNode,
    *,
    claim_parent: ClaimParent,
    adopt_fanout: AdoptFanout,
) -> ClaimResult | None:
    """Read siblings first in a fresh writer transaction, then claim and adopt once.

    ``claim_parent`` supplies T4c's generation CAS. ``adopt_fanout`` is only the
    T0b Protocol; this driver never transforms or writes a report itself.
    """
    # fanout_support validates this optional protocol before the driver binds.
    rows = await cast("ManagedFanoutStore", store).get_managed_children(
        node.run_id, [b.child_run_id for b in node.bindings], tenant_id=node.tenant_id
    )
    if len(rows) != len(node.bindings) or {r.run_id for r in rows} != {
        b.child_run_id for b in node.bindings
    }:
        raise ManagedStateError(reason="child_missing", run_id=node.run_id)
    children = {child.run_id: child for child in node.children}
    for row in rows:
        header = ManagedHeader.from_dict(row.header, run_id=row.run_id)
        call = next(b.tool_call_id for b in node.bindings if b.child_run_id == row.run_id)
        if (
            row.parent_run_id != node.run_id
            or header.parent_run_id != node.run_id
            or header.path != children[row.run_id].agent_path
            or row.tenant_id != node.tenant_id
            or header.root_run_id != node.root_run_id
            or header.tool_call_id != call
        ):
            raise ManagedStateError(reason="child_mismatch", run_id=node.run_id)
    if any(r.status not in _TERMINAL for r in rows):
        return None

    if await _managed_cancel.cancellation_requested(store, node.run_id):
        return None
    claim = await claim_parent(store, node)
    return None if claim is None else await adopt_fanout(store, node, claim)


async def _run_prepared(
    branch: ManagedInvocation,
    prepared: PreparedChild,
    claim: ClaimResult,
    *,
    notifier: LoopNotifier | None,
    event_sink: EventSink | None = None,
) -> RunResult:

    lifecycle = _run_start._RunLifecycle(
        branch.child_run_id,
        claimed_store(branch.store, branch.child_run_id, claim),
        getattr(branch.child, "_task_manager", None),
        owned=True,
    )
    if event_sink is not None:
        notifier = _BranchEvents(notifier, event_sink)
    if lifecycle.manager is not None:
        lifecycle.manager.register_current(branch.child_run_id)
        lifecycle.registered = True
    try:
        try:
            return await _execute_prepared(lifecycle, branch, prepared, notifier, event_sink)
        except (_managed_cancel.ManagedRootStoppedError, RunFencedError) as stop:
            result = await lifecycle._stopped(stop, lifecycle.recorder, notifier)
            if isinstance(stop, RunFencedError):
                result.meta[_managed.NOT_OWNER_KEY] = True
            return result
        except Exception as error:  # noqa: BLE001, RUF100 - the owned run lifecycle persists failure before branch isolation.
            await lifecycle._failed(
                _storage_error(error, branch.child_run_id), {}, lifecycle.recorder, notifier
            )
            raise
    except asyncio.CancelledError:
        cancelled, swallow = await lifecycle._cancelled(None, {}, lifecycle.recorder, notifier)
        if swallow:
            return cancelled
        raise
    finally:
        lifecycle._release()


async def _execute_prepared(
    lifecycle: _run_start._RunLifecycle,
    branch: ManagedInvocation,
    prepared: PreparedChild,
    notifier: LoopNotifier | None,
    event_sink: EventSink | None,
) -> RunResult:

    child, run_id = branch.child, branch.child_run_id
    provider = child._require_provider()
    loop = _run_entry._new_run_loop(child, None)
    session = SkillSession.prepare(child, loop=loop)
    if lifecycle.store is None:
        raise ManagedStateError(reason="binding_conflict", run_id=run_id)
    with session.bind():
        await _run_entry._verify_projected_creation(
            prepared.projection.error is not None, lifecycle.store, run_id
        )
        link, notifier = await _run_start._start_new_run(
            lifecycle,
            child,
            provider,
            branch.tenant_id,
            {"seed_traces": prepared.spec.child_seed_traces},
            prepared.projection.engine,
            notifier,
            None,
            {"agent_name": child.name},
        )
        token = _context.set_delegation_context(_owner_context(branch, lifecycle.store, link))
        try:
            mcp_meta: dict[str, object] = {}
            await _run_tools._discover_runtime_tools(
                child, lifecycle.recorder, notifier, run_id, loop, mcp_meta
            )
            result = await _run_entry._run_projected_loop(
                prepared.projection,
                loop,
                None,
                agent=child,
                provider=provider,
                strategy=NativeToolCalling(),
                user_input=str((prepared.spec.child_input_data or {}).get("input", "")),
                run_id=run_id,
                recorder=lifecycle.recorder,
                notifier=notifier,
                initial_history=None,
                provider_kwargs=None,
                output_type=child.output_type,
                state_store=lifecycle.store,
            )
            result = await _run_outcome._persist_loop_outcome(
                state_store=lifecycle.store,
                run_id=run_id,
                result=result,
                recorder=lifecycle.recorder,
            )
            await lifecycle._finish(lifecycle.recorder, notifier, result)
            return result
        finally:
            _context.reset_delegation_context(token)


def _owner_context(
    branch: ManagedInvocation, store: StateStore, link: str | None
) -> DelegationContext:

    return DelegationContext(
        run_id=branch.child_run_id,
        trace_link=link,
        delegation_level=branch.parent_level + 1,
        persisted=True,
        store_identity=_context.get_store_identity(store),
        max_delegation_depth=_tighten(branch.max_depth, branch.child.max_delegation_depth),
        tenant_id=branch.tenant_id,
        tenant_explicit=True,
        state_store=store,
        lease_interval_s=branch.child.lease_interval.total_seconds(),
        loop=asyncio.get_running_loop(),
    )


class _PreparationStore:
    # Dynamic store adapter: plugins expose optional preparation capabilities via getattr.
    # The real store remains shared; only its format read is cached for this batch.
    def __init__(self, store: StateStore, storage_format: int) -> None:
        self._store = store
        self._format = storage_format

    @property
    def capabilities(self) -> StoreCapabilities:
        return capabilities(self._store)

    async def _creation_storage_format(self) -> int:
        return self._format

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)


async def _prepare_batch(
    store: StateStore, branches: Sequence[ManagedInvocation]
) -> tuple[PreparedChild, ...]:
    reader = getattr(store, "_creation_storage_format", None)
    storage_format = await reader() if callable(reader) else STORAGE_FORMAT_CHECKPOINTED
    if storage_format != STORAGE_FORMAT_CHECKPOINTED:
        raise ManagedStateError(reason="storage_format", run_id=branches[0].parent_run_id)
    # The adapter forwards every store method; preparation only overrides the format read.
    preparation_store = cast("StateStore", _PreparationStore(store, storage_format))
    return tuple(
        [
            await prepare_child_creation(b.child, replace(b, store=preparation_store))
            for b in branches
        ]
    )


class _BranchEvents:
    # The dynamic notifier hook boundary is the same forwarding seam as _TraceNotifier.
    def __init__(self, inner: LoopNotifier | None, sink: EventSink) -> None:
        self.inner = inner or BaseNotifier()
        self._sink = sink

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    async def on_tool_started(self, run_id: str, call: ToolCall, iteration: int) -> None:
        await _hooks.notify_tool_started(self.inner, run_id, call, iteration)
        await self._sink(
            RunEvent(
                type=RunEventType.TOOL_USE_START,
                run_id=run_id,
                tool_name=call.name,
                tool_call_id=call.id,
                iteration=iteration,
            )
        )
        await self._sink(
            _managed._structural(
                RunEvent(
                    type=RunEventType.TOOL_USE_END,
                    tool_call=call,
                    iteration=iteration,
                ),
                run_id,
            )
        )

    async def on_tool_completed(
        self, run_id: str, call: ToolCall, result: ToolResult, iteration: int
    ) -> None:
        await _hooks.notify_tool(self.inner, run_id, call, result, iteration)
        await self._sink(
            _managed._structural(
                RunEvent(
                    type=RunEventType.TOOL_RESULT,
                    tool_call=call,
                    tool_result=result,
                    iteration=iteration,
                ),
                run_id,
            )
        )


async def drive_fanout(
    invocation: FanoutInvocation,
    *,
    claim_parent: ClaimParent,
    adopt_fanout: AdoptFanout,
    notifier: LoopNotifier | None = None,
    event_sink: EventSink | None = None,
) -> RunResult:
    """Join the owned branches, adopt through the Protocol, and continue in this task.

    Claim and completion are injected until T4c merges. A not-owner marker lets
    the enclosing trampoline return its fresh root view without executing a child.
    """
    run = await start_fanout(invocation, notifier=notifier, event_sink=event_sink)
    await run.wait()
    if run.bound.outcome != "created":
        return _managed.not_owner_result(invocation.parent_run_id)
    claim = await settle_parent(
        invocation.store,
        wait_node(invocation),
        claim_parent=claim_parent,
        adopt_fanout=adopt_fanout,
    )
    if claim is None:
        return _managed.not_owner_result(invocation.parent_run_id)
    return await _reclaim.continue_claimed(
        invocation.parent_agent,
        invocation.store,
        invocation.parent_run_id,
        claim,
        notifier=notifier,
    )


def _execution_child(child: Agent, parent: Agent) -> Agent:
    execution = _managed_budget.inherit(child, parent)
    execution._task_manager = parent._task_manager
    return execution


def _storage_error(error: Exception, run_id: str) -> Exception:
    if isinstance(error, (ProviderError, PersistenceError, ManagedStateError)):
        return error
    safe = ManagedStateError(reason="fanout_invalid", run_id=run_id)
    safe.__cause__ = error
    return safe
