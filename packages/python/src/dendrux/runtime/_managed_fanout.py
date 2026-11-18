"""Prepare a fan-out, transfer claims to branch owners, and join before fan-in.

Single-child workflows stay inline in ``_managed`` through durable bind,
adoption and continuation. Routing, reports and stream cleanup remain in the
managed helper modules; this driver owns only multi-child claim custody.
"""

from __future__ import annotations

import asyncio
import logging
from contextvars import Context, copy_context
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, cast

from dendrux import _codes
from dendrux._lifecycle import STORAGE_FORMAT_CHECKPOINTED, TERMINAL_STATUSES
from dendrux._managed_wait import MANAGED_FANOUT_SCHEMA_VERSION, MANAGED_WAIT_REASON
from dendrux.errors import (
    ManagedStateError,
    RunCancelledError,
    RunFencedError,
    RunNotFoundError,
    RunNotReclaimableError,
)
from dendrux.loops import _helpers as _hooks
from dendrux.loops._managed_batch import MANAGED_HANDOFF_KEY, ManagedHandoff
from dendrux.loops.base import BaseNotifier, LoopNotifier
from dendrux.runtime import (
    _managed,
    _managed_adoption,
    _managed_budget,
    _managed_cancel,
    _managed_fanout_store,
    _reclaim,
    _run_events,
    _run_execute,
)
from dendrux.runtime import context as _context
from dendrux.runtime._capabilities import capabilities, fanout_support
from dendrux.runtime._fanout_ledger import FanoutLedger
from dendrux.runtime._managed import ManagedInvocation
from dendrux.runtime._managed_claim import scoped_run
from dendrux.runtime._managed_definitions import require_bindable, require_claimable
from dendrux.runtime._managed_route import _Waiting
from dendrux.runtime._managed_tree import BranchSpec, FanoutInvocation, LeafNode, WaitNode
from dendrux.runtime._run_entry import PreparedChild, prepare_child_creation
from dendrux.runtime._run_pending import _build_cached_result
from dendrux.runtime.durability import _fanout_ownership_handoff
from dendrux.runtime.state import (
    ClaimResult,
    LoopStatePatch,
    ManagedHeader,
    StoreCapabilities,
    claimed_store,
)
from dendrux.runtime.tasks import _drain_cleanup
from dendrux.types import (
    ManagedFanout,
    ManagedWait,
    PauseState,
    RunAlreadyActiveError,
    RunEvent,
    RunEventType,
    RunResult,
    RunStatus,
    generate_ulid,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from dendrux.agent import Agent
    from dendrux.loops.base import LoopRecorder
    from dendrux.runtime._managed_fanout_store import BindManagedChildrenResult
    from dendrux.runtime.state import StateStore, TenantScope
    from dendrux.types import ToolCall, ToolResult

    EventSink = Callable[[RunEvent], Awaitable[None]]
    AdoptSettled = Callable[[StateStore, WaitNode], Awaitable[ClaimResult | None]]

logger = logging.getLogger(__name__)
_TERMINAL = frozenset(s.value for s in TERMINAL_STATUSES)


# Exact library-owned names include optional transports without loading their extras.
_INTERNAL_CONTEXT_NAMES = frozenset(
    {
        "entry_guardrail_engine",
        "dendrux_provider_call",
        "dendrux_retry_run_id",
        "dendrux_retry_iteration",
        "dendrux_retry_sinks",
        "dendrux_retry_attempts",
        "dendrux_sdk_request",
        "mcp_tool_http_attempt",
        "mcp_discovery_read",
        "dendrux_blocked_target",
        "dendrux_claim_call",
        "dendrux_blocked_dispatch",
        "dendrux_delegation",
        "dendrux_managed_invocation",
        "dendrux_managed_root_reclaim",
        "dendrux_database_operation",
        "dendrux_database_cancellation",
        "dendrux_live_update_scope",
        "dendrux_sweep_deadlines",
        "dendrux_failed_page_marks",
        "dendrux_handoff_commit",
        "dendrux_creation_loop",
        "dendrux_finalized_run_read",
        "mysql_increment_site",
        "skill_session",
        "dendrux_tool_result_cap",
        "dendrux_tool_image_allowance",
    }
)


def owner_context() -> Context:
    """Copy developer context while clearing every library-owned execution scope.

    Read defaults in an empty context, then set them synchronously in the copy.
    Neither the caller's values nor its tokens move to an owner task. Unset
    variables already have their defaults and require no reset.
    """
    copied, defaults = copy_context(), Context()
    for variable in copied:
        if variable.name in _INTERNAL_CONTEXT_NAMES:
            copied.run(variable.set, defaults.run(variable.get))
    return copied


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
    No provider generator or child session opens before a branch accepts its claim.
    Typed cancel/fence startup refusals leave the 0.7 policy to the driver; a
    post-bind startup error has already drained custody and must still propagate.
    """
    branches = branch_invocations(invocation)
    try:
        if not fanout_support(invocation.store).capable:
            raise ManagedStateError(reason="fanout_capability", run_id=invocation.parent_run_id)
        prepared = await _prepare_batch(invocation.store, branches)
        request = _managed_fanout_store.BindManagedChildrenRequest(
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
            lambda: cast(
                "_managed_fanout_store.ManagedFanoutStore", invocation.store
            ).bind_managed_children(request),
            run_id=invocation.parent_run_id,
        )
    except Exception as error:  # noqa: BLE001, RUF100 - pre-bind isolation preserves the single-child refusal policy.
        if isinstance(error, (RunCancelledError, RunFencedError)):
            raise
        await _managed._after_child_error(
            branches[0],
            error,
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
                    coro, name=f"fanout:{branch.child_run_id}", context=owner_context()
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
                result = await _run_execute._run_once(
                    branch.child,
                    provider=branch.child._require_provider(),
                    user_input=str((prepared.spec.child_input_data or {}).get("input", "")),
                    state_store=branch.store,
                    tenant_id=branch.tenant_id,
                    extra_notifier=_BranchEvents(notifier, event_sink) if event_sink else notifier,
                    _preclaimed=_run_execute._PreclaimedChild(branch, prepared, claim),
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
    adopt_settled: AdoptSettled | None = None,
) -> ClaimResult | None:
    """Read siblings first in a fresh writer transaction, then claim and adopt once.

    The injected T4c entry owns cancellation admission, claim and completions.
    This driver never transforms or writes a report itself.
    """
    # fanout_support validates this optional protocol before the driver binds.
    rows = await cast("_managed_fanout_store.ManagedFanoutStore", store).get_managed_children(
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

    return await (adopt_settled or _managed_adoption.adopt_settled)(store, node)


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
    adopt_settled: AdoptSettled | None = None,
    notifier: LoopNotifier | None = None,
    event_sink: EventSink | None = None,
) -> RunResult:
    """Join the owned branches, adopt through the Protocol, and continue in this task.

    The adopter owns the claim and per-call completions. A not-owner marker lets
    the enclosing trampoline return its fresh root view without executing a child.
    """
    try:
        run = await start_fanout(invocation, notifier=notifier, event_sink=event_sink)
    except (RunCancelledError, RunFencedError) as stop:
        # The policy re-raises if a bind occurred; task-creation failures stay original.
        result, _ = await _managed._after_child_error(branch_invocations(invocation)[0], stop)
        return result
    await run.wait()
    if run.bound.outcome != "created":
        return _managed.not_owner_result(invocation.parent_run_id)
    claim = await settle_parent(
        invocation.store,
        wait_node(invocation),
        adopt_settled=adopt_settled,
    )
    if claim is None:
        return _managed.not_owner_result(invocation.parent_run_id)
    return await continue_parent(
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


async def prepare_handoff(
    store: StateStore,
    run_id: str,
    result: RunResult,
    recorder: LoopRecorder | None,
    provider_kwargs: dict[str, Any] | None,
) -> RunResult | None:
    """Checkpoint all admitted calls and build the v4 invocation in the parent task."""
    if not fanout_support(_managed._shared(store)).capable:
        raise ManagedStateError(reason="fanout_capability", run_id=run_id)
    handoff: ManagedHandoff = result.meta[MANAGED_HANDOFF_KEY]
    branches = _handoff_branches(handoff, run_id)
    if await _managed_cancel.cancellation_requested(store, run_id):
        return None
    header = ManagedHeader.from_dict(
        (getattr(store, "loop_state", None) or {}).get("managed"), run_id=run_id
    )
    ctx = _context.get_delegation_context()
    if ctx is None or ctx.run_id != run_id:
        raise ManagedStateError(reason="header_invalid", run_id=run_id)
    if ctx.max_delegation_depth is not None and ctx.delegation_level + 1 > ctx.max_delegation_depth:
        raise _context.DelegationDepthExceededError(
            ctx.delegation_level + 1, ctx.max_delegation_depth
        )
    pause: PauseState = result.meta["pause_state"]
    pause.history[:] = [await _hooks.project_message(recorder, m) for m in pause.history]
    state = getattr(recorder, "_loop_state", None)
    if not isinstance(state, dict):
        raise ManagedStateError(reason="binding_conflict", run_id=run_id)
    state[_managed.ADMITTED_KEY] = [c.id for c in handoff.calls]
    checkpoint = getattr(recorder, "checkpoint_outcome", None)
    if checkpoint is not None:
        await checkpoint(result, provider_kwargs)
    branches = tuple(replace(b, child_path=(*header.path, b.child.name)) for b in branches)
    wait = _fanout_wait(pause, handoff.calls, branches, state, provider_kwargs)
    invocation = FanoutInvocation(
        store=_managed._shared(store),
        tenant_id=ctx.tenant_id,
        root_run_id=header.root_run_id,
        parent_run_id=run_id,
        parent_epoch=int(getattr(store, "claim_epoch", 0)),
        parent_level=ctx.delegation_level,
        parent_agent=handoff.parent,
        wait=wait,
        iteration=result.iteration_count,
        parent_usage=result.usage,
        max_depth=ctx.max_delegation_depth,
        branches=branches,
        unpriced_accepted=_managed_budget.accepted_count(
            state.get(_managed_budget.UNPRICED_ACCEPTED)
        ),
    )
    return RunResult(
        run_id=run_id,
        status=RunStatus.RUNNING,
        iteration_count=result.iteration_count,
        usage=result.usage,
        meta={_managed.DESCEND_KEY: invocation},
    )


def _handoff_branches(
    handoff: ManagedHandoff,
    run_id: str,
) -> tuple[BranchSpec, ...]:
    branches = []
    for call in handoff.calls:
        child = _managed._registered(handoff.parent, call.name, run_id)
        require_bindable(handoff.parent, child)
        branches.append(
            BranchSpec(
                call,
                _managed_budget.inherit(child, handoff.parent),
                (),
                generate_ulid(),
            )
        )
    return tuple(branches)


async def continue_parent(
    agent: Agent,
    store: StateStore,
    run_id: str,
    claim: ClaimResult,
    *,
    notifier: LoopNotifier | None,
) -> RunResult:
    """Continue the claimed parent in this task with its runtime-computed loop policy."""
    return await _reclaim.continue_claimed(agent, store, run_id, claim, notifier=notifier)


def _fanout_wait(
    pause: PauseState,
    calls: Sequence[ToolCall],
    branches: Sequence[BranchSpec],
    state: dict[str, Any],
    provider_kwargs: dict[str, Any] | None,
) -> PauseState:
    return replace(
        pause,
        pending_tool_calls=list(calls),
        pending_targets={c.id: "agent" for c in calls},
        pause_reason=MANAGED_WAIT_REASON,
        pause_id=generate_ulid(),
        schema_version=MANAGED_FANOUT_SCHEMA_VERSION,
        loop_state_ref=True,
        managed_wait=None,
        managed_fanout=ManagedFanout(
            tuple(ManagedWait(b.call.id, b.child_run_id) for b in branches)
        ),
        approval_gates=None,
        pending_decisions=None,
        clarification=None,
        image_view=None,
        image_view_ref=state.get("image_view_ref"),
        provider_kwargs=dict(provider_kwargs or pause.provider_kwargs or {}),
    )


async def drive_node(
    store: StateStore,
    node: WaitNode | LeafNode,
    *,
    tenant_id: TenantScope,
    fanout_tree: bool,
    notifier: LoopNotifier | None = None,
) -> RunResult:
    """Reconnect a validated tree, joining branch owners before fresh fan-in.

    ``fanout_tree`` is true for every node beneath a workflow containing a v4
    wait. Only there may a live lease or a running SingleCall be left to its
    owner or developer-called sweep. Legacy trees keep their recovery errors.
    The caller owns and joins every task created here, including on cancellation.
    """
    record = await scoped_run(store, node.run_id, tenant_id)
    if record is None:
        raise RunNotFoundError(node.run_id)
    if record.status in _TERMINAL:
        return await _build_cached_result(store, node.run_id)
    if isinstance(node, LeafNode) or record.status != RunStatus.PAUSED.value:
        return await _reconnect_leaf(store, node, record.status, tenant_id, fanout_tree, notifier)
    await _reconnect_children(store, node, fanout_tree=fanout_tree, notifier=notifier)
    if node.version == 4:
        claim = await settle_parent(store, node)
        if claim is None:
            return await _build_cached_result(store, node.run_id)
        result = await continue_parent(node.agent, store, node.run_id, claim, notifier=notifier)
    else:
        child = node.children[0]
        if child.call is None:
            raise ManagedStateError(reason="binding_missing", run_id=node.run_id)
        waiting = _Waiting(
            node.agent, node.run_id, child.agent, child.agent_path, child.call, child.run_id
        )
        result = await _managed.adopt(waiting, store=store, notifier=notifier)
    return await _managed.drive(result, notifier=notifier)


async def _reconnect_leaf(
    store: StateStore,
    node: WaitNode | LeafNode,
    status: str,
    tenant_id: TenantScope,
    fanout_tree: bool,
    notifier: LoopNotifier | None,
) -> RunResult:
    if status != RunStatus.RUNNING.value:
        return await _build_cached_result(store, node.run_id)
    state = await store.get_loop_state(node.run_id)
    if state is None:
        raise RunNotReclaimableError(node.run_id, "recovery_state_missing")
    loop = state.get("loop")
    if fanout_tree and loop == "single_call":
        return await _build_cached_result(store, node.run_id)
    require_claimable(node.run_id, loop)
    try:
        result = await _reclaim.reclaim(
            node.agent,
            store,
            node.run_id,
            tenant_id=tenant_id,
            item=None,
            notifier=notifier,
            via_root=bool(node.agent_path),
        )
    except (RunAlreadyActiveError, RunFencedError, RunNotReclaimableError) as exc:
        if not fanout_tree or (
            isinstance(exc, RunNotReclaimableError)
            and exc.reason not in {"lease_fresh", "not_running", "already_reclaimed"}
        ):
            raise
        if await scoped_run(store, node.run_id, tenant_id) is None:
            raise RunNotFoundError(node.run_id) from None
        return await _build_cached_result(store, node.run_id)
    return await _managed.drive(result, notifier=notifier)


async def _reconnect_children(
    store: StateStore, node: WaitNode, *, fanout_tree: bool, notifier: LoopNotifier | None
) -> None:
    tasks: list[asyncio.Task[RunResult | Exception]] = []
    cancellation = None
    try:
        for child in node.children:
            operation = _reconnect_branch(store, child, node, fanout_tree, notifier)
            try:
                task = asyncio.create_task(operation, context=owner_context())
            except BaseException:
                operation.close()
                raise
            tasks.append(task)
        if tasks:
            await asyncio.wait(tasks)
    except asyncio.CancelledError as exc:
        cancellation = exc
    finally:
        _, cleanup_cancel = await _drain_cleanup(
            _join_reconnect(tasks, cancel=cancellation is not None), run_id=node.run_id
        )
        if cancellation is not None:
            raise cancellation
        if cleanup_cancel is not None:
            raise cleanup_cancel
    _raise_reconnect_error(tasks)


def _raise_reconnect_error(tasks: Sequence[asyncio.Task[RunResult | Exception]]) -> None:
    # A child's own cancellation propagates after the join, even in an uncancelled caller.
    for task in tasks:
        if task.cancelled():
            task.result()
    for task in tasks:
        result = task.result()
        if isinstance(result, Exception):
            raise result


async def _reconnect_branch(
    store: StateStore,
    child: WaitNode | LeafNode,
    parent: WaitNode,
    fanout_tree: bool,
    notifier: LoopNotifier | None,
) -> RunResult | Exception:
    try:
        return await drive_node(
            store, child, tenant_id=parent.tenant_id, fanout_tree=fanout_tree, notifier=notifier
        )
    except Exception as error:  # noqa: BLE001, RUF100 - owner isolation retains failures until every sibling joins.
        _codes.log_code(
            logger,
            _codes.MANAGED_FANOUT_BRANCH_ERROR,
            run_id=child.run_id,
            error_type=type(error).__name__,
        )
        return error


async def _join_reconnect(
    tasks: list[asyncio.Task[RunResult | Exception]], *, cancel: bool
) -> None:
    if cancel:
        for task in tasks:
            if not task.done() and not task.cancelling():
                task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
