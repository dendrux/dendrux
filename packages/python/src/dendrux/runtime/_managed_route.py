"""Managed subagents: routing a root-id continuation to the right node.

A root-id control (``submit_*``, ``resume``, ``resume_stream``) reaches the
workflow's active leaf through the persisted bindings: :func:`walk` follows each
wait's exact child id from the root, validating tenant, parent link, header and
path at every hop, and rebuilds the waiting parents (:class:`_Waiting`).
:func:`root_view` and :func:`_projected_pause` are what a caller that does not
own the next step sees. The claim and the driver stay in
:mod:`dendrux.runtime._managed`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, TypeVar

from dendrux import _codes
from dendrux._lifecycle import PAUSE_STATUSES, TERMINAL_STATUS_VALUES
from dendrux.errors import (
    ManagedChildControlError,
    ManagedStateError,
    PauseStatusMismatchError,
    RunNotFoundError,
    RunNotPausedError,
)
from dendrux.runtime import _managed_budget, _managed_cancel, _managed_definitions
from dendrux.runtime._bounds import MAX_PAGE
from dendrux.runtime._entry_checks import entry_checks
from dendrux.runtime._managed_limits import _saved_walk_limit
from dendrux.runtime._managed_pending import (
    fanout_children,
    fanout_wait,
    require_allowance,
    require_fanout,
)
from dendrux.runtime._managed_tree import LeafNode, WaitNode, WorkflowTree
from dendrux.runtime._run_pending import _build_cached_result, attach_pendings
from dendrux.runtime.state import (
    MANAGED_BINDING_KEY,
    MANAGED_HEADER_KEY,
    UNSCOPED,
    ManagedHeader,
    unclaimed_store,
)
from dendrux.runtime.submit import _pause_id_required
from dendrux.types import PauseState, RunResult, RunStatus, ToolCall, UsageStats

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine

    from dendrux.agent import Agent
    from dendrux.runtime._entry_checks import EntryPoint
    from dendrux.runtime._managed_fanout_store import ManagedChildRow
    from dendrux.runtime.state import RunRecord, StateStore, TenantScope

logger = logging.getLogger(__name__)
T = TypeVar("T")

MANAGED_WAIT_REASON = "managed_wait"


@dataclass(frozen=True)
class _Waiting:
    """A parent waiting on one bound child (rebuilt from storage after a restart)."""

    parent_agent: Agent
    parent_run_id: str
    child: Agent
    child_path: tuple[str, ...]
    call: ToolCall
    child_run_id: str
    child_loop: str | None = None  # the bound child's saved loop marker (restart only)

    @property
    def call_id(self) -> str:
        return self.call.id


# ---------------------------------------------------------------------------
# The root-facing view
# ---------------------------------------------------------------------------


async def root_view(store: Any, root_run_id: str) -> RunResult:
    """The root's persisted, root-facing state, for a caller that does not own the next step.

    A root waiting on a subagent reports its active branch: the leaf's external
    pause, or RUNNING while a subagent runs or awaits adoption.
    """
    return await _build_cached_result(store, root_run_id)


def _root_result(record: RunRecord, *, status: RunStatus, **fields: Any) -> RunResult:
    return RunResult(
        run_id=record.id,
        status=status,
        iteration_count=record.iteration_count,
        usage=UsageStats(
            input_tokens=record.total_input_tokens,
            output_tokens=record.total_output_tokens,
            total_tokens=record.total_input_tokens + record.total_output_tokens,
            cost_usd=record.total_cost_usd,
            reasoning_tokens=record.total_reasoning_tokens,
        ),
        **fields,
    )


async def _projected_pause(
    store: Any, root_run_id: str, waiting: list[_Waiting], leaf: RunResult
) -> RunResult:
    """A computed root view of a branch's external pauses; the parents keep waiting.

    ``leaf`` is the result of the run the innermost parent waits on: its own pause, or,
    when it is itself parked on a fan-out, every pause below it. Each is re-addressed
    to the root (``agent_path`` from the root's name, ``origin_run_id`` the paused run).
    """
    record = await store.get_run(root_run_id)
    if record is None:
        raise ManagedStateError(reason="child_missing", run_id=root_run_id)
    above = [record.agent_name, *waiting[-1].child_path[:-1]]
    own = leaf.pendings or ((leaf.pending,) if leaf.pending is not None else ())
    pendings = [
        replace(
            pending,
            run_id=root_run_id,
            agent_name=pending.agent_name if pending.origin_run_id else waiting[-1].child.name,
            agent_path=[*above, *(pending.agent_path or [])]
            if pending.origin_run_id
            else [*above, waiting[-1].child_path[-1]],
            origin_run_id=pending.origin_run_id or leaf.run_id,
        )
        for pending in own
    ]
    view = _root_result(record, status=leaf.status)
    return attach_pendings(view, pendings)


# ---------------------------------------------------------------------------
# Root routing: the persisted bindings, root to leaf (0.7, T3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Route:
    """A root-id control resolved to the workflow's active leaf (in memory).

    ``tenant_id`` is the root's actual tenant; every hop carries it. ``waiting``
    lists the parents from the root down to the leaf's parent; the leaf is the
    root itself when nothing is waiting.
    """

    root_run_id: str
    tenant_id: str | None
    waiting: tuple[_Waiting, ...]
    leaf: RunRecord
    leaf_agent: Agent

    @property
    def routed(self) -> bool:
        """The leaf is a subagent, not the root itself."""
        return bool(self.waiting)


async def walk(root_agent: Agent, store: Any, root_run_id: str) -> tuple[list[_Waiting], RunRecord]:
    """Follow each v3 wait's exact child id, validating every hop; bounded.

    A v4 fan-out wait ends the branch as its leaf (callers read its view); on a store
    without fan-out it is refused (``fanout_capability``) before anything is claimed.
    """
    record = await store.get_run(root_run_id)
    if record is None:
        raise RunNotFoundError(root_run_id)
    waiting: list[_Waiting] = []
    agent: Agent = root_agent
    path: tuple[str, ...] = ()
    limit = await _saved_walk_limit(store, record)
    for index in range(limit):
        wait = await _outgoing(store, record)
        if wait is not None and wait.managed_fanout is not None:
            require_fanout(store, record.id)  # a fan-out ends this branch walk (0.8)
        if wait is None or wait.managed_wait is None:
            return waiting, record
        if index + 1 == limit:
            break
        binding = wait.managed_wait
        child, path, loop = await _bound_child(store, record, binding, root_run_id, path)
        child_agent = _execution(_registered(agent, path[-1], record.id), root_agent)
        intent = await _intent_call(store, record.id, binding.tool_call_id, wait.iteration)
        waiting.append(_Waiting(agent, record.id, child_agent, path, intent, child.id, loop))
        agent, record = child_agent, child
    raise ManagedStateError(reason="binding_conflict", run_id=root_run_id)


async def _outgoing(store: Any, record: RunRecord) -> PauseState | None:
    """The v3 managed wait of a parked parent, else None.

    A pause carrying either managed-wait mark is decoded strictly, so a corrupt
    wait is invalid state, never read as the root's own ordinary pause.
    """
    if record.status != RunStatus.PAUSED.value:
        return None
    raw = await store.get_pause_state(record.id)
    if not raw or (
        raw.get("schema_version") != 3 and raw.get("pause_reason") != MANAGED_WAIT_REASON
    ):
        return None
    return PauseState.from_managed_dict(raw, run_id=record.id)


async def _bound_child(
    store: Any, parent: RunRecord, binding: Any, root_run_id: str, path: tuple[str, ...]
) -> tuple[RunRecord, tuple[str, ...], Any]:
    child, child_path, state = await _bound_child_state(
        store, parent.id, parent.tenant_id, binding, root_run_id, path
    )
    return child, child_path, state.get("loop")


async def _bound_child_state(
    store: Any,
    parent_id: str,
    parent_tenant: str | None,
    binding: Any,
    root_run_id: str,
    path: tuple[str, ...],
) -> tuple[RunRecord, tuple[str, ...], dict[str, Any]]:
    """A v3 wait's child: its record, validated path and checkpoint (two reads)."""
    child = await store.get_run(binding.child_run_id)
    if child is None:
        raise ManagedStateError(reason="child_missing", run_id=parent_id)
    if child.parent_run_id != parent_id or child.tenant_id != parent_tenant:
        raise ManagedStateError(reason="child_mismatch", run_id=parent_id)
    state = await store.get_loop_state(child.id) or {}
    header = ManagedHeader.from_dict(state.get(MANAGED_HEADER_KEY), run_id=child.id)
    if (
        header.root_run_id != root_run_id
        or header.parent_run_id != parent_id
        or header.tool_call_id != binding.tool_call_id
        or header.path[:-1] != path
    ):
        raise ManagedStateError(reason="child_mismatch", run_id=parent_id)
    return child, header.path, state


async def _intent_call(store: Any, parent_run_id: str, call_id: str, iteration: int) -> ToolCall:
    """The bound call as the parent's model made it (its brief lives on the intent row)."""
    return (await _intent_calls(store, parent_run_id, (call_id,), iteration))[call_id]


async def _intent_calls(
    store: Any, parent_run_id: str, call_ids: tuple[str, ...], iteration: int
) -> dict[str, ToolCall]:
    """The bound calls of one wait, by call id, from one paged read of its iteration.

    Read only the wait's own iteration: its rows are one model batch, bounded by
    ``Limits.max_pending_calls``, so the paged read ends; earlier history is never read.
    A bound call without its intent row is ``binding_missing``.
    """
    wanted, found = set(call_ids), {}
    offset = 0
    while True:
        page = await store.get_tool_calls(
            parent_run_id, iteration_index=iteration, limit=MAX_PAGE, offset=offset
        )
        for row in page:
            if row.tool_call_id in wanted and row.target == "agent":
                found[row.tool_call_id] = ToolCall(
                    name=row.tool_name,
                    params=dict(row.params or {}),
                    id=row.tool_call_id,
                    provider_tool_call_id=row.provider_tool_call_id,
                )
        if len(found) == len(wanted):
            return found
        if len(page) < MAX_PAGE:
            raise ManagedStateError(reason="binding_missing", run_id=parent_run_id)
        offset += MAX_PAGE


def _execution(child: Agent, root_agent: Agent) -> Agent:
    """The registered child as one routed control runs it.

    It borrows the root's budget policy and the root's task manager, so the task that
    runs it is registered where ``cancel_run`` on the root signals (as fan-out owners do).
    """
    execution = _managed_budget.inherit(child, root_agent)
    execution._task_manager = root_agent._task_manager
    return execution


def _registered(parent: Agent, name: str, parent_run_id: str) -> Agent:
    """The parent's registered child by name; a missing path refuses before any claim."""
    for child in parent.subagents:
        if child.name == name:
            return child
    raise ManagedStateError(reason="path_missing", run_id=parent_run_id)


# ---------------------------------------------------------------------------
# The workflow tree (0.8): v3 and v4 waits, root to every leaf
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Seen:
    """One run as the walk first read it: a run record (root, v3 child) or a fan-out row."""

    run_id: str
    tenant_id: str | None
    status: str
    loop: str | None
    binding: Any
    record: RunRecord | None = None
    row_pause: tuple[str | None, str | None] = (None, None)
    row_waiting: bool = False

    @classmethod
    def of_record(cls, record: RunRecord, state: dict[str, Any]) -> _Seen:
        return cls(
            record.id,
            record.tenant_id,
            record.status,
            state.get("loop"),
            state.get(MANAGED_BINDING_KEY),
            record=record,
        )

    @classmethod
    def of_row(cls, row: ManagedChildRow) -> _Seen:
        waiting = row.pause_reason == MANAGED_WAIT_REASON or isinstance(row.binding, dict)
        return cls(
            row.run_id,
            row.tenant_id,
            row.status,
            row.loop,
            row.binding,
            row_pause=(row.pause_id, row.pause_reason),
            row_waiting=waiting,
        )


class _StateMemo:
    """Forward the root's saved-cap read and keep the checkpoint it returned."""

    def __init__(self, store: Any) -> None:
        self._store = store
        self.state: dict[str, Any] = {}

    async def get_loop_state(self, run_id: str) -> dict[str, Any]:
        self.state = await self._store.get_loop_state(run_id) or {}
        return self.state


class _TreeWalk:
    """One bounded walk from a root, resolving each run's registered agent.

    Reads: the root's record and checkpoint; per v3 wait its pause, the child's record
    and checkpoint and one intent page; per v4 wait its pause envelope, one
    ``get_managed_children`` and one intent read of its iteration; per paused v3 leaf
    its pause (a fan-out child's token comes with its row).
    """

    def __init__(self, root_agent: Agent, store: Any, root: RunRecord, limit: int) -> None:
        self.root_agent, self.store, self.root, self.limit = root_agent, store, root, limit
        self.fetched = 1  # the root

    async def node(
        self, seen: _Seen, agent: Agent, path: tuple[str, ...], call: ToolCall | None
    ) -> WaitNode | LeafNode:
        raw = await self._pause(seen)
        wait = _managed_wait_of(seen, raw)
        if wait is None:
            return self._leaf(seen, agent, path, call, raw)
        if wait.managed_fanout is not None:
            children = await self._fanout(seen, wait, raw, agent, path)
        else:
            children = await self._single(seen, wait, agent, path)
        return WaitNode(
            run_id=seen.run_id,
            root_run_id=self.root.id,
            tenant_id=self.root.tenant_id,
            agent=agent,
            agent_path=path,
            pause_id=str(wait.pause_id),
            version=4 if wait.managed_fanout is not None else 3,
            iteration=wait.iteration,
            children=children,
            call=call,
        )

    async def _pause(self, seen: _Seen) -> dict[str, Any] | None:
        """The run's pause, read only when it may be a wait or its token is not known."""
        if seen.status not in PAUSE_STATUSES:
            return None
        if seen.record is None and not seen.row_waiting:
            return dict(zip(("pause_id", "pause_reason"), seen.row_pause, strict=True))
        raw = await self.store.get_pause_state(seen.run_id)
        return raw if isinstance(raw, dict) else {}

    def _leaf(
        self,
        seen: _Seen,
        agent: Agent,
        path: tuple[str, ...],
        call: ToolCall | None,
        raw: dict[str, Any] | None,
    ) -> LeafNode:
        pause = raw or {}
        return LeafNode(
            run_id=seen.run_id,
            agent=agent,
            agent_path=path,
            status=seen.status,
            loop=seen.loop,
            pause_id=pause.get("pause_id"),
            pause_reason=pause.get("pause_reason"),
            call=call,
        )

    async def _single(
        self, seen: _Seen, wait: PauseState, agent: Agent, path: tuple[str, ...]
    ) -> tuple[WaitNode | LeafNode, ...]:
        binding = wait.managed_wait
        if binding is None:
            raise ManagedStateError(reason="wait_invalid", run_id=seen.run_id)
        self._admit(1, seen.run_id)
        child, child_path, state = await _bound_child_state(
            self.store, seen.run_id, seen.tenant_id, binding, self.root.id, path
        )
        child_agent = self._agent(agent, child_path, seen.run_id)
        call = await _intent_call(self.store, seen.run_id, binding.tool_call_id, wait.iteration)
        return (await self.node(_Seen.of_record(child, state), child_agent, child_path, call),)

    async def _fanout(
        self,
        seen: _Seen,
        wait: PauseState,
        raw: dict[str, Any] | None,
        agent: Agent,
        path: tuple[str, ...],
    ) -> tuple[WaitNode | LeafNode, ...]:
        fanout = fanout_wait(self.store, seen.run_id, raw, seen.binding)
        self._admit(len(fanout.bindings), seen.run_id)
        rows = await fanout_children(
            self.store, seen.run_id, seen.tenant_id, fanout.bindings, self.root.id, path
        )
        ids = tuple(b.tool_call_id for b in fanout.bindings)
        calls = await _intent_calls(self.store, seen.run_id, ids, wait.iteration)
        children = []
        for (row, child_path), call_id in zip(rows, ids, strict=True):
            child_agent = self._agent(agent, child_path, seen.run_id)
            children.append(
                await self.node(_Seen.of_row(row), child_agent, child_path, calls[call_id])
            )
        return tuple(children)

    def _admit(self, batch: int, run_id: str) -> None:
        """Count ``batch`` runs against the root's saved cap before reading them."""
        require_allowance(batch, self.limit - self.fetched, run_id)
        self.fetched += batch

    def _agent(self, parent: Agent, path: tuple[str, ...], parent_run_id: str) -> Agent:
        return _execution(_registered(parent, path[-1], parent_run_id), self.root_agent)


def _managed_wait_of(seen: _Seen, raw: dict[str, Any] | None) -> PauseState | None:
    """A parked parent's managed wait, decoded strictly; None for any other run.

    A pause carrying either managed-wait mark is a wait, never read as the run's own
    ordinary pause (as :func:`_outgoing`).
    """
    if seen.status != RunStatus.PAUSED.value or not raw:
        return None
    if raw.get("schema_version") not in (3, 4) and raw.get("pause_reason") != MANAGED_WAIT_REASON:
        return None
    return PauseState.from_managed_dict(raw, run_id=seen.run_id)


async def _walk(root_agent: Agent, store: Any, root: RunRecord) -> WorkflowTree:
    memo = _StateMemo(store)
    limit = await _saved_walk_limit(memo, root)
    walker = _TreeWalk(root_agent, store, root, limit)
    node = await walker.node(_Seen.of_record(root, memo.state), root_agent, (), None)
    return WorkflowTree(root.id, root.tenant_id, node)


async def walk_tree(
    root_agent: Agent, store: StateStore, root_run_id: str, *, tenant_id: TenantScope
) -> WorkflowTree:
    """Read the workflow under a root into one validated tree (``WalkTree``).

    A missing or foreign root is not found before anything is read below it; tenant,
    parent link, header and path are checked at every hop. Bounded by the root's saved
    run limit. Read-only.
    """
    record = await _scoped_root(store, root_run_id, tenant_id)
    return await _walk(root_agent, store, record)


async def _scoped_root(store: Any, run_id: str, tenant_id: Any) -> RunRecord:
    record: RunRecord | None = await store.get_run(run_id)
    if record is None or (tenant_id is not UNSCOPED and record.tenant_id != tenant_id):
        raise RunNotFoundError(run_id)
    return record


async def route_tree(root_agent: Agent, store: Any, run_id: str, *, tenant_id: Any) -> WorkflowTree:
    """Resolve a root-id control to its workflow tree (v3 and v4 waits, every leaf).

    The same entry rules as 0.7 routing, before anything below the root is read: a
    missing or foreign run is not found, a managed child id is refused
    (``managed_root_required``) and the root's budget policy is checked.
    """
    record = await _scoped_root(store, run_id, tenant_id)
    if await _managed_cancel.managed_child_header(store, record):
        raise ManagedChildControlError(run_id)
    await _managed_budget.route_checkpoint(root_agent, store, record)
    return await _walk(root_agent, store, record)


async def reaches_fanout(store: Any, found: Route) -> bool:
    """The 0.7 branch walk stopped at a v4 fan-out wait: it is the route's leaf (0.8)."""
    if found.leaf.status != RunStatus.PAUSED.value:
        return False
    wait = await _outgoing(store, found.leaf)
    return wait is not None and wait.managed_fanout is not None


def find_leaf(tree: WorkflowTree, pause_id: str) -> LeafNode:
    """The paused leaf holding ``pause_id`` (``FindLeaf``).

    Raises:
        PauseStatusMismatchError: ``reason="pause_id"`` when no paused leaf holds it;
            ``current_pause_id`` is the first paused leaf's token (stable order), or None.
    """
    paused = tree.paused_leaves()
    for leaf in paused:
        if leaf.pause_id == pause_id:
            return leaf
    first = paused[0] if paused else None
    status = RunStatus(first.status) if first is not None else RunStatus.RUNNING
    raise PauseStatusMismatchError(
        tree.root_run_id,
        status,
        status,
        expected_pause_id=pause_id,
        current_pause_id=first.pause_id if first is not None else None,
        reason="pause_id",
    )


def _required_leaf(tree: WorkflowTree, pause_id: str | None) -> LeafNode:
    """A fan-out continuation names its leaf by token; with none it is refused."""
    if pause_id is not None:
        return find_leaf(tree, pause_id)
    if not tree.paused_leaves():
        raise RunNotPausedError(tree.root_run_id, RunStatus.RUNNING)
    _codes.log_code(logger, _codes.MANAGED_PAUSE_ID_REQUIRED, run_id=tree.root_run_id)
    raise _pause_id_required()


async def submit_routed(
    root_agent: Agent,
    store: Any,
    found: Route,
    operation: EntryPoint,
    expected_pause_id: str | None,
    submit: Callable[[Agent, str, Any], Awaitable[RunResult]],
    notifier: Any,
    **fields: Any,
) -> RunResult:
    """Reconnect a root without external pauses, or continue its token-selected leaf.

    ``found`` is the 0.7 route that stopped at the fan-out (its root already in scope).
    The whole tree is walked from the root again (entry rules first); the leaf is
    validated with its own registered definition (a paused leaf continues as ReAct),
    checked against its own entry rules, claimed and run in the caller's task.
    """
    tree = await route_tree(root_agent, store, found.root_run_id, tenant_id=found.tenant_id)
    return await _submit_or_reconnect(
        root_agent, store, tree, operation, expected_pause_id, submit, notifier, fields
    )


async def _submit_or_reconnect(
    root_agent: Agent,
    store: Any,
    tree: WorkflowTree,
    operation: EntryPoint,
    expected_pause_id: str | None,
    submit: Callable[[Agent, str, Any], Awaitable[RunResult]],
    notifier: Any,
    fields: dict[str, Any],
) -> RunResult:
    # Legacy external pauses still block reconnect when their generation is missing.
    if (
        operation == "resume"
        and expected_pause_id is None
        and all(value is None for value in fields.values())
        and not any(leaf.status in PAUSE_STATUSES for leaf in tree.leaves())
    ):
        return await root_agent.reclaim(
            tree.root_run_id, tenant_id=tree.tenant_id, notifier=notifier
        )
    leaf = _required_leaf(tree, expected_pause_id)
    parent = tree.ancestors(leaf.run_id)[0]
    _managed_definitions._require_supported_loop(parent.agent, leaf.agent, paused=True)
    entry_checks(
        leaf.agent, operation, tree.tenant_id, default_tenant=UNSCOPED, run_id=leaf.run_id, **fields
    )
    result = await submit(leaf.agent, leaf.run_id, _InlineTasks())
    return await unwind_tree(tree, leaf, store, result, notifier=notifier)


class _InlineTasks:
    """Runs a claimed leaf continuation in the caller's task (spec §8.2 step 3).

    Takes the place of the Agent's task manager for one routed submit: ``spawn``
    returns the continuation itself, which the submit then awaits in this task, so the
    leaf's provider calls, tools and ContextVars are the caller's. The runner still
    registers this task as the run's owner, so in-process cancellation reaches it.
    """

    def spawn(self, run_id: str, continuation: Coroutine[Any, Any, T]) -> Coroutine[Any, Any, T]:
        del run_id
        return continuation


async def unwind_tree(
    tree: WorkflowTree, leaf: LeafNode, store: Any, result: RunResult, *, notifier: Any
) -> RunResult:
    """Unwind from a continued leaf toward the root, its immediate parent first.

    While the run just driven is terminal, settle the wait above it: a v3 wait by the
    0.7 adoption, a v4 wait by the fan-in (``settle_parent``: a fresh read of every
    child, then one claim and the completions in call order). A parent that adopts continues in
    this task, delegating again if its model does. Stops at the first wait that is not
    adoptable, is claimed by another caller, or whose run ends paused, and then returns
    a fresh root view (the root's own result when the root itself was continued).
    """
    # cycle: _managed imports this module at load; removed when dispatch is extracted.
    from dendrux.runtime import _managed, _managed_fanout, _reclaim

    shared: Any = unclaimed_store(store)
    result = await _managed.drive(result, notifier=notifier)
    current = leaf.run_id
    for wait in tree.ancestors(leaf.run_id):
        record = await shared.get_run(current)
        if record is None or record.status not in TERMINAL_STATUS_VALUES:
            break
        if wait.version == 3:
            claim = await _managed.claim_adoption(_waiting_of(wait), shared)
        else:
            claim = await _managed_fanout.settle_parent(shared, wait)
        if claim is None:
            break
        resumed = await _reclaim.continue_claimed(
            wait.agent, shared, wait.run_id, claim, notifier=notifier
        )
        result = await _managed.drive(resumed, notifier=notifier)
        current = wait.run_id
    if current == tree.root_run_id:
        return result
    return await root_view(shared, tree.root_run_id)


def _waiting_of(wait: WaitNode) -> _Waiting:
    """A v3 wait as the 0.7 adoption takes it."""
    child = wait.children[0]
    if child.call is None:
        raise ManagedStateError(reason="binding_missing", run_id=wait.run_id)
    return _Waiting(
        wait.agent,
        wait.run_id,
        child.agent,
        child.agent_path,
        child.call,
        child.run_id,
        child.loop if isinstance(child, LeafNode) else None,
    )


_Waiting.__module__ = "dendrux.runtime._managed"
Route.__module__ = "dendrux.runtime._managed"
