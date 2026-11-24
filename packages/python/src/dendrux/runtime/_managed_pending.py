"""The shape of a managed workflow as a read-only reader sees it (0.7 chains, 0.8 fan-out).

A parent waiting on subagents is PAUSED with a private managed wait; what an application sees
is every external pause below it, addressed through the run it asked about. This module reads
that shape without history and without locks:

- a v3 wait (one child): one projected primary-key read of the bound child (0.7);
- a v4 wait (2..32 children): its pause envelope, decoded strictly and checked against the
  checkpoint binding, then one ``get_managed_children`` read covering all its bindings.

Every hop is validated (parent link, tenant, root, invocation, path) before anything below it
is read, and the walk is bounded. The per-leaf pending projection belongs to the caller
(:mod:`dendrux.runtime._pending`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from sqlalchemy import Integer, String, bindparam, column, func, literal, select, true, union_all
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.exc import DBAPIError

from dendrux import _codes
from dendrux._lifecycle import (
    PAUSE_STATUSES,
    PENDING_KIND_BY_STATUS,
    GovernanceEventType,
    StoredEventType,
)
from dendrux._managed_wait import ManagedFanout, decode_managed_binding
from dendrux.db.models import AgentRun, RunEvent
from dendrux.errors import ManagedStateError, RunNotFoundError
from dendrux.runtime._bounds import MAX_PAGE
from dendrux.runtime._capabilities import fanout_support
from dendrux.runtime.durability import _safe_database_call
from dendrux.runtime.state import (
    PAUSE_SUMMARY_KEYS,
    UNSCOPED,
    ManagedHeader,
    PauseSummary,
    _apply_tenant_scope,
    unclaimed_store,
)
from dendrux.types import _ULID_PATTERN, RunStatus

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from dendrux._managed_wait import ManagedWait
    from dendrux.runtime._managed_fanout_store import ManagedChildRow
    from dendrux.runtime._pending import PendingPause
    from dendrux.runtime.state import TenantScope

    ReadHop = Callable[[str, TenantScope], Awaitable["ManagedHop | None"]]

MANAGED_WAIT_REASON = "managed_wait"
MAX_MANAGED_WALK = 257
"""Bound on the runs one walk reads: the definition-node cap plus the root."""


@dataclass(frozen=True)
class ManagedHop:
    """One run on a managed branch: projected columns and checkpoint keys, never history."""

    run_id: str
    status: str
    tenant_id: str | None
    parent_run_id: str | None
    agent_name: str
    header: Any
    binding: Any

    @property
    def waiting(self) -> bool:
        """A parent parked on its bound subagents (a v3 or a v4 wait)."""
        return self.status == RunStatus.PAUSED.value and isinstance(self.binding, dict)

    @property
    def fanout(self) -> bool:
        """A parent parked on a v4 fan-out wait (its binding maps several calls)."""
        return self.waiting and "bindings" in self.binding


@dataclass(frozen=True)
class FacingView:
    """What a run looks like to its caller: root-facing for a managed workflow.

    ``pendings`` lists every external pause the run or its workflow waits on, in stable
    order (depth-first, siblings in call order); ``pending`` is the first of them.
    """

    status: str
    pending: PendingPause | None
    managed: bool = False
    pendings: tuple[PendingPause, ...] = ()


def facing_status(status: str, view: FacingView | None) -> str:
    """The status a caller sees: the stored one, unless this read classified a managed view.

    A managed view (a root waiting on subagents) carries the status the same reads
    observed: the first leaf's waiting status, or RUNNING while subagents run or
    await adoption. Nothing is inferred from a missing pending: an ordinary run
    keeps its stored status even when it moved on before its pending read.
    """
    if view is not None and view.managed:
        return view.status
    return status


def is_managed_wait(status: str, pause: Mapping[str, Any]) -> bool:
    """A parent waiting on managed subagents (its own pause is private)."""
    return status == RunStatus.PAUSED.value and pause.get("pause_reason") == MANAGED_WAIT_REASON


def _bound_child_id(parent: ManagedHop) -> str:
    child_id = parent.binding.get("child_run_id")
    call_id = parent.binding.get("tool_call_id")
    if (
        not isinstance(child_id, str)
        or not _ULID_PATTERN.fullmatch(child_id)
        or not isinstance(call_id, str)
    ):
        raise ManagedStateError(reason="wait_invalid", run_id=parent.run_id)
    return child_id


def _checked_child(
    parent: ManagedHop, child: ManagedHop | None, root_run_id: str, path: tuple[str, ...]
) -> tuple[ManagedHop, tuple[str, ...]]:
    """Validate one v3 hop (parent link, tenant, root, invocation, path): ``(child, path)``."""
    if child is None:
        raise ManagedStateError(reason="child_missing", run_id=parent.run_id)
    call_id = parent.binding.get("tool_call_id")
    return child, checked_child(parent.run_id, parent.tenant_id, root_run_id, path, call_id, child)


class BoundRun(Protocol):
    """The identity columns of a bound child, from any projection."""

    @property
    def run_id(self) -> str: ...
    @property
    def parent_run_id(self) -> str | None: ...
    @property
    def tenant_id(self) -> str | None: ...
    @property
    def header(self) -> Any: ...


def checked_child(
    parent_run_id: str,
    parent_tenant: str | None,
    root_run_id: str,
    path: tuple[str, ...],
    call_id: object,
    child: BoundRun,
) -> tuple[str, ...]:
    """Validate a bound child against its parent's binding and return the child's path.

    Raises:
        ManagedStateError: ``child_mismatch`` for another parent, tenant, root, call or
            path; ``header_invalid`` for a missing or malformed header.
    """
    if child.parent_run_id != parent_run_id or child.tenant_id != parent_tenant:
        raise ManagedStateError(reason="child_mismatch", run_id=parent_run_id)
    header = ManagedHeader.from_dict(child.header, run_id=child.run_id)
    if (
        header.root_run_id != root_run_id
        or header.parent_run_id != parent_run_id
        or header.tool_call_id != call_id
        or header.path[:-1] != path
    ):
        raise ManagedStateError(reason="child_mismatch", run_id=parent_run_id)
    return header.path


# ---------------------------------------------------------------------------
# 0.8: v4 fan-out waits
# ---------------------------------------------------------------------------


def require_allowance(batch: int, allowance: int, run_id: str) -> None:
    """Refuse a walk's next read of ``batch`` runs when only ``allowance`` remain."""
    if batch > allowance:
        raise ManagedStateError(reason="binding_conflict", run_id=run_id)


def require_fanout(state: Any, run_id: str) -> None:
    """Refuse v4 fan-out data on a store that does not declare fan-out (``fanout_capability``).

    Called before anything below the wait is read or claimed.
    """
    if not fanout_support(unclaimed_store(state)).capable:
        raise ManagedStateError(reason="fanout_capability", run_id=run_id)


def fanout_wait(state: Any, run_id: str, raw: object, binding: object) -> ManagedFanout:
    """The ordered bindings of a v4 wait, checked against the parent's checkpoint record.

    Refuses before any child is read: ``fanout_capability`` on a store that does not
    declare fan-out, ``wait_invalid`` for an envelope the strict decoder refuses, and
    ``fanout_invalid`` when the pause envelope and the checkpoint binding disagree.
    """
    require_fanout(state, run_id)
    decoded = decode_managed_binding(raw, run_id=run_id)
    if not isinstance(decoded, ManagedFanout) or not isinstance(raw, dict):
        raise ManagedStateError(reason="fanout_invalid", run_id=run_id)
    if binding != decoded.to_checkpoint(pause_id=raw["pause_id"]):
        raise ManagedStateError(reason="fanout_invalid", run_id=run_id)
    return decoded


async def fanout_children(
    state: Any,
    parent_run_id: str,
    parent_tenant: str | None,
    bindings: tuple[ManagedWait, ...],
    root_run_id: str,
    path: tuple[str, ...],
) -> list[tuple[ManagedChildRow, tuple[str, ...]]]:
    """Each binding's child row and validated path, in call order (one projection read).

    The read is scoped to the parent and its tenant; a missing row is ``child_missing``.
    """
    store = unclaimed_store(state)
    rows = await store.get_managed_children(  # type: ignore[union-attr]  # declared fan-out
        parent_run_id, [b.child_run_id for b in bindings], tenant_id=parent_tenant
    )
    found = {row.run_id: row for row in rows}
    children: list[tuple[ManagedChildRow, tuple[str, ...]]] = []
    for binding in bindings:
        row = found.get(binding.child_run_id)
        if row is None:
            raise ManagedStateError(reason="child_missing", run_id=parent_run_id)
        child_path = checked_child(
            parent_run_id, parent_tenant, root_run_id, path, binding.tool_call_id, row
        )
        children.append((row, child_path))
    return children


# ---------------------------------------------------------------------------
# The walk: every leaf below a waiting run, depth-first in call order
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Workflow:
    """A waiting run and every leaf below it (runs that do not wait on children).

    ``leaves`` are depth-first with siblings in their parent batch's call order; each
    carries its registered path. ``start_path`` is the waiting run's own path.
    """

    start: ManagedHop
    start_path: tuple[str, ...]
    leaves: tuple[tuple[ManagedHop, tuple[str, ...]], ...]

    def paused(self) -> tuple[tuple[ManagedHop, tuple[str, ...]], ...]:
        """The leaves at an external pause, in :attr:`leaves` order."""
        return tuple(leaf for leaf in self.leaves if leaf[0].status in PAUSE_STATUSES)

    def agent_path(self, path: tuple[str, ...]) -> list[str]:
        """A leaf's agent names from the waiting run down, as a pending shows them."""
        return [self.start.agent_name, *path[len(self.start_path) :]]


async def read_workflow(
    state: Any, run_id: str, tenant_id: TenantScope, *, read_hop: ReadHop
) -> Workflow | None:
    """Walk every bound child below ``run_id``; None when ``run_id`` is not waiting.

    ``read_hop`` projects one run within a tenant scope (one statement on SQL stores).
    ``tenant_id`` scopes the first read (a missing or foreign run is not found); every
    later hop must carry its parent's tenant. Bounded by :data:`MAX_MANAGED_WALK` runs;
    inconsistent rows raise ``ManagedStateError``. Read-only: no claim, no lock.
    """
    start = await read_hop(run_id, tenant_id)
    if start is None:
        raise RunNotFoundError(run_id)
    if not start.waiting:
        return None
    header = ManagedHeader.from_dict(start.header, run_id=run_id)
    stack = [(start, header.path)]
    leaves: list[tuple[ManagedHop, tuple[str, ...]]] = []
    fetched = 1
    while stack:
        hop, path = stack.pop()
        if not hop.waiting:
            leaves.append((hop, path))
            continue
        allowance = MAX_MANAGED_WALK - fetched
        below = await _children(state, hop, header.root_run_id, path, read_hop, allowance)
        fetched += len(below)
        stack.extend(reversed(below))
    return Workflow(start, header.path, tuple(leaves))


async def _children(
    state: Any,
    hop: ManagedHop,
    root_run_id: str,
    path: tuple[str, ...],
    read_hop: ReadHop,
    allowance: int,
) -> list[tuple[ManagedHop, tuple[str, ...]]]:
    """A waiting hop's bound children with their paths, in call order.

    ``allowance`` is how many more runs the walk may read; a batch larger than that is
    refused (``binding_conflict``) before it is read.
    """
    if not hop.fanout:
        require_allowance(1, allowance, hop.run_id)
        child = await read_hop(_bound_child_id(hop), UNSCOPED)
        return [_checked_child(hop, child, root_run_id, path)]
    store: Any = unclaimed_store(state)
    raw = await _fanout_pause(store, hop)
    fanout = fanout_wait(state, hop.run_id, raw, hop.binding)
    require_allowance(len(fanout.bindings), allowance, hop.run_id)
    rows = await fanout_children(
        state, hop.run_id, hop.tenant_id, fanout.bindings, root_run_id, path
    )
    return [
        (
            ManagedHop(
                row.run_id,
                row.status,
                row.tenant_id,
                row.parent_run_id,
                child_path[-1],
                row.header,
                row.binding,
            ),
            child_path,
        )
        for row, child_path in rows
    ]


@_safe_database_call
async def _fanout_pause(store: Any, hop: ManagedHop) -> Any:
    factory = getattr(store, "_session_factory", None)
    if not callable(factory) or not callable(getattr(store, "pause_projection", None)):
        return await store.get_pause_state(hop.run_id)
    query = select(AgentRun.pause_data).where(AgentRun.id == hop.run_id)
    async with factory() as session:
        result = await session.execute(_apply_tenant_scope(query, hop.tenant_id))
        return result.scalar_one_or_none()


def needs_approval_display(summary: PauseSummary) -> bool:
    kind = PENDING_KIND_BY_STATUS.get(summary.status)
    if kind != "paused":
        return kind in ("approval", "input")
    targets = summary.pause.get("pending_targets") or {}
    gates = summary.pause.get("approval_gates") or {}
    return any(
        isinstance(call, dict)
        and (targets.get(call.get("id")) != "client" or call.get("id") in gates)
        for call in summary.pause.get("pending_tool_calls") or ()
    )


def _latest_event(*kinds: str, before: Any = None, inputs: tuple[Any, ...] = ()) -> Any:
    statement = select(func.max(RunEvent.sequence_index)).where(
        RunEvent.agent_run_id == AgentRun.id, RunEvent.event_type.in_(kinds)
    )
    if before is not None:
        statement = statement.where(RunEvent.sequence_index < before)
    return statement.correlate(AgentRun, *inputs).scalar_subquery()


def _summary_batch_query(store: Any, ids: list[str], tenant: TenantScope, *, whole: bool) -> Any:
    seq = _latest_event(StoredEventType.RUN_PAUSED.value).label("seq")
    statement = _apply_tenant_scope(
        select(AgentRun.id, AgentRun.status, seq, AgentRun.pii_mapping).where(
            store.id_in(AgentRun.id, ids)
        ),
        tenant,
    )
    if whole:
        return statement.add_columns(AgentRun.pause_data)
    if "postgresql" in store.dialect_names:
        # As in the single-run adapter, parse plain JSON once before extracting its keys.
        source = (
            statement.add_columns(AgentRun.pause_data.cast(JSONB).label("doc")).offset(0).subquery()
        )
        doc = source.c.doc
        return select(
            source.c.id,
            source.c.status,
            source.c.seq,
            source.c.pii_mapping,
            *(doc[key] for key in PAUSE_SUMMARY_KEYS),
            doc["steps"][-1],
        )
    return statement.add_columns(*store.pause_projection(AgentRun.pause_data))


def _decode_summary(row: Any, *, whole: bool) -> PauseSummary:
    _, status, seq, mapping, *parts = row
    if whole:
        doc = parts[0] if isinstance(parts[0], dict) else {}
        steps = doc.get("steps")
        last = steps[-1] if isinstance(steps, list) and steps else None
        parts = [doc.get(key) for key in PAUSE_SUMMARY_KEYS] + [last]
    last = parts.pop()
    pause = {k: v for k, v in zip(PAUSE_SUMMARY_KEYS, parts, strict=True) if v is not None}
    return PauseSummary(
        status,
        pause,
        last if isinstance(last, dict) else None,
        seq,
        mapping if isinstance(mapping, dict) else None,
    )


async def read_pause_summaries(
    store: Any, ids: list[str], tenant: TenantScope, *, whole: bool
) -> dict[str, PauseSummary]:
    statement = _summary_batch_query(store, ids, tenant, whole=whole)
    async with store._session_factory() as session:
        rows = (await session.execute(statement)).all()
    return {row.id: _decode_summary(row, whole=whole) for row in rows}


@_safe_database_call
async def leaf_summaries(store: Any, ids: list[str], tenant: str | None) -> dict[str, PauseSummary]:
    """Read bounded pause columns, falling back to legacy plain JSON on conversion errors."""
    try:
        return await read_pause_summaries(store, ids, tenant, whole=False)
    except DBAPIError as exc:
        if not store.json_conversion_error(exc):
            raise
    return await _recover_leaf_summaries(store, ids, tenant)


async def _recover_leaf_summaries(
    store: Any, ids: list[str], tenant: str | None
) -> dict[str, PauseSummary]:
    """Isolate the rows PostgreSQL rejected so the diagnostic names each affected run."""
    if len(ids) == 1:
        _codes.log_code(logging.getLogger(__name__), _codes.PAUSE_DATA_NOT_JSONB, run_id=ids[0])
        return await read_pause_summaries(store, ids, tenant, whole=True)
    # Healthy groups still use the bounded projection. Only failed groups split again.
    middle = len(ids) // 2
    summaries: dict[str, PauseSummary] = await leaf_summaries(store, ids[:middle], tenant)
    summaries.update(await leaf_summaries(store, ids[middle:], tenant))
    return summaries


def _active_generation(seq: Any, *inputs: Any) -> Any:
    event = RunEvent.__table__.c
    tail = RunEvent.__table__.alias("later")
    abort = RunEvent.__table__.alias("abort")
    paused = StoredEventType.RUN_PAUSED.value
    # The summary pins an exact pause. Only events after it can invalidate it.
    at_pause = (
        select(1)
        .where(
            event.agent_run_id == AgentRun.id,
            event.sequence_index == seq,
            event.event_type == paused,
        )
        .correlate(AgentRun, *inputs)
        .exists()
    )
    undone = (
        select(1)
        .where(
            abort.c.agent_run_id == tail.c.agent_run_id,
            abort.c.sequence_index > tail.c.sequence_index,
            abort.c.event_type == StoredEventType.RUN_RESUME_ABORTED.value,
        )
        .correlate(tail)
        .exists()
    )
    invalid = (
        paused,
        StoredEventType.RUN_COMPLETED.value,
        StoredEventType.RUN_ERROR.value,
        StoredEventType.RUN_CANCELLED.value,
    )
    moved = (
        select(1)
        .where(
            tail.c.agent_run_id == AgentRun.id,
            tail.c.sequence_index > seq,
            tail.c.event_type.in_(invalid)
            | ((tail.c.event_type == StoredEventType.RUN_RESUMED.value) & ~undone),
        )
        .correlate(AgentRun, *inputs)
        .exists()
    )
    return at_pause & ~moved


def _approval_window(seq: Any, *inputs: Any) -> Any:
    boundary = _latest_event(
        StoredEventType.RUN_PAUSED.value,
        StoredEventType.RUN_RESUMED.value,
        before=seq,
        inputs=inputs,
    )
    return (
        select(RunEvent.agent_run_id, RunEvent.sequence_index, RunEvent.data)
        .where(
            RunEvent.agent_run_id == AgentRun.id,
            RunEvent.event_type == GovernanceEventType.APPROVAL_REQUESTED.value,
            RunEvent.sequence_index.between(func.coalesce(boundary, -1) + 1, seq - 1),
        )
        .correlate(AgentRun, *inputs)
        .order_by(RunEvent.sequence_index.desc())
        .limit(MAX_PAGE)
    )


def _approval_batch_query(store: Any, generations: dict[str, int], tenant: str | None) -> Any:
    # Recheck the generation and scope in the statement that releases display arguments.
    # A bounded index walk for each run avoids ranking the workflow's entire history.
    if "postgresql" in store.dialect_names:
        inputs = (
            func.unnest(
                bindparam("ids", list(generations), type_=ARRAY(String)),
                bindparam("generations", list(generations.values()), type_=ARRAY(Integer)),
            )
            .table_valued(column("run_id", String), column("seq", Integer))
            .render_derived(name="pauses")
        )
        events = _approval_window(inputs.c.seq, inputs).lateral()
        return _apply_tenant_scope(
            select(events)
            .select_from(inputs)
            .join(AgentRun, AgentRun.id == inputs.c.run_id)
            .join(events, true())
            .where(
                store.id_in(AgentRun.id, list(generations)),
                AgentRun.status.in_(PAUSE_STATUSES),
                _active_generation(inputs.c.seq, inputs),
            ),
            tenant,
        )
    queries = []
    for run_id, generation in generations.items():
        seq = literal(generation)
        events = (
            _approval_window(seq)
            .select_from(AgentRun)
            .where(
                AgentRun.id == run_id, AgentRun.status.in_(PAUSE_STATUSES), _active_generation(seq)
            )
        )
        queries.append(select(_apply_tenant_scope(events, tenant).subquery()))
    return union_all(*queries)


@_safe_database_call
async def leaf_approval_params(
    store: Any, generations: dict[str, int], tenant: str | None
) -> dict[str, dict[str, dict[str, Any]]]:
    """Latest recorded arguments per call within each active pause's own interval."""
    if not generations:
        return {}
    async with store._session_factory() as session:
        rows = (await session.execute(_approval_batch_query(store, generations, tenant))).all()
    # Neither a UNION nor a lateral join promises output order.
    by_run: dict[str, dict[str, dict[str, Any]]] = {}
    for run_id, _, data in sorted(rows, key=lambda row: row.sequence_index):
        if isinstance(data, dict) and isinstance(call_id := data.get("call_id"), str):
            calls = by_run.setdefault(run_id, {})
            if isinstance(params := data.get("params"), dict):
                calls[call_id] = params
            else:
                calls.pop(call_id, None)
    return by_run


async def sql_leaf_data(
    store: Any, ids: list[str], tenant: str | None
) -> tuple[dict[str, PauseSummary], dict[str, dict[str, dict[str, Any]]]]:
    """Read summaries and the arguments evidenced for those exact pause generations."""
    summaries = await leaf_summaries(store, ids, tenant)
    generations = {
        run_id: summary.paused_event_sequence
        for run_id, summary in summaries.items()
        if summary.paused_event_sequence is not None and needs_approval_display(summary)
    }
    return summaries, await leaf_approval_params(store, generations, tenant)
