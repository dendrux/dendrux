"""The shape of a managed workflow as a read-only reader sees it (0.7 chains, 0.8 fan-out).

A parent waiting on subagents is PAUSED with a private managed wait; what an application sees
is every external pause below it, addressed through the run it asked about. This module reads
that shape without history and without locks:

- a v3 wait (one child): one projected primary-key read of the bound child (0.7);
- a v4 wait (2..32 children): its pause envelope, decoded strictly and checked against the
  checkpoint binding, then one ``get_managed_children`` read for every binding.

Every hop is validated (parent link, tenant, root, invocation, path) before anything below it
is read, and the walk is bounded. The per-leaf pending projection belongs to the caller
(:mod:`dendrux.runtime._pending`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from dendrux._lifecycle import PAUSE_STATUSES
from dendrux._managed_wait import ManagedFanout, decode_managed_binding
from dendrux.errors import ManagedStateError, RunNotFoundError
from dendrux.runtime._capabilities import fanout_support
from dendrux.runtime.state import (
    UNSCOPED,
    ManagedHeader,
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


def fanout_wait(state: Any, run_id: str, raw: object, binding: object) -> ManagedFanout:
    """The ordered bindings of a v4 wait, checked against the parent's checkpoint record.

    Refuses before any child is read: ``fanout_capability`` on a store that does not
    declare fan-out, ``wait_invalid`` for an envelope the strict decoder refuses, and
    ``fanout_invalid`` when the pause envelope and the checkpoint binding disagree.
    """
    if not fanout_support(unclaimed_store(state)).capable:
        raise ManagedStateError(reason="fanout_capability", run_id=run_id)
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
    visited = 0
    while stack:
        visited += 1
        if visited > MAX_MANAGED_WALK:
            raise ManagedStateError(reason="binding_conflict", run_id=run_id)
        hop, path = stack.pop()
        if not hop.waiting:
            leaves.append((hop, path))
            continue
        below = await _children(state, hop, header.root_run_id, path, read_hop)
        stack.extend(reversed(below))
    return Workflow(start, header.path, tuple(leaves))


async def _children(
    state: Any, hop: ManagedHop, root_run_id: str, path: tuple[str, ...], read_hop: ReadHop
) -> list[tuple[ManagedHop, tuple[str, ...]]]:
    """A waiting hop's bound children with their paths, in call order."""
    if not hop.fanout:
        child = await read_hop(_bound_child_id(hop), UNSCOPED)
        return [_checked_child(hop, child, root_run_id, path)]
    store: Any = unclaimed_store(state)
    raw = await store.get_pause_state(hop.run_id)
    fanout = fanout_wait(state, hop.run_id, raw, hop.binding)
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
