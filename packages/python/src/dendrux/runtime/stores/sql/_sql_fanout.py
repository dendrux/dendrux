"""Bind a fan-out of 2..32 managed children in one transaction, and project their status (0.8)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from dendrux._lifecycle import STORAGE_FORMAT_CHECKPOINTED
from dendrux._managed_wait import _compact_json_bytes
from dendrux._meta_keys import GOVERNANCE_KEY, STORE_SESSION_KEY
from dendrux.db.models import AgentRun, ToolCallRecord
from dendrux.errors import ManagedStateError
from dendrux.runtime._capabilities import require_managed_fanout
from dendrux.runtime._fanout_adoption_state import (
    MAX_FANOUT,
    MIN_FANOUT,
    ROW_STARTED,
    IntentRow,
    fanout_adoption,
    resolved_prefix,
)
from dendrux.runtime._managed_fanout_store import (
    FANOUT_WAIT_SCHEMA_VERSION,
    MAX_CHILD_READ,
    BindManagedChildrenRequest,
    BindManagedChildrenResult,
    BoundChild,
    ManagedChildRow,
    ManagedChildSpec,
    checkpoint_binding,
)
from dendrux.runtime.durability import retry_transient_db
from dendrux.runtime.state import (
    _TERMINAL_STATUS_VALUES,
    MANAGED_BINDING_KEY,
    MANAGED_HEADER_KEY,
    ClaimResult,
    ManagedHeader,
    _apply_tenant_scope,
)
from dendrux.runtime.stores.sql import _sql_support
from dendrux.runtime.stores.sql._sql_managed import _bound_child_id, _SqlManaged
from dendrux.types import (
    MANAGED_PROTOCOL_VERSION,
    MANAGED_WAIT_MAX_BYTES,
    RunStatus,
    _validate_persistable,
    generate_ulid,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from dendrux.runtime.state import TenantScope

_CONFLICT = "binding_conflict"
_Bindings = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _BatchCall(ManagedChildSpec):
    """One child of a batch with its parent's identity: what the single-bind writers read."""

    parent_run_id: str = ""
    iteration_index: int = 0

    @classmethod
    def of(cls, plan: BindManagedChildrenRequest, child: ManagedChildSpec) -> _BatchCall:
        own = {f.name: getattr(child, f.name) for f in fields(ManagedChildSpec)}
        return cls(**own, parent_run_id=plan.parent_run_id, iteration_index=plan.iteration_index)


def _refuse_batch_size(request: BindManagedChildrenRequest) -> None:
    """A batch is 2..32 calls; anything else is refused before any read or write."""
    if not MIN_FANOUT <= len(request.children) <= MAX_FANOUT:
        raise ManagedStateError(reason="fanout_limit", run_id=request.parent_run_id)


def _refuse_oversized_wait(plan: BindManagedChildrenRequest) -> None:
    """The stored wait is bounded like every managed wait: compact UTF-8 JSON, 32 KiB."""
    wait = plan.encoded_wait
    if wait is None or _compact_json_bytes(wait) > MANAGED_WAIT_MAX_BYTES:
        raise ManagedStateError(reason="wait_invalid", run_id=plan.parent_run_id)


def _row_bindings(rows: Sequence[Any], request: BindManagedChildrenRequest) -> _Bindings:
    """Every call's one managed intent row, as ``(call, child)`` in call order; else conflict."""
    by_call: dict[str, list[Any]] = {}
    for row in rows:
        by_call.setdefault(row.tool_call_id, []).append(row)
    bindings = []
    for child in request.children:
        child_id = _bound_child_id(by_call.get(child.tool_call_id, ()))
        if child_id is None:
            raise ManagedStateError(reason=_CONFLICT, run_id=request.parent_run_id)
        bindings.append((child.tool_call_id, child_id))
    if len(rows) != len(bindings) or len({c for _, c in bindings}) != len(bindings):
        raise ManagedStateError(reason=_CONFLICT, run_id=request.parent_run_id)
    return tuple(bindings)


def _child_matches(
    row: Any, request: BindManagedChildrenRequest, call_id: str, path: tuple[str, ...]
) -> bool:
    """A bound child links back to this parent, tenant, root, path and invocation."""
    if row.tenant_id != request.tenant_id or row.parent_run_id != request.parent_run_id:
        return False
    try:
        header = ManagedHeader.from_dict(row.header, run_id=row.id)
    except ManagedStateError:
        return False
    return (
        header.root_run_id == request.root_run_id
        and header.path == path
        and (header.parent_run_id, header.tool_call_id) == (request.parent_run_id, call_id)
    )


def _names_any(state: dict[str, Any], calls: set[str]) -> bool:
    """The checkpoint still binds, or still has pending, any of ``calls``."""
    outgoing = state.get(MANAGED_BINDING_KEY)
    bound: set[Any] = set()
    if isinstance(outgoing, dict):
        bound = {outgoing.get("tool_call_id")}
        if isinstance(outgoing.get("bindings"), dict):
            bound |= set(outgoing["bindings"])
    pending = {p.get("call_id") for p in state.get("pending") or () if isinstance(p, dict)}
    return bool(calls & (bound | pending))


def _parked_at(parent: Any, bindings: _Bindings) -> str | None:
    """The wait generation when pause and checkpoint agree on exactly these bindings."""
    state = parent.loop_state or {}
    outgoing = state.get(MANAGED_BINDING_KEY)
    pending = {p.get("call_id") for p in state.get("pending") or () if isinstance(p, dict)}
    envelope = {
        "v": MANAGED_PROTOCOL_VERSION,
        "n": len(bindings),
        "bindings": [{"tool_call_id": c, "child_run_id": ch} for c, ch in bindings],
    }
    agrees = (
        parent.status == RunStatus.PAUSED.value
        and parent.wait_version == FANOUT_WAIT_SCHEMA_VERSION
        and isinstance(parent.wait_id, str)
        and parent.wait_fanout == envelope
        and outgoing
        == {"v": MANAGED_PROTOCOL_VERSION, "pause_id": parent.wait_id, "bindings": dict(bindings)}
        and {c for c, _ in bindings} <= pending
    )
    return parent.wait_id if agrees else None


def _existing_wait_id(
    parent: Any, rows: Sequence[Any], bindings: _Bindings, parent_id: str
) -> str | None:
    """Classify an established batch against the parent's state (spec §5.3 table).

    Terminal parent: read back, no generation. Parked at the fan-out wait: the generation.
    Claimed for adoption (v4 marker): resolved rows form a call-order prefix; the generation.
    Fully adopted: all rows resolved and nothing still binds or waits on the calls. Anything
    else is ``binding_conflict``.
    """
    if parent.status in _TERMINAL_STATUS_VALUES:
        return None
    state = parent.loop_state or {}
    intents = [IntentRow(r.tool_call_id, r.outcome, r.content_digest, ch) for r, (_, ch) in rows]
    calls = tuple(c for c, _ in bindings)
    # Historical first: a fully resolved batch the checkpoint no longer names reads back as
    # itself, whatever later batch the parent is waiting on or adopting now.
    historical = not any(row.outcome == ROW_STARTED for row, _ in rows)
    if (
        historical
        and not _names_any(state, set(calls))
        and resolved_prefix(intents, bindings, run_id=parent_id) == len(bindings)
    ):
        return None
    adoption = fanout_adoption(state, run_id=parent_id)
    if adoption is not None:
        outgoing = state.get(MANAGED_BINDING_KEY) or {}
        if (
            adoption != calls
            or parent.status != RunStatus.RUNNING.value
            or outgoing.get("bindings") != dict(bindings)
            or not isinstance(outgoing.get("pause_id"), str)
        ):
            raise ManagedStateError(reason=_CONFLICT, run_id=parent_id)
        resolved_prefix(intents, bindings, run_id=parent_id)
        return str(outgoing["pause_id"])
    if all(row.outcome == ROW_STARTED for row, _ in rows):
        wait_id = _parked_at(parent, bindings)
        if wait_id is not None:
            return wait_id
    raise ManagedStateError(reason=_CONFLICT, run_id=parent_id)


def _child_ids(parent_run_id: object, child_run_ids: Sequence[str]) -> list[str]:
    ids = list(child_run_ids)
    if (
        not isinstance(parent_run_id, str)
        or len(ids) > MAX_CHILD_READ
        or any(not isinstance(i, str) for i in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError(f"Read at most {MAX_CHILD_READ} distinct child run ids of one parent")
    return ids


class _SqlFanout(_SqlManaged):
    async def bind_managed_children(
        self, request: BindManagedChildrenRequest
    ) -> BindManagedChildrenResult:
        """Atomically bind a fan-out batch to N fresh running children (internal, 0.8).

        Works on a detached snapshot of ``request`` taken before the first await. One
        transaction under the root lock, then the parent row lock: an established batch is
        read back (never with execution authority); otherwise admission runs once for the
        batch, the parent parks at the v4 wait with its epoch retired, and N intents (one
        private nonce), N children, their seeds and N intent events are written. Only the
        invocation whose nonce reads back receives the N children's execution tokens.
        """
        require_managed_fanout(self)
        _refuse_batch_size(request)
        for child in request.children:
            for value, name in (
                (child.child_meta, "metadata"),
                (child.child_input_data, "input"),
                (child.params, "params"),
            ):
                _validate_persistable(value, name)
        plan = request.detached()
        _refuse_oversized_wait(plan)
        for child in plan.children:
            _sql_support._check_create_identifiers(
                child.child_run_id, child.child_agent_name, plan.tenant_id, None, child.child_model
            )
        # Private to this method invocation and kept across its transient retries.
        nonce = generate_ulid()

        async def _attempt() -> BindManagedChildrenResult:
            async with self._session_factory() as session:
                session.info[STORE_SESSION_KEY] = self
                await self._lock_managed_root(session, plan)
                parent = await self._bind_parent(session, plan)
                existing = await self._fanout_existing(session, plan, parent)
                if existing is not None:
                    await session.rollback()
                    return existing
                states = await self._admit_fanout(session, plan, parent)
                await self._fanout_writes(session, plan, parent, states, nonce)
            wait_id = plan.encoded_wait["pause_id"] if plan.encoded_wait else None
            return BindManagedChildrenResult(
                outcome="created",
                parent_wait_id=wait_id,
                children=tuple(
                    BoundChild(
                        c.tool_call_id,
                        c.child_run_id,
                        ClaimResult(0, STORAGE_FORMAT_CHECKPOINTED, state, None),
                    )
                    for c, state in zip(plan.children, states, strict=True)
                ),
            )

        return await retry_transient_db(
            _attempt,
            label="bind_managed_children",
            run_id=plan.parent_run_id,
            classify=self.classify_db_error,
            _raw_driver_errors=True,
        )

    async def _fanout_existing(
        self, session: AsyncSession, plan: BindManagedChildrenRequest, parent: Any
    ) -> BindManagedChildrenResult | None:
        """Read back an established batch; refuse a partial or conflicting one."""
        calls = [child.tool_call_id for child in plan.children]
        rows = (
            await session.execute(
                select(
                    ToolCallRecord.tool_call_id,
                    ToolCallRecord.target,
                    ToolCallRecord.meta,
                    ToolCallRecord.outcome,
                    ToolCallRecord.content_digest,
                )
                .where(
                    ToolCallRecord.agent_run_id == plan.parent_run_id,
                    ToolCallRecord.tool_call_id.in_(calls),
                )
                .limit(len(calls) + 1)
            )
        ).all()
        if not rows:
            return None
        bindings = _row_bindings(rows, plan)
        await self._check_fanout_children(session, plan, bindings)
        by_call = {row.tool_call_id: row for row in rows}
        paired = [(by_call[c], (c, ch)) for c, ch in bindings]
        wait_id = _existing_wait_id(parent, paired, bindings, plan.parent_run_id)
        return BindManagedChildrenResult(
            outcome="existing",
            parent_wait_id=wait_id,
            children=tuple(BoundChild(c, ch, None) for c, ch in bindings),
        )

    async def _check_fanout_children(
        self, session: AsyncSession, plan: BindManagedChildrenRequest, bindings: _Bindings
    ) -> None:
        """One read of the N bound children; each must match its invocation exactly."""
        rows = (
            await session.execute(
                select(
                    AgentRun.id,
                    AgentRun.parent_run_id,
                    AgentRun.tenant_id,
                    self.json_get(AgentRun.loop_state, (MANAGED_HEADER_KEY,), "json").label(
                        "header"
                    ),
                ).where(AgentRun.id.in_([child_id for _, child_id in bindings]))
            )
        ).all()
        by_id = {row.id: row for row in rows}
        paths = {child.tool_call_id: child.child_path for child in plan.children}
        for call_id, child_id in bindings:
            row = by_id.get(child_id)
            if row is None or not _child_matches(row, plan, call_id, paths[call_id]):
                raise ManagedStateError(reason=_CONFLICT, run_id=plan.parent_run_id)

    async def _admit_fanout(
        self, session: AsyncSession, plan: BindManagedChildrenRequest, parent: Any
    ) -> list[dict[str, Any]]:
        """Admission once for the batch; returns each child's checkpoint, in call order."""
        children = plan.children
        header = self._admit_parent(
            plan, parent, children[0].child_run_id, [c.child_path for c in children]
        )
        headers = [
            self._admit_call(plan, parent, header, c.tool_call_id, c.child_path) for c in children
        ]
        if not len(children) <= plan.max_parallel <= MAX_FANOUT:
            raise ManagedStateError(reason="fanout_limit", run_id=plan.parent_run_id)
        await self._admit_managed_capacity(session, plan, parent, extra=len(children))
        states = []
        for child, child_header in zip(children, headers, strict=True):
            state = {**child.child_loop_state, MANAGED_HEADER_KEY: child_header.to_dict()}
            _sql_support._validate_checkpoint(child.child_run_id, state)
            states.append(state)
        return states

    async def _fanout_writes(
        self,
        session: AsyncSession,
        plan: BindManagedChildrenRequest,
        parent: Any,
        states: list[dict[str, Any]],
        nonce: str,
    ) -> None:
        """Park the parent, then write N intents, N children, seeds and events; commit once."""
        wait = plan.encoded_wait
        if wait is None:  # Only a detached plan carries its encoded wait.
            raise ValueError("bind_managed_children needs a detached request")
        await self._bind_park_parent(session, plan, parent, wait, checkpoint_binding(wait))
        calls = [_BatchCall.of(plan, child) for child in plan.children]
        for call, state in zip(calls, states, strict=True):
            await self._bind_intent(session, call, nonce)
            await self._bind_child(session, call, parent, state, blocked=call.child_blocked)
        for call in calls:
            await self._bind_seeds(session, call)
            await self._bind_event(session, call)
        runs = (plan.parent_run_id, *(call.child_run_id for call in calls))
        await self._write_tx(session, run_ids=runs)

    async def get_managed_children(
        self, parent_run_id: str, child_run_ids: Sequence[str], *, tenant_id: TenantScope
    ) -> list[ManagedChildRow]:
        """Project up to 32 children of a parent in one fresh primary-key read (internal, 0.8).

        Scoped to the parent and the explicit tenant scope; input order; missing omitted. No
        history, no lock; its own session, so each call starts a new transaction.
        """
        ids = _child_ids(parent_run_id, child_run_ids)
        # Built first: an invalid tenant scope is refused even for an empty read.
        statement = _apply_tenant_scope(
            select(
                AgentRun.id,
                AgentRun.parent_run_id,
                AgentRun.tenant_id,
                AgentRun.status,
                self.json_get(AgentRun.loop_state, (MANAGED_HEADER_KEY,), "json"),
                self.json_get(AgentRun.loop_state, (MANAGED_BINDING_KEY,), "json"),
                self.json_get(AgentRun.loop_state, ("loop",), "str"),
                self.json_get(AgentRun.pause_data, ("pause_id",), "str"),
                self.json_get(AgentRun.pause_data, ("pause_reason",), "str"),
                self.json_get(AgentRun.meta, (GOVERNANCE_KEY,), "json"),
            ).where(AgentRun.id.in_(ids), AgentRun.parent_run_id == parent_run_id),
            tenant_id,
        )
        if not ids:
            return []

        async def _attempt() -> list[Any]:
            async with self._session_factory() as session:
                return list((await session.execute(statement)).all())

        rows = await retry_transient_db(
            _attempt,
            label="get_managed_children",
            run_id=parent_run_id,
            classify=self.classify_db_error,
            _raw_driver_errors=True,
        )
        found = {row[0]: ManagedChildRow(*row) for row in rows}
        return [found[run_id] for run_id in ids if run_id in found]
