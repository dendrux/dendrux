"""Bind and admit a managed child while atomically parking its parent."""

from __future__ import annotations

import datetime as _dt
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

import dendrux.runtime.state as _state
from dendrux import _codes
from dendrux._lifecycle import STORAGE_FORMAT_CHECKPOINTED
from dendrux._meta_keys import STORE_SESSION_KEY
from dendrux.runtime.durability import retry_transient_db
from dendrux.runtime.state import (
    _TERMINAL_STATUS_VALUES,
    MANAGED_BINDING_KEY,
    MANAGED_HEADER_KEY,
    BindManagedChildRequest,
    BindManagedChildResult,
    ClaimResult,
    ManagedHeader,
    _apply_loop_patch,
    _usage_totals,
)
from dendrux.runtime.stores.sql import _sql_support
from dendrux.runtime.stores.sql._managed_reads import ManagedReads
from dendrux.runtime.stores.sql._sql_evidence import _SqlEvidence
from dendrux.runtime.stores.sql._sql_support import logger
from dendrux.types import RunStatus, generate_ulid

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession


def _bound_child_id(rows: Sequence[Any]) -> str | None:
    """The child id of exactly one managed intent row, else None (a conflict)."""
    from dendrux.types import _ULID_PATTERN

    if len(rows) != 1 or rows[0].target != "agent":
        return None
    managed = (rows[0].meta or {}).get("managed")
    if not isinstance(managed, dict) or managed.get("v") != 1:
        return None
    child_id = managed.get("child_run_id")
    if not isinstance(child_id, str) or not _ULID_PATTERN.fullmatch(child_id):
        return None
    return child_id


def _child_matches(
    child: Any, child_id: str, request: BindManagedChildRequest, tenant_id: str | None
) -> bool:
    """A bound child links back to exactly this invocation, root, path and tenant.

    Tenant and parent link are compared before the child's header is parsed.
    """
    from dendrux.errors import ManagedStateError

    if child.tenant_id != tenant_id or child.parent_run_id != request.parent_run_id:
        return False
    try:
        header = ManagedHeader.from_dict(
            (child.loop_state or {}).get(MANAGED_HEADER_KEY), run_id=child_id
        )
    except ManagedStateError:
        return False
    return (
        header.root_run_id == request.root_run_id
        and header.path == request.child_path
        and (header.parent_run_id, header.tool_call_id)
        == (request.parent_run_id, request.tool_call_id)
    )


def _existing_wait_id(
    parent: Any, outcome: str, outgoing: Any, request: BindManagedChildRequest, child_id: str
) -> str | None:
    """Classify an established binding and check the evidence for that state.

    * Terminal parent: finalization may clear the pause; the intent and child
      binding stand. Read back, no generation.
    * Completed (adopted) intent: historical. The checkpoint must no longer name
      the call as outgoing or pending.
    * Started intent on a live parent: active. Checkpoint binding, pending call,
      v3 pause generation and envelope must all agree.
    """
    from dendrux.errors import ManagedStateError

    parent_id, call_id = request.parent_run_id, request.tool_call_id
    names_call = isinstance(outgoing, dict) and outgoing.get("tool_call_id") == call_id
    pending = any(
        isinstance(p, dict) and p.get("call_id") == call_id
        for p in (parent.loop_state or {}).get("pending") or ()
    )
    if parent.status in _TERMINAL_STATUS_VALUES:
        return None
    if outcome != "started":
        if names_call or pending:
            raise ManagedStateError(reason="binding_conflict", run_id=parent_id)
        return None
    if outgoing is None:
        raise ManagedStateError(reason="binding_missing", run_id=parent_id)
    if not names_call:
        raise ManagedStateError(reason="binding_conflict", run_id=parent_id)
    if not pending:
        raise ManagedStateError(reason="pending_missing", run_id=parent_id)
    return _active_wait_id(parent, outgoing, request, child_id)


def _active_wait_id(
    parent: Any, outgoing: dict[str, Any], request: BindManagedChildRequest, child_id: str
) -> str:
    """The active binding must agree across checkpoint, v3 pause generation and intent.

    The projected envelope must already be a JSON object; a stored scalar or
    string (even one that encodes an object) is refused without parsing it.
    """
    from dendrux.errors import ManagedStateError

    envelope = parent.wait_binding
    pause_id = outgoing.get("pause_id")
    if (
        outgoing.get("child_run_id") != child_id
        or parent.wait_version != 3
        or not isinstance(pause_id, str)
        or parent.wait_id != pause_id
        or not isinstance(envelope, dict)
        or (envelope.get("tool_call_id"), envelope.get("child_run_id"))
        != (request.tool_call_id, child_id)
    ):
        raise ManagedStateError(reason="binding_conflict", run_id=request.parent_run_id)
    return pause_id


class _SqlManaged(_SqlEvidence, ManagedReads):
    async def bind_managed_child(self, request: BindManagedChildRequest) -> BindManagedChildResult:
        """Atomically bind one managed invocation to a fresh running child (internal, 0.7).

        Works on a detached, validated snapshot of ``request`` taken before the
        first await. One transaction under the parent's row lock: tenant, then an
        existing binding (read back, never with execution authority), then the
        parent's running status, claim epoch, cancellation, managed header,
        outstanding binding, pending call and depth. It then parks the parent as a
        v3 managed wait with its epoch retired, inserts the unique managed tool
        intent, creates the child running with its managed header and seed, and
        writes the intent event. Only the invocation that proves its own private
        nonce on the intent row receives the child's execution token. No
        provider, hook or user code runs inside.
        """
        from dendrux.runtime._capabilities import require_managed_subagents
        from dendrux.types import _validate_persistable

        require_managed_subagents(self)
        for value, name in (
            (request.child_meta, "metadata"),
            (request.child_input_data, "input"),
            (request.params, "params"),
        ):
            _validate_persistable(value, name)
        plan = request.detached()
        _sql_support._check_create_identifiers(
            plan.child_run_id, plan.child_agent_name, plan.tenant_id, None, plan.child_model
        )
        wait = plan.parent_wait.to_dict()
        # Private to this method invocation and kept across its transient retries:
        # a caller replaying an identical request cannot prove it.
        nonce = generate_ulid()

        async def _attempt() -> BindManagedChildResult:
            async with self._session_factory() as session:
                session.info[STORE_SESSION_KEY] = self
                await self._lock_managed_root(session, plan)
                parent = await self._bind_parent(session, plan)
                existing = await self._bind_existing(session, plan, parent)
                if existing is not None:
                    await session.rollback()
                    return existing
                header = await self._admit_managed_limit(session, plan, parent)
                child_state = {**plan.child_loop_state, MANAGED_HEADER_KEY: header.to_dict()}
                _sql_support._validate_checkpoint(plan.child_run_id, child_state)
                await self._bind_park_parent(session, plan, parent, wait)
                await self._bind_intent(session, plan, nonce)
                await self._bind_child(session, plan, parent, child_state)
                await self._bind_seeds(session, plan)
                await self._bind_event(session, plan)
                await self._write_tx(session, run_ids=(plan.parent_run_id, plan.child_run_id))
            return BindManagedChildResult(
                outcome="created",
                child_run_id=plan.child_run_id,
                parent_wait_id=wait["pause_id"],
                child_claim=ClaimResult(0, STORAGE_FORMAT_CHECKPOINTED, child_state, None),
            )

        return await retry_transient_db(
            _attempt,
            label="bind_managed_child",
            run_id=plan.parent_run_id,
            classify=self.classify_db_error,
            _raw_driver_errors=True,
        )

    async def _bind_parent(self, session: AsyncSession, request: BindManagedChildRequest) -> Any:
        """Lock the parent row; a foreign or missing parent is not found (no disclosure)."""
        from dendrux.db.models import AgentRun
        from dendrux.errors import RunNotFoundError

        parent = (
            await session.execute(
                self.lock_clause(
                    select(
                        AgentRun.tenant_id,
                        AgentRun.status,
                        AgentRun.claim_epoch,
                        AgentRun.cancel_requested,
                        self._managed_stop_predicate().label("root_stopped"),
                        AgentRun.storage_format,
                        AgentRun.delegation_level,
                        AgentRun.loop_state,
                        self.json_get(AgentRun.pause_data, ("schema_version",), "int").label(
                            "wait_version"
                        ),
                        self.json_get(AgentRun.pause_data, ("pause_id",), "str").label("wait_id"),
                        self.json_get(AgentRun.pause_data, ("managed_wait",), "json").label(
                            "wait_binding"
                        ),
                    ).where(AgentRun.id == request.parent_run_id),
                    "update",
                )
            )
        ).first()
        if parent is None or parent.tenant_id != request.tenant_id:
            raise RunNotFoundError(request.parent_run_id)
        return parent

    async def _bind_existing(
        self, session: AsyncSession, request: BindManagedChildRequest, parent: Any
    ) -> BindManagedChildResult | None:
        """Read back this invocation's saved binding; refuse a missing or conflicting one.

        The state is classified from the parent's lifecycle and the intent
        outcome, never from one binding copy: see ``_existing_wait_id``.
        """
        from dendrux.db.models import AgentRun, ToolCallRecord
        from dendrux.errors import ManagedStateError

        parent_id, call_id = request.parent_run_id, request.tool_call_id
        outgoing = (parent.loop_state or {}).get(MANAGED_BINDING_KEY)
        rows = (
            await session.execute(
                select(ToolCallRecord.target, ToolCallRecord.meta, ToolCallRecord.outcome)
                .where(
                    ToolCallRecord.agent_run_id == parent_id,
                    ToolCallRecord.tool_call_id == call_id,
                )
                .limit(2)
            )
        ).all()
        if not rows:
            if isinstance(outgoing, dict) and outgoing.get("tool_call_id") == call_id:
                raise ManagedStateError(reason="binding_missing", run_id=parent_id)
            return None
        child_id = _bound_child_id(rows)
        if child_id is None:
            raise ManagedStateError(reason="binding_conflict", run_id=parent_id)
        child = (
            await session.execute(
                select(AgentRun.parent_run_id, AgentRun.tenant_id, AgentRun.loop_state).where(
                    AgentRun.id == child_id
                )
            )
        ).first()
        if child is None:
            raise ManagedStateError(reason="child_missing", run_id=parent_id)
        if not _child_matches(child, child_id, request, parent.tenant_id):
            raise ManagedStateError(reason="child_mismatch", run_id=parent_id)
        wait_id = _existing_wait_id(parent, rows[0].outcome, outgoing, request, child_id)
        return BindManagedChildResult(
            outcome="existing", child_run_id=child_id, parent_wait_id=wait_id, child_claim=None
        )

    def _bind_admit(self, request: BindManagedChildRequest, parent: Any) -> ManagedHeader:
        """Admission under the parent lock; returns the child's managed header."""
        from dendrux.errors import ManagedStateError, RunCancelledError, RunFencedError
        from dendrux.runtime.context import DelegationDepthExceededError

        parent_id = request.parent_run_id
        cancelled = parent.cancel_requested or parent.root_stopped
        if cancelled or parent.status in _TERMINAL_STATUS_VALUES:
            raise RunCancelledError(request.child_run_id, parent_id)
        if (
            parent.status != RunStatus.RUNNING.value
            or parent.claim_epoch != request.parent_claim_epoch
        ):
            _codes.log_code(logger, _codes.RUN_FENCED, run_id=parent_id)
            raise RunFencedError(parent_id, claim_epoch=request.parent_claim_epoch)
        if parent.storage_format != STORAGE_FORMAT_CHECKPOINTED:
            raise ManagedStateError(reason="storage_format", run_id=parent_id)
        state = parent.loop_state or {}
        header = ManagedHeader.from_dict(state.get(MANAGED_HEADER_KEY), run_id=parent_id)
        if header.root_run_id != request.root_run_id or request.child_path[:-1] != header.path:
            raise ManagedStateError(reason="header_invalid", run_id=parent_id)
        if state.get(MANAGED_BINDING_KEY) is not None:
            raise ManagedStateError(reason="binding_outstanding", run_id=parent_id)
        if not any(
            isinstance(p, dict) and p.get("call_id") == request.tool_call_id
            for p in state.get("pending") or ()
        ):
            raise ManagedStateError(reason="pending_missing", run_id=parent_id)
        level = int(parent.delegation_level or 0) + 1
        if request.max_delegation_depth is not None and level > request.max_delegation_depth:
            raise DelegationDepthExceededError(level, request.max_delegation_depth)
        return ManagedHeader(
            root_run_id=request.root_run_id,
            path=request.child_path,
            parent_run_id=parent_id,
            tool_call_id=request.tool_call_id,
        )

    async def _bind_park_parent(
        self,
        session: AsyncSession,
        request: BindManagedChildRequest,
        parent: Any,
        wait: dict[str, Any],
    ) -> None:
        """Save the v3 wait and outgoing binding, and retire the parent's epoch."""
        from sqlalchemy import update

        from dendrux.db.models import AgentRun
        from dendrux.errors import RunFencedError

        parent_id, epoch = request.parent_run_id, request.parent_claim_epoch
        state = _apply_loop_patch(dict(parent.loop_state or {}), request.parent_patch)
        # Both binding copies come from the one wait value that is persisted.
        state[MANAGED_BINDING_KEY] = {**wait["managed_wait"], "pause_id": wait["pause_id"]}
        _sql_support._validate_checkpoint(parent_id, state)
        values: dict[str, Any] = {
            **_sql_support._status_values(RunStatus.PAUSED.value),
            "pause_data": wait,
            "pause_requested": False,
            "claim_epoch": epoch + 1,
            "loop_state": state,
        }
        if request.parent_iteration_count is not None:
            values["iteration_count"] = request.parent_iteration_count
        if request.parent_total_usage is not None:
            values.update(_usage_totals(request.parent_total_usage))
        self._remember_run_values(session, parent_id, values)
        stmt = self._dispatch_filter(
            update(AgentRun).where(
                AgentRun.id == parent_id,
                AgentRun.claim_epoch == epoch,
                AgentRun.status == RunStatus.RUNNING.value,
                AgentRun.cancel_requested.is_(False),
            ),
            parent_id,
        )
        now = _dt.datetime.now(_dt.UTC)
        result = await session.execute(
            stmt.ordered_values(*self._ordered(AgentRun.__table__, dict(updated_at=now, **values)))
        )
        if not getattr(result, "rowcount", 0):
            await self._check_dispatch_block(session, parent_id, epoch)
            _codes.log_code(logger, _codes.RUN_FENCED, run_id=parent_id)
            raise RunFencedError(parent_id, claim_epoch=epoch)

    async def _bind_intent(
        self, session: AsyncSession, request: BindManagedChildRequest, nonce: str
    ) -> None:
        """The unique managed tool intent, started, carrying the binding and creator nonce."""
        from dendrux.db.models import ToolCallRecord
        from dendrux.runtime._digest import content_digest

        digest = content_digest(
            dict(
                tool_call_id=request.tool_call_id,
                tool_name=request.tool_name,
                target="agent",
                params=request.params,
            )
        )
        meta = {"managed": {"v": 1, "child_run_id": request.child_run_id, "bind": nonce}}
        await self._insert_retry(
            session,
            ToolCallRecord,
            dict(
                id=request.row_id,
                agent_run_id=request.parent_run_id,
                tool_call_id=request.tool_call_id,
                provider_tool_call_id=request.provider_tool_call_id,
                tool_name=request.tool_name,
                target="agent",
                params=request.params,
                iteration_index=request.iteration_index,
                outcome="started",
                result="",
                success=False,
                duration_ms=0,
                idempotent=False,
                meta=meta,
                content_digest=digest,
            ),
            digest=digest,
        )
        # A lost acknowledgement grants the token only if this nonce is read back.
        self._remember_write(
            session, ToolCallRecord.__table__, ToolCallRecord.id == request.row_id, {"meta": meta}
        )

    async def _bind_child(
        self,
        session: AsyncSession,
        request: BindManagedChildRequest,
        parent: Any,
        child_state: dict[str, Any],
    ) -> None:
        """Create the child running at epoch 0 with its managed header."""
        from sqlalchemy.exc import IntegrityError

        from dendrux.db.enums import AgentRunStatus
        from dendrux.db.models import AgentRun

        child_id = request.child_run_id
        session.add(
            AgentRun(
                storage_format=STORAGE_FORMAT_CHECKPOINTED,
                pii_mapping=request.child_pii_mapping,
                loop_state=child_state,
                id=child_id,
                agent_name=request.child_agent_name,
                **_sql_support._status_values(AgentRunStatus.RUNNING, initial=True),
                input_data=request.child_input_data,
                model=request.child_model,
                strategy=request.child_strategy,
                parent_run_id=request.parent_run_id,
                delegation_level=int(parent.delegation_level or 0) + 1,
                tenant_id=parent.tenant_id,
                meta=_sql_support._creation_metadata(session, self, child_id, request.child_meta),
                last_progress_at=_dt.datetime.now(_dt.UTC),
                claim_epoch=0,
            )
        )
        self._remember_write(
            session,
            AgentRun.__table__,
            AgentRun.id == child_id,
            {
                "id": child_id,
                "parent_run_id": request.parent_run_id,
                "tenant_id": parent.tenant_id,
                "agent_name": request.child_agent_name,
                "claim_epoch": 0,
                "loop_state": child_state,
            },
        )
        try:
            await session.flush()
        except IntegrityError as error:
            if self.classify_db_error(error).kind == "unique_violation":
                self._write_conflict(request.parent_run_id)
            raise

    async def _bind_seeds(self, session: AsyncSession, request: BindManagedChildRequest) -> None:
        for trace in request.child_seed_traces:
            await self._insert_trace(session, request.child_run_id, trace)

    async def _bind_event(self, session: AsyncSession, request: BindManagedChildRequest) -> None:
        await _state._write_event(session, request.parent_run_id, request.intent_event)
        self._mark_announce(session, (request.parent_run_id, request.child_run_id))
