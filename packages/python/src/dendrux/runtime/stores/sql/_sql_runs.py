"""Create or deduplicate a run with its supported storage format."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from sqlalchemy.exc import SQLAlchemyError

import dendrux.runtime._blob_catalog as _blobs
from dendrux import _codes
from dendrux._lifecycle import STORAGE_FORMAT_CHECKPOINTED, TERMINAL_STATUSES
from dendrux._meta_keys import STORE_SESSION_KEY
from dendrux.runtime.durability import (
    _CommitNotAppliedError,
    _safe_database_call,
    retry_transient_db,
)
from dendrux.runtime.state import _TERMINAL_STATUS_VALUES, TraceWrite, _loop_state_at_creation
from dendrux.runtime.stores.sql import _sql_support
from dendrux.runtime.stores.sql._sql_evidence import _SqlEvidence
from dendrux.runtime.stores.sql._sql_support import logger
from dendrux.types import CreateRunResult, IdempotencyConflictError, RunStatus

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from dendrux.media import BlobEdge


class _SqlRuns(_SqlEvidence):
    @_safe_database_call
    async def _creation_storage_format(self, session: AsyncSession | None = None) -> int:
        """Read the same settings pin for entry validation and creation."""
        from sqlalchemy import select

        from dendrux.db.models import DendruxSetting

        checkpoint_capable = all(
            (
                self.capabilities.fenced_writes,
                self.capabilities.retry_ids,
                self.capabilities.checkpointing,
                self.capabilities.atomic_completion,
            )
        )
        if not checkpoint_capable:
            return 1
        if self._format2_active:
            return 2
        if session is None:
            async with self._session_factory() as entry_session:
                return cast("int", await self._creation_storage_format(entry_session))
        storage_format = int(
            await session.scalar(
                select(DendruxSetting.value).where(DendruxSetting.key == "storage_format")
            )
            or "1"
        )
        # Activation is one-way; format 1 must still observe later activation.
        if storage_format == STORAGE_FORMAT_CHECKPOINTED:
            self._format2_active = True
        return storage_format

    async def create_run(
        self,
        run_id: str,
        agent_name: str,
        *,
        input_data: dict[str, Any] | None = None,
        model: str | None = None,
        strategy: str | None = None,
        parent_run_id: str | None = None,
        delegation_level: int = 0,
        tenant_id: str | None = None,
        meta: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        idempotency_fingerprint: str | None = None,
        retry_of_run_id: str | None = None,
        seed_traces: Sequence[TraceWrite] | None = None,
        pii_mapping: dict[str, str] | None = None,
        blob_refs: Sequence[BlobEdge] | None = None,
        blob_refs_scope: str | None = None,
    ) -> CreateRunResult:
        """Create a pinned run; materialize seed traces only for format 2.

        Format-1 seeds are deliberately ignored: the legacy runtime retains its
        original user/assistant trace writes and history behavior. Callers must
        use the returned storage_format to choose their recorder protocol.
        """
        from dendrux.types import _validate_persistable

        _sql_support._check_create_identifiers(
            run_id, agent_name, tenant_id, idempotency_key, model
        )
        _validate_persistable(meta, "metadata")
        _validate_persistable(input_data, "input")
        import datetime as _dt

        from sqlalchemy import select
        from sqlalchemy.exc import IntegrityError

        from dendrux.db.enums import AgentRunStatus
        from dendrux.db.models import AgentRun

        if tenant_id == "":
            raise ValueError("tenant_id must be a non-empty string or None, not ''.")
        if idempotency_key is not None:
            idempotency_fingerprint = idempotency_fingerprint or ""
        # Fast path: if an idempotency key is provided, check for an existing
        # run of this tenant (keys are unique per tenant, not globally).
        if idempotency_key is not None:
            existing = await self._check_idempotency(
                idempotency_key, idempotency_fingerprint or "", tenant_id=tenant_id
            )
            if existing is not None:
                return existing
        absent_retry = False

        async def _prove_creation(session: AsyncSession) -> CreateRunResult:
            from dendrux.errors import StoreOutcomeUnknown
            from dendrux.runtime._digest import content_digest

            try:
                await self._prepare_commit_readback(session)
                row = (
                    await session.execute(
                        self.lock_clause(select(AgentRun).where(AgentRun.id == run_id), "share")
                    )
                ).scalar_one_or_none()
                if (
                    row is not None
                    and row.agent_name == agent_name
                    and row.tenant_id == tenant_id
                    and content_digest(row.input_data) == content_digest(input_data)
                    and row.idempotency_fingerprint == idempotency_fingerprint
                    and row.idempotency_key == idempotency_key
                    and _sql_support._creation_mark_proven(self, run_id, row.meta)
                ):
                    return CreateRunResult(
                        run_id=run_id,
                        outcome="created",
                        status=RunStatus(row.status),
                        storage_format=row.storage_format,
                    )
            except SQLAlchemyError:
                pass
            _codes.log_code(logger, _codes.STORE_OUTCOME_UNKNOWN, run_id=run_id)
            raise StoreOutcomeUnknown() from None

        async def _attempt() -> CreateRunResult:
            nonlocal absent_retry
            async with self._session_factory() as session:
                session.info[STORE_SESSION_KEY] = self
                if parent_run_id is not None:
                    await self.reserve_parent_writer(session, parent_run_id)
                storage_format = await self._creation_storage_format(session)
                if parent_run_id is not None:
                    from dendrux.errors import RunCancelledError, RunNotFoundError

                    parent = (
                        await session.execute(
                            self.lock_clause(
                                select(
                                    AgentRun.cancel_requested, AgentRun.status, AgentRun.tenant_id
                                ).where(AgentRun.id == parent_run_id),
                                "share",
                            )
                        )
                    ).first()
                    if parent is None:
                        raise RunNotFoundError(parent_run_id)
                    if parent.cancel_requested or parent.status in _TERMINAL_STATUS_VALUES:
                        raise RunCancelledError(run_id, parent_run_id)
                run = AgentRun(
                    storage_format=storage_format,
                    pii_mapping=pii_mapping
                    if storage_format == STORAGE_FORMAT_CHECKPOINTED
                    else None,
                    loop_state=_loop_state_at_creation(self, run_id, meta)
                    if storage_format == STORAGE_FORMAT_CHECKPOINTED
                    else None,
                    id=run_id,
                    agent_name=agent_name,
                    **_sql_support._status_values(AgentRunStatus.RUNNING, initial=True),
                    input_data=input_data,
                    model=model,
                    strategy=strategy,
                    parent_run_id=parent_run_id,
                    delegation_level=delegation_level,
                    tenant_id=tenant_id,
                    meta=_sql_support._creation_metadata(session, self, run_id, meta),
                    last_progress_at=_dt.datetime.now(_dt.UTC),
                    idempotency_key=idempotency_key,
                    idempotency_fingerprint=idempotency_fingerprint,
                    retry_of_run_id=retry_of_run_id,
                )
                session.add(run)
                self._remember_write(
                    session,
                    AgentRun.__table__,
                    (AgentRun.idempotency_key == idempotency_key)
                    & self.idempotency_scope_predicate(tenant_id)
                    if idempotency_key is not None
                    else AgentRun.id == run_id,
                    {
                        "id": run_id,
                        "agent_name": agent_name,
                        "tenant_id": tenant_id,
                        "input_data": input_data,
                        "idempotency_fingerprint": idempotency_fingerprint,
                    },
                )
                try:
                    if seed_traces and storage_format == STORAGE_FORMAT_CHECKPOINTED:
                        await session.flush()
                        for trace in seed_traces:
                            await self._insert_trace(session, run_id, trace)
                    if blob_refs:
                        await session.flush()
                        await _blobs.write_blob_refs(
                            session,
                            run_id=run_id,
                            blob_refs=blob_refs,
                            blob_refs_scope=blob_refs_scope,
                            run_tenant_id=tenant_id,
                        )
                    self._mark_announce(session, (run_id,))
                    await self._write_tx(session)
                except _CommitNotAppliedError:
                    absent_retry = True
                    raise
                except IntegrityError as exc:
                    await session.rollback()
                    error = self.classify_db_error(exc)
                    identity_conflict = self._idempotency_violation(exc) or (
                        error.kind == "unique_violation"
                        and error.constraint
                        in self._conflict(AgentRun.__table__, ["id"]).constraints
                    )
                    if not identity_conflict:
                        raise
                    existing = None
                    if idempotency_key is not None:
                        try:
                            existing = await self._check_idempotency(
                                idempotency_key, idempotency_fingerprint or "", tenant_id=tenant_id
                            )
                        except IdempotencyConflictError:
                            if not absent_retry:
                                raise
                    if absent_retry:
                        # An invisible in-flight INSERT can win the unique check
                        # after an absent readback. Re-prove our full identity.
                        return await _prove_creation(session)
                    if existing is not None:
                        return existing
                    if idempotency_key is None or not self._idempotency_violation(exc):
                        raise
                    # The key is held by another tenant's run under a pre-0.5
                    # global unique index (legacy SQLite file). Fail closed
                    # without disclosing that run.
                    _codes.log_code(logger, _codes.IDEMPOTENCY_CONFLICT_LEGACY, run_id=run_id)
                    raise IdempotencyConflictError(
                        run_id=None, idempotency_key=idempotency_key
                    ) from None
            return CreateRunResult(
                run_id=run_id,
                outcome="created",
                status=RunStatus.RUNNING,
                storage_format=storage_format,
            )

        return await retry_transient_db(
            _attempt,
            label="create_run",
            run_id=run_id,
            classify=self.classify_db_error,
            _raw_driver_errors=True,
        )

    def idempotency_scope_predicate(self, tenant_id: str | None) -> Any:
        """Build the tenant portion shared by creation lookup and commit proof."""
        from sqlalchemy import func, literal_column

        from dendrux.db.models import AgentRun

        return func.coalesce(AgentRun.tenant_id, literal_column("''")) == (tenant_id or "")

    async def _check_idempotency(
        self,
        idempotency_key: str,
        fingerprint: str,
        *,
        tenant_id: str | None = None,
    ) -> CreateRunResult | None:
        """Check for an existing run of ``tenant_id`` with this idempotency key.

        Keys are scoped per tenant (``None`` is the untenanted scope, stored
        as ``''`` in the index), so another tenant's run is never returned.
        The predicate spells the index expression so it uses
        ``uq_agent_runs_tenant_idempotency_key``.

        Returns None if no match (caller should create fresh).
        Returns CreateRunResult if match found.
        Raises IdempotencyConflictError if key reused with different input.
        """
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        terminal = TERMINAL_STATUSES

        async with self._session_factory() as session:
            stmt = select(AgentRun).where(
                self.idempotency_scope_predicate(tenant_id),
                AgentRun.idempotency_key == idempotency_key,
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()

            if row is None:
                return None

            # Conflict check: same key, different request
            if (row.idempotency_fingerprint or "") != fingerprint:
                raise IdempotencyConflictError(
                    run_id=row.id,
                    idempotency_key=idempotency_key,
                )

            # Classify by status
            status = RunStatus(row.status)
            if row.status in terminal:
                return CreateRunResult(
                    run_id=row.id,
                    outcome="existing_terminal",
                    status=status,
                    storage_format=row.storage_format,
                )
            return CreateRunResult(
                run_id=row.id,
                outcome="existing_active",
                status=status,
                storage_format=row.storage_format,
            )
