"""Allocate, persist and deliver the durable run event/note journal."""

from __future__ import annotations

import datetime as _dt
from typing import TYPE_CHECKING, Any, NoReturn

from sqlalchemy import select

from dendrux import _codes
from dendrux._meta_keys import FRESH_EVIDENCE_SESSION_KEY
from dendrux.runtime._bounds import MAX_PAGE, PAUSE_PAGE_MAX
from dendrux.runtime.durability import retry_transient_db
from dendrux.runtime.state import (
    _TERMINAL_STATUS_VALUES,
    UNSCOPED,
    NoteRecord,
    RunEventRecord,
    TenantScope,
    _apply_tenant_scope,
    _insert_event,
    _require_aware_utc,
    _run_in_scope,
    _to_aware_utc,
    _Unscoped,
)
from dendrux.runtime.stores.sql import _sql_support
from dendrux.runtime.stores.sql._sql_support import logger
from dendrux.runtime.stores.sql._sql_writes import _SqlWrites
from dendrux.runtime.stores.sql._types import RunLockScope, SelectSource
from dendrux.types import generate_ulid

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql.base import Executable


class _SqlEvents(_SqlWrites):
    async def add_note(
        self,
        run_id: str,
        *,
        note_id: str,
        text: str,
        metadata: Any = None,
        tenant_id: TenantScope = UNSCOPED,
        max_notes: int | None = None,
        claim_epoch: int | None = None,
    ) -> bool:
        """Conditional ``INSERT ... SELECT`` guarded by status and tenant, row share-locked.

        With ``max_notes`` the statement also requires ``COUNT(notes) < max_notes``.
        On Postgres a transaction-scoped advisory lock per run is taken first,
        so concurrent note submitters of one run count and insert one at a time
        (each statement sees the previous commit) without blocking the runner's
        own writes to the run row. SQLite serializes writers already.
        """
        import datetime as _dt

        from sqlalchemy import (
            DateTime,
            String,
            Text,
            func,
            insert,
            literal,
            null,
            select,
        )

        from dendrux.db.models import AgentRun, RunInbox

        if max_notes is not None and (isinstance(max_notes, bool) or max_notes < 1):
            raise ValueError("max_notes must be a positive int or None.")

        def _note_count() -> Any:
            return select(func.count()).select_from(RunInbox).where(RunInbox.run_id == run_id)

        async def _attempt() -> bool:
            async with self._session_factory() as session:
                if max_notes is not None:
                    if self._note_lock_requires_scope_precheck and (
                        await session.scalar(
                            _apply_tenant_scope(
                                select(AgentRun.id).where(AgentRun.id == run_id), tenant_id
                            )
                        )
                        is None
                    ):
                        return False
                    await self.serialize_in_tx(session, owner=RunLockScope(run_id, tenant_id))
                if claim_epoch is not None:
                    await self._fence(session, run_id, claim_epoch)
                source = select(
                    literal(note_id, String(26)),
                    AgentRun.id,
                    literal(text, Text()),
                    null() if metadata is None else literal(metadata, RunInbox.metadata_.type),
                    literal(_dt.datetime.now(_dt.UTC), DateTime(timezone=True)),
                ).where(
                    AgentRun.id == run_id,
                    AgentRun.status.notin_(_TERMINAL_STATUS_VALUES),
                )
                if max_notes is not None:
                    source = source.where(_note_count().scalar_subquery() < max_notes)
                # FOR SHARE (Postgres; a no-op on SQLite, whose writes are
                # serialized): a concurrent finalize waits for this insert, or
                # this insert re-checks the status the finalize committed.
                source = self.lock_clause(_apply_tenant_scope(source, tenant_id), "share")
                result = await session.execute(
                    insert(RunInbox).from_select(
                        [
                            RunInbox.id,
                            RunInbox.run_id,
                            RunInbox.text,
                            RunInbox.metadata_,
                            RunInbox.created_at,
                        ],
                        source,
                    )
                )
                inserted = bool(result.rowcount and result.rowcount > 0)
                if not inserted:
                    await self._refuse_note_insert(
                        session, run_id, tenant_id, max_notes, _note_count()
                    )
                    await session.rollback()
                    return False
                await self._write_tx(session)
                return inserted

        return await retry_transient_db(
            _attempt,
            label="add_note",
            run_id=run_id,
            classify=self.classify_db_error,
            _raw_driver_errors=True,
        )

    async def _refuse_note_insert(
        self,
        session: AsyncSession,
        run_id: str,
        tenant_id: TenantScope,
        max_notes: int | None,
        count_query: Any,
    ) -> None:
        from sqlalchemy import select

        from dendrux.db.models import AgentRun
        from dendrux.errors import NoteLimitExceededError

        if max_notes is None:
            return
        status = await session.scalar(
            _apply_tenant_scope(select(AgentRun.status).where(AgentRun.id == run_id), tenant_id)
        )
        full = (await session.scalar(count_query) or 0) >= max_notes
        if status is not None and status not in _TERMINAL_STATUS_VALUES and full:
            await session.rollback()
            _codes.log_code(logger, _codes.NOTE_LIMIT, run_id=run_id, limit=max_notes)
            raise NoteLimitExceededError(run_id, limit=max_notes)

    async def get_notes(
        self,
        run_id: str,
        *,
        undelivered_only: bool = False,
        tenant_id: TenantScope = UNSCOPED,
    ) -> list[NoteRecord]:
        from sqlalchemy import select

        from dendrux.db.models import AgentRun, RunInbox

        async with self._session_factory() as session:
            stmt = select(RunInbox).where(RunInbox.run_id == run_id)
            if not isinstance(tenant_id, _Unscoped):
                owner = _apply_tenant_scope(
                    select(AgentRun.id).where(AgentRun.id == run_id), tenant_id
                )
                stmt = stmt.where(owner.exists())
            if undelivered_only:
                stmt = stmt.where(RunInbox.delivered_at.is_(None))
            rows = (
                (await session.execute(stmt.order_by(RunInbox.created_at, RunInbox.id)))
                .scalars()
                .all()
            )
            return [
                NoteRecord(
                    id=r.id,
                    run_id=r.run_id,
                    text=r.text,
                    metadata=r.metadata_,
                    created_at=_to_aware_utc(r.created_at),
                    delivered_at=_to_aware_utc(r.delivered_at),
                    delivered_iteration=r.delivered_iteration,
                )
                for r in rows
            ]

    async def mark_notes_delivered(
        self, run_id: str, note_ids: list[str], *, iteration: int, claim_epoch: int | None = None
    ) -> int:
        import datetime as _dt

        from sqlalchemy import update

        from dendrux.db.models import RunInbox

        async def _attempt() -> int:
            async with self._session_factory() as session:
                if claim_epoch is not None:
                    await self._fence(session, run_id, claim_epoch)
                result = await session.execute(
                    update(RunInbox)
                    .where(
                        RunInbox.run_id == run_id,
                        RunInbox.id.in_(note_ids),
                        RunInbox.delivered_at.is_(None),
                    )
                    .values(
                        delivered_at=_dt.datetime.now(_dt.UTC),
                        delivered_iteration=iteration,
                    )
                )
                await self._write_tx(session)
                return int(result.rowcount or 0)

        return await retry_transient_db(
            _attempt,
            label="mark_notes_delivered",
            run_id=run_id,
            classify=self.classify_db_error,
            _raw_driver_errors=True,
        )

    async def get_next_event_sequence(self, run_id: str) -> int:
        """Return ``max(sequence_index) + 1`` for the run, or 0 if none."""
        from sqlalchemy import func, select

        from dendrux.db.models import RunEvent

        async def _attempt() -> int:
            async with self._session_factory() as session:
                stmt = select(func.max(RunEvent.sequence_index)).where(
                    RunEvent.agent_run_id == run_id
                )
                row = (await session.execute(stmt)).first()
                if row is None or row[0] is None:
                    return 0
                return int(row[0]) + 1

        return await retry_transient_db(
            _attempt,
            label="get_next_event_sequence",
            run_id=run_id,
            classify=self.classify_db_error,
            _raw_driver_errors=True,
        )

    async def save_run_event(
        self,
        run_id: str,
        *,
        event_type: str,
        iteration_index: int = 0,
        correlation_id: str | None = None,
        data: dict[str, Any] | None = None,
        claim_epoch: int | None = None,
        event_id: str | None = None,
        content_digest: str | None = None,
    ) -> int:
        """Append under the leading run-row fence; retries reuse the event identity."""
        async with self._session_factory() as session:
            await self._fence(session, run_id, claim_epoch)
            seq = await _insert_event(
                session,
                run_id,
                event_type,
                data,
                iteration_index=iteration_index,
                correlation_id=correlation_id,
                event_id=event_id,
                content_digest=content_digest,
            )
            self._mark_announce(session, (run_id,))
            await self._write_tx(session)
            return seq

    async def get_events_for_runs(
        self, heads: Mapping[str, int], *, limit_per_run: int
    ) -> dict[str, list[RunEventRecord]]:
        """Read bounded events for already-authorized run ids."""
        from dendrux.runtime.stores.sql._batched_reads import _group_events, _validate_event_batch

        _validate_event_batch(heads, limit_per_run)
        if not heads:
            return {}
        async with self._session_factory() as session:
            rows = (
                await session.execute(
                    self.events_for_runs_query(heads, limit_per_run=limit_per_run)
                )
            ).all()
        return _group_events(rows)

    def events_for_runs_query(self, heads: Mapping[str, int], *, limit_per_run: int) -> Executable:
        """Build an executable SQLAlchemy query for already-authorized event reads.

        ``heads`` maps 1 to 200 run ids to their last-read sequence index; -1
        includes sequence 0. ``limit_per_run`` is validated to be 1 to 100.
        Return all ``run_events`` columns for at most that many earliest events
        above each head, omitting missing runs. The base executes the query,
        groups rows by run and sorts each group by sequence index.

        Store authors may override this hook for their dialect. Without an
        override, a portable UNION ALL combines one bounded subquery per run.
        Empty inputs are handled by ``get_events_for_runs`` without this hook.
        """
        from sqlalchemy import union_all

        from dendrux.db.models import RunEvent

        queries = [
            select(
                select(RunEvent.__table__)
                .where(RunEvent.agent_run_id == run_id, RunEvent.sequence_index > head)
                .order_by(RunEvent.sequence_index)
                .limit(limit_per_run)
                .subquery()
            )
            for run_id, head in heads.items()
        ]
        return union_all(*queries)

    async def get_run_events(
        self,
        run_id: str,
        *,
        after_sequence_index: int | None = None,
        limit: int | None = None,
        event_types: list[str] | None = None,
        before: datetime | None = None,
        tenant_id: TenantScope = UNSCOPED,
        _pause_page: bool = False,
    ) -> list[RunEventRecord]:
        """Events of a run by ``sequence_index``; empty when outside ``tenant_id``."""
        from sqlalchemy import false, select

        from dendrux.db.models import RunEvent

        _require_aware_utc("before", before)

        async with self._session_factory() as session:
            stmt = (
                select(RunEvent)
                .where(RunEvent.agent_run_id == run_id, _run_in_scope(run_id, tenant_id))
                .order_by(RunEvent.sequence_index)
            )

            if after_sequence_index is not None:
                stmt = stmt.where(RunEvent.sequence_index > after_sequence_index)

            if before is not None:
                stmt = stmt.where(RunEvent.created_at < _to_aware_utc(before))

            if event_types is not None:
                if not event_types:
                    # Empty list = "match no event_type" — symmetric with status=[].
                    stmt = stmt.where(false())
                else:
                    stmt = stmt.where(RunEvent.event_type.in_(event_types))

            if limit is not None:
                # Clamp to 1..1000
                clamped = max(1, min(limit, PAUSE_PAGE_MAX if _pause_page else MAX_PAGE))
                stmt = stmt.limit(clamped)

            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                RunEventRecord(
                    id=r.id,
                    event_type=r.event_type,
                    sequence_index=r.sequence_index,
                    iteration_index=r.iteration_index,
                    correlation_id=r.correlation_id,
                    data=r.data,
                    created_at=_to_aware_utc(r.created_at),
                )
                for r in rows
            ]

    async def _insert_event(
        self,
        session: AsyncSession,
        run_id: str,
        event_type: str,
        data: dict[str, Any] | None,
        *,
        iteration_index: int = 0,
        correlation_id: str | None = None,
        event_id: str | None = None,
        content_digest: str | None = None,
    ) -> int:
        """Allocate sequence inside the insert; only a lost acknowledgement reads back."""
        from sqlalchemy import DateTime, Integer, String, func, literal, select
        from sqlalchemy.exc import IntegrityError

        from dendrux.db.models import RunEvent
        from dendrux.errors import PersistenceError
        from dendrux.runtime._digest import event_digest

        digest = event_digest(event_type, iteration_index, correlation_id, data)
        _sql_support._note_blocked_event(session, run_id, event_type)

        def conflict() -> NoReturn:
            _codes.log_code(logger, _codes.WRITE_CONFLICT, run_id=run_id)
            raise PersistenceError(code="write_conflict")

        if content_digest is not None and content_digest != digest:
            conflict()
        row_id = event_id or generate_ulid()
        next_index = (
            select(func.coalesce(func.max(RunEvent.sequence_index), -1) + 1)
            .where(RunEvent.agent_run_id == run_id)
            .scalar_subquery()
        )
        source = select(
            literal(row_id, String(26)),
            literal(run_id, String(26)),
            literal(event_type, String(50)),
            self.event_sequence_expr(next_index),
            literal(iteration_index, Integer()),
            literal(correlation_id, String(26)),
            literal(data, RunEvent.data.type),
            literal(_dt.datetime.now(_dt.UTC), DateTime(timezone=True)),
            literal(digest, String(64)),
        )
        self._remember_write(
            session,
            RunEvent.__table__,
            RunEvent.id == row_id,
            {"agent_run_id": run_id, "content_digest": digest},
        )
        conflict_spec = self._conflict(RunEvent.__table__, ["id"])
        plan = self.insert_if_absent_stmt(
            RunEvent.__table__,
            SelectSource(
                [
                    "id",
                    "agent_run_id",
                    "event_type",
                    "sequence_index",
                    "iteration_index",
                    "correlation_id",
                    "data",
                    "created_at",
                    "content_digest",
                ],
                source,
            ),
            conflict=conflict_spec,
            returning=[RunEvent.sequence_index],
        )
        try:
            result = await self._execute_insert_plan(session, plan, conflict_spec)
            inserted = result is not None and (
                result.rowcount != 0 if not self.supports_insert_returning else True
            )
            seq = (
                result.scalar_one_or_none()
                if result is not None and self.supports_insert_returning
                else None
            )
            if inserted and not self.supports_insert_returning:
                seq = self.inserted_event_sequence(result)
                if seq is None:
                    seq = await session.scalar(
                        select(RunEvent.sequence_index).where(RunEvent.id == row_id)
                    )
        except IntegrityError as error:
            if self.classify_db_error(error).kind == "unique_violation":
                conflict()
            raise
        if seq is None:
            row = (
                await session.execute(select(RunEvent).where(RunEvent.id == row_id))
            ).scalar_one()
            if row.agent_run_id != run_id or row.content_digest != digest:
                conflict()
            return int(row.sequence_index)
        session.info[FRESH_EVIDENCE_SESSION_KEY] = True
        self._mark_announce(session, (run_id,))
        return int(seq)
