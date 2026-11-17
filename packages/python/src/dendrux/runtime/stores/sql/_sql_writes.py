"""Apply dialect-aware fenced SQL writes through commit proof and hint boundaries."""

from __future__ import annotations

import datetime as _dt
from typing import TYPE_CHECKING, Any, NoReturn

from sqlalchemy import select

from dendrux import _codes
from dendrux._lifecycle import STORAGE_FORMAT_CHECKPOINTED, FailureReason, GovernanceEventType
from dendrux._meta_keys import (
    ANNOUNCE_MARKS_SESSION_KEY,
    BLOCKED_INSERT_SESSION_KEY,
    BLOCKED_META_KEY,
    BLOCKED_RUNS_SESSION_KEY,
    COMMIT_PROOFS_SESSION_KEY,
    FRESH_EVIDENCE_SESSION_KEY,
    STORE_SESSION_KEY,
    TOUCHED_RUNS_SESSION_KEY,
)
from dendrux.runtime._bounds import IN_CHUNK as _IN_CHUNK
from dendrux.runtime._bounds import MAX_PAGE
from dendrux.runtime.durability import (
    _CommitNotAppliedError,
    _enqueue_live_updates,
    _mark_commit_issued,
    _safe_database_call,
)
from dendrux.runtime.state import _apply_tenant_scope
from dendrux.runtime.stores.sql import _sql_support
from dendrux.runtime.stores.sql._sql_host import _StoreHost
from dendrux.runtime.stores.sql._sql_support import logger
from dendrux.runtime.stores.sql._types import ConflictSpec, SetFields

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession


class _UpdatedRows:
    def __init__(self, rows: Sequence[Any]) -> None:
        self._rows = list(rows)

    def first(self) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(**self._rows[0]) if self._rows else None

    def scalar_one_or_none(self) -> Any:
        if len(self._rows) > 1:
            from sqlalchemy.exc import MultipleResultsFound

            raise MultipleResultsFound()
        return next(iter(self._rows[0].values())) if self._rows else None


class _SqlWrites(_StoreHost):
    def _conflict(self, table: Any, names: Sequence[str], *constraints: str) -> ConflictSpec:
        known = set(constraints)
        known.add(", ".join(f"{table.name}.{name}" for name in names))
        if list(names) == [col.name for col in table.primary_key.columns]:
            known.add(table.primary_key.name or f"{table.name}_pkey")
        return ConflictSpec([table.c[name] for name in names], constraints=frozenset(known))

    async def _execute_insert_plan(
        self, session: AsyncSession, plan: Any, conflict: ConflictSpec
    ) -> Any:
        from sqlalchemy.exc import IntegrityError

        if plan.conflict_mode == "targeted_ignore":
            return await session.execute(plan.statement)
        try:
            # An expected unique error aborts the statement/savepoint, never the
            # owning transaction (which may already contain a fence/evidence).
            async with session.begin_nested():
                return await session.execute(plan.statement)
        except IntegrityError as exc:
            classified = self.classify_db_error(exc)
            if (
                classified.kind == "unique_violation"
                and classified.constraint in conflict.constraints
            ):
                return None
            raise

    def _remember_write(
        self, session: AsyncSession, table: Any, predicate: Any, expected: dict[str, Any]
    ) -> None:
        session.info.setdefault(COMMIT_PROOFS_SESSION_KEY, []).append((table, predicate, expected))
        if "agent_run_id" in expected:
            session.info.setdefault(TOUCHED_RUNS_SESSION_KEY, set()).add(expected["agent_run_id"])

    def _remember_run_values(
        self, session: AsyncSession, run_id: str, values: dict[str, Any]
    ) -> None:
        from sqlalchemy.sql import ClauseElement

        from dendrux.db.models import AgentRun

        session.info.setdefault(TOUCHED_RUNS_SESSION_KEY, set()).add(run_id)
        expected = {
            key: value for key, value in values.items() if not isinstance(value, ClauseElement)
        }
        if expected:
            self._remember_write(session, AgentRun.__table__, AgentRun.id == run_id, expected)

    def _commit_operation(self, session: AsyncSession, run_ids: Sequence[str]) -> Any:
        from dendrux.runtime.durability import _database_operation

        operation = _database_operation.get()
        _, _, arguments = operation or (None, "", {})
        touched = session.info.setdefault(TOUCHED_RUNS_SESSION_KEY, set())
        touched.update(run_ids)
        touched.update(arguments.get("run_ids", ()))
        run_id = arguments.get("run_id")
        if run_id is not None:
            touched.add(run_id)
        return operation

    def _mark_announce(self, session: AsyncSession, run_ids: Sequence[str]) -> None:
        session.info.setdefault(ANNOUNCE_MARKS_SESSION_KEY, set()).update(run_ids)

    def _enqueue_committed(
        self, session: AsyncSession, commit_issued: bool, succeeded: bool
    ) -> None:
        marks = session.info.pop(ANNOUNCE_MARKS_SESSION_KEY, ())
        if commit_issued and marks and self._live_updates is not None:
            _enqueue_live_updates(self, marks, succeeded)

    async def _write_tx(self, session: AsyncSession, *, run_ids: Sequence[str] = ()) -> None:
        """Commit once; prove identity after a lost acknowledgement or fail closed."""
        from dendrux.runtime.stores.sql._commit import _CommitReadback

        operation = self._commit_operation(session, run_ids)
        # ORM flush remains pre-COMMIT and cannot publish hints on failure.
        await session.flush()
        await self._mark_blocked_writes(session)
        commit_issued = False
        succeeded = False
        try:
            try:
                commit_issued = True
                _mark_commit_issued()
                await session.commit()
            except Exception as error:
                await _CommitReadback(self, session, operation).resolve(error)
            session.info.pop(COMMIT_PROOFS_SESSION_KEY, None)
            session.info.pop(FRESH_EVIDENCE_SESSION_KEY, None)
            succeeded = True
        except _CommitNotAppliedError:
            commit_issued = False
            raise
        finally:
            self._enqueue_committed(session, commit_issued, succeeded)

    async def _mark_blocked_writes(self, session: AsyncSession) -> None:
        from sqlalchemy import update

        from dendrux.db.models import AgentRun
        from dendrux.runtime._blocked_content import _blocked_target

        marked = session.info.setdefault(BLOCKED_RUNS_SESSION_KEY, set())
        target = _blocked_target.get()
        if (
            target is not None
            and target[0] is self
            and target[1] in session.info.get(TOUCHED_RUNS_SESSION_KEY, ())
        ):
            marked.add(target[1])
        # The initial INSERT has the mark; retain it in the COMMIT readback proof.
        for run_id in sorted(marked - {session.info.get(BLOCKED_INSERT_SESSION_KEY)}):
            changed = self.json_patch_expr(AgentRun.meta, SetFields({BLOCKED_META_KEY: True}))
            if changed is None:
                from dendrux.errors import PersistenceError

                raise PersistenceError()
            await session.execute(
                update(AgentRun).where(AgentRun.id == run_id).values(meta=changed)
            )

    def _blocked_runs_query(self, run_ids: Sequence[str], tenant_id: str | None) -> Any:
        from sqlalchemy import func, or_

        from dendrux.db.models import AgentRun, RunEvent

        legacy_event = (
            select(RunEvent.id)
            .where(
                RunEvent.agent_run_id == AgentRun.id,
                RunEvent.event_type == GovernanceEventType.GUARDRAIL_BLOCKED.value,
            )
            .exists()
        )
        blocked = or_(
            self.json_get(AgentRun.meta, (BLOCKED_META_KEY,), "bool").is_(True),
            AgentRun.failure_reason == FailureReason.GUARDRAIL_BLOCKED.value,
            legacy_event,
        )
        return _apply_tenant_scope(
            select(AgentRun.id, func.coalesce(blocked, False)).where(
                self.id_in(AgentRun.id, run_ids)
            ),
            tenant_id,
        )

    def _native_blocked_mark(self) -> object:
        """Private capability: blocked writes and dispatches here commit the run mark, or
        refuse, in the same transaction (the runtime then skips the custom-store path)."""
        from dendrux.runtime._blocked_content import _NATIVE_BLOCKED_MARK

        return _NATIVE_BLOCKED_MARK

    @_safe_database_call
    async def _blocked_run_ids(self, run_ids: Sequence[str], *, tenant_id: str | None) -> set[str]:
        from dendrux.errors import RunNotFoundError

        if tenant_id is not None and (not isinstance(tenant_id, str) or not tenant_id):
            raise ValueError("Blocked source lookup requires an explicit tenant scope")
        result: set[str] = set()
        async with self._session_factory() as session:
            for offset in range(0, len(run_ids), _IN_CHUNK):
                batch = run_ids[offset : offset + _IN_CHUNK]
                rows = dict(
                    (await session.execute(self._blocked_runs_query(batch, tenant_id))).all()
                )
                for run_id in batch:
                    if run_id not in rows:
                        raise RunNotFoundError(run_id)
                    if rows[run_id]:
                        result.add(run_id)
        return result

    async def update_one(
        self,
        session: AsyncSession,
        *,
        table: Any,
        pk: tuple[Any, ...],
        predicate: Any,
        values: Sequence[tuple[Any, Any]],
        returning: Sequence[Any],
        lock_projection: Sequence[Any] = (),
        validate_locked: Callable[[Any], None] | None = None,
    ) -> Any:
        """Update a predicate match; locked readback uses the complete immutable PK."""
        from sqlalchemy import and_, select, update

        session.info[STORE_SESSION_KEY] = self
        columns = tuple(table.primary_key.columns)
        if not columns or len(pk) != len(columns):
            raise ValueError("pk must supply the complete immutable primary key")
        if not returning:
            raise ValueError("returning projection must not be empty")
        if any(getattr(col, "name", None) in {c.name for c in columns} for col, _ in values):
            raise ValueError("primary keys are immutable")
        key = and_(*(col == value for col, value in zip(columns, pk, strict=True)))
        stmt = update(table).where(key, predicate).ordered_values(*values)
        if self.supports_update_returning and not lock_projection and validate_locked is None:
            return (await session.execute(stmt.returning(*returning))).mappings().first()
        projection = list(dict.fromkeys((*columns, *lock_projection)))
        locking_read = self.lock_clause(
            select(*projection).where(key, predicate),
            "update_no_key" if table.name == "dendrux_blobs" else "update",
        )
        if not self.supports_row_locks:
            reservation = (
                update(table).where(key, predicate).ordered_values(*self._reservation_values(table))
            )
            if self.supports_update_returning:
                # Reserve SQLite's writer and project validation data in one
                # statement; a SELECT in legacy driver mode starts no snapshot.
                locking_read = reservation.returning(*projection)
            else:
                await session.execute(reservation)
        row = (await session.execute(locking_read)).mappings().first()
        if row is None:
            return None
        if validate_locked is not None:
            validate_locked(row)
        if any(getattr(col, "name", None) in {c.name for c in columns} for col, _ in values):
            raise ValueError("primary keys are immutable")
        stmt = update(table).where(key).ordered_values(*values)
        if self.supports_update_returning:
            return (await session.execute(stmt.returning(*returning))).mappings().one()
        await session.execute(stmt)
        return (await session.execute(select(*returning).where(key))).mappings().one()

    def _reservation_values(self, table: Any) -> list[tuple[Any, Any]]:
        # Explicit self-assignments suppress model onupdate defaults: reserving
        # a writer must not renew progress or change a status timestamp.
        columns = [next(iter(table.primary_key.columns))]
        columns.extend(
            col for col in table.columns if col.onupdate is not None and col not in columns
        )
        return [(col, col) for col in columns]

    def _ordered(self, table: Any, values: dict[str, Any]) -> list[tuple[Any, Any]]:
        """Readers precede writers on engines evaluating SET left to right."""
        from sqlalchemy.sql import ClauseElement, visitors

        dependencies: dict[str, set[str]] = {key: set() for key in values}
        for reader, expression in values.items():
            if not isinstance(expression, ClauseElement):
                expression = getattr(expression, "__clause_element__", lambda: None)()
            if not isinstance(expression, ClauseElement):
                continue
            for node in visitors.iterate(expression):
                for writer in values:
                    if writer != reader and getattr(node, "shares_lineage", lambda col: False)(
                        table.c[writer]
                    ):
                        dependencies[writer].add(reader)
        ordered = []
        while dependencies:
            ready = next((key for key, readers in dependencies.items() if not readers), None)
            if ready is None:
                raise ValueError("Cyclic SET assignment dependencies")
            ordered.append((table.c[ready], values[ready]))
            del dependencies[ready]
            for readers in dependencies.values():
                readers.discard(ready)
        return ordered

    async def _update_one_rows(
        self,
        session: AsyncSession,
        *,
        table: Any,
        pk: tuple[Any, ...],
        predicate: Any,
        values: dict[str, Any],
        returning: Sequence[Any],
    ) -> Any:
        if table.name == "agent_runs":
            self._remember_run_values(session, str(pk[0]), values)
        row = await self.update_one(
            session,
            table=table,
            pk=pk,
            predicate=predicate,
            values=self._ordered(table, values),
            returning=returning,
        )
        return _UpdatedRows([] if row is None else [row])

    async def _update_many_rows(
        self,
        session: AsyncSession,
        *,
        table: Any,
        predicate: Any,
        values: dict[str, Any],
        returning: Sequence[Any],
        page_size: int,
    ) -> Any:
        if self.supports_update_returning:
            from sqlalchemy import update

            rows = list(
                (
                    await session.execute(
                        update(table)
                        .where(predicate)
                        .ordered_values(*self._ordered(table, values))
                        .returning(*returning)
                    )
                )
                .mappings()
                .all()
            )
            return _UpdatedRows(rows)
        rows = await self.update_many(
            session,
            table=table,
            predicate=predicate,
            values=self._ordered(table, values),
            returning=returning,
            page_size=page_size,
        )
        return _UpdatedRows(rows)

    async def update_many(
        self,
        session: AsyncSession,
        *,
        table: Any,
        predicate: Any,
        values: Sequence[tuple[Any, Any]],
        returning: Sequence[Any],
        page_size: int,
    ) -> list[Any]:
        """Lock a bounded, ordered PK set and update/read back exactly that set."""
        from sqlalchemy import false, select, tuple_, update

        _sql_support._positive_page(page_size)
        if page_size > MAX_PAGE:
            raise ValueError("page_size must not exceed 1000")
        session.info[STORE_SESSION_KEY] = self
        columns = tuple(table.primary_key.columns)
        if not columns or not returning:
            raise ValueError("a complete primary key and returning projection are required")
        if any(getattr(col, "name", None) in {c.name for c in columns} for col, _ in values):
            raise ValueError("primary keys are immutable")
        page = select(*columns).where(predicate).order_by(*columns).limit(page_size)
        locking_read = self.lock_clause(
            page, "update_no_key" if table.name == "dendrux_blobs" else "update"
        )
        if not self.supports_row_locks:
            if self.supports_update_returning:
                locking_read = (
                    update(table)
                    .where(tuple_(*columns).in_(page))
                    .ordered_values(*self._reservation_values(table))
                    .returning(*columns)
                )
            else:
                await session.execute(
                    update(table).where(false()).ordered_values(*self._reservation_values(table))
                )
        keys = list((await session.execute(locking_read)).all())
        if not keys:
            return []
        membership = tuple_(*columns).in_(keys)
        stmt = update(table).where(membership).ordered_values(*values)
        if self.supports_update_returning:
            return list((await session.execute(stmt.returning(*returning))).mappings().all())
        await session.execute(stmt)
        return list(
            (await session.execute(select(*returning).where(membership).order_by(*columns)))
            .mappings()
            .all()
        )

    def _dispatch_filter(self, stmt: Any, run_id: str) -> Any:
        from dendrux.runtime._blocked_content import _dispatch_guarded

        if _dispatch_guarded(self, run_id):
            from dendrux.db.models import AgentRun

            return stmt.where(
                self.json_get(AgentRun.meta, (BLOCKED_META_KEY,), "bool").is_not(True)
            ).execution_options(synchronize_session=False)
        return stmt

    async def _check_dispatch_block(
        self, session: AsyncSession, run_id: str, claim_epoch: int | None
    ) -> None:
        from dendrux.runtime._blocked_content import _dispatch_guarded

        if not _dispatch_guarded(self, run_id):
            return
        from dendrux.db.models import AgentRun
        from dendrux.loops._helpers import _BlockedObservationError
        from dendrux.types import Message, Role, ToolResult

        stmt = select(self.json_get(AgentRun.meta, (BLOCKED_META_KEY,), "bool")).where(
            AgentRun.id == run_id
        )
        if claim_epoch is not None:
            stmt = stmt.where(AgentRun.claim_epoch == claim_epoch)
        marked = failed = False
        try:
            marked = await session.scalar(stmt) is True
        except Exception:
            failed = True
        if failed:
            # The refused write cannot be explained: governance cannot decide, so refuse.
            from dendrux.runtime._blocked_content import _DispatchUndecided

            raise _DispatchUndecided()
        if marked:
            raise _BlockedObservationError(
                Message(Role.USER, "[blocked: 1 finding]"),
                [],
                "Run stopped because it contains blocked content.",
                ToolResult(name="", call_id="", payload="[blocked: 1 finding]"),
            )

    async def _fence(
        self, session: AsyncSession, run_id: str, claim_epoch: int | None, **values: Any
    ) -> None:
        from sqlalchemy import case, update

        from dendrux.db.models import AgentRun
        from dendrux.errors import RunFencedError

        session.info[STORE_SESSION_KEY] = self

        stmt = update(AgentRun).where(AgentRun.id == run_id)
        if claim_epoch is not None:
            stmt = stmt.where(AgentRun.claim_epoch == claim_epoch)
        stmt = self._dispatch_filter(stmt, run_id)
        now = _dt.datetime.now(_dt.UTC)
        # Only the executing owner can renew via a fenced write. Administrative
        # events without a claim must not keep a crashed worker alive.
        if claim_epoch is not None:
            values.setdefault(
                "last_progress_at",
                case(
                    (AgentRun.storage_format == STORAGE_FORMAT_CHECKPOINTED, now),
                    else_=AgentRun.last_progress_at,
                ),
            )
        self._remember_run_values(session, run_id, values)
        result = await session.execute(
            stmt.ordered_values(*self._ordered(AgentRun.__table__, dict(updated_at=now, **values)))
        )
        if not getattr(result, "rowcount", 0):
            await self._check_dispatch_block(session, run_id, claim_epoch)
        if claim_epoch is not None and not getattr(result, "rowcount", 0):
            _codes.log_code(logger, _codes.RUN_FENCED, run_id=run_id)
            raise RunFencedError(run_id, claim_epoch=claim_epoch)

    async def _insert_retry(
        self,
        session: AsyncSession,
        model: Any,
        values: dict[str, Any],
        *,
        digest: str | None = None,
        equality: tuple[str, ...] = (),
    ) -> str:
        from sqlalchemy import select
        from sqlalchemy.exc import IntegrityError

        from dendrux.runtime._digest import content_digest

        self._remember_write(
            session,
            model.__table__,
            model.id == values["id"],
            {
                key: values[key]
                for key in (
                    ("agent_run_id", "content_digest")
                    if digest is not None
                    else ("agent_run_id", *equality)
                )
            },
        )
        conflict = self._conflict(model.__table__, ["id"])
        plan = self.insert_if_absent_stmt(
            model.__table__, values, conflict=conflict, returning=[model.id]
        )
        try:
            result = await self._execute_insert_plan(session, plan, conflict)
            inserted = (
                (
                    result.scalar_one_or_none()
                    if self.supports_insert_returning
                    else values["id"]
                    if result is not None and result.rowcount
                    else None
                )
                if result is not None
                else None
            )
        except IntegrityError as error:
            if self.classify_db_error(error).kind == "unique_violation":
                self._write_conflict(values["agent_run_id"])
            raise
        if inserted is not None:
            session.info[FRESH_EVIDENCE_SESSION_KEY] = True
            return str(inserted)
        row: Any = (
            await session.execute(select(model).where(model.id == values["id"]))
        ).scalar_one()
        equal = row.agent_run_id == values["agent_run_id"]
        if digest is not None:
            equal = equal and row.content_digest == digest
        else:

            def normalize(value: Any) -> Any:
                from decimal import Decimal

                return float(value) if isinstance(value, Decimal) else value

            def field_value(key: str, value: Any) -> Any:
                if key == "cost_usd" and value is not None:
                    from decimal import ROUND_HALF_UP, Decimal

                    return str(
                        Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
                    )
                return normalize(value)

            equal = equal and content_digest(
                {k: field_value(k, getattr(row, k)) for k in equality}
            ) == content_digest({k: field_value(k, values[k]) for k in equality})
        if not equal:
            self._write_conflict(values["agent_run_id"])
        return str(row.id)

    def _write_conflict(self, run_id: str) -> NoReturn:
        from dendrux.errors import PersistenceError

        _codes.log_code(logger, _codes.WRITE_CONFLICT, run_id=run_id)
        raise PersistenceError(code="write_conflict")


_UpdatedRows.__module__ = "dendrux.runtime.stores.sql.base"
