"""Shared SQL persistence. Lock order: run, ordered children, ordered blobs, settings."""

from __future__ import annotations

import datetime as _dt
import json
import logging
import re
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast

from sqlalchemy import and_, select, true

from dendrux import _codes
from dendrux.runtime._blob_catalog import SQLBlobCatalog, write_blob_refs
from dendrux.runtime.durability import (
    _safe_database_call,
    retry_transient_db,
    safe_database_operations,
)
from dendrux.types import (
    CreateRunResult,
    IdempotencyConflictError,
    RunStatus,
    UsageStats,
    generate_ulid,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence
    from datetime import datetime, timedelta

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from dendrux.db.models import AgentRun
    from dendrux.media import BlobEdge
    from dendrux.runtime.stores._errors import DbError
    from dendrux.runtime.stores.sql._types import DialectFacts, InsertPlan

from dendrux.runtime.state import (
    _COST_DECIMALS,
    _IN_CHUNK,
    _PERSISTED_VALUE_FIELDS,
    _TERMINAL_STATUS_VALUES,
    PAUSE_REQUEST_META_KEY,
    PAUSE_SUMMARY_KEYS,
    UNSCOPED,
    ClaimResult,
    ConversationEntry,
    ConversationPage,
    DelegationInfo,
    EventWrite,
    LLMInteractionRecord,
    LoopStatePatch,
    MarkResult,
    NoteRecord,
    ParentRef,
    PauseSummary,
    ReclaimableRun,
    RunBrief,
    RunControl,
    RunEventRecord,
    RunRecord,
    StoreCapabilities,
    SubtreeNodeRecord,
    SubtreeRecord,
    SubtreeSummary,
    SweptRun,
    TenantScope,
    ToolCallReadRecord,
    TraceRecord,
    TraceWrite,
    UsageRollup,
    _apply_loop_patch,
    _apply_tenant_scope,
    _approval_event_object,
    _check_offset,
    _extract_status,
    _finalized_output_data,
    _finalized_run_read,
    _FinalizedRunRead,
    _finite_json_float,
    _insert_event,
    _insert_event_locked,
    _loop_state_at_creation,
    _reclaimable,
    _require_aware_utc,
    _run_in_scope,
    _run_to_record,
    _to_aware_utc,
    _Unscoped,
    _usage_totals,
    _validate_sweep_age,
    _write_event,
)
from dendrux.runtime.stores.sql._types import (
    ConflictSpec,
    PatchLoopState,
    RunLockScope,
    RunParentCursor,
    SelectSource,
    SetFields,
)

logger = logging.getLogger(__name__)


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


class _SerializedSessionFactory:
    """Session factory that holds the pool's session lock for each session's life.

    The session is closed (ROLLBACK and pool reset included) before the lock
    is released, even when the caller is cancelled during the close.
    """

    def __init__(self, engine: AsyncEngine, factory: Callable[[], AsyncSession]) -> None:
        self._engine = engine
        engine.sync_engine.hide_parameters = True
        self._factory = factory

    @asynccontextmanager
    async def __call__(self) -> AsyncIterator[AsyncSession]:
        from dendrux.db._pool_guard import close_fully, serialized

        async with serialized(self._engine):
            session = self._factory()
            try:
                yield session
            finally:
                await close_fully(session.close)


class _CheckedSessionFactory:
    def __init__(self, store: SQLStateStore, factory: Any) -> None:
        self._store = store
        self._factory = factory

    @asynccontextmanager
    async def __call__(self) -> AsyncIterator[AsyncSession]:
        async with self._factory() as session:
            await self._store._check_connection(session)
            yield session


def _status_values(status: Any, *, initial: bool = False) -> dict[str, Any]:
    from sqlalchemy import case

    from dendrux.db.models import AgentRun

    now = _dt.datetime.now(_dt.UTC)
    return {
        "status": status,
        "status_changed_at": now
        if initial
        else case((AgentRun.status != status, now), else_=AgentRun.status_changed_at),
    }


def _check_create_identifiers(
    run_id: str,
    agent_name: str,
    tenant_id: str | None,
    idempotency_key: str | None,
    model: str | None,
) -> None:
    from dendrux.types import _validate_identifier

    _validate_identifier(run_id, "run_id", maximum=26)
    _validate_identifier(agent_name, "agent_name")
    for name, optional in (
        ("tenant_id", tenant_id),
        ("idempotency_key", idempotency_key),
        ("model", model),
    ):
        if optional is not None:
            _validate_identifier(optional, name)


def _checkpoint_error(run_id: str) -> NoReturn:
    from dendrux.errors import PersistenceError

    _codes.log_code(logger, _codes.CHECKPOINT_TOO_LARGE, run_id=run_id)
    raise PersistenceError(code="checkpoint_too_large")


def _json_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _validate_checkpoint(run_id: str, value: dict[str, Any]) -> None:
    from dendrux.types import _validate_persistable

    _validate_persistable(value, "loop_state")
    if _json_bytes(value) > 32768 or _json_bytes(value.get("provider_kwargs", {})) > 2048:
        _checkpoint_error(run_id)


def _validate_trace_write(run_id: str, trace: TraceWrite) -> None:
    from dendrux.types import _validate_persistable

    _validate_persistable(trace.content, "content")
    _validate_persistable(trace.meta, "meta")
    if trace.trace_kind not in ("seed", "turn", "rewrite", "state"):
        raise ValueError("Unsupported trace kind")
    if trace.trace_kind in ("rewrite", "state"):
        if len(trace.content.encode("utf-8")) > 1024 * 1024:
            _checkpoint_error(run_id)
        value = json.loads(trace.content)
        if trace.trace_kind == "rewrite":
            if not isinstance(value, list) or len(value) > 999:
                _checkpoint_error(run_id)
            if any(
                not isinstance(m, dict)
                or not isinstance(m.get("role"), str)
                or not isinstance(m.get("content"), str)
                for m in value
            ):
                raise ValueError("Rewrite entries require role and content")
            if any(not isinstance(m.get("meta", {}), dict) for m in value):
                _codes.log_code(logger, _codes.REWRITE_INVALID, run_id=run_id)
                raise ValueError("Rewrite entry metadata must be an object")
        elif not isinstance(value, dict) or value.get("kind") not in (
            "image_view",
            "skills_session",
            "clarification",
        ):
            raise ValueError("State traces require a versioned state payload")


def _positive_page(limit: int) -> None:
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")


def _trace_record(row: Any) -> TraceRecord:
    return TraceRecord(
        id=row.id,
        role=row.role,
        content=row.content,
        order_index=row.order_index,
        meta=row.meta,
        created_at=_to_aware_utc(row.created_at),
        trace_kind=row.trace_kind,
    )


def _validated_store_write(method: Any) -> Any:
    from functools import wraps
    from inspect import signature

    sig = signature(method)

    @wraps(method)
    async def write(*args: Any, **kwargs: Any) -> Any:
        from dendrux.types import _validate_persistable

        values = sig.bind(*args, **kwargs).arguments
        for name in _PERSISTED_VALUE_FIELDS.intersection(values):
            value = values[name]
            if isinstance(value, LoopStatePatch):
                value = {
                    "remove_pending": list(value.remove_pending),
                    "set_phase": value.set_phase,
                    "set_fields": dict(value.set_fields),
                    "set_call_phase": dict(value.set_call_phase),
                }
            _validate_persistable(value, name)
        return await method(*args, **kwargs)

    return write


def _validated_store_writes(cls: Any) -> Any:
    from inspect import iscoroutinefunction

    for name, method in list(vars(cls).items()):
        if iscoroutinefunction(method) and name.startswith(
            (
                "save_",
                "create_",
                "finalize_",
                "pause_",
                "try_pause_",
                "submit_",
                "add_note",
                "update_run_meta",
                "request_pause",
                "checkpoint_step",
                "start_tool_call",
                "complete_tool_call",
                "consume_submission",
                "set_trace_link",
            )
        ):
            setattr(cls, name, _validated_store_write(method))
    return cls


@safe_database_operations
@_validated_store_writes
class SQLStateStore(SQLBlobCatalog):
    """Default StateStore implementation using SQLAlchemy async.

    Works with SQLite (zero-config) and Postgres (via an explicit engine).
    Also the blob catalog (``dendrux.media.BlobCatalog``) for image blobs.

    ``create_run``, ``save_trace``, ``save_tool_call`` and ``submit_and_claim``
    take optional ``blob_refs`` + ``blob_refs_scope``: the record's blob edges,
    written in the record's transaction after asserting the run's tenant and
    claiming each blob row (``write_blob_refs``). Passed only when non-empty.
    """

    dialect_names: tuple[str, ...] = ()
    supports_update_returning = False
    supports_insert_returning = False
    supports_row_locks = True
    ddl_transactional = False
    _requires_persisted_claim = True

    @property
    def _uses_combined_sweep(self) -> bool:
        return (
            type(self).sweep_stale_runs is SQLStateStore.sweep_stale_runs
            and type(self).mark_lease_expired is SQLStateStore.mark_lease_expired
        )

    async def _complete_run_control(self, run_id: str, control: RunControl) -> RunControl:
        from dataclasses import replace

        persisted = await SQLStateStore.get_run_control(self, run_id)
        return replace(control, claim_epoch=persisted.claim_epoch, status=persisted.status)

    capabilities: StoreCapabilities = StoreCapabilities(
        fenced_writes=True,
        retry_ids=True,
        checkpointing=True,
        atomic_completion=True,
        notify=True,
        recovery=True,
    )

    def _insert(self, table: Any) -> Any:
        from sqlalchemy import insert

        return insert(table)

    def insert_if_absent_stmt(
        self, table: Any, source: Any, *, conflict: ConflictSpec, returning: Sequence[Any] = ()
    ) -> InsertPlan:
        from dendrux.runtime.stores.sql._types import InsertPlan

        stmt = self._insert(table)
        if isinstance(source, SelectSource):
            stmt = stmt.from_select(source.columns, source.select)
        elif source:
            stmt = stmt.values(source)
        if returning and self.supports_insert_returning:
            stmt = stmt.returning(*returning)
        return InsertPlan(stmt, "raise_unique")

    def lock_clause(self, stmt: Any, intent: Literal["share", "update", "update_no_key"]) -> Any:
        if not self.supports_row_locks:
            return stmt
        return stmt.with_for_update(read=intent == "share")

    async def serialize_in_tx(self, session: AsyncSession, *, owner: RunLockScope) -> None:
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        await session.execute(
            self.lock_clause(
                _apply_tenant_scope(
                    select(AgentRun.id).where(AgentRun.id == owner.run_id), owner.tenant_scope
                ),
                "update",
            )
        )

    def id_in(self, col: Any, ids: Sequence[Any]) -> Any:
        return col.in_(ids)

    def _idempotency_violation(self, exc: BaseException) -> bool:
        error = self.classify_db_error(exc)
        return error.kind == "unique_violation" and error.constraint in (
            "uq_agent_runs_tenant_idempotency_key",
            "ix_agent_runs_idempotency_key",
            "agent_runs_idempotency_key_key",
            "agent_runs.idempotency_key",
        )

    def classify_db_error(self, exc: BaseException) -> DbError:
        from dendrux.runtime.stores._errors import classify_error

        return classify_error(exc)

    async def connection_checks(self, conn: Any) -> DialectFacts:
        from dendrux.runtime.stores.sql._types import DialectFacts

        return DialectFacts(conn.dialect.name, tuple(conn.dialect.server_version_info or ()))

    async def reserve_parent_writer(self, session: AsyncSession, run_id: str) -> None:
        return None

    def blob_now(self) -> Any:
        return _dt.datetime.now(_dt.UTC)

    def json_get(self, col: Any, path: Any, as_type: Any) -> Any:
        raise NotImplementedError("SQL adapters must implement JSON projection")

    def json_contains(self, col: Any, path: Any, value: Any) -> Any:
        raise NotImplementedError("SQL adapters must implement JSON containment")

    def json_is_null(self, col: Any, path: Any) -> Any:
        raise NotImplementedError("SQL adapters must implement JSON null predicates")

    def json_patch_expr(self, col: Any, patch: Any) -> Any:
        return None

    def pause_projection(self, col: Any) -> list[Any]:
        return [self.json_get(col, (key,), "json") for key in PAUSE_SUMMARY_KEYS] + [
            self.json_get(col, ("steps", -1), "json")
        ]

    def json_conversion_error(self, exc: BaseException) -> bool:
        return False

    def upsert_setting(self, table: Any, *, key: str, value: str, updated_at: datetime) -> Any:
        raise NotImplementedError("SQL adapters must implement settings upsert")

    def child_page_query(
        self, *, runs: Any, parents: Any, tenant_scope: Any, page_size: int
    ) -> Any:
        from sqlalchemy import and_, or_, select

        _positive_page(page_size)
        groups: dict[str | None, list[str]] = {}
        for parent in parents:
            groups.setdefault(parent.after, []).append(parent.run_id)
        predicates = [
            and_(
                self.id_in(runs.c.parent_run_id, ids),
                runs.c.id > after if after is not None else true(),
            )
            for after, ids in groups.items()
        ]
        return (
            self._frontier_scope(select(runs.c.id).where(or_(*predicates)), tenant_scope)
            .order_by(runs.c.id)
            .limit(page_size)
        )

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
        session.info.setdefault("dendrux.commit_proofs", []).append((table, predicate, expected))
        if "agent_run_id" in expected:
            session.info.setdefault("dendrux.touched_runs", set()).add(expected["agent_run_id"])

    def _remember_run_values(
        self, session: AsyncSession, run_id: str, values: dict[str, Any]
    ) -> None:
        from sqlalchemy.sql import ClauseElement

        from dendrux.db.models import AgentRun

        session.info.setdefault("dendrux.touched_runs", set()).add(run_id)
        expected = {
            key: value for key, value in values.items() if not isinstance(value, ClauseElement)
        }
        if expected:
            self._remember_write(session, AgentRun.__table__, AgentRun.id == run_id, expected)

    async def _write_tx(self, session: AsyncSession, *, run_ids: Sequence[str] = ()) -> None:
        """Commit once; prove identity after a lost acknowledgement or fail closed."""
        from sqlalchemy import select
        from sqlalchemy.exc import SQLAlchemyError

        from dendrux.runtime.durability import (
            StoreOutcomeUnknown,
            _CommitNotAppliedError,
            _database_operation,
            is_transient_db_error,
        )

        operation = _database_operation.get()
        owner, name, arguments = operation if operation is not None else (None, "", {})
        run_id = arguments.get("run_id")
        touched = session.info.setdefault("dendrux.touched_runs", set())
        touched.update(run_ids)
        touched.update(arguments.get("run_ids", ()))
        if run_id is not None:
            touched.add(run_id)

        def unknown() -> NoReturn:
            _codes.log_code(logger, _codes.STORE_OUTCOME_UNKNOWN, run_id=run_id)
            raise StoreOutcomeUnknown() from None

        try:
            await session.commit()
            session.info.pop("dendrux.commit_proofs", None)
            session.info.pop("dendrux.fresh_evidence", None)
            return
        except SQLAlchemyError as exc:
            classified = self.classify_db_error(exc)
            try:
                await session.rollback()
            except SQLAlchemyError:
                _codes.log_code(logger, _codes.PERSISTENCE_ERROR, run_id=run_id)
                if classified.kind == "connection_lost":
                    unknown()
                from dendrux.errors import PersistenceError

                raise PersistenceError() from None
            if classified.kind in ("deadlock_or_serialization", "lock_timeout") or not (
                is_transient_db_error(exc, classify=self.classify_db_error)
            ):
                raise

        # Claim identity cannot be reconstructed from a later epoch. This also
        # covers submission claims and every other CAS without a retry identity.
        if name in ("claim_paused_run", "submit_and_claim", "reclaim_run", "release_run_claim") or (
            name == "create_run" and arguments.get("idempotency_key") is None
        ):
            unknown()
        proofs = session.info.pop("dendrux.commit_proofs", [])
        fresh_evidence = session.info.pop("dendrux.fresh_evidence", False)
        transitions = {
            "finalize_run",
            "finalize_run_if_status_in",
            "finalize_with_event",
            "try_pause_run",
            "pause_with_event",
            "pause_run",
            "consume_submission",
        }
        claim_epoch = arguments.get("claim_epoch")
        prove_fence = name in transitions and claim_epoch is not None and not fresh_evidence
        if name in (
            "save_trace",
            "save_tool_call",
            "save_usage",
            "save_llm_interaction",
            "save_run_event",
            "start_tool_call",
            "complete_tool_call",
            "consume_submission",
        ):
            evidence = [proof for proof in proofs if proof[0].name != "agent_runs"]
            if evidence and not prove_fence:
                # An immutable retry identity proves publication even after a
                # sibling writer advances the run's mutable execution state.
                proofs = evidence
        if not proofs:
            unknown()

        async def lost() -> NoReturn:
            from dendrux.db.models import AgentRun

            if name not in transitions or run_id is None:
                unknown()
            predicate = AgentRun.id == run_id
            epoch = arguments.get("claim_epoch")
            if epoch is not None:
                predicate &= AgentRun.claim_epoch == epoch
            statuses = arguments.get("allowed_current_statuses")
            if statuses is not None:
                predicate &= AgentRun.status.in_(statuses)
            elif name == "finalize_run" and arguments.get("expected_current_status") is not None:
                predicate &= AgentRun.status == arguments["expected_current_status"]
            elif name in ("try_pause_run", "pause_with_event"):
                predicate &= (AgentRun.status == "running") & AgentRun.cancel_requested.is_(False)
            elif name in ("pause_run", "consume_submission") and epoch is not None:
                predicate &= AgentRun.status == "running"
            if name == "consume_submission":
                predicate &= (
                    self.json_get(AgentRun.loop_state, ("submission", "submission_id"), "str")
                    == arguments["submission_id"]
                )
            if await session.scalar(select(AgentRun.id).where(predicate)) is not None:
                # Raise rather than return a CAS miss: callers such as the
                # recorder discard return values, and the runner cancels on a miss.
                raise _CommitNotAppliedError() from None
            unknown()

        try:
            from decimal import ROUND_HALF_UP, Decimal

            fenced_run_proven = False
            for table, predicate, expected in proofs:
                check_epoch = prove_fence and table.name == "agent_runs"
                if check_epoch:
                    # A fenced transition's owner identity must match in the
                    # same readback as its postcondition, including epoch zero.
                    expected = {**expected, "claim_epoch": claim_epoch}
                stored = (
                    (
                        await session.execute(
                            select(*[table.c[key] for key in expected]).where(predicate)
                        )
                    )
                    .mappings()
                    .one_or_none()
                )
                if stored is None:
                    await lost()
                for key, value in expected.items():
                    actual = stored[key]
                    if isinstance(actual, Decimal) and value is not None:
                        value = Decimal(str(value)).quantize(
                            Decimal("0.000001"), rounding=ROUND_HALF_UP
                        )
                    if isinstance(actual, _dt.datetime) and isinstance(value, _dt.datetime):
                        actual, value = _to_aware_utc(actual), _to_aware_utc(value)
                    if isinstance(actual, (dict, list)) or isinstance(value, (dict, list)):
                        from dendrux.runtime._digest import content_digest

                        actual, value = content_digest(actual), content_digest(value)
                    if actual != value:
                        await lost()
                fenced_run_proven |= check_epoch
            if name in transitions and not fresh_evidence and not fenced_run_proven:
                unknown()
        except SQLAlchemyError:
            unknown()

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

        session.info["dendrux.store"] = self
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

        _positive_page(page_size)
        if page_size > 1000:
            raise ValueError("page_size must not exceed 1000")
        session.info["dendrux.store"] = self
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
        keys = list((await session.execute(locking_read)).tuples().all())
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

    def __init__(self, engine: AsyncEngine) -> None:
        import asyncio

        if engine.dialect.name not in self.dialect_names:
            raise ValueError(
                f"{type(self).__name__} does not support dialect {engine.dialect.name!r}; "
                "use store_for_engine or a SQLStateStore subclass declaring dialect_names."
            )
        self._connection_check_lock = asyncio.Lock()
        self._dialect_facts: Any = None
        self._format2_active = False
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy.orm import sessionmaker

        from dendrux.db._pool_guard import shares_one_connection, warn_if_shared_connection

        self._engine = engine
        self._session_factory = sessionmaker(  # type: ignore[call-overload]
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            info={
                "dendrux.store": self,
                "dendrux.event_hook": lambda session, run_id: self._after_event_insert(
                    session, run_id
                ),
            },
        )
        if shares_one_connection(engine):
            # One shared connection = one shared transaction: overlapping sessions
            # would roll back each other's pending writes (see db/_pool_guard.py).
            warn_if_shared_connection(engine)
            self._session_factory = _SerializedSessionFactory(engine, self._session_factory)
        self._session_factory = _CheckedSessionFactory(self, self._session_factory)

    async def _check_connection(self, session: AsyncSession) -> None:
        if self._dialect_facts is not None:
            return
        async with self._connection_check_lock:
            if self._dialect_facts is not None:
                return
            facts = await self.connection_checks(await session.connection())
            for name in (
                "supports_update_returning",
                "supports_insert_returning",
                "supports_row_locks",
                "ddl_transactional",
            ):
                if name in facts.features:
                    setattr(self, name, facts.features[name])
            self._dialect_facts = facts

    @property
    def store_identity(self) -> str:
        """Comparable identity for this store — the database URL.

        Two stores pointing at the same database have the same identity,
        even if they are different Python objects wrapping different engine
        instances. Used by delegation context to determine safe parent-child
        linking.
        """
        return str(self._engine.url)

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
        if storage_format == 2:
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

        _check_create_identifiers(run_id, agent_name, tenant_id, idempotency_key, model)
        _validate_persistable(meta, "metadata")
        _validate_persistable(input_data, "input")
        import datetime as _dt

        from sqlalchemy import select
        from sqlalchemy.exc import IntegrityError

        from dendrux.db.enums import AgentRunStatus
        from dendrux.db.models import AgentRun

        if tenant_id == "":
            raise ValueError("tenant_id must be a non-empty string or None, not ''.")
        # Fast path: if an idempotency key is provided, check for an existing
        # run of this tenant (keys are unique per tenant, not globally).
        if idempotency_key is not None:
            existing = await self._check_idempotency(
                idempotency_key, idempotency_fingerprint or "", tenant_id=tenant_id
            )
            if existing is not None:
                return existing

        async def _attempt() -> CreateRunResult:
            async with self._session_factory() as session:
                session.info["dendrux.store"] = self
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
                    pii_mapping=pii_mapping if storage_format == 2 else None,
                    loop_state=_loop_state_at_creation(self, run_id, meta)
                    if storage_format == 2
                    else None,
                    id=run_id,
                    agent_name=agent_name,
                    **_status_values(AgentRunStatus.RUNNING, initial=True),
                    input_data=input_data,
                    model=model,
                    strategy=strategy,
                    parent_run_id=parent_run_id,
                    delegation_level=delegation_level,
                    tenant_id=tenant_id,
                    meta=meta,
                    last_progress_at=_dt.datetime.now(_dt.UTC),
                    idempotency_key=idempotency_key,
                    idempotency_fingerprint=idempotency_fingerprint,
                    retry_of_run_id=retry_of_run_id,
                )
                session.add(run)
                if idempotency_key is not None:
                    self._remember_write(
                        session,
                        AgentRun.__table__,
                        (AgentRun.idempotency_key == idempotency_key)
                        & self.idempotency_scope_predicate(tenant_id),
                        {"id": run_id, "idempotency_fingerprint": idempotency_fingerprint},
                    )
                try:
                    if seed_traces and storage_format == 2:
                        await session.flush()
                        for trace in seed_traces:
                            await self._insert_trace(session, run_id, trace)
                    if blob_refs:
                        await session.flush()
                        await write_blob_refs(
                            session,
                            run_id=run_id,
                            blob_refs=blob_refs,
                            blob_refs_scope=blob_refs_scope,
                            run_tenant_id=tenant_id,
                        )
                    await self._write_tx(session)
                except IntegrityError as exc:
                    await session.rollback()
                    # Only resolve via idempotency when the violated constraint is
                    # an idempotency-key unique index, not an arbitrary integrity
                    # error. IntegrityError is NOT transient — don't let
                    # retry_transient_db see it.
                    if idempotency_key is None or not self._idempotency_violation(exc):
                        raise
                    existing = await self._check_idempotency(
                        idempotency_key, idempotency_fingerprint or "", tenant_id=tenant_id
                    )
                    if existing is not None:
                        return existing
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
            _attempt, label="create_run", run_id=run_id, classify=self.classify_db_error
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

        from dendrux.db.enums import AgentRunStatus
        from dendrux.db.models import AgentRun

        terminal = {
            AgentRunStatus.SUCCESS,
            AgentRunStatus.ERROR,
            AgentRunStatus.CANCELLED,
            AgentRunStatus.MAX_ITERATIONS,
        }

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
            if row.idempotency_fingerprint != fingerprint:
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

    async def _fence(
        self, session: AsyncSession, run_id: str, claim_epoch: int | None, **values: Any
    ) -> None:
        from sqlalchemy import case, update

        from dendrux.db.models import AgentRun
        from dendrux.errors import RunFencedError

        session.info["dendrux.store"] = self

        stmt = update(AgentRun).where(AgentRun.id == run_id)
        if claim_epoch is not None:
            stmt = stmt.where(AgentRun.claim_epoch == claim_epoch)
        now = _dt.datetime.now(_dt.UTC)
        # Only the executing owner can renew via a fenced write. Administrative
        # events without a claim must not keep a crashed worker alive.
        if claim_epoch is not None:
            values.setdefault(
                "last_progress_at",
                case((AgentRun.storage_format == 2, now), else_=AgentRun.last_progress_at),
            )
        self._remember_run_values(session, run_id, values)
        result = await session.execute(
            stmt.ordered_values(*self._ordered(AgentRun.__table__, dict(updated_at=now, **values)))
        )
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
            session.info["dendrux.fresh_evidence"] = True
            return str(inserted)
        row = (await session.execute(select(model).where(model.id == values["id"]))).scalar_one()
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

    async def save_trace(
        self,
        run_id: str,
        role: str,
        content: str,
        *,
        order_index: int,
        meta: dict[str, Any] | None = None,
        blob_refs: Sequence[BlobEdge] | None = None,
        blob_refs_scope: str | None = None,
        claim_epoch: int | None = None,
        trace_id: str | None = None,
        trace_kind: str = "turn",
        content_digest: str | None = None,
    ) -> str:
        trace = TraceWrite(
            trace_id or generate_ulid(),
            role,
            content,
            order_index,
            trace_kind,
            meta,
            blob_refs,
            blob_refs_scope,
            content_digest,
        )
        async with self._session_factory() as session:
            if claim_epoch is not None:
                await self._fence(session, run_id, claim_epoch)
            row_id = await self._insert_trace(session, run_id, trace)
            await self._write_tx(session)
            return row_id

    async def save_tool_call(
        self,
        run_id: str,
        *,
        tool_call_id: str,
        provider_tool_call_id: str | None,
        tool_name: str,
        target: str,
        params: dict[str, Any] | None,
        result_payload: str,
        success: bool,
        duration_ms: int,
        iteration_index: int,
        error_message: str | None = None,
        meta: dict[str, Any] | None = None,
        blob_refs: Sequence[BlobEdge] | None = None,
        blob_refs_scope: str | None = None,
        claim_epoch: int | None = None,
        row_id: str | None = None,
        content_digest: str | None = None,
        outcome: str = "completed",
    ) -> str:
        from dendrux.db.models import ToolCallRecord
        from dendrux.runtime._digest import content_digest as digest_of

        digest = digest_of(
            dict(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                target=target,
                params=params,
                result_payload=result_payload,
                success=success,
                outcome=outcome,
            )
        )
        if content_digest is not None and content_digest != digest:
            self._write_conflict(run_id)
        try:
            result_dict = json.loads(
                result_payload, parse_constant=_finite_json_float, parse_float=_finite_json_float
            )
        except (ValueError, TypeError):
            logger.warning("Tool call result JSON decode failed", extra={"run_id": run_id})
            result_dict = {"raw": result_payload}
        async with self._session_factory() as session:
            if claim_epoch is not None:
                await self._fence(session, run_id, claim_epoch)
            saved = await self._insert_retry(
                session,
                ToolCallRecord,
                dict(
                    id=row_id or generate_ulid(),
                    agent_run_id=run_id,
                    tool_call_id=tool_call_id,
                    provider_tool_call_id=provider_tool_call_id,
                    tool_name=tool_name,
                    target=target,
                    params=params,
                    result=result_dict,
                    success=success,
                    outcome=outcome,
                    duration_ms=duration_ms,
                    iteration_index=iteration_index,
                    error_message=error_message,
                    meta=meta,
                    content_digest=digest,
                ),
                digest=digest,
            )
            if blob_refs:
                await write_blob_refs(
                    session, run_id=run_id, blob_refs=blob_refs, blob_refs_scope=blob_refs_scope
                )
            await self._write_tx(session)
            return saved

    async def save_usage(
        self,
        run_id: str,
        *,
        iteration_index: int,
        usage: UsageStats,
        model: str | None = None,
        provider: str | None = None,
        duration_ms: int | None = None,
        meta: dict[str, Any] | None = None,
        usage_id: str | None = None,
        claim_epoch: int | None = None,
    ) -> None:
        from decimal import ROUND_HALF_UP, Decimal

        from dendrux.db.models import TokenUsage

        values = dict(
            id=usage_id or generate_ulid(),
            agent_run_id=run_id,
            iteration_index=iteration_index,
            model=model,
            provider=provider,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=(
                Decimal(str(usage.cost_usd)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
                if usage.cost_usd is not None
                else None
            ),
            duration_ms=duration_ms,
            cache_read_input_tokens=usage.cache_read_input_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            meta=meta,
        )
        async with self._session_factory() as session:
            if claim_epoch is not None:
                await self._fence(session, run_id, claim_epoch)
            # §12.18: equality covers persisted columns, excluding non-persisted UsageStats fields.
            await self._insert_retry(
                session,
                TokenUsage,
                values,
                equality=(
                    "iteration_index",
                    "model",
                    "provider",
                    "duration_ms",
                    "input_tokens",
                    "output_tokens",
                    "cost_usd",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                    "reasoning_tokens",
                ),
            )
            await self._write_tx(session)

    async def save_llm_interaction(
        self,
        run_id: str,
        *,
        iteration_index: int,
        usage: UsageStats,
        model: str | None = None,
        provider: str | None = None,
        duration_ms: int | None = None,
        semantic_request: dict[str, Any] | None = None,
        semantic_response: dict[str, Any] | None = None,
        provider_request: dict[str, Any] | None = None,
        provider_response: dict[str, Any] | None = None,
        guardrail_findings: dict[str, Any] | None = None,
        interaction_id: str | None = None,
        claim_epoch: int | None = None,
    ) -> None:
        from dendrux.db.models import LLMInteraction

        values = dict(
            id=interaction_id or generate_ulid(),
            agent_run_id=run_id,
            iteration_index=iteration_index,
            model=model,
            provider=provider,
            semantic_request=semantic_request,
            semantic_response=semantic_response,
            provider_request=provider_request,
            provider_response=provider_response,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=usage.cost_usd,
            duration_ms=duration_ms,
            cache_read_input_tokens=usage.cache_read_input_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            guardrail_findings=guardrail_findings,
        )
        async with self._session_factory() as session:
            if claim_epoch is not None:
                await self._fence(session, run_id, claim_epoch)
            await self._insert_retry(
                session,
                LLMInteraction,
                values,
                equality=(
                    "iteration_index",
                    "model",
                    "provider",
                    "semantic_request",
                    "semantic_response",
                    "provider_request",
                    "provider_response",
                ),
            )
            await self._write_tx(session)

    async def _insert_trace(self, session: AsyncSession, run_id: str, trace: TraceWrite) -> str:
        from dendrux.db.models import ReactTrace
        from dendrux.runtime._digest import trace_digest

        _validate_trace_write(run_id, trace)
        digest = trace_digest(
            trace.role, trace.content, trace.order_index, trace.trace_kind, trace.meta
        )
        if trace.content_digest is not None and trace.content_digest != digest:
            self._write_conflict(run_id)
        row_id = await self._insert_retry(
            session,
            ReactTrace,
            dict(
                id=trace.trace_id,
                agent_run_id=run_id,
                role=trace.role,
                content=trace.content,
                order_index=trace.order_index,
                trace_kind=trace.trace_kind,
                meta=trace.meta,
                content_digest=digest,
            ),
            digest=digest,
        )
        if trace.trace_kind == "rewrite":
            # A run-owned pointer avoids scanning every conversation row to
            # discover whether a rewrite exists. It commits with the trace.
            from sqlalchemy import update

            from dendrux.db.models import AgentRun

            pointer = {"trace_id": trace.trace_id, "order_index": trace.order_index}
            current_order = self.json_get(
                AgentRun.meta, ("dendrux.latest_rewrite", "order_index"), "int"
            )
            metadata = self.json_patch_expr(
                AgentRun.meta, SetFields({"dendrux.latest_rewrite": pointer})
            )
            if metadata is None:
                current_meta = await session.scalar(
                    self.lock_clause(select(AgentRun.meta).where(AgentRun.id == run_id), "update")
                )
                metadata = {**(current_meta or {}), "dendrux.latest_rewrite": pointer}
            await session.execute(
                update(AgentRun)
                .where(
                    AgentRun.id == run_id,
                    (current_order.is_(None)) | (current_order <= trace.order_index),
                )
                .values(meta=metadata)
            )
        if trace.blob_refs:
            await write_blob_refs(
                session,
                run_id=run_id,
                blob_refs=trace.blob_refs,
                blob_refs_scope=trace.blob_refs_scope,
            )
        return row_id

    async def checkpoint_step(
        self,
        run_id: str,
        *,
        claim_epoch: int,
        loop_state: dict[str, Any],
        pii_mapping: dict[str, str] | None,
        trace: TraceWrite | None,
    ) -> None:
        """Commit a bounded execution projection and optional trace under one fence."""
        _validate_checkpoint(run_id, loop_state)
        values: dict[str, Any] = {"loop_state": loop_state}
        if pii_mapping is not None:
            values["pii_mapping"] = pii_mapping
        async with self._session_factory() as session:
            await self._fence(session, run_id, claim_epoch, **values)
            if trace is not None:
                await self._insert_trace(session, run_id, trace)
            await self._write_tx(session)

    async def _patch_fence(
        self,
        session: AsyncSession,
        run_id: str,
        claim_epoch: int,
        patch: LoopStatePatch,
        *,
        count_tool: str | None = None,
        pii_mapping: dict[str, str] | None = None,
    ) -> None:
        from sqlalchemy import select, update

        from dendrux.db.models import AgentRun
        from dendrux.errors import RunFencedError

        if "pending" in patch.set_fields:
            raise ValueError("A patch cannot replace pending; remove calls by identity")
        _validate_checkpoint(run_id, dict(patch.set_fields))
        changed = self.json_patch_expr(AgentRun.loop_state, PatchLoopState(patch, count_tool))
        if changed is not None:
            result = await self._update_one_rows(
                session,
                table=AgentRun.__table__,
                pk=(run_id,),
                predicate=and_(AgentRun.id == run_id, AgentRun.claim_epoch == claim_epoch),
                values=dict(
                    updated_at=_dt.datetime.now(_dt.UTC),
                    last_progress_at=_dt.datetime.now(_dt.UTC),
                    loop_state=changed,
                    **{"pii_mapping": pii_mapping} if pii_mapping is not None else {},
                ),
                returning=[AgentRun.loop_state],
            )
            merged = result.scalar_one_or_none()
            if merged is not None:
                # Same compact UTF-8 measure as checkpoint_step and SQLite.
                # A failed bound rolls back the patch and the whole completion.
                _validate_checkpoint(run_id, merged)
                self._remember_run_values(session, run_id, {"loop_state": merged})
            else:
                epoch = await session.scalar(
                    select(AgentRun.claim_epoch).where(AgentRun.id == run_id)
                )
                if epoch == claim_epoch:
                    _checkpoint_error(run_id)
                _codes.log_code(logger, _codes.RUN_FENCED, run_id=run_id)
                raise RunFencedError(run_id, claim_epoch=claim_epoch)
        else:
            await self._fence(
                session,
                run_id,
                claim_epoch,
                **({"pii_mapping": pii_mapping} if pii_mapping is not None else {}),
            )
            current = await session.scalar(select(AgentRun.loop_state).where(AgentRun.id == run_id))
            changed = _apply_loop_patch(current or {}, patch)
            if count_tool is not None:
                counts = dict(changed.get("tool_counts", {}))
                counts[count_tool] = counts.get(count_tool, 0) + 1
                changed["tool_counts"] = counts
            _validate_checkpoint(run_id, changed)
            self._remember_run_values(session, run_id, {"loop_state": changed})
            await session.execute(
                update(AgentRun).where(AgentRun.id == run_id).values(loop_state=changed)
            )

    async def start_tool_call(
        self,
        run_id: str,
        *,
        claim_epoch: int,
        row_id: str,
        tool_call_id: str,
        provider_tool_call_id: str | None,
        tool_name: str,
        target: str,
        params: dict[str, Any] | None,
        iteration_index: int,
        idempotent: bool,
        event: EventWrite,
        meta: dict[str, Any] | None = None,
        blob_refs: Sequence[BlobEdge] | None = None,
        blob_refs_scope: str | None = None,
    ) -> None:
        """Commit server-side intent and its event before invoking a tool."""
        from dendrux.db.models import ToolCallRecord
        from dendrux.runtime._digest import content_digest

        if target == "client":
            raise ValueError("Client calls cannot have execution intents")
        digest = content_digest(
            dict(tool_call_id=tool_call_id, tool_name=tool_name, target=target, params=params)
        )
        async with self._session_factory() as session:
            await self._fence(session, run_id, claim_epoch)
            await self._insert_retry(
                session,
                ToolCallRecord,
                dict(
                    id=row_id,
                    agent_run_id=run_id,
                    tool_call_id=tool_call_id,
                    provider_tool_call_id=provider_tool_call_id,
                    tool_name=tool_name,
                    target=target,
                    params=params,
                    iteration_index=iteration_index,
                    outcome="started",
                    result="",
                    success=False,
                    duration_ms=0,
                    idempotent=idempotent,
                    meta=meta,
                    content_digest=digest,
                ),
                digest=digest,
            )
            if blob_refs:
                await write_blob_refs(
                    session, run_id=run_id, blob_refs=blob_refs, blob_refs_scope=blob_refs_scope
                )
            await _write_event(session, run_id, event)
            await self._write_tx(session)

    async def complete_tool_call(
        self,
        run_id: str,
        *,
        claim_epoch: int,
        tool_call_id: str,
        outcome: str,
        result_payload: str,
        success: bool,
        duration_ms: int,
        error_message: str | None,
        replayed: bool,
        observation: TraceWrite,
        event: EventWrite,
        loop_state_patch: LoopStatePatch,
        meta: dict[str, Any] | None = None,
        blob_refs: Sequence[BlobEdge] | None = None,
        blob_refs_scope: str | None = None,
        pii_mapping: dict[str, str] | None = None,
    ) -> None:
        """Atomically resolve one intent, its reserved observation and pending entry."""
        from sqlalchemy import select

        from dendrux.db.models import ToolCallRecord
        from dendrux.runtime._digest import content_digest

        if outcome not in ("completed", "failed", "unknown"):
            raise ValueError("Unsupported tool outcome")
        digest = content_digest(
            dict(
                tool_call_id=tool_call_id,
                outcome=outcome,
                result_payload=result_payload,
                success=success,
                error_message=error_message,
            )
        )
        try:
            payload = json.loads(
                result_payload, parse_constant=_finite_json_float, parse_float=_finite_json_float
            )
        except (ValueError, TypeError):
            payload = {"raw": result_payload}
        values: dict[str, Any] = dict(
            outcome=outcome,
            result=payload,
            success=success,
            duration_ms=duration_ms,
            error_message=error_message,
            content_digest=digest,
        )
        if replayed:
            values["replayed_at"] = _dt.datetime.now(_dt.UTC)
        if meta is not None:
            values["meta"] = meta
        async with self._session_factory() as session:
            await self._patch_fence(
                session,
                run_id,
                claim_epoch,
                loop_state_patch,
                count_tool=(observation.meta or {}).get("tool_name"),
                pii_mapping=pii_mapping,
            )
            row_id = (
                await self._update_many_rows(
                    session,
                    table=ToolCallRecord.__table__,
                    page_size=1,
                    predicate=and_(
                        ToolCallRecord.agent_run_id == run_id,
                        ToolCallRecord.tool_call_id == tool_call_id,
                        ToolCallRecord.outcome == "started",
                    ),
                    values=dict(**values),
                    returning=[ToolCallRecord.id],
                )
            ).scalar_one_or_none()
            if row_id is None:
                stored = await session.scalar(
                    select(ToolCallRecord.content_digest).where(
                        ToolCallRecord.agent_run_id == run_id,
                        ToolCallRecord.tool_call_id == tool_call_id,
                    )
                )
                await session.rollback()
                if stored != digest:
                    self._write_conflict(run_id)
                return
            await self._insert_trace(session, run_id, observation)
            if blob_refs:
                await write_blob_refs(
                    session, run_id=run_id, blob_refs=blob_refs, blob_refs_scope=blob_refs_scope
                )
            await _write_event(session, run_id, event)
            await self._write_tx(session)

    async def consume_submission(
        self,
        run_id: str,
        *,
        claim_epoch: int,
        submission_id: str,
        traces: Sequence[TraceWrite],
        patch: LoopStatePatch,
        pii_mapping: dict[str, str] | None = None,
    ) -> bool:
        """Consume a durable submission once, even after an acknowledgement is lost."""
        from sqlalchemy import select, update

        from dendrux.db.models import AgentRun

        async with self._session_factory() as session:
            await self._fence(
                session,
                run_id,
                claim_epoch,
                **({"pii_mapping": pii_mapping} if pii_mapping is not None else {}),
            )
            row = (
                await session.execute(
                    select(AgentRun.loop_state, AgentRun.pause_data).where(AgentRun.id == run_id)
                )
            ).one()
            current = row.loop_state or {}
            if (current.get("submission") or {}).get("submission_id") != submission_id:
                await session.rollback()
                return False
            changed = _apply_loop_patch(current, patch)
            changed["submission"] = None
            self._remember_run_values(session, run_id, {"loop_state": changed})
            _validate_checkpoint(run_id, changed)
            pause = dict(row.pause_data or {})
            pause.pop("submission", None)
            for trace in traces:
                await self._insert_trace(session, run_id, trace)
            await session.execute(
                update(AgentRun)
                .where(AgentRun.id == run_id)
                .values(loop_state=changed, pause_data=pause)
            )
            await self._write_tx(session)
            return True

    async def set_trace_link(self, run_id: str, *, claim_epoch: int, trace_link: str) -> None:
        """Persist a validated traceparent under the execution fence."""
        from dendrux.runtime._tracelink import validate_trace_link

        if validate_trace_link(trace_link) is None:
            _codes.log_code(logger, _codes.TRACE_LINK_INVALID, run_id=run_id)
            raise ValueError("Invalid trace link")
        async with self._session_factory() as session:
            await self._fence(session, run_id, claim_epoch, trace_link=trace_link)
            await self._write_tx(session)

    def _frontier_predicate(self, column: Any, run_ids: Sequence[str]) -> Any:
        return self.id_in(column, run_ids)

    def _frontier_scope(self, statement: Any, tenant_id: TenantScope) -> Any:
        from dendrux.db.models import AgentRun
        from dendrux.runtime.state import _check_scope

        if isinstance(tenant_id, _Unscoped):
            return statement
        _check_scope(tenant_id)
        return statement.where(AgentRun.tenant_id.is_not_distinct_from(tenant_id))

    async def mark_cancel_requested(self, run_ids: Sequence[str], *, tenant_id: TenantScope) -> int:
        """Mark a frontier in one statement before scanning its children."""
        from sqlalchemy import update

        from dendrux.db.models import AgentRun

        if not run_ids:
            return 0

        async def _attempt() -> int:
            async with self._session_factory() as session:
                result = await session.execute(
                    self._frontier_scope(
                        update(AgentRun)
                        .where(
                            self._frontier_predicate(AgentRun.id, run_ids),
                            AgentRun.status.notin_(_TERMINAL_STATUS_VALUES),
                        )
                        .values(cancel_requested=True)
                        .execution_options(synchronize_session=False),
                        tenant_id,
                    )
                )
                await self._write_tx(session)
                return int(getattr(result, "rowcount", 0))

        return await retry_transient_db(
            _attempt,
            label="mark_cancel_requested",
            run_id=run_ids[0],
            classify=self.classify_db_error,
        )

    async def get_children_page(
        self, parent_ids: Sequence[str], *, after: str | None, limit: int, tenant_id: TenantScope
    ) -> list[str]:
        """Read a tenant-scoped keyset page, capped at 500 children of a marked frontier."""

        from dendrux.db.models import AgentRun

        _positive_page(limit)
        if not parent_ids:
            return []
        size = min(limit, 500)
        statement = self.child_page_query(
            runs=AgentRun.__table__,
            parents=[RunParentCursor(parent, after) for parent in sorted(set(parent_ids))],
            tenant_scope=tenant_id,
            page_size=size,
        )

        async def _attempt() -> list[str]:
            async with self._session_factory() as session:
                return list((await session.scalars(statement)).all())

        return await retry_transient_db(
            _attempt,
            label="get_children_page",
            run_id=parent_ids[0],
            classify=self.classify_db_error,
        )

    async def get_loop_state(self, run_id: str) -> dict[str, Any] | None:
        """Read the authoritative execution projection."""
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        async with self._session_factory() as session:
            value = await session.scalar(select(AgentRun.loop_state).where(AgentRun.id == run_id))
            return cast("dict[str, Any] | None", value)

    async def get_trace(
        self, run_id: str, trace_id: str, *, tenant_id: TenantScope = UNSCOPED
    ) -> TraceRecord | None:
        """Read a referenced state trace by id, including superseded prefixes."""
        from sqlalchemy import select

        from dendrux.db.models import ReactTrace

        async with self._session_factory() as session:
            row = await session.scalar(
                select(ReactTrace).where(
                    ReactTrace.agent_run_id == run_id,
                    ReactTrace.id == trace_id,
                    _run_in_scope(run_id, tenant_id),
                )
            )
            return _trace_record(row) if row is not None else None

    async def get_traces_after(
        self,
        run_id: str,
        *,
        after_order_index: int | None,
        limit: int = 500,
        kinds: tuple[str, ...] = ("seed", "turn", "rewrite", "state"),
        tenant_id: TenantScope = UNSCOPED,
    ) -> list[TraceRecord]:
        """Read a bounded keyset page of physical trace rows."""
        from sqlalchemy import select

        from dendrux.db.models import ReactTrace

        _positive_page(limit)
        async with self._session_factory() as session:
            stmt = select(ReactTrace).where(
                ReactTrace.agent_run_id == run_id,
                ReactTrace.trace_kind.in_(kinds),
                _run_in_scope(run_id, tenant_id),
            )
            if after_order_index is not None:
                stmt = stmt.where(ReactTrace.order_index > after_order_index)
            rows = (
                await session.scalars(stmt.order_by(ReactTrace.order_index).limit(min(limit, 1000)))
            ).all()
            return [_trace_record(row) for row in rows]

    async def conversation_traces(
        self,
        run_id: str,
        *,
        after_position: int | None,
        limit: int = 500,
        include_superseded: bool = False,
        tenant_id: TenantScope = UNSCOPED,
    ) -> ConversationPage:
        """Project the latest replacement prefix followed by its conversation suffix."""
        from sqlalchemy import select

        from dendrux.db.models import AgentRun, ReactTrace

        _positive_page(limit)
        limit = min(limit, 1000)
        after = -1 if after_position is None else after_position
        entries: list[ConversationEntry] = []
        async with self._session_factory() as session:
            base = select(ReactTrace).where(
                ReactTrace.agent_run_id == run_id, _run_in_scope(run_id, tenant_id)
            )
            rewrite = None
            rewrite_order = None
            if not include_superseded:
                pointer = await session.scalar(
                    select(AgentRun.meta["dendrux.latest_rewrite"]).where(
                        AgentRun.id == run_id,
                        _run_in_scope(run_id, tenant_id),
                    )
                )
                if isinstance(pointer, dict):
                    rewrite_order = pointer["order_index"]
                    if after < rewrite_order * 1000 + 999:
                        rewrite = await session.scalar(
                            base.where(ReactTrace.id == pointer["trace_id"])
                        )
            if rewrite is not None:
                for index, message in enumerate(json.loads(rewrite.content), 1):
                    position = rewrite.order_index * 1000 + index
                    if position > after:
                        entries.append(
                            ConversationEntry(
                                position,
                                message["role"],
                                message["content"],
                                {
                                    **(rewrite.meta or {}),
                                    **(message.get("meta") or {}),
                                    "expanded_index": index,
                                },
                                "rewrite",
                                rewrite.id,
                            )
                        )
                        if len(entries) > limit:
                            break
            if len(entries) <= limit:
                kinds = ("seed", "turn", "rewrite") if include_superseded else ("seed", "turn")
                stmt = base.where(
                    ReactTrace.trace_kind.in_(kinds), ReactTrace.order_index > after // 1000
                )
                if rewrite_order is not None:
                    stmt = stmt.where(ReactTrace.order_index > rewrite_order)
                rows = (
                    await session.scalars(
                        stmt.order_by(ReactTrace.order_index).limit(limit + 1 - len(entries))
                    )
                ).all()
                entries.extend(
                    ConversationEntry(
                        row.order_index * 1000,
                        row.role,
                        row.content,
                        row.meta or {},
                        row.trace_kind,
                        row.id,
                    )
                    for row in rows
                )
        more = len(entries) > limit
        entries = entries[:limit]
        return ConversationPage(entries, entries[-1].position if more else None)

    async def get_llm_interactions(
        self,
        run_id: str,
        *,
        iteration_index: int | None = None,
        limit: int | None = None,
        offset: int = 0,
        before: datetime | None = None,
        tenant_id: TenantScope = UNSCOPED,
    ) -> list[LLMInteractionRecord]:
        """LLM interactions of a run; empty when the run is outside ``tenant_id``."""
        from sqlalchemy import select

        _require_aware_utc("before", before)

        from dendrux.db.models import LLMInteraction

        async with self._session_factory() as session:
            stmt = (
                select(LLMInteraction)
                .where(LLMInteraction.agent_run_id == run_id, _run_in_scope(run_id, tenant_id))
                .order_by(LLMInteraction.iteration_index, LLMInteraction.id)
            )
            if iteration_index is not None:
                stmt = stmt.where(LLMInteraction.iteration_index == iteration_index)
            if before is not None:
                stmt = stmt.where(LLMInteraction.created_at < _to_aware_utc(before))
            if limit is not None:
                stmt = stmt.limit(max(1, min(limit, 1000)))
            if offset:
                stmt = stmt.offset(max(0, offset))
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                LLMInteractionRecord(
                    id=r.id,
                    iteration_index=r.iteration_index,
                    model=r.model,
                    provider=r.provider,
                    semantic_request=r.semantic_request,
                    semantic_response=r.semantic_response,
                    provider_request=r.provider_request,
                    provider_response=r.provider_response,
                    input_tokens=r.input_tokens,
                    output_tokens=r.output_tokens,
                    cost_usd=float(r.cost_usd) if r.cost_usd is not None else None,
                    duration_ms=r.duration_ms,
                    cache_read_input_tokens=r.cache_read_input_tokens,
                    cache_creation_input_tokens=r.cache_creation_input_tokens,
                    reasoning_tokens=r.reasoning_tokens,
                    created_at=_to_aware_utc(r.created_at),
                )
                for r in rows
            ]

    async def finalize_run(
        self,
        run_id: str,
        *,
        status: str,
        answer: str | None = None,
        error: str | None = None,
        iteration_count: int | None = None,
        total_usage: UsageStats | None = None,
        expected_current_status: str | None = None,
        pii_mapping: dict[str, str] | None = None,
        claim_epoch: int | None = None,
        error_info: dict[str, Any] | None = None,
        failure_reason: str | None = None,
    ) -> bool:
        """Finalize a run. Returns True if the update was applied.

        Args:
            expected_current_status: If provided, only updates the row if the
                current DB status matches. Returns False if status has changed
                (e.g. already cancelled). This prevents cancel/finalize races.
            pii_mapping: Final guardrail PII mapping. Persists past finalize
                (audit-first — framework never auto-clears).
        """
        from sqlalchemy import update

        from dendrux.db.models import AgentRun

        async def _attempt() -> bool:
            async with self._session_factory() as session:
                # updated_at is filled by the model's onupdate hook (Python aware-UTC).
                values: dict[str, Any] = {
                    **_status_values(status),
                }
                if iteration_count is not None:
                    values["iteration_count"] = iteration_count
                if answer is not None or total_usage is not None:
                    values["output_data"] = await _finalized_output_data(
                        session, run_id, answer, total_usage
                    )
                if error is not None:
                    values["error"] = error
                if total_usage is not None:
                    values.update(_usage_totals(total_usage))
                if pii_mapping is not None:
                    values["pii_mapping"] = pii_mapping

                if failure_reason is not None:
                    values["failure_reason"] = failure_reason
                # Clear pause_data on finalize (D1: execution state cleaned up)
                values["pause_data"] = None
                # Clear cooperative cancel flag — terminal state is final;
                # a stale True here would confuse later audits.
                values["cancel_requested"] = False
                values["pause_requested"] = False

                # Conditional finalize: only update if status is still 'running'
                # (or expected_status if provided). Prevents cancel/finalize races.
                if expected_current_status is not None:
                    stmt = (
                        update(AgentRun)
                        .where(AgentRun.id == run_id, AgentRun.status == expected_current_status)
                        .ordered_values(*self._ordered(AgentRun.__table__, dict(**values)))
                    )
                else:
                    stmt = (
                        update(AgentRun)
                        .where(AgentRun.id == run_id)
                        .ordered_values(*self._ordered(AgentRun.__table__, dict(**values)))
                    )
                if claim_epoch is not None:
                    stmt = stmt.where(AgentRun.claim_epoch == claim_epoch)
                self._remember_run_values(session, run_id, values)
                result = await session.execute(stmt)
                if result.rowcount and error_info is not None:
                    from sqlalchemy import select

                    old_meta = (
                        await session.execute(select(AgentRun.meta).where(AgentRun.id == run_id))
                    ).scalar_one_or_none()
                    await session.execute(
                        update(AgentRun)
                        .where(AgentRun.id == run_id)
                        .values(meta={**(old_meta or {}), "dendrux.error_info": error_info})
                    )
                await self._write_tx(session)
                return (
                    bool(result.rowcount and result.rowcount > 0)
                    if expected_current_status or claim_epoch is not None
                    else True
                )

        return await retry_transient_db(
            _attempt, label="finalize_run", run_id=run_id, classify=self.classify_db_error
        )

    async def finalize_run_if_status_in(
        self,
        run_id: str,
        *,
        status: str,
        allowed_current_statuses: list[str],
        answer: str | None = None,
        error: str | None = None,
        iteration_count: int | None = None,
        total_usage: UsageStats | None = None,
        pii_mapping: dict[str, str] | None = None,
        failure_reason: str | None = None,
        claim_epoch: int | None = None,
    ) -> bool:
        """Atomic CAS finalize over multiple acceptable current statuses.

        One SQL ``UPDATE ... WHERE status IN (...)`` — collapses what
        used to be N sequential CAS attempts into a single round-trip.
        Mirrors :meth:`finalize_run`'s side-effects: clears ``pause_data``
        and ``cancel_requested``.
        """

        async def _attempt() -> bool:
            async with self._session_factory() as session:
                won = await self._cas_finalize(
                    session,
                    run_id,
                    status=status,
                    allowed_current_statuses=allowed_current_statuses,
                    answer=answer,
                    error=error,
                    iteration_count=iteration_count,
                    total_usage=total_usage,
                    pii_mapping=pii_mapping,
                    failure_reason=failure_reason,
                    claim_epoch=claim_epoch,
                )
                await self._write_tx(session)
                return won

        return await retry_transient_db(
            _attempt,
            label="finalize_run_if_status_in",
            run_id=run_id,
            classify=self.classify_db_error,
        )

    async def finalize_with_event(
        self,
        run_id: str,
        *,
        status: str,
        allowed_current_statuses: list[str],
        event_type: str,
        event_data: dict[str, Any] | None = None,
        answer: str | None = None,
        error: str | None = None,
        iteration_count: int | None = None,
        total_usage: UsageStats | None = None,
        pii_mapping: dict[str, str] | None = None,
        failure_reason: str | None = None,
        claim_epoch: int | None = None,
        trace: TraceWrite | None = None,
        event: EventWrite | None = None,
    ) -> int | None:
        """:meth:`finalize_run_if_status_in` plus its lifecycle event, in one transaction.

        The event's sequence index is allocated after the CAS took the row
        lock, so nothing can observe the new status before the event is
        durable and no two transition events share an index. Returns the
        event's sequence index, or None when the CAS lost (nothing written).
        """

        async def _attempt() -> int | None:
            async with self._session_factory() as session:
                won = await self._cas_finalize(
                    session,
                    run_id,
                    status=status,
                    allowed_current_statuses=allowed_current_statuses,
                    answer=answer,
                    error=error,
                    iteration_count=iteration_count,
                    total_usage=total_usage,
                    pii_mapping=pii_mapping,
                    failure_reason=failure_reason,
                    claim_epoch=claim_epoch,
                )
                if not won:
                    await session.rollback()
                    return None
                if trace is not None:
                    await self._insert_trace(session, run_id, trace)
                seq = (
                    await _write_event(session, run_id, event)
                    if event is not None
                    else await _insert_event_locked(session, run_id, event_type, event_data)
                )
                await self._write_tx(session)
                return seq

        return await retry_transient_db(
            _attempt, label="finalize_with_event", run_id=run_id, classify=self.classify_db_error
        )

    async def _cas_finalize(
        self,
        session: AsyncSession,
        run_id: str,
        *,
        status: str,
        allowed_current_statuses: list[str],
        answer: str | None,
        error: str | None,
        iteration_count: int | None,
        total_usage: UsageStats | None,
        pii_mapping: dict[str, str] | None,
        failure_reason: str | None = None,
        claim_epoch: int | None = None,
    ) -> bool:
        from sqlalchemy import case, update

        from dendrux.db.models import AgentRun

        # updated_at is filled by the model's onupdate hook (Python aware-UTC).
        values: dict[str, Any] = {
            **_status_values(status),
            "pause_data": None,
            "cancel_requested": False,
            "pause_requested": False,
        }
        finalized_state = self.json_patch_expr(
            AgentRun.loop_state, SetFields({"phase": "finalizing"})
        )
        if finalized_state is None:
            current_state = await session.scalar(
                self.lock_clause(select(AgentRun.loop_state).where(AgentRun.id == run_id), "update")
            )
            finalized_state = {**(current_state or {}), "phase": "finalizing"}
        values["loop_state"] = case(
            (AgentRun.storage_format == 2, finalized_state), else_=AgentRun.loop_state
        )
        if iteration_count is not None:
            values["iteration_count"] = iteration_count
        if answer is not None or total_usage is not None:
            values["output_data"] = await _finalized_output_data(
                session, run_id, answer, total_usage
            )
        if error is not None:
            values["error"] = error
        if total_usage is not None:
            values.update(_usage_totals(total_usage))
        if pii_mapping is not None:
            values["pii_mapping"] = pii_mapping
        if failure_reason is not None:
            values["failure_reason"] = failure_reason

        self._remember_run_values(session, run_id, values)
        stmt = update(AgentRun).where(
            AgentRun.id == run_id, AgentRun.status.in_(allowed_current_statuses)
        )
        if claim_epoch is not None:
            stmt = stmt.where(AgentRun.claim_epoch == claim_epoch)
        capture = _finalized_run_read.get()
        if capture is not None and capture[0] is self and capture[1] == run_id:
            row = (
                await self._update_one_rows(
                    session,
                    table=AgentRun.__table__,
                    pk=(run_id,),
                    predicate=and_(
                        AgentRun.status.in_(allowed_current_statuses),
                        AgentRun.claim_epoch == claim_epoch if claim_epoch is not None else true(),
                    ),
                    values=values,
                    returning=[
                        AgentRun.id,
                        AgentRun.tenant_id,
                        AgentRun.reclaim_attempts,
                        AgentRun.last_reclaimed_at,
                    ],
                )
            ).first()
            if row is not None:
                capture[2][:] = [
                    _FinalizedRunRead(
                        row.id,
                        row.tenant_id,
                        row.reclaim_attempts,
                        _to_aware_utc(row.last_reclaimed_at) if row.last_reclaimed_at else None,
                    )
                ]
            return row is not None
        result: Any = await session.execute(
            stmt.ordered_values(*self._ordered(AgentRun.__table__, dict(**values)))
        )
        return bool(result.rowcount and result.rowcount > 0)

    async def request_cancel(self, run_id: str) -> bool:
        """Set ``cancel_requested=True`` on a non-terminal run.

        The terminal-status guard is part of the SQL UPDATE — no race
        window between read and write.
        """
        from sqlalchemy import update

        from dendrux.db.models import AgentRun

        async def _attempt() -> bool:
            async with self._session_factory() as session:
                # updated_at is filled by the model's onupdate hook (Python aware-UTC).
                stmt = (
                    update(AgentRun)
                    .where(
                        AgentRun.id == run_id,
                        AgentRun.status.notin_(_TERMINAL_STATUS_VALUES),
                    )
                    .values(cancel_requested=True)
                )
                result = await session.execute(stmt)
                await self._write_tx(session)
                return bool(result.rowcount and result.rowcount > 0)

        return await retry_transient_db(
            _attempt, label="request_cancel", run_id=run_id, classify=self.classify_db_error
        )

    async def update_run_meta(
        self, run_id: str, patch: dict[str, Any], *, claim_epoch: int | None = None
    ) -> None:
        """Shallow-merge ``patch`` into ``agent_runs.meta`` for ``run_id``.

        Read-modify-write in one transaction; only Dendrux-owned keys
        (``dendrux.*``) are written through here. No-op for missing runs.
        """
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        async def _attempt() -> None:
            async with self._session_factory() as session:
                if claim_epoch is not None:
                    await self._fence(session, run_id, claim_epoch)
                elif not self.supports_row_locks:
                    await self.reserve_parent_writer(session, run_id)
                row = (
                    await session.execute(
                        self.lock_clause(select(AgentRun).where(AgentRun.id == run_id), "update")
                    )
                ).scalar_one_or_none()
                if row is None:
                    return
                row.meta = {**(row.meta or {}), **patch}
                await self._write_tx(session)

        await retry_transient_db(
            _attempt, label="update_run_meta", run_id=run_id, classify=self.classify_db_error
        )

    async def is_cancel_requested(self, run_id: str) -> bool:
        """Read ``cancel_requested`` flag. False for missing runs."""
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        async def _attempt() -> bool:
            async with self._session_factory() as session:
                stmt = select(AgentRun.cancel_requested).where(AgentRun.id == run_id)
                result = await session.execute(stmt)
                row = result.first()
                return bool(row[0]) if row is not None else False

        return await retry_transient_db(
            _attempt, label="is_cancel_requested", run_id=run_id, classify=self.classify_db_error
        )

    async def get_run_control(self, run_id: str) -> RunControl:
        """One read of cancel / pause flags plus an EXISTS over undelivered notes."""
        from sqlalchemy import literal, select

        from dendrux.db.models import AgentRun, RunInbox

        pending = (
            select(literal(1))
            .where(RunInbox.run_id == AgentRun.id, RunInbox.delivered_at.is_(None))
            .exists()
        )

        async def _attempt() -> RunControl:
            async with self._session_factory() as session:
                stmt = select(
                    AgentRun.cancel_requested,
                    AgentRun.pause_requested,
                    pending,
                    AgentRun.claim_epoch,
                    AgentRun.status,
                ).where(AgentRun.id == run_id)
                row = (await session.execute(stmt)).first()
                if row is None:
                    return RunControl()
                return RunControl(
                    cancel_requested=bool(row[0]),
                    pause_requested=bool(row[1]),
                    has_pending_notes=bool(row[2]),
                    claim_epoch=row[3],
                    status=row[4],
                )

        return await retry_transient_db(
            _attempt, label="get_run_control", run_id=run_id, classify=self.classify_db_error
        )

    async def request_pause(self, run_id: str, *, reason: str | None = None) -> bool:
        """Set ``pause_requested`` on a RUNNING row; status guard in the UPDATE."""
        from sqlalchemy import select, update

        from dendrux.db.models import AgentRun

        async def _attempt() -> bool:
            async with self._session_factory() as session:
                result = await session.execute(
                    update(AgentRun)
                    .where(AgentRun.id == run_id, AgentRun.status == RunStatus.RUNNING.value)
                    .values(pause_requested=True)
                )
                if not result.rowcount:
                    await session.rollback()
                    return False
                # The UPDATE holds the row lock; record the reason (or clear a
                # stale one) under it.
                meta = (
                    await session.execute(select(AgentRun.meta).where(AgentRun.id == run_id))
                ).scalar_one_or_none() or {}
                if reason is not None or PAUSE_REQUEST_META_KEY in meta:
                    await session.execute(
                        update(AgentRun)
                        .where(AgentRun.id == run_id)
                        .values(meta={**meta, PAUSE_REQUEST_META_KEY: {"reason": reason}})
                    )
                await self._write_tx(session)
                return True

        return await retry_transient_db(
            _attempt, label="request_pause", run_id=run_id, classify=self.classify_db_error
        )

    async def try_pause_run(
        self,
        run_id: str,
        *,
        status: str,
        pause_data: dict[str, Any],
        iteration_count: int | None = None,
        pii_mapping: dict[str, str] | None = None,
        total_usage: UsageStats | None = None,
        claim_epoch: int | None = None,
    ) -> bool:
        """CAS pause: RUNNING and no pending cancel, else False (caller honors winner)."""
        from sqlalchemy import update

        from dendrux.db.models import AgentRun

        async def _attempt() -> bool:
            async with self._session_factory() as session:
                values: dict[str, Any] = {
                    **_status_values(status),
                    "pause_data": pause_data,
                    "pause_requested": False,
                }
                if iteration_count is not None:
                    values["iteration_count"] = iteration_count
                if pii_mapping is not None:
                    values["pii_mapping"] = pii_mapping
                if total_usage is not None:
                    values.update(_usage_totals(total_usage))
                self._remember_run_values(session, run_id, values)
                result = await session.execute(
                    update(AgentRun)
                    .where(
                        AgentRun.id == run_id,
                        AgentRun.status == RunStatus.RUNNING.value,
                        AgentRun.cancel_requested.is_(False),
                        *([AgentRun.claim_epoch == claim_epoch] if claim_epoch is not None else []),
                    )
                    .ordered_values(*self._ordered(AgentRun.__table__, dict(**values)))
                )
                await self._write_tx(session)
                return bool(result.rowcount and result.rowcount > 0)

        return await retry_transient_db(
            _attempt, label="try_pause_run", run_id=run_id, classify=self.classify_db_error
        )

    async def pause_with_event(
        self,
        run_id: str,
        *,
        status: str,
        pause_data: dict[str, Any],
        event_data: dict[str, Any] | None = None,
        iteration_count: int | None = None,
        pii_mapping: dict[str, str] | None = None,
        total_usage: UsageStats | None = None,
        claim_epoch: int | None = None,
    ) -> int | None:
        """:meth:`try_pause_run` plus the ``run.paused`` event, in one transaction.

        A finalizer of paused runs (``cancel_run``, the sweeper) can only act
        once this commits, so it always sees ``run.paused`` and allocates a
        later sequence index. Returns the event's sequence index, or None
        when the CAS lost (a pending cancel or a non-RUNNING row).
        """
        from sqlalchemy import update

        from dendrux.db.models import AgentRun

        async def _attempt() -> int | None:
            async with self._session_factory() as session:
                values: dict[str, Any] = {
                    **_status_values(status),
                    "pause_data": pause_data,
                    "pause_requested": False,
                }
                if iteration_count is not None:
                    values["iteration_count"] = iteration_count
                if pii_mapping is not None:
                    values["pii_mapping"] = pii_mapping
                if total_usage is not None:
                    values.update(_usage_totals(total_usage))
                self._remember_run_values(session, run_id, values)
                result = await session.execute(
                    update(AgentRun)
                    .where(
                        AgentRun.id == run_id,
                        AgentRun.status == RunStatus.RUNNING.value,
                        AgentRun.cancel_requested.is_(False),
                        *([AgentRun.claim_epoch == claim_epoch] if claim_epoch is not None else []),
                    )
                    .ordered_values(*self._ordered(AgentRun.__table__, dict(**values)))
                )
                if not result.rowcount:
                    await session.rollback()
                    return None
                seq = await _insert_event_locked(
                    session,
                    run_id,
                    "run.paused",
                    event_data,
                    iteration_index=iteration_count or 0,
                )
                await self._write_tx(session)
                return seq

        return await retry_transient_db(
            _attempt, label="pause_with_event", run_id=run_id, classify=self.classify_db_error
        )

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
            JSON,
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
        from dendrux.errors import NoteLimitExceededError

        if max_notes is not None and (isinstance(max_notes, bool) or max_notes < 1):
            raise ValueError("max_notes must be a positive int or None.")

        def _note_count() -> Any:
            return select(func.count()).select_from(RunInbox).where(RunInbox.run_id == run_id)

        async def _attempt() -> bool:
            async with self._session_factory() as session:
                if max_notes is not None:
                    await self.serialize_in_tx(session, owner=RunLockScope(run_id, tenant_id))
                if claim_epoch is not None:
                    await self._fence(session, run_id, claim_epoch)
                source = select(
                    literal(note_id, String(26)),
                    AgentRun.id,
                    literal(text, Text()),
                    null() if metadata is None else literal(metadata, JSON()),
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
                if not inserted and max_notes is not None:
                    # Still under the lock: tell "full" from missing/terminal/foreign.
                    status = await session.scalar(
                        _apply_tenant_scope(
                            select(AgentRun.status).where(AgentRun.id == run_id), tenant_id
                        )
                    )
                    full = (await session.scalar(_note_count()) or 0) >= max_notes
                    if status is not None and status not in _TERMINAL_STATUS_VALUES and full:
                        await session.rollback()
                        _codes.log_code(logger, _codes.NOTE_LIMIT, run_id=run_id, limit=max_notes)
                        raise NoteLimitExceededError(run_id, limit=max_notes)
                await self._write_tx(session)
                return inserted

        return await retry_transient_db(
            _attempt, label="add_note", run_id=run_id, classify=self.classify_db_error
        )

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
            _attempt, label="mark_notes_delivered", run_id=run_id, classify=self.classify_db_error
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
        )

    async def pause_run(
        self,
        run_id: str,
        *,
        status: str,
        pause_data: dict[str, Any],
        iteration_count: int | None = None,
        pii_mapping: dict[str, str] | None = None,
        total_usage: UsageStats | None = None,
        claim_epoch: int | None = None,
    ) -> None:
        """Persist pause state and set the paused status.

        ``total_usage`` writes the run's running totals (``total_*``).
        """
        from sqlalchemy import update

        from dendrux.db.models import AgentRun

        async def _attempt() -> None:
            async with self._session_factory() as session:
                # updated_at is filled by the model's onupdate hook (Python aware-UTC).
                values: dict[str, Any] = {
                    **_status_values(status),
                    "pause_data": pause_data,
                }
                if iteration_count is not None:
                    values["iteration_count"] = iteration_count
                if pii_mapping is not None:
                    values["pii_mapping"] = pii_mapping
                if total_usage is not None:
                    values.update(_usage_totals(total_usage))
                stmt = (
                    update(AgentRun)
                    .where(AgentRun.id == run_id)
                    .ordered_values(*self._ordered(AgentRun.__table__, dict(**values)))
                )
                if claim_epoch is not None:
                    stmt = stmt.where(
                        AgentRun.claim_epoch == claim_epoch, AgentRun.status == "running"
                    )
                self._remember_run_values(session, run_id, values)
                await session.execute(stmt)
                await self._write_tx(session)

        await retry_transient_db(
            _attempt, label="pause_run", run_id=run_id, classify=self.classify_db_error
        )

    async def get_pause_state(self, run_id: str) -> dict[str, Any] | None:
        """Retrieve pause_data for a run. Returns None if no run or no pause data."""
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        async with self._session_factory() as session:
            stmt = select(AgentRun.pause_data).where(AgentRun.id == run_id)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            # row is None if run doesn't exist OR if pause_data column is NULL
            if row is None:
                return None
            return dict(row)  # type narrowing for mypy

    async def get_run_status(self, run_id: str, *, tenant_id: TenantScope = UNSCOPED) -> str | None:
        """Status-only projection for polling; never loads execution history."""
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        async with self._session_factory() as session:
            value = (
                await session.execute(
                    _apply_tenant_scope(
                        select(AgentRun.status).where(AgentRun.id == run_id), tenant_id
                    )
                )
            ).scalar_one_or_none()
            return None if value is None else str(value)

    async def get_active_pause_sequence(
        self, run_id: str, *, tenant_id: TenantScope = UNSCOPED
    ) -> int | None:
        from sqlalchemy import func, select

        from dendrux.db.models import AgentRun, RunEvent

        seq = (
            select(func.max(RunEvent.sequence_index))
            .where(RunEvent.agent_run_id == run_id, RunEvent.event_type == "run.paused")
            .scalar_subquery()
        )
        invalidated = (
            select(func.coalesce(func.max(RunEvent.sequence_index), -1))
            .where(
                RunEvent.agent_run_id == run_id,
                RunEvent.event_type.in_(
                    ["run.resumed", "run.completed", "run.error", "run.cancelled"]
                ),
            )
            .scalar_subquery()
        )
        async with self._session_factory() as session:
            value = (
                await session.execute(
                    _apply_tenant_scope(
                        select(seq).where(
                            AgentRun.id == run_id,
                            seq > invalidated,
                            AgentRun.status.in_(
                                [
                                    "waiting_client_tool",
                                    "waiting_human_input",
                                    "waiting_approval",
                                    "paused",
                                ]
                            ),
                        ),
                        tenant_id,
                    )
                )
            ).scalar_one_or_none()
            return None if value is None else int(value)

    async def get_approval_params(
        self, run_id: str, pause_sequence: int, *, tenant_id: TenantScope = UNSCOPED
    ) -> dict[str, dict[str, Any]]:
        """Latest recorded model args per call in this pause's event interval.

        Read at most 1000 approval requests. Missing/overflowed calls fail closed
        in the display layer when PII mapping exists.
        """
        from sqlalchemy import func, select

        from dendrux.db.models import RunEvent

        boundary = (
            select(func.coalesce(func.max(RunEvent.sequence_index), -1))
            .where(
                RunEvent.agent_run_id == run_id,
                RunEvent.event_type.in_(["run.paused", "run.resumed"]),
                RunEvent.sequence_index < pause_sequence,
            )
            .scalar_subquery()
        )
        stmt = (
            select(RunEvent.data)
            .where(
                RunEvent.agent_run_id == run_id,
                _run_in_scope(run_id, tenant_id),
                RunEvent.event_type == "approval.requested",
                RunEvent.sequence_index > boundary,
                RunEvent.sequence_index < pause_sequence,
            )
            .order_by(RunEvent.sequence_index.desc())
            .limit(1000)
        )
        async with self._session_factory() as session:
            rows = (await session.execute(stmt)).scalars().all()
        result: dict[str, dict[str, Any]] = {}
        seen: set[str] = set()
        for data in rows:
            data = _approval_event_object(data)
            call_id = data.get("call_id")
            if isinstance(call_id, str) and call_id not in seen:
                seen.add(call_id)
                if isinstance(data.get("params"), dict):
                    result[call_id] = data["params"]
        return result

    async def get_pause_summary(
        self, run_id: str, *, tenant_id: TenantScope = UNSCOPED
    ) -> PauseSummary | None:
        """Read bounded pause fields, including legacy plain-JSON documents."""
        return await self._read_pause_summary(run_id, tenant_id, pending=False)

    async def get_pending_pause_summary(
        self, run_id: str, *, tenant_id: TenantScope = UNSCOPED
    ) -> PauseSummary | None:
        """Read pending fields using the adapter's bounded query plan."""
        return await self._read_pause_summary(run_id, tenant_id, pending=True)

    def pause_summary_query(self, run_id: str, tenant_scope: TenantScope, *, pending: bool) -> Any:
        from sqlalchemy import func, select

        from dendrux.db.models import AgentRun, RunEvent

        paused_seq = (
            select(func.max(RunEvent.sequence_index))
            .where(RunEvent.agent_run_id == run_id, RunEvent.event_type == "run.paused")
            .scalar_subquery()
        )
        parts = self.pause_projection(AgentRun.pause_data)
        return _apply_tenant_scope(
            select(AgentRun.status, paused_seq, AgentRun.pii_mapping, *parts).where(
                AgentRun.id == run_id
            ),
            tenant_scope,
        )

    async def _read_pause_summary(
        self, run_id: str, tenant_id: TenantScope, *, pending: bool
    ) -> PauseSummary | None:
        from sqlalchemy.exc import DBAPIError

        stmt = self.pause_summary_query(run_id, tenant_id, pending=pending)
        try:
            async with self._session_factory() as session:
                row = (await session.execute(stmt)).first()
        except DBAPIError as exc:
            if not self.json_conversion_error(exc):
                raise
            from dendrux.runtime._pending import _whole_pause_summary

            _codes.log_code(logger, _codes.PAUSE_DATA_NOT_JSONB, run_id=run_id)
            return await _whole_pause_summary(self, run_id, tenant_id)
        if row is None:
            return None
        status, seq, mapping, *values = row
        mapping = dict(mapping) if isinstance(mapping, dict) else None
        decoded = list(values)
        pause = {
            k: v for k, v in zip(PAUSE_SUMMARY_KEYS, decoded[:-1], strict=True) if v is not None
        }
        last = decoded[-1]
        last_step = last if isinstance(last, dict) else None
        return PauseSummary(str(status), pause, last_step, seq, mapping)

    async def get_pii_mapping(
        self, run_id: str, *, tenant_id: TenantScope = UNSCOPED
    ) -> dict[str, str] | None:
        """Retrieve pii_mapping for a run. None if not set or outside ``tenant_id``."""
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        async with self._session_factory() as session:
            stmt = _apply_tenant_scope(
                select(AgentRun.pii_mapping).where(AgentRun.id == run_id), tenant_id
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return dict(row)

    async def claim_paused_run(
        self,
        run_id: str,
        *,
        expected_status: str,
        expected_pause_id: str | None = None,
        validate: Callable[[dict[str, Any]], None] | None = None,
    ) -> ClaimResult | Literal[False]:
        """Atomically transition a paused run to RUNNING.

        Returns the new claim epoch if the claim succeeded.
        Returns False if someone else already claimed it or status didn't match.
        Uses UPDATE ... WHERE status=? for atomicity — no race window.
        The claim clears ``pause_requested`` in the same statement, so a
        resumed PAUSED run does not immediately re-pause on a stale flag.
        """
        from sqlalchemy import select, update

        from dendrux.db.models import AgentRun

        async def _attempt() -> ClaimResult | Literal[False]:
            async with self._session_factory() as session:
                if not self.supports_update_returning:

                    def validate_locked(row: Any) -> None:
                        pause = row.get("pause_data") or {}
                        if (
                            expected_pause_id is not None
                            and pause.get("pause_id") != expected_pause_id
                        ):
                            from dendrux.errors import PauseStatusMismatchError

                            _codes.log_code(logger, _codes.PAUSE_ID_MISMATCH, run_id=run_id)
                            raise PauseStatusMismatchError(
                                run_id,
                                RunStatus(expected_status),
                                RunStatus(expected_status),
                                expected_pause_id=expected_pause_id,
                                current_pause_id=pause.get("pause_id"),
                                reason="pause_id",
                            )
                        if validate is not None:
                            validate(pause)

                    row = await self.update_one(
                        session,
                        table=AgentRun.__table__,
                        pk=(run_id,),
                        predicate=AgentRun.status == expected_status,
                        values=self._ordered(
                            AgentRun.__table__,
                            {
                                **_status_values("running"),
                                "last_progress_at": _dt.datetime.now(_dt.UTC),
                                "pause_requested": False,
                                "claim_epoch": AgentRun.claim_epoch + 1,
                            },
                        ),
                        returning=[
                            AgentRun.claim_epoch,
                            AgentRun.storage_format,
                            AgentRun.loop_state,
                        ],
                        lock_projection=[AgentRun.__table__.c.pause_data]
                        if validate is not None or expected_pause_id is not None
                        else (),
                        validate_locked=validate_locked
                        if validate is not None or expected_pause_id is not None
                        else None,
                    )
                    await self._write_tx(session)
                    return (
                        False
                        if row is None
                        else ClaimResult(
                            row["claim_epoch"],
                            row["storage_format"],
                            row["loop_state"],
                            (row["loop_state"] or {}).get("submission"),
                        )
                    )
                # updated_at is filled by the model's onupdate hook (Python aware-UTC).
                stmt = (
                    update(AgentRun)
                    .where(AgentRun.id == run_id, AgentRun.status == expected_status)
                    .ordered_values(
                        *self._ordered(
                            AgentRun.__table__,
                            dict(
                                **_status_values("running"),
                                last_progress_at=_dt.datetime.now(_dt.UTC),
                                pause_requested=False,
                                claim_epoch=AgentRun.claim_epoch + 1,
                            ),
                        )
                    )
                )
                result = await session.execute(stmt)
                if result.rowcount:
                    row = (
                        await session.execute(
                            select(
                                AgentRun.claim_epoch, AgentRun.storage_format, AgentRun.loop_state
                            ).where(AgentRun.id == run_id)
                        )
                    ).one()
                    token = ClaimResult(
                        row.claim_epoch,
                        row.storage_format,
                        row.loop_state,
                        (row.loop_state or {}).get("submission"),
                    )
                if result.rowcount and (validate is not None or expected_pause_id is not None):
                    pause = (
                        await session.execute(
                            select(AgentRun.pause_data).where(AgentRun.id == run_id)
                        )
                    ).scalar_one_or_none()
                    try:
                        if expected_pause_id is not None:
                            from dendrux.errors import PauseStatusMismatchError

                            current_id = (pause or {}).get("pause_id")
                            if current_id != expected_pause_id:
                                _codes.log_code(logger, _codes.PAUSE_ID_MISMATCH, run_id=run_id)
                                raise PauseStatusMismatchError(
                                    run_id,
                                    RunStatus(expected_status),
                                    RunStatus(expected_status),
                                    expected_pause_id=expected_pause_id,
                                    current_pause_id=current_id,
                                    reason="pause_id",
                                )
                        if validate is not None:
                            validate(pause or {})
                    except BaseException:
                        await session.rollback()
                        raise
                await self._write_tx(session)
                return token if result.rowcount else False

        return await retry_transient_db(
            _attempt, label="claim_paused_run", run_id=run_id, classify=self.classify_db_error
        )

    async def submit_and_claim(
        self,
        run_id: str,
        *,
        expected_status: str,
        submitted_data: dict[str, Any],
        submission: dict[str, Any] | None = None,
        validate: Callable[[dict[str, Any]], None] | None = None,
        blob_refs: Sequence[BlobEdge] | None = None,
        blob_refs_scope: str | None = None,
    ) -> ClaimResult | Literal[False]:
        """Atomically save submitted data and claim the run.

        Implementation (one transaction):
          1. Conditional UPDATE status -> 'running' WHERE status still equals
             *expected_status*. Losing this CAS returns False. Winning it
             takes the row's write lock (Postgres row lock / SQLite write
             lock), so the pause_data read next cannot change underneath.
          2. Read pause_data and run *validate* on it. A failure rolls the
             whole transaction back (status stays *expected_status*) and
             re-raises.
          3. Merge submitted_data into pause_data.
          4. Write ``blob_refs`` (tenant asserted, blobs claimed) and commit.
             A failure rolls the claim back too.
        """
        from sqlalchemy import select, update

        from dendrux.db.models import AgentRun

        async def _attempt() -> ClaimResult | Literal[False]:
            async with self._session_factory() as session:
                # 1. Claim first — the CAS on status is the concurrency
                # boundary. updated_at is filled by the model's onupdate hook.
                claim = await session.execute(
                    update(AgentRun)
                    .where(AgentRun.id == run_id, AgentRun.status == expected_status)
                    .ordered_values(
                        *self._ordered(
                            AgentRun.__table__,
                            dict(
                                **_status_values("running"),
                                last_progress_at=_dt.datetime.now(_dt.UTC),
                                pause_requested=False,
                                claim_epoch=AgentRun.claim_epoch + 1,
                            ),
                        )
                    )
                )
                if not claim.rowcount:
                    await session.rollback()
                    return False

                # 2. Read the pause being claimed, under the claim's lock.
                run = (
                    await session.execute(
                        select(
                            AgentRun.pause_data, AgentRun.storage_format, AgentRun.loop_state
                        ).where(AgentRun.id == run_id)
                    )
                ).one()
                current_pause_data = run.pause_data
                if current_pause_data is None:
                    await session.rollback()
                    return False
                if validate is not None:
                    try:
                        validate(current_pause_data)
                    except BaseException:
                        await session.rollback()
                        raise

                # 3. Merge submitted_data into pause_data.
                merged = dict(current_pause_data)
                values: dict[str, Any] = {}
                if run.storage_format == 2:
                    if submission is None:
                        from dendrux.errors import PersistenceError

                        _codes.log_code(logger, _codes.SUBMISSION_MISSING, run_id=run_id)
                        raise PersistenceError(code="submission_missing")
                    merged["submission"] = {
                        **{
                            k: submission[k]
                            for k in ("submission_id", "kind", "pause_id", "call_ids")
                        },
                        "payload": dict(submitted_data),
                    }
                    loop_state = dict(run.loop_state or {})
                    loop_state["submission"] = {
                        k: v for k, v in submission.items() if k != "payload"
                    }
                    _validate_checkpoint(run_id, loop_state)
                    values["loop_state"] = loop_state
                else:
                    merged.update(submitted_data)
                await session.execute(
                    update(AgentRun)
                    .where(AgentRun.id == run_id)
                    .ordered_values(
                        *self._ordered(AgentRun.__table__, dict(pause_data=merged, **values))
                    )
                )
                if blob_refs:
                    await write_blob_refs(
                        session,
                        run_id=run_id,
                        blob_refs=blob_refs,
                        blob_refs_scope=blob_refs_scope,
                    )
                row = (
                    await session.execute(
                        select(
                            AgentRun.claim_epoch, AgentRun.storage_format, AgentRun.loop_state
                        ).where(AgentRun.id == run_id)
                    )
                ).one()
                token = ClaimResult(
                    row.claim_epoch,
                    row.storage_format,
                    row.loop_state,
                    (row.loop_state or {}).get("submission"),
                )
                await self._write_tx(session)
                return token

        return await retry_transient_db(
            _attempt, label="submit_and_claim", run_id=run_id, classify=self.classify_db_error
        )

    async def release_run_claim(self, run_id: str, *, status: str, claim_epoch: int) -> bool:
        """Return an owned claim to its pause after post-claim validation fails."""
        from sqlalchemy import update

        from dendrux.db.models import AgentRun

        async def attempt() -> bool:
            async with self._session_factory() as session:
                result = await session.execute(
                    update(AgentRun)
                    .where(
                        AgentRun.id == run_id,
                        AgentRun.status == "running",
                        AgentRun.claim_epoch == claim_epoch,
                    )
                    .ordered_values(
                        *self._ordered(AgentRun.__table__, dict(**_status_values(status)))
                    )
                )
                await self._write_tx(session)
                return bool(result.rowcount)

        return await retry_transient_db(
            attempt, label="release_run_claim", run_id=run_id, classify=self.classify_db_error
        )

    @_safe_database_call
    async def _get_result_record(self, run_id: str, *, tenant_id: TenantScope) -> RunRecord | None:
        """Project one browser-result record without prompts or pause history.

        Private optimization; custom stores keep the existing scoped get_run
        contract. All reads use the same tenant predicate as get_run.
        """
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        stop_key = "dendrux.stop_reason"

        fields = (
            AgentRun.strategy,
            AgentRun.storage_format,
            AgentRun.claim_epoch,
            AgentRun.status_changed_at,
            AgentRun.reclaim_attempts,
            AgentRun.reclaim_limit,
            AgentRun.lease_expired_at,
            AgentRun.last_reclaimed_at,
            AgentRun.trace_link,
            AgentRun.id,
            AgentRun.agent_name,
            AgentRun.status,
            AgentRun.tenant_id,
            AgentRun.iteration_count,
            AgentRun.error,
            AgentRun.cancel_requested,
            AgentRun.total_input_tokens,
            AgentRun.total_output_tokens,
            AgentRun.total_cache_read_tokens,
            AgentRun.total_cache_creation_tokens,
            AgentRun.total_reasoning_tokens,
            AgentRun.total_cost_usd,
            AgentRun.output_data["answer"].as_string().label("answer"),
            AgentRun.meta["dendrux.error_info"].label("error_info"),
            AgentRun.meta[stop_key].as_string().label("stop_reason"),
        )
        stmt = _apply_tenant_scope(select(*fields).where(AgentRun.id == run_id), tenant_id)
        async with self._session_factory() as session:
            row = (await session.execute(stmt)).mappings().first()
        if row is None:
            return None
        values = dict(row)
        for name in ("status_changed_at", "lease_expired_at", "last_reclaimed_at"):
            values[name] = _to_aware_utc(values[name])
        info, stop_reason = values.pop("error_info"), values.pop("stop_reason")
        values["total_cost_usd"] = (
            float(values["total_cost_usd"]) if values["total_cost_usd"] is not None else None
        )
        return RunRecord(
            **values,
            meta={"dendrux.error_info": info, stop_key: stop_reason},
        )

    async def get_run(self, run_id: str, *, tenant_id: TenantScope = UNSCOPED) -> RunRecord | None:
        """The run, or None when it does not exist or is outside ``tenant_id``.

        The tenant predicate is part of the query, so a foreign run reads
        exactly like a missing one.
        """
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        async with self._session_factory() as session:
            stmt = _apply_tenant_scope(select(AgentRun).where(AgentRun.id == run_id), tenant_id)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return _run_to_record(row)

    async def get_next_trace_index(self, run_id: str) -> int:
        """Read the next trace order from the run/order index, without payloads."""
        from sqlalchemy import select

        from dendrux.db.models import ReactTrace

        async with self._session_factory() as session:
            value = await session.scalar(
                select(ReactTrace.order_index)
                .where(ReactTrace.agent_run_id == run_id)
                .order_by(ReactTrace.order_index.desc())
                .limit(1)
            )
            return int(value) + 1 if value is not None else 0

    async def get_traces(
        self,
        run_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
        tenant_id: TenantScope = UNSCOPED,
    ) -> list[TraceRecord]:
        """Trace rows of a run; empty when the run is outside ``tenant_id``."""
        from sqlalchemy import select

        from dendrux.db.models import ReactTrace

        async with self._session_factory() as session:
            stmt = (
                select(ReactTrace)
                .where(ReactTrace.agent_run_id == run_id, _run_in_scope(run_id, tenant_id))
                .order_by(ReactTrace.order_index)
            )
            if limit is not None:
                stmt = stmt.limit(max(1, min(limit, 1000)))
            if offset:
                stmt = stmt.offset(max(0, offset))
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                TraceRecord(
                    id=r.id,
                    trace_kind=r.trace_kind,
                    role=r.role,
                    content=r.content,
                    order_index=r.order_index,
                    meta=r.meta,
                    created_at=_to_aware_utc(r.created_at),
                )
                for r in rows
            ]

    async def get_tool_calls(
        self,
        run_id: str,
        *,
        iteration_index: int | None = None,
        limit: int | None = None,
        offset: int = 0,
        before: datetime | None = None,
        tenant_id: TenantScope = UNSCOPED,
    ) -> list[ToolCallReadRecord]:
        """Tool calls of a run; empty when the run is outside ``tenant_id``."""
        from sqlalchemy import select

        _require_aware_utc("before", before)

        from dendrux.db.models import ToolCallRecord

        async with self._session_factory() as session:
            stmt = (
                select(ToolCallRecord)
                .where(ToolCallRecord.agent_run_id == run_id, _run_in_scope(run_id, tenant_id))
                .order_by(ToolCallRecord.created_at, ToolCallRecord.id)
            )
            if iteration_index is not None:
                stmt = stmt.where(ToolCallRecord.iteration_index == iteration_index)
            if before is not None:
                stmt = stmt.where(ToolCallRecord.created_at < _to_aware_utc(before))
            if limit is not None:
                stmt = stmt.limit(max(1, min(limit, 1000)))
            if offset:
                stmt = stmt.offset(max(0, offset))
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                ToolCallReadRecord(
                    id=r.id,
                    outcome=r.outcome,
                    observation=(r.meta or {}).get("dendrux.observation"),
                    idempotent=r.idempotent,
                    replayed_at=_to_aware_utc(r.replayed_at),
                    tool_call_id=r.tool_call_id,
                    provider_tool_call_id=r.provider_tool_call_id,
                    tool_name=r.tool_name,
                    target=r.target,
                    params=r.params,
                    result=r.result,
                    success=r.success,
                    duration_ms=r.duration_ms,
                    iteration_index=r.iteration_index,
                    error_message=r.error_message,
                    created_at=_to_aware_utc(r.created_at),
                )
                for r in rows
            ]

    async def _after_event_insert(self, session: AsyncSession, run_id: str) -> None:
        return None

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
            await self._write_tx(session)
            return seq

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
                clamped = max(1, min(limit, 2002 if _pause_page else 1000))
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

    # ------------------------------------------------------------------
    # Progress tracking & stale-run sweep
    # ------------------------------------------------------------------

    async def touch_progress(self, run_id: str, *, claim_epoch: int | None = None) -> bool:
        """Renew only a running run with the supplied epoch, in one statement."""
        from sqlalchemy import update

        from dendrux.db.models import AgentRun

        async with self._session_factory() as session:
            stmt = update(AgentRun).where(AgentRun.id == run_id, AgentRun.status == "running")
            if claim_epoch is not None:
                stmt = stmt.where(AgentRun.claim_epoch == claim_epoch)
            result = await session.execute(stmt.values(last_progress_at=_dt.datetime.now(_dt.UTC)))
            await self._write_tx(session)
            return bool(result.rowcount)

    async def mark_lease_expired(
        self,
        *,
        cutoff: datetime,
        batch_size: int,
        max_runs: int,
        max_returned: int,
        reclaim_limit: int,
        cursor: str | None,
        finalise_unrecoverable: bool = False,
    ) -> MarkResult:
        """Scan running rows once by id, atomically fencing each expired lease."""
        return cast(
            "MarkResult",
            await self._scan_expired_runs(
                cutoff=cutoff,
                batch_size=batch_size,
                max_runs=max_runs,
                max_returned=max_returned,
                reclaim_limit=reclaim_limit,
                cursor=cursor,
                on_stale="reclaim" if finalise_unrecoverable else None,
            ),
        )

    @_safe_database_call
    async def _scan_expired_runs(
        self,
        *,
        cutoff: datetime,
        batch_size: int,
        max_runs: int,
        max_returned: int,
        reclaim_limit: int,
        cursor: str | None,
        on_stale: Literal["error", "reclaim"] | None = None,
    ) -> MarkResult:
        from sqlalchemy import case, func, literal_column, or_, select

        from dendrux.db.models import AgentRun, RunEvent

        _require_aware_utc("cutoff", cutoff)
        for bound in (batch_size, max_runs, max_returned, reclaim_limit):
            _positive_page(bound)
        if max(batch_size, max_runs, max_returned) > 100000:
            _codes.log_code(logger, _codes.RECOVERY_SCAN_LIMIT, run_id=None)
            raise ValueError("Recovery scan bounds must not exceed 100000")
        if reclaim_limit > 32767:
            raise ValueError("reclaim_limit exceeds SMALLINT")
        result = MarkResult([], [], [], 0, None)
        inspected, after = 0, cursor
        started = (
            select(RunEvent.id)
            .where(
                RunEvent.agent_run_id == AgentRun.id,
                RunEvent.event_type == literal_column("'run.started'"),
            )
            .limit(1)
            .correlate(AgentRun)
            .scalar_subquery()
        )
        fields = (
            AgentRun.id,
            AgentRun.agent_name,
            AgentRun.tenant_id,
            AgentRun.status,
            AgentRun.last_progress_at,
            AgentRun.created_at,
            AgentRun.storage_format,
            AgentRun.reclaim_attempts,
            AgentRun.reclaim_limit,
            AgentRun.claim_epoch,
            AgentRun.lease_expired_at,
            AgentRun.loop_state["loop"].as_string().label("loop"),
            AgentRun.cancel_requested,
            AgentRun.iteration_count,
            started.label("started"),
        )
        while inspected < max_runs and len(result.reclaimable) < max_returned:
            statement = select(*fields).where(AgentRun.status.between("running", "running"))
            if after is not None:
                statement = statement.where(AgentRun.id > after)
            async with self._session_factory() as session:
                ids = (
                    await session.execute(
                        statement.order_by(AgentRun.status, AgentRun.id).limit(
                            min(batch_size, max_runs - inspected)
                        )
                    )
                ).all()
            if not ids:
                return MarkResult(
                    result.reclaimable, result.finalised, result.legacy, inspected, None
                )
            for candidate in ids:
                run_id = candidate.id

                async def attempt(
                    row: Any = candidate, run_id: str = run_id
                ) -> ReclaimableRun | SweptRun | None:
                    async with self._session_factory() as session:
                        age = _to_aware_utc(row.last_progress_at or row.created_at)
                        if (on_stale == "error" or row.lease_expired_at is None) and age >= cutoff:
                            return None
                        if row.storage_format == 2 and row.loop not in ("react", "single_call"):
                            if on_stale != "error":
                                _codes.log_code(logger, _codes.LOOP_UNKNOWN, run_id=run_id)
                            if on_stale is None:
                                return None
                        limit = (
                            row.reclaim_limit
                            if row.lease_expired_at is not None and row.reclaim_limit is not None
                            else reclaim_limit
                        )
                        legacy_reason = (
                            "stale_running" if row.started or on_stale is None else "never_started"
                        )
                        reason = (
                            legacy_reason
                            if on_stale == "error"
                            else "stale_running_legacy"
                            if row.storage_format != 2
                            else "reclaim_limit"
                            if row.reclaim_attempts >= limit
                            else legacy_reason
                            if row.loop != "react"
                            else None
                        )
                        if reason is not None or row.cancel_requested:
                            flagged = AgentRun.cancel_requested.is_(True)
                            status_expr = case((flagged, "cancelled"), else_="error")
                            changed = (
                                await self._update_one_rows(
                                    session,
                                    table=AgentRun.__table__,
                                    pk=(run_id,),
                                    predicate=and_(
                                        AgentRun.id == run_id,
                                        AgentRun.status == "running",
                                        AgentRun.claim_epoch == row.claim_epoch,
                                        or_(
                                            AgentRun.lease_expired_at.is_not(None)
                                            if on_stale != "error"
                                            else False,
                                            func.coalesce(
                                                AgentRun.last_progress_at, AgentRun.created_at
                                            )
                                            < cutoff,
                                        ),
                                    ),
                                    values=dict(
                                        **_status_values(status_expr),
                                        claim_epoch=AgentRun.claim_epoch + 1,
                                        failure_reason=reason,
                                        error=case((flagged, None), else_=f"Run swept as {reason}"),
                                        pause_data=None,
                                        pause_requested=False,
                                        cancel_requested=False,
                                    ),
                                    returning=[AgentRun.status],
                                )
                            ).scalar_one_or_none()
                            if changed is None:
                                return None
                            status = changed
                            await _insert_event_locked(
                                session,
                                run_id,
                                "run.cancelled" if status == "cancelled" else "run.interrupted",
                                {
                                    "failure_reason": reason,
                                    **(
                                        {"reason": "cancel_requested"}
                                        if status == "cancelled"
                                        else {}
                                    ),
                                },
                                iteration_index=row.iteration_count or 0,
                            )
                            await self._write_tx(session)
                            _codes.log_code(
                                logger,
                                _codes.REGISTRY[
                                    "legacy_run_swept"
                                    if reason == "stale_running_legacy"
                                    else reason or "cancel_requested"
                                ],
                                run_id=run_id,
                            )
                            return SweptRun(
                                run_id,
                                row.agent_name,
                                "running",
                                reason or "cancel_requested",
                                _to_aware_utc(row.last_progress_at),
                                _dt.datetime.now(_dt.UTC),
                                status,
                            )
                        if row.lease_expired_at is not None:
                            return _reclaimable(row)
                        now = _dt.datetime.now(_dt.UTC)
                        # This session holds scalar projections, not ORM entities.
                        # Avoid identity synchronization of the CAS's RETURNING row.
                        marked = (
                            await self._update_one_rows(
                                session,
                                table=AgentRun.__table__,
                                pk=(run_id,),
                                predicate=and_(
                                    AgentRun.id == run_id,
                                    AgentRun.status == "running",
                                    AgentRun.storage_format == 2,
                                    AgentRun.lease_expired_at.is_(None),
                                    AgentRun.claim_epoch == row.claim_epoch,
                                    func.coalesce(AgentRun.last_progress_at, AgentRun.created_at)
                                    < cutoff,
                                ),
                                values=dict(
                                    claim_epoch=AgentRun.claim_epoch + 1,
                                    lease_expired_at=now,
                                    reclaim_limit=limit,
                                ),
                                returning=[*fields],
                            )
                        ).first()
                        if marked is None:
                            current = (
                                await session.execute(
                                    select(*fields).where(
                                        AgentRun.id == run_id,
                                        AgentRun.status == "running",
                                        AgentRun.lease_expired_at.is_not(None),
                                    )
                                )
                            ).first()
                            return _reclaimable(current) if current is not None else None
                        await _insert_event_locked(
                            session,
                            run_id,
                            "run.lease_expired",
                            {"gap_s": max(0.0, (now - age).total_seconds())},
                        )
                        item = _reclaimable(marked)
                        await self._write_tx(session)
                        return item

                item = await retry_transient_db(
                    attempt,
                    label="mark_lease_expired",
                    run_id=run_id,
                    classify=self.classify_db_error,
                )
                inspected += 1
                after = run_id
                if isinstance(item, ReclaimableRun):
                    result.reclaimable.append(item)
                elif isinstance(item, SweptRun):
                    result.finalised.append(item)
                    if item.failure_reason == "stale_running_legacy":
                        result.legacy.append(item)
                if inspected >= max_runs or len(result.reclaimable) >= max_returned:
                    if inspected >= max_runs:
                        _codes.log_code(
                            logger,
                            _codes.SWEEP_CAP_REACHED,
                            category="stale_running",
                            cap=max_runs,
                            inspected=inspected,
                        )
                    return MarkResult(
                        result.reclaimable, result.finalised, result.legacy, inspected, after
                    )
        return MarkResult(result.reclaimable, result.finalised, result.legacy, inspected, after)

    async def get_lease_expired(self, *, limit: int, cursor: str | None) -> list[ReclaimableRun]:
        """Rediscover marks without changing epochs or emitting duplicate events."""
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        _positive_page(limit)
        stmt = select(
            AgentRun.id,
            AgentRun.tenant_id,
            AgentRun.agent_name,
            AgentRun.last_progress_at,
            AgentRun.lease_expired_at,
            AgentRun.reclaim_attempts,
            AgentRun.reclaim_limit,
            AgentRun.claim_epoch,
        ).where(
            AgentRun.status == "running",
            AgentRun.storage_format == 2,
            AgentRun.lease_expired_at.is_not(None),
        )
        if cursor is not None:
            stmt = stmt.where(AgentRun.id > cursor)
        async with self._session_factory() as session:
            return [
                _reclaimable(row)
                for row in (
                    await session.execute(stmt.order_by(AgentRun.id).limit(min(limit, 1000)))
                ).all()
            ]

    async def reclaim_run(
        self,
        run_id: str,
        *,
        tenant_id: TenantScope,
        agent_name: str,
        expected_claim_epoch: int | None,
        governance: dict[str, Any],
        governance_policy: Literal["warn", "raise"],
        skill_pins: dict[str, str],
        loop_type: str,
    ) -> ClaimResult | Literal[False]:
        """Validate and reclaim under the same immutable-PK lock used by the update."""
        from types import SimpleNamespace

        from dendrux.db.models import AgentRun
        from dendrux.errors import RunNotFoundError, RunNotReclaimableError
        from dendrux.runtime.state import _check_scope

        if governance_policy not in ("warn", "raise"):
            _codes.log_code(logger, _codes.GOVERNANCE_MISMATCH, run_id=run_id)
            raise ValueError("governance_policy must be warn or raise")
        _check_scope(tenant_id)
        table = AgentRun.__table__
        scope = (
            true()
            if isinstance(tenant_id, _Unscoped)
            else table.c.tenant_id.is_not_distinct_from(tenant_id)
        )
        projection = [
            table.c[name]
            for name in (
                "id",
                "tenant_id",
                "storage_format",
                "status",
                "lease_expired_at",
                "claim_epoch",
                "reclaim_attempts",
                "reclaim_limit",
                "cancel_requested",
                "agent_name",
                "meta",
                "loop_state",
                "last_progress_at",
                "created_at",
                "last_reclaimed_at",
            )
        ]

        async def attempt() -> ClaimResult | Literal[False]:
            assignments: list[tuple[Any, Any]] = []
            previous_epoch = 0
            cancelled = False
            governance_changed = False
            gap = 0.0

            def validate(row_mapping: Any) -> None:
                nonlocal previous_epoch, cancelled, governance_changed, gap
                row = SimpleNamespace(**row_mapping)
                checks = [
                    (row.storage_format == 2, "legacy_run"),
                    (self.capabilities.recovery, "store_capability"),
                    (row.status == "running", "not_running"),
                    (row.lease_expired_at is not None, "lease_fresh"),
                    (
                        expected_claim_epoch is None or row.claim_epoch == expected_claim_epoch,
                        "already_reclaimed",
                    ),
                    (
                        row.reclaim_limit is not None and row.reclaim_attempts < row.reclaim_limit,
                        "reclaim_limit",
                    ),
                ]
                for ok, reason in checks:
                    if not ok:
                        raise RunNotReclaimableError(run_id, reason)
                previous_epoch = row.claim_epoch
                cancelled = bool(row.cancel_requested)
                if cancelled:
                    assignments.extend(
                        self._ordered(
                            table,
                            {
                                **_status_values("cancelled"),
                                "claim_epoch": table.c.claim_epoch + 1,
                                "pause_data": None,
                                "cancel_requested": False,
                            },
                        )
                    )
                    return
                if row.agent_name != agent_name:
                    raise RunNotReclaimableError(run_id, "agent_mismatch")
                saved = (row.meta or {}).get("dendrux.governance")
                governance_changed = saved is not None and saved.get("digest") != governance.get(
                    "digest"
                )
                if governance_changed:
                    _codes.log_code(logger, _codes.GOVERNANCE_MISMATCH, run_id=run_id)
                    if governance_policy == "raise":
                        raise RunNotReclaimableError(run_id, "governance_changed")
                if (((row.loop_state or {}).get("skills") or {}).get("pins", {})) != skill_pins:
                    raise RunNotReclaimableError(run_id, "skill_mismatch")
                saved_loop = (row.loop_state or {}).get("loop")
                if saved_loop not in ("react", "single_call"):
                    raise RunNotReclaimableError(run_id, "loop_unknown")
                if loop_type != "react" or saved_loop != "react":
                    raise RunNotReclaimableError(run_id, "loop_unsupported")
                now = _dt.datetime.now(_dt.UTC)
                gap = max(
                    0.0,
                    (now - _to_aware_utc(row.last_progress_at or row.created_at)).total_seconds(),
                )
                assignments.extend(
                    self._ordered(
                        table,
                        dict(
                            claim_epoch=table.c.claim_epoch + 1,
                            reclaim_attempts=table.c.reclaim_attempts + 1,
                            lease_expired_at=None,
                            last_progress_at=now,
                            last_reclaimed_at=now,
                        ),
                    )
                )

            async with self._session_factory() as session:
                session.info["dendrux.store"] = self
                changed = await self.update_one(
                    session,
                    table=table,
                    pk=(run_id,),
                    predicate=scope,
                    values=assignments,
                    returning=[
                        table.c.claim_epoch,
                        table.c.storage_format,
                        table.c.loop_state,
                        table.c.reclaim_attempts,
                    ],
                    lock_projection=projection,
                    validate_locked=validate,
                )
                if changed is None:
                    raise RunNotFoundError(run_id)
                if cancelled:
                    await _insert_event_locked(
                        session, run_id, "run.cancelled", {"reason": "cancel_requested"}
                    )
                    await self._write_tx(session)
                    raise RunNotReclaimableError(run_id, "cancelled")
                await _insert_event_locked(
                    session,
                    run_id,
                    "run.reclaimed",
                    dict(
                        attempt=changed["reclaim_attempts"],
                        previous_epoch=previous_epoch,
                        gap_s=gap,
                        **({"governance_changed": True} if governance_changed else {}),
                    ),
                )
                token = ClaimResult(
                    changed["claim_epoch"],
                    changed["storage_format"],
                    changed["loop_state"],
                    (changed["loop_state"] or {}).get("submission"),
                )
                await self._write_tx(session)
                return token

        return await retry_transient_db(
            attempt, label="reclaim_run", run_id=run_id, classify=self.classify_db_error
        )

    async def get_replayed_tool_calls(
        self, run_id: str, *, limit: int = 100, tenant_id: TenantScope = UNSCOPED
    ) -> tuple[list[str], bool]:
        """Read replayed call ids with one bounded lookahead for truncation."""
        from sqlalchemy import select

        from dendrux.db.models import ToolCallRecord

        _positive_page(limit)
        limit = min(limit, 1000)
        async with self._session_factory() as session:
            rows = (
                await session.scalars(
                    select(ToolCallRecord.tool_call_id)
                    .where(
                        ToolCallRecord.agent_run_id == run_id,
                        ToolCallRecord.replayed_at.is_not(None),
                        _run_in_scope(run_id, tenant_id),
                    )
                    .order_by(ToolCallRecord.replayed_at, ToolCallRecord.id)
                    .limit(limit + 1)
                )
            ).all()
            return list(rows[:limit]), len(rows) > limit

    async def sweep_stale_runs(
        self,
        older_than: timedelta,
        *,
        batch_size: int = 100,
        max_runs: int = 10000,
    ) -> list[SweptRun]:
        """Inspect running candidates in projected pages; cap 10,000 by default.

        max_runs accepts 1..100,000. Fresh rows and losing CAS attempts count
        toward it; hitting the cap emits sweep_cap_reached.
        """
        return await self._sweep_runs(
            older_than, waiting=False, batch_size=batch_size, max_runs=max_runs
        )

    async def sweep_abandoned_runs(
        self,
        older_than: timedelta,
        *,
        batch_size: int = 100,
        max_runs: int = 10000,
    ) -> list[SweptRun]:
        """Inspect waiting candidates in projected pages; cap 10,000 by default.

        max_runs accepts 1..100,000 across all waiting statuses. The final CAS
        requires both status_changed_at and updated_at to be stale. Losing CAS
        attempts count toward the cap; hitting it emits sweep_cap_reached.
        """
        return await self._sweep_runs(
            older_than, waiting=True, batch_size=batch_size, max_runs=max_runs
        )

    async def _sweep_runs(
        self,
        older_than: timedelta,
        *,
        waiting: bool,
        batch_size: int,
        max_runs: int,
    ) -> list[SweptRun]:
        from sqlalchemy import case, func, literal_column, select, true, tuple_

        from dendrux.db.models import AgentRun, RunEvent

        for name, value, ceiling in (
            ("batch_size", batch_size, 1000),
            ("max_runs", max_runs, 100000),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= ceiling:
                raise ValueError(f"{name} must be between 1 and {ceiling}")
        _validate_sweep_age(older_than)
        now = _dt.datetime.now(_dt.UTC)
        cutoff = now - older_than
        age = (
            AgentRun.status_changed_at
            if waiting
            else func.coalesce(AgentRun.last_progress_at, AgentRun.created_at)
        )
        statuses = (
            ["waiting_client_tool", "waiting_human_input", "waiting_approval", "paused"]
            if waiting
            else ["running"]
        )
        # A scalar LIMIT subquery prevents a hashed scan of every started event.
        started = (
            select(RunEvent.id)
            .where(
                RunEvent.agent_run_id == AgentRun.id,
                RunEvent.event_type == literal_column("'run.started'"),
            )
            .limit(1)
            .correlate(AgentRun)
            .scalar_subquery()
        )
        swept: list[SweptRun] = []
        visited = 0
        for status_filter in statuses:
            after: tuple[_dt.datetime, str] | str | None = None
            while visited < max_runs:
                statement = select(
                    AgentRun.id,
                    AgentRun.agent_name,
                    AgentRun.status,
                    AgentRun.last_progress_at,
                    AgentRun.iteration_count,
                    AgentRun.claim_epoch,
                    started.label("started"),
                    age.label("sweep_age"),
                ).limit(min(batch_size, max_runs - visited))
                if waiting:
                    statement = statement.where(
                        AgentRun.status == status_filter, age < cutoff
                    ).order_by(age, AgentRun.id)
                    if after is not None:
                        statement = statement.where(tuple_(age, AgentRun.id) > after)
                else:
                    # Equal range bounds preserve the (status, id) ordering in
                    # the planner. Equality alone lets it prefer the primary key
                    # and filter through terminal history on every page.
                    statement = statement.where(
                        AgentRun.status.between(status_filter, status_filter)
                    ).order_by(AgentRun.status, AgentRun.id)
                    if after is not None:
                        statement = statement.where(
                            tuple_(AgentRun.status, AgentRun.id) > (status_filter, after)
                        )
                async with self._session_factory() as session:
                    rows = (await session.execute(statement)).all()
                if not rows:
                    break
                visited += len(rows)
                after = (rows[-1].sweep_age, rows[-1].id) if waiting else rows[-1].id
                for row in rows:
                    # Running rows are bounded by live concurrency. Count every
                    # inspected row against the work cap, including fresh ones.
                    if not waiting and _to_aware_utc(row.sweep_age) >= cutoff:
                        continue
                    reason = (
                        "abandoned_waiting"
                        if waiting
                        else "stale_running"
                        if row.started
                        else "never_started"
                    )
                    flagged = AgentRun.cancel_requested.is_(True)
                    new_status = "error" if waiting else case((flagged, "cancelled"), else_="error")
                    async with self._session_factory() as session:
                        result = await self._update_one_rows(
                            session,
                            table=AgentRun.__table__,
                            pk=(row.id,),
                            predicate=and_(
                                AgentRun.id == row.id,
                                AgentRun.status == row.status,
                                AgentRun.claim_epoch == row.claim_epoch,
                                age < cutoff,
                                AgentRun.updated_at < cutoff if waiting else true(),
                            ),
                            values=dict(
                                **_status_values(new_status),
                                claim_epoch=AgentRun.claim_epoch + 1,
                                failure_reason=reason,
                                pause_data=None,
                                error=f"Run swept as {reason}"
                                if waiting
                                else case((flagged, None), else_=f"Run swept as {reason}"),
                                cancel_requested=False,
                                pause_requested=False,
                            ),
                            returning=[AgentRun.status],
                        )
                        status = result.scalar_one_or_none()
                        if status is None:
                            continue
                        event_type = "run.abandoned" if waiting else "run.interrupted"
                        data = {"failure_reason": reason}
                        if status == "cancelled":
                            event_type = "run.cancelled"
                            data["reason"] = "cancel_requested"
                        await _insert_event_locked(
                            session,
                            row.id,
                            event_type,
                            data,
                            iteration_index=row.iteration_count or 0,
                        )
                        await self._write_tx(session)
                    swept.append(
                        SweptRun(
                            run_id=row.id,
                            agent_name=row.agent_name,
                            previous_status=row.status,
                            failure_reason=reason,
                            last_progress_at=_to_aware_utc(row.last_progress_at),
                            swept_at=_to_aware_utc(now),
                            status=status,
                        )
                    )
        if visited == max_runs:
            _codes.log_code(
                logger,
                _codes.SWEEP_CAP_REACHED,
                category="abandoned_waiting" if waiting else "stale_running",
                cap=max_runs,
                inspected=visited,
            )
        return swept

    # ------------------------------------------------------------------
    # Delegation queries
    # ------------------------------------------------------------------

    async def get_delegation_info(
        self, run_id: str, *, max_rows: int = 1000, max_depth: int = 32
    ) -> DelegationInfo | None:
        """Bounded delegation context; summaries cover only the visited subtree.

        max_rows includes the root, ancestors and descendants (1..10000).
        max_depth bounds each direction (0..100). One lookahead row detects
        truncation; large JSON execution/payload columns are never selected.
        """
        from sqlalchemy import select
        from sqlalchemy.orm import load_only

        from dendrux.db.models import AgentRun

        if (
            isinstance(max_rows, bool)
            or not isinstance(max_rows, int)
            or not 1 <= max_rows <= 10000
        ):
            raise ValueError("max_rows must be an int between 1 and 10000.")
        if (
            isinstance(max_depth, bool)
            or not isinstance(max_depth, int)
            or not 0 <= max_depth <= 100
        ):
            raise ValueError("max_depth must be an int between 0 and 100.")
        projection = load_only(
            AgentRun.id,
            AgentRun.parent_run_id,
            AgentRun.agent_name,
            AgentRun.status,
            AgentRun.delegation_level,
            AgentRun.total_input_tokens,
            AgentRun.total_output_tokens,
            AgentRun.total_cost_usd,
        )

        def brief(row: AgentRun) -> RunBrief:
            return RunBrief(
                run_id=row.id,
                agent_name=row.agent_name,
                status=_extract_status(row),
                delegation_level=row.delegation_level,
            )

        async with self._session_factory() as session:
            run_row = (
                await session.execute(
                    select(AgentRun).options(projection).where(AgentRun.id == run_id)
                )
            ).scalar_one_or_none()
            if run_row is None:
                return None
            remaining = max_rows - 1
            seen = {run_id}
            ancestry: list[RunBrief] = []
            parent = (
                ParentRef(run_id=run_row.parent_run_id, resolved=False)
                if run_row.parent_run_id
                else None
            )
            current = run_row.parent_run_id
            truncated = False
            ancestry_complete = True
            while current:
                if current in seen:
                    ancestry_complete = False
                    break
                if remaining == 0 or len(ancestry) >= max_depth:
                    truncated = True
                    ancestry_complete = False
                    break
                row = (
                    await session.execute(
                        select(AgentRun).options(projection).where(AgentRun.id == current)
                    )
                ).scalar_one_or_none()
                if row is None:
                    ancestry_complete = False
                    break
                seen.add(row.id)
                remaining -= 1
                ancestry.append(brief(row))
                if len(ancestry) == 1:
                    parent = ParentRef(
                        run_id=row.id,
                        resolved=True,
                        agent_name=row.agent_name,
                        status=_extract_status(row),
                        delegation_level=row.delegation_level,
                    )
                current = row.parent_run_id
            ancestry.reverse()
            descendants: list[tuple[AgentRun, int]] = []
            frontier = [run_id]
            depth = 0
            seen = {run_id}
            while frontier:
                depth += 1
                next_level = []
                stop = False
                for i in range(0, len(frontier), _IN_CHUNK):
                    allowance = remaining if depth <= max_depth else 0
                    stmt = (
                        select(AgentRun)
                        .options(projection)
                        .where(
                            AgentRun.parent_run_id.in_(frontier[i : i + _IN_CHUNK]),
                            AgentRun.id.not_in(seen),
                        )
                        .order_by(AgentRun.created_at, AgentRun.id)
                        .limit(allowance + 1)
                    )
                    rows = list((await session.execute(stmt)).scalars())
                    if len(rows) > allowance:
                        truncated = True
                        stop = True
                    for row in rows[:allowance]:
                        seen.add(row.id)
                        remaining -= 1
                        descendants.append((row, depth))
                        next_level.append(row.id)
                    if stop:
                        break
                if stop:
                    break
                frontier = next_level
            children, summary = await self._traverse_subtree_bfs(
                session, run_row, descendants=descendants
            )
            return DelegationInfo(
                parent=parent,
                children=children,
                ancestry=ancestry,
                subtree_summary=summary,
                ancestry_complete=ancestry_complete,
                truncated=truncated,
            )

    async def _walk_ancestry(
        self,
        session: AsyncSession,
        run_row: AgentRun,
    ) -> tuple[list[RunBrief], bool]:
        """Walk parent chain upward. Returns (ancestry, complete) root-first."""
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        ancestry: list[RunBrief] = []
        complete = True
        current_parent_id = run_row.parent_run_id
        visited: set[str] = {run_row.id}

        while current_parent_id and current_parent_id not in visited:
            visited.add(current_parent_id)
            stmt = select(AgentRun).where(AgentRun.id == current_parent_id)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                complete = False
                break
            ancestry.append(
                RunBrief(
                    run_id=row.id,
                    agent_name=row.agent_name,
                    status=_extract_status(row),
                    delegation_level=row.delegation_level,
                )
            )
            current_parent_id = row.parent_run_id

        # Cycle detected — parent_run_id pointed back into visited set
        if current_parent_id and current_parent_id in visited:
            complete = False

        ancestry.reverse()
        return ancestry, complete

    async def _bfs_descendants(
        self,
        session: AsyncSession,
        root_id: str,
        *,
        tenant_id: TenantScope = UNSCOPED,
        light: bool = False,
        max_nodes: int | None = None,
    ) -> list[tuple[AgentRun, int]]:
        """Descendants of ``root_id`` as ``(row, depth)`` in BFS order, root excluded.

        ``depth`` is 1 for direct children. Cycle-safe via a seen set. With
        ``tenant_id`` set, other tenants' runs are skipped and so is
        everything below them. ``light`` loads only id / parent / tenant so
        usage-only callers don't pull large JSON columns.
        """
        from sqlalchemy import select
        from sqlalchemy.orm import load_only

        from dendrux.db.models import AgentRun

        seen: set[str] = {root_id}
        current_level = [root_id]
        found: list[tuple[AgentRun, int]] = []
        depth = 0

        while current_level:
            depth += 1
            rows: list[AgentRun] = []
            for i in range(0, len(current_level), _IN_CHUNK):
                stmt = select(AgentRun).where(
                    AgentRun.parent_run_id.in_(current_level[i : i + _IN_CHUNK])
                )
                if light:
                    stmt = stmt.options(
                        load_only(AgentRun.id, AgentRun.parent_run_id, AgentRun.tenant_id)
                    )
                stmt = _apply_tenant_scope(stmt, tenant_id)
                if max_nodes is not None:
                    stmt = stmt.limit(max_nodes - len(found) - len(rows))
                result = await session.execute(stmt.order_by(AgentRun.created_at, AgentRun.id))
                rows.extend(result.scalars().all())
                if max_nodes is not None and len(found) + len(rows) >= max_nodes:
                    import logging

                    from dendrux.errors import BudgetConfigurationError

                    _codes.log_code(
                        logging.getLogger(__name__), _codes.SUBTREE_LIMIT, run_id=root_id
                    )
                    raise BudgetConfigurationError(reason="subtree_limit")

            next_level: list[str] = []
            for cr in rows:
                if cr.id in seen:
                    continue
                seen.add(cr.id)
                next_level.append(cr.id)
                found.append((cr, depth))
            current_level = next_level

        return found

    async def _traverse_subtree_bfs(
        self,
        session: AsyncSession,
        run_row: AgentRun,
        *,
        descendants: list[tuple[AgentRun, int]] | None = None,
    ) -> tuple[list[RunBrief], SubtreeSummary]:
        """BFS downward from run_row. Returns (direct_children, summary).

        Summary includes the root run itself in token/cost/status counts.
        """
        if descendants is None:
            descendants = await self._bfs_descendants(session, run_row.id)

        sum_input = run_row.total_input_tokens or 0
        sum_output = run_row.total_output_tokens or 0
        root_cost = run_row.total_cost_usd
        has_cost = root_cost is not None
        sum_cost = float(root_cost) if root_cost is not None else 0.0
        unknown_cost_count = 0 if has_cost else 1
        status_counts: dict[str, int] = {_extract_status(run_row): 1}
        children: list[RunBrief] = []

        for cr, depth in descendants:
            sum_input += cr.total_input_tokens or 0
            sum_output += cr.total_output_tokens or 0
            if cr.total_cost_usd is not None:
                sum_cost += float(cr.total_cost_usd)
                has_cost = True
            else:
                unknown_cost_count += 1
            s = _extract_status(cr)
            status_counts[s] = status_counts.get(s, 0) + 1
            if depth == 1:
                children.append(
                    RunBrief(
                        run_id=cr.id,
                        agent_name=cr.agent_name,
                        status=s,
                        delegation_level=cr.delegation_level,
                    )
                )

        summary = SubtreeSummary(
            direct_child_count=len(children),
            descendant_count=len(descendants),
            max_depth=max((d for _, d in descendants), default=0),
            subtree_input_tokens=sum_input,
            subtree_output_tokens=sum_output,
            subtree_cost_usd=sum_cost if has_cost else None,
            unknown_cost_count=unknown_cost_count,
            status_counts=status_counts,
        )
        return children, summary

    async def _usage_by_run(
        self, session: AsyncSession, run_ids: list[str]
    ) -> dict[str, UsageRollup]:
        """Per-run live usage, one grouped query per chunk of ids.

        Runs without any ``llm_interactions`` row are absent from the result.
        """
        from sqlalchemy import case, func, select

        from dendrux.db.models import LLMInteraction as L

        out: dict[str, UsageRollup] = {}
        for i in range(0, len(run_ids), _IN_CHUNK):
            stmt = (
                select(
                    L.agent_run_id,
                    func.coalesce(func.sum(L.input_tokens), 0),
                    func.coalesce(func.sum(L.output_tokens), 0),
                    func.coalesce(func.sum(L.cache_read_input_tokens), 0),
                    func.coalesce(func.sum(L.cache_creation_input_tokens), 0),
                    func.coalesce(func.sum(L.reasoning_tokens), 0),
                    func.coalesce(func.sum(L.cost_usd), 0),
                    func.coalesce(func.sum(case((L.cost_usd.is_(None), 1), else_=0)), 0),
                    func.count(),
                )
                .where(L.agent_run_id.in_(run_ids[i : i + _IN_CHUNK]))
                .group_by(L.agent_run_id)
            )
            for row in (await session.execute(stmt)).all():
                out[row[0]] = UsageRollup(
                    input_tokens=int(row[1]),
                    output_tokens=int(row[2]),
                    cache_read_tokens=int(row[3]),
                    cache_write_tokens=int(row[4]),
                    reasoning_tokens=int(row[5]),
                    known_cost_usd=round(float(row[6]), _COST_DECIMALS),
                    unpriced_calls=int(row[7]),
                    llm_calls=int(row[8]),
                )
        return out

    async def get_subtree(
        self, run_id: str, *, tenant_id: TenantScope = UNSCOPED
    ) -> SubtreeRecord | None:
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        async with self._session_factory() as session:
            stmt = _apply_tenant_scope(select(AgentRun).where(AgentRun.id == run_id), tenant_id)
            root = (await session.execute(stmt)).scalar_one_or_none()
            if root is None:
                return None

            rows = [(root, 0), *await self._bfs_descendants(session, root.id, tenant_id=tenant_id)]
            usage = await self._usage_by_run(session, [r.id for r, _ in rows])
            nodes = [
                SubtreeNodeRecord(
                    run=_run_to_record(r), depth=d, usage=usage.get(r.id, UsageRollup())
                )
                for r, d in rows
            ]
            return SubtreeRecord(nodes=nodes, usage=UsageRollup.combine(n.usage for n in nodes))

    async def get_descendant_page(
        self,
        run_id: str,
        *,
        after: str | None = None,
        limit: int = 500,
        tenant_id: TenantScope = UNSCOPED,
    ) -> list[str]:
        """Read one page of descendant ids, without run payloads.

        A recursive UNION deduplicates ids, including malformed cycles. Each
        page uses a fresh statement so cancellation writes never hold its cursor.
        """
        from sqlalchemy import select

        from dendrux.db.models import AgentRun

        tree = _apply_tenant_scope(select(AgentRun.id).where(AgentRun.id == run_id), tenant_id).cte(
            "descendants", recursive=True
        )
        children = _apply_tenant_scope(
            select(AgentRun.id).join(tree, AgentRun.parent_run_id == tree.c.id), tenant_id
        )
        tree = tree.union(children)
        stmt = select(tree.c.id).where(tree.c.id != run_id)
        if after is not None:
            stmt = stmt.where(tree.c.id > after)
        stmt = stmt.order_by(tree.c.id).limit(max(1, min(limit, 500)))
        async with self._session_factory() as session:
            return list((await session.scalars(stmt)).all())

    async def subtree_usage(self, run_id: str, *, max_nodes: int = 10000) -> UsageRollup:
        if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < 1:
            raise ValueError("max_nodes must be a positive integer")
        async with self._session_factory() as session:
            descendants = await self._bfs_descendants(
                session, run_id, light=True, max_nodes=max_nodes
            )
            usage = await self._usage_by_run(session, [run_id, *(r.id for r, _ in descendants)])
            return UsageRollup.combine(usage.values())

    async def run_usage(self, run_id: str) -> UsageRollup:
        async with self._session_factory() as session:
            return (await self._usage_by_run(session, [run_id])).get(run_id, UsageRollup())

    _MAX_LIST_LIMIT = 1000

    async def list_runs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        tenant_id: TenantScope = UNSCOPED,
        status: str | list[str] | None = None,
        agent_name: str | None = None,
        parent_run_id: str | None = None,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RunRecord]:
        """Runs newest first, filtered in SQL. Never loads ``pause_data`` or ``pii_mapping``.

        ``limit`` is clamped to 1..1000; ``offset`` above 10,000 raises
        ``ValueError`` (deep OFFSET pages scan every skipped row).
        """
        from sqlalchemy import select
        from sqlalchemy.orm import defer

        from dendrux.db.models import AgentRun

        capped_limit = min(max(1, limit), self._MAX_LIST_LIMIT)
        clamped_offset = _check_offset(offset)

        async with self._session_factory() as session:
            # Secondary sort by id keeps pagination stable when created_at
            # ties — SQLite's DateTime is second-resolution, so concurrent
            # runs share timestamps and offset paging would otherwise skip
            # or duplicate rows across pages.
            stmt = (
                select(AgentRun)
                .options(
                    defer(AgentRun.pause_data, raiseload=True),
                    defer(AgentRun.pii_mapping, raiseload=True),
                )
                .order_by(AgentRun.created_at.desc(), AgentRun.id.desc())
            )
            stmt = self._apply_run_filters(
                stmt,
                tenant_id=tenant_id,
                status=status,
                agent_name=agent_name,
                parent_run_id=parent_run_id,
                started_after=started_after,
                started_before=started_before,
                metadata_filter=metadata_filter,
            )
            stmt = stmt.limit(capped_limit).offset(clamped_offset)

            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [_run_to_record(r) for r in rows]

    async def count_pauses_per_run(
        self,
        run_ids: list[str],
    ) -> dict[str, int]:
        if not run_ids:
            return {}

        from sqlalchemy import func, select

        from dendrux.db.models import RunEvent

        async with self._session_factory() as session:
            stmt = (
                select(RunEvent.agent_run_id, func.count())
                .where(RunEvent.agent_run_id.in_(run_ids))
                .where(RunEvent.event_type == "run.paused")
                .group_by(RunEvent.agent_run_id)
            )
            result = await session.execute(stmt)
            return {row[0]: int(row[1]) for row in result.all()}

    async def count_runs(
        self,
        *,
        tenant_id: TenantScope = UNSCOPED,
        status: str | list[str] | None = None,
        agent_name: str | None = None,
        parent_run_id: str | None = None,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        """Count runs matching the ``list_runs`` filters (same tenant semantics)."""
        from sqlalchemy import func, select

        from dendrux.db.models import AgentRun

        async with self._session_factory() as session:
            stmt = select(func.count()).select_from(AgentRun)
            stmt = self._apply_run_filters(
                stmt,
                tenant_id=tenant_id,
                status=status,
                agent_name=agent_name,
                parent_run_id=parent_run_id,
                started_after=started_after,
                started_before=started_before,
                metadata_filter=metadata_filter,
            )
            result = await session.execute(stmt)
            total = result.scalar_one()
            return int(total)

    def _apply_run_filters(
        self,
        stmt: Any,
        *,
        tenant_id: TenantScope,
        status: str | list[str] | None,
        agent_name: str | None,
        parent_run_id: str | None,
        started_after: datetime | None,
        started_before: datetime | None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> Any:
        from sqlalchemy import false

        from dendrux.db.models import AgentRun

        # Naive datetimes here would silently break PG comparisons against
        # TIMESTAMPTZ. Reject at the boundary so future filter additions
        # inherit the guard automatically by routing through this helper.
        _require_aware_utc("started_after", started_after)
        _require_aware_utc("started_before", started_before)

        stmt = _apply_tenant_scope(stmt, tenant_id)
        if status is not None:
            if isinstance(status, list):
                if not status:
                    # Empty list = "match no status" (vs. "no filter at all").
                    stmt = stmt.where(false())
                else:
                    stmt = stmt.where(AgentRun.status.in_(status))
            else:
                stmt = stmt.where(AgentRun.status == status)
        if agent_name is not None:
            stmt = stmt.where(AgentRun.agent_name == agent_name)
        if parent_run_id is not None:
            stmt = stmt.where(AgentRun.parent_run_id == parent_run_id)
        # Normalized to UTC: SQLite stores UTC wall-clock text, so a non-UTC
        # offset would otherwise compare against the wrong instant.
        if started_after is not None:
            stmt = stmt.where(AgentRun.created_at >= _to_aware_utc(started_after))
        if started_before is not None:
            stmt = stmt.where(AgentRun.created_at < _to_aware_utc(started_before))
        if metadata_filter:
            stmt = self._apply_metadata_filter(stmt, metadata_filter)
        return stmt

    # JSON keys are limited to a conservative identifier shape so the
    # path string we splice into the SQL (``$.<key>`` for SQLite) cannot
    # smuggle path operators or quotes. Values are always parameter-bound,
    # so only the key is sensitive here.
    _METADATA_KEY_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

    def _apply_metadata_filter(self, stmt: Any, metadata_filter: dict[str, Any]) -> Any:
        """Filter by exact-match equality on top-level ``meta`` keys.

        AND across all keys. ``thread_id="abc"`` is the canonical chatbot
        case; arbitrary scalar values (str/int/bool) work too. Nested
        paths (``user.id``) and list-membership checks are intentionally
        unsupported in v1 — add them when a real use case arrives.
        """

        from dendrux.db.models import AgentRun
        from dendrux.types import _validate_persistable

        _validate_persistable(metadata_filter, "metadata_filter")
        for key in metadata_filter:
            if not isinstance(key, str) or not self._METADATA_KEY_RE.match(key):
                raise ValueError(
                    f"metadata_filter keys must match {self._METADATA_KEY_RE.pattern!r}; "
                    f"got {key!r}. Nested paths are not supported in v1."
                )

        return stmt.where(self.metadata_filter_predicate(AgentRun.meta, metadata_filter))

    def metadata_filter_predicate(self, col: Any, metadata_filter: dict[str, Any]) -> Any:
        from sqlalchemy import and_, true

        return and_(true(), *(self.json_contains(col, (k,), v) for k, v in metadata_filter.items()))

    def event_sequence_expr(self, next_index: Any) -> Any:
        """Build sequence allocation without changing the owning INSERT shape."""
        return next_index

    def inserted_event_sequence(self, result: Any) -> int | None:
        """Return a dialect-certified inserted sequence, or request PK readback."""
        return None

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
        from sqlalchemy import JSON, DateTime, Integer, String, func, literal, select
        from sqlalchemy.exc import IntegrityError

        from dendrux.db.models import RunEvent
        from dendrux.errors import PersistenceError
        from dendrux.runtime._digest import event_digest

        digest = event_digest(event_type, iteration_index, correlation_id, data)

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
            literal(data, JSON()),
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
        session.info["dendrux.fresh_evidence"] = True
        hook = session.info.get("dendrux.event_hook")
        if hook is not None:
            await hook(session, run_id)
        return int(seq)
