"""Shared SQL persistence. Lock order: run, ordered children, ordered blobs, settings.

Live updates are lossy hints. If create_run commits but loses its acknowledgement
and readback wrongly proves it absent, the retry may recover the created run from
the unique conflict without sending a hint. Polling still observes that run.
"""

from __future__ import annotations

import datetime as _dt
import json as json
import re
from inspect import getmembers, iscoroutinefunction
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast

from sqlalchemy import and_, select, true
from sqlalchemy.exc import SQLAlchemyError as SQLAlchemyError

import dendrux.runtime._blob_catalog as _blobs
from dendrux import _codes
from dendrux._lifecycle import (
    PAUSE_STATUS_VALUES,
    STORAGE_FORMAT_CHECKPOINTED,
    CancelReason,
    FailureReason,
    StoredEventType,
)
from dendrux._lifecycle import (
    TERMINAL_STATUSES as TERMINAL_STATUSES,
)
from dendrux._lifecycle import (
    GovernanceEventType as GovernanceEventType,
)
from dendrux._meta_keys import (
    ERROR_INFO_META_KEY as ERROR_INFO_META_KEY,
)
from dendrux._meta_keys import (
    FRESH_EVIDENCE_SESSION_KEY,
    GOVERNANCE_KEY,
    STORE_SESSION_KEY,
)
from dendrux._meta_keys import (
    LATEST_REWRITE_META_KEY as LATEST_REWRITE_META_KEY,
)
from dendrux._meta_keys import (
    OBSERVATION_META_KEY as OBSERVATION_META_KEY,
)
from dendrux._meta_keys import (
    STOP_REASON_META_KEY as STOP_REASON_META_KEY,
)
from dendrux.runtime._blob_catalog import SQLBlobCatalog
from dendrux.runtime._blob_catalog import write_blob_refs as write_blob_refs
from dendrux.runtime._bounds import (
    DEFAULT_EVENT_LIMIT as DEFAULT_EVENT_LIMIT,
)
from dendrux.runtime._bounds import (
    DEFAULT_LIST_LIMIT as DEFAULT_LIST_LIMIT,
)
from dendrux.runtime._bounds import (
    DEFAULT_READ_LIMIT as DEFAULT_READ_LIMIT,
)
from dendrux.runtime._bounds import DEFAULT_SUBTREE_DEPTH as DEFAULT_SUBTREE_DEPTH
from dendrux.runtime._bounds import DEFAULT_SUBTREE_ROWS as DEFAULT_SUBTREE_ROWS
from dendrux.runtime._bounds import (
    MAX_PAGE,
    PAUSE_PAGE_MAX,
    RECOVERY_SCAN_MAX,
    SWEEP_BATCH_DEFAULT,
    SWEEP_BATCH_MAX,
    SWEEP_MAX_RECLAIMS_MAX,
    SWEEP_MAX_RUNS_DEFAULT,
    SWEEP_MAX_RUNS_MAX,
)
from dendrux.runtime._bounds import MAX_SMALL_PAGE as MAX_SMALL_PAGE
from dendrux.runtime._bounds import MAX_SUBTREE_DEPTH as MAX_SUBTREE_DEPTH
from dendrux.runtime._bounds import MAX_SUBTREE_ROWS as MAX_SUBTREE_ROWS
from dendrux.runtime.durability import (
    _CommitNotAppliedError as _CommitNotAppliedError,
)
from dendrux.runtime.durability import (
    _live_updates_page,
    _safe_database_call,
    retry_transient_db,
    safe_database_operations,
)
from dendrux.runtime.state import _COST_DECIMALS as _COST_DECIMALS
from dendrux.runtime.state import DelegationInfo as DelegationInfo
from dendrux.runtime.state import ParentRef as ParentRef
from dendrux.runtime.state import RunBrief as RunBrief
from dendrux.runtime.state import SubtreeNodeRecord as SubtreeNodeRecord
from dendrux.runtime.state import SubtreeRecord as SubtreeRecord
from dendrux.runtime.state import SubtreeSummary as SubtreeSummary
from dendrux.runtime.state import UsageRollup as UsageRollup
from dendrux.runtime.state import _extract_status as _extract_status
from dendrux.runtime.state import _finalized_run_read as _finalized_run_read
from dendrux.runtime.state import _FinalizedRunRead as _FinalizedRunRead
from dendrux.runtime.stores.sql._types import RunParentCursor as RunParentCursor
from dendrux.runtime.stores.sql._types import SetFields as SetFields
from dendrux.types import (
    CreateRunResult as CreateRunResult,
)
from dendrux.types import (
    IdempotencyConflictError as IdempotencyConflictError,
)
from dendrux.types import (
    RunStatus,
    generate_ulid,
)
from dendrux.types import (
    UsageStats as UsageStats,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from datetime import datetime, timedelta

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
    from sqlalchemy.sql.base import Executable

    from dendrux.db._live_updates import LiveUpdates
    from dendrux.media import BlobEdge
    from dendrux.runtime.stores._errors import DbError
    from dendrux.runtime.stores.sql._types import DialectFacts, InsertPlan

from dendrux.runtime import state as _state
from dendrux.runtime.state import (
    _PERSISTED_VALUE_FIELDS,
    _TERMINAL_STATUS_VALUES,
    PAUSE_SUMMARY_KEYS,
    UNSCOPED,
    ClaimResult,
    EventWrite,
    LoopStatePatch,
    MarkResult,
    NoteRecord,
    ReclaimableRun,
    RunControl,
    RunEventRecord,
    StoreCapabilities,
    SweptRun,
    TenantScope,
    TraceWrite,
    _apply_loop_patch,
    _apply_tenant_scope,
    _insert_event,
    _reclaimable,
    _require_aware_utc,
    _run_in_scope,
    _to_aware_utc,
    _Unscoped,
    _validate_sweep_age,
)
from dendrux.runtime.state import (
    MANAGED_BINDING_KEY as MANAGED_BINDING_KEY,
)
from dendrux.runtime.state import (
    MANAGED_HEADER_KEY as MANAGED_HEADER_KEY,
)
from dendrux.runtime.state import (
    PAUSE_REQUEST_META_KEY as PAUSE_REQUEST_META_KEY,
)
from dendrux.runtime.state import (
    BindManagedChildRequest as BindManagedChildRequest,
)
from dendrux.runtime.state import (
    BindManagedChildResult as BindManagedChildResult,
)
from dendrux.runtime.state import (
    ConversationEntry as ConversationEntry,
)
from dendrux.runtime.state import (
    ConversationPage as ConversationPage,
)
from dendrux.runtime.state import (
    LLMInteractionRecord as LLMInteractionRecord,
)
from dendrux.runtime.state import (
    ManagedHeader as ManagedHeader,
)
from dendrux.runtime.state import (
    PauseSummary as PauseSummary,
)
from dendrux.runtime.state import (
    RunRecord as RunRecord,
)
from dendrux.runtime.state import (
    ToolCallReadRecord as ToolCallReadRecord,
)
from dendrux.runtime.state import (
    TraceRecord as TraceRecord,
)
from dendrux.runtime.state import (
    _approval_event_object as _approval_event_object,
)
from dendrux.runtime.state import (
    _check_offset as _check_offset,
)
from dendrux.runtime.state import (
    _finite_json_float as _finite_json_float,
)
from dendrux.runtime.state import (
    _loop_state_at_creation as _loop_state_at_creation,
)
from dendrux.runtime.state import (
    _run_to_record as _run_to_record,
)
from dendrux.runtime.state import (
    _usage_totals as _usage_totals,
)
from dendrux.runtime.state import (
    _write_event as _write_event,
)
from dendrux.runtime.stores.sql import _sql_sessions, _sql_support, _sql_sweep_scan
from dendrux.runtime.stores.sql._managed_reads import ManagedReads
from dendrux.runtime.stores.sql._sql_evidence import _SqlEvidence
from dendrux.runtime.stores.sql._sql_host import _StoreHost
from dendrux.runtime.stores.sql._sql_lifecycle import _SqlLifecycle
from dendrux.runtime.stores.sql._sql_managed import _active_wait_id as _active_wait_id
from dendrux.runtime.stores.sql._sql_managed import _bound_child_id as _bound_child_id
from dendrux.runtime.stores.sql._sql_managed import _child_matches as _child_matches
from dendrux.runtime.stores.sql._sql_managed import _existing_wait_id as _existing_wait_id
from dendrux.runtime.stores.sql._sql_managed import _SqlManaged
from dendrux.runtime.stores.sql._sql_pause import _SqlPause
from dendrux.runtime.stores.sql._sql_reads import _TRACE_POSITION_STRIDE as _TRACE_POSITION_STRIDE
from dendrux.runtime.stores.sql._sql_reads import _SqlReads
from dendrux.runtime.stores.sql._sql_runs import _SqlRuns
from dendrux.runtime.stores.sql._sql_sessions import (
    _CheckedSessionFactory as _CheckedSessionFactory,
)
from dendrux.runtime.stores.sql._sql_sessions import (
    _SerializedSessionFactory as _SerializedSessionFactory,
)
from dendrux.runtime.stores.sql._sql_support import (
    _RECOVERY_DESCRIPTOR_BYTES as _RECOVERY_DESCRIPTOR_BYTES,
)
from dendrux.runtime.stores.sql._sql_support import _RECOVERY_MAX_MESSAGES as _RECOVERY_MAX_MESSAGES
from dendrux.runtime.stores.sql._sql_support import _RECOVERY_TRACE_BYTES as _RECOVERY_TRACE_BYTES
from dendrux.runtime.stores.sql._sql_support import (
    PAUSE_INVALIDATION_EVENT_TYPES as PAUSE_INVALIDATION_EVENT_TYPES,
)
from dendrux.runtime.stores.sql._sql_support import _advancing_turn as _advancing_turn
from dendrux.runtime.stores.sql._sql_support import (
    _check_create_identifiers as _check_create_identifiers,
)
from dendrux.runtime.stores.sql._sql_support import _checkpoint_error as _checkpoint_error
from dendrux.runtime.stores.sql._sql_support import _checkpoint_progress as _checkpoint_progress
from dendrux.runtime.stores.sql._sql_support import _creation_mark_proven as _creation_mark_proven
from dendrux.runtime.stores.sql._sql_support import _creation_metadata as _creation_metadata
from dendrux.runtime.stores.sql._sql_support import _final_output_values as _final_output_values
from dendrux.runtime.stores.sql._sql_support import _final_progress as _final_progress
from dendrux.runtime.stores.sql._sql_support import _json_bytes as _json_bytes
from dendrux.runtime.stores.sql._sql_support import _merge_public_meta as _merge_public_meta
from dendrux.runtime.stores.sql._sql_support import _note_blocked_event as _note_blocked_event
from dendrux.runtime.stores.sql._sql_support import _positive_page as _positive_page
from dendrux.runtime.stores.sql._sql_support import _progress_values as _progress_values
from dendrux.runtime.stores.sql._sql_support import (
    _refuse_direct_managed_child as _refuse_direct_managed_child,
)
from dendrux.runtime.stores.sql._sql_support import _status_values as _status_values
from dendrux.runtime.stores.sql._sql_support import _submission_progress as _submission_progress
from dendrux.runtime.stores.sql._sql_support import _trace_record as _trace_record
from dendrux.runtime.stores.sql._sql_support import _validate_checkpoint as _validate_checkpoint
from dendrux.runtime.stores.sql._sql_support import (
    _validate_lease_batch_size as _validate_lease_batch_size,
)
from dendrux.runtime.stores.sql._sql_support import (
    _validate_rewrite_entries as _validate_rewrite_entries,
)
from dendrux.runtime.stores.sql._sql_support import _validate_trace_write as _validate_trace_write
from dendrux.runtime.stores.sql._sql_support import logger as logger
from dendrux.runtime.stores.sql._sql_sweep_scan import _WAIT_MARK_LABELS as _WAIT_MARK_LABELS
from dendrux.runtime.stores.sql._sql_sweep_scan import _SweepScan as _SweepScan
from dendrux.runtime.stores.sql._sql_tree import _SqlTree
from dendrux.runtime.stores.sql._sql_writes import _SqlWrites
from dendrux.runtime.stores.sql._sql_writes import _UpdatedRows as _UpdatedRows
from dendrux.runtime.stores.sql._types import (
    ConflictSpec,
    RunLockScope,
    SelectSource,
)
from dendrux.runtime.stores.sql._types import (
    PatchLoopState as PatchLoopState,
)

_COMMIT_READBACK_LOCK_TIMEOUT_MS = 5000


async def _write_release_event(
    session: AsyncSession, run_id: str, won: Any, event: EventWrite | None
) -> None:
    if won and event is not None:
        await _state._write_event(session, run_id, event)


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
    for name, method in getmembers(cls, iscoroutinefunction):
        if name.startswith(
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
class SQLStateStore(
    _SqlRuns,
    _SqlManaged,
    _SqlTree,
    _SqlPause,
    _SqlReads,
    _SqlLifecycle,
    _SqlEvidence,
    _SqlWrites,
    _StoreHost,
    ManagedReads,
    SQLBlobCatalog,
):
    """Default StateStore implementation using SQLAlchemy async.

    Works with SQLite (zero-config) and Postgres (via an explicit engine).
    Also the blob catalog (``dendrux.media.BlobCatalog``) for image blobs.

    ``create_run``, ``save_trace``, ``save_tool_call`` and ``submit_and_claim``
    take optional ``blob_refs`` + ``blob_refs_scope``: the record's blob edges,
    written in the record's transaction after asserting the run's tenant and
    claiming each blob row (``write_blob_refs``). Passed only when non-empty.
    """

    _note_lock_requires_scope_precheck = False

    dialect_names: tuple[str, ...] = ()
    supports_update_returning = False
    supports_insert_returning = False
    supports_row_locks = True
    ddl_transactional = False
    _requires_persisted_claim = True

    @property
    def _refuses_managed_child_reclaim(self) -> bool:
        """This store's own claim refuses a direct managed-child reclaim (0.7)."""
        return type(self).reclaim_run is SQLStateStore.reclaim_run

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
        recovery=True,
        managed_subagents=True,
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

    async def _prepare_commit_readback(self, session: AsyncSession) -> None:
        pass

    async def connection_checks(self, conn: Any) -> DialectFacts:
        from dendrux.runtime.stores.sql._types import DialectFacts

        return DialectFacts(conn.dialect.name, tuple(conn.dialect.server_version_info or ()))

    async def reserve_parent_writer(self, session: AsyncSession, run_id: str) -> None:
        return None

    def blob_now(self) -> Any:
        return _dt.datetime.now(_dt.UTC)

    def json_get(self, col: Any, path: Any, as_type: Any) -> Any:
        raise NotImplementedError("SQL adapters must implement json_get")

    def json_contains(self, col: Any, path: Any, value: Any) -> Any:
        raise NotImplementedError("SQL adapters must implement json_contains")

    def json_is_null(self, col: Any, path: Any) -> Any:
        raise NotImplementedError("SQL adapters must implement json_is_null")

    def json_patch_expr(self, col: Any, patch: Any) -> Any:
        return None

    def pause_projection(self, col: Any) -> list[Any]:
        return [self.json_get(col, (key,), "json") for key in PAUSE_SUMMARY_KEYS] + [
            self.json_get(col, ("steps", -1), "json")
        ]

    def json_conversion_error(self, exc: BaseException) -> bool:
        return False

    def upsert_setting(self, table: Any, *, key: str, value: str, updated_at: datetime) -> Any:
        raise NotImplementedError("SQL adapters must implement upsert_setting")

    def child_page_query(
        self, *, runs: Any, parents: Any, tenant_scope: Any, page_size: int
    ) -> Any:
        from sqlalchemy import and_, or_, select

        _sql_support._positive_page(page_size)
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

    def __init__(
        self, engine: AsyncEngine, *, live_updates: LiveUpdates | bool | None = None
    ) -> None:
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
                STORE_SESSION_KEY: self,
            },
        )
        if shares_one_connection(engine):
            # One shared connection = one shared transaction: overlapping sessions
            # would roll back each other's pending writes (see db/_pool_guard.py).
            warn_if_shared_connection(engine)
            self._session_factory = _sql_sessions._SerializedSessionFactory(
                engine, self._session_factory
            )
        self._session_factory = _sql_sessions._CheckedSessionFactory(self, self._session_factory)
        self._live_updates = self._bind_live_updates(live_updates)

    @property
    def live_updates(self) -> LiveUpdates | None:
        """The live update bus owned by this store, if enabled."""
        return self._live_updates

    def _native_live_updates(self) -> LiveUpdates | None:
        return None

    def _bind_live_updates(self, updates: LiveUpdates | bool | None) -> LiveUpdates | None:
        if updates is None or updates is False:
            return None
        from dendrux.runtime.stores.sql._live_registry import register

        updates = self._resolve_live_updates(updates)
        updates.bind(self)
        register(self)
        return updates

    def _resolve_live_updates(self, updates: LiveUpdates | bool) -> LiveUpdates:
        from dendrux.db._live_updates import LiveUpdates

        resolved = self._native_live_updates() if updates is True else updates
        if not isinstance(resolved, LiveUpdates):
            raise ValueError(
                f"{type(self).__name__} requires a LiveUpdates instance; use RedisUpdates"
            )
        return resolved

    async def close(self) -> None:
        """Close store-owned live updates without disposing the engine."""
        from dendrux.runtime.stores.sql._live_registry import deregister

        if self._live_updates is not None:
            await self._live_updates.close()
            deregister(self)

    async def _check_connection(self, session: AsyncSession) -> None:
        if self._dialect_facts is not None:
            return
        async with self._connection_check_lock:
            if self._dialect_facts is not None:
                return
            conn = await session.connection()
            if conn.dialect.name in {"postgresql", "sqlite"}:

                def autocommit(sync: Any) -> bool:
                    dbapi = sync.connection.dbapi_connection
                    return getattr(dbapi, "autocommit", False) is True or (
                        sync.dialect.name == "sqlite"
                        and dbapi.isolation_level is None
                        and not dbapi.driver_connection.in_transaction
                    )

                if await conn.run_sync(autocommit):
                    raise ValueError(
                        "SQLStateStore requires transactions; AUTOCOMMIT is not supported"
                    )
            facts = await self.connection_checks(conn)
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
            values = {"loop_state": changed, **_sql_support._submission_progress(current, traces)}
            self._remember_run_values(session, run_id, values)
            _sql_support._validate_checkpoint(run_id, changed)
            pause = dict(row.pause_data or {})
            pause.pop("submission", None)
            for trace in traces:
                await self._insert_trace(session, run_id, trace)
            await session.execute(
                update(AgentRun).where(AgentRun.id == run_id).values(**values, pause_data=pause)
            )
            self._mark_announce(session, (run_id,))
            await self._write_tx(session)
            return True

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
                                **_sql_support._status_values(RunStatus.RUNNING.value),
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
                    self._mark_announce(session, (run_id,))
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
                                **_sql_support._status_values(RunStatus.RUNNING.value),
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
                self._mark_announce(session, (run_id,))
                await self._write_tx(session)
                return token if result.rowcount else False

        return await retry_transient_db(
            _attempt,
            label="claim_paused_run",
            run_id=run_id,
            classify=self.classify_db_error,
            _raw_driver_errors=True,
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
                                **_sql_support._status_values(RunStatus.RUNNING.value),
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
                if run.storage_format == STORAGE_FORMAT_CHECKPOINTED:
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
                    _sql_support._validate_checkpoint(run_id, loop_state)
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
                    await _blobs.write_blob_refs(
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
                self._mark_announce(session, (run_id,))
                await self._write_tx(session)
                return token

        return await retry_transient_db(
            _attempt,
            label="submit_and_claim",
            run_id=run_id,
            classify=self.classify_db_error,
            _raw_driver_errors=True,
        )

    async def _finish_claim_release(
        self,
        session: AsyncSession,
        run_id: str,
        claim_epoch: int,
        won: bool,
        event: EventWrite | None,
    ) -> bool:
        if won:
            await _write_release_event(session, run_id, won, event)
            return True
        return await self._cancel_claim_release(session, run_id, claim_epoch)

    async def _cancel_claim_release(
        self, session: AsyncSession, run_id: str, claim_epoch: int
    ) -> bool:
        from dendrux.db.models import AgentRun

        requested = await session.scalar(
            self.lock_clause(
                select(AgentRun.cancel_requested).where(
                    AgentRun.id == run_id,
                    AgentRun.status == RunStatus.RUNNING.value,
                    AgentRun.claim_epoch == claim_epoch,
                    AgentRun.cancel_requested.is_(True),
                ),
                "update",
            )
        )
        if not requested:
            return False
        won = await self._cas_finalize(
            session,
            run_id,
            status=RunStatus.CANCELLED.value,
            allowed_current_statuses=[RunStatus.RUNNING.value],
            claim_epoch=claim_epoch,
            answer=None,
            error=None,
            iteration_count=None,
            total_usage=None,
            pii_mapping=None,
        )
        if won:
            await _state._insert_event_locked(
                session,
                run_id,
                StoredEventType.RUN_CANCELLED.value,
                {"reason": CancelReason.CANCEL_REQUESTED.value},
            )
        return won

    async def release_run_claim(
        self, run_id: str, *, status: str, claim_epoch: int, event: EventWrite | None = None
    ) -> bool:
        """Return an owned claim to its pause after post-claim validation fails."""
        from sqlalchemy import update

        from dendrux.db.models import AgentRun

        async def attempt() -> bool:
            async with self._session_factory() as session:
                result = await session.execute(
                    update(AgentRun)
                    .where(
                        AgentRun.id == run_id,
                        AgentRun.status == RunStatus.RUNNING.value,
                        AgentRun.claim_epoch == claim_epoch,
                        AgentRun.cancel_requested.is_(False),
                    )
                    .ordered_values(
                        *self._ordered(
                            AgentRun.__table__, dict(**_sql_support._status_values(status))
                        )
                    )
                )
                won = await self._finish_claim_release(
                    session, run_id, claim_epoch, bool(result.rowcount), event
                )
                self._mark_announce(session, (run_id,))
                await self._write_tx(session)
                return won

        return await retry_transient_db(
            attempt,
            label="release_run_claim",
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

    # ------------------------------------------------------------------
    # Progress tracking & stale-run sweep
    # ------------------------------------------------------------------

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
        _sql_support._validate_lease_batch_size(
            batch_size=batch_size,
            cap=SWEEP_BATCH_MAX,
        )
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
            _sql_support._positive_page(bound)
        if max(batch_size, max_runs, max_returned) > RECOVERY_SCAN_MAX:
            _codes.log_code(logger, _codes.RECOVERY_SCAN_LIMIT, run_id=None)
            raise ValueError("Recovery scan bounds must not exceed 100000")
        if reclaim_limit > SWEEP_MAX_RECLAIMS_MAX:
            raise ValueError("reclaim_limit exceeds SMALLINT")
        result = MarkResult([], [], [], 0, None)
        inspected, after = 0, cursor
        started = (
            select(RunEvent.id)
            .where(
                RunEvent.agent_run_id == AgentRun.id,
                RunEvent.event_type == literal_column(f"'{StoredEventType.RUN_STARTED.value}'"),
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
            statement = select(*fields).where(
                AgentRun.status.between(RunStatus.RUNNING.value, RunStatus.RUNNING.value)
            )
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
            async with _live_updates_page(self):
                for candidate in ids:
                    run_id = candidate.id

                    async def attempt(
                        row: Any = candidate, run_id: str = run_id
                    ) -> ReclaimableRun | SweptRun | None:
                        async with self._session_factory() as session:
                            age = _to_aware_utc(row.last_progress_at or row.created_at)
                            if (
                                on_stale == "error" or row.lease_expired_at is None
                            ) and age >= cutoff:
                                return None
                            if (
                                row.storage_format == STORAGE_FORMAT_CHECKPOINTED
                                and row.loop
                                not in (
                                    "react",
                                    "single_call",
                                )
                            ):
                                if on_stale != "error":
                                    _codes.log_code(logger, _codes.LOOP_UNKNOWN, run_id=run_id)
                                if on_stale is None:
                                    return None
                            limit = (
                                row.reclaim_limit
                                if row.lease_expired_at is not None
                                and row.reclaim_limit is not None
                                else reclaim_limit
                            )
                            legacy_reason = (
                                FailureReason.STALE_RUNNING.value
                                if row.started or on_stale is None
                                else FailureReason.NEVER_STARTED.value
                            )
                            reason = (
                                legacy_reason
                                if on_stale == "error"
                                else FailureReason.STALE_RUNNING_LEGACY.value
                                if row.storage_format != STORAGE_FORMAT_CHECKPOINTED
                                else FailureReason.RECLAIM_LIMIT.value
                                if row.reclaim_attempts >= limit
                                else legacy_reason
                                if row.loop != "react"
                                else None
                            )
                            if reason is not None or row.cancel_requested:
                                flagged = AgentRun.cancel_requested.is_(True)
                                status_expr = case(
                                    (flagged, RunStatus.CANCELLED.value),
                                    else_=RunStatus.ERROR.value,
                                )
                                changed = (
                                    await self._update_one_rows(
                                        session,
                                        table=AgentRun.__table__,
                                        pk=(run_id,),
                                        predicate=and_(
                                            AgentRun.id == run_id,
                                            AgentRun.status == RunStatus.RUNNING.value,
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
                                            **_sql_support._status_values(status_expr),
                                            claim_epoch=AgentRun.claim_epoch + 1,
                                            failure_reason=reason,
                                            error=case(
                                                (flagged, None), else_=f"Run swept as {reason}"
                                            ),
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
                                await _state._insert_event_locked(
                                    session,
                                    run_id,
                                    StoredEventType.RUN_CANCELLED.value
                                    if status == RunStatus.CANCELLED.value
                                    else StoredEventType.RUN_INTERRUPTED.value,
                                    {
                                        "failure_reason": reason,
                                        **(
                                            {"reason": CancelReason.CANCEL_REQUESTED.value}
                                            if status == RunStatus.CANCELLED.value
                                            else {}
                                        ),
                                    },
                                    iteration_index=row.iteration_count or 0,
                                )
                                self._mark_announce(session, (run_id,))
                                await self._write_tx(session)
                                _codes.log_code(
                                    logger,
                                    _codes.REGISTRY[
                                        "legacy_run_swept"
                                        if reason == FailureReason.STALE_RUNNING_LEGACY.value
                                        else reason or CancelReason.CANCEL_REQUESTED.value
                                    ],
                                    run_id=run_id,
                                )
                                return SweptRun(
                                    run_id,
                                    row.agent_name,
                                    RunStatus.RUNNING.value,
                                    reason or CancelReason.CANCEL_REQUESTED.value,
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
                                        AgentRun.status == RunStatus.RUNNING.value,
                                        AgentRun.storage_format == STORAGE_FORMAT_CHECKPOINTED,
                                        AgentRun.lease_expired_at.is_(None),
                                        AgentRun.claim_epoch == row.claim_epoch,
                                        func.coalesce(
                                            AgentRun.last_progress_at, AgentRun.created_at
                                        )
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
                                            AgentRun.status == RunStatus.RUNNING.value,
                                            AgentRun.lease_expired_at.is_not(None),
                                        )
                                    )
                                ).first()
                                return _reclaimable(current) if current is not None else None
                            await _state._insert_event_locked(
                                session,
                                run_id,
                                StoredEventType.RUN_LEASE_EXPIRED.value,
                                {"gap_s": max(0.0, (now - age).total_seconds())},
                            )
                            item = _reclaimable(marked)
                            self._mark_announce(session, (run_id,))
                            await self._write_tx(session)
                            return item

                    item = await retry_transient_db(
                        attempt,
                        label="mark_lease_expired",
                        run_id=run_id,
                        classify=self.classify_db_error,
                        _raw_driver_errors=True,
                    )
                    inspected += 1
                    after = run_id
                    if isinstance(item, ReclaimableRun):
                        result.reclaimable.append(item)
                    elif isinstance(item, SweptRun):
                        result.finalised.append(item)
                        if item.failure_reason == FailureReason.STALE_RUNNING_LEGACY.value:
                            result.legacy.append(item)
                    if inspected >= max_runs or len(result.reclaimable) >= max_returned:
                        if inspected >= max_runs:
                            _codes.log_code(
                                logger,
                                _codes.SWEEP_CAP_REACHED,
                                category=FailureReason.STALE_RUNNING.value,
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

        _sql_support._positive_page(limit)
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
            AgentRun.status == RunStatus.RUNNING.value,
            AgentRun.storage_format == STORAGE_FORMAT_CHECKPOINTED,
            AgentRun.lease_expired_at.is_not(None),
        )
        if cursor is not None:
            stmt = stmt.where(AgentRun.id > cursor)
        async with self._session_factory() as session:
            return [
                _reclaimable(row)
                for row in (
                    await session.execute(stmt.order_by(AgentRun.id).limit(min(limit, MAX_PAGE)))
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
                FailureReason.RECLAIM_LIMIT.value,
                CancelReason.CANCEL_REQUESTED.value,
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
                _sql_support._refuse_direct_managed_child(run_id, row.loop_state)
                checks = [
                    (row.storage_format == STORAGE_FORMAT_CHECKPOINTED, "legacy_run"),
                    (self.capabilities.recovery, "store_capability"),
                    (row.status == RunStatus.RUNNING.value, "not_running"),
                    (row.lease_expired_at is not None, "lease_fresh"),
                    (
                        expected_claim_epoch is None or row.claim_epoch == expected_claim_epoch,
                        "already_reclaimed",
                    ),
                    (
                        row.reclaim_limit is not None and row.reclaim_attempts < row.reclaim_limit,
                        FailureReason.RECLAIM_LIMIT.value,
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
                                **_sql_support._status_values(RunStatus.CANCELLED.value),
                                "claim_epoch": table.c.claim_epoch + 1,
                                "pause_data": None,
                                "cancel_requested": False,
                            },
                        )
                    )
                    return
                if row.agent_name != agent_name:
                    raise RunNotReclaimableError(run_id, "agent_mismatch")
                saved = (row.meta or {}).get(GOVERNANCE_KEY)
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
                session.info[STORE_SESSION_KEY] = self
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
                    await _state._insert_event_locked(
                        session,
                        run_id,
                        StoredEventType.RUN_CANCELLED.value,
                        {"reason": CancelReason.CANCEL_REQUESTED.value},
                    )
                    self._mark_announce(session, (run_id,))
                    await self._write_tx(session)
                    raise RunNotReclaimableError(run_id, "cancelled")
                await _state._insert_event_locked(
                    session,
                    run_id,
                    StoredEventType.RUN_RECLAIMED.value,
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
                self._mark_announce(session, (run_id,))
                await self._write_tx(session)
                return token

        return await retry_transient_db(
            attempt,
            label="reclaim_run",
            run_id=run_id,
            classify=self.classify_db_error,
            _raw_driver_errors=True,
        )

    def _waiting_sweep_page(
        self, statement: Any, after: tuple[datetime, str] | None, limit: int
    ) -> Any:
        from sqlalchemy import tuple_

        from dendrux.db.models import AgentRun

        if after is not None:
            statement = statement.where(tuple_(AgentRun.status_changed_at, AgentRun.id) > after)
        return statement

    async def sweep_stale_runs(
        self,
        older_than: timedelta,
        *,
        batch_size: int = SWEEP_BATCH_DEFAULT,
        max_runs: int = SWEEP_MAX_RUNS_DEFAULT,
    ) -> list[SweptRun]:
        """Inspect running candidates in projected pages; cap 10,000 by default.

        max_runs accepts 1..100,000. Fresh rows and losing CAS attempts count
        toward it; hitting the cap emits sweep_cap_reached.
        """
        return cast(
            "list[SweptRun]",
            (
                await self._sweep_runs(
                    older_than, waiting=False, batch_size=batch_size, max_runs=max_runs
                )
            )[0],
        )

    async def sweep_abandoned_runs(
        self,
        older_than: timedelta,
        *,
        batch_size: int = SWEEP_BATCH_DEFAULT,
        max_runs: int = SWEEP_MAX_RUNS_DEFAULT,
    ) -> list[SweptRun]:
        """Inspect waiting candidates in projected pages; cap 10,000 by default.

        max_runs accepts 1..100,000 across all waiting statuses. The final CAS
        requires both status_changed_at and updated_at to be stale. Losing CAS
        attempts count toward the cap; hitting it emits sweep_cap_reached.
        """
        return cast(
            "list[SweptRun]",
            (
                await self._sweep_runs(
                    older_than, waiting=True, batch_size=batch_size, max_runs=max_runs
                )
            )[0],
        )

    @property
    def _uses_waiting_sweep_cursor(self) -> bool:
        return type(self).sweep_abandoned_runs is SQLStateStore.sweep_abandoned_runs

    async def _scan_abandoned_runs(
        self,
        older_than: timedelta,
        *,
        batch_size: int,
        max_runs: int,
        cursor: tuple[str, datetime, str] | None,
    ) -> tuple[list[SweptRun], tuple[str, datetime, str] | None]:
        return cast(
            "tuple[list[SweptRun], tuple[str, datetime, str] | None]",
            await self._sweep_runs(
                older_than, waiting=True, batch_size=batch_size, max_runs=max_runs, cursor=cursor
            ),
        )

    @_safe_database_call
    async def _sweep_runs(
        self,
        older_than: timedelta,
        *,
        waiting: bool,
        batch_size: int,
        max_runs: int,
        cursor: tuple[str, datetime, str] | None = None,
    ) -> tuple[list[SweptRun], tuple[str, datetime, str] | None]:
        for name, value, ceiling in (
            ("batch_size", batch_size, SWEEP_BATCH_MAX),
            ("max_runs", max_runs, SWEEP_MAX_RUNS_MAX),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= ceiling:
                raise ValueError(f"{name} must be between 1 and {ceiling}")
        _validate_sweep_age(older_than)
        scan = _sql_sweep_scan._SweepScan(self, waiting=waiting, older_than=older_than)
        swept: list[SweptRun] = []
        visited = 0
        statuses = list(PAUSE_STATUS_VALUES) if waiting else [RunStatus.RUNNING.value]
        after = None
        if cursor is not None:
            statuses = statuses[statuses.index(cursor[0]) :]
            after = cursor[1:]
        for status_filter in statuses:
            done, count, end = await scan.collect(
                status_filter, after, batch_size, max_runs - visited
            )
            swept.extend(done)
            visited += count
            if visited == max_runs:
                _codes.log_code(
                    logger,
                    _codes.SWEEP_CAP_REACHED,
                    category=FailureReason.ABANDONED_WAITING.value
                    if waiting
                    else FailureReason.STALE_RUNNING.value,
                    cap=max_runs,
                    inspected=visited,
                )
                return swept, (status_filter, *end) if waiting else None
            after = None
        return swept, None

    # ------------------------------------------------------------------
    # Delegation queries
    # ------------------------------------------------------------------

    _MAX_LIST_LIMIT = MAX_PAGE

    # JSON keys are limited to a conservative identifier shape so the
    # path string we splice into the SQL (``$.<key>`` for SQLite) cannot
    # smuggle path operators or quotes. Values are always parameter-bound,
    # so only the key is sensitive here.
    _METADATA_KEY_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

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
