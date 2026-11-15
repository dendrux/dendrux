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
from typing import TYPE_CHECKING, Any, Literal

from sqlalchemy import true
from sqlalchemy.exc import SQLAlchemyError as SQLAlchemyError

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
    LATEST_REWRITE_META_KEY as LATEST_REWRITE_META_KEY,
)
from dendrux._meta_keys import (
    OBSERVATION_META_KEY as OBSERVATION_META_KEY,
)
from dendrux._meta_keys import (
    STOP_REASON_META_KEY as STOP_REASON_META_KEY,
)
from dendrux._meta_keys import (
    STORE_SESSION_KEY,
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
)
from dendrux.runtime._bounds import MAX_SMALL_PAGE as MAX_SMALL_PAGE
from dendrux.runtime._bounds import MAX_SUBTREE_DEPTH as MAX_SUBTREE_DEPTH
from dendrux.runtime._bounds import MAX_SUBTREE_ROWS as MAX_SUBTREE_ROWS
from dendrux.runtime.durability import (
    _CommitNotAppliedError as _CommitNotAppliedError,
)
from dendrux.runtime.durability import (
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
    UsageStats as UsageStats,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from dendrux.db._live_updates import LiveUpdates
    from dendrux.runtime.stores._errors import DbError
    from dendrux.runtime.stores.sql._types import DialectFacts, InsertPlan

from dendrux.runtime.state import (
    _PERSISTED_VALUE_FIELDS,
    PAUSE_SUMMARY_KEYS,
    LoopStatePatch,
    RunControl,
    StoreCapabilities,
    _apply_tenant_scope,
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
from dendrux.runtime.stores.sql import _sql_sessions, _sql_support
from dendrux.runtime.stores.sql._managed_reads import ManagedReads
from dendrux.runtime.stores.sql._sql_claims import _SqlClaims
from dendrux.runtime.stores.sql._sql_claims import _write_release_event as _write_release_event
from dendrux.runtime.stores.sql._sql_events import _SqlEvents
from dendrux.runtime.stores.sql._sql_evidence import _SqlEvidence
from dendrux.runtime.stores.sql._sql_host import _StoreHost
from dendrux.runtime.stores.sql._sql_leases import _SqlLeases
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
from dendrux.runtime.stores.sql._sql_sweep import _SqlSweep
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
    _SqlEvents,
    _SqlClaims,
    _SqlLifecycle,
    _SqlEvidence,
    _SqlLeases,
    _SqlSweep,
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

    # ------------------------------------------------------------------
    # Progress tracking & stale-run sweep
    # ------------------------------------------------------------------

    @property
    def _uses_waiting_sweep_cursor(self) -> bool:
        return type(self).sweep_abandoned_runs is SQLStateStore.sweep_abandoned_runs

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
