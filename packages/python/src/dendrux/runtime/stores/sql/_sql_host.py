"""Specify the typed execution context required by SQL operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    import re
    from abc import abstractmethod
    from collections.abc import Sequence
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession

    from dendrux.db._live_updates import LiveUpdates
    from dendrux.runtime.state import StoreCapabilities
    from dendrux.runtime.stores._errors import DbError
    from dendrux.runtime.stores.sql._types import ConflictSpec, InsertPlan, RunLockScope


class _StoreHost:
    """Runtime annotation-only context; signatures exist only for static checking."""

    _MAX_LIST_LIMIT: int
    _METADATA_KEY_RE: re.Pattern[str]
    _format2_active: bool
    _live_updates: LiveUpdates | None
    _note_lock_requires_scope_precheck: bool
    _session_factory: Any
    capabilities: StoreCapabilities
    supports_insert_returning: bool
    supports_row_locks: bool
    supports_update_returning: bool

    if TYPE_CHECKING:

        @abstractmethod
        async def _check_connection(self, session: AsyncSession) -> None: ...

        @abstractmethod
        def _idempotency_violation(self, exc: BaseException) -> bool: ...

        @abstractmethod
        async def _prepare_commit_readback(self, session: AsyncSession) -> None: ...

        @abstractmethod
        def child_page_query(
            self, *, runs: Any, parents: Any, tenant_scope: Any, page_size: int
        ) -> Any: ...

        @abstractmethod
        def classify_db_error(self, exc: BaseException) -> DbError: ...

        @abstractmethod
        def event_sequence_expr(self, next_index: Any) -> Any: ...

        @abstractmethod
        def id_in(self, col: Any, ids: Sequence[Any]) -> Any: ...

        @abstractmethod
        def insert_if_absent_stmt(
            self, table: Any, source: Any, *, conflict: ConflictSpec, returning: Sequence[Any] = ()
        ) -> InsertPlan: ...

        @abstractmethod
        def inserted_event_sequence(self, result: Any) -> int | None: ...

        @abstractmethod
        def json_conversion_error(self, exc: BaseException) -> bool: ...

        @abstractmethod
        def json_get(self, col: Any, path: Any, as_type: Any) -> Any: ...

        @abstractmethod
        def json_patch_expr(self, col: Any, patch: Any) -> Any: ...

        @abstractmethod
        def lock_clause(
            self, stmt: Any, intent: Literal["share", "update", "update_no_key"]
        ) -> Any: ...

        @abstractmethod
        def metadata_filter_predicate(self, col: Any, metadata_filter: dict[str, Any]) -> Any: ...

        @abstractmethod
        def pause_projection(self, col: Any) -> list[Any]: ...

        @abstractmethod
        async def reserve_parent_writer(self, session: AsyncSession, run_id: str) -> None: ...

        @abstractmethod
        async def serialize_in_tx(self, session: AsyncSession, *, owner: RunLockScope) -> None: ...


class _SweepStore(Protocol):
    _session_factory: Any

    def _mark_announce(self, session: AsyncSession, run_ids: Sequence[str]) -> None: ...

    async def _update_one_rows(
        self,
        session: AsyncSession,
        *,
        table: Any,
        pk: tuple[Any, ...],
        predicate: Any,
        values: dict[str, Any],
        returning: Sequence[Any],
    ) -> Any: ...

    def _waiting_sweep_page(
        self, statement: Any, after: tuple[datetime, str] | None, limit: int
    ) -> Any: ...

    async def _write_tx(self, session: AsyncSession, *, run_ids: Sequence[str] = ()) -> None: ...

    def json_get(self, col: Any, path: Any, as_type: Any) -> Any: ...
