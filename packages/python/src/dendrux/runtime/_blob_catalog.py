"""Blob catalog, lifecycle, resolver and accessor (0.4.0 image persistence).

Internal module. Implements the ``media`` interfaces against storage:

- :class:`SQLBlobCatalog` — the :class:`~dendrux.media.BlobCatalog` over the
  ``dendrux_blobs`` / ``dendrux_blob_refs`` tables. ``SQLAlchemyStateStore``
  inherits it. Every lifecycle transition is one conditional statement keyed
  by ``(tenant_scope, sha256)`` (and ``generation`` where it matters).
- :func:`write_blob_refs` — the edge-write protocol run inside a record's
  transaction (tenant assertion, claim by conditional UPDATE, idempotent
  edge inserts).
- :class:`InMemoryBlobCatalog` — the same state machine over dicts, for
  agents without a database (non-durable; edges released at run end).
- :class:`CachingBlobResolver` — verified reads with a generation-keyed LRU.
- :class:`CatalogBlobAccessor` — ``agent.blobs`` / ``RunStore.blobs``: read,
  info, references, purge and garbage collection (incl. crash recovery).
- :class:`BlobServices` — wires the above for a state store or for memory.

Clock: on Postgres every blob timestamp and every cutoff uses the database
clock (``now()``), so a host with a skewed clock cannot shrink the GC grace.
SQLite (one host, one file) uses Python UTC, matching the stored format.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

from dendrux.errors import (
    BlobNotFoundError,
    ImageLimitError,
    ImagePersistenceUnsupportedError,
    InvalidImageError,
)
from dendrux.media._catalog import BlobEdge
from dendrux.media._ids import image_id
from dendrux.media._runtime import tenant_scope_for
from dendrux.media._sniff import verify_blob_bytes
from dendrux.media._store import (
    BlobInfo,
    BlobKey,
    BlobReference,
    GcResult,
    InMemoryBlobStore,
)
from dendrux.runtime.durability import retry_transient_db
from dendrux.types import ImagePart, ImageRef, generate_ulid

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from dendrux.media._catalog import BlobCatalog, BlobResolver
    from dendrux.media._store import BlobState, BlobStore

logger = logging.getLogger(__name__)

__all__ = [
    "BlobServices",
    "CachingBlobResolver",
    "CatalogBlobAccessor",
    "GcOnFullBlobStore",
    "InMemoryBlobCatalog",
    "SQLBlobCatalog",
    "as_blob_catalog",
    "edges_for_parts",
    "write_blob_refs",
]

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IN_CHUNK = 500
DEFAULT_CACHE_BYTES = 128 * 1024 * 1024
MIN_GC_AGE = timedelta(hours=1)
_PURGE_REASON_MAX = 200
_TRANSIENT_STATES = ("pending", "deleting")
_READER_STATES = ("live", "purging", "purged")


def _py_now() -> datetime:
    return datetime.now(UTC)


def _aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _require_aware(dt: datetime) -> datetime:
    out = _aware(dt)
    assert out is not None
    return out


def _chunks(items: Sequence[str]) -> list[Sequence[str]]:
    return [items[i : i + _IN_CHUNK] for i in range(0, len(items), _IN_CHUNK)]


def _check_sha(sha256: str) -> str:
    if not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256):
        raise ValueError("sha256 must be 64 lowercase hex characters")
    return sha256


def as_blob_catalog(store: Any) -> BlobCatalog | None:
    """``store`` if it implements the blob catalog (built-in SQL store, memory), else None."""
    needed = ("get_current", "touch_live", "insert_row", "mark_purging", "run_references")
    if store is not None and all(hasattr(store, name) for name in needed):
        return store  # type: ignore[no-any-return]
    return None


# ---------------------------------------------------------------------------
# Edges for persisted parts / image lists
# ---------------------------------------------------------------------------


def edges_for_parts(
    scope: str, owner_kind: str, owner_id: str, part_dicts: Sequence[dict[str, Any]]
) -> list[BlobEdge]:
    """One edge per image dict, ``ordinal`` = its index in ``part_dicts``."""
    return [
        BlobEdge(
            tenant_scope=scope,
            sha256=d["sha256"],
            owner_kind=owner_kind,
            owner_id=owner_id,
            ordinal=i,
        )
        for i, d in enumerate(part_dicts)
        if isinstance(d, dict) and d.get("type") == "image"
    ]


# ---------------------------------------------------------------------------
# SQL catalog
# ---------------------------------------------------------------------------


def _blob_info_columns() -> list[Any]:
    from dendrux.db.models import Blob

    return [
        Blob.sha256,
        Blob.media_type,
        Blob.byte_length,
        Blob.width,
        Blob.height,
        Blob.frame_count,
        Blob.storage,
        Blob.state,
        Blob.generation,
        Blob.created_at,
        Blob.last_used_at,
        Blob.purged_at,
        Blob.purge_reason,
        Blob.ingest_transform,
        Blob.state_changed_at,
    ]


def _row_to_info(row: Any) -> BlobInfo:
    return BlobInfo(
        sha256=row.sha256,
        media_type=row.media_type,
        byte_length=int(row.byte_length),
        width=int(row.width),
        height=int(row.height),
        frame_count=int(row.frame_count),
        storage=row.storage,
        state=row.state,
        generation=row.generation,
        created_at=_require_aware(row.created_at),
        last_used_at=_require_aware(row.last_used_at),
        purged_at=_aware(row.purged_at),
        purge_reason=row.purge_reason,
        ingest_transform=row.ingest_transform,
        state_changed_at=_aware(row.state_changed_at),
    )


def _db_now(dialect_name: str) -> Any:
    """Timestamp value for a write: ``now()`` on Postgres, Python UTC elsewhere."""
    if dialect_name == "postgresql":
        from sqlalchemy import func

        return func.now()
    return _py_now()


def _dialect_insert(dialect_name: str, table: Any) -> Any:
    if dialect_name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        return pg_insert(table)
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

    return sqlite_insert(table)


async def _lock_in_sha_order(
    session: Any, dialect_name: str, scope: str, shas: Sequence[str]
) -> None:
    """Postgres: lock the blob rows in ``sha256`` order before a multi-row UPDATE.

    A multi-row ``UPDATE ... WHERE sha256 IN (...)`` locks rows in plan order,
    so two transactions claiming overlapping sets can deadlock. Taking
    ``FOR NO KEY UPDATE`` row locks in one global order first makes the
    following UPDATE wait instead. ``NO KEY UPDATE`` does not conflict with
    the ``KEY SHARE`` locks foreign-key checks take. SQLite serialises writers
    on the database lock and needs nothing.
    """
    if dialect_name != "postgresql" or not shas:
        return
    from sqlalchemy import select

    from dendrux.db.models import Blob

    await session.execute(
        select(Blob.sha256)
        .where(Blob.tenant_scope == scope, Blob.sha256.in_(list(shas)))
        .order_by(Blob.sha256)
        .with_for_update(key_share=True)
    )


def _tenant_matches(column: Any, scope: str) -> Any:
    """SQL predicate: ``agent_runs.tenant_id`` belongs to ``scope``."""
    if scope == "-":
        return column.is_(None)
    if not scope.startswith("t:"):
        from sqlalchemy import false

        return false()
    return column == scope[2:]


class SQLBlobCatalog:
    """Blob catalog over ``dendrux_blobs`` / ``dendrux_blob_refs``.

    Mixin for ``SQLAlchemyStateStore``: needs ``self._engine`` and
    ``self._session_factory``.
    """

    _engine: Any
    _session_factory: Any

    @property
    def _dialect(self) -> str:
        return str(self._engine.dialect.name)

    async def current_time(self) -> datetime:
        """The catalog clock: the database's ``now()`` on Postgres, Python UTC on SQLite."""
        if self._dialect != "postgresql":
            return _py_now()
        from sqlalchemy import func, select

        async with self._session_factory() as session:
            value = (await session.execute(select(func.now()))).scalar_one()
        return _require_aware(value)

    # --- lookup / reuse ---------------------------------------------------

    async def get_current(self, scope: str, sha256s: Sequence[str]) -> dict[str, BlobInfo]:
        from sqlalchemy import select

        from dendrux.db.models import Blob

        shas = list(dict.fromkeys(sha256s))
        out: dict[str, BlobInfo] = {}
        if not shas:
            return out
        async with self._session_factory() as session:
            for chunk in _chunks(shas):
                rows = await session.execute(
                    select(*_blob_info_columns()).where(
                        Blob.tenant_scope == scope,
                        Blob.sha256.in_(chunk),
                        Blob.state.in_(_READER_STATES),
                    )
                )
                for row in rows:
                    out[row.sha256] = _row_to_info(row)
        return out

    async def touch_live(self, scope: str, sha256s: Sequence[str]) -> dict[str, BlobInfo]:
        shas = sorted(set(sha256s))
        if not shas:
            return {}

        async def _attempt() -> dict[str, BlobInfo]:
            out: dict[str, BlobInfo] = {}
            async with self._session_factory() as session:
                await self._touch_live_in(session, scope, shas, out)
                await session.commit()
            return out

        return await retry_transient_db(_attempt, label="touch_live", run_id="-")

    async def _touch_live_in(
        self, session: Any, scope: str, shas: list[str], out: dict[str, BlobInfo]
    ) -> None:
        from sqlalchemy import select, update

        from dendrux.db.models import Blob

        now = _db_now(self._dialect)
        for chunk in _chunks(shas):
            await _lock_in_sha_order(session, self._dialect, scope, chunk)
            stmt = (
                update(Blob)
                .where(
                    Blob.tenant_scope == scope,
                    Blob.sha256.in_(chunk),
                    Blob.state == "live",
                )
                .values(last_used_at=now)
                .execution_options(synchronize_session=False)
            )
            if self._engine.dialect.update_returning:
                rows = await session.execute(stmt.returning(*_blob_info_columns()))
                for row in rows:
                    out[row.sha256] = _row_to_info(row)
            else:  # pragma: no cover - every supported backend has RETURNING
                await session.execute(stmt)
                rows = await session.execute(
                    select(*_blob_info_columns()).where(
                        Blob.tenant_scope == scope,
                        Blob.sha256.in_(chunk),
                        Blob.state == "live",
                    )
                )
                for row in rows:
                    out[row.sha256] = _row_to_info(row)

    # --- lifecycle --------------------------------------------------------

    async def get_row(self, scope: str, sha256: str) -> BlobInfo | None:
        from sqlalchemy import select

        from dendrux.db.models import Blob

        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(*_blob_info_columns()).where(
                        Blob.tenant_scope == scope, Blob.sha256 == sha256
                    )
                )
            ).first()
        return None if row is None else _row_to_info(row)

    async def insert_row(
        self,
        scope: str,
        sha256: str,
        info: BlobInfo,
        *,
        state: Literal["pending", "live"],
        generation: str,
        data: bytes | None,
    ) -> bool:
        from dendrux.db.models import Blob

        now = _db_now(self._dialect)
        stmt = (
            _dialect_insert(self._dialect, Blob)
            .values(
                tenant_scope=scope,
                sha256=sha256,
                generation=generation,
                state=state,
                media_type=info.media_type,
                byte_length=info.byte_length,
                width=info.width,
                height=info.height,
                frame_count=info.frame_count,
                storage=info.storage,
                data=data if info.storage == "db" else None,
                ingest_transform=info.ingest_transform,
                created_at=now,
                last_used_at=now,
                state_changed_at=now,
            )
            .on_conflict_do_nothing(index_elements=["tenant_scope", "sha256"])
        )
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            await session.commit()
        return bool(result.rowcount == 1)

    async def _conditional_update(self, where: list[Any], values: dict[str, Any]) -> bool:
        from sqlalchemy import update

        from dendrux.db.models import Blob

        async with self._session_factory() as session:
            result = await session.execute(
                update(Blob)
                .where(*where)
                .values(**values)
                .execution_options(synchronize_session=False)
            )
            await session.commit()
        return bool(result.rowcount == 1)

    async def promote(self, scope: str, sha256: str, generation: str) -> bool:
        from dendrux.db.models import Blob

        now = _db_now(self._dialect)
        return await self._conditional_update(
            [
                Blob.tenant_scope == scope,
                Blob.sha256 == sha256,
                Blob.state == "pending",
                Blob.generation == generation,
            ],
            {"state": "live", "last_used_at": now, "state_changed_at": now},
        )

    async def adopt_pending(
        self, scope: str, sha256: str, old_generation: str, new_generation: str
    ) -> bool:
        from dendrux.db.models import Blob

        now = _db_now(self._dialect)
        return await self._conditional_update(
            [
                Blob.tenant_scope == scope,
                Blob.sha256 == sha256,
                Blob.state == "pending",
                Blob.generation == old_generation,
            ],
            {"generation": new_generation, "last_used_at": now, "state_changed_at": now},
        )

    async def revive(
        self,
        scope: str,
        sha256: str,
        old_generation: str,
        new_generation: str,
        *,
        to_state: Literal["pending", "live"],
        data: bytes | None,
    ) -> bool:
        from dendrux.db.models import Blob

        now = _db_now(self._dialect)
        return await self._conditional_update(
            [
                Blob.tenant_scope == scope,
                Blob.sha256 == sha256,
                Blob.state == "deleting",
                Blob.generation == old_generation,
            ],
            {
                "state": to_state,
                "generation": new_generation,
                "storage": "db" if data is not None else "external",
                "data": data,
                "last_used_at": now,
                "state_changed_at": now,
            },
        )

    def _no_edge(self, scope: str, sha256: str) -> Any:
        from sqlalchemy import exists, select

        from dendrux.db.models import BlobRef

        return ~exists(
            select(BlobRef.id).where(BlobRef.tenant_scope == scope, BlobRef.sha256 == sha256)
        )

    async def mark_deleting(
        self, scope: str, sha256: str, cutoff: datetime, *, stale_pending_cutoff: datetime
    ) -> str | None:
        marked = await self.mark_deleting_row(
            scope, sha256, cutoff, stale_pending_cutoff=stale_pending_cutoff
        )
        return None if marked is None else marked[0]

    async def mark_deleting_row(
        self, scope: str, sha256: str, cutoff: datetime, *, stale_pending_cutoff: datetime
    ) -> tuple[str, str] | None:
        """:meth:`mark_deleting` returning ``(generation, storage)`` of the marked row.

        On Postgres the ``last_used_at < cutoff`` predicate is the real guard
        against a concurrent edge claim: an UPDATE that waited on the claimed
        row re-evaluates its WHERE against the new row version (the claim
        refreshed ``last_used_at``), but the ``NOT EXISTS`` subquery keeps the
        statement snapshot and cannot see the just-committed edge. Correctness
        therefore relies on the GC grace (``MIN_GC_AGE``, 1 h) exceeding any
        edge transaction; never lower it. The edge re-check and ``delete_row``'s
        ``NOT EXISTS`` (and the FK RESTRICT) are the backstops.
        """
        from sqlalchemy import and_, or_, update

        from dendrux.db.models import Blob

        stmt = (
            update(Blob)
            .where(
                Blob.tenant_scope == scope,
                Blob.sha256 == sha256,
                or_(
                    and_(Blob.state == "live", Blob.last_used_at < cutoff),
                    and_(Blob.state == "pending", Blob.state_changed_at < stale_pending_cutoff),
                ),
                self._no_edge(scope, sha256),
            )
            .values(state="deleting", state_changed_at=_db_now(self._dialect))
            .returning(Blob.generation, Blob.storage)
            .execution_options(synchronize_session=False)
        )
        async with self._session_factory() as session:
            row = (await session.execute(stmt)).first()
            await session.commit()
        return None if row is None else (str(row.generation), str(row.storage))

    async def revert_deleting(self, scope: str, sha256: str, generation: str) -> bool:
        """``deleting → live`` for this generation (GC lost a race at the edge re-check)."""
        from dendrux.db.models import Blob

        now = _db_now(self._dialect)
        return await self._conditional_update(
            [
                Blob.tenant_scope == scope,
                Blob.sha256 == sha256,
                Blob.state == "deleting",
                Blob.generation == generation,
            ],
            {"state": "live", "last_used_at": now, "state_changed_at": now},
        )

    async def delete_row(self, scope: str, sha256: str, generation: str) -> bool:
        from sqlalchemy import delete

        from dendrux.db.models import Blob

        async with self._session_factory() as session:
            result = await session.execute(
                delete(Blob)
                .where(
                    Blob.tenant_scope == scope,
                    Blob.sha256 == sha256,
                    Blob.state == "deleting",
                    Blob.generation == generation,
                    self._no_edge(scope, sha256),
                )
                .execution_options(synchronize_session=False)
            )
            await session.commit()
        return bool(result.rowcount == 1)

    async def mark_purging(self, scope: str, sha256: str, reason: str | None) -> str | None:
        marked = await self.mark_purging_row(scope, sha256, reason)
        return None if marked is None else marked[0]

    async def mark_purging_row(
        self, scope: str, sha256: str, reason: str | None
    ) -> tuple[str, str] | None:
        """:meth:`mark_purging` returning ``(generation, storage)`` of the row it updated."""
        from sqlalchemy import update

        from dendrux.db.models import Blob

        now = _db_now(self._dialect)
        stmt = (
            update(Blob)
            .where(
                Blob.tenant_scope == scope,
                Blob.sha256 == sha256,
                Blob.state.in_(("pending", "live", "deleting")),
            )
            .values(
                state="purging",
                data=None,
                purged_at=now,
                purge_reason=reason,
                state_changed_at=now,
            )
            .returning(Blob.generation, Blob.storage)
            .execution_options(synchronize_session=False)
        )
        async with self._session_factory() as session:
            row = (await session.execute(stmt)).first()
            await session.commit()
        return None if row is None else (str(row.generation), str(row.storage))

    async def finish_purge(self, scope: str, sha256: str, generation: str) -> None:
        from dendrux.db.models import Blob

        await self._conditional_update(
            [
                Blob.tenant_scope == scope,
                Blob.sha256 == sha256,
                Blob.state == "purging",
                Blob.generation == generation,
            ],
            {"state": "purged", "state_changed_at": _db_now(self._dialect)},
        )

    async def stuck_rows(
        self, older_than: datetime, limit: int, *, include_external: bool = True
    ) -> list[BlobInfo]:
        """Rows a crash left mid-transition that recovery can act on.

        ``purging`` rows, ``deleting`` rows without edges (with edges they wait
        for a revive, so listing them would starve the page) and stale
        ``pending`` rows; ``include_external=False`` drops externally stored
        rows (no blob store to delete their objects).
        """
        from sqlalchemy import and_, exists, or_, select

        from dendrux.db.models import Blob, BlobRef

        no_edge = ~exists(
            select(BlobRef.id).where(
                BlobRef.tenant_scope == Blob.tenant_scope, BlobRef.sha256 == Blob.sha256
            )
        )
        where = [
            or_(
                Blob.state == "purging",
                and_(Blob.state == "deleting", no_edge),
                and_(Blob.state == "pending", Blob.state_changed_at < older_than),
            )
        ]
        if not include_external:
            where.append(Blob.storage != "external")
        stmt = (
            select(Blob.tenant_scope, *_blob_info_columns())
            .where(*where)
            .order_by(Blob.state_changed_at, Blob.tenant_scope, Blob.sha256)
            .limit(limit)
        )
        async with self._session_factory() as session:
            rows = (await session.execute(stmt)).all()
        return [_ScopedInfo.of(row.tenant_scope, _row_to_info(row)) for row in rows]

    async def gc_candidates(
        self, cutoff: datetime, limit: int, *, include_external: bool = True
    ) -> list[BlobInfo]:
        """Live rows unused since ``cutoff`` and stale pending rows, all without edges."""
        from sqlalchemy import and_, exists, or_, select

        from dendrux.db.models import Blob, BlobRef

        no_edge = ~exists(
            select(BlobRef.id).where(
                BlobRef.tenant_scope == Blob.tenant_scope, BlobRef.sha256 == Blob.sha256
            )
        )
        stmt = (
            select(Blob.tenant_scope, *_blob_info_columns())
            .where(
                or_(
                    and_(Blob.state == "live", Blob.last_used_at < cutoff),
                    and_(Blob.state == "pending", Blob.state_changed_at < cutoff),
                ),
                no_edge,
                *([] if include_external else [Blob.storage != "external"]),
            )
            .order_by(Blob.last_used_at, Blob.tenant_scope, Blob.sha256)
            .limit(limit)
        )
        async with self._session_factory() as session:
            rows = (await session.execute(stmt)).all()
        return [_ScopedInfo.of(row.tenant_scope, _row_to_info(row)) for row in rows]

    async def has_edges(self, scope: str, sha256: str) -> bool:
        """Whether any edge references ``(scope, sha256)``."""
        from sqlalchemy import select

        from dendrux.db.models import BlobRef

        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(BlobRef.id)
                    .where(BlobRef.tenant_scope == scope, BlobRef.sha256 == sha256)
                    .limit(1)
                )
            ).first()
        return row is not None

    async def read_blob_data(self, key: BlobKey) -> bytes | None:
        from sqlalchemy import select

        from dendrux.db.models import Blob

        async with self._session_factory() as session:
            value = (
                await session.execute(
                    select(Blob.data).where(
                        Blob.tenant_scope == key.tenant_scope,
                        Blob.sha256 == key.sha256,
                        Blob.generation == key.generation,
                        Blob.storage == "db",
                        Blob.state == "live",
                    )
                )
            ).scalar_one_or_none()
        return None if value is None else bytes(value)

    # --- edges & authorisation ---------------------------------------------

    async def add_blob_refs(self, run_id: str, run_scope: str, edges: Sequence[BlobEdge]) -> None:
        if not edges:
            return

        async def _attempt() -> None:
            async with self._session_factory() as session:
                await write_blob_refs(
                    session, run_id=run_id, blob_refs=edges, blob_refs_scope=run_scope
                )
                await session.commit()

        await retry_transient_db(_attempt, label="add_blob_refs", run_id=run_id)

    async def run_references(self, run_scope: str, run_id: str, sha256: str) -> bool:
        from sqlalchemy import select

        from dendrux.db.models import AgentRun, BlobRef

        stmt = (
            select(BlobRef.id)
            .join(AgentRun, AgentRun.id == BlobRef.run_id)
            .where(
                BlobRef.run_id == run_id,
                BlobRef.tenant_scope == run_scope,
                BlobRef.sha256 == sha256,
                _tenant_matches(AgentRun.tenant_id, run_scope),
            )
            .limit(1)
        )
        async with self._session_factory() as session:
            row = (await session.execute(stmt)).first()
        return row is not None

    async def blob_references(self, scope: str, sha256: str) -> list[BlobReference]:
        from sqlalchemy import select

        from dendrux.db.models import BlobRef

        stmt = (
            select(
                BlobRef.run_id,
                BlobRef.owner_kind,
                BlobRef.owner_id,
                BlobRef.ordinal,
                BlobRef.created_at,
            )
            .where(BlobRef.tenant_scope == scope, BlobRef.sha256 == sha256)
            .order_by(BlobRef.created_at, BlobRef.id)
        )
        async with self._session_factory() as session:
            rows = (await session.execute(stmt)).all()
        return [
            BlobReference(
                run_id=r.run_id,
                owner_kind=r.owner_kind,
                owner_id=r.owner_id,
                ordinal=int(r.ordinal),
                created_at=_require_aware(r.created_at),
            )
            for r in rows
        ]

    async def delete_orphan_edges(self, limit: int) -> int:
        from sqlalchemy import delete, exists, select

        from dendrux.db.models import AgentRun, BlobRef

        orphan_ids = (
            select(BlobRef.id)
            .where(~exists(select(AgentRun.id).where(AgentRun.id == BlobRef.run_id)))
            .limit(limit)
        )
        async with self._session_factory() as session:
            ids = [r[0] for r in (await session.execute(orphan_ids)).all()]
            if not ids:
                return 0
            deleted = 0
            for chunk in _chunks(ids):
                result = await session.execute(
                    delete(BlobRef)
                    .where(
                        BlobRef.id.in_(chunk),
                        ~exists(select(AgentRun.id).where(AgentRun.id == BlobRef.run_id)),
                    )
                    .execution_options(synchronize_session=False)
                )
                deleted += int(result.rowcount or 0)
            await session.commit()
        return deleted


@dataclass(frozen=True)
class _ScopedInfo(BlobInfo):
    """A :class:`BlobInfo` that also carries its tenant scope (GC work lists)."""

    tenant_scope: str = "-"

    @classmethod
    def of(cls, scope: str, info: BlobInfo) -> _ScopedInfo:
        return cls(**{**info.__dict__, "tenant_scope": scope})


def _scope_of(info: BlobInfo) -> str:
    return getattr(info, "tenant_scope", "-")


# ---------------------------------------------------------------------------
# Edge-write protocol (inside the record's transaction)
# ---------------------------------------------------------------------------


class _Unset:
    pass


_UNSET = _Unset()


async def write_blob_refs(
    session: AsyncSession,
    *,
    run_id: str,
    blob_refs: Sequence[BlobEdge],
    blob_refs_scope: str | None,
    run_tenant_id: str | None | _Unset = _UNSET,
) -> None:
    """Write ``blob_refs`` for ``run_id`` in the caller's open transaction.

    1. Resolve the run's tenant (``run_tenant_id`` for a row being inserted,
       else ``agent_runs.tenant_id``) and require it to equal
       ``blob_refs_scope`` and every edge's scope.
    2. Claim the blobs with ``UPDATE ... SET last_used_at WHERE state='live'``
       (the lock GC's ``mark_deleting`` waits on / re-evaluates against).
    3. Insert the edges, ignoring an already-written owner key (a retried
       write after a lost commit acknowledgement is a no-op).

    Raises (the caller's transaction must then be rolled back):
        ImagePersistenceUnsupportedError: edges without a scope.
        BlobNotFoundError: tenant mismatch, unknown run, or a blob that is
            absent / pending / deleting (uniform; no existence oracle).
        InvalidImageError: ``"purged"`` for a purging or purged blob.
    """
    from sqlalchemy import select, update

    from dendrux.db.models import AgentRun, Blob, BlobRef

    if not blob_refs:
        return
    if blob_refs_scope is None:
        raise ImagePersistenceUnsupportedError("state_store")
    first_id = image_id(blob_refs[0].sha256)

    # 1. Tenant assertion.
    if isinstance(run_tenant_id, _Unset):
        found = (
            await session.execute(select(AgentRun.tenant_id).where(AgentRun.id == run_id))
        ).first()
        run_scope: str | None = None if found is None else tenant_scope_for(found[0])
    else:
        run_scope = tenant_scope_for(run_tenant_id)
    if run_scope is None or run_scope != blob_refs_scope:
        _log_mismatch(run_id, blob_refs[0].owner_kind)
        raise BlobNotFoundError(first_id)
    for edge in blob_refs:
        if edge.tenant_scope != run_scope:
            _log_mismatch(run_id, edge.owner_kind)
            raise BlobNotFoundError(image_id(edge.sha256))

    _check_batch_owner_keys(blob_refs)

    # 2. Claim the blob rows with a write (rows locked in sha order first on PG).
    dialect = session.bind.dialect
    shas = sorted({e.sha256 for e in blob_refs})
    now = _db_now(dialect.name)
    claimed: set[str] = set()
    for chunk in _chunks(shas):
        await _lock_in_sha_order(session, dialect.name, run_scope, chunk)
        stmt = (
            update(Blob)
            .where(Blob.tenant_scope == run_scope, Blob.sha256.in_(chunk), Blob.state == "live")
            .values(last_used_at=now)
            .execution_options(synchronize_session=False)
        )
        if dialect.update_returning:
            claimed.update(r[0] for r in (await session.execute(stmt.returning(Blob.sha256))))
        else:  # pragma: no cover - the write lock is held after the UPDATE
            await session.execute(stmt)
            claimed.update(
                r[0]
                for r in await session.execute(
                    select(Blob.sha256).where(
                        Blob.tenant_scope == run_scope,
                        Blob.sha256.in_(chunk),
                        Blob.state == "live",
                    )
                )
            )
    missing = [s for s in shas if s not in claimed]
    # Trace edges skip a blob that is no longer live (a purged image in a
    # retried/replayed trace): its input/submission/seed edge already holds
    # reachability and history shows the purge placeholder. Every other
    # owner kind needs the blob live.
    required = [
        s for s in missing if any(e.sha256 == s and e.owner_kind != "trace" for e in blob_refs)
    ]
    if missing and not required:
        logger.info(
            "Blob edges skipped for %d non-live image(s) in a trace (run=%s)", len(missing), run_id
        )
        blob_refs = [e for e in blob_refs if e.sha256 in claimed]
        if not blob_refs:
            return
    missing = required
    if missing:
        states = {
            r.sha256: r.state
            for r in await session.execute(
                select(Blob.sha256, Blob.state).where(
                    Blob.tenant_scope == run_scope, Blob.sha256.in_(missing)
                )
            )
        }
        for sha in missing:
            if states.get(sha) in ("purging", "purged"):
                raise InvalidImageError("purged")
        raise BlobNotFoundError(image_id(missing[0]))

    # 3. Idempotent edge inserts.
    created = _db_now(dialect.name)
    rows = [
        {
            "id": generate_ulid(),
            "tenant_scope": e.tenant_scope,
            "sha256": e.sha256,
            "run_id": run_id,
            "owner_kind": e.owner_kind,
            "owner_id": e.owner_id,
            "ordinal": e.ordinal,
            "created_at": created,
        }
        for e in blob_refs
    ]
    stmt_ins = (
        _dialect_insert(dialect.name, BlobRef)
        .values(rows)
        .on_conflict_do_nothing(index_elements=["run_id", "owner_kind", "owner_id", "ordinal"])
    )
    await session.execute(stmt_ins)

    # An owner key that already names a different blob is a collision, never a
    # silent no-op (the new record's image would otherwise have no edge).
    from sqlalchemy import tuple_

    wanted = {(e.owner_kind, e.owner_id, e.ordinal): (e.tenant_scope, e.sha256) for e in blob_refs}
    keys = list(wanted)
    for i in range(0, len(keys), _IN_CHUNK):
        part = keys[i : i + _IN_CHUNK]
        stored = await session.execute(
            select(
                BlobRef.owner_kind,
                BlobRef.owner_id,
                BlobRef.ordinal,
                BlobRef.tenant_scope,
                BlobRef.sha256,
            ).where(
                BlobRef.run_id == run_id,
                tuple_(BlobRef.owner_kind, BlobRef.owner_id, BlobRef.ordinal).in_(part),
            )
        )
        for r in stored:
            if wanted[(r.owner_kind, r.owner_id, int(r.ordinal))] != (r.tenant_scope, r.sha256):
                logger.error(
                    "Blob edge write rejected: owner key already references another image "
                    "(run=%s owner_kind=%s owner_id=%s ordinal=%s)",
                    run_id,
                    r.owner_kind,
                    r.owner_id,
                    r.ordinal,
                )
                raise BlobNotFoundError(
                    image_id(wanted[(r.owner_kind, r.owner_id, int(r.ordinal))][1])
                )


def _check_batch_owner_keys(edges: Sequence[BlobEdge]) -> None:
    seen: dict[tuple[str, str, int], str] = {}
    for edge in edges:
        prior = seen.setdefault((edge.owner_kind, edge.owner_id, edge.ordinal), edge.sha256)
        if prior != edge.sha256:
            raise ValueError("blob_refs repeats an owner key with different images")


def _log_mismatch(run_id: str, owner_kind: str) -> None:
    logger.error(
        "Blob edge write rejected: tenant scope does not match run (run=%s owner_kind=%s)",
        run_id,
        owner_kind,
    )


# ---------------------------------------------------------------------------
# In-memory catalog (no database)
# ---------------------------------------------------------------------------


@dataclass
class _MemRow:
    info: BlobInfo
    data: bytes | None


@dataclass
class _MemEdge:
    edge: BlobEdge
    run_id: str
    created_at: datetime


class InMemoryBlobCatalog:
    """The blob catalog over dicts, for agents without a database (non-durable).

    Implements the same state machine as :class:`SQLBlobCatalog`. Edges are
    tracked per run in memory: the runtime registers them with
    :meth:`add_blob_refs` and drops them with :meth:`release_run` when the
    run reaches a terminal state (runs without a database cannot pause).
    Every method is atomic (a lock, no awaits inside). Edges are indexed by
    blob and by run, so GC and release cost O(rows) and O(edges of the run).
    """

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], _MemRow] = {}
        self._edges: dict[tuple[str, str, str, int], _MemEdge] = {}
        self._by_blob: dict[tuple[str, str], set[tuple[str, str, str, int]]] = {}
        self._by_run: dict[str, set[tuple[str, str, str, int]]] = {}
        self._run_scopes: dict[str, str] = {}
        self._lock = threading.Lock()

    def _add_edge(self, key: tuple[str, str, str, int], edge: _MemEdge) -> None:
        self._edges[key] = edge
        self._by_blob.setdefault((edge.edge.tenant_scope, edge.edge.sha256), set()).add(key)
        self._by_run.setdefault(edge.run_id, set()).add(key)

    def _drop_edge(self, key: tuple[str, str, str, int]) -> None:
        edge = self._edges.pop(key, None)
        if edge is None:
            return
        blob_key = (edge.edge.tenant_scope, edge.edge.sha256)
        by_blob = self._by_blob.get(blob_key)
        if by_blob is not None:
            by_blob.discard(key)
            if not by_blob:
                del self._by_blob[blob_key]
        by_run = self._by_run.get(edge.run_id)
        if by_run is not None:
            by_run.discard(key)
            if not by_run:
                del self._by_run[edge.run_id]

    async def current_time(self) -> datetime:
        return _py_now()

    def _set(self, key: tuple[str, str], **changes: Any) -> None:
        row = self._rows[key]
        data = changes.pop("data", row.data)
        row.info = replace(row.info, **changes)
        row.data = data

    def _edged(self, scope: str, sha256: str) -> bool:
        return bool(self._by_blob.get((scope, sha256)))

    async def get_current(self, scope: str, sha256s: Sequence[str]) -> dict[str, BlobInfo]:
        with self._lock:
            out: dict[str, BlobInfo] = {}
            for sha in sha256s:
                row = self._rows.get((scope, sha))
                if row is not None and row.info.state in _READER_STATES:
                    out[sha] = row.info
            return out

    async def touch_live(self, scope: str, sha256s: Sequence[str]) -> dict[str, BlobInfo]:
        with self._lock:
            out: dict[str, BlobInfo] = {}
            now = _py_now()
            for sha in sha256s:
                row = self._rows.get((scope, sha))
                if row is not None and row.info.state == "live":
                    self._set((scope, sha), last_used_at=now)
                    out[sha] = row.info
            return out

    async def get_row(self, scope: str, sha256: str) -> BlobInfo | None:
        with self._lock:
            row = self._rows.get((scope, sha256))
            return None if row is None else row.info

    async def insert_row(
        self,
        scope: str,
        sha256: str,
        info: BlobInfo,
        *,
        state: Literal["pending", "live"],
        generation: str,
        data: bytes | None,
    ) -> bool:
        with self._lock:
            if (scope, sha256) in self._rows:
                return False
            now = _py_now()
            self._rows[(scope, sha256)] = _MemRow(
                info=replace(
                    info,
                    sha256=sha256,
                    state=state,
                    generation=generation,
                    created_at=now,
                    last_used_at=now,
                    state_changed_at=now,
                    purged_at=None,
                    purge_reason=None,
                ),
                data=data if info.storage == "db" else None,
            )
            return True

    def _cas(
        self,
        scope: str,
        sha256: str,
        states: tuple[BlobState, ...],
        expect_generation: str | None,
        **changes: Any,
    ) -> BlobInfo | None:
        row = self._rows.get((scope, sha256))
        if row is None or row.info.state not in states:
            return None
        if expect_generation is not None and row.info.generation != expect_generation:
            return None
        before = row.info
        self._set((scope, sha256), **changes)
        return before

    async def promote(self, scope: str, sha256: str, generation: str) -> bool:
        with self._lock:
            now = _py_now()
            return (
                self._cas(
                    scope,
                    sha256,
                    ("pending",),
                    generation,
                    state="live",
                    last_used_at=now,
                    state_changed_at=now,
                )
                is not None
            )

    async def adopt_pending(
        self, scope: str, sha256: str, old_generation: str, new_generation: str
    ) -> bool:
        with self._lock:
            now = _py_now()
            return (
                self._cas(
                    scope,
                    sha256,
                    ("pending",),
                    old_generation,
                    generation=new_generation,
                    last_used_at=now,
                    state_changed_at=now,
                )
                is not None
            )

    async def revive(
        self,
        scope: str,
        sha256: str,
        old_generation: str,
        new_generation: str,
        *,
        to_state: Literal["pending", "live"],
        data: bytes | None,
    ) -> bool:
        with self._lock:
            now = _py_now()
            return (
                self._cas(
                    scope,
                    sha256,
                    ("deleting",),
                    old_generation,
                    state=to_state,
                    generation=new_generation,
                    storage="db" if data is not None else "external",
                    data=data,
                    last_used_at=now,
                    state_changed_at=now,
                )
                is not None
            )

    async def mark_deleting(
        self, scope: str, sha256: str, cutoff: datetime, *, stale_pending_cutoff: datetime
    ) -> str | None:
        marked = await self.mark_deleting_row(
            scope, sha256, cutoff, stale_pending_cutoff=stale_pending_cutoff
        )
        return None if marked is None else marked[0]

    async def mark_deleting_row(
        self, scope: str, sha256: str, cutoff: datetime, *, stale_pending_cutoff: datetime
    ) -> tuple[str, str] | None:
        with self._lock:
            row = self._rows.get((scope, sha256))
            if row is None or self._edged(scope, sha256):
                return None
            info = row.info
            changed = info.state_changed_at or info.last_used_at
            eligible = (info.state == "live" and info.last_used_at < cutoff) or (
                info.state == "pending" and changed < stale_pending_cutoff
            )
            if not eligible:
                return None
            self._set((scope, sha256), state="deleting", state_changed_at=_py_now())
            return info.generation, info.storage

    async def revert_deleting(self, scope: str, sha256: str, generation: str) -> bool:
        with self._lock:
            now = _py_now()
            return (
                self._cas(
                    scope,
                    sha256,
                    ("deleting",),
                    generation,
                    state="live",
                    last_used_at=now,
                    state_changed_at=now,
                )
                is not None
            )

    async def delete_row(self, scope: str, sha256: str, generation: str) -> bool:
        with self._lock:
            row = self._rows.get((scope, sha256))
            if (
                row is None
                or row.info.state != "deleting"
                or row.info.generation != generation
                or self._edged(scope, sha256)
            ):
                return False
            del self._rows[(scope, sha256)]
            return True

    async def mark_purging(self, scope: str, sha256: str, reason: str | None) -> str | None:
        marked = await self.mark_purging_row(scope, sha256, reason)
        return None if marked is None else marked[0]

    async def mark_purging_row(
        self, scope: str, sha256: str, reason: str | None
    ) -> tuple[str, str] | None:
        with self._lock:
            now = _py_now()
            before = self._cas(
                scope,
                sha256,
                ("pending", "live", "deleting"),
                None,
                state="purging",
                data=None,
                purged_at=now,
                purge_reason=reason,
                state_changed_at=now,
            )
            return None if before is None else (before.generation, before.storage)

    async def finish_purge(self, scope: str, sha256: str, generation: str) -> None:
        with self._lock:
            self._cas(
                scope,
                sha256,
                ("purging",),
                generation,
                state="purged",
                state_changed_at=_py_now(),
            )

    async def stuck_rows(
        self, older_than: datetime, limit: int, *, include_external: bool = True
    ) -> list[BlobInfo]:
        with self._lock:
            out: list[BlobInfo] = []
            for (scope, sha), row in self._rows.items():
                info = row.info
                if not include_external and info.storage == "external":
                    continue
                changed = info.state_changed_at or info.last_used_at
                if (
                    info.state == "purging"
                    or (info.state == "deleting" and not self._edged(scope, sha))
                    or (info.state == "pending" and changed < older_than)
                ):
                    out.append(_ScopedInfo.of(scope, info))
            return out[:limit]

    async def gc_candidates(
        self, cutoff: datetime, limit: int, *, include_external: bool = True
    ) -> list[BlobInfo]:
        with self._lock:
            out: list[BlobInfo] = []
            for (scope, sha), row in self._rows.items():
                info = row.info
                if not include_external and info.storage == "external":
                    continue
                changed = info.state_changed_at or info.last_used_at
                old = (info.state == "live" and info.last_used_at < cutoff) or (
                    info.state == "pending" and changed < cutoff
                )
                if old and not self._edged(scope, sha):
                    out.append(_ScopedInfo.of(scope, info))
            return out[:limit]

    async def has_edges(self, scope: str, sha256: str) -> bool:
        with self._lock:
            return self._edged(scope, sha256)

    async def read_blob_data(self, key: BlobKey) -> bytes | None:
        with self._lock:
            row = self._rows.get((key.tenant_scope, key.sha256))
            if (
                row is None
                or row.info.generation != key.generation
                or row.info.storage != "db"
                or row.info.state != "live"
            ):
                return None
            return row.data

    async def add_blob_refs(self, run_id: str, run_scope: str, edges: Sequence[BlobEdge]) -> None:
        """Register in-memory edges for ``run_id`` (same checks as the SQL protocol)."""
        if not edges:
            return
        with self._lock:
            known = self._run_scopes.get(run_id)
            if (known is not None and known != run_scope) or any(
                e.tenant_scope != run_scope for e in edges
            ):
                _log_mismatch(run_id, edges[0].owner_kind)
                raise BlobNotFoundError(image_id(edges[0].sha256))
            _check_batch_owner_keys(edges)
            for e in edges:
                key = (run_id, e.owner_kind, e.owner_id, e.ordinal)
                existing = self._edges.get(key)
                if existing is not None and (
                    existing.edge.tenant_scope,
                    existing.edge.sha256,
                ) != (e.tenant_scope, e.sha256):
                    logger.error(
                        "Blob edge write rejected: owner key already references another image "
                        "(run=%s owner_kind=%s)",
                        run_id,
                        e.owner_kind,
                    )
                    raise BlobNotFoundError(image_id(e.sha256))
            live: list[BlobEdge] = []
            for e in edges:
                row = self._rows.get((run_scope, e.sha256))
                if row is None or row.info.state != "live":
                    if e.owner_kind == "trace":
                        continue  # same skip-non-live rule as the SQL edge write
                    if row is not None and row.info.state in ("purging", "purged"):
                        raise InvalidImageError("purged")
                    raise BlobNotFoundError(image_id(e.sha256))
                live.append(e)
            now = _py_now()
            self._run_scopes[run_id] = run_scope
            for e in live:
                self._set((run_scope, e.sha256), last_used_at=now)
                key = (run_id, e.owner_kind, e.owner_id, e.ordinal)
                if key not in self._edges:
                    self._add_edge(key, _MemEdge(edge=e, run_id=run_id, created_at=now))

    def release_run(self, run_id: str) -> None:
        """Drop every in-memory edge of ``run_id`` (call when the run is terminal)."""
        with self._lock:
            self._run_scopes.pop(run_id, None)
            for key in list(self._by_run.get(run_id, ())):
                self._drop_edge(key)

    async def run_references(self, run_scope: str, run_id: str, sha256: str) -> bool:
        with self._lock:
            if self._run_scopes.get(run_id) != run_scope:
                return False
            return any(
                self._edges[k].edge.tenant_scope == run_scope
                and self._edges[k].edge.sha256 == sha256
                for k in self._by_run.get(run_id, ())
            )

    async def blob_references(self, scope: str, sha256: str) -> list[BlobReference]:
        with self._lock:
            return [
                BlobReference(
                    run_id=e.run_id,
                    owner_kind=e.edge.owner_kind,
                    owner_id=e.edge.owner_id,
                    ordinal=e.edge.ordinal,
                    created_at=e.created_at,
                )
                for e in (self._edges[k] for k in self._by_blob.get((scope, sha256), ()))
            ]

    async def delete_orphan_edges(self, limit: int) -> int:
        with self._lock:
            orphans = [
                k
                for run_id, keys in self._by_run.items()
                if run_id not in self._run_scopes
                for k in keys
            ][:limit]
            for key in orphans:
                self._drop_edge(key)
            return len(orphans)


# ---------------------------------------------------------------------------
# Resolver (verified reads + generation-keyed LRU)
# ---------------------------------------------------------------------------


class CachingBlobResolver:
    """Reads verified bytes of a live blob from the DB or the external store.

    Every miss is verified (length, SHA-256, sniffed type, header dims and
    frames) before it is cached. The cache is keyed by
    ``(scope, sha256, generation)`` and used only for rows the caller just
    read as ``live``, so a purge or GC takes effect from the next request.
    """

    def __init__(
        self,
        catalog: BlobCatalog,
        blob_store: BlobStore | None = None,
        *,
        max_bytes: int = DEFAULT_CACHE_BYTES,
    ) -> None:
        self._catalog = catalog
        self._store = blob_store
        self._max_bytes = max_bytes
        self._cache: OrderedDict[tuple[str, str, str], bytes] = OrderedDict()
        self._size = 0
        self._lock = threading.Lock()

    @property
    def cached_bytes(self) -> int:
        """Bytes currently held by the cache."""
        return self._size

    def _get(self, key: tuple[str, str, str]) -> bytes | None:
        with self._lock:
            data = self._cache.get(key)
            if data is not None:
                self._cache.move_to_end(key)
            return data

    def _put(self, key: tuple[str, str, str], data: bytes) -> None:
        if len(data) > self._max_bytes:
            return
        with self._lock:
            if key in self._cache:
                return
            self._cache[key] = data
            self._size += len(data)
            while self._size > self._max_bytes:
                _, old = self._cache.popitem(last=False)
                self._size -= len(old)

    def evict(self, scope: str, sha256: str) -> None:
        """Drop every cached generation of ``(scope, sha256)`` (called on purge)."""
        with self._lock:
            for key in [k for k in self._cache if k[0] == scope and k[1] == sha256]:
                self._size -= len(self._cache.pop(key))

    async def read(self, scope: str, info: BlobInfo) -> bytes:
        if info.state != "live":
            raise BlobNotFoundError(image_id(info.sha256))
        cache_key = (scope, info.sha256, info.generation)
        hit = self._get(cache_key)
        if hit is not None:
            return hit
        key = BlobKey(scope, info.sha256, info.generation)
        if info.storage == "db":
            data = await self._catalog.read_blob_data(key)
        else:
            if self._store is None:
                raise ImagePersistenceUnsupportedError("blob_store_not_configured")
            data = await self._store.get(key)
        if data is None:
            raise BlobNotFoundError(image_id(info.sha256))
        data = bytes(data)
        verify_blob_bytes(data, info)
        self._put(cache_key, data)
        return data


# ---------------------------------------------------------------------------
# Accessor: read / info / references / purge / collect_garbage
# ---------------------------------------------------------------------------


class CatalogBlobAccessor:
    """``agent.blobs`` / ``RunStore.blobs`` over a blob catalog.

    ``tenant_id`` is the authorisation boundary: absence and a tenant mismatch
    both raise :class:`~dendrux.errors.BlobNotFoundError`. Purge and GC run
    the row-as-work-item state machine; a crash at any step leaves a row the
    next :meth:`collect_garbage` finishes.
    """

    def __init__(
        self,
        catalog: BlobCatalog | None,
        *,
        blob_store: BlobStore | None = None,
        resolver: BlobResolver | None = None,
        store_type: str | None = None,
    ) -> None:
        self._catalog = catalog
        self._store = blob_store
        self._resolver = resolver
        self._store_type = store_type

    def _require_catalog(self) -> BlobCatalog:
        if self._catalog is None:
            raise ImagePersistenceUnsupportedError("state_store", store_type=self._store_type)
        return self._catalog

    def _require_resolver(self) -> BlobResolver:
        catalog = self._require_catalog()
        if self._resolver is None:
            self._resolver = CachingBlobResolver(catalog, self._store)
        return self._resolver

    async def read(
        self, image: ImageRef | ImagePart | str, *, tenant_id: str | None = None
    ) -> bytes:
        """Verified bytes of a live blob in ``tenant_id``'s scope.

        ``tenant_id=None`` (the default) is the untenanted scope only, not
        "any tenant": blobs are tenant-scoped and a blob read has no run to
        take a scope from. ``RunStore.read_image`` is the run-scoped read.

        Raises:
            BlobNotFoundError: absent, other tenant, pending, deleting or purged.
            InvalidImageError: ``"descriptor_mismatch"`` when an ``ImageRef``
                disagrees with the catalog.
            BlobIntegrityError: the stored bytes failed verification.
        """
        ref: ImageRef | None
        if isinstance(image, ImagePart):
            ref = image.ref
        elif isinstance(image, ImageRef):
            ref = image
        else:
            ref = None
        sha = _check_sha(ref.sha256 if ref is not None else str(image))
        scope = tenant_scope_for(tenant_id)
        info = (await self._require_catalog().get_current(scope, [sha])).get(sha)
        if info is None or info.state != "live":
            raise BlobNotFoundError(image_id(sha))
        if ref is not None and info.to_ref() != ref:
            raise InvalidImageError("descriptor_mismatch")
        return await self._require_resolver().read(scope, info)

    async def info(self, sha256: str, *, tenant_id: str | None = None) -> BlobInfo | None:
        """Catalog metadata (live, purging or purged rows), or None."""
        sha = _check_sha(sha256)
        scope = tenant_scope_for(tenant_id)
        info = (await self._require_catalog().get_current(scope, [sha])).get(sha)
        return None if info is None else _plain(info)

    async def references(self, sha256: str, *, tenant_id: str | None = None) -> list[BlobReference]:
        """Every recorded use of the blob in ``tenant_id``'s scope (kept after purge)."""
        sha = _check_sha(sha256)
        return await self._require_catalog().blob_references(tenant_scope_for(tenant_id), sha)

    async def purge(
        self, sha256: str, *, tenant_id: str | None = None, reason: str | None = None
    ) -> bool:
        """Erase the blob's bytes; returns True once the purge is recorded, False if no row.

        Purge wins over every ingest: the row moves to ``purging`` (bytes
        dropped), the external object is deleted, then the row is ``purged``.
        Re-ingesting the same bytes then fails with ``InvalidImageError("purged")``.
        Edges are kept (audit). This process's verified-bytes cache is evicted;
        other processes drop theirs by LRU (their reads already fail, since
        every read checks the row first). For an externally stored blob with
        no ``blob_store`` configured, the intent is recorded and then
        ``ImagePersistenceUnsupportedError("blob_store_not_configured")`` is
        raised; a later ``collect_garbage`` with the store deletes the object.
        """
        sha = _check_sha(sha256)
        if reason is not None and len(reason) > _PURGE_REASON_MAX:
            raise ValueError(f"purge reason must be at most {_PURGE_REASON_MAX} characters")
        catalog = self._require_catalog()
        scope = tenant_scope_for(tenant_id)
        # Record intent first (bytes unreadable, re-ingest blocked); the external
        # delete is decided from the row this statement updated.
        marked = await _mark_purging_row(catalog, scope, sha, reason)
        self._evict(scope, sha)
        if marked is None:
            row = await catalog.get_row(scope, sha)
            if row is None:
                return False
            if row.state != "purging":
                return True  # already purged
            marked = (row.generation, row.storage)
        generation, storage = marked
        if storage == "external" and self._store is None:
            # Intent is durable; the next collect_garbage with a blob_store finishes it.
            raise ImagePersistenceUnsupportedError("blob_store_not_configured")
        await self._finish_purge(catalog, scope, sha, generation, storage)
        return True

    def _evict(self, scope: str, sha: str) -> None:
        evict = getattr(self._resolver, "evict", None)
        if evict is not None:
            evict(scope, sha)

    async def _finish_purge(
        self, catalog: BlobCatalog, scope: str, sha: str, generation: str, storage: str
    ) -> None:
        if storage == "external" and self._store is not None:
            await self._store.delete(BlobKey(scope, sha, generation))
        await catalog.finish_purge(scope, sha, generation)

    async def collect_garbage(
        self, *, older_than: timedelta = timedelta(hours=24), limit: int = 1000
    ) -> GcResult:
        """Recover stuck rows, drop orphan edges, then delete unreferenced old blobs.

        ``older_than`` (>= 1 hour) is the grace since a blob was last used.
        Idempotent and restartable; never compares across tenant scopes.
        A blob that gains a reference while being collected is kept (counted
        in ``skipped_referenced``); per-row failures are logged, never raised.
        """
        if older_than < MIN_GC_AGE:
            raise ValueError("collect_garbage(older_than=...) must be at least 1 hour")
        if limit <= 0:
            raise ValueError("collect_garbage(limit=...) must be positive")
        catalog = self._require_catalog()
        now = await _catalog_now(catalog)
        cutoff = now - older_than
        stats = _GcStats()

        # 1. Recovery of rows a crash left mid-transition.
        has_store = self._store is not None
        for info in await _call(catalog, "stuck_rows", cutoff, limit, include_external=has_store):
            await self._guarded(self._recover, catalog, info, cutoff, stats)
        # 2. Edges whose run is gone (SQLite has no FK cascade).
        try:
            stats.deleted_orphan_edges += await catalog.delete_orphan_edges(limit)
        except Exception:
            logger.error("Blob GC: deleting orphan edges failed", exc_info=True)
        # 3. Unreferenced blobs past the grace period.
        for info in await _call(
            catalog, "gc_candidates", cutoff, limit, include_external=has_store
        ):
            await self._guarded(self._collect, catalog, info, cutoff, stats)
        return stats.result()

    async def _guarded(
        self, step: Any, catalog: BlobCatalog, info: BlobInfo, cutoff: datetime, stats: _GcStats
    ) -> None:
        try:
            await step(catalog, info, cutoff, stats)
        except Exception:
            logger.error(
                "Blob GC: step failed (scope=%s image=%s state=%s)",
                _scope_of(info),
                image_id(info.sha256),
                info.state,
                exc_info=True,
            )

    def _can_touch_object(self, info: BlobInfo) -> bool:
        if info.storage == "external" and self._store is None:
            logger.warning(
                "Blob GC: skipping externally stored image %s (no blob_store configured)",
                image_id(info.sha256),
            )
            return False
        return True

    async def _recover(
        self, catalog: BlobCatalog, info: BlobInfo, cutoff: datetime, stats: _GcStats
    ) -> None:
        scope = _scope_of(info)
        if not self._can_touch_object(info):
            return
        if info.state == "purging":
            await self._finish_purge(catalog, scope, info.sha256, info.generation, info.storage)
            stats.completed_purges += 1
            return
        if info.state == "deleting":
            if await _call(catalog, "has_edges", scope, info.sha256):
                stats.skipped_referenced += 1
                return
            await self._delete_object(scope, info)
            if await catalog.delete_row(scope, info.sha256, info.generation):
                stats.retried_deletes += 1
                stats.deleted_blobs += 1
                stats.bytes_freed += info.byte_length
            return
        if info.state == "pending":
            await self._collect(catalog, info, cutoff, stats)

    async def _collect(
        self, catalog: BlobCatalog, info: BlobInfo, cutoff: datetime, stats: _GcStats
    ) -> None:
        scope, sha = _scope_of(info), info.sha256
        if not self._can_touch_object(info):
            return
        marked = await _call(
            catalog, "mark_deleting_row", scope, sha, cutoff, stale_pending_cutoff=cutoff
        )
        if marked is None:
            return
        generation, storage = marked
        # Defence in depth: an edge that slipped in keeps the blob.
        if await _call(catalog, "has_edges", scope, sha):
            if storage == "db":
                # Bytes are in the row (kept while deleting): safe to go back to live.
                await _call(catalog, "revert_deleting", scope, sha, generation)
            # External: never revert. A concurrent ingest may already have deleted
            # key(generation) to revive the row; reverting would leave a live row
            # without its object. The row stays ``deleting`` for that revive.
            logger.error(
                "Blob GC: image %s gained a reference while being collected; kept (scope=%s)",
                image_id(sha),
                scope,
            )
            stats.skipped_referenced += 1
            return
        await self._delete_object(scope, replace(info, generation=generation, storage=storage))
        if await catalog.delete_row(scope, sha, generation):
            stats.deleted_blobs += 1
            stats.bytes_freed += info.byte_length
            return
        row = await catalog.get_row(scope, sha)
        if row is not None and row.state == "deleting" and row.generation == generation:
            logger.error(
                "Blob GC: image %s referenced after its object was deleted; row left for "
                "revive (scope=%s)",
                image_id(sha),
                scope,
            )
            stats.skipped_referenced += 1

    async def _delete_object(self, scope: str, info: BlobInfo) -> None:
        if info.storage == "external" and self._store is not None:
            await self._store.delete(BlobKey(scope, info.sha256, info.generation))


@dataclass
class _GcStats:
    deleted_blobs: int = 0
    deleted_orphan_edges: int = 0
    completed_purges: int = 0
    retried_deletes: int = 0
    skipped_referenced: int = 0
    bytes_freed: int = 0

    def result(self) -> GcResult:
        return GcResult(**self.__dict__)


def _plain(info: BlobInfo) -> BlobInfo:
    if type(info) is BlobInfo:
        return info
    fields = {k: v for k, v in info.__dict__.items() if k != "tenant_scope"}
    return BlobInfo(**fields)


async def _catalog_now(catalog: Any) -> datetime:
    getter = getattr(catalog, "current_time", None)
    if getter is None:
        return _py_now()
    return _require_aware(await getter())


async def _call(catalog: Any, name: str, *args: Any, **kwargs: Any) -> Any:
    method = getattr(catalog, name, None)
    if method is None:
        raise ImagePersistenceUnsupportedError("state_store", store_type=type(catalog).__name__)
    return await method(*args, **kwargs)


async def _mark_purging_row(
    catalog: Any, scope: str, sha: str, reason: str | None
) -> tuple[str, str] | None:
    marker = getattr(catalog, "mark_purging_row", None)
    if marker is not None:
        result: tuple[str, str] | None = await marker(scope, sha, reason)
        return result
    generation = await catalog.mark_purging(scope, sha, reason)
    if generation is None:
        return None
    row = await catalog.get_row(scope, sha)
    return (generation, row.storage if row is not None else "external")


# ---------------------------------------------------------------------------
# In-memory store that collects garbage once when full
# ---------------------------------------------------------------------------


class GcOnFullBlobStore:
    """Wraps a :class:`BlobStore`: when ``put`` reports the store full, run one
    ``collect_garbage(older_than=1h)`` pass and retry the put once."""

    def __init__(self, inner: BlobStore) -> None:
        self.inner = inner
        self.accessor: CatalogBlobAccessor | None = None

    async def put(self, key: BlobKey, data: bytes, *, media_type: str) -> None:
        try:
            await self.inner.put(key, data, media_type=media_type)
            return
        except ImageLimitError as exc:
            if self.accessor is None or exc.scope != "store_bytes":
                raise
        await self.accessor.collect_garbage(older_than=MIN_GC_AGE)
        await self.inner.put(key, data, media_type=media_type)

    async def get(self, key: BlobKey) -> bytes | None:
        return await self.inner.get(key)

    async def delete(self, key: BlobKey) -> None:
        await self.inner.delete(key)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlobServices:
    """Catalog, store, resolver and accessor for one Agent / RunStore.

    ``catalog`` and ``resolver`` are ``None`` for a state store without a blob
    catalog (then the accessor and ingest raise
    ``ImagePersistenceUnsupportedError(reason="state_store")``).
    """

    catalog: BlobCatalog | None
    blob_store: BlobStore | None
    resolver: BlobResolver | None
    accessor: CatalogBlobAccessor
    memory: InMemoryBlobCatalog | None = field(default=None)

    @classmethod
    def for_state_store(
        cls,
        state_store: Any,
        *,
        blob_store: BlobStore | None = None,
        cache_bytes: int = DEFAULT_CACHE_BYTES,
    ) -> BlobServices:
        """Services over a state store (bytes in the DB unless ``blob_store`` is given)."""
        catalog = as_blob_catalog(state_store)
        resolver = (
            CachingBlobResolver(catalog, blob_store, max_bytes=cache_bytes)
            if catalog is not None
            else None
        )
        accessor = CatalogBlobAccessor(
            catalog,
            blob_store=blob_store,
            resolver=resolver,
            store_type=type(state_store).__name__,
        )
        return cls(catalog=catalog, blob_store=blob_store, resolver=resolver, accessor=accessor)

    @classmethod
    def in_memory(
        cls,
        *,
        blob_store: BlobStore | None = None,
        cache_bytes: int = DEFAULT_CACHE_BYTES,
    ) -> BlobServices:
        """Process-local services for an agent without a database (non-durable).

        Bytes go to ``blob_store`` (default: a new :class:`InMemoryBlobStore`),
        wrapped so a full store triggers one GC pass before failing.
        """
        catalog = InMemoryBlobCatalog()
        store = GcOnFullBlobStore(blob_store if blob_store is not None else InMemoryBlobStore())
        resolver = CachingBlobResolver(catalog, store, max_bytes=cache_bytes)
        accessor = CatalogBlobAccessor(catalog, blob_store=store, resolver=resolver)
        store.accessor = accessor
        return cls(
            catalog=catalog, blob_store=store, resolver=resolver, accessor=accessor, memory=catalog
        )


# ---------------------------------------------------------------------------
# provider_request scrub (defence in depth; adapters sanitise first)
# ---------------------------------------------------------------------------

# Any data URL with a base64 payload (any media type, any parameters).
_DATA_URL_RE = re.compile(r"data:[^,\s\"'<>]*?;base64,([A-Za-z0-9+/_=-]*)", re.IGNORECASE)
# A base64 run long enough to be payload, not prose (the §5.5 >= 200 rule).
_B64_RUN_RE = re.compile(r"[A-Za-z0-9+/_-]{200,}={0,2}")
IMAGE_MARKER_PREFIX = "dendrux-image:sha256:"
REDACTED_MARKER = "dendrux-image:redacted"


def _marker_for_b64(b64: str) -> str:
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        return REDACTED_MARKER
    return IMAGE_MARKER_PREFIX + hashlib.sha256(raw).hexdigest()


def _scrub_str(text: str) -> str:
    if len(text) < 200 and ";base64," not in text.lower():
        return text
    out = _DATA_URL_RE.sub(lambda m: _marker_for_b64(m.group(1)), text)
    out = _B64_RUN_RE.sub(lambda m: _marker_for_b64(m.group(0)), out)
    return text if out == text else out


def scrub_inline_images(value: Any) -> Any:
    """Replace any inline base64 payload left in a captured request by a marker.

    Covers data URLs of any media type and parameters
    (``data:image/png;charset=x;base64,...``), Anthropic base64 sources
    (``{"type": "base64", "data": ...}``, any length), and any base64 run of
    200+ characters anywhere in a string (bare values or embedded in text).
    Each payload becomes ``"dendrux-image:sha256:<hex>"`` (hash of the decoded
    bytes) or ``"dendrux-image:redacted"`` when it does not decode. Returns
    the input object itself when nothing changed.
    """
    if isinstance(value, str):
        return _scrub_str(value)
    if isinstance(value, dict):
        changed = False
        out: dict[Any, Any] = {}
        is_b64_source = value.get("type") == "base64" and isinstance(value.get("data"), str)
        for k, v in value.items():
            if (
                is_b64_source
                and k == "data"
                and not v.startswith((IMAGE_MARKER_PREFIX, REDACTED_MARKER))
            ):
                new: Any = _marker_for_b64(v)
            else:
                new = scrub_inline_images(v)
            changed = changed or new is not v
            out[k] = new
        return out if changed else value
    if isinstance(value, list):
        items = [scrub_inline_images(v) for v in value]
        return items if any(a is not b for a, b in zip(items, value, strict=True)) else value
    return value
