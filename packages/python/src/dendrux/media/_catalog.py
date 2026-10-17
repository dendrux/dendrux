"""Blob catalog and resolver interfaces (implemented by the persistence layer).

Defined here, below ``runtime``, so the media pipeline can depend on them
without importing the state store. ``SQLAlchemyStateStore`` implements
:class:`BlobCatalog`; ``runtime/_blob_catalog.py`` implements
:class:`BlobResolver`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from dendrux.media._store import BlobInfo, BlobKey, BlobReference


@dataclass(frozen=True)
class BlobEdge:
    """One reference from a run-owned record to a blob, written in the record's transaction."""

    tenant_scope: str
    sha256: str
    owner_kind: str
    owner_id: str
    ordinal: int


@runtime_checkable
class BlobCatalog(Protocol):
    """Blob rows, lifecycle transitions and edges. Every transition is one conditional statement."""

    async def get_current(self, scope: str, sha256s: Sequence[str]) -> dict[str, BlobInfo]: ...

    async def touch_live(self, scope: str, sha256s: Sequence[str]) -> dict[str, BlobInfo]: ...

    async def get_row(self, scope: str, sha256: str) -> BlobInfo | None: ...

    async def insert_row(
        self,
        scope: str,
        sha256: str,
        info: BlobInfo,
        *,
        state: Literal["pending", "live"],
        generation: str,
        data: bytes | None,
    ) -> bool: ...

    async def promote(self, scope: str, sha256: str, generation: str) -> bool: ...

    async def revive(
        self,
        scope: str,
        sha256: str,
        old_generation: str,
        new_generation: str,
        *,
        to_state: Literal["pending", "live"],
        data: bytes | None,
    ) -> bool: ...

    async def mark_deleting(
        self, scope: str, sha256: str, cutoff: datetime, *, stale_pending_cutoff: datetime
    ) -> str | None: ...

    async def delete_row(self, scope: str, sha256: str, generation: str) -> bool: ...

    async def mark_purging(self, scope: str, sha256: str, reason: str | None) -> str | None: ...

    async def finish_purge(self, scope: str, sha256: str, generation: str) -> None: ...

    async def stuck_rows(self, older_than: datetime, limit: int) -> list[BlobInfo]: ...

    async def read_blob_data(self, key: BlobKey) -> bytes | None: ...

    async def add_blob_refs(
        self, run_id: str, run_scope: str, edges: Sequence[BlobEdge]
    ) -> None: ...

    async def run_references(self, run_scope: str, run_id: str, sha256: str) -> bool: ...

    async def blob_references(self, scope: str, sha256: str) -> list[BlobReference]: ...

    async def delete_orphan_edges(self, limit: int) -> int: ...


@runtime_checkable
class BlobResolver(Protocol):
    """Reads verified bytes of a live blob row (DB or external store, cached).

    ``info`` must come from this request's ``get_current`` query. Raises
    ``BlobIntegrityError`` on a verification failure and ``BlobNotFoundError``
    when the bytes are absent.
    """

    async def read(self, scope: str, info: BlobInfo) -> bytes: ...
