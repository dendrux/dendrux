"""Blob storage protocol, the in-memory store, and blob metadata records."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, runtime_checkable
from urllib.parse import quote

from dendrux.errors import ImageLimitError

if TYPE_CHECKING:
    from datetime import datetime

    from dendrux.types import ImageRef

BlobState: TypeAlias = Literal["pending", "live", "purging", "purged", "deleting"]
BlobStorage: TypeAlias = Literal["db", "external"]


@dataclass(frozen=True)
class BlobKey:
    """Address of one stored object: tenant scope, content hash and row generation."""

    tenant_scope: str
    sha256: str
    generation: str

    @property
    def object_name(self) -> str:
        """Suggested object key: ``"<quoted scope>/<sha256>/<generation>"``."""
        return f"{quote(self.tenant_scope, safe='')}/{self.sha256}/{self.generation}"


@runtime_checkable
class BlobStore(Protocol):
    """External byte storage for image blobs (S3, GCS, disk, ...).

    Dendrux never trusts a store: every ``get`` is verified before use.
    ``put`` and ``delete`` must be idempotent per key; ``get`` returns ``None``
    when the object is absent.
    """

    async def put(self, key: BlobKey, data: bytes, *, media_type: str) -> None: ...

    async def get(self, key: BlobKey) -> bytes | None: ...

    async def delete(self, key: BlobKey) -> None: ...


class InMemoryBlobStore:
    """Process-local :class:`BlobStore` (non-durable). Raises ``ImageLimitError`` when full.

    ``calls`` records ``(operation, object_name)`` for every call, in order.
    """

    def __init__(self, max_bytes: int = 256 * 1024 * 1024) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("InMemoryBlobStore.max_bytes must be a positive int")
        self.max_bytes = max_bytes
        self._objects: dict[str, bytes] = {}
        self._size = 0
        self._lock = threading.Lock()
        self.calls: list[tuple[str, str]] = []

    @property
    def total_bytes(self) -> int:
        """Bytes currently stored."""
        return self._size

    def object_names(self) -> list[str]:
        """Names of the objects currently stored, sorted."""
        with self._lock:
            return sorted(self._objects)

    async def put(self, key: BlobKey, data: bytes, *, media_type: str) -> None:
        name = key.object_name
        with self._lock:
            self.calls.append(("put", name))
            previous = self._objects.get(name)
            new_size = self._size - (len(previous) if previous is not None else 0) + len(data)
            if new_size > self.max_bytes:
                over = new_size
            else:
                self._objects[name] = bytes(data)
                self._size = new_size
                return
        raise ImageLimitError("store_bytes", limit=self.max_bytes, actual=over)

    async def get(self, key: BlobKey) -> bytes | None:
        name = key.object_name
        with self._lock:
            self.calls.append(("get", name))
            return self._objects.get(name)

    async def delete(self, key: BlobKey) -> None:
        name = key.object_name
        with self._lock:
            self.calls.append(("delete", name))
            data = self._objects.pop(name, None)
            if data is not None:
                self._size -= len(data)


@dataclass(frozen=True)
class BlobInfo:
    """Catalog metadata of one blob row (never the bytes)."""

    sha256: str
    media_type: str
    byte_length: int
    width: int
    height: int
    frame_count: int
    storage: BlobStorage
    state: BlobState
    generation: str
    created_at: datetime
    last_used_at: datetime
    purged_at: datetime | None = None
    purge_reason: str | None = None
    ingest_transform: str | None = None
    # When the row last changed state (pending-staleness check in ingest).
    state_changed_at: datetime | None = None

    def to_ref(self) -> ImageRef:
        """The canonical :class:`~dendrux.types.ImageRef` for this row."""
        from dendrux.types import ImageRef

        return ImageRef(
            sha256=self.sha256,
            media_type=self.media_type,
            byte_length=self.byte_length,
            width=self.width,
            height=self.height,
        )


@dataclass(frozen=True)
class BlobReference:
    """One recorded use of a blob by a run (an edge)."""

    run_id: str
    owner_kind: str
    owner_id: str
    ordinal: int
    created_at: datetime


@dataclass(frozen=True)
class GcResult:
    """What one ``collect_garbage`` pass did."""

    deleted_blobs: int = 0
    deleted_orphan_edges: int = 0
    completed_purges: int = 0
    retried_deletes: int = 0
    skipped_referenced: int = 0
    bytes_freed: int = 0


@runtime_checkable
class BlobAccessor(Protocol):
    """Blob reads and lifecycle operations (``agent.blobs``, ``RunStore(...).blobs``).

    ``tenant_id`` is the authorisation boundary; absence and a tenant mismatch
    both raise :class:`~dendrux.errors.BlobNotFoundError`.
    """

    async def read(self, image: ImageRef | str, *, tenant_id: str | None = None) -> bytes: ...

    async def info(self, sha256: str, *, tenant_id: str | None = None) -> BlobInfo | None: ...

    async def references(
        self, sha256: str, *, tenant_id: str | None = None
    ) -> list[BlobReference]: ...

    async def purge(
        self, sha256: str, *, tenant_id: str | None = None, reason: str | None = None
    ) -> bool: ...

    async def collect_garbage(
        self, *, older_than: timedelta = timedelta(hours=24), limit: int = 1000
    ) -> GcResult: ...
