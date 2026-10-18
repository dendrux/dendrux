"""Per-run image runtime: policy, catalog, stores and tenant scope, passed explicitly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from dendrux.media._catalog import BlobCatalog, BlobResolver
    from dendrux.media._policy import ImagePolicy
    from dendrux.media._store import BlobStore


def tenant_scope_for(tenant_id: str | None) -> str:
    """Tenant scope of a tenant id: ``"t:" + tenant_id``, or ``"-"`` when absent."""
    return "-" if tenant_id is None else "t:" + tenant_id


@dataclass(frozen=True)
class ImageRuntime:
    """Everything image ingest and request preparation need for one run.

    Built by the runner once the run's tenant is known and passed explicitly
    (never through a ContextVar). ``catalog`` is ``None`` when the state store
    has no blob catalog; ingest then raises
    ``ImagePersistenceUnsupportedError(reason="state_store")``. ``blob_store``
    ``None`` means bytes live in the database.
    """

    policy: ImagePolicy
    catalog: BlobCatalog | None
    blob_store: BlobStore | None
    resolver: BlobResolver | None
    tenant_scope: str
    run_id: str | None = None
    # Optional "one garbage-collection pass" (e.g. ``agent.blobs.collect_garbage``
    # with ``older_than=1h``); ingest calls it once when the blob store is full.
    collect_garbage: Callable[[], Awaitable[object]] | None = None

    @property
    def storage(self) -> Literal["db", "external"]:
        """Where newly ingested bytes go."""
        return "db" if self.blob_store is None else "external"
