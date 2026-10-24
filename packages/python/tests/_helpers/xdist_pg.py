"""Per-worker Postgres database names for ``pytest -n``."""

from __future__ import annotations

from urllib.parse import urlparse, urlunparse

_SUFFIX = "_test"


def worker_scoped_pg_url(url: str, worker_id: str | None) -> str:
    """Return ``url`` pointing at a database private to the xdist worker.

    ``dendrux_test`` becomes ``dendrux_gw0_test``; with no worker id (a serial run)
    the URL is unchanged. The result always ends in ``_test``, as the integration
    conftest requires before it will run destructive tests.
    """
    if not worker_id:
        return url
    parsed = urlparse(url)
    name = parsed.path.lstrip("/").removesuffix(_SUFFIX)
    return urlunparse(parsed._replace(path=f"/{name}_{worker_id}{_SUFFIX}"))
