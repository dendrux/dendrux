"""SSRF-hardened, fetch-once URL reader for ``Image.from_url``."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dendrux.media._policy import UrlPolicy


async def fetch_url(url: str, *, policy: UrlPolicy, max_bytes: int) -> bytes:
    """Fetch ``url`` under ``policy`` with a hard ``max_bytes`` ceiling.

    Raises:
        ImageFetchError: on any refusal or failure; only the host is reported.
    """
    raise NotImplementedError
