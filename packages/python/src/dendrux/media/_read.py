"""Bounded readers: every source is capped before allocation."""

from __future__ import annotations


def check_bytes_source(data: bytes, max_source_bytes: int, *, index: int | None = None) -> bytes:
    """Reject raw bytes over ``max_source_bytes`` (``ImageLimitError("source_bytes")``)."""
    raise NotImplementedError


async def read_path_bounded(path: str, max_source_bytes: int, *, index: int | None = None) -> bytes:
    """Read a regular file of at most ``max_source_bytes`` (stat first, then a bounded read)."""
    raise NotImplementedError


def decode_base64_bounded(
    text: str, max_source_bytes: int, *, index: int | None = None
) -> tuple[bytes, str | None]:
    """Strictly decode base64 or a ``data:<mt>;base64,`` URL; oversize text is rejected undecoded.

    Returns the bytes and the data URL's media type (``None`` for plain base64).
    """
    raise NotImplementedError
