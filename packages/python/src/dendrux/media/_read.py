"""Bounded readers: every source is capped before allocation."""

from __future__ import annotations

import asyncio
import binascii
import os
import re
import stat

from dendrux.errors import ImageLimitError, InvalidImageError

_DATA_URL_RE = re.compile(r"data:([A-Za-z0-9.+/-]+);base64,", re.ASCII)
_CHUNK_CHARS = 64 * 1024  # multiple of 4 -> decodes on quantum boundaries


def check_bytes_source(data: bytes, max_source_bytes: int, *, index: int | None = None) -> bytes:
    """Reject raw bytes over ``max_source_bytes`` (``ImageLimitError("source_bytes")``)."""
    if len(data) > max_source_bytes:
        raise ImageLimitError("source_bytes", limit=max_source_bytes, actual=len(data), index=index)
    return data


_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NONBLOCK", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_BINARY", 0)
)


def _read_path_sync(path: str, max_source_bytes: int) -> tuple[bytes | None, str | None, int]:
    """Return (data, error, size). error is "path" or "too_large".

    ``stat`` rejects early without opening; the file is then opened
    non-blocking and re-checked with ``fstat`` on the open descriptor, so a
    path swapped for a FIFO or device after the ``stat`` can never block.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None, "path", 0
    if not stat.S_ISREG(st.st_mode):
        return None, "path", 0
    if st.st_size > max_source_bytes:
        return None, "too_large", st.st_size
    try:
        fd = os.open(path, _OPEN_FLAGS)
    except OSError:
        return None, "path", 0
    try:
        fst = os.fstat(fd)
        if not stat.S_ISREG(fst.st_mode):
            return None, "path", 0
        if fst.st_size > max_source_bytes:
            return None, "too_large", fst.st_size
        chunks: list[bytes] = []
        remaining = max_source_bytes + 1
        while remaining > 0:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
    except OSError:
        return None, "path", 0
    finally:
        os.close(fd)
    if len(data) > max_source_bytes:
        return None, "too_large", len(data)
    return data, None, len(data)


async def read_path_bounded(path: str, max_source_bytes: int, *, index: int | None = None) -> bytes:
    """Read a regular file of at most ``max_source_bytes`` (stat first, then a bounded read).

    Raises:
        InvalidImageError: ``reason="path"`` (missing, unreadable or not a regular file).
        ImageLimitError: ``scope="source_bytes"``.
    """
    data, error, size = await asyncio.to_thread(_read_path_sync, path, max_source_bytes)
    if error == "path":
        raise InvalidImageError("path", index=index)
    if error == "too_large":
        raise ImageLimitError("source_bytes", limit=max_source_bytes, actual=size, index=index)
    assert data is not None
    return data


def max_base64_chars(max_bytes: int) -> int:
    """Longest padded base64 text that can decode to at most ``max_bytes``."""
    return 4 * -(-max_bytes // 3)


_ASCII_WS = {ord(c): None for c in " \t\n\r\x0b\x0c"}


def max_lenient_base64_chars(max_bytes: int) -> int:
    """Raw-length ceiling for whitespace-wrapped base64 (checked before stripping)."""
    return int(1.05 * max_base64_chars(max_bytes)) + 4096


def decode_base64_bounded(
    text: str, max_source_bytes: int, *, index: int | None = None, lenient: bool = False
) -> tuple[bytes, str | None]:
    """Strictly decode base64 or a ``data:<mt>;base64,`` URL; oversize text is rejected undecoded.

    Strict: no whitespace is allowed anywhere. Lenient (MIME-wrapped data):
    the raw text must fit :func:`max_lenient_base64_chars` before ASCII
    whitespace is stripped; the stripped text then gets the strict checks.
    Returns the bytes and the data URL's media type (``None`` for plain base64).

    Raises:
        InvalidImageError: ``reason="bad_base64"``.
        ImageLimitError: ``scope="source_bytes"``.
    """
    if lenient:
        if len(text) > max_lenient_base64_chars(max_source_bytes):
            raise ImageLimitError(
                "source_bytes",
                limit=max_source_bytes,
                actual=(len(text) // 4) * 3,
                index=index,
            )
        text = text.lstrip(" \t\n\r\x0b\x0c")
    media_type: str | None = None
    start = 0
    if text.startswith("data:"):
        m = _DATA_URL_RE.match(text)
        if m is None:
            raise InvalidImageError("bad_base64", index=index)
        media_type = m.group(1).lower()
        start = m.end()
    if lenient:
        text = text[start:].translate(_ASCII_WS)
        start = 0
    n_chars = len(text) - start
    if n_chars > max_base64_chars(max_source_bytes):
        # Lower bound on the decoded size, reported without decoding.
        raise ImageLimitError(
            "source_bytes", limit=max_source_bytes, actual=(n_chars // 4) * 3, index=index
        )
    if n_chars == 0:
        return b"", media_type
    if n_chars % 4:
        raise InvalidImageError("bad_base64", index=index)
    out = bytearray()
    bad = False
    for pos in range(start, len(text), _CHUNK_CHARS):
        chunk = text[pos : pos + _CHUNK_CHARS]
        try:
            out += binascii.a2b_base64(chunk.encode("ascii"), strict_mode=True)
        except (binascii.Error, UnicodeEncodeError, ValueError):
            bad = True
            break
        # Padding may only appear in the final quantum.
        if "=" in chunk and pos + _CHUNK_CHARS < len(text):
            bad = True
            break
        if len(out) > max_source_bytes:
            break
    if bad:
        raise InvalidImageError("bad_base64", index=index)
    if len(out) > max_source_bytes:
        raise ImageLimitError("source_bytes", limit=max_source_bytes, actual=len(out), index=index)
    return bytes(out), media_type
