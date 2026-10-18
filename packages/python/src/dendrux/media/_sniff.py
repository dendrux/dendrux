"""Magic-byte sniffing and header-only parsing of png/jpeg/gif/webp (no pixel decoding).

Parsers walk the in-memory buffer only; nothing is allocated in proportion to
declared dimensions. Structural checks catch truncation and inconsistent
lengths; pixel-level corruption is left to an ingest hook or the provider.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dendrux.errors import BlobIntegrityError, InvalidImageError

if TYPE_CHECKING:
    from dendrux.media._store import BlobInfo

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class ImageHeader:
    """What the header parser read: sniffed type, dimensions and frame count."""

    media_type: str
    width: int
    height: int
    frame_count: int = 1


class _CorruptError(Exception):
    """Internal: structural problem; converted to InvalidImageError("corrupt")."""


def sniff_media_type(data: bytes) -> str | None:
    """Media type from magic bytes, or ``None`` when not one of the four supported types."""
    if data.startswith(PNG_SIGNATURE):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def inspect_image(data: bytes, *, index: int | None = None) -> ImageHeader:
    """Sniff, structurally validate and header-parse ``data``.

    Raises:
        InvalidImageError: ``"empty"``, ``"unsupported_type"`` or ``"corrupt"``.
    """
    if not data:
        raise InvalidImageError("empty", index=index, byte_length=0)
    media_type = sniff_media_type(data)
    if media_type is None:
        raise InvalidImageError("unsupported_type", index=index, byte_length=len(data))
    parser = _PARSERS[media_type]
    corrupt = False
    header: ImageHeader | None = None
    try:
        header = parser(data)
    except (_CorruptError, struct.error, IndexError):
        corrupt = True
    if corrupt or header is None or header.width <= 0 or header.height <= 0:
        raise InvalidImageError(
            "corrupt", index=index, media_type=media_type, byte_length=len(data)
        )
    return header


def _u32be(data: bytes, pos: int) -> int:
    if pos + 4 > len(data):
        raise _CorruptError
    return int.from_bytes(data[pos : pos + 4], "big")


def _parse_png(data: bytes) -> ImageHeader:
    pos = len(PNG_SIGNATURE)
    end = len(data)
    width = height = 0
    first = True
    while True:
        if pos + 8 > end:
            raise _CorruptError  # no IEND
        length = _u32be(data, pos)
        kind = data[pos + 4 : pos + 8]
        body_start = pos + 8
        crc_pos = body_start + length
        if length > 0x7FFFFFFF or crc_pos + 4 > end:
            raise _CorruptError
        if first:
            if kind != b"IHDR" or length != 13:
                raise _CorruptError
            body = data[body_start:crc_pos]
            if zlib.crc32(kind + body) != _u32be(data, crc_pos):
                raise _CorruptError
            width, height = struct.unpack(">II", body[:8])
            first = False
        elif kind == b"IHDR":
            raise _CorruptError
        if kind == b"IEND":
            break
        pos = crc_pos + 4
    return ImageHeader("image/png", width, height, 1)


# SOF markers carrying frame dimensions: C0-C3, C5-C7, C9-CB, CD-CF.
_SOF_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
_STANDALONE = frozenset({0x01, *range(0xD0, 0xD8)})


def _parse_jpeg(data: bytes) -> ImageHeader:
    end = len(data)
    pos = 2  # after SOI
    width = height = 0
    while True:
        # Find the next marker (skip fill bytes 0xFF).
        if pos >= end or data[pos] != 0xFF:
            raise _CorruptError
        while pos < end and data[pos] == 0xFF:
            pos += 1
        if pos >= end:
            raise _CorruptError
        marker = data[pos]
        pos += 1
        if marker == 0xD9:  # EOI
            break
        if marker == 0xD8 or marker == 0x00:
            raise _CorruptError
        if marker in _STANDALONE:
            continue
        if pos + 2 > end:
            raise _CorruptError
        seg_len = int.from_bytes(data[pos : pos + 2], "big")
        if seg_len < 2 or pos + seg_len > end:
            raise _CorruptError
        if marker in _SOF_MARKERS and width == 0:
            if seg_len < 8:
                raise _CorruptError
            height = int.from_bytes(data[pos + 3 : pos + 5], "big")
            width = int.from_bytes(data[pos + 5 : pos + 7], "big")
            if width == 0 or height == 0:
                raise _CorruptError
        pos += seg_len
        if marker == 0xDA:  # SOS: skip entropy-coded data to the next real marker
            while True:
                ff = data.find(b"\xff", pos)
                if ff < 0 or ff + 1 >= end:
                    raise _CorruptError
                nxt = data[ff + 1]
                if nxt == 0x00 or 0xD0 <= nxt <= 0xD7 or nxt == 0xFF:
                    pos = ff + 1 if nxt == 0xFF else ff + 2
                    continue
                pos = ff
                break
    if width == 0:
        raise _CorruptError
    return ImageHeader("image/jpeg", width, height, 1)


def _skip_gif_subblocks(data: bytes, pos: int) -> int:
    end = len(data)
    while True:
        if pos >= end:
            raise _CorruptError
        size = data[pos]
        pos += 1
        if size == 0:
            return pos
        pos += size
        if pos > end:
            raise _CorruptError


def _parse_gif(data: bytes) -> ImageHeader:
    end = len(data)
    if end < 13:
        raise _CorruptError
    width, height, flags = struct.unpack("<HHB", data[6:11])
    pos = 13
    if flags & 0x80:
        pos += 3 * (2 << (flags & 0x07))
    frames = 0
    while True:
        if pos >= end:
            raise _CorruptError  # no trailer
        block = data[pos]
        if block == 0x3B:
            break
        if block == 0x21:  # extension: label + sub-blocks
            if pos + 2 > end:
                raise _CorruptError
            pos = _skip_gif_subblocks(data, pos + 2)
        elif block == 0x2C:  # image descriptor
            if pos + 10 > end:
                raise _CorruptError
            local = data[pos + 9]
            pos += 10
            if local & 0x80:
                pos += 3 * (2 << (local & 0x07))
            pos += 1  # LZW minimum code size
            pos = _skip_gif_subblocks(data, pos)
            frames += 1
        else:
            raise _CorruptError
    if frames == 0:
        raise _CorruptError
    return ImageHeader("image/gif", width, height, frames)


def _parse_webp(data: bytes) -> ImageHeader:
    riff_size = int.from_bytes(data[4:8], "little")
    if riff_size + 8 != len(data) or riff_size < 4:
        raise _CorruptError
    end = len(data)
    pos = 12
    width = height = 0
    frames = 1
    animated = False
    anmf = 0
    has_image = False
    first = True
    while pos < end:
        if pos + 8 > end:
            raise _CorruptError
        kind = data[pos : pos + 4]
        size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        body = pos + 8
        if body + size > end:
            raise _CorruptError
        if kind == b"VP8 ":
            if size < 10 or data[body + 3 : body + 6] != b"\x9d\x01\x2a":
                raise _CorruptError
            w = int.from_bytes(data[body + 6 : body + 8], "little") & 0x3FFF
            h = int.from_bytes(data[body + 8 : body + 10], "little") & 0x3FFF
            if first:
                width, height = w, h
            has_image = True
        elif kind == b"VP8L":
            if size < 5 or data[body] != 0x2F:
                raise _CorruptError
            bits = int.from_bytes(data[body + 1 : body + 5], "little")
            w = (bits & 0x3FFF) + 1
            h = ((bits >> 14) & 0x3FFF) + 1
            if first:
                width, height = w, h
            has_image = True
        elif kind == b"VP8X":
            if not first or size < 10:
                raise _CorruptError
            animated = bool(data[body] & 0x02)
            width = int.from_bytes(data[body + 4 : body + 7], "little") + 1
            height = int.from_bytes(data[body + 7 : body + 10], "little") + 1
        elif kind == b"ANMF":
            if size < 16:
                raise _CorruptError
            anmf += 1
            has_image = True
        first = False
        pos = body + size + (size & 1)
    if pos > end + 1 or not has_image:
        raise _CorruptError
    if animated:
        if anmf == 0:
            raise _CorruptError
        frames = anmf
    return ImageHeader("image/webp", width, height, frames)


_PARSERS = {
    "image/png": _parse_png,
    "image/jpeg": _parse_jpeg,
    "image/gif": _parse_gif,
    "image/webp": _parse_webp,
}


def verify_blob_bytes(data: bytes, info: BlobInfo) -> None:
    """Verify stored bytes against their catalog row (length, sha256, type, dims, frames).

    Raises:
        BlobIntegrityError: with ``check`` naming the first failed check.
    """
    image_id = "img_" + info.sha256[:12]
    check: str | None = None
    if len(data) != info.byte_length:
        check = "length"
    elif hashlib.sha256(data).hexdigest() != info.sha256:
        check = "sha256"
    else:
        try:
            header: ImageHeader | None = inspect_image(data)
        except InvalidImageError:
            header = None
        if header is None or header.media_type != info.media_type:
            check = "media_type"
        elif (header.width, header.height, header.frame_count) != (
            info.width,
            info.height,
            info.frame_count,
        ):
            check = "dimensions"
    if check is not None:
        raise BlobIntegrityError(image_id, check)
