"""Magic-byte sniffing and header-only parsing of png/jpeg/gif/webp (no pixel decoding)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dendrux.media._store import BlobInfo


@dataclass(frozen=True)
class ImageHeader:
    """What the header parser read: sniffed type, dimensions and frame count."""

    media_type: str
    width: int
    height: int
    frame_count: int = 1


def sniff_media_type(data: bytes) -> str | None:
    """Media type from magic bytes, or ``None`` when not one of the four supported types."""
    raise NotImplementedError


def inspect_image(data: bytes, *, index: int | None = None) -> ImageHeader:
    """Sniff, structurally validate and header-parse ``data``.

    Raises:
        InvalidImageError: ``"unsupported_type"`` or ``"corrupt"``.
    """
    raise NotImplementedError


def verify_blob_bytes(data: bytes, info: BlobInfo) -> None:
    """Verify stored bytes against their catalog row (length, sha256, type, dims, frames).

    Raises:
        BlobIntegrityError: with ``check`` naming the first failed check.
    """
    raise NotImplementedError
