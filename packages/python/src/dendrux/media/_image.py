"""Image inputs (:class:`Image`) and the tool return wrapper (:class:`ToolOutput`)."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias
from urllib.parse import urlsplit

from dendrux.types import ImageDetail, ImagePart, ImageRef

Detail: TypeAlias = ImageDetail
ImageSource: TypeAlias = Literal["bytes", "path", "base64", "url"]
_DETAILS = ("auto", "low", "high")


def _norm_media_type(media_type: str | None) -> str | None:
    if media_type is None:
        return None
    if not isinstance(media_type, str):
        raise TypeError("media_type must be a string or None")
    mt = media_type.split(";", 1)[0].strip().lower()
    if not mt:
        return None  # blank / parameters-only: treated as undeclared (the sniff decides)
    return "image/jpeg" if mt == "image/jpg" else mt


def _check_detail(detail: str | None) -> None:
    if detail is not None and detail not in _DETAILS:
        raise ValueError(f"detail must be one of {_DETAILS} or None")


@dataclass(frozen=True, repr=False)
class Image:
    """An image to hand to Dendrux: bytes, a file path, base64 text or a URL to fetch once.

    Build one with :meth:`from_bytes`, :meth:`from_path`, :meth:`from_base64`
    or :meth:`from_url`. Nothing is read, decoded or fetched until ingest,
    which validates, caps and stores the bytes as a blob. ``repr`` shows the
    source kind and size only, never content, paths or URL paths/queries.
    """

    source: ImageSource
    data: bytes | None = field(default=None, compare=True)
    path: str | None = None
    text: str | None = None
    url: str | None = None
    media_type: str | None = None
    detail: Detail | None = None

    @classmethod
    def from_bytes(
        cls, data: bytes, *, media_type: str | None = None, detail: Detail | None = None
    ) -> Image:
        """Raw image bytes."""
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(f"Image.from_bytes() needs bytes, got {type(data).__name__}")
        _check_detail(detail)
        return cls(
            source="bytes", data=bytes(data), media_type=_norm_media_type(media_type), detail=detail
        )

    @classmethod
    def from_path(
        cls,
        path: str | os.PathLike[str],
        *,
        media_type: str | None = None,
        detail: Detail | None = None,
    ) -> Image:
        """A local file, read (bounded) at ingest."""
        fs_path = os.fspath(path)
        if not isinstance(fs_path, str):
            raise TypeError("Image.from_path() needs a str path")
        _check_detail(detail)
        return cls(
            source="path", path=fs_path, media_type=_norm_media_type(media_type), detail=detail
        )

    @classmethod
    def from_base64(
        cls, data: str, *, media_type: str | None = None, detail: Detail | None = None
    ) -> Image:
        """Base64 text, or a ``data:<media type>;base64,<data>`` URL. Decoded strictly at ingest."""
        if isinstance(data, (bytes, bytearray)):
            raw = bytes(data)
            if not raw.isascii():
                raise TypeError("Image.from_base64() needs ASCII text")
            data = raw.decode("ascii")
        if not isinstance(data, str):
            raise TypeError(f"Image.from_base64() needs a str, got {type(data).__name__}")
        _check_detail(detail)
        return cls(
            source="base64", text=data, media_type=_norm_media_type(media_type), detail=detail
        )

    @classmethod
    def from_url(cls, url: str, *, detail: Detail | None = None) -> Image:
        """A URL fetched once at ingest (SSRF-hardened) and stored; the URL is never persisted."""
        if not isinstance(url, str):
            raise TypeError(f"Image.from_url() needs a str, got {type(url).__name__}")
        _check_detail(detail)
        return cls(source="url", url=url, detail=detail)

    def __post_init__(self) -> None:
        payload = {"bytes": self.data, "path": self.path, "base64": self.text, "url": self.url}
        if self.source not in payload:
            raise ValueError(f"Image.source must be one of {tuple(payload)}, got {self.source!r}")
        if payload[self.source] is None or sum(v is not None for v in payload.values()) != 1:
            raise ValueError("build an Image with Image.from_bytes/from_path/from_base64/from_url")
        _check_detail(self.detail)

    def __repr__(self) -> str:
        fields = [f"source={self.source!r}"]
        if self.source == "bytes":
            fields.append(f"bytes={len(self.data or b'')}")
        elif self.source == "base64":
            fields.append(f"chars={len(self.text or '')}")
        elif self.source == "url":
            fields.append(f"host={_safe_host(self.url)!r}")
        if self.source != "url":
            fields.append(f"media_type={self.media_type!r}")
        if self.detail is not None:
            fields.append(f"detail={self.detail!r}")
        return f"Image({', '.join(fields)})"

    __str__ = __repr__


def _safe_host(url: str | None) -> str | None:
    if not url:
        return None
    try:
        return urlsplit(url).hostname
    except ValueError:
        return None


ImageInput: TypeAlias = Image | ImageRef | ImagePart
UserContent: TypeAlias = str | Sequence[str | ImageInput]


def _check_image_inputs(images: Sequence[Any], owner: str) -> tuple[ImageInput, ...]:
    if isinstance(images, (str, bytes, Image, ImageRef, ImagePart)):
        raise TypeError(f"{owner}.images must be a sequence of images")
    out = tuple(images)
    for img in out:
        if not isinstance(img, (Image, ImageRef, ImagePart)):
            raise TypeError(
                f"{owner}.images items must be Image, ImageRef or ImagePart, "
                f"got {type(img).__name__}"
            )
    return out


@dataclass(frozen=True)
class ToolOutput:
    """A tool return value with images.

    ``result`` is serialised exactly like a plain return (``json.dumps``);
    ``images`` are appended after that text, in order.
    """

    result: Any = None
    images: Sequence[ImageInput] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "images", _check_image_inputs(self.images, "ToolOutput"))
