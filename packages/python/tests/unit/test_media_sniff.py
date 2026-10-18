"""Header-only sniffing/parsing of png/jpeg/gif/webp and stored-blob verification."""

from __future__ import annotations

import hashlib
import struct
import tracemalloc
import zlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from dendrux.errors import BlobIntegrityError, InvalidImageError
from dendrux.media import BlobInfo, ImageHeader, inspect_image, sniff_media_type, verify_blob_bytes

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "images"


def fx(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def chunk(kind: bytes, data: bytes, *, crc: int | None = None) -> bytes:
    c = zlib.crc32(kind + data) if crc is None else crc
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", c)


def png(width: int = 2, height: int = 1, *, ihdr_crc: int | None = None, end: bool = True) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = (b"\x00" + b"\x00\x00\x00" * min(width, 4)) * min(height, 4)
    out = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr, crc=ihdr_crc)
    out += chunk(b"IDAT", zlib.compress(raw))
    if end:
        out += chunk(b"IEND", b"")
    return out


def gif(frames: int = 1, *, trailer: bool = True, w: int = 1, h: int = 1) -> bytes:
    out = b"GIF89a" + struct.pack("<HHBBB", w, h, 0x80, 0, 0) + b"\x00\x00\x00\xff\xff\xff"
    for _ in range(frames):
        out += b"\x2c" + struct.pack("<HHHHB", 0, 0, w, h, 0) + b"\x02\x02\x44\x01\x00"
    return out + (b"\x3b" if trailer else b"")


def jpeg(*, sof: bool = True, eoi: bool = True, w: int = 3, h: int = 2) -> bytes:
    out = b"\xff\xd8"
    out += b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    if sof:
        out += b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, h, w, 1) + b"\x01\x11\x00"
    out += b"\xff\xda" + struct.pack(">H", 8) + b"\x01\x01\x00\x00\x3f\x00"
    out += b"\x12\x34\xff\x00\x56\xff\xd0\x78"  # stuffed byte + restart marker in scan data
    if eoi:
        out += b"\xff\xd9"
    return out


def webp_vp8l(w: int = 5, h: int = 7, *, riff_delta: int = 0) -> bytes:
    bits = (w - 1) | ((h - 1) << 14)
    payload = b"\x2f" + bits.to_bytes(4, "little") + b"\x00"
    body = b"WEBP" + b"VP8L" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", len(body) + riff_delta) + body


class TestFixtures:
    @pytest.mark.parametrize(
        ("name", "media_type", "size", "frames"),
        [
            ("zebra42.png", "image/png", (400, 200), 1),
            ("kiwi77.png", "image/png", (400, 200), 1),
            ("zebra42.jpg", "image/jpeg", (400, 200), 1),
            ("zebra42_progressive.jpg", "image/jpeg", (400, 200), 1),
            ("photo_4032x3024.jpg", "image/jpeg", (4032, 3024), 1),
            ("zebra42.gif", "image/gif", (400, 200), 1),
            ("animated.gif", "image/gif", (32, 32), 2),
            ("zebra42.webp", "image/webp", (400, 200), 1),
            ("zebra42_lossless.webp", "image/webp", (400, 200), 1),
            ("alpha_vp8x.webp", "image/webp", (64, 64), 1),
            ("animated.webp", "image/webp", (64, 64), 2),
        ],
    )
    def test_fixture(self, name: str, media_type: str, size: tuple[int, int], frames: int) -> None:
        data = fx(name)
        assert sniff_media_type(data) == media_type
        assert inspect_image(data) == ImageHeader(media_type, size[0], size[1], frames)

    def test_trailing_bytes_after_end_marker_accepted(self) -> None:
        assert inspect_image(fx("zebra42.jpg") + b"trailer").width == 400
        assert inspect_image(fx("zebra42.png") + b"x").width == 400


class TestRejections:
    def test_empty(self) -> None:
        with pytest.raises(InvalidImageError) as ei:
            inspect_image(b"", index=3)
        assert ei.value.reason == "empty" and ei.value.index == 3

    @pytest.mark.parametrize(
        "data",
        [b"BM" + b"\x00" * 60, b"<svg/>", b"%PDF-1.7", b"II*\x00", b"RIFF\x00\x00\x00\x00WAVE"],
    )
    def test_unsupported_type(self, data: bytes) -> None:
        assert sniff_media_type(data) is None
        with pytest.raises(InvalidImageError) as ei:
            inspect_image(data)
        assert ei.value.reason == "unsupported_type"

    @pytest.mark.parametrize(
        "data",
        [
            png(ihdr_crc=0),  # IHDR CRC mismatch
            png(end=False),  # no IEND
            png()[:20],  # truncated inside IHDR
            png()[:-6],  # truncated inside IEND
            png(width=0),  # zero width
            b"\x89PNG\r\n\x1a\n" + chunk(b"IDAT", b"x") + chunk(b"IEND", b""),  # IHDR not first
            b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 0xFFFFFF) + b"IHDR" + b"\x00" * 20,  # length
        ],
    )
    def test_png_corrupt(self, data: bytes) -> None:
        with pytest.raises(InvalidImageError) as ei:
            inspect_image(data)
        assert ei.value.reason == "corrupt"
        assert ei.value.__context__ is None

    def test_png_duplicate_ihdr(self) -> None:
        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        data = sig + chunk(b"IHDR", ihdr) + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")
        with pytest.raises(InvalidImageError, match="corrupt"):
            inspect_image(data)

    @pytest.mark.parametrize(
        "data",
        [
            jpeg(eoi=False),
            jpeg(sof=False),
            jpeg(w=0),
            fx("zebra42.jpg")[:-200],
            b"\xff\xd8\xff\xe0\xff\xff",  # segment length beyond buffer
            b"\xff\xd8\xff\xe0\x00\x01\x00",  # segment length < 2
            b"\xff\xd8\xff\xd8\xff\xd9",  # second SOI
            b"\xff\xd8\xff\xe0\x00\x04\x00\x00\x00\x00",  # no marker after a segment
        ],
    )
    def test_jpeg_corrupt(self, data: bytes) -> None:
        with pytest.raises(InvalidImageError, match="corrupt"):
            inspect_image(data)

    def test_jpeg_minimal(self) -> None:
        assert inspect_image(jpeg()) == ImageHeader("image/jpeg", 3, 2, 1)

    @pytest.mark.parametrize(
        "data",
        [
            gif(trailer=False),
            gif(frames=0),
            gif(w=0),
            gif()[:-4],  # sub-block overrun
            b"GIF89a\x01\x00",  # header truncated
            gif()[:-1] + b"\x99\x3b",  # unknown block
        ],
    )
    def test_gif_corrupt(self, data: bytes) -> None:
        with pytest.raises(InvalidImageError, match="corrupt"):
            inspect_image(data)

    def test_gif_frames(self) -> None:
        assert inspect_image(gif(frames=3)).frame_count == 3
        ext = gif()[:19] + b"\x21\xfe\x03abc\x00" + gif()[19:]
        assert inspect_image(ext).frame_count == 1

    @pytest.mark.parametrize(
        "data",
        [
            webp_vp8l(riff_delta=4),  # RIFF size larger than file
            webp_vp8l(riff_delta=-2),  # RIFF size smaller than file
            webp_vp8l()[:-2],
            b"RIFF" + struct.pack("<I", 12) + b"WEBPVP8 " + struct.pack("<I", 0),  # no data
            b"RIFF" + struct.pack("<I", 4) + b"WEBP",  # no image chunk
        ],
    )
    def test_webp_corrupt(self, data: bytes) -> None:
        with pytest.raises(InvalidImageError, match="corrupt"):
            inspect_image(data)

    def test_webp_bad_vp8_start_code(self) -> None:
        data = bytearray(fx("zebra42.webp"))
        data[20 + 3] ^= 0xFF
        with pytest.raises(InvalidImageError, match="corrupt"):
            inspect_image(bytes(data))

    def test_webp_vp8l_dims(self) -> None:
        assert inspect_image(webp_vp8l(5, 7)) == ImageHeader("image/webp", 5, 7, 1)


class TestBombHeaders:
    def test_png_bomb_header_parsed_without_allocation(self) -> None:
        data = png(width=100_000, height=100_000)
        tracemalloc.start()
        try:
            header = inspect_image(data)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert (header.width, header.height) == (100_000, 100_000)
        assert peak < 64 * 1024

    def test_jpeg_and_gif_declared_dims(self) -> None:
        assert inspect_image(jpeg(w=65535, h=65535)).width == 65535
        assert inspect_image(gif(w=65535, h=65535)).height == 65535


def _info(data: bytes, **kw: object) -> BlobInfo:
    h = inspect_image(data)
    now = datetime.now(UTC)
    base: dict[str, object] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "media_type": h.media_type,
        "byte_length": len(data),
        "width": h.width,
        "height": h.height,
        "frame_count": h.frame_count,
        "storage": "db",
        "state": "live",
        "generation": "01G",
        "created_at": now,
        "last_used_at": now,
    }
    base.update(kw)
    return BlobInfo(**base)  # type: ignore[arg-type]


class TestVerify:
    def test_ok(self) -> None:
        data = fx("zebra42.png")
        verify_blob_bytes(data, _info(data))

    @pytest.mark.parametrize(
        ("mutate", "check"),
        [
            (lambda d, i: (d + b"x", i), "length"),
            (lambda d, i: (d[:-1] + bytes([d[-1] ^ 1]), i), "sha256"),
            (lambda d, i: (d, _info(d, media_type="image/jpeg")), "media_type"),
            (lambda d, i: (d, _info(d, width=401)), "dimensions"),
            (lambda d, i: (d, _info(d, frame_count=2)), "dimensions"),
        ],
    )
    def test_failures(self, mutate: object, check: str) -> None:
        data = fx("zebra42.png")
        bad_data, info = mutate(data, _info(data))  # type: ignore[operator]
        with pytest.raises(BlobIntegrityError) as ei:
            verify_blob_bytes(bad_data, info)
        assert ei.value.check == check
        assert ei.value.__context__ is None

    def test_corrupt_bytes_with_matching_hash(self) -> None:
        data = png(ihdr_crc=0)
        info = _info(
            fx("zebra42.png"), sha256=hashlib.sha256(data).hexdigest(), byte_length=len(data)
        )
        with pytest.raises(BlobIntegrityError) as ei:
            verify_blob_bytes(data, info)
        assert ei.value.check == "media_type"
