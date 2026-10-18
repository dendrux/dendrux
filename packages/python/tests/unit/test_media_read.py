"""Bounded readers: raw bytes, paths, strict and lenient base64."""

from __future__ import annotations

import base64
import os
import tracemalloc
from pathlib import Path

import pytest

from dendrux.errors import ImageLimitError, InvalidImageError
from dendrux.media import _read
from dendrux.media._read import (
    check_bytes_source,
    decode_base64_bounded,
    max_base64_chars,
    max_lenient_base64_chars,
    read_path_bounded,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "images"
PNG = (FIXTURES / "zebra42.png").read_bytes()


def test_bytes_cap() -> None:
    assert check_bytes_source(b"abc", 3) == b"abc"
    with pytest.raises(ImageLimitError) as ei:
        check_bytes_source(b"abcd", 3, index=2)
    assert (ei.value.scope, ei.value.limit, ei.value.actual, ei.value.index) == (
        "source_bytes",
        3,
        4,
        2,
    )


class TestPath:
    async def test_reads_file(self, tmp_path: Path) -> None:
        p = tmp_path / "a.png"
        p.write_bytes(PNG)
        assert await read_path_bounded(str(p), len(PNG)) == PNG

    async def test_missing_and_directory(self, tmp_path: Path) -> None:
        for target in (tmp_path / "missing.png", tmp_path):
            with pytest.raises(InvalidImageError) as ei:
                await read_path_bounded(str(target), 100, index=1)
            assert ei.value.reason == "path" and ei.value.index == 1
            assert str(tmp_path) not in str(ei.value)
            assert ei.value.__context__ is None

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs mkfifo")
    async def test_fifo_rejected_without_blocking(self, tmp_path: Path) -> None:
        fifo = tmp_path / "pipe"
        os.mkfifo(fifo)
        with pytest.raises(InvalidImageError, match="path"):
            await read_path_bounded(str(fifo), 100)

    async def test_oversize_rejected_by_stat(self, tmp_path: Path) -> None:
        cap = 8 * 1024 * 1024
        p = tmp_path / "big.bin"
        with open(p, "wb") as f:
            f.truncate(cap + 1)  # sparse: nothing is read
        tracemalloc.start()
        try:
            with pytest.raises(ImageLimitError) as ei:
                await read_path_bounded(str(p), cap)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert ei.value.scope == "source_bytes" and ei.value.actual == cap + 1
        assert peak < cap // 8

    async def test_file_that_grows_after_stat(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        p = tmp_path / "grows.bin"
        p.write_bytes(b"x" * 200)
        real_stat = os.stat

        class SmallStat:
            def __init__(self, st: os.stat_result) -> None:
                self.st_mode = st.st_mode
                self.st_size = 10

        monkeypatch.setattr(_read.os, "stat", lambda path: SmallStat(real_stat(path)))
        with pytest.raises(ImageLimitError) as ei:
            await read_path_bounded(str(p), 100)
        assert ei.value.actual == 101  # read stopped at cap + 1


class TestBase64Strict:
    def test_plain_and_data_url(self) -> None:
        b64 = base64.b64encode(PNG).decode()
        assert decode_base64_bounded(b64, len(PNG)) == (PNG, None)
        assert decode_base64_bounded("data:image/PNG;base64," + b64, len(PNG)) == (
            PNG,
            "image/png",
        )
        assert decode_base64_bounded("", 10) == (b"", None)

    def test_multi_chunk(self) -> None:
        data = os.urandom(200_000)
        b64 = base64.b64encode(data).decode()
        assert len(b64) > 2 * 64 * 1024
        assert decode_base64_bounded(b64, len(data))[0] == data

    @pytest.mark.parametrize(
        "text",
        [
            "aGVs bG8=",  # space
            "aGVsbG8=\n",  # newline
            "aGVs*G8=",  # alphabet
            "aGVsbG8",  # length not a multiple of 4
            "aG==bG8=",  # padding in the middle
            "data:image/png;base64",  # malformed data URL
            "data:;base64,aGVsbG8=",
            "aGVsbG8=é===",  # non-ascii
        ],
    )
    def test_bad_base64(self, text: str) -> None:
        with pytest.raises(InvalidImageError) as ei:
            decode_base64_bounded(text, 1000, index=4)
        assert ei.value.reason == "bad_base64" and ei.value.index == 4
        assert ei.value.__context__ is None and ei.value.__cause__ is None

    def test_padding_only_in_final_quantum_across_chunks(self) -> None:
        first = "QUFB" * (64 * 1024 // 4 - 1) + "QQ=="
        with pytest.raises(InvalidImageError, match="bad_base64"):
            decode_base64_bounded(first + "QUFB", 10**6)

    def test_oversize_rejected_before_decode(self) -> None:
        cap = 1024 * 1024
        text = "A" * (max_base64_chars(cap) + 4)
        tracemalloc.start()
        try:
            with pytest.raises(ImageLimitError) as ei:
                decode_base64_bounded(text, cap)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert ei.value.scope == "source_bytes"
        assert peak < 2 * cap
        assert peak < 64 * 1024  # nothing was decoded

    def test_exact_cap_boundary(self) -> None:
        data = os.urandom(3000)
        b64 = base64.b64encode(data).decode()
        assert decode_base64_bounded(b64, 3000)[0] == data
        with pytest.raises(ImageLimitError):
            decode_base64_bounded(b64, 2999)


class TestBase64Lenient:
    def test_mime_wrapped(self) -> None:
        wrapped = base64.encodebytes(PNG).decode().replace("\n", "\r\n")
        assert "\r\n" in wrapped
        assert decode_base64_bounded(wrapped, len(PNG), lenient=True)[0] == PNG
        with pytest.raises(InvalidImageError):
            decode_base64_bounded(wrapped, 10 * len(PNG))
        url = "  data:image/png;base64,\n" + wrapped
        assert decode_base64_bounded(url, len(PNG), lenient=True) == (PNG, "image/png")

    def test_raw_length_precheck(self) -> None:
        cap = 1000
        text = " " * (max_lenient_base64_chars(cap) + 1)
        with pytest.raises(ImageLimitError):
            decode_base64_bounded(text, cap, lenient=True)

    def test_stripped_length_still_checked(self) -> None:
        cap = 300
        b64 = base64.b64encode(os.urandom(cap + 3)).decode()
        with pytest.raises(ImageLimitError):
            decode_base64_bounded(b64, cap, lenient=True)

    def test_non_ascii_whitespace_still_rejected(self) -> None:
        with pytest.raises(InvalidImageError):
            decode_base64_bounded("aGVs bG8=", 100, lenient=True)
