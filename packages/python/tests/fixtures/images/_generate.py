"""Regenerate the image fixtures in this directory.

Run from this directory: ``python _generate.py``. Needs ``cjpeg``, ``cwebp`` and
``img2webp`` (libjpeg-turbo / libwebp) on PATH. The outputs are committed;
tests never run this script. PNG and GIF are encoded here in pure Python.
"""

from __future__ import annotations

import struct
import subprocess
import tempfile
import zlib
from pathlib import Path

HERE = Path(__file__).parent

# 5x7 bitmap glyphs (rows top to bottom, 5 bits each, MSB = left).
FONT = {
    "Z": [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
    "E": [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
    "B": [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
    "R": [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
    "A": [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    "K": [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
    "I": [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    "W": [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010],
    "2": [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
    "4": [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
    "7": [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
}


def render(text: str, w: int, h: int, bg: tuple[int, int, int], fg=(255, 255, 255), scale=8):
    rows = [[bg] * w for _ in range(h)]
    tw = len(text) * 6 * scale - scale
    x0, y0 = (w - tw) // 2, (h - 7 * scale) // 2
    for ci, ch in enumerate(text):
        glyph = FONT[ch]
        for gy, bits in enumerate(glyph):
            for gx in range(5):
                if bits & (1 << (4 - gx)):
                    for dy in range(scale):
                        for dx in range(scale):
                            rows[y0 + gy * scale + dy][x0 + ci * 6 * scale + gx * scale + dx] = fg
    return rows


def png_bytes(rows, alpha: int | None = None) -> bytes:
    h, w = len(rows), len(rows[0])
    color_type = 6 if alpha is not None else 2
    raw = bytearray()
    for row in rows:
        raw.append(0)
        for px in row:
            raw.extend(px)
            if alpha is not None:
                raw.append(alpha)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def ppm_bytes(rows) -> bytes:
    h, w = len(rows), len(rows[0])
    return f"P6 {w} {h} 255\n".encode() + b"".join(bytes(px) for row in rows for px in row)


def _lzw_literal(pixels: list[int], min_code_size: int = 8) -> bytes:
    # Literal-only LZW: a clear code every 250 pixels keeps every code at 9 bits.
    clear, eoi, size = 1 << min_code_size, (1 << min_code_size) + 1, min_code_size + 1
    codes: list[int] = []
    for i, p in enumerate(pixels):
        if i % 250 == 0:
            codes.append(clear)
        codes.append(p)
    codes.append(eoi)
    acc = nbits = 0
    out = bytearray()
    for c in codes:
        acc |= c << nbits
        nbits += size
        while nbits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            nbits -= 8
    if nbits:
        out.append(acc & 0xFF)
    blocks = bytearray([min_code_size])
    for i in range(0, len(out), 255):
        part = out[i : i + 255]
        blocks.append(len(part))
        blocks.extend(part)
    blocks.append(0)
    return bytes(blocks)


def gif_bytes(
    frames: list[list[int]], w: int, h: int, palette: list[tuple[int, int, int]]
) -> bytes:
    pal = list(palette) + [(0, 0, 0)] * (4 - len(palette))
    out = bytearray(b"GIF89a")
    out += struct.pack("<HHBBB", w, h, 0b10000001, 0, 0)  # GCT, 4 entries
    for c in pal[:4]:
        out += bytes(c)
    if len(frames) > 1:
        out += b"\x21\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00"
    for pixels in frames:
        out += b"\x21\xf9\x04\x00\x0a\x00\x00\x00"  # graphic control, 100 ms
        out += b"\x2c" + struct.pack("<HHHHB", 0, 0, w, h, 0)
        out += _lzw_literal(pixels)
    out += b"\x3b"
    return bytes(out)


def run(*cmd: str) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def main() -> None:
    red, green = (220, 30, 30), (30, 160, 60)
    zebra = render("ZEBRA42", 400, 200, red)
    kiwi = render("KIWI77", 400, 200, green)
    (HERE / "zebra42.png").write_bytes(png_bytes(zebra))
    (HERE / "kiwi77.png").write_bytes(png_bytes(kiwi))
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        (t / "zebra.ppm").write_bytes(ppm_bytes(zebra))
        run("cjpeg", "-quality", "85", "-outfile", str(HERE / "zebra42.jpg"), str(t / "zebra.ppm"))
        run(
            "cjpeg",
            "-quality",
            "85",
            "-progressive",
            "-outfile",
            str(HERE / "zebra42_progressive.jpg"),
            str(t / "zebra.ppm"),
        )
        big = [[(90, 120, 200)] * 4032 for _ in range(3024)]
        (t / "big.ppm").write_bytes(ppm_bytes(big))
        run(
            "cjpeg",
            "-quality",
            "30",
            "-outfile",
            str(HERE / "photo_4032x3024.jpg"),
            str(t / "big.ppm"),
        )
        run(
            "cwebp",
            "-quiet",
            "-q",
            "80",
            str(HERE / "zebra42.png"),
            "-o",
            str(HERE / "zebra42.webp"),
        )
        run(
            "cwebp",
            "-quiet",
            "-lossless",
            str(HERE / "zebra42.png"),
            "-o",
            str(HERE / "zebra42_lossless.webp"),
        )
        small = render("Z", 64, 64, red, scale=6)
        (t / "alpha.png").write_bytes(png_bytes(small, alpha=128))
        run(
            "cwebp", "-quiet", "-q", "80", str(t / "alpha.png"), "-o", str(HERE / "alpha_vp8x.webp")
        )
        (t / "f1.png").write_bytes(png_bytes(render("Z", 64, 64, red, scale=6)))
        (t / "f2.png").write_bytes(png_bytes(render("Z", 64, 64, green, scale=6)))
        run(
            "img2webp",
            "-loop",
            "0",
            str(t / "f1.png"),
            str(t / "f2.png"),
            "-o",
            str(HERE / "animated.webp"),
        )

    # GIF: 2-colour palette (bg, white).
    def gif_pixels(rows, bg):
        return [0 if px == bg else 1 for row in rows for px in row]

    (HERE / "zebra42.gif").write_bytes(
        gif_bytes([gif_pixels(zebra, red)], 400, 200, [red, (255, 255, 255)])
    )
    a = gif_pixels(render("Z", 32, 32, red, scale=3), red)
    b = [1 - p for p in a]
    (HERE / "animated.gif").write_bytes(gif_bytes([a, b], 32, 32, [red, (255, 255, 255)]))


if __name__ == "__main__":
    main()
