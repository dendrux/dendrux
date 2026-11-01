"""Duplicate-code gate (pylint's similarity engine, ``symilar``): a new copy-pasted block
fails; a known block may not grow beyond a small tolerance.

Settings: at least 10 similar lines; imports, signatures, comments and docstrings ignored.
Baseline: ``tools/duplicate_baseline.json``, ``{"module_a|module_b": [sizes, largest
first]}``. A block is matched to a baseline block of the same file pair whose size is
within the tolerance (the larger of 2 lines and 15 percent), so unrelated edits that
shift a block by a line or two do not break the gate.

    python tools/duplicate_ratchet.py check
    python tools/duplicate_ratchet.py stamp      # after improving; refuses to add blocks
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

try:
    from tools import _ratchet as r
except ImportError:  # run as a script from packages/python/tools
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import _ratchet as r

MIN_LINES = 10
BASELINE = Path(__file__).resolve().parent / "duplicate_baseline.json"
_HEADER = re.compile(r"^(\d+) similar lines in (\d+) files$")
_FILE = re.compile(r"^==(?P<path>.+?):\[\d+:\d+\]$")


@dataclass(frozen=True)
class Block:
    a: str
    b: str
    size: int

    @property
    def pair(self) -> str:
        return f"{self.a}|{self.b}"


def _module(path: str) -> str:
    return r.module_name(Path(path).resolve())


def parse_symilar(text: str) -> list[Block]:
    blocks: list[Block] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        header = _HEADER.match(lines[i])
        i += 1
        if not header or int(header.group(2)) != 2:
            continue
        files = []
        while i < len(lines) and (m := _FILE.match(lines[i])):
            files.append(_module(m.group("path")))
            i += 1
        if len(files) == 2:
            a, b = sorted(files)
            blocks.append(Block(a, b, int(header.group(1))))
    return blocks


def baseline_from_blocks(blocks: list[Block]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for block in blocks:
        grouped[block.pair].append(block.size)
    return {pair: sorted(sizes, reverse=True) for pair, sizes in sorted(grouped.items())}


def _tolerance(size: int) -> int:
    return max(2, math.ceil(size * 0.15))


def compare(blocks: list[Block], baseline: Mapping[str, list[int]]) -> r.CountComparison:
    result = r.CountComparison()
    unused = {pair: sorted(sizes) for pair, sizes in baseline.items()}
    for block in sorted(blocks, key=lambda b: (b.pair, -b.size)):
        candidates = [s for s in unused.get(block.pair, []) if block.size <= s + _tolerance(s)]
        if candidates:
            unused[block.pair].remove(candidates[0])
        else:
            result.new.append(f"{block.pair} ({block.size} lines)")
    for pair, sizes in sorted(unused.items()):
        result.improved.extend(f"{pair} ({size} lines)" for size in sizes)
    return result


def collect_blocks(src: Path = r.SRC) -> list[Block]:
    files = [str(p) for p in r.iter_sources(src)]
    # run_symilar reads sys.argv[1:]; `-c` keeps this independent of the PATH.
    code = "from pylint import run_symilar; run_symilar()"
    cmd = [sys.executable, "-c", code, "-d", str(MIN_LINES), "-i", "--ignore-docstrings"]
    cmd += ["--ignore-imports", "--ignore-signatures", *files]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return parse_symilar(proc.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    r.add_common_args(parser)
    parser.add_argument("--src", type=Path, default=r.SRC)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args(argv)
    blocks = collect_blocks(args.src.resolve())
    old = r.baseline_or_none(args.baseline)
    return r.finish(
        args.command,
        label="duplicates",
        comparison=None if old is None else compare(blocks, old),
        new_baseline=baseline_from_blocks(blocks),
        path=args.baseline,
        allow_regress=args.allow_regress,
        reason=args.reason,
        describe=lambda text: f"new duplicated block {text}",
    )


if __name__ == "__main__":
    raise SystemExit(main())
