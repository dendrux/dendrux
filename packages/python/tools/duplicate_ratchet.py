"""Duplicate-code gate (pylint's similarity engine, ``Symilar``): a new copy-pasted block
fails; a known block may not grow beyond a small tolerance. Every pairwise commonality is
read from the engine (not its summarised report), so a third identical copy is three pairs.

Settings: at least 10 similar lines; imports, signatures, comments and docstrings ignored.
Baseline: ``tools/duplicate_baseline.json``, ``{"module_a|module_b": [sizes, largest
first]}``. A block is matched to a baseline block of the same file pair whose size is
within the tolerance (the larger of 2 lines and 15 percent), so unrelated edits that
shift a block by a line or two do not break the gate. A stamp keeps the old size for a
block that grew inside the tolerance, so it can lower a ceiling and never raise one.

    python tools/duplicate_ratchet.py check
    python tools/duplicate_ratchet.py stamp      # after improving; refuses to add blocks
"""

from __future__ import annotations

import argparse
import math
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


@dataclass(frozen=True)
class Block:
    a: str
    b: str
    size: int

    @property
    def pair(self) -> str:
        return "|".join(sorted((self.a, self.b)))


def baseline_from_blocks(
    blocks: list[Block], previous: Mapping[str, list[int]] | None = None
) -> dict[str, list[int]]:
    """Baseline for ``blocks``. Given the ``previous`` baseline, a block that matches an
    old entry keeps the old size when it grew within the tolerance, so a stamp can lower a
    ceiling but never raise one, and a block with no old entry is left out (``check`` would
    have failed on it). Without ``previous`` every block is recorded as it is."""
    if previous is None:
        grouped: dict[str, list[int]] = defaultdict(list)
        for block in blocks:
            grouped[block.pair].append(block.size)
        return {pair: sorted(sizes, reverse=True) for pair, sizes in sorted(grouped.items())}
    kept: dict[str, list[int]] = defaultdict(list)
    for block, ceiling in _match(blocks, previous)[0]:
        kept[block.pair].append(min(block.size, ceiling))
    return {pair: sorted(sizes, reverse=True) for pair, sizes in sorted(kept.items())}


def _tolerance(size: int) -> int:
    return max(2, math.ceil(size * 0.15))


def _match(
    blocks: list[Block], baseline: Mapping[str, list[int]]
) -> tuple[list[tuple[Block, int]], list[Block], dict[str, list[int]]]:
    """(blocks with the baseline size they matched, unmatched blocks, unused baseline sizes)."""
    unused = {pair: sorted(sizes) for pair, sizes in baseline.items()}
    matched: list[tuple[Block, int]] = []
    unmatched: list[Block] = []
    for block in sorted(blocks, key=lambda b: (b.pair, -b.size)):
        free = unused.get(block.pair, [])
        # the smallest entry that is not smaller than the block, else the closest one below
        # it that still covers the block within its tolerance
        roomy = [s for s in free if s >= block.size]
        grown = [s for s in free if s < block.size <= s + _tolerance(s)]
        pick = min(roomy) if roomy else (max(grown) if grown else None)
        if pick is not None:
            free.remove(pick)
            matched.append((block, pick))
        else:
            unmatched.append(block)
    return matched, unmatched, unused


def compare(blocks: list[Block], baseline: Mapping[str, list[int]]) -> r.CountComparison:
    result = r.CountComparison()
    _, unmatched, unused = _match(blocks, baseline)
    result.new.extend(f"{b.pair} ({b.size} lines)" for b in unmatched)
    for pair, sizes in sorted(unused.items()):
        result.improved.extend(f"{pair} ({size} lines)" for size in sizes)
    return result


def collect_blocks(src: Path = r.SRC) -> list[Block]:
    """Every pairwise commonality, before pylint's report summary merges the pairs that
    share a block (which hides a third identical copy of a duplicated body)."""
    from pylint.checkers.symilar import Symilar  # dev tool, imported only when the gate runs

    engine = Symilar(
        min_lines=MIN_LINES,
        ignore_comments=True,
        ignore_docstrings=True,
        ignore_imports=True,
        ignore_signatures=True,
    )
    for path in r.iter_sources(src):
        with path.open(encoding="utf-8") as stream:
            engine.append_stream(str(path.resolve()), stream)
    found: set[tuple[str, str, int, int, int, int, int]] = set()
    for c in engine._iter_sims():  # noqa: SLF001 - the pinned engine's pairwise iterator
        first, second = r.module_name(c.fst_lset.name), r.module_name(c.snd_lset.name)
        ends = (c.fst_file_start, c.fst_file_end, c.snd_file_start, c.snd_file_end)
        if second < first:
            first, second = second, first
            ends = (ends[2], ends[3], ends[0], ends[1])
        found.add((first, second, c.cmn_lines_nb, *(int(n) for n in ends)))
    return [Block(a, b, size) for a, b, size, *_ in sorted(found)]


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
        new_baseline=baseline_from_blocks(blocks, None if args.allow_regress else old),
        path=args.baseline,
        allow_regress=args.allow_regress,
        reason=args.reason,
        describe=lambda text: f"new duplicated block {text}",
    )


if __name__ == "__main__":
    raise SystemExit(main())
