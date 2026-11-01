"""tools/duplicate_ratchet.py: a new duplicated block fails; known blocks may not grow."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import pytest

from tools import duplicate_ratchet as dup

if TYPE_CHECKING:
    from pathlib import Path

SYMILAR = """
20 similar lines in 2 files
==src/dendrux/llm/openai.py:[614:638]
==src/dendrux/llm/openai_responses.py:[646:670]
           registry = WireImageRegistry()
           x = 1

12 similar lines in 2 files
==src/dendrux/llm/openai_responses.py:[1:13]
==src/dendrux/llm/openai.py:[1:13]
           y = 2

TOTAL lines=100 duplicates=32 percent=0.32
"""


def test_parse_collects_blocks_with_a_sorted_module_pair_and_size() -> None:
    blocks = dup.parse_symilar(SYMILAR)
    assert blocks == [
        dup.Block("dendrux.llm.openai", "dendrux.llm.openai_responses", 20),
        dup.Block("dendrux.llm.openai", "dendrux.llm.openai_responses", 12),
    ]


def test_baseline_groups_sizes_by_pair_largest_first() -> None:
    base = dup.baseline_from_blocks(dup.parse_symilar(SYMILAR))
    assert base == {"dendrux.llm.openai|dendrux.llm.openai_responses": [20, 12]}


def test_the_same_blocks_pass() -> None:
    blocks = dup.parse_symilar(SYMILAR)
    assert not dup.compare(blocks, dup.baseline_from_blocks(blocks)).failed


def test_a_block_in_a_new_pair_fails() -> None:
    base = dup.baseline_from_blocks(dup.parse_symilar(SYMILAR))
    blocks = dup.parse_symilar(SYMILAR) + [dup.Block("dendrux.a", "dendrux.b", 10)]
    result = dup.compare(blocks, base)
    assert result.new == ["dendrux.a|dendrux.b (10 lines)"]


def test_an_extra_block_in_a_known_pair_fails() -> None:
    blocks = dup.parse_symilar(SYMILAR)
    base = dup.baseline_from_blocks(blocks)
    blocks.append(dup.Block("dendrux.llm.openai", "dendrux.llm.openai_responses", 11))
    assert dup.compare(blocks, base).new


def test_a_block_that_grows_past_the_tolerance_fails() -> None:
    base = {"dendrux.a|dendrux.b": [20]}
    assert dup.compare([dup.Block("dendrux.a", "dendrux.b", 22)], base).failed is False
    assert dup.compare([dup.Block("dendrux.a", "dendrux.b", 40)], base).failed is True


def test_a_block_that_shrinks_or_disappears_is_an_improvement() -> None:
    base = {"dendrux.a|dendrux.b": [20, 12]}
    result = dup.compare([dup.Block("dendrux.a", "dendrux.b", 12)], base)
    assert not result.failed
    assert result.improved


@pytest.mark.skipif(shutil.which("symilar") is None, reason="pylint not installed")
def test_end_to_end_finds_a_real_copy_paste(tmp_path: Path) -> None:
    src = tmp_path / "src" / "dendrux"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("")
    body = "".join(f"    total_{i} = value + {i}\n    print(total_{i}, '{i}')\n" for i in range(8))
    (src / "one.py").write_text(f"def one(value):\n{body}    return value\n")
    (src / "two.py").write_text(f"def two(value, extra):\n{body}    return extra\n")
    blocks = dup.collect_blocks(src)
    assert blocks and blocks[0].pair == "dendrux.one|dendrux.two"
    baseline = tmp_path / "base.json"
    assert dup.main(["check", "--src", str(src), "--baseline", str(baseline)]) == 2
    assert dup.main(["stamp", "--src", str(src), "--baseline", str(baseline)]) == 0
    assert dup.main(["check", "--src", str(src), "--baseline", str(baseline)]) == 0
    (src / "three.py").write_text(f"def three(value, more):\n{body}    return more\n")
    assert dup.main(["check", "--src", str(src), "--baseline", str(baseline)]) == 1
