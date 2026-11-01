"""tools/duplicate_ratchet.py: a new duplicated block fails; known blocks may not grow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tools import duplicate_ratchet as dup

if TYPE_CHECKING:
    from pathlib import Path

PAIR = "dendrux.llm.openai|dendrux.llm.openai_responses"
OPENAI = dup.Block("dendrux.llm.openai", "dendrux.llm.openai_responses", 20)
OPENAI_SMALL = dup.Block("dendrux.llm.openai", "dendrux.llm.openai_responses", 12)


def test_blocks_are_a_sorted_module_pair_and_size() -> None:
    assert dup.Block("dendrux.b", "dendrux.a", 11).pair == "dendrux.a|dendrux.b"


def test_baseline_groups_sizes_by_pair_largest_first() -> None:
    assert dup.baseline_from_blocks([OPENAI_SMALL, OPENAI]) == {PAIR: [20, 12]}


def test_the_same_blocks_pass() -> None:
    blocks = [OPENAI, OPENAI_SMALL]
    assert not dup.compare(blocks, dup.baseline_from_blocks(blocks)).failed


def test_a_block_in_a_new_pair_fails() -> None:
    base = dup.baseline_from_blocks([OPENAI, OPENAI_SMALL])
    blocks = [OPENAI, OPENAI_SMALL] + [dup.Block("dendrux.a", "dendrux.b", 10)]
    result = dup.compare(blocks, base)
    assert result.new == ["dendrux.a|dendrux.b (10 lines)"]


def test_an_extra_block_in_a_known_pair_fails() -> None:
    blocks = [OPENAI, OPENAI_SMALL]
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


def test_a_stamp_keeps_the_old_ceiling_for_a_block_that_grew_within_the_tolerance() -> None:
    previous = {"dendrux.a|dendrux.b": [20]}
    grown = [dup.Block("dendrux.a", "dendrux.b", 22)]
    assert dup.baseline_from_blocks(grown, previous) == {"dendrux.a|dendrux.b": [20]}


def test_a_stamp_lowers_the_ceiling_when_a_block_shrank() -> None:
    previous = {"dendrux.a|dendrux.b": [20, 12]}
    shrunk = [dup.Block("dendrux.a", "dendrux.b", 18)]
    assert dup.baseline_from_blocks(shrunk, previous) == {"dendrux.a|dendrux.b": [18]}


def test_a_stamp_never_adds_a_block_it_was_not_given_a_ceiling_for() -> None:
    previous = {"dendrux.a|dendrux.b": [20]}
    blocks = [dup.Block("dendrux.a", "dendrux.b", 20), dup.Block("dendrux.a", "dendrux.c", 12)]
    # an unmatched block is recorded only when there is nothing to match against
    assert dup.baseline_from_blocks(blocks, previous) == {"dendrux.a|dendrux.b": [20]}


def _write_copies(src: Path, names: tuple[str, ...]) -> None:
    body = "".join(f"    total_{i} = value + {i}\n    print(total_{i}, '{i}')\n" for i in range(8))
    for name in names:
        (src / f"{name}.py").write_text(f"def {name}(value):\n{body}    return value\n")


def test_end_to_end_a_third_identical_copy_fails(tmp_path: Path) -> None:
    src = tmp_path / "src" / "dendrux"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("")
    _write_copies(src, ("a", "b"))
    baseline = tmp_path / "base.json"
    args = ["--src", str(src), "--baseline", str(baseline)]
    assert dup.main(["stamp", *args]) == 0
    assert dup.main(["check", *args]) == 0
    _write_copies(src, ("c",))
    assert dup.collect_blocks(src)  # all three pairs are seen
    assert {b.pair for b in dup.collect_blocks(src)} == {
        "dendrux.a|dendrux.b",
        "dendrux.a|dendrux.c",
        "dendrux.b|dendrux.c",
    }
    assert dup.main(["check", *args]) == 1
    assert dup.main(["stamp", *args]) == 1


def test_end_to_end_stamp_does_not_raise_a_ceiling(tmp_path: Path) -> None:
    src = tmp_path / "src" / "dendrux"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("")
    _write_copies(src, ("a", "b"))
    baseline = tmp_path / "base.json"
    args = ["--src", str(src), "--baseline", str(baseline)]
    assert dup.main(["stamp", *args]) == 0
    before = baseline.read_text()
    # b grows by two lines that a also has: the common block gets longer
    extra = "    extra_one = value\n    print(extra_one)\n"
    for name in ("a", "b"):
        text = (src / f"{name}.py").read_text()
        (src / f"{name}.py").write_text(
            text.replace("    return value", extra + "    return value")
        )
    assert dup.main(["check", *args]) == 0  # within the tolerance
    assert dup.main(["stamp", *args]) == 0
    assert baseline.read_text() == before


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


def test_a_stamp_over_the_same_blocks_changes_nothing() -> None:
    sizes = [20, 16, 14, 12]
    blocks = [dup.Block("dendrux.a", "dendrux.b", n) for n in sizes]
    previous = {"dendrux.a|dendrux.b": sizes}
    assert dup.baseline_from_blocks(blocks, previous) == previous
