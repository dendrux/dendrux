"""Physical size limits and shrink-only size baselines."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


@pytest.fixture
def size() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "tools" / "size_ratchet.py"
    assert path.is_file(), "R0 size ratchet CLI has not been implemented"
    return importlib.import_module("tools.size_ratchet")


def _source(tmp_path: Path) -> Path:
    src = tmp_path / "src" / "dendrux"
    src.mkdir(parents=True)
    return src


def test_physical_spans_include_decorators_comments_blanks_and_nested_functions(
    size: ModuleType, tmp_path: Path
) -> None:
    src = _source(tmp_path)
    (src / "a.py").write_text(
        "class Box:\n"
        "    @decorator(\n        1,\n    )\n"
        "    async def method(self):\n"
        "        # a physical line\n\n"
        "        def inner():\n            return 1\n"
        "        return inner()\n"
    )
    assert size.collect_scores(src) == {
        "dendrux.a": 10,
        "dendrux.a:Box.method": 9,
        "dendrux.a:Box.method.<locals>.inner": 2,
    }


def test_overloads_excluded_including_aliases_but_other_stubs_count(
    size: ModuleType, tmp_path: Path
) -> None:
    src = _source(tmp_path)
    (src / "a.py").write_text(
        "import typing as t\nfrom typing import overload as ov\n"
        "from typing_extensions import overload as ext\n"
        "@t.overload\ndef f(x: int): ...\n"
        "@ov\ndef f(x: str): ...\n"
        "def f(x):\n    return x\n"
        "@ext\ndef extra(x): ...\n"
        "def stub(): ...\n"
    )
    scores = size.collect_scores(src)
    assert scores["dendrux.a:f"] == 2
    assert "dendrux.a:extra" not in scores
    assert scores["dendrux.a:stub"] == 1


def test_same_qualname_keeps_longest_implementation(size: ModuleType, tmp_path: Path) -> None:
    src = _source(tmp_path)
    (src / "a.py").write_text(
        "class Box:\n    @property\n    def x(self):\n        return 1\n"
        "    @x.setter\n    def x(self, value):\n        self.value = value\n        return\n"
    )
    assert size.collect_scores(src)["dendrux.a:Box.x"] == 4


def test_physical_lines_do_not_split_unicode_string_content(
    size: ModuleType, tmp_path: Path
) -> None:
    src = _source(tmp_path)
    (src / "a.py").write_text('VALUE = "one\u2028two"\n\n')
    assert size.collect_scores(src)["dendrux.a"] == 2


def test_parenthesized_decorator_counts_from_at_sign(size: ModuleType, tmp_path: Path) -> None:
    src = _source(tmp_path)
    (src / "a.py").write_text("@(\n    decorator\n)\ndef f():\n    pass\n")
    assert size.collect_scores(src)["dendrux.a:f"] == 5


def test_stacked_decorators_start_at_first_decorator(size: ModuleType, tmp_path: Path) -> None:
    src = _source(tmp_path)
    (src / "a.py").write_text("@first\n@second(\n    value,\n)\ndef f():\n    pass\n")
    assert size.collect_scores(src)["dendrux.a:f"] == 6


@pytest.mark.parametrize("length", [1000, 1001])
@pytest.mark.parametrize("trailing_newline", [False, True])
def test_physical_file_boundary_with_or_without_final_newline(
    size: ModuleType, tmp_path: Path, length: int, trailing_newline: bool
) -> None:
    src = _source(tmp_path)
    source = "\n".join(["# physical line"] * length) + ("\n" if trailing_newline else "")
    (src / "a.py").write_text(source)
    scores = size.collect_scores(src)
    assert scores["dendrux.a"] == length
    assert size.compare(scores, {}).failed is (length > 1000)


def test_same_name_max_keeps_a_longer_first_definition(size: ModuleType, tmp_path: Path) -> None:
    src = _source(tmp_path)
    (src / "a.py").write_text(
        "class Box:\n    @property\n    def x(self):\n        value = 1\n        return value\n"
        "    @x.setter\n    def x(self, value):\n        self.value = value\n"
    )
    assert size.collect_scores(src)["dendrux.a:Box.x"] == 4


def test_scope_excludes_migrations_and_only_exact_data_modules(
    size: ModuleType, tmp_path: Path
) -> None:
    src = _source(tmp_path)
    for name in ("_codes.py", "types.py", "db/migrations/x.py", "testing/types.py"):
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def f():\n" + "    pass\n" * 1001)
    scores = size.collect_scores(src)
    assert "dendrux._codes" not in scores
    assert "dendrux.types" not in scores
    assert not any("migrations" in key for key in scores)
    assert scores["dendrux.testing.types"] == 1002


@pytest.mark.parametrize(
    ("scores", "baseline", "new", "worse"),
    [
        ({"dendrux.a": 1000, "dendrux.a:f": 80}, {}, [], []),
        ({"dendrux.a": 1001}, {}, ["dendrux.a"], []),
        ({"dendrux.a:f": 81}, {}, ["dendrux.a:f"], []),
        ({"dendrux.a": 1201}, {"dendrux.a": 1200}, [], ["dendrux.a"]),
        ({"dendrux.a:f": 91}, {"dendrux.a:f": 90}, [], ["dendrux.a:f"]),
        ({"dendrux.a:f": 85}, {"dendrux.a:f": 90}, [], []),
        ({}, {"dendrux.a:f": 90}, [], []),
        ({"dendrux.renamed": 1201}, {"dendrux.a": 1200}, ["dendrux.renamed"], []),
        ({"dendrux.a:renamed": 91}, {"dendrux.a:f": 90}, ["dendrux.a:renamed"], []),
    ],
)
def test_size_comparison(
    size: ModuleType, scores: dict, baseline: dict, new: list, worse: list
) -> None:
    result = size.compare(scores, baseline)
    assert result.new == new
    assert result.worse == worse
    assert result.failed == bool(new or worse)


def test_baseline_records_only_measured_offenders(size: ModuleType) -> None:
    assert size.baseline_from_scores(
        {"dendrux.a": 1001, "dendrux.a:f": 81, "dendrux.b": 1000, "dendrux.b:g": 80}
    ) == {"dendrux.a": 1001, "dendrux.a:f": 81}


def test_cli_check_stamp_report_and_refusal_preserves_baseline(
    size: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = _source(tmp_path)
    path = src / "a.py"
    path.write_text("def f():\n" + "    pass\n" * 81)
    baseline = tmp_path / "size.json"
    args = ["--src", str(src), "--baseline", str(baseline)]
    assert size.main(["check", *args]) == 2
    assert size.main(["report", *args]) == 0
    assert not baseline.exists()
    assert "dendrux.a:f" in capsys.readouterr().out
    assert size.main(["stamp", *args]) == 0
    assert json.loads(baseline.read_text()) == {"dendrux.a:f": 82}
    assert size.main(["check", *args]) == 0
    before = baseline.read_bytes()
    path.write_text(path.read_text() + "    pass\n")
    assert size.main(["check", *args]) == 1
    assert size.main(["stamp", *args]) == 1
    assert baseline.read_bytes() == before
    path.write_text("def f():\n" + "    pass\n" * 80)
    assert size.main(["stamp", *args]) == 0
    assert json.loads(baseline.read_text()) == {"dendrux.a:f": 81}
    path.rename(src / "renamed.py")
    assert size.main(["check", *args]) == 1
    assert size.main(["stamp", *args]) == 1
    path = src / "renamed.py"
    path.write_text("def renamed():\n    pass\n")
    assert size.main(["stamp", *args]) == 0
    assert json.loads(baseline.read_text()) == {}


def test_cli_has_no_regression_override(size: ModuleType) -> None:
    with pytest.raises(SystemExit) as exc:
        size.main(["stamp", "--allow-regress", "--reason", "no increases"])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("sources", "baseline", "pairs", "expected", "reason"),
    [
        (
            {"new.py": 81},
            {"dendrux.old:f": 81},
            ["dendrux.old:f", "dendrux.new:f"],
            {"dendrux.new:f": 81},
            "equal function move",
        ),
        (
            {"new.py": 81},
            {"dendrux.old:f": 90, "dendrux.other:f": 91},
            ["dendrux.old:f", "dendrux.new:f"],
            {"dendrux.new:f": 81, "dendrux.other:f": 91},
            "shrink and preserve unrelated baseline",
        ),
        (
            {"new.py": 1001},
            {"dendrux.old": 1001},
            ["dendrux.old", "dendrux.new"],
            {"dendrux.new": 1001},
            "equal file move",
        ),
        (
            {"new.py": 1001, "second.py": 1002},
            {"dendrux.old": 2003},
            ["dendrux.old", "dendrux.new", "dendrux.old", "dendrux.second"],
            {"dendrux.new": 1001, "dendrux.second": 1002},
            "file split at total ceiling",
        ),
        (
            {"new.py": 1001, "second.py": 1002},
            {"dendrux.old": 2100},
            ["dendrux.old", "dendrux.new", "dendrux.old", "dendrux.second"],
            {"dendrux.new": 1001, "dendrux.second": 1002},
            "file split lowers total",
        ),
        (
            {"old.py": 2, "new.py": 81},
            {"dendrux.old:f": 90},
            ["dendrux.old:f", "dendrux.new:f"],
            None,
            "source function still exists below limit",
        ),
        (
            {"old.py": 2, "new.py": 1001},
            {"dendrux.old": 1100},
            ["dendrux.old", "dendrux.new"],
            {"dendrux.old": 2, "dendrux.new": 1001},
            "retained source below limit",
        ),
        ({"new.py": 81}, {}, ["dendrux.old:f", "dendrux.new:f"], None, "no source allowance"),
        (
            {"new.py": 81},
            {"dendrux.old:f": 80},
            ["dendrux.old:f", "dendrux.new:f"],
            None,
            "source allowance is not an offender",
        ),
        (
            {"new.py": 80},
            {"dendrux.old:f": 90},
            ["dendrux.old:f", "dendrux.new:f"],
            None,
            "destination function at limit",
        ),
        (
            {"new.py": 1000},
            {"dendrux.old": 1100},
            ["dendrux.old", "dendrux.new"],
            None,
            "destination file at limit",
        ),
        (
            {},
            {"dendrux.old:f": 90},
            ["dendrux.old:f", "dendrux.missing:f"],
            None,
            "destination absent",
        ),
        (
            {"new.py": 91},
            {"dendrux.old:f": 90},
            ["dendrux.old:f", "dendrux.new:f"],
            None,
            "grown function",
        ),
        (
            {"new.py": 1101},
            {"dendrux.old": 1100},
            ["dendrux.old", "dendrux.new"],
            None,
            "grown file",
        ),
        (
            {"new.py": 1001, "second.py": 1002},
            {"dendrux.old": 2002},
            ["dendrux.old", "dendrux.new", "dendrux.old", "dendrux.second"],
            None,
            "file split exceeds total allowance",
        ),
        (
            {"new.py": 81, "second.py": 81},
            {"dendrux.old:f": 200},
            ["dendrux.old:f", "dendrux.new:f", "dendrux.old:f", "dendrux.second:f"],
            None,
            "function allowances cannot split",
        ),
        (
            {"new.py": 1001},
            {"dendrux.old:f": 2000},
            ["dendrux.old:f", "dendrux.new"],
            None,
            "function to file",
        ),
        (
            {"new.py": 81},
            {"dendrux.old": 1001},
            ["dendrux.old", "dendrux.new:f"],
            None,
            "file to function",
        ),
        (
            {"new.py": 81},
            {"dendrux.old:f": 90, "dendrux.new:f": 85},
            ["dendrux.old:f", "dendrux.new:f"],
            None,
            "destination already baselined",
        ),
        (
            {"new.py": 81},
            {"dendrux.old:f": 90},
            ["dendrux.old:f", "dendrux.new:f", "dendrux.old:f", "dendrux.new:f"],
            None,
            "duplicate destination",
        ),
        (
            {"new.py": 81},
            {"dendrux.old:f": 90, "dendrux.second:f": 90},
            ["dendrux.old:f", "dendrux.new:f", "dendrux.second:f", "dendrux.new:f"],
            None,
            "two sources cannot claim one destination",
        ),
        (
            {"new.py": 81},
            {"dendrux.old:f": 90},
            ["dendrux.old:f", "dendrux.new:f", "dendrux.missing:f", "dendrux.nope:f"],
            None,
            "later invalid pair makes batch atomic",
        ),
    ],
)
def test_move_allowances(
    size: ModuleType,
    tmp_path: Path,
    sources: dict[str, int],
    baseline: dict[str, int],
    pairs: list[str],
    expected: dict[str, int] | None,
    reason: str,
) -> None:
    src = _source(tmp_path)
    for name, length in sources.items():
        # Comment-only fixtures isolate file allowances from function allowances.
        source = (
            "def f():\n" + "    pass\n" * (length - 1)
            if length <= 200
            else "# physical line\n" * length
        )
        (src / name).write_text(source)
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline) + "\n")
    before = path.read_bytes()
    args = ["--src", str(src), "--baseline", str(path)]
    assert size.main(["move", *pairs, *args]) == (1 if expected is None else 0), reason
    if expected is None:
        assert path.read_bytes() == before, reason
    else:
        assert json.loads(path.read_text()) == expected, reason
        assert size.main(["check", *args]) == 0


def test_move_rejects_source_function_replaced_by_overload(
    size: ModuleType, tmp_path: Path
) -> None:
    src = _source(tmp_path)
    (src / "old.py").write_text("from typing import overload\n@overload\ndef f(): ...\n")
    (src / "new.py").write_text("def f():\n" + "    pass\n" * 80)
    path = tmp_path / "baseline.json"
    path.write_text('{"dendrux.old:f": 90}\n')
    before = path.read_bytes()
    assert (
        size.main(
            ["move", "dendrux.old:f", "dendrux.new:f", "--src", str(src), "--baseline", str(path)]
        )
        == 1
    )
    assert path.read_bytes() == before


@pytest.mark.parametrize("pairs", [[], ["dendrux.old"], ["dendrux.old", "dendrux.new", "extra"]])
def test_move_requires_complete_nonempty_pairs(
    size: ModuleType, tmp_path: Path, pairs: list[str]
) -> None:
    path = tmp_path / "baseline.json"
    path.write_text("{}\n")
    with pytest.raises(SystemExit) as exc:
        size.main(["move", *pairs, "--baseline", str(path)])
    assert exc.value.code == 2
    assert path.read_text() == "{}\n"


def test_move_requires_existing_baseline(size: ModuleType, tmp_path: Path) -> None:
    src = _source(tmp_path)
    path = tmp_path / "missing.json"
    assert (
        size.main(
            ["move", "dendrux.old", "dendrux.new", "--src", str(src), "--baseline", str(path)]
        )
        == 2
    )
    assert not path.exists()


@pytest.mark.parametrize(
    ("old_size", "allowance", "destinations", "success"),
    [
        (1200, 2201, [1001], True),
        (1200, 3300, [1001, 1002], True),
        (1200, 3203, [1001, 1002], True),
        (1200, 2200, [1001], False),
        (1200, 1200, [1001], False),
        (1201, 1200, [1001], False),
        (1200, 3202, [1001, 1002], False),
    ],
)
def test_split_keeps_shrinking_source_file(
    size: ModuleType,
    tmp_path: Path,
    old_size: int,
    allowance: int,
    destinations: list[int],
    success: bool,
) -> None:
    src = _source(tmp_path)
    (src / "old.py").write_text("# retained line\n" * old_size)
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"dendrux.old": allowance}) + "\n")
    before = path.read_bytes()
    pairs = []
    expected = {"dendrux.old": old_size}
    for i, value in enumerate(destinations):
        key = f"dendrux.new_{i}"
        (src / f"new_{i}.py").write_text("# moved line\n" * value)
        pairs.extend(["dendrux.old", key])
        expected[key] = value
    args = ["--src", str(src), "--baseline", str(path)]
    assert size.main(["move", *pairs, *args]) == (0 if success else 1)
    if success:
        assert json.loads(path.read_text()) == expected
        assert size.main(["check", *args]) == 0
        # The retained file gets the measured ceiling, not the former allowance.
        with (src / "old.py").open("a") as out:
            out.write("# extra line\n")
        assert size.main(["check", *args]) == 1
    else:
        assert path.read_bytes() == before
