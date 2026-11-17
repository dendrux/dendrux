"""Per-file, per-rule counts use the project Ruff configuration."""

from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


@pytest.fixture
def ruff() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "tools" / "ruff_ratchet.py"
    assert path.is_file(), "R0 Ruff ratchet CLI has not been implemented"
    return importlib.import_module("tools.ruff_ratchet")


def _tree(tmp_path: Path) -> Path:
    src = tmp_path / "src" / "dendrux"
    src.mkdir(parents=True)
    return src


def test_report_filters_scope_rules_and_groups_counts(ruff: ModuleType, tmp_path: Path) -> None:
    src = _tree(tmp_path)
    report = [
        {"filename": str(src / name), "code": code}
        for name, code in [
            ("a.py", "PLC0415"),
            ("a.py", "PLC0415"),
            ("a.py", "BLE001"),
            ("pkg/__init__.py", "PLR0911"),
            ("testing/a.py", "PLR0912"),
            ("db/migrations/a.py", "PLC0415"),
            ("a.py", "F401"),
            ("../outside.py", "BLE001"),
        ]
    ]
    assert ruff.scores_from_report(report, src) == {
        "dendrux.a:PLC0415": 2,
        "dendrux.a:BLE001": 1,
        "dendrux.pkg:PLR0911": 1,
        "dendrux.testing.a:PLR0912": 1,
    }


@pytest.mark.parametrize(
    ("current", "baseline", "failed"),
    [
        ({"dendrux.a:PLC0415": 2}, {"dendrux.a:PLC0415": 2}, False),
        ({"dendrux.a:PLC0415": 3}, {"dendrux.a:PLC0415": 2}, True),
        ({"dendrux.a:PLC0415": 1}, {"dendrux.a:PLC0415": 2}, False),
        ({}, {"dendrux.a:PLC0415": 2}, False),
        ({"dendrux.a:BLE001": 1}, {"dendrux.a:PLC0415": 2}, True),
        ({"dendrux.b:PLC0415": 1}, {"dendrux.a:PLC0415": 2}, True),
    ],
)
def test_comparison_is_per_file_and_rule(
    ruff: ModuleType, current: dict, baseline: dict, failed: bool
) -> None:
    assert ruff.compare(current, baseline).failed is failed


def test_subprocess_extends_project_rules_and_tool_errors_fail_closed(
    ruff: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _tree(tmp_path)
    seen = []

    def run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 2, "[]", "invalid Ruff configuration")

    monkeypatch.setattr(ruff.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        ruff.collect_scores(src)
    cmd, kwargs = seen[0]
    assert "--extend-select" in cmd
    assert set(cmd[cmd.index("--extend-select") + 1].split(",")) == set(ruff.RULES)
    assert "--select" not in cmd
    assert Path(cmd[cmd.index("--config") + 1]).name == "pyproject.toml"
    assert "--force-exclude" in cmd
    assert kwargs.get("shell") is not True


def test_real_ruff_check_stamp_report_new_rules_and_shrink(
    ruff: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = _tree(tmp_path)
    path = src / "a.py"
    path.write_text("def f():\n    import os\n    return os.name\n")
    assert ruff.collect_scores(src) == {"dendrux.a:PLC0415": 1}
    baseline = tmp_path / "ruff.json"
    args = ["--src", str(src), "--baseline", str(baseline)]
    assert ruff.main(["check", *args]) == 2
    assert ruff.main(["report", *args]) == 0
    assert not baseline.exists()
    assert "PLC0415: 1" in capsys.readouterr().out
    assert ruff.main(["stamp", *args]) == 0
    assert ruff.main(["check", *args]) == 0
    before = baseline.read_bytes()
    path.write_text(
        "def f():\n    import os\n    try:\n        return os.name\n"
        "    except Exception:\n        return None\n"
    )
    assert ruff.main(["check", *args]) == 1
    assert ruff.main(["stamp", *args]) == 1
    assert baseline.read_bytes() == before
    path.write_text("def f():\n    return 1\n")
    assert ruff.main(["stamp", *args]) == 0
    assert json.loads(baseline.read_text()) == {}


def test_real_ruff_honors_noqa_and_excludes_migrations(ruff: ModuleType, tmp_path: Path) -> None:
    src = _tree(tmp_path)
    (src / "a.py").write_text("def f():\n    import os  # noqa: PLC0415\n    return os.name\n")
    migration = src / "db" / "migrations" / "x.py"
    migration.parent.mkdir(parents=True)
    migration.write_text("def f():\n    import os\n    return os.name\n")
    assert ruff.collect_scores(src) == {}


def test_cli_has_no_regression_override(ruff: ModuleType) -> None:
    with pytest.raises(SystemExit) as exc:
        ruff.main(["stamp", "--allow-regress", "--reason", "no increases"])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("sources", "baseline", "pairs", "expected", "reason"),
    [
        (
            {"new": 2},
            {"dendrux.old:PLC0415": 2},
            ["dendrux.old:PLC0415", "dendrux.new:PLC0415"],
            {"dendrux.new:PLC0415": 2},
            "equal file/rule move",
        ),
        (
            {"new": 1},
            {"dendrux.old:PLC0415": 2, "dendrux.other:BLE001": 3},
            ["dendrux.old:PLC0415", "dendrux.new:PLC0415"],
            {"dendrux.new:PLC0415": 1, "dendrux.other:BLE001": 3},
            "lower and preserve unrelated",
        ),
        (
            {"new": 1, "second": 2},
            {"dendrux.old:PLC0415": 3},
            [
                "dendrux.old:PLC0415",
                "dendrux.new:PLC0415",
                "dendrux.old:PLC0415",
                "dendrux.second:PLC0415",
            ],
            {"dendrux.new:PLC0415": 1, "dendrux.second:PLC0415": 2},
            "file split at ceiling",
        ),
        (
            {"new": 1, "second": 2},
            {"dendrux.old:PLC0415": 4},
            [
                "dendrux.old:PLC0415",
                "dendrux.new:PLC0415",
                "dendrux.old:PLC0415",
                "dendrux.second:PLC0415",
            ],
            {"dendrux.new:PLC0415": 1, "dendrux.second:PLC0415": 2},
            "file split lowers total",
        ),
        (
            {"old": 0, "new": 1},
            {"dendrux.old:PLC0415": 2},
            ["dendrux.old:PLC0415", "dendrux.new:PLC0415"],
            {"dendrux.old:PLC0415": 0, "dendrux.new:PLC0415": 1},
            "retained source with no findings",
        ),
        (
            {"new": 1},
            {},
            ["dendrux.old:PLC0415", "dendrux.new:PLC0415"],
            None,
            "source not baselined",
        ),
        (
            {"new": 1},
            {"dendrux.old:PLC0415": 0},
            ["dendrux.old:PLC0415", "dendrux.new:PLC0415"],
            None,
            "nonpositive allowance",
        ),
        (
            {"new": 0},
            {"dendrux.old:PLC0415": 2},
            ["dendrux.old:PLC0415", "dendrux.new:PLC0415"],
            None,
            "destination not offender",
        ),
        (
            {},
            {"dendrux.old:PLC0415": 2},
            ["dendrux.old:PLC0415", "dendrux.new:PLC0415"],
            None,
            "destination missing",
        ),
        (
            {"new": 3},
            {"dendrux.old:PLC0415": 2},
            ["dendrux.old:PLC0415", "dendrux.new:PLC0415"],
            None,
            "count grew",
        ),
        (
            {"new": 1, "second": 2},
            {"dendrux.old:PLC0415": 2},
            [
                "dendrux.old:PLC0415",
                "dendrux.new:PLC0415",
                "dendrux.old:PLC0415",
                "dendrux.second:PLC0415",
            ],
            None,
            "split sum grew",
        ),
        (
            {"new": 1},
            {"dendrux.old:BLE001": 2},
            ["dendrux.old:BLE001", "dendrux.new:PLC0415"],
            None,
            "cross-rule transfer",
        ),
        (
            {"new": 1},
            {"dendrux.old:UNKNOWN": 2},
            ["dendrux.old:UNKNOWN", "dendrux.new:PLC0415"],
            None,
            "unknown rule",
        ),
        (
            {"new": 1},
            {"dendrux.old:PLC0415": 2, "dendrux.new:PLC0415": 1},
            ["dendrux.old:PLC0415", "dendrux.new:PLC0415"],
            None,
            "destination already baselined",
        ),
        (
            {"new": 1},
            {"dendrux.old:PLC0415": 2},
            [
                "dendrux.old:PLC0415",
                "dendrux.new:PLC0415",
                "dendrux.old:PLC0415",
                "dendrux.new:PLC0415",
            ],
            None,
            "duplicate destination",
        ),
        (
            {"new": 1},
            {"dendrux.old:PLC0415": 2},
            [
                "dendrux.old:PLC0415",
                "dendrux.new:PLC0415",
                "dendrux.missing:PLC0415",
                "dendrux.nope:PLC0415",
            ],
            None,
            "invalid second pair",
        ),
    ],
)
def test_move_file_rule_allowances(
    ruff: ModuleType,
    tmp_path: Path,
    sources: dict[str, int],
    baseline: dict[str, int],
    pairs: list[str],
    expected: dict[str, int] | None,
    reason: str,
) -> None:
    src = _tree(tmp_path)
    for module, count in sources.items():
        (src / f"{module}.py").write_text("def f():\n" + "    import os\n" * count + "    pass\n")
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline) + "\n")
    before = path.read_bytes()
    args = ["--src", str(src), "--baseline", str(path)]
    assert ruff.main(["move", *pairs, *args]) == (1 if expected is None else 0), reason
    if expected is None:
        assert path.read_bytes() == before, reason
    else:
        assert json.loads(path.read_text()) == expected, reason
        assert ruff.main(["check", *args]) == 0


@pytest.mark.parametrize("pairs", [[], ["dendrux.old"], ["dendrux.old", "dendrux.new", "extra"]])
def test_move_requires_complete_nonempty_pairs(
    ruff: ModuleType, tmp_path: Path, pairs: list[str]
) -> None:
    path = tmp_path / "baseline.json"
    path.write_text("{}\n")
    with pytest.raises(SystemExit) as exc:
        ruff.main(["move", *pairs, "--baseline", str(path)])
    assert exc.value.code == 2
    assert path.read_text() == "{}\n"


def test_move_requires_existing_baseline(ruff: ModuleType, tmp_path: Path) -> None:
    src = _tree(tmp_path)
    path = tmp_path / "missing.json"
    assert (
        ruff.main(
            [
                "move",
                "dendrux.old:PLC0415",
                "dendrux.new:PLC0415",
                "--src",
                str(src),
                "--baseline",
                str(path),
            ]
        )
        == 2
    )
    assert not path.exists()


@pytest.mark.parametrize(
    ("old_count", "allowance", "destinations", "success"),
    [
        (1, 3, [2], True),
        (1, 5, [1, 2], True),
        (1, 4, [1, 2], True),
        (1, 2, [2], False),
        (2, 2, [1], False),
        (3, 2, [1], False),
        (1, 3, [1, 2], False),
    ],
)
def test_split_keeps_shrinking_source_counts(
    ruff: ModuleType,
    tmp_path: Path,
    old_count: int,
    allowance: int,
    destinations: list[int],
    success: bool,
) -> None:
    src = _tree(tmp_path)
    (src / "old.py").write_text("def f():\n" + "    import os\n" * old_count + "    pass\n")
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"dendrux.old:PLC0415": allowance}) + "\n")
    before = path.read_bytes()
    pairs = []
    expected = {"dendrux.old:PLC0415": old_count}
    for i, count in enumerate(destinations):
        key = f"dendrux.new_{i}:PLC0415"
        (src / f"new_{i}.py").write_text("def f():\n" + "    import os\n" * count + "    pass\n")
        pairs.extend(["dendrux.old:PLC0415", key])
        expected[key] = count
    args = ["--src", str(src), "--baseline", str(path)]
    assert ruff.main(["move", *pairs, *args]) == (0 if success else 1)
    if success:
        assert json.loads(path.read_text()) == expected
        assert ruff.main(["check", *args]) == 0
    else:
        assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("source", "bad_codes"),
    [
        ("    import os  # noqa: PLC0415\n", ["PLC0415"]),
        ("    import os  # noqa: PLC0415 - optional extra\n", []),
        ("    import os  # noqa: PLC0415 — optional extra\n", []),
        ("    import os  # noqa: PLC0415 - 可选依赖\n", []),
        ("    # optional extra stays lazy\n    import os  # noqa: PLC0415\n", []),
        ("    #\n    import os  # noqa: PLC0415\n", ["PLC0415"]),
        ("    # PLC0415\n    import os  # noqa: PLC0415\n", ["PLC0415"]),
        ("    # optional extra\n\n    import os  # noqa: PLC0415\n", ["PLC0415"]),
        ("    import os  # noqa: PLC0415, BLE001\n", ["BLE001", "PLC0415"]),
        ("    import os  # noqa: PLC0415, BLE001 - optional isolation\n", []),
        ("    pass  # noqa: F401\n", []),
        ("    value = '# noqa: BLE001'\n", []),
        ("    import os  # noqa\n", sorted(["PLC0415", "PLR0911", "PLR0912", "PLR0915", "BLE001"])),
        ("    # noqa: BLE001\n    import os  # noqa: PLC0415\n", ["BLE001", "PLC0415"]),
        ("    import os  # noqa: PLC0415 - ---\n", ["PLC0415"]),
    ],
)
def test_noqa_requires_inline_or_immediately_preceding_reason(
    ruff: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    source: str,
    bad_codes: list[str],
) -> None:
    src = _tree(tmp_path)
    (src / "a.py").write_text("def f():\n" + source + "    pass\n")
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}\n")
    args = ["--src", str(src), "--baseline", str(baseline)]
    assert ruff.main(["check", *args]) == (1 if bad_codes else 0)
    output = capsys.readouterr().out
    for code in bad_codes:
        assert code in output
        assert f"{src / 'a.py'}:" in output
        assert "noqa lacks reason" in output


def test_noqa_reason_debt_is_separate_shrink_only_and_has_locations(
    ruff: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = _tree(tmp_path)
    path = src / "a.py"
    path.write_text("def f():\n    import os  # noqa: PLC0415\n    pass\n")
    baseline = tmp_path / "baseline.json"
    args = ["--src", str(src), "--baseline", str(baseline)]
    assert ruff.main(["stamp", *args]) == 0
    assert list(json.loads(baseline.read_text()).values()) == [1]
    assert ruff.main(["check", *args]) == 0
    capsys.readouterr()
    with path.open("a") as out:
        out.write("    import sys  # noqa: PLC0415\n")
    before = baseline.read_bytes()
    assert ruff.main(["check", *args]) == 1
    assert f"{path}:4:" in capsys.readouterr().out
    assert ruff.main(["stamp", *args]) == 1
    assert baseline.read_bytes() == before
    path.write_text("def f():\n    import os  # noqa: PLC0415 - optional extra\n    pass\n")
    assert ruff.main(["stamp", *args]) == 0
    assert json.loads(baseline.read_text()) == {}


def test_file_level_noqa_reasons_and_migration_scope(ruff: ModuleType, tmp_path: Path) -> None:
    src = _tree(tmp_path)
    path = src / "a.py"
    path.write_text("# ruff: noqa: PLC0415\ndef f():\n    import os\n    pass\n")
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}\n")
    args = ["--src", str(src), "--baseline", str(baseline)]
    assert ruff.main(["check", *args]) == 1
    path.write_text("# optional extras stay lazy\n" + path.read_text())
    assert ruff.main(["check", *args]) == 0
    migration = src / "db" / "migrations" / "x.py"
    migration.parent.mkdir(parents=True)
    migration.write_text("# noqa: BLE001\n")
    assert ruff.main(["check", *args]) == 0
