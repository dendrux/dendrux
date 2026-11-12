"""tools/stale_patch_lint.py: a patch of a pure re-export silently does nothing."""

from __future__ import annotations

import json
import textwrap
from typing import TYPE_CHECKING

from tools import stale_patch_lint as sp

if TYPE_CHECKING:
    from pathlib import Path


def build(tmp_path: Path, module_source: str, test_source: str) -> tuple[Path, Path]:
    src = tmp_path / "src"
    pkg = src / "dendrux" / "runtime"
    pkg.mkdir(parents=True)
    (src / "dendrux" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "runner.py").write_text(textwrap.dedent(module_source))
    (pkg / "events.py").write_text("def emit():\n    return 1\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_a.py").write_text(textwrap.dedent(test_source))
    return src, tests


def patching(target: str, name: str = "test_x") -> str:
    return f'def {name}(monkeypatch):\n    monkeypatch.setattr("{target}", 1)\n'


RUNNER = """\
from dendrux.runtime.events import emit as _emit_event
from dendrux.runtime import events
from dendrux.runtime.events import emit as used_emit


def defined():
    return used_emit()


assigned = 3
__all__ = ["_emit_event"]
"""


def stale(tmp_path: Path, test_source: str, module: str = RUNNER) -> list[str]:
    src, tests = build(tmp_path, module, test_source)
    return [f"{s.module}.{s.attr}" for s in sp.find_stale(tests, src)]


def test_patching_a_pure_re_export_by_string_is_stale(tmp_path: Path) -> None:
    found = stale(
        tmp_path,
        patching("dendrux.runtime.runner._emit_event"),
    )
    assert found == ["dendrux.runtime.runner._emit_event"]


def test_patch_and_patch_object_with_aliases_are_stale_too(tmp_path: Path) -> None:
    found = stale(
        tmp_path,
        """\
        from unittest.mock import patch
        from dendrux.runtime import runner

        @patch("dendrux.runtime.runner._emit_event")
        def test_a(m):
            with patch.object(runner, "_emit_event"):
                pass

        def test_b(monkeypatch):
            monkeypatch.setattr(runner, "_emit_event", 1)
        """,
    )
    assert found == ["dendrux.runtime.runner._emit_event"] * 3


def test_patching_a_definition_is_fine(tmp_path: Path) -> None:
    assert (
        stale(
            tmp_path,
            patching("dendrux.runtime.runner.defined")
            + patching("dendrux.runtime.runner.assigned", "test_y"),
        )
        == []
    )


def test_patching_an_import_the_module_uses_is_fine(tmp_path: Path) -> None:
    # patch-where-used: runner calls used_emit(), so this patch takes effect.
    assert (
        stale(
            tmp_path,
            patching("dendrux.runtime.runner.used_emit"),
        )
        == []
    )


def test_the_attribute_of_a_class_in_the_module_is_judged_by_the_class_name(tmp_path: Path) -> None:
    module = RUNNER + "\nclass Thing:\n    def m(self):\n        return 1\n"
    ok = patching("dendrux.runtime.runner.Thing.m")
    assert stale(tmp_path, ok, module) == []


def test_non_library_and_unknown_targets_are_ignored(tmp_path: Path) -> None:
    assert (
        stale(
            tmp_path,
            """\
        def test_x(monkeypatch):
            monkeypatch.setattr("os.getcwd", 1)
            monkeypatch.setattr("dendrux.nowhere.thing", 1)
            monkeypatch.setattr("dendrux.runtime.runner.missing", 1)
        """,
        )
        == []
    )


def test_a_package_re_export_is_stale(tmp_path: Path) -> None:
    src, tests = build(
        tmp_path,
        RUNNER,
        patching("dendrux.runtime.Reexported"),
    )
    (src / "dendrux" / "runtime" / "__init__.py").write_text(
        "from dendrux.runtime.runner import defined as Reexported\n"
    )
    assert [f"{s.module}.{s.attr}" for s in sp.find_stale(tests, src)] == [
        "dendrux.runtime.Reexported"
    ]


def test_the_cli_checks_against_a_shrink_only_baseline(tmp_path: Path, capsys) -> None:
    src, tests = build(
        tmp_path,
        RUNNER,
        patching("dendrux.runtime.runner._emit_event"),
    )
    baseline = tmp_path / "baseline.json"
    args = ["--tests", str(tests), "--src", str(src), "--baseline", str(baseline)]
    assert sp.main(["check", *args]) == 1  # no baseline file means zero allowed
    assert sp.main(["stamp", *args]) == 1  # a stamp never records a new problem
    assert sp.main(["stamp", *args, "--allow-regress", "--reason", "test"]) == 0
    assert list(json.loads(baseline.read_text()).values()) == [1]
    assert sp.main(["check", *args]) == 0
    (tests / "test_b.py").write_text(patching("dendrux.runtime.runner._emit_event", "test_y"))
    assert sp.main(["check", *args]) == 1
    assert "re-export" in capsys.readouterr().out
    (tests / "test_b.py").unlink()
    (tests / "test_a.py").write_text("def test_x():\n    pass\n")
    assert sp.main(["check", *args]) == 0  # improvement is not a failure


def test_patching_an_attribute_of_a_re_exported_class_is_not_a_module_patch(tmp_path: Path) -> None:
    # Patching a method of an imported object patches that object, not the module.
    found = stale(
        tmp_path,
        """\
        from dendrux.runtime.runner import _emit_event


        def test_x(monkeypatch):
            monkeypatch.setattr(_emit_event, "__init__", 1)
        """,
    )
    assert found == []


def test_a_clean_tree_needs_no_baseline_file(tmp_path: Path) -> None:
    src, tests = build(tmp_path, RUNNER, "def test_x():\n    pass\n")
    args = ["--tests", str(tests), "--src", str(src), "--baseline", str(tmp_path / "none.json")]
    assert sp.main(["check", *args]) == 0
