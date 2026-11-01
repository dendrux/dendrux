"""tools/semgrep_ratchet.py: custom Semgrep rules, ratcheted by rule + module + function."""

from __future__ import annotations

import os
import textwrap
from typing import TYPE_CHECKING

import pytest

from tools import semgrep_ratchet as sr

if TYPE_CHECKING:
    from pathlib import Path

SOURCE = textwrap.dedent(
    """\
    class Store:
        async def append(self, session):
            await session.commit()

        def helper(self):
            def inner():
                return 1
            return inner

    TOP = 1
    """
)


def test_enclosing_function_is_the_innermost_qualified_name() -> None:
    tree = sr.function_spans(SOURCE)
    assert sr.enclosing(tree, 3) == "Store.append"
    assert sr.enclosing(tree, 6) == "Store.helper.<locals>.inner"
    assert sr.enclosing(tree, 8) == "Store.helper"
    assert sr.enclosing(tree, 10) == "<module>"


def test_scope_uses_include_and_exclude_prefixes_under_the_package() -> None:
    meta = {"include": ["runtime/stores/sql/"], "exclude": ["runtime/stores/sql/mysql.py"]}
    assert sr.in_scope("src/dendrux/runtime/stores/sql/base.py", meta)
    assert not sr.in_scope("src/dendrux/runtime/stores/sql/mysql.py", meta)
    assert not sr.in_scope("src/dendrux/db/session.py", meta)
    assert sr.in_scope("src/dendrux/anything.py", {})
    assert not sr.in_scope("src/dendrux/types.py", {"exclude": ["types.py"]})


def test_keys_count_matches_per_rule_module_and_function(tmp_path: Path) -> None:
    src = tmp_path / "src" / "dendrux"
    (src / "runtime").mkdir(parents=True)
    (src / "runtime" / "m.py").write_text(SOURCE)
    results = [
        {
            "check_id": "semgrep.some-rule",
            "path": str(src / "runtime" / "m.py"),
            "start": {"line": 3},
            "extra": {"metadata": {}},
        },
        {
            "check_id": "semgrep.some-rule",
            "path": str(src / "runtime" / "m.py"),
            "start": {"line": 3},
            "extra": {"metadata": {}},
        },
        {
            "check_id": "x.other",
            "path": str(src / "runtime" / "m.py"),
            "start": {"line": 10},
            "extra": {"metadata": {}},
        },
    ]
    assert sr.keys_from_results(results) == {
        "other|dendrux.runtime.m|<module>": 1,
        "some-rule|dendrux.runtime.m|Store.append": 2,
    }


def test_a_new_match_fails_and_a_known_one_passes() -> None:
    known = {"r|dendrux.a|f": 1}
    assert not sr.compare(known, known).failed
    assert sr.compare({**known, "r|dendrux.a|g": 1}, known).new == ["r|dendrux.a|g"]
    assert sr.compare({"r|dendrux.a|f": 2}, known).worse == ["r|dendrux.a|f"]


def test_without_semgrep_the_gate_says_how_to_install_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEMGREP", raising=False)
    monkeypatch.setattr(sr.shutil, "which", lambda name: None)
    with pytest.raises(sr.SemgrepMissingError, match="pip install semgrep=="):
        sr.find_semgrep()


# --- end to end, only where semgrep is installed (CI quality job, or SEMGREP=... locally) ---

HAVE_SEMGREP = False
try:
    sr.find_semgrep()
    HAVE_SEMGREP = True
except sr.SemgrepMissingError:
    pass


@pytest.mark.skipif(not HAVE_SEMGREP, reason="semgrep not installed (set SEMGREP=/path/to/semgrep)")
def test_the_rules_pass_their_own_fixtures() -> None:
    assert sr.main(["test"]) == 0


@pytest.mark.skipif(not HAVE_SEMGREP, reason="semgrep not installed (set SEMGREP=/path/to/semgrep)")
def test_end_to_end_flags_a_new_commit_outside_write_tx(tmp_path: Path) -> None:
    src = tmp_path / "src" / "dendrux" / "runtime" / "stores" / "sql"
    src.mkdir(parents=True)
    for parent in (src, src.parent, src.parent.parent, src.parent.parent.parent):
        (parent / "__init__.py").write_text("")
    mod = src / "base.py"
    mod.write_text("class S:\n    async def _write_tx(self, s):\n        await s.commit()\n")
    baseline = tmp_path / "base.json"
    common = ["--src", str(tmp_path / "src" / "dendrux"), "--baseline", str(baseline)]
    assert sr.main(["stamp", *common]) == 0
    assert sr.main(["check", *common]) == 0
    mod.write_text(mod.read_text() + "    async def other(self, s):\n        await s.commit()\n")
    assert sr.main(["check", *common]) == 1
    assert os.path.exists(baseline)
