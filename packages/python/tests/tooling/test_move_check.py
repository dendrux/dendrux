"""tools/move_check.py proves a refactor only moved code."""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

from tools import move_check as mc

OLD = '''\
"""Old module."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

LIMIT = 10
NAME: str = "x"


def helper(a, b=1):
    """Add."""
    return a + b


async def fetch(url: str, *, retries: int = 3) -> str:
    return url * retries


class Thing(Base):
    """A thing."""

    size = 3

    def alpha(self):
        return os.getcwd()

    def beta(self, x=2):
        return x
'''


FUT = "from __future__ import annotations\n"


def d(text: str) -> str:
    return textwrap.dedent(text)


def check(after: dict[str, str], old: str = OLD, allow_new: tuple[str, ...] = ()):
    return mc.check_move(old, after, allow_new=allow_new)


def test_a_pure_move_passes() -> None:
    old_stays = d('''\
        """Old module."""
        from __future__ import annotations
        from new_a import helper, fetch
        from new_b import LIMIT, NAME
        import os

        class Thing(Base):
            """A thing."""

            size = 3

            def alpha(self):
                return os.getcwd()

            def beta(self, x=2):
                return x
        ''')
    new_a = d('''\
        from __future__ import annotations

        def helper(a, b=1):
            """Add."""
            return a + b


        async def fetch(url: str, *, retries: int = 3) -> str:
            return url * retries
        ''')
    new_b = "from __future__ import annotations\nLIMIT = 10\nNAME: str = 'x'\n"
    report = check({"old": old_stays, "new_a": new_a, "new_b": new_b})
    assert report.problems == []
    assert report.moved == {"helper": "new_a", "fetch": "new_a", "LIMIT": "new_b", "NAME": "new_b"}


def test_a_method_moved_into_a_mixin_class_passes() -> None:
    old_after = d('''\
        """Old module."""
        from __future__ import annotations
        class Thing(_Beta, Base):
            """A thing."""

            size = 3

            def alpha(self):
                return os.getcwd()
        ''')
    mixin = d("""\
        from __future__ import annotations
        class _Beta:
            def beta(self, x=2):
                return x
        """)
    rest = d('''\
        from __future__ import annotations
        LIMIT = 10
        NAME: str = "x"

        def helper(a, b=1):
            """Add."""
            return a + b

        async def fetch(url: str, *, retries: int = 3) -> str:
            return url * retries
        ''')
    report = check({"old": old_after, "mixin": mixin, "rest": rest})
    assert report.problems == []
    assert report.moved["Thing.beta"] == "mixin"


def test_identical_source_passes_with_no_moves() -> None:
    report = check({"old": OLD})
    assert report.problems == [] and report.moved == {}


def test_a_changed_line_fails_with_a_diff() -> None:
    after = {"old": OLD.replace("return a + b", "return b + a")}
    report = check(after)
    [problem] = report.problems
    assert "helper" in problem and "changed" in problem
    assert "-" in problem and "+" in problem  # readable unified diff


def test_a_changed_default_fails() -> None:
    after = {"old": OLD.replace("b=1", "b=2")}
    assert any("helper" in p for p in check(after).problems)


def test_a_changed_method_default_in_a_mixin_fails() -> None:
    old_after = (
        FUT + "class Thing(_Beta, Base):\n    size = 3\n\n"
        "    def alpha(self):\n        return os.getcwd()\n"
    )
    mixin = FUT + "class _Beta:\n    def beta(self, x=3):\n        return x\n"
    rest = OLD.split("class Thing")[0]
    report = check({"old": old_after, "mixin": mixin, "rest": rest})
    assert any("Thing.beta" in p for p in report.problems)


def test_a_dropped_function_fails() -> None:
    after = {"old": OLD.replace('def helper(a, b=1):\n    """Add."""\n    return a + b\n', "")}
    problems = check(after).problems
    assert any("dropped" in p and "helper" in p for p in problems)


def test_a_duplicated_function_fails() -> None:
    copy = FUT + 'def helper(a, b=1):\n    """Add."""\n    return a + b\n'
    problems = check({"old": OLD, "other": copy}).problems
    assert any("duplicated" in p and "helper" in p for p in problems)


def test_an_extra_definition_fails_unless_allowed() -> None:
    extra = FUT + "def brand_new():\n    return 1\n"
    problems = check({"old": OLD, "other": extra}).problems
    assert any("extra" in p and "brand_new" in p for p in problems)
    assert check({"old": OLD, "other": extra}, allow_new=("brand_new",)).problems == []


def test_unexplained_imports_are_not_ignored() -> None:
    after = OLD.replace("import os\n", "import os\nimport sys\nfrom a import b\n")
    problems = check({"old": after}).problems
    assert any("extra-import" in p and "import sys" in p for p in problems)
    assert any("extra-import" in p and "from a import b" in p for p in problems)


def test_the_module_docstring_is_not_compared_but_a_new_dunder_all_in_the_old_module_fails() -> (
    None
):
    after = OLD.replace('"""Old module."""', '"""Rewritten."""')
    assert check({"old": after}).problems == []
    assert any("__all__" in p for p in check({"old": after + '\n__all__ = ["helper"]\n'}).problems)


def test_a_literal_dunder_all_in_a_new_module_is_not_compared() -> None:
    old_after = OLD.replace('def helper(a, b=1):\n    """Add."""\n    return a + b\n', "")
    new = FUT + 'def helper(a, b=1):\n    """Add."""\n    return a + b\n__all__ = ["helper"]\n'
    assert check({"old": old_after, "new": new}).problems == []


def test_a_mixin_that_is_not_a_base_of_the_class_fails() -> None:
    old_after = (
        "class Thing(Base):\n    size = 3\n\n    def alpha(self):\n        return os.getcwd()\n"
    )
    mixin = FUT + "class _Beta:\n    def beta(self, x=2):\n        return x\n"
    rest = OLD.split("class Thing")[0]
    report = check({"old": old_after, "mixin": mixin, "rest": rest})
    assert any("_Beta" in p and "base" in p for p in report.problems)


def test_a_changed_base_list_fails() -> None:
    assert check({"old": OLD.replace("Thing(Base)", "Thing(Other)")}).problems


def test_require_reexports_flags_a_moved_public_name_the_old_module_lost() -> None:
    old_after = OLD.replace('def helper(a, b=1):\n    """Add."""\n    return a + b\n', "")
    new = FUT + 'def helper(a, b=1):\n    """Add."""\n    return a + b\n'
    report = mc.check_move(OLD, {"old": old_after, "new": new}, require_reexports=True)
    assert any("re-export" in p and "helper" in p for p in report.problems)
    reexported = old_after + "\nfrom new import helper\n"
    assert (
        mc.check_move(OLD, {"old": reexported, "new": new}, require_reexports=True).problems == []
    )


def test_the_cli_compares_against_a_git_base_ref(tmp_path: Path, capsys) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    git("config", "user.email", "a@b.c")
    git("config", "user.name", "t")
    (tmp_path / "old.py").write_text(OLD)
    git("add", ".")
    git("commit", "-q", "-m", "base")
    (tmp_path / "new.py").write_text(
        FUT + 'def helper(a, b=1):\n    """Add."""\n    return a + b\n'
    )
    (tmp_path / "old.py").write_text(
        OLD.replace('def helper(a, b=1):\n    """Add."""\n    return a + b\n', "")
    )
    import os

    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        ok = mc.main(["--base", "HEAD", "--old", "old.py", "--new", "new.py"])
        (tmp_path / "new.py").write_text(FUT + "def helper(a, b=1):\n    return a\n")
        bad = mc.main(["--base", "HEAD", "--old", "old.py", "--new", "new.py"])
    finally:
        os.chdir(cwd)
    assert (ok, bad) == (0, 1)
    out = capsys.readouterr().out
    assert "helper" in out and "changed" in out


def test_the_cli_can_emit_a_patch_map(tmp_path: Path) -> None:
    import json
    import os

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    git("config", "user.email", "a@b.c")
    git("config", "user.name", "t")
    pkg = tmp_path / "src" / "dendrux" / "runtime"
    pkg.mkdir(parents=True)
    (pkg / "runner.py").write_text("def helper():\n    return 1\n\nkeep = 2\n")
    git("add", ".")
    git("commit", "-q", "-m", "base")
    (pkg / "runner.py").write_text("keep = 2\n")
    (pkg / "extra.py").write_text("def helper():\n    return 1\n")
    out = tmp_path / "map.json"
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        code = mc.main(
            [
                "--base",
                "HEAD",
                "--old",
                "src/dendrux/runtime/runner.py",
                "--new",
                "src/dendrux/runtime/extra.py",
                "--emit-map",
                str(out),
            ]
        )
    finally:
        os.chdir(cwd)
    assert code == 0
    assert json.loads(out.read_text()) == {
        "dendrux.runtime.runner": {"helper": "dendrux.runtime.extra"}
    }
