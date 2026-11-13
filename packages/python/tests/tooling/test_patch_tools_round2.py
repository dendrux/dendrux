"""Patch tooling: scopes, relative imports, factories and import reuse (reproduced false passes)."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest

from tools import patch_codemod as pc
from tools import stale_patch_lint as sp

if TYPE_CHECKING:
    from pathlib import Path

MAPPING = {"dendrux.runtime.runner": {"X": "dendrux.runtime.events"}}


def build(tmp_path: Path, module_source: str, test_source: str) -> tuple[Path, Path]:
    src = tmp_path / "src"
    pkg = src / "dendrux" / "runtime"
    pkg.mkdir(parents=True)
    (src / "dendrux" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "runner.py").write_text(textwrap.dedent(module_source))
    (pkg / "events.py").write_text("X = 1\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_a.py").write_text(textwrap.dedent(test_source))
    return src, tests


def stale(tmp_path: Path, module: str, test: str) -> int:
    src, tests = build(tmp_path, module, test)
    return len(sp.find_stale(tests, src))


def unresolved(tmp_path: Path, module: str, test: str) -> list[str]:
    src, tests = build(tmp_path, module, test)
    return [f"{p.name}:{n}" for p, _, n in sp.scan(tests, src).unresolved]


# the three call forms the lint understands, against a module that re-exports X
FORMS = """\
from unittest.mock import patch
from dendrux.runtime import runner


def test_string(monkeypatch):
    monkeypatch.setattr("dendrux.runtime.runner.X", 1)


def test_object(monkeypatch):
    with patch.object(runner, "X"):
        pass


def test_mock_string():
    with patch("dendrux.runtime.runner.X"):
        pass
"""


# ---- relative imports are imports -------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "from .events import X",
        "from . import events as X",
        "from ..runtime.events import X",
        "from .events import X as X",
    ],
)
def test_a_relative_re_export_is_stale_like_an_absolute_one(tmp_path: Path, line: str) -> None:
    assert stale(tmp_path, line + "\n", FORMS) == 3


def test_a_relative_import_beyond_the_package_is_not_a_crash(tmp_path: Path) -> None:
    assert stale(tmp_path, "from .... import X\n", FORMS) in (0, 3)


# ---- comprehension scopes, global and nonlocal ---------------------------------

IMPORT = "from dendrux.runtime.events import X\n"


@pytest.mark.parametrize(
    "tail",
    [
        "xs = [X for X in range(2)]\n",
        "xs = {X: 1 for X in range(2)}\n",
        "xs = list(X for X in range(2))\n",
        "def unrelated():\n    global X\n    return 1\n",
        "def outer():\n    X = 1\n    def inner():\n        nonlocal X\n        X = 2\n"
        "    return inner\n",
    ],
)
def test_neither_a_comprehension_variable_nor_a_bare_declaration_hides_the_import(
    tmp_path: Path, tail: str
) -> None:
    assert stale(tmp_path, IMPORT + tail, FORMS) == 3


def test_a_comprehension_variable_does_not_hide_a_test_module_alias(tmp_path: Path) -> None:
    test = FORMS.replace(
        "def test_object",
        "xs = [runner for runner in range(2)]\n\n\ndef test_object",
    )
    assert stale(tmp_path, IMPORT, test) == 3


def test_the_first_iterable_of_a_comprehension_is_read_in_the_enclosing_scope(
    tmp_path: Path,
) -> None:
    module = IMPORT + "def f():\n    return [y for y in X]\n"
    assert stale(tmp_path, module, FORMS) == 0  # a real read at call time


def test_a_function_that_rebinds_the_name_through_global_is_a_definition(tmp_path: Path) -> None:
    module = IMPORT + "def reset():\n    global X\n    X = None\n"
    assert stale(tmp_path, module, FORMS) == 0


def test_an_ambiguous_patch_target_fails_the_lint(tmp_path: Path) -> None:
    test = """\
        from dendrux.runtime import runner

        runner = object()


        def test_a(monkeypatch):
            monkeypatch.setattr(runner, "X", 1)
        """
    found = unresolved(tmp_path, IMPORT, test)
    assert found and "test_a.py" in found[0]
    src, tests = build(tmp_path / "again", IMPORT, test)
    assert sp.main(["check", "--tests", str(tests), "--src", str(src)]) == 1


def test_a_deterministic_shadow_does_not_fail_the_lint(tmp_path: Path) -> None:
    test = """\
        from dendrux.runtime import runner


        def test_a(runner, monkeypatch):
            monkeypatch.setattr(runner, "X", 1)
        """
    assert unresolved(tmp_path, IMPORT, test) == []


def test_a_re_exported_patch_helper_cannot_be_resolved_and_fails(tmp_path: Path) -> None:
    test = """\
        from .helpers import patch


        def test_a():
            with patch("dendrux.runtime.runner.X"):
                pass
        """
    assert unresolved(tmp_path, IMPORT, test)


# ---- MonkeyPatch factories are resolved, never guessed -------------------------

CALL = 'mp.setattr("dendrux.runtime.runner.X", 1)'


@pytest.mark.parametrize(
    "test",
    [
        "from pytest import MonkeyPatch as MP\nmp = MP()\n" + CALL + "\n",
        "import pytest\nmp = pytest.MonkeyPatch()\n" + CALL + "\n",
        "import pytest as pt\nwith pt.MonkeyPatch.context() as mp:\n    " + CALL + "\n",
        "from _pytest.monkeypatch import MonkeyPatch\nmp = MonkeyPatch()\n" + CALL + "\n",
        "import pytest\n\n\ndef f(mp: pytest.MonkeyPatch):\n    " + CALL + "\n",
        "from pytest import MonkeyPatch as MP\n\n\ndef f(mp: MP):\n    " + CALL + "\n",
        "def f(monkeypatch):\n    with monkeypatch.context() as mp:\n        " + CALL + "\n",
    ],
)
def test_real_monkeypatch_objects_are_found(tmp_path: Path, test: str) -> None:
    assert stale(tmp_path, IMPORT, test) == 1


@pytest.mark.parametrize(
    "test",
    [
        "from custom import VendorMonkeyPatch\nmp = VendorMonkeyPatch()\n" + CALL + "\n",
        "import vendor\nmp = vendor.MonkeyPatch()\n" + CALL + "\n",
        "from custom import MonkeyPatch\n\n\ndef f(mp: MonkeyPatch):\n    " + CALL + "\n",
        "def MonkeyPatch():\n    ...\n\n\nmp = MonkeyPatch()\n" + CALL + "\n",
    ],
)
def test_lookalike_objects_are_not_patch_sites(tmp_path: Path, test: str) -> None:
    assert stale(tmp_path, IMPORT, test) == 0


def test_the_codemod_rewrites_only_the_real_monkeypatch() -> None:
    real = pc.rewrite_source(
        "from pytest import MonkeyPatch as MP\nmp = MP()\n" + CALL + "\n", MAPPING
    ).source
    assert "dendrux.runtime.events.X" in real
    fake = "from custom import VendorMonkeyPatch\nmp = VendorMonkeyPatch()\n" + CALL + "\n"
    assert pc.rewrite_source(fake, MAPPING).source == fake


# ---- reuse only an unconditional module-level import placed before the use -------------

ALIAS_PATCH = """\
from dendrux.runtime import runner
{head}

def test_a(monkeypatch):
    monkeypatch.setattr(runner, "X", 1)
{tail}"""


def rewrite(source: str) -> str:
    return pc.rewrite_source(source, MAPPING).source


def test_an_import_before_the_use_is_reused() -> None:
    out = rewrite(ALIAS_PATCH.format(head="from dendrux.runtime import events", tail=""))
    assert 'monkeypatch.setattr(events, "X", 1)' in out
    assert out.count("import events") == 1


@pytest.mark.parametrize(
    "head,tail",
    [
        ("", "    from dendrux.runtime import events\n"),  # a local import after the use
        ("", "\nfrom dendrux.runtime import events\n"),  # a module import after the function
        (
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n"
            "    from dendrux.runtime import events",
            "",
        ),
        (
            "try:\n    from dendrux.runtime import events\nexcept ImportError:\n    events = None",
            "",
        ),
        ("if True:\n    from dendrux.runtime import events", ""),
    ],
)
def test_an_import_that_may_not_have_run_is_not_reused(head: str, tail: str) -> None:
    out = rewrite(ALIAS_PATCH.format(head=head, tail=tail))
    assert 'monkeypatch.setattr(events_mod, "X", 1)' in out
    assert "from dendrux.runtime import events as events_mod\n" in out
    assert out.index("as events_mod") < out.index("def test_a")


def test_a_module_import_after_a_decorated_test_is_not_reused() -> None:
    source = """\
        from unittest.mock import patch
        from dendrux.runtime import runner


        @patch.object(runner, "X")
        def test_a(mock):
            pass


        from dendrux.runtime import events
        """
    out = rewrite(textwrap.dedent(source))
    assert '@patch.object(events_mod, "X")' in out
    assert out.index("as events_mod") < out.index("@patch.object")
