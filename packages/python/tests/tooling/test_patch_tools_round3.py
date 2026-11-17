"""Patch tooling respects scope declarations and refuses relative helper calls."""

from __future__ import annotations

import ast
import textwrap
from typing import TYPE_CHECKING

import pytest

from tests.tooling.test_patch_tools_round2 import IMPORT, MAPPING, build
from tools import _patch_sites as ps
from tools import _scopes as sc
from tools import patch_codemod as pc
from tools import stale_patch_lint as sp

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("declaration", ["global", "nonlocal"])
@pytest.mark.parametrize("ambiguous", [False, True])
def test_declared_patch_targets_resolve_or_fail_unresolved(
    tmp_path: Path, declaration: str, ambiguous: bool
) -> None:
    binding = "from dendrux.runtime import runner\n"
    if ambiguous:
        binding += "runner = object()\n"
    if declaration == "global":
        source = binding + "\ndef outer():\n    runner = object()\n"
    else:
        source = "runner = object()\n\ndef outer():\n" + textwrap.indent(binding, "    ")
    source += (
        "    def inner(monkeypatch):\n"
        f"        {declaration} runner\n"
        '        monkeypatch.setattr(runner, "X", 99)\n'
        "    return inner\n"
    )
    compile(source, "test_a.py", "exec")
    sites = ps.find_sites(sc.ScopeIndex(ast.parse(source)))
    assert [site.kind for site in sites] == ["unresolved" if ambiguous else "alias"]
    src, tests = build(tmp_path, IMPORT, source)
    result = sp.scan(tests, src)
    assert (len(result.stale), len(result.unresolved)) == ((0, 1) if ambiguous else (1, 0))
    assert sp.main(["check", "--tests", str(tests), "--src", str(src)]) == 1
    rewritten = pc.rewrite_source(source, MAPPING)
    if ambiguous:
        assert rewritten.source == source
        assert rewritten.notes and "shadowed" not in " ".join(rewritten.notes)
    else:
        assert 'monkeypatch.setattr(events, "X", 99)' in rewritten.source
        assert not rewritten.notes


def test_a_global_declaration_also_applies_to_nested_reads() -> None:
    source = """\
        from dendrux.runtime import runner

        def outer():
            runner = object()
            def middle():
                global runner
                def inner(monkeypatch):
                    monkeypatch.setattr(runner, "X", 99)
                return inner
            return middle
        """
    rewritten = pc.rewrite_source(textwrap.dedent(source), MAPPING)
    assert 'monkeypatch.setattr(events, "X", 99)' in rewritten.source
    assert not rewritten.notes


def test_a_global_patch_callable_ignores_an_enclosing_local() -> None:
    source = """\
        from unittest.mock import patch

        def outer():
            patch = object()
            def inner():
                global patch
                with patch(target="dendrux.runtime.runner.X"):
                    pass
            return inner
        """
    rewritten = pc.rewrite_source(textwrap.dedent(source), MAPPING)
    assert 'target="dendrux.runtime.events.X"' in rewritten.source
    assert not rewritten.notes


def test_a_global_read_keeps_a_module_import_live() -> None:
    source = (
        IMPORT
        + """\
def outer():
    X = object()
    def inner():
        global X
        return X
    return inner
"""
    )
    assert "X" not in sp.analyse(source).reexported


def test_a_declared_non_import_target_is_never_reported_shadowed(tmp_path: Path) -> None:
    source = """\
        from dendrux.runtime import runner

        def outer():
            runner = object()
            def inner(monkeypatch):
                nonlocal runner
                monkeypatch.setattr(runner, "X", 99)
            return inner
        """
    source = textwrap.dedent(source)
    sites = ps.find_sites(sc.ScopeIndex(ast.parse(source)))
    assert [site.kind for site in sites] == ["unresolved"]
    src, tests = build(tmp_path, IMPORT, source)
    assert sp.main(["check", "--tests", str(tests), "--src", str(src)]) == 1


def test_an_unbound_declared_global_target_is_unresolved(tmp_path: Path) -> None:
    source = """\
        def outer():
            from dendrux.runtime import runner
            def inner(monkeypatch):
                global runner
                monkeypatch.setattr(runner, "X", 99)
            return inner
        """
    source = textwrap.dedent(source)
    src, tests = build(tmp_path, IMPORT, source)
    assert len(sp.scan(tests, src).unresolved) == 1
    assert sp.main(["check", "--tests", str(tests), "--src", str(src)]) == 1


@pytest.mark.parametrize("package", [None, "tests"])
@pytest.mark.parametrize(
    "call",
    [
        'patch("dendrux.runtime.runner.X")',
        'patch(target="dendrux.runtime.runner.X")',
        'patch.object(runner, "X")',
        'patch.object(target=runner, attribute="X")',
        'patch.object(runner, attribute="X")',
    ],
)
def test_relative_patch_helpers_are_unresolved_for_every_argument_form(
    tmp_path: Path, package: str | None, call: str
) -> None:
    source = (
        "from .helpers import patch\n"
        "from dendrux.runtime import runner\n\n"
        f"def test_a():\n    with {call}:\n        pass\n"
    )
    sites = ps.find_sites(sc.ScopeIndex(ast.parse(source), package))
    assert [site.kind for site in sites] == ["unresolved"]
    rewritten = pc.rewrite_source(source, MAPPING, package)
    assert rewritten.source == source
    assert rewritten.notes and "cannot be resolved" in rewritten.notes[0]
    src, tests = build(tmp_path, IMPORT, source)
    if package:
        (tests / "__init__.py").write_text("")
    assert len(sp.scan(tests, src).unresolved) == 1
    assert sp.main(["check", "--tests", str(tests), "--src", str(src)]) == 1
