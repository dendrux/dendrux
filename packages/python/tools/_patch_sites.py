"""Find the patch call sites in a test module (shared by patch_codemod and stale_patch_lint).

A call is a patch call only when its callable *resolves* to one of the supported APIs through
the file's imports and the lexical scope of the call, never by its spelling:

* ``unittest.mock.patch(target, ...)`` and ``unittest.mock.patch.object(target, attribute, ...)``,
  however imported or aliased;
* ``mocker.patch`` / ``mocker.patch.object`` on the pytest-mock fixture;
* ``monkeypatch.setattr`` / ``monkeypatch.delattr`` on the pytest fixture (a parameter named
  ``monkeypatch`` or annotated ``MonkeyPatch``, or a ``MonkeyPatch`` context object).

Target forms: ``string`` (``"pkg.mod.name"``) and ``alias`` (a module object plus an attribute
name). The target may be positional or a keyword. A target name that is shadowed or bound
ambiguously is reported as ``unresolved`` so callers leave it alone and say so.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

try:
    from tools import _scopes as sc
except ImportError:  # run as a script from packages/python/tools
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import _scopes as sc

if TYPE_CHECKING:
    from tools._scopes import Scope


@dataclass(frozen=True)
class PatchSite:
    kind: Literal["string", "alias", "unresolved"]
    dotted: str  # string form: whole target; alias form: resolved object path
    attr: str  # alias form: the attribute name; "" otherwise
    node: ast.expr  # the target expression to rewrite
    lineno: int
    scope: Scope
    note: str = ""


_PATCH = "unittest.mock.patch"
_FIXTURE_APIS = {
    "monkeypatch.setattr": "attr",
    "monkeypatch.delattr": "attr",
    "mocker.patch": "string",
    "mocker.patch.object": "object",
}
_MOCK_APIS = {_PATCH: "string", f"{_PATCH}.object": "object"}


def _api(call: ast.Call, scope: Scope) -> str | None:
    """'string' | 'object' | 'attr' when the call is a supported patch API."""
    text = sc.dotted(call.func)
    if text is None:
        return None
    head, _, tail = text.partition(".")
    binding = scope.resolve(head)
    if binding is None:
        return None
    if binding.kind == "import":
        return _MOCK_APIS.get(f"{binding.value}.{tail}" if tail else binding.value)
    fixture = None
    if binding.kind == "mp" or (
        binding.kind == "param" and (head == "monkeypatch" or binding.value.endswith("MonkeyPatch"))
    ):
        fixture = "monkeypatch"
    elif binding.kind == "param" and head == "mocker":
        fixture = "mocker"
    if fixture is None:
        return None
    return _FIXTURE_APIS.get(f"{fixture}.{tail}")


def _arg(call: ast.Call, index: int, *keywords: str) -> ast.expr | None:
    if len(call.args) > index and not isinstance(call.args[index], ast.Starred):
        return call.args[index]
    for kw in call.keywords:
        if kw.arg in keywords:
            return kw.value
    return None


def _str(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _alias_site(call: ast.Call, target: ast.expr, attr: str, scope: Scope) -> PatchSite | None:
    text = sc.dotted(target)
    if text is None:
        return None
    head, _, tail = text.partition(".")
    if scope.find(head) is None:
        return None  # not a name of this file at all
    binding = scope.resolve(head)
    if binding is None or binding.kind != "import":
        if not scope.hides_import(head):
            return None  # an object patch (a class, a local), not a module alias
        note = f"line {call.lineno}: target {head!r} is shadowed or bound more than once; left"
        return PatchSite("unresolved", text, attr, target, call.lineno, scope, note)
    full = f"{binding.value}.{tail}" if tail else binding.value
    return PatchSite("alias", full, attr.split(".")[0], target, call.lineno, scope)


def _site(call: ast.Call, scope: Scope) -> PatchSite | None:
    api = _api(call, scope)
    if api is None:
        return None
    target = _arg(call, 0, "target")
    if target is None:
        return None
    as_string = _str(target)
    if api == "string" or (api == "attr" and as_string is not None):
        if as_string is not None and "." in as_string:
            return PatchSite("string", as_string, "", target, target.lineno, scope)
        return None
    name = _str(_arg(call, 1, "attribute", "name"))
    if name is None or as_string is not None:
        return None
    return _alias_site(call, target, name, scope)


def find_sites(index: sc.ScopeIndex) -> list[PatchSite]:
    sites = [s for call, scope in index.calls if (s := _site(call, scope)) is not None]
    return sorted(sites, key=lambda s: (s.lineno, s.node.col_offset))
