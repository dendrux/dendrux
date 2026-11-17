"""Only exact reviewed scaffold identities explain peer type imports."""

from __future__ import annotations

import pytest

from tests.tooling.test_move_manifest import manifest
from tools.move_check import check_move

HEAD = "from __future__ import annotations\nfrom typing import TYPE_CHECKING\n"
CONTRACT = "class _StoreHost:\n    _session_factory: object\n"
FUNCTION = "def _creation_mark_proven(store: SQLStateStore):\n    return store\n"
IMPORT = "from host import _StoreHost"


def _proof(imports: str = IMPORT, function: str | None = None, *, scaffold: str = CONTRACT):
    old = HEAD + "class SQLStateStore: pass\n" + FUNCTION
    after = {
        "old": HEAD + "from new import _creation_mark_proven\nclass SQLStateStore: pass\n",
        "host": HEAD + scaffold,
        "new": HEAD
        + "if TYPE_CHECKING:\n    "
        + imports.replace("\n", "\n    ")
        + "\n"
        + (function or FUNCTION.replace("SQLStateStore", "_StoreHost")),
    }
    owners = {"SQLStateStore": "old", "_creation_mark_proven": "new"}
    evidence = manifest(
        old,
        after,
        owners,
        scaffolding={"_StoreHost": {"module": "host", "source": scaffold}},
        annotations=[
            {"root": "_creation_mark_proven", "module": "new", "host": "_StoreHost", "count": 1}
        ],
    )
    return check_move(old, after, move_manifest=evidence)


def test_exact_declared_scaffold_type_import_passes() -> None:
    report = _proof()
    assert report.ok, report.problems


@pytest.mark.parametrize(
    "imports, unexplained",
    [
        ("from wrong import _StoreHost", "wrong._StoreHost"),
        (IMPORT + ", _Unreviewed", "host._Unreviewed"),
        (IMPORT + "\nfrom http.client import HTTPConnection", "http.client.HTTPConnection"),
    ],
)
def test_other_peer_imports_still_fail(imports: str, unexplained: str) -> None:
    report = _proof(imports)
    assert not report.ok
    assert any("extra-import:" in p and unexplained in p for p in report.problems)


@pytest.mark.parametrize(
    "function",
    [
        "def _creation_mark_proven(store: _StoreHost):\n    return _StoreHost()\n",
        "def _creation_mark_proven(store: _StoreHost = _StoreHost):\n    return store\n",
        "@_StoreHost\ndef _creation_mark_proven(store: _StoreHost):\n    return store\n",
    ],
)
def test_type_only_scaffold_cannot_be_used_at_runtime(function: str) -> None:
    report = _proof(function=function)
    assert not report.ok
    assert any("type-only-runtime:" in p and "_StoreHost" in p for p in report.problems)


def test_peer_import_cannot_authorize_executable_scaffold() -> None:
    report = _proof(scaffold=CONTRACT + "    runtime_value = 1\n")
    assert not report.ok
    assert any("no runtime members" in p for p in report.problems)
