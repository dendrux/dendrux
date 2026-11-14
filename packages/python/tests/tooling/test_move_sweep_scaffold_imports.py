"""The reviewed sweep Protocol explains only exact typing imports."""

from __future__ import annotations

import pytest

from tests.tooling.test_move_manifest import manifest
from tools.move_check import check_move

HEAD = "from __future__ import annotations\nfrom typing import TYPE_CHECKING, Protocol\n"
CONTRACT = "class _SweepStore(Protocol):\n    _session_factory: object\n"
SCAN = (
    "class _SweepScan:\n    def __init__(self, store: SQLStateStore):\n        self.store = store\n"
)
IMPORT = "from host import _SweepStore"


def _proof(imports: str = IMPORT, scan: str | None = None):
    old = HEAD + "class SQLStateStore: pass\n" + SCAN
    after = {
        "old": HEAD + "from new import _SweepScan\nclass SQLStateStore: pass\n",
        "host": HEAD + CONTRACT,
        "new": HEAD
        + "if TYPE_CHECKING:\n    "
        + imports.replace("\n", "\n    ")
        + "\n"
        + (scan or SCAN.replace("SQLStateStore", "_SweepStore"))
        + '_SweepScan.__module__ = "old"\n',
    }
    evidence = manifest(
        old,
        after,
        {"SQLStateStore": "old", "_SweepScan": "new", "_SweepScan.__init__": "new"},
        scaffolding={"_SweepStore": {"module": "host", "source": CONTRACT}},
        annotations=[{"root": "_SweepScan", "module": "new", "host": "_SweepStore", "count": 1}],
        metadata=[
            {"root": "_SweepScan", "module": "new", "attribute": "__module__", "value": "old"}
        ],
    )
    return check_move(old, after, move_manifest=evidence)


def test_exact_declared_sweep_protocol_type_import_passes() -> None:
    report = _proof()
    assert report.ok, report.problems


@pytest.mark.parametrize(
    "imports, unexplained",
    [
        ("from wrong import _SweepStore", "wrong._SweepStore"),
        (IMPORT + ", _Unreviewed", "host._Unreviewed"),
        (IMPORT + "\nfrom http.client import HTTPConnection", "http.client.HTTPConnection"),
    ],
)
def test_other_sweep_peer_imports_still_fail(imports: str, unexplained: str) -> None:
    report = _proof(imports)
    assert not report.ok
    assert any("extra-import:" in problem and unexplained in problem for problem in report.problems)


@pytest.mark.parametrize(
    "scan",
    [
        SCAN.replace("SQLStateStore", "_SweepStore").replace(
            "self.store = store", "self.store = _SweepStore()"
        ),
        SCAN.replace("SQLStateStore", "_SweepStore = _SweepStore"),
        "@_SweepStore\n" + SCAN.replace("SQLStateStore", "_SweepStore"),
        SCAN.replace("SQLStateStore", "_SweepStore").replace(
            "self.store = store", "self.store = _SweepStore"
        ),
    ],
)
def test_sweep_type_import_cannot_be_used_at_runtime(scan: str) -> None:
    report = _proof(scan=scan)
    assert not report.ok
    assert any(
        "type-only-runtime:" in problem and "_SweepStore" in problem for problem in report.problems
    )
