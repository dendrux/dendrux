"""0.8 T0 part C: the managed driver split keeps every name, identity and patch seam."""

import ast
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from dendrux.runtime import _managed
from dendrux.types import RunResult, RunStatus

FACADE = "dendrux.runtime._managed"
MOVED = {
    "dendrux.runtime._managed_adoption": (
        "logger",
        "REPORT_TOO_LARGE",
        "CHILD_FAILED",
        "CHILD_CANCELLED",
        "REPORT_INVALID",
        "child_report",
        "_report_payload",
        "_FAILURE_TEXT",
        "_FAILURE_CODES",
        "_failed_report",
        "AcceptedReport",
        "accept_report",
        "_transformed",
        "project_report",
    ),
    "dendrux.runtime._managed_route": (
        "MANAGED_WAIT_REASON",
        "_Waiting",
        "root_view",
        "_root_result",
        "_projected_pause",
        "Route",
        "walk",
        "_outgoing",
        "_bound_child",
        "_intent_call",
        "_registered",
    ),
}
# Tests patch these on the facade; the driver must keep reading them there.
PATCH_SEAMS = ("route", "claim_adoption", "adopt", "complete_adoption", "_run_child", "bind_child")


@pytest.mark.parametrize(
    ("module", "name"), [(module, name) for module, names in MOVED.items() for name in names]
)
def test_moved_name_is_the_same_object_through_the_facade(module: str, name: str) -> None:
    owner = importlib.import_module(module)
    assert name in vars(owner)
    assert getattr(_managed, name) is vars(owner)[name]


@pytest.mark.parametrize(
    "name",
    ["_Waiting", "Route", "AcceptedReport"],
)
def test_moved_class_keeps_its_original_identity(name: str) -> None:
    cls = getattr(_managed, name)
    assert (cls.__module__, cls.__qualname__) == (FACADE, name)


def test_logger_keeps_the_original_name() -> None:
    adoption = importlib.import_module("dendrux.runtime._managed_adoption")
    assert _managed.logger is adoption.logger
    assert _managed.logger.name == FACADE


@pytest.mark.parametrize("name", PATCH_SEAMS)
def test_patch_seam_stays_defined_in_the_facade(name: str) -> None:
    assert getattr(_managed, name).__module__ == FACADE
    assert name in vars(_managed)


@pytest.mark.parametrize("module", sorted(MOVED))
def test_split_module_never_imports_the_facade_at_runtime(module: str) -> None:
    owner = importlib.import_module(module)
    tree = ast.parse(Path(inspect.getfile(owner)).read_text(encoding="utf-8"))
    runtime = [n for n in tree.body if isinstance(n, ast.Import | ast.ImportFrom)]
    targets = {
        f"{n.module}.{a.name}" if isinstance(n, ast.ImportFrom) else a.name
        for n in runtime
        for a in n.names
    } | {n.module for n in runtime if isinstance(n, ast.ImportFrom)}
    assert FACADE not in targets


async def test_facade_patch_of_adopt_still_reaches_the_driver(monkeypatch) -> None:
    calls: list[object] = []

    async def adopt(waiting, *, store, notifier):
        calls.append(waiting)
        return RunResult(run_id="root", status=RunStatus.SUCCESS)

    monkeypatch.setattr(_managed, "adopt", adopt)
    waiting = SimpleNamespace(parent_run_id="parent")
    done = RunResult(run_id="child", status=RunStatus.SUCCESS)
    result = await _managed._trampoline(
        done, [waiting], root_run_id="root", store=None, notifier=None
    )
    assert calls == [waiting] and result.run_id == "root"


async def test_driver_reads_the_root_view_through_its_new_owner(monkeypatch) -> None:
    route = importlib.import_module("dendrux.runtime._managed_route")
    view = RunResult(run_id="root", status=RunStatus.RUNNING)

    async def root_view(store, root_run_id):
        return view

    monkeypatch.setattr(route, "root_view", root_view)
    not_owner = RunResult(
        run_id="parent", status=RunStatus.RUNNING, meta={_managed.NOT_OWNER_KEY: True}
    )
    result = await _managed._trampoline(
        not_owner, [], root_run_id="root", store=None, notifier=None
    )
    assert result is view
