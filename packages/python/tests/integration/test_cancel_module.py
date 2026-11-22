"""The standalone ``dendrux.cancel_run``: sources, tenancy, bound, normalization and result."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

import dendrux
from dendrux import UNSCOPED, Agent, Budget, cancel_run
from dendrux.errors import (
    ManagedChildControlError,
    RunNotFoundError,
    TenantScopeRequiredError,
)
from dendrux.runtime import _cancel, sweep
from dendrux.runtime.state import claimed_store
from dendrux.types import InputValidationError, RunStatus, generate_ulid
from tests.integration.test_cancel_service import seed
from tests.integration.test_managed_cancel import TENANT, chain

if TYPE_CHECKING:
    from pathlib import Path


class NoIO:
    """A store that fails any use, proving validation happens before I/O."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"store used: {name}")


def track_disposal(monkeypatch: Any) -> list[int]:
    disposed: list[int] = []
    real = AsyncEngine.dispose

    async def dispose(self: AsyncEngine, *args: Any, **kwargs: Any) -> None:
        disposed.append(id(self))
        await real(self, *args, **kwargs)

    monkeypatch.setattr(AsyncEngine, "dispose", dispose)
    return disposed


def test_exported_from_the_package() -> None:
    assert "cancel_run" in dendrux.__all__
    assert dendrux.cancel_run is cancel_run


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"state_store": NoIO(), "database_url": "sqlite+aiosqlite:///{path}"},
    ],
    ids=["neither", "both"],
)
async def test_exactly_one_source_before_io(tmp_path: Path, kwargs: dict[str, Any]) -> None:
    path = tmp_path / "never.db"
    options = {k: v.format(path=path) if isinstance(v, str) else v for k, v in kwargs.items()}
    with pytest.raises(ValueError, match="database_url|state_store"):
        await cancel_run("r", tenant_id=TENANT, **options)
    assert not path.exists()


def test_tenant_id_is_a_required_keyword() -> None:
    with pytest.raises(TypeError):
        cancel_run("r", state_store=NoIO())  # type: ignore[call-arg]


async def test_unscoped_is_refused_with_a_logged_code(caplog: Any) -> None:
    caplog.set_level(logging.WARNING)
    run_id = generate_ulid()
    with pytest.raises(TenantScopeRequiredError) as info:
        await cancel_run(run_id, tenant_id=UNSCOPED, state_store=NoIO())  # type: ignore[arg-type]
    assert info.value.operation == "cancel_run"
    assert str(info.value) == (
        "cancel_run() requires tenant_id=<str or None>; UNSCOPED is not accepted."
    )
    record = next(r for r in caplog.records if getattr(r, "code", None) == "tenant_scope_required")
    assert (record.run_id, record.entry) == (run_id, "cancel_run")


@pytest.mark.parametrize("tenant", ["", 5, b"t1"])
async def test_invalid_tenant_is_refused_before_io(tenant: Any) -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        await cancel_run("r", tenant_id=tenant, state_store=NoIO())


@pytest.mark.parametrize("run_id", ["", 5, "x" * 300])
async def test_invalid_run_id_is_refused_before_io(run_id: Any) -> None:
    with pytest.raises(InputValidationError):
        await cancel_run(run_id, tenant_id=TENANT, state_store=NoIO())


@pytest.mark.parametrize("cascade", [None, 1, "yes"])
async def test_cascade_must_be_a_bool(cascade: Any) -> None:
    with pytest.raises(ValueError, match="cascade"):
        await cancel_run("r", tenant_id=TENANT, state_store=NoIO(), cascade=cascade)


@pytest.mark.parametrize("bound", [0, -1, True, 1.5, "10"])
async def test_invalid_max_nodes_is_refused_before_io(bound: Any) -> None:
    with pytest.raises(ValueError, match="max_nodes"):
        await cancel_run("r", tenant_id=TENANT, state_store=NoIO(), max_nodes=bound)


@pytest.mark.parametrize(("given", "expected"), [(None, Budget.max_subtree_nodes), (7, 7)])
async def test_bound_default_and_override_reach_the_service(
    store: Any, monkeypatch: Any, given: int | None, expected: int
) -> None:
    root, _ = await seed(store)
    seen: list[int] = []
    service = _cancel.cancel_run

    async def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs["max_nodes"])
        return await service(*args, **kwargs)

    monkeypatch.setattr(_cancel, "cancel_run", spy)
    await cancel_run(root, tenant_id=TENANT, state_store=store, max_nodes=given)
    assert seen == [expected]


async def test_bound_truncates_like_the_agent(store: Any) -> None:
    root, kids = await seed(store, children=3)
    result = await cancel_run(root, tenant_id=TENANT, state_store=store, max_nodes=2)
    assert (result.meta["cascade_count"], result.meta["cascade_truncated"]) == (2, True)
    assert result.meta["cascade"][0] == {"run_id": kids[0], "outcome": "cancelled"}


async def test_cascade_defaults_to_true(store: Any) -> None:
    root, kids = await seed(store, children=2)
    result = await cancel_run(root, tenant_id=TENANT, state_store=store)
    assert result.status == RunStatus.CANCELLED
    assert [item["run_id"] for item in result.meta["cascade"]] == [*kids, root]
    for kid in kids:
        assert (await store.get_run(kid)).status == "cancelled"


async def test_cascade_false_leaves_legacy_children(store: Any) -> None:
    root, kids = await seed(store, children=1)
    result = await cancel_run(root, tenant_id=TENANT, state_store=store, cascade=False)
    assert "cascade" not in result.meta
    assert (await store.get_run(kids[0])).status == "paused"


async def test_result_matches_the_agent_result(store: Any) -> None:
    def shape(result: Any, ids: list[str]) -> Any:
        names = {rid: f"n{i}" for i, rid in enumerate(ids)}
        meta = dict(result.meta)
        meta["cascade"] = [
            {**item, "run_id": names[item["run_id"]]} for item in meta.get("cascade", [])
        ]
        return (result.status, result.usage, result.recovery, result.stop_reason, meta)

    module_root, module_kids = await seed(store, children=2)
    agent_root, agent_kids = await seed(store, children=2)
    agent = Agent(name="cancel", prompt="p", state_store=store, strict_tenancy=False)
    via_module = await cancel_run(module_root, tenant_id=TENANT, state_store=store)
    via_agent = await agent.cancel_run(agent_root, tenant_id=TENANT, cascade=True)
    assert shape(via_module, [module_root, *module_kids]) == shape(
        via_agent, [agent_root, *agent_kids]
    )


async def test_running_run_gets_a_durable_request_only(store: Any) -> None:
    root, _ = await seed(store, paused=False)
    result = await cancel_run(root, tenant_id=TENANT, state_store=store)
    assert result.status == RunStatus.RUNNING
    assert result.meta["cancel_requested"] is True
    assert (await store.get_run(root)).status == "running"


async def test_tenant_none_reaches_only_untenanted_runs(store: Any) -> None:
    root, _ = await seed(store)
    with pytest.raises(RunNotFoundError):
        await cancel_run(root, tenant_id=None, state_store=store)
    untenanted = generate_ulid()
    await store.create_run(untenanted, "cancel")
    await store.pause_run(untenanted, status="paused", pause_data={})
    result = await cancel_run(untenanted, tenant_id=None, state_store=store)
    assert result.status == RunStatus.CANCELLED


async def test_foreign_tenant_is_not_found(store: Any) -> None:
    root, _ = await seed(store)
    with pytest.raises(RunNotFoundError):
        await cancel_run(root, tenant_id="other", state_store=store)
    assert not (await store.get_run(root)).cancel_requested


async def test_managed_child_is_refused(store: Any) -> None:
    root, child, _ = await chain(store)
    with pytest.raises(ManagedChildControlError):
        await cancel_run(child, tenant_id=TENANT, state_store=store)
    assert not (await store.get_run(root)).cancel_requested


async def test_managed_root_cancels_its_workflow(store: Any) -> None:
    root, child, leaf = await chain(store)
    result = await cancel_run(root, tenant_id=TENANT, state_store=store, cascade=False)
    assert {item["run_id"] for item in result.meta["cascade"]} == {root, child, leaf}
    for rid in (root, child, leaf):
        row = await store.get_run(rid)
        assert row.cancel_requested or row.status == "cancelled"


async def test_claimed_store_is_normalized_to_the_unclaimed_store(store: Any) -> None:
    root, _ = await seed(store)
    stale = claimed_store(store, root, 999)
    result = await cancel_run(root, tenant_id=TENANT, state_store=stale)
    assert result.status == RunStatus.CANCELLED
    row = await store.get_run(root)
    assert row.status == "cancelled"


async def test_callers_store_is_never_closed(store: Any, monkeypatch: Any) -> None:
    root, _ = await seed(store)
    disposed = track_disposal(monkeypatch)
    closed: list[bool] = []
    monkeypatch.setattr(store, "close", lambda *a, **k: closed.append(True))
    await cancel_run(root, tenant_id=TENANT, state_store=store)
    assert (disposed, closed) == ([], [])
    assert (await store.get_run(root)).status == "cancelled"


@pytest.mark.parametrize("engine", ["sqlite"], indirect=True)
@pytest.mark.parametrize("missing", [False, True])
async def test_database_url_store_is_created_and_disposed(
    engine: Any, tmp_path: Path, monkeypatch: Any, missing: bool
) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'cancel.db'}"
    seeding, seed_engine = await sweep._create_temp_store(url)
    root, _ = await seed(seeding)
    await seed_engine.dispose()
    created: list[int] = []
    real = sweep._create_temp_store

    async def tracked(*args: Any, **kwargs: Any) -> Any:
        store, temp = await real(*args, **kwargs)
        created.append(id(temp))
        return store, temp

    monkeypatch.setattr(sweep, "_create_temp_store", tracked)
    disposed = track_disposal(monkeypatch)
    target = generate_ulid() if missing else root
    if missing:
        with pytest.raises(RunNotFoundError):
            await cancel_run(target, tenant_id=TENANT, database_url=url)
    else:
        result = await cancel_run(target, tenant_id=TENANT, database_url=url)
        assert result.status == RunStatus.CANCELLED
    assert len(created) == 1
    assert disposed == created


async def test_same_bound_retry_does_not_continue(store: Any) -> None:
    """There is no cursor: only a larger ``max_nodes`` reaches the rest of the tree."""
    root, kids = await seed(store, children=3)
    first = await cancel_run(root, tenant_id=TENANT, state_store=store, max_nodes=2)
    second = await cancel_run(root, tenant_id=TENANT, state_store=store, max_nodes=2)
    assert first.meta["cascade_truncated"] and second.meta["cascade_truncated"]
    visited = [[item["run_id"] for item in r.meta["cascade"]] for r in (first, second)]
    assert visited[0] == visited[1] == [kids[0], root]
    for kid in kids[1:]:
        row = await store.get_run(kid)
        assert (row.status, row.cancel_requested) == ("paused", False)
    wider = await cancel_run(root, tenant_id=TENANT, state_store=store, max_nodes=4)
    assert wider.meta["cascade_truncated"] is False
    for kid in kids:
        assert (await store.get_run(kid)).status == "cancelled"
