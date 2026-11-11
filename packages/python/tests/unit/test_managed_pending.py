"""Managed subagents (0.7, T3): the root-facing pending view and its bounded walk.

Pure read logic over a custom store (no SQL session): the walk follows each
waiting parent's exact bound child, validates every hop and never guesses.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from dendrux.errors import ManagedStateError, RunNotFoundError
from dendrux.runtime._pending import (
    MAX_MANAGED_WALK,
    PendingPause,
    _PendingReader,
    managed_branch,
)
from dendrux.runtime.state import UNSCOPED
from dendrux.types import generate_ulid

TENANT = "t1"


def pause_shape(pending: PendingPause) -> set[str]:
    return set(pending.to_dict())


def test_origin_fields_default_to_none_and_keep_the_event_shape() -> None:
    own = PendingPause("R", "waiting_approval", "approval", "P")
    assert (own.agent_name, own.agent_path, own.origin_run_id) == (None, None, None)
    assert not {"agent_name", "agent_path", "origin_run_id"} & pause_shape(own)

    routed = PendingPause(
        "R",
        "waiting_approval",
        "approval",
        "P",
        agent_name="auditor",
        agent_path=["lead", "reviewer", "auditor"],
        origin_run_id="L",
    )
    shown = routed.to_dict()
    assert shown["run_id"] == "R" and shown["pause_id"] == "P"
    assert shown["agent_name"] == "auditor"
    assert shown["agent_path"] == ["lead", "reviewer", "auditor"]
    assert shown["origin_run_id"] == "L"
    assert pause_shape(routed) - pause_shape(own) == {"agent_name", "agent_path", "origin_run_id"}


class FakeStore:
    """A custom store: runs and checkpoints by id, counting reads."""

    def __init__(self) -> None:
        self.runs: dict[str, Any] = {}
        self.states: dict[str, dict[str, Any]] = {}
        self.pauses: dict[str, dict[str, Any]] = {}
        self.reads = 0

    async def get_run(self, run_id: str) -> Any:
        self.reads += 1
        return self.runs.get(run_id)

    async def get_loop_state(self, run_id: str) -> Any:
        self.reads += 1
        return self.states.get(run_id)

    async def get_pause_state(self, run_id: str) -> Any:
        self.reads += 1
        return self.pauses.get(run_id)

    def add(
        self,
        name: str,
        *,
        status: str,
        parent: str | None = None,
        tenant: str | None = TENANT,
        root: str | None = None,
        path: tuple[str, ...] = (),
        call_id: str | None = None,
    ) -> str:
        run_id = generate_ulid()
        self.runs[run_id] = SimpleNamespace(
            id=run_id,
            status=status,
            tenant_id=tenant,
            parent_run_id=parent,
            agent_name=name,
        )
        header: dict[str, Any] = {"v": 1, "root_run_id": root or run_id, "path": list(path)}
        if parent is not None:
            header["invocation"] = {"parent_run_id": parent, "tool_call_id": call_id}
        self.states[run_id] = {"managed": header}
        return run_id

    def bind(self, parent: str, child: str, call_id: str) -> None:
        self.runs[parent].status = "paused"
        self.states[parent]["managed_wait"] = {
            "v": 1,
            "tool_call_id": call_id,
            "child_run_id": child,
            "pause_id": generate_ulid(),
        }
        self.pauses[parent] = {"pause_reason": "managed_wait", "pause_id": generate_ulid()}


def chain(store: FakeStore, depth: int, leaf_status: str = "running") -> list[str]:
    """root -> c1 -> ... -> c{depth}; every parent waits on its child."""
    root = store.add("lead", status="running")
    ids = [root]
    path: tuple[str, ...] = ()
    for level in range(1, depth + 1):
        path = (*path, f"a{level}")
        call_id = f"call{level}"
        child = store.add(
            f"a{level}",
            status=leaf_status if level == depth else "running",
            parent=ids[-1],
            root=root,
            path=path,
            call_id=call_id,
        )
        store.bind(ids[-1], child, call_id)
        ids.append(child)
    return ids


async def test_walk_follows_bound_children_to_the_leaf() -> None:
    store = FakeStore()
    ids = chain(store, 3)
    hops, start, leaf = await managed_branch(store, ids[0], TENANT)
    assert [h.run_id for h in hops] == ids
    assert start == () and leaf == ("a1", "a2", "a3")
    # Two reads per hop on a custom store (run, checkpoint); nothing else.
    assert store.reads == 2 * len(ids)


async def test_walk_from_a_middle_run_is_relative_to_it() -> None:
    store = FakeStore()
    ids = chain(store, 2)
    hops, start, leaf = await managed_branch(store, ids[1], TENANT)
    assert [h.run_id for h in hops] == ids[1:]
    assert start == ("a1",) and leaf == ("a1", "a2")


async def test_a_run_that_is_not_waiting_is_its_own_branch() -> None:
    store = FakeStore()
    run_id = store.add("solo", status="waiting_approval")
    hops, start, leaf = await managed_branch(store, run_id, TENANT)
    assert [h.run_id for h in hops] == [run_id] and start == leaf == ()


async def test_foreign_scope_is_not_found() -> None:
    store = FakeStore()
    ids = chain(store, 1)
    with pytest.raises(RunNotFoundError):
        await managed_branch(store, ids[0], "other")
    with pytest.raises(RunNotFoundError):
        await managed_branch(store, "01ZZZZZZZZZZZZZZZZZZZZZZZZ", UNSCOPED)


def _corrupt(store: FakeStore, ids: list[str], how: str) -> None:
    root, child = ids
    if how == "child_missing":
        del store.runs[child]
    elif how == "foreign_tenant":
        store.runs[child].tenant_id = "other"
    elif how == "wrong_parent":
        store.runs[child].parent_run_id = generate_ulid()
    elif how == "other_root":
        store.states[child]["managed"]["root_run_id"] = generate_ulid()
    elif how == "other_call":
        store.states[child]["managed"]["invocation"]["tool_call_id"] = "other"
    elif how == "other_path":
        store.states[child]["managed"]["path"] = ["x", "y"]
    elif how == "no_header":
        del store.states[child]["managed"]
    elif how == "bad_binding":
        store.states[root]["managed_wait"]["child_run_id"] = "not-a-ulid"
    elif how == "root_header":
        del store.states[root]["managed"]


@pytest.mark.parametrize(
    ("how", "reason"),
    [
        ("child_missing", "child_missing"),
        ("foreign_tenant", "child_mismatch"),
        ("wrong_parent", "child_mismatch"),
        ("other_root", "child_mismatch"),
        ("other_call", "child_mismatch"),
        ("other_path", "child_mismatch"),
        ("no_header", "header_invalid"),
        ("bad_binding", "wait_invalid"),
        ("root_header", "header_invalid"),
    ],
)
async def test_every_hop_is_validated_and_never_guessed(how: str, reason: str) -> None:
    store = FakeStore()
    ids = chain(store, 1)
    _corrupt(store, ids, how)
    with pytest.raises(ManagedStateError) as caught:
        await managed_branch(store, ids[0], TENANT)
    assert caught.value.reason == reason
    assert caught.value.code == "managed_state_invalid"


async def test_walk_is_bounded() -> None:
    store = FakeStore()
    ids = chain(store, MAX_MANAGED_WALK + 1)
    with pytest.raises(ManagedStateError) as caught:
        await managed_branch(store, ids[0], TENANT)
    assert caught.value.reason == "binding_conflict"
    assert store.reads <= 2 * (MAX_MANAGED_WALK + 1)


class SummaryStore(FakeStore):
    """A custom store whose full pause state stands in for the summary read."""


async def test_root_reads_running_while_its_subagent_runs_or_awaits_adoption() -> None:
    for leaf_status in ("running", "success", "error"):
        store = SummaryStore()
        ids = chain(store, 2, leaf_status=leaf_status)
        view = await _PendingReader(store).facing(ids[0], tenant_id=TENANT)
        assert (view.status, view.pending, view.managed) == ("running", None, True)


def test_wire_fallback_keeps_the_managed_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Over the display budget the wire hides arguments but keeps the pause identity."""
    from dendrux import _wire
    from dendrux.runtime._pending import PendingToolCall

    pending = PendingPause(
        "R",
        "waiting_approval",
        "approval",
        "P",
        tool_calls=[PendingToolCall("c1", "publish", "server", {"text": "SENTINEL-arg"})],
        agent_name="auditor",
        agent_path=["lead", "reviewer", "auditor"],
        origin_run_id="L",
    )
    monkeypatch.setattr(_wire, "_redacted", lambda data, mapping: None)  # budget exceeded
    shown = _wire._pending_wire(pending, None)
    assert (shown["run_id"], shown["pause_id"]) == ("R", "P")
    assert shown["pending_tool_calls"][0]["params"] is None
    assert "SENTINEL-arg" not in repr(shown)
    assert shown["agent_name"] == "auditor"
    assert shown["agent_path"] == ["lead", "reviewer", "auditor"]
    assert shown["origin_run_id"] == "L"
    own = PendingPause("R", "waiting_approval", "approval", "P")
    assert "origin_run_id" not in _wire._pending_wire(own, None)
