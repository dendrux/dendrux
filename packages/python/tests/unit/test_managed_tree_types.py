"""The 0.8 workflow-tree records and the Protocols later tasks implement (spec §5.2, §6, §8.2)."""

from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path
from typing import Any, cast

import pytest

from dendrux.errors import ManagedStateError
from dendrux.runtime import _managed_tree as tree_module
from dendrux.runtime._fanout_adoption_state import FanoutBinding
from dendrux.runtime._managed_tree import (
    MAX_FANOUT,
    AdoptFanout,
    AttachPendings,
    BranchSpec,
    FanoutInvocation,
    FindLeaf,
    LeafNode,
    SettleParent,
    WaitNode,
    WalkTree,
    WorkflowTree,
)
from dendrux.types import PauseState, ToolCall, UsageStats

AGENT = cast("Any", object())
ROOT = "01JC0000000000000000000001"
W = "01JC0000000000000000000002"
OTHER = "01JC0000000000000000000003"


def _call(name: str) -> ToolCall:
    return ToolCall(name=name, params={}, provider_tool_call_id=f"p-{name}")


def _leaf(run_id: str, *, status: str = "running", pause_id: str | None = None) -> LeafNode:
    return LeafNode(
        run_id=run_id,
        agent=AGENT,
        agent_path=(run_id,),
        status=status,
        pause_id=pause_id,
        call=_call(run_id),
    )


def _wait(
    run_id: str,
    *children: Any,
    version: int = 4,
    root: bool = False,
    root_run_id: str = ROOT,
    tenant_id: str | None = "t1",
) -> WaitNode:
    return WaitNode(
        run_id=run_id,
        root_run_id=root_run_id,
        tenant_id=tenant_id,
        agent=AGENT,
        agent_path=() if root else (run_id,),
        pause_id=f"gen-{run_id}",
        version=cast("Any", version),
        iteration=1,
        children=children,
        call=None if root else _call(run_id),
    )


def _branch(name: str, child_run_id: str | None = None) -> BranchSpec:
    return BranchSpec(
        call=_call(name), child=AGENT, child_path=(name,), child_run_id=child_run_id or f"r-{name}"
    )


def _invocation(*branches: BranchSpec, **overrides: Any) -> FanoutInvocation:
    fields: dict[str, Any] = {
        "store": object(),
        "tenant_id": "t1",
        "root_run_id": ROOT,
        "parent_run_id": ROOT,
        "parent_epoch": 3,
        "parent_level": 0,
        "parent_agent": AGENT,
        "wait": cast("PauseState", object()),
        "iteration": 2,
        "parent_usage": UsageStats(),
        "max_depth": None,
        "branches": branches,
    }
    return FanoutInvocation(**{**fields, **overrides})


# --- records are frozen ------------------------------------------------------

RECORDS = (WaitNode, LeafNode, WorkflowTree, BranchSpec, FanoutInvocation)


@pytest.mark.parametrize("record", RECORDS, ids=lambda r: r.__name__)
def test_records_are_frozen_dataclasses(record: type) -> None:
    assert dataclasses.is_dataclass(record)
    assert record.__dataclass_params__.frozen  # type: ignore[attr-defined]


def test_a_leaf_cannot_be_reassigned() -> None:
    leaf = _leaf("a")
    with pytest.raises(dataclasses.FrozenInstanceError):
        leaf.status = "success"  # type: ignore[misc]


# --- LeafNode ----------------------------------------------------------------

LEAF_STATES = {
    "running": ("running", None, False, False),
    "paused externally": ("waiting_approval", "P1", False, True),
    "success": ("success", None, True, False),
    "error": ("error", None, True, False),
    "cancelled": ("cancelled", None, True, False),
    "max_iterations": ("max_iterations", None, True, False),
}


@pytest.mark.parametrize(
    ("status", "pause_id", "terminal", "paused"), LEAF_STATES.values(), ids=LEAF_STATES.keys()
)
def test_leaf_state_flags(status: str, pause_id: str | None, terminal: bool, paused: bool) -> None:
    leaf = _leaf("a", status=status, pause_id=pause_id)
    assert (leaf.is_terminal, leaf.is_paused) == (terminal, paused)


def test_a_terminal_leaf_with_a_stale_pause_id_is_not_paused() -> None:
    assert not _leaf("a", status="success", pause_id="P1").is_paused


# --- WaitNode ----------------------------------------------------------------


def test_wait_exposes_its_bindings_in_call_order() -> None:
    a, b, c = _leaf("a"), _leaf("b"), _leaf("c")
    wait = _wait("w", a, b, c)
    assert wait.bindings == (
        FanoutBinding(a.call.id, "a"),  # type: ignore[union-attr]
        FanoutBinding(b.call.id, "b"),  # type: ignore[union-attr]
        FanoutBinding(c.call.id, "c"),  # type: ignore[union-attr]
    )


def test_wait_children_are_copied_into_a_tuple() -> None:
    children = [_leaf("a"), _leaf("b")]
    wait = WaitNode(
        run_id="w",
        root_run_id=ROOT,
        tenant_id=None,
        agent=AGENT,
        agent_path=(),
        pause_id="g",
        version=4,
        iteration=0,
        children=cast("Any", children),
    )
    children.append(_leaf("c"))
    assert isinstance(wait.children, tuple)
    assert len(wait.children) == 2


def test_a_v3_wait_has_exactly_one_child() -> None:
    assert len(_wait("w", _leaf("a"), version=3).children) == 1


def _shared_call() -> tuple[LeafNode, LeafNode]:
    call = _call("same")
    return (
        dataclasses.replace(_leaf("a"), call=call),
        dataclasses.replace(_leaf("b"), call=call),
    )


BAD_WAITS = {
    "v3 with two children": lambda: _wait(W, _leaf("a"), _leaf("b"), version=3),
    "v4 with one child": lambda: _wait(W, _leaf("a")),
    "v4 with no children": lambda: _wait(W),
    "v4 above the cap": lambda: _wait(W, *(_leaf(f"c{i}") for i in range(MAX_FANOUT + 1))),
    "unknown version": lambda: _wait(W, _leaf("a"), _leaf("b"), version=5),
    "two children with one run id": lambda: _wait(W, _leaf("a"), _leaf("a")),
    "two children answering one call": lambda: _wait(W, *_shared_call()),
    "child without a call": lambda: _wait(
        W, _leaf("a"), dataclasses.replace(_leaf("b"), call=None)
    ),
}


@pytest.mark.parametrize("build", BAD_WAITS.values(), ids=BAD_WAITS.keys())
def test_a_malformed_wait_is_invalid_managed_state(build: Any) -> None:
    with pytest.raises(ManagedStateError) as raised:
        build()
    assert raised.value.reason == "binding_conflict"


def test_a_wait_at_the_cap_is_accepted() -> None:
    wait = _wait("w", *(_leaf(f"c{i}") for i in range(MAX_FANOUT)))
    assert len(wait.children) == MAX_FANOUT == 32


# --- WorkflowTree --------------------------------------------------------------


def _nested() -> WorkflowTree:
    """ROOT waits on [a, w2, d]; w2 waits on [b (paused), c]; d is paused."""
    inner = _wait("w2", _leaf("b", status="waiting_approval", pause_id="P-b"), _leaf("c"))
    root = _wait(
        ROOT,
        _leaf("a", status="success"),
        inner,
        _leaf("d", status="waiting_human_input", pause_id="P-d"),
        root=True,
    )
    return WorkflowTree(root_run_id=ROOT, tenant_id="t1", root=root)


def test_leaves_are_depth_first_with_siblings_in_call_order() -> None:
    assert [leaf.run_id for leaf in _nested().leaves()] == ["a", "b", "c", "d"]


def test_paused_leaves_keep_the_same_order() -> None:
    assert [leaf.pause_id for leaf in _nested().paused_leaves()] == ["P-b", "P-d"]


def test_ancestors_run_from_the_immediate_parent_to_the_root() -> None:
    tree = _nested()
    assert [w.run_id for w in tree.ancestors("b")] == ["w2", ROOT]
    assert [w.run_id for w in tree.ancestors("d")] == [ROOT]
    assert [w.run_id for w in tree.ancestors("w2")] == [ROOT]
    assert tree.ancestors(ROOT) == ()


def test_ancestors_of_an_unknown_run_is_an_error() -> None:
    with pytest.raises(LookupError):
        _nested().ancestors("nope")


def test_a_single_leaf_tree_is_valid() -> None:
    root = LeafNode(run_id=ROOT, agent=AGENT, agent_path=(), status="running")
    tree = WorkflowTree(root_run_id=ROOT, tenant_id=None, root=root)
    assert tree.leaves() == (root,)
    assert tree.ancestors(ROOT) == ()


def test_a_deep_chain_does_not_recurse() -> None:
    node: Any = _leaf("leaf")
    for depth in range(3000):
        node = _wait(f"w{depth}", node, version=3)
    node = dataclasses.replace(node, run_id=ROOT, call=None)
    tree = WorkflowTree(root_run_id=ROOT, tenant_id="t1", root=node)
    assert [leaf.run_id for leaf in tree.leaves()] == ["leaf"]
    assert len(tree.ancestors("leaf")) == 3000


BAD_TREES = {
    "root id differs from the root node": lambda: WorkflowTree(
        root_run_id=OTHER,
        tenant_id="t1",
        root=_wait(ROOT, _leaf("a"), _leaf("b"), root=True, root_run_id=OTHER),
    ),
    "root node answers a call": lambda: WorkflowTree(
        root_run_id=OTHER, tenant_id=None, root=_leaf(OTHER)
    ),
    "a nested wait trusting another root": lambda: WorkflowTree(
        root_run_id=ROOT,
        tenant_id="t1",
        root=_wait(
            ROOT, _leaf("a"), _wait("w2", _leaf("b"), _leaf("c"), root_run_id=OTHER), root=True
        ),
    ),
    "a nested wait in another tenant": lambda: WorkflowTree(
        root_run_id=ROOT,
        tenant_id="t1",
        root=_wait(
            ROOT, _leaf("a"), _wait("w2", _leaf("b"), _leaf("c"), tenant_id="t2"), root=True
        ),
    ),
    "an untenanted wait in a tenant's tree": lambda: WorkflowTree(
        root_run_id=ROOT,
        tenant_id="t1",
        root=_wait(ROOT, _leaf("a"), _leaf("b"), root=True, tenant_id=None),
    ),
    "one run in two places": lambda: WorkflowTree(
        root_run_id=ROOT,
        tenant_id="t1",
        root=_wait(ROOT, _leaf("a"), _wait("w2", _leaf("a"), _leaf("b")), root=True),
    ),
}


@pytest.mark.parametrize("build", BAD_TREES.values(), ids=BAD_TREES.keys())
def test_a_malformed_tree_is_invalid_managed_state(build: Any) -> None:
    with pytest.raises(ManagedStateError) as raised:
        build()
    assert raised.value.reason == "binding_conflict"


# --- BranchSpec and FanoutInvocation -------------------------------------------


def test_invocation_carries_the_shared_parent_fields_of_a_managed_invocation() -> None:
    from dendrux.runtime._managed import ManagedInvocation

    single = {f.name for f in dataclasses.fields(ManagedInvocation)}
    fanout = {f.name for f in dataclasses.fields(FanoutInvocation)}
    per_branch = {"call", "child", "child_path", "child_run_id"}
    assert {f.name for f in dataclasses.fields(BranchSpec)} == per_branch
    assert fanout == (single - per_branch) | {"branches"}


def test_invocation_exposes_call_ids_and_bindings_in_call_order() -> None:
    x, y = _branch("x"), _branch("y")
    invocation = _invocation(x, y)
    assert invocation.call_ids == (x.call.id, y.call.id)
    assert invocation.bindings == (
        FanoutBinding(x.call.id, "r-x"),
        FanoutBinding(y.call.id, "r-y"),
    )
    assert invocation.unpriced_accepted == 0


def test_invocation_branches_are_copied_into_a_tuple() -> None:
    branches = [_branch("x"), _branch("y")]
    invocation = _invocation(**{"branches": branches})
    branches.append(_branch("z"))
    assert isinstance(invocation.branches, tuple)
    assert len(invocation.branches) == 2


def _same_call_twice() -> FanoutInvocation:
    x = _branch("x")
    return _invocation(x, dataclasses.replace(x, child_run_id="r-other"))


BAD_INVOCATIONS = {
    "no branches": lambda: _invocation(),
    "one branch (N=1 keeps the 0.7 path)": lambda: _invocation(_branch("x")),
    "above the cap": lambda: _invocation(*(_branch(f"b{i}") for i in range(MAX_FANOUT + 1))),
    "one call twice": _same_call_twice,
    "one child run id twice": lambda: _invocation(_branch("x", "r1"), _branch("y", "r1")),
}


@pytest.mark.parametrize("build", BAD_INVOCATIONS.values(), ids=BAD_INVOCATIONS.keys())
def test_a_malformed_invocation_is_refused(build: Any) -> None:
    with pytest.raises(ValueError, match="fan-out"):
        build()


def test_an_invocation_at_the_cap_is_accepted() -> None:
    assert len(_invocation(*(_branch(f"b{i}") for i in range(MAX_FANOUT))).branches) == 32


# --- Protocols -------------------------------------------------------------------

SIGNATURES = {
    WalkTree: (
        True,
        "(self, root_agent: 'Agent', store: 'StateStore', root_run_id: 'str', *, "
        "tenant_id: 'TenantScope') -> 'WorkflowTree'",
    ),
    FindLeaf: (False, "(self, tree: 'WorkflowTree', pause_id: 'str') -> 'LeafNode'"),
    SettleParent: (
        True,
        "(self, store: 'StateStore', node: 'WaitNode') -> 'ClaimResult | None'",
    ),
    AdoptFanout: (
        True,
        "(self, store: 'StateStore', node: 'WaitNode', claim: 'ClaimResult') -> 'ClaimResult'",
    ),
    AttachPendings: (
        False,
        "(self, result: 'RunResult', pendings: 'Sequence[PendingPause]') -> 'RunResult'",
    ),
}


@pytest.mark.parametrize("protocol", SIGNATURES, ids=lambda p: p.__name__)
def test_protocol_signatures_are_pinned(protocol: Any) -> None:
    is_async, expected = SIGNATURES[protocol]
    assert getattr(protocol, "_is_protocol", False)
    assert inspect.iscoroutinefunction(protocol.__call__) is is_async
    assert str(inspect.signature(protocol.__call__)) == expected


def test_protocol_names_cover_the_five_spec_functions() -> None:
    expected = {
        "walk_tree": WalkTree,
        "find_leaf": FindLeaf,
        "settle_parent": SettleParent,
        "adopt_fanout": AdoptFanout,
        "attach_pendings": AttachPendings,
    }
    assert expected == tree_module.PROTOCOLS


async def test_a_fake_settle_parent_can_stand_in_for_the_protocol() -> None:
    seen: list[str] = []

    async def fake(store: Any, node: WaitNode) -> None:
        seen.append(node.run_id)

    settle: SettleParent = fake
    tree = _nested()
    for wait in tree.ancestors("b"):
        assert await settle(object(), wait) is None  # type: ignore[arg-type]
    assert seen == ["w2", ROOT]


class _ChildStore:
    """A store stand-in holding child rows under (parent, tenant), as §5.4 scopes them."""

    def __init__(self, rows: dict[tuple[str, str | None], dict[str, dict[str, str]]]) -> None:
        self.rows = rows
        self.reads: list[tuple[str, tuple[str, ...], str | None]] = []

    async def get_managed_children(
        self, parent_run_id: str, child_run_ids: list[str], *, tenant_id: str | None
    ) -> list[dict[str, str]]:
        self.reads.append((parent_run_id, tuple(child_run_ids), tenant_id))
        scoped = self.rows.get((parent_run_id, tenant_id), {})
        return [scoped[run_id] for run_id in child_run_ids if run_id in scoped]


async def _settle(store: Any, node: WaitNode) -> None:
    """Module-level, as the real one is: everything it trusts comes from its arguments."""
    ids = [binding.child_run_id for binding in node.bindings]
    rows = await store.get_managed_children(node.run_id, ids, tenant_id=node.tenant_id)
    if len(rows) != len(ids):
        raise ManagedStateError(reason="child_missing", run_id=node.run_id)
    if any(row["root_run_id"] != node.root_run_id for row in rows):
        raise ManagedStateError(reason="child_mismatch", run_id=node.run_id)


def _nested_wait() -> WaitNode:
    tree = _nested()
    (inner,) = (w for w in tree.ancestors("b") if w.run_id == "w2")
    assert (inner.root_run_id, inner.tenant_id) == (ROOT, "t1")
    return dataclasses.replace(inner, run_id=W)


async def test_a_nested_wait_carries_the_tenant_scope_for_its_child_read() -> None:
    settle: SettleParent = _settle
    node = _nested_wait()
    rows = {"b": {"root_run_id": ROOT}, "c": {"root_run_id": ROOT}}
    store = _ChildStore({(W, "t1"): rows})
    assert await settle(cast("Any", store), node) is None
    assert store.reads == [(W, ("b", "c"), "t1")]
    other_tenant = _ChildStore({(W, "t2"): rows})
    with pytest.raises(ManagedStateError) as raised:
        await settle(cast("Any", other_tenant), node)
    assert raised.value.reason == "child_missing"


async def test_a_nested_wait_carries_the_expected_root_for_its_children() -> None:
    settle: SettleParent = _settle
    rows = {"b": {"root_run_id": ROOT}, "c": {"root_run_id": OTHER}}
    with pytest.raises(ManagedStateError) as raised:
        await settle(cast("Any", _ChildStore({(W, "t1"): rows})), _nested_wait())
    assert raised.value.reason == "child_mismatch"


# --- the interface module stays off the live path ---------------------------------


def test_runtime_imports_stay_below_the_driver() -> None:
    """Agent, store and driver types are annotations only: nothing live is pulled in."""
    source = Path(tree_module.__file__).read_text()
    runtime_imports = {
        node.module
        for node in ast.parse(source).body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert runtime_imports == {
        "__future__",
        "dataclasses",
        "typing",
        "dendrux._lifecycle",
        "dendrux._sentinel",
        "dendrux.errors",
        "dendrux.runtime._fanout_adoption_state",
    }


def test_unsupplied_tree_policy_fails_closed():
    from dendrux.runtime import _managed_route
    from dendrux.runtime._managed_fanout_store import _UNPROJECTED

    for cls in (WaitNode, LeafNode, _managed_route._Seen):
        assert (
            next(f for f in dataclasses.fields(cls) if f.name == "governance").default
            is _UNPROJECTED
        )
    wait = _wait(ROOT, _leaf(W), version=3, root=True)
    with pytest.raises(ManagedStateError) as refused:
        _managed_route._check_wait_governance([wait])
    assert refused.value.reason == "wait_invalid"
