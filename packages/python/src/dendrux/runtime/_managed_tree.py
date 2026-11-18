"""The shape of a managed workflow with fan-out, and the steps that act on it (0.8).

Interface only: the frozen records a tree walk and a fan-out handoff produce,
and the call signatures of the five steps later modules implement. Nothing here
reads a store or runs a child, and no live path imports this module yet.

A workflow is a tree of runs. A :class:`WaitNode` is a parent parked on the
children one model turn started (one child for a v3 wait, 2..32 for a v4
fan-out); a :class:`LeafNode` is any run that is not parked on children. The
records are shallow-frozen: agents, tool calls and the wait are shared
references that must not be mutated through them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, Protocol

from dendrux._lifecycle import TERMINAL_STATUSES
from dendrux.errors import ManagedStateError
from dendrux.runtime._fanout_adoption_state import MAX_FANOUT, MIN_FANOUT, FanoutBinding

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from dendrux.agent import Agent
    from dendrux.runtime._pending import PendingPause
    from dendrux.runtime.state import ClaimResult, StateStore, TenantScope
    from dendrux.types import PauseState, RunResult, ToolCall, UsageStats

_TERMINAL = frozenset(status.value for status in TERMINAL_STATUSES)
_CONFLICT = "binding_conflict"


# ---------------------------------------------------------------------------
# The workflow tree (built by a walk from persisted state)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeafNode:
    """A run that is not parked on children: running, externally paused or terminal.

    ``call`` is the parent's tool call this run answers (None for the root).
    ``pause_id`` and ``pause_reason`` are set only while the run is at an
    external pause; ``loop`` is the run's saved loop marker.
    """

    run_id: str
    agent: Agent
    agent_path: tuple[str, ...]
    status: str
    loop: str | None = None
    pause_id: str | None = None
    pause_reason: str | None = None
    call: ToolCall | None = None

    @property
    def is_terminal(self) -> bool:
        """The run has ended; it is fan-in material and is never driven again."""
        return self.status in _TERMINAL

    @property
    def is_paused(self) -> bool:
        """The run waits on an external answer addressed by ``pause_id``."""
        return self.pause_id is not None and not self.is_terminal


@dataclass(frozen=True)
class WaitNode:
    """A parent parked on the children one model turn started.

    ``pause_id`` is the wait generation an adoption claims. ``children`` are in
    the batch's call order and each carries the call it answers. ``version`` is
    the persisted wait format: 3 binds exactly one child, 4 binds 2..32.

    ``root_run_id`` and ``tenant_id`` are the workflow root and the root's
    resolved tenant, as the walk (or the live driver) validated them. A step
    handed only this node scopes its child read with ``tenant_id`` and checks
    each child's header against ``root_run_id``; it never takes either from a
    child's own row.

    Raises:
        ManagedStateError: ``binding_conflict`` when the children do not fit
            the version, repeat a run or a call, or lack their call.
    """

    run_id: str
    root_run_id: str
    tenant_id: str | None
    agent: Agent
    agent_path: tuple[str, ...]
    pause_id: str
    version: Literal[3, 4]
    iteration: int
    children: tuple[WaitNode | LeafNode, ...]
    call: ToolCall | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "children", tuple(self.children))
        count = len(self.children)
        fits = (
            count == 1
            if self.version == 3
            else self.version == 4 and MIN_FANOUT <= count <= MAX_FANOUT
        )
        calls = {child.call.id for child in self.children if child.call is not None}
        runs = {child.run_id for child in self.children}
        if not fits or len(calls) != count or len(runs) != count:
            raise ManagedStateError(reason=_CONFLICT, run_id=self.run_id)

    @property
    def bindings(self) -> tuple[FanoutBinding, ...]:
        """``(tool_call_id, child_run_id)`` per child, in call order."""
        return tuple(
            FanoutBinding(child.call.id, child.run_id)
            for child in self.children
            if child.call is not None
        )


@dataclass(frozen=True)
class WorkflowTree:
    """One consistent view of a root and every run bound beneath it.

    Raises:
        ManagedStateError: ``binding_conflict`` when the root node is not the
            root run, answers a call, a run appears twice in the tree, or a
            wait carries another root or tenant than the tree's.
    """

    root_run_id: str
    tenant_id: str | None
    root: WaitNode | LeafNode

    def __post_init__(self) -> None:
        nodes = [node for node, _ in _descend(self.root)]
        runs = {node.run_id for node in nodes}
        scope = (self.root_run_id, self.tenant_id)
        rooted = self.root.run_id == self.root_run_id and self.root.call is None
        scoped = all(
            (node.root_run_id, node.tenant_id) == scope
            for node in nodes
            if isinstance(node, WaitNode)
        )
        if not rooted or not scoped or len(runs) != len(nodes):
            raise ManagedStateError(reason=_CONFLICT, run_id=self.root_run_id)

    def leaves(self) -> tuple[LeafNode, ...]:
        """Every leaf, depth-first, with siblings in their parent's call order."""
        return tuple(node for node, _ in _descend(self.root) if isinstance(node, LeafNode))

    def paused_leaves(self) -> tuple[LeafNode, ...]:
        """The externally paused leaves, in :meth:`leaves` order."""
        return tuple(leaf for leaf in self.leaves() if leaf.is_paused)

    def ancestors(self, run_id: str) -> tuple[WaitNode, ...]:
        """The waits above ``run_id``, from its immediate parent up to the root.

        Raises:
            LookupError: ``run_id`` is not in this tree.
        """
        for node, above in _descend(self.root):
            if node.run_id == run_id:
                return above
        raise LookupError("run is not in this workflow tree")


def _descend(
    root: WaitNode | LeafNode,
) -> Iterable[tuple[WaitNode | LeafNode, tuple[WaitNode, ...]]]:
    """Each node with its ancestors (nearest first), depth-first in call order.

    Iterative: a deep chain of single-child waits must not hit the recursion limit.
    """
    stack: list[tuple[WaitNode | LeafNode, tuple[WaitNode, ...]]] = [(root, ())]
    while stack:
        node, above = stack.pop()
        yield node, above
        if isinstance(node, WaitNode):
            stack.extend((child, (node, *above)) for child in reversed(node.children))


# ---------------------------------------------------------------------------
# The fan-out handoff (built in memory by the parent's frame)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BranchSpec:
    """One admitted call of a fan-out: the child definition and its new run id."""

    call: ToolCall
    child: Agent
    child_path: tuple[str, ...]
    child_run_id: str


@dataclass(frozen=True)
class FanoutInvocation:
    """The admitted managed calls of one parent turn at its claim epoch (N ≥ 2).

    The parent fields are those of the single-call ``ManagedInvocation``;
    ``branches`` holds one :class:`BranchSpec` per call, in call order.

    Raises:
        ValueError: fewer than 2 or more than 32 branches, or a repeated call
            or child run id.
    """

    store: StateStore
    tenant_id: str | None
    root_run_id: str
    parent_run_id: str
    parent_epoch: int
    parent_level: int
    parent_agent: Agent
    wait: PauseState
    iteration: int
    parent_usage: UsageStats
    max_depth: int | None
    branches: tuple[BranchSpec, ...]
    unpriced_accepted: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "branches", tuple(self.branches))
        count = len(self.branches)
        if not MIN_FANOUT <= count <= MAX_FANOUT:
            raise ValueError(f"a fan-out needs 2..{MAX_FANOUT} branches, got {count}")
        if (
            len(set(self.call_ids)) != count
            or len({b.child_run_id for b in self.branches}) != count
        ):
            raise ValueError("a fan-out branch repeats a call or a child run id")

    @property
    def call_ids(self) -> tuple[str, ...]:
        """The tool call ids, in call order."""
        return tuple(branch.call.id for branch in self.branches)

    @property
    def bindings(self) -> tuple[FanoutBinding, ...]:
        """``(tool_call_id, child_run_id)`` per branch, in call order."""
        return tuple(FanoutBinding(b.call.id, b.child_run_id) for b in self.branches)


# ---------------------------------------------------------------------------
# The steps (signatures only; implemented by later modules)
# ---------------------------------------------------------------------------


class WalkTree(Protocol):
    """``walk_tree``: read the workflow under a root into one validated tree.

    Bounded by the root's saved run limit; tenant, parent link, header and path
    are checked at every hop before anything is disclosed.
    """

    async def __call__(
        self, root_agent: Agent, store: StateStore, root_run_id: str, *, tenant_id: TenantScope
    ) -> WorkflowTree: ...


class FindLeaf(Protocol):
    """``find_leaf``: the paused leaf a pause token addresses.

    Raises the stale-at-lookup error when no paused leaf holds ``pause_id``.
    """

    def __call__(self, tree: WorkflowTree, pause_id: str) -> LeafNode: ...


class SettleParent(Protocol):
    """``settle_parent``: adopt a wait's children if every one of them is terminal.

    Reads the children in a new writer transaction, then claims and completes
    the adoption. Returns the parent's claim, ready to continue, or None when a
    child is not terminal, the parent is being cancelled, or another caller won.
    """

    async def __call__(self, store: StateStore, node: WaitNode) -> ClaimResult | None: ...


class AdoptFanout(Protocol):
    """``adopt_fanout``: complete a claimed v4 adoption's unresolved rows, in call order.

    Runs under ``claim`` (the parent's adoption claim). Returns the claim state
    after the last completion, with the wait cleared.
    """

    async def __call__(
        self, store: StateStore, node: WaitNode, claim: ClaimResult
    ) -> ClaimResult: ...


class AttachPendings(Protocol):
    """``attach_pendings``: the one writer of a result's pause fields.

    Sets ``pending``, ``pendings``, ``pause_id`` and the pause-dependent status
    from ``pendings`` (stable order); an empty sequence clears them.
    """

    def __call__(self, result: RunResult, pendings: Sequence[PendingPause]) -> RunResult: ...


PROTOCOLS: Final[dict[str, type]] = {
    "walk_tree": WalkTree,
    "find_leaf": FindLeaf,
    "settle_parent": SettleParent,
    "adopt_fanout": AdoptFanout,
    "attach_pendings": AttachPendings,
}
"""Spec function name → the Protocol its implementation satisfies."""
