"""Managed subagents: routing a root-id continuation to the right node.

A root-id control (``submit_*``, ``resume``, ``resume_stream``) reaches the
workflow's active leaf through the persisted bindings: :func:`walk` follows each
wait's exact child id from the root, validating tenant, parent link, header and
path at every hop, and rebuilds the waiting parents (:class:`_Waiting`).
:func:`root_view` and :func:`_projected_pause` are what a caller that does not
own the next step sees. The claim and the driver stay in
:mod:`dendrux.runtime._managed`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from dendrux.errors import ManagedStateError, RunNotFoundError
from dendrux.runtime import _managed_budget
from dendrux.runtime._managed_limits import _saved_walk_limit
from dendrux.runtime.state import MANAGED_HEADER_KEY, ManagedHeader
from dendrux.types import PauseState, RunResult, RunStatus, ToolCall, UsageStats

if TYPE_CHECKING:
    from dendrux.agent import Agent
    from dendrux.runtime.state import RunRecord

MANAGED_WAIT_REASON = "managed_wait"


@dataclass(frozen=True)
class _Waiting:
    """A parent waiting on one bound child (rebuilt from storage after a restart)."""

    parent_agent: Agent
    parent_run_id: str
    child: Agent
    child_path: tuple[str, ...]
    call: ToolCall
    child_run_id: str
    child_loop: str | None = None  # the bound child's saved loop marker (restart only)

    @property
    def call_id(self) -> str:
        return self.call.id


# ---------------------------------------------------------------------------
# The root-facing view
# ---------------------------------------------------------------------------


async def root_view(store: Any, root_run_id: str) -> RunResult:
    """The root's persisted, root-facing state, for a caller that does not own the next step.

    A root waiting on a subagent reports its active branch: the leaf's external
    pause, or RUNNING while a subagent runs or awaits adoption.
    """
    from dendrux.runtime.runner import _build_cached_result

    return await _build_cached_result(store, root_run_id)


def _root_result(record: RunRecord, *, status: RunStatus, **fields: Any) -> RunResult:
    return RunResult(
        run_id=record.id,
        status=status,
        iteration_count=record.iteration_count,
        usage=UsageStats(
            input_tokens=record.total_input_tokens,
            output_tokens=record.total_output_tokens,
            total_tokens=record.total_input_tokens + record.total_output_tokens,
            cost_usd=record.total_cost_usd,
            reasoning_tokens=record.total_reasoning_tokens,
        ),
        **fields,
    )


async def _projected_pause(
    store: Any, root_run_id: str, waiting: list[_Waiting], leaf: RunResult
) -> RunResult:
    """A computed root view of a leaf's external pause; the parents keep waiting."""
    record = await store.get_run(root_run_id)
    if record is None:
        raise ManagedStateError(reason="child_missing", run_id=root_run_id)
    pending = leaf.pending
    if pending is not None:
        pending = replace(
            pending,
            run_id=root_run_id,
            agent_name=waiting[-1].child.name,
            agent_path=[record.agent_name, *waiting[-1].child_path],
            origin_run_id=leaf.run_id,
        )
    return _root_result(
        record,
        status=leaf.status,
        pause_id=pending.pause_id if pending is not None else None,
        pending=pending,
    )


# ---------------------------------------------------------------------------
# Root routing: the persisted bindings, root to leaf (0.7, T3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Route:
    """A root-id control resolved to the workflow's active leaf (in memory).

    ``tenant_id`` is the root's actual tenant; every hop carries it. ``waiting``
    lists the parents from the root down to the leaf's parent; the leaf is the
    root itself when nothing is waiting.
    """

    root_run_id: str
    tenant_id: str | None
    waiting: tuple[_Waiting, ...]
    leaf: RunRecord
    leaf_agent: Agent

    @property
    def routed(self) -> bool:
        """The leaf is a subagent, not the root itself."""
        return bool(self.waiting)


async def walk(root_agent: Agent, store: Any, root_run_id: str) -> tuple[list[_Waiting], RunRecord]:
    """Follow each v3 wait's exact child id, validating every hop; bounded."""
    record = await store.get_run(root_run_id)
    if record is None:
        raise RunNotFoundError(root_run_id)
    waiting: list[_Waiting] = []
    agent: Agent = root_agent
    path: tuple[str, ...] = ()
    limit = await _saved_walk_limit(store, record)
    for index in range(limit):
        wait = await _outgoing(store, record)
        if wait is None or wait.managed_wait is None:
            return waiting, record
        if index + 1 == limit:
            break
        binding = wait.managed_wait
        child, path, loop = await _bound_child(store, record, binding, root_run_id, path)
        child_agent = _managed_budget.inherit(_registered(agent, path[-1], record.id), root_agent)
        intent = await _intent_call(store, record.id, binding.tool_call_id, wait.iteration)
        waiting.append(_Waiting(agent, record.id, child_agent, path, intent, child.id, loop))
        agent, record = child_agent, child
    raise ManagedStateError(reason="binding_conflict", run_id=root_run_id)


async def _outgoing(store: Any, record: RunRecord) -> PauseState | None:
    """The v3 managed wait of a parked parent, else None.

    A pause carrying either managed-wait mark is decoded strictly, so a corrupt
    wait is invalid state, never read as the root's own ordinary pause.
    """
    if record.status != RunStatus.PAUSED.value:
        return None
    raw = await store.get_pause_state(record.id)
    if not raw or (
        raw.get("schema_version") != 3 and raw.get("pause_reason") != MANAGED_WAIT_REASON
    ):
        return None
    return PauseState.from_managed_dict(raw, run_id=record.id)


async def _bound_child(
    store: Any, parent: RunRecord, binding: Any, root_run_id: str, path: tuple[str, ...]
) -> tuple[RunRecord, tuple[str, ...], Any]:
    child = await store.get_run(binding.child_run_id)
    if child is None:
        raise ManagedStateError(reason="child_missing", run_id=parent.id)
    if child.parent_run_id != parent.id or child.tenant_id != parent.tenant_id:
        raise ManagedStateError(reason="child_mismatch", run_id=parent.id)
    state = await store.get_loop_state(child.id) or {}
    header = ManagedHeader.from_dict(state.get(MANAGED_HEADER_KEY), run_id=child.id)
    if (
        header.root_run_id != root_run_id
        or header.parent_run_id != parent.id
        or header.tool_call_id != binding.tool_call_id
        or header.path[:-1] != path
    ):
        raise ManagedStateError(reason="child_mismatch", run_id=parent.id)
    return child, header.path, state.get("loop")


async def _intent_call(store: Any, parent_run_id: str, call_id: str, iteration: int) -> ToolCall:
    """The bound call as the parent's model made it (its brief lives on the intent row).

    Read only the wait's own iteration: its rows are one model batch, bounded by
    ``Limits.max_pending_calls``, so the paged read ends; earlier history is never read.
    """
    from dendrux.runtime._bounds import MAX_PAGE

    offset = 0
    while True:
        page = await store.get_tool_calls(
            parent_run_id, iteration_index=iteration, limit=MAX_PAGE, offset=offset
        )
        for row in page:
            if row.tool_call_id == call_id and row.target == "agent":
                return ToolCall(
                    name=row.tool_name,
                    params=dict(row.params or {}),
                    id=row.tool_call_id,
                    provider_tool_call_id=row.provider_tool_call_id,
                )
        if len(page) < MAX_PAGE:
            raise ManagedStateError(reason="binding_missing", run_id=parent_run_id)
        offset += MAX_PAGE


def _registered(parent: Agent, name: str, parent_run_id: str) -> Agent:
    """The parent's registered child by name; a missing path refuses before any claim."""
    for child in parent.subagents:
        if child.name == name:
            return child
    raise ManagedStateError(reason="path_missing", run_id=parent_run_id)


_Waiting.__module__ = "dendrux.runtime._managed"
Route.__module__ = "dendrux.runtime._managed"
