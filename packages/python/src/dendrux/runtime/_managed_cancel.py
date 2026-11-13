"""Implement managed cancellation contracts, traversal, admission checks and settlement."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol, cast

from dendrux import _codes
from dendrux._lifecycle import STORAGE_FORMAT_CHECKPOINTED, CancelReason
from dendrux.errors import (
    ManagedChildControlError,
    ManagedStateError,
    RunCancelledError,
    RunFencedError,
)
from dendrux.runtime._bounds import MAX_SMALL_PAGE
from dendrux.types import Budget, RunResult, RunStatus, ToolResult, _usage_from_dict

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dendrux.loops.base import LoopNotifier, LoopRecorder
    from dendrux.runtime.state import (
        BindManagedChildRequest,
        BindManagedChildResult,
        RunControl,
        RunRecord,
        TenantScope,
    )

logger = logging.getLogger(__name__)
_CANCEL_OUTCOME_LIMIT = 999


def _state() -> Any:
    # Cycle: state -> _managed_cancel -> state; TODO(T5a-F4): separate the managed contract.
    return importlib.import_module(".state", __package__)


def _runner() -> Any:
    # Cycle: runner -> _managed_cancel -> runner; TODO(T5a-F6): share terminal finalization.
    return importlib.import_module(".runner", __package__)


class ManagedStoreCapability(Protocol):
    """The atomic bind and cancel-read contracts required by managed workflows."""

    async def bind_managed_child(
        self, request: BindManagedChildRequest
    ) -> BindManagedChildResult: ...

    async def get_managed_run_count(
        self, root_run_id: str, *, limit: int, tenant_id: str | None
    ) -> int | None:
        """Count explicit reachable managed membership, including root and completed runs.

        Return min(count, limit), or None if scope validation is inconclusive.
        Never infer membership from parent_run_id alone or load histories.
        Bound child candidates, including legacy rows, to limit+1; exhaustion
        without a conclusive count returns None.
        Bind must perform the same bounded count in its root-serialized transaction,
        after replay detection and before writes. The standalone read is advisory
        during concurrent binds.
        """
        ...

    async def get_managed_cancel_children_page(
        self, parent_ids: Sequence[str], *, after: str | None, limit: int, tenant_id: TenantScope
    ) -> list[tuple[str, str | None, Any]]:
        """Project child id, parent id and raw managed header without loading history.

        Return distinct child IDs in ascending order, strictly after the keyset
        cursor, scoped to the explicit tenant (None means only the null tenant;
        UNSCOPED is an explicit administrative scope). Return at most min(limit,
        500) rows; a non-positive or non-integer limit refuses. Empty parents
        return an empty page. Keep malformed non-null headers in the projection
        so the cancellation caller can skip and report them.
        """
        ...

    async def is_managed_root_stopped(self, run_id: str) -> bool:
        """Read root eligibility without histories, using the own row's tenant.

        Missing own rows fail closed. A missing, foreign-tenant, marked or terminal
        referenced root is stopped; a legacy row with no managed root is not.
        """
        ...

    async def get_managed_run_control(self, run_id: str) -> RunControl:
        """Read own control/ownership and root stop in one admission snapshot.

        Preserve own pause, pending-note, epoch and status fields. Cancel is own
        cancellation OR root stopped, using the same tenant rule as the root read.
        A missing own row returns cancel_requested=True, failing closed.
        """
        ...


class ManagedRootStoppedError(RunCancelledError):
    """Managed execution reached a checkpoint after its root stopped."""


@dataclass(frozen=True)
class CancelPage:
    """One normalized frontier page; the cursor advances across filtered-out rows too."""

    run_ids: tuple[str, ...]
    after: str | None
    has_more: bool
    skipped_ids: tuple[str, ...] = ()


class CancelPageReader(Protocol):
    """Read bounded child IDs while preserving the underlying keyset cursor."""

    @property
    def skipped_run_ids(self) -> list[str]: ...

    async def read(
        self,
        parent_ids: Sequence[str],
        *,
        after: str | None,
        limit: int,
        tenant_id: str | None,
    ) -> CancelPage: ...


@dataclass(frozen=True)
class LegacyCancelPages:
    """Adapt the existing legacy ID page without changing its query or membership."""

    store: Any
    skipped_run_ids: list[str] = field(default_factory=list)

    async def read(
        self,
        parent_ids: Sequence[str],
        *,
        after: str | None,
        limit: int,
        tenant_id: str | None,
    ) -> CancelPage:
        rows = await self.store.get_children_page(
            parent_ids, after=after, limit=limit, tenant_id=tenant_id
        )
        return CancelPage(tuple(rows), rows[-1] if rows else after, len(rows) >= limit)


@dataclass(frozen=True)
class ManagedCancelPages:
    """Filter explicit managed edges, retaining the projected page's raw cursor."""

    store: Any
    root_run_id: str
    skipped_run_ids: list[str] = field(default_factory=list)

    def _child_id(self, row: Any) -> str | None:
        return cancel_child_id(row, self.root_run_id)

    async def read(
        self,
        parent_ids: Sequence[str],
        *,
        after: str | None,
        limit: int,
        tenant_id: str | None,
    ) -> CancelPage:
        rows = await self.store.get_managed_cancel_children_page(
            parent_ids, after=after, limit=limit, tenant_id=tenant_id
        )
        ids, skipped = [], []
        for row in rows:
            try:
                child = self._child_id(row)
            except ManagedStateError:
                skipped.append(str(row[0]))
            else:
                if child is not None:
                    ids.append(child)
        return CancelPage(
            tuple(ids), rows[-1][0] if rows else after, len(rows) >= limit, tuple(skipped)
        )


class AllCancelPages(ManagedCancelPages):
    """Include legacy edges while validating every non-null managed child header."""

    def _child_id(self, row: Any) -> str:
        if row[2] is not None:
            _state().ManagedHeader.from_dict(row[2], run_id=row[0])
        return str(row[0])


def can_mark_cancel(store: Any) -> bool:
    """Whether cancellation can mark each frontier before scanning its child pages."""
    return all(
        callable(getattr(store, name, None))
        for name in ("mark_cancel_requested", "get_children_page")
    )


async def cancel_scope(
    store: Any, record: RunRecord, *, cascade: bool
) -> tuple[bool, CancelPageReader | None]:
    """Force managed traversal; legacy inclusion is still caller-selected."""
    if record.storage_format != STORAGE_FORMAT_CHECKPOINTED:
        return cascade, None
    getter = getattr(store, "get_loop_state", None)
    value = (await getter(record.id) or {}).get(_state().MANAGED_HEADER_KEY) if getter else None
    if value is None:
        return cascade, None
    header = _state().ManagedHeader.from_dict(value, run_id=record.id)
    if header.parent_run_id is not None:
        raise ManagedChildControlError(record.id)
    if header.root_run_id != record.id:
        raise ManagedStateError(reason="header_invalid", run_id=record.id)
    require_cancellation(store, record.id)
    pages: CancelPageReader = (
        AllCancelPages(store, record.id) if cascade else ManagedCancelPages(store, record.id)
    )
    return True, pages


def cancel_child_id(item: Any, root_run_id: str) -> str | None:
    """Filter an explicit managed edge from a bounded projected child page."""
    run_id, parent_id, value = item
    if value is None:
        return None
    header = _state().ManagedHeader.from_dict(value, run_id=run_id)
    if header.root_run_id != root_run_id or header.parent_run_id != parent_id:
        return None
    return str(run_id)


async def checkpoint(
    store: Any, run_id: str, state: dict[str, Any], *, claim_epoch: int | None = None
) -> RunControl | None:
    """One managed admission read; None preserves ordinary legacy control reads."""
    if _state().MANAGED_HEADER_KEY not in state:
        return None
    header = _state().ManagedHeader.from_dict(state[_state().MANAGED_HEADER_KEY], run_id=run_id)
    control = await _managed_control(store, run_id, claim_epoch=claim_epoch)
    if control.cancel_requested:
        raise ManagedRootStoppedError(run_id, header.root_run_id)
    return control


async def _managed_control(
    store: Any, run_id: str, *, claim_epoch: int | None = None
) -> RunControl:
    control: RunControl = await _state().require_store_method(store, "get_managed_run_control")(
        run_id
    )
    if claim_epoch is not None and (
        control.claim_epoch != claim_epoch or control.status != RunStatus.RUNNING.value
    ):
        _codes.log_code(logger, _codes.RUN_FENCED, run_id=run_id)
        raise RunFencedError(run_id, claim_epoch=claim_epoch)
    return control


async def cancellation_requested(store: Any, run_id: str) -> bool:
    return (
        await _managed_control(store, run_id, claim_epoch=getattr(store, "claim_epoch", None))
    ).cancel_requested


async def cancelled_result(
    store: Any,
    run_id: str,
    recorder: Any,
    notifier: Any,
    *,
    recorder_started: bool = False,
    notifier_started: bool = False,
) -> RunResult:
    source = getattr(recorder, "_loop_source", None)
    data = source() if source is not None else {}
    state = getattr(recorder, "_loop_state", None) or getattr(store, "loop_state", None) or {}
    result = RunResult(
        run_id=run_id,
        status=RunStatus.CANCELLED,
        steps=list(data.get("steps", ())),
        iteration_count=int(state.get("iteration", 0)),
        usage=replace(data["usage"])
        if "usage" in data
        else _usage_from_dict(state.get("usage") or {}),
    )
    result = await _runner()._finalize_loop_cancelled(
        store,
        run_id,
        result,
        None,
        {},
        [RunStatus.RUNNING.value],
        {"reason": CancelReason.CANCEL_REQUESTED.value},
    )
    await _runner()._finish_fenced_lifecycle(
        run_id,
        result,
        recorder if recorder_started else None,
        notifier if notifier_started else None,
    )
    return cast("RunResult", result)


def require_cancellation(store: Any, run_id: str | None = None) -> None:
    # Cycle: _capabilities -> state -> _managed_cancel -> _capabilities.
    # TODO(T5a-F4): separate the managed store contract.
    importlib.import_module("._capabilities", __package__).require_managed_subagents(store)


def outcome_code(report: ToolResult) -> str:
    if report.meta.get("code") == _codes.MANAGED_CHILD_CANCELLED.name:
        return "cancelled"
    return "succeeded" if report.success else "failed"


async def legacy_cancel_levels(
    store: Any, run_id: str, tenant_id: str | None, limit: int
) -> tuple[list[list[str]], bool]:
    subtree = await _state().require_store_method(store, "get_subtree")(run_id, tenant_id=tenant_id)
    nodes = (
        sorted(
            (node for node in subtree.nodes if node.run.id != run_id), key=lambda node: node.depth
        )
        if subtree is not None
        else []
    )
    by_depth = {0: [run_id]}
    for node in nodes[: limit - 1]:
        by_depth.setdefault(node.depth, []).append(node.run.id)
    return [by_depth[depth] for depth in sorted(by_depth)], len(nodes) + 1 > limit


class CancelAdmission:
    """Apply managed admission control through the recorder's borrowed store and state."""

    async def on_run_started(
        self: Any, run_id: str, *, agent_name: str | None = None, agent_model: str | None = None
    ) -> None:
        """Check managed root control after creation or continuation claims."""
        await self._managed_checkpoint(run_id)

    async def _managed_checkpoint(self: Any, run_id: str) -> RunControl | None:
        return await checkpoint(
            self._store_calls,
            run_id,
            self._loop_state,
            claim_epoch=getattr(self._store, "claim_epoch", None),
        )

    async def _admission_control(
        self: Any, run_id: str, control: RunControl | None = None
    ) -> RunControl:
        if control is None:
            control = await self._managed_checkpoint(run_id)
        if control is None:
            control = await _state().read_run_control(self._store_calls, run_id)
        return control


class ManagedStopLifecycle:
    """Settle an owned managed stop using the lifecycle's existing start flags."""

    async def _stopped(
        self: Any,
        stop: ManagedRootStoppedError | RunFencedError,
        recorder: LoopRecorder | None,
        notifier: LoopNotifier | None,
    ) -> RunResult:
        if isinstance(stop, ManagedRootStoppedError):
            return await cancelled_result(
                self.store,
                self.run_id,
                recorder,
                notifier,
                recorder_started=self.recorder_started,
                notifier_started=self.notifier_started,
            )
        return cast("RunResult", await self._fenced(recorder, notifier))


async def stopped_resume_result(
    stop: ManagedRootStoppedError | RunFencedError,
    store: Any,
    run_id: str,
    ctx: Any,
) -> RunResult:
    """Settle a resume stop while pairing only the sinks whose starts completed."""
    if not isinstance(stop, ManagedRootStoppedError):
        return cast("RunResult", await _runner()._fenced_resume_result(store, run_id, ctx))
    recorder, notifier = (ctx.recorder, ctx.notifier) if ctx is not None else (None, None)
    return await cancelled_result(
        store,
        run_id,
        recorder,
        notifier,
        recorder_started=ctx is not None and ctx.recorder_started,
        notifier_started=ctx is not None and ctx.notifier_started,
    )


async def enter_child(invocation: Any, store: Any) -> bool:
    """Refuse a stopped parent's child entry before its bind or other writes."""
    if await cancellation_requested(store, invocation.parent_run_id):
        raise RunCancelledError(invocation.child_run_id, invocation.root_run_id)
    return False


async def route_checkpoint(store: Any, record: RunRecord) -> None:
    """Keep terminal replay available while refusing continuation of a stopped root."""
    if record.status not in _state()._TERMINAL_STATUS_VALUES and await cancellation_requested(
        store, record.id
    ):
        raise RunCancelledError(record.id, record.id)


async def managed_child_header(store: Any, record: RunRecord) -> bool:
    """True when ``record`` is a managed child (its header names an invocation)."""
    if getattr(record, "parent_run_id", None) is None:
        return False
    header = (await store.get_loop_state(record.id) or {}).get(_state().MANAGED_HEADER_KEY)
    return isinstance(header, dict) and "invocation" in header


async def mark_subtree(
    agent: Any,
    store: Any,
    run_id: str,
    *,
    tenant_id: str | None,
    pages: CancelPageReader | None = None,
) -> tuple[list[list[str]], bool]:
    """Mark before scanning each bounded frontier, and retain skipped-child diagnostics."""
    limit = agent._budget.max_subtree_nodes if agent._budget else Budget.max_subtree_nodes
    if not can_mark_cancel(store):
        return await legacy_cancel_levels(store, run_id, tenant_id, limit)
    mark = _state().require_store_method(store, "mark_cancel_requested")
    pages = pages or LegacyCancelPages(store)
    levels, visited, frontier = [[run_id]], {run_id}, [run_id]
    truncated = False
    while frontier:
        # Wait for creators holding parent FOR SHARE before scanning committed children.
        if await mark(frontier, tenant_id=tenant_id):
            for rid in frontier:
                agent._task_manager.cancel(rid)
        if truncated:
            break
        following, truncated = await _scan_frontier(pages, frontier, tenant_id, visited, limit)
        if following:
            levels.append(following)
        frontier = following
    if truncated:
        _codes.log_code(
            logger,
            _codes.CASCADE_TRUNCATED,
            run_id=run_id,
            limit=limit,
            count=sum(len(level) for level in levels),
        )
    return levels, truncated or bool(pages.skipped_run_ids)


async def _scan_frontier(
    pages: CancelPageReader,
    frontier: list[str],
    tenant_id: str | None,
    visited: set[str],
    limit: int,
) -> tuple[list[str], bool]:
    following: list[str] = []
    after = None
    while True:
        page = await pages.read(frontier, after=after, limit=MAX_SMALL_PAGE, tenant_id=tenant_id)
        skipped = set(page.skipped_ids)
        for child in sorted((*page.run_ids, *page.skipped_ids)):
            if child in visited:
                continue
            if len(visited) >= limit:
                return following, True
            visited.add(child)
            _record_child(pages, child, skipped, following)
        if not page.has_more:
            return following, False
        after = page.after


def _record_child(
    pages: CancelPageReader, child: str, skipped: set[str], following: list[str]
) -> None:
    if child in skipped:
        if len(pages.skipped_run_ids) < _CANCEL_OUTCOME_LIMIT:
            pages.skipped_run_ids.append(child)
        _codes.log_code(logger, _codes.CASCADE_CHILD_SKIPPED, run_id=child)
    else:
        following.append(child)


async def cancel_descendants(
    agent: Any,
    store: Any,
    run_id: str,
    tenant_id: str | None,
    frontier_signals: bool,
    outcomes: list[dict[str, str]],
    to_signal: list[str],
    pages: CancelPageReader | None = None,
) -> tuple[int, bool]:
    """Settle deepest descendants and project bounded cancellation/skip outcomes."""
    levels, truncated = await agent._mark_cancel_subtree(
        store, run_id, tenant_id=tenant_id, pages=pages
    )
    if pages is not None:
        outcomes.extend({"run_id": rid, "outcome": "skipped"} for rid in pages.skipped_run_ids)
    count = 0
    descendants = (rid for level in reversed(levels) for rid in level if rid != run_id)
    for rid in descendants:
        outcome = await agent._cancel_one(store, rid, signal=False)
        if outcome != "terminal" and not frontier_signals:
            to_signal.append(rid)
        count += 1
        if len(outcomes) < _CANCEL_OUTCOME_LIMIT:
            outcomes.append({"run_id": rid, "outcome": outcome})
    return count, truncated
