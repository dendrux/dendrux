"""Deliver definition refusals discovered during a stream's pre-claim setup."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

from dendrux.errors import GovernanceMismatchError, RunNotReclaimableError, SkillContinuityError
from dendrux.runtime import (
    _managed_route,
    _managed_stream,
    _refusal_view,
    _run_images,
)
from dendrux.skills._session import check_image_manifest
from dendrux.types import RunResult, RunStatus

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from dendrux.agent import Agent
    from dendrux.runtime.state import RunRecord, StateStore, TenantScope

T = TypeVar("T")
DefinitionRefusal = GovernanceMismatchError | SkillContinuityError | ValueError


async def open_leaf(
    state: Any, waiting: Any, first: Any, resumed: Any, admitted: _refusal_view.AdmittedRoot
) -> Any:
    events = first.__aiter__()
    head = await anext(events, None)
    if head is None or head.type in _managed_stream._TERMINAL_EVENTS:
        await events.aclose()
        state.waiting = []
        return _managed_stream._Single(await _terminal_event(head, resumed.run_id, admitted)), None
    started = _managed_stream._Started(first, events, head)
    state.waiting, state.current, state.root_active = waiting, started, False
    return started, resumed


async def _terminal_event(head: Any, root_id: str, admitted: _refusal_view.AdmittedRoot) -> Any:
    if head and head.run_result and head.run_result.run_id == root_id:
        return head
    error = head.error if head else None
    try:
        result = await admitted.latest()
    except _refusal_view.ProjectionUnavailableError:
        if not error:
            raise
        # The leaf already reported its error. No root state can safely supplement it.
        result = RunResult(run_id=root_id, status=RunStatus.ERROR)
    result.error = error
    return _managed_stream._event_with_error(result, error)


def routed_open(
    open_first: Callable[..., Any],
    route: Callable[..., Any],
    run_id: str,
    tenant_id: TenantScope,
    pause_id: str | None,
) -> Callable[..., Any]:
    @wraps(open_first)
    async def open_checked(state: Any) -> Any:
        state.store = await state.root_agent._resolve_state_store()
        root = await _managed_route._scoped_root(state.store, run_id, tenant_id)
        admitted = _refusal_view.AdmittedRoot(state.store, root)
        try:
            found = await route(
                state.root_agent, state.store, run_id, tenant_id=root.tenant_id, admitted=root
            )
            state.root_run_id, state.tenant_id = run_id, found.tenant_id
            target = await _managed_route.fanout_target(
                state.root_agent, state.store, found, pause_id
            )
        except (GovernanceMismatchError, RunNotReclaimableError) as exc:
            if isinstance(exc, RunNotReclaimableError) and exc.reason not in {
                "skill_mismatch",
                "agent_mismatch",
            }:
                raise
            state.root_active, state.root_run_id = True, run_id
            return _managed_stream._Single(await admitted.refusal_event(exc)), None
        return await open_first(state, found, target, admitted)

    return open_checked


class _InvalidSkillManifestError(Exception):
    def __init__(self, error: ValueError):
        self.error = error
        super().__init__(str(error))


async def image_run(agent: Agent, store: StateStore, run_id: str) -> RunRecord:
    record = await _run_images.read_resume_image_run(store, run_id)
    try:
        check_image_manifest(agent, record, run_id)
    except ValueError as exc:
        # Only a definition error belongs to the terminal event. Ordinary image errors propagate.
        raise _InvalidSkillManifestError(exc) from exc
    return record


async def image_setup(operation: Awaitable[T]) -> T | DefinitionRefusal:
    """Image errors raise on first pull. Definition errors use the stream's failure handler."""
    try:
        return await operation
    except _InvalidSkillManifestError as refusal:
        return refusal.error
    except (GovernanceMismatchError, SkillContinuityError) as refusal:
        return refusal


def checked_images(outcome: T | DefinitionRefusal) -> T:
    if isinstance(outcome, DefinitionRefusal):
        raise outcome
    return outcome
