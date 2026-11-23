"""Deliver definition refusals discovered during a stream's pre-claim setup."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

from dendrux.errors import GovernanceMismatchError, RunNotReclaimableError, SkillContinuityError
from dendrux.runtime import _managed_stream, _run_images
from dendrux.skills._session import check_image_manifest

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from dendrux.agent import Agent
    from dendrux.runtime.state import RunRecord, StateStore

T = TypeVar("T")
DefinitionRefusal = GovernanceMismatchError | SkillContinuityError | ValueError


def routed_open(open_first: Callable[..., Any], run_id: str) -> Callable[..., Any]:
    @wraps(open_first)
    async def open_checked(state: Any) -> Any:
        try:
            return await open_first(state)
        except (GovernanceMismatchError, RunNotReclaimableError) as exc:
            if isinstance(exc, RunNotReclaimableError) and exc.reason not in {
                "skill_mismatch",
                "agent_mismatch",
            }:
                raise
            # Routing has admitted this tenant. A refusal owns no run, even if closed here.
            state.root_active, state.root_run_id = True, run_id
            event = await _managed_stream._stream_failure(state, exc, run_id)
            return _managed_stream._Single(event), None

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
