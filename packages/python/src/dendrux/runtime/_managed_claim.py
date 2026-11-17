"""Managed-subagent submission helpers for the submit methods (0.7).

Moved out of ``dendrux.runtime.submit``, which imports every name back so
existing callers and patch seams are unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dendrux.types import PauseState, ToolResult, ToolTarget

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


def refuse_managed_results(
    run_id: str, pending_targets: Mapping[str, Any] | None, results: list[ToolResult]
) -> None:
    """Refuse an external result for a generated subagent call (0.7).

    Only the bound child run reports a subagent call. This holds even for a
    client pause older code persisted for it (its target stays ``agent``).
    """
    from dendrux.errors import ManagedResultRefusedError

    targets = pending_targets or {}
    if any(targets.get(result.call_id) == ToolTarget.AGENT.value for result in results):
        raise ManagedResultRefusedError(run_id)


def managed_submission_descriptor(wait: PauseState, call_id: str) -> dict[str, Any]:
    """The format-2 submission descriptor of a managed adoption (0.7).

    A v3 managed wait still uses the closed format-2 envelope: kind
    ``tool_results`` for exactly the bound call, at the wait's private generation.
    """
    from dendrux.types import generate_ulid

    return {
        "submission_id": generate_ulid(),
        "kind": "tool_results",
        "pause_id": wait.pause_id,
        "payload_ref": {"pause_data_key": "submission"},
        "call_ids": [call_id],
        "reserved_order_index": wait.trace_order_offset,
    }


def _managed_adoption_validator(
    run_id: str, pause_id: str, call_id: str, child_run_id: str
) -> Callable[[dict[str, Any]], None]:
    """Inside the claim: the same v3 generation, still bound to this call and child."""
    from dendrux.errors import ManagedStateError

    def validate(current: dict[str, Any]) -> None:
        wait = PauseState.from_managed_dict(current, run_id=run_id)
        binding = wait.managed_wait
        if (
            wait.pause_id != pause_id
            or binding is None
            or binding.tool_call_id != call_id
            or binding.child_run_id != child_run_id
        ):
            raise ManagedStateError(reason="binding_conflict", run_id=run_id)

    return validate
