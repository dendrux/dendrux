"""Managed-subagent submission helpers for the submit methods (0.7).

Moved out of ``dendrux.runtime.submit``, which imports every name back so
existing callers and patch seams are unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dendrux import _codes
from dendrux._meta_keys import GOVERNANCE_KEY
from dendrux.types import PauseState, RunStatus, ToolResult, ToolTarget

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from dendrux.agent import Agent
    from dendrux.runtime.state import ClaimResult, StateStore


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


async def _admit_managed_parent(agent: Agent, store: StateStore, run_id: str) -> dict[str, Any]:
    """The parent's current definition may accept the report: name, governance, skills.

    The same admission an ordinary reclaim or submission applies, before any
    claim or write. The saved evidence (run governance snapshot, checkpoint
    skill pins) changes only under a claim, and the claim below is a CAS on the
    wait generation read here, so no other claim can slip between check and use.
    Returns the saved governance view for the ``warn`` report.
    """
    from dendrux.errors import GovernanceMismatchError, RunNotFoundError, RunNotReclaimableError
    from dendrux.runtime import (
        submit as _submit,  # cycle: submit re-exports this module (AGENTS 6.2)
    )
    from dendrux.skills._session import SkillSession

    record = await store.get_run(run_id)
    if record is None:
        raise RunNotFoundError(run_id)
    if record.agent_name != agent.name:
        raise RunNotReclaimableError(run_id, "agent_mismatch")
    saved = {GOVERNANCE_KEY: (record.meta or {}).get(GOVERNANCE_KEY)}
    if saved[GOVERNANCE_KEY] is None:
        saved.pop(GOVERNANCE_KEY)
    change = _submit._governance_change(agent, saved)
    if change is not None and agent._on_governance_change == "raise":
        import logging

        _codes.log_code(logging.getLogger(__name__), _codes.GOVERNANCE_MISMATCH, run_id=run_id)
        raise GovernanceMismatchError(run_id, **change)
    state = await store.get_loop_state(run_id) or {}
    pins = {s.name: s.digest for s in SkillSession.prepare(agent).available.values()}
    if ((state.get("skills") or {}).get("pins", {})) != pins:
        raise RunNotReclaimableError(run_id, "skill_mismatch")
    return saved


async def claim_managed_adoption(
    store: StateStore,
    run_id: str,
    *,
    agent: Agent,
    call_id: str,
    child_run_id: str,
    result: ToolResult,
    blocked: str | None = None,
) -> ClaimResult | bool:
    """Durably accept a child's report into its waiting parent and claim the parent.

    ``submit_and_claim`` on the parent's private managed-wait generation: the
    report and the parent's new epoch commit together. Returns False when the
    parent is no longer at that wait (another caller adopted, or it moved on).
    The report is consumed only by the managed adoption, never by
    ``consume_submission``.
    """
    from functools import partial

    from dendrux.runtime import (
        submit as _submit,  # cycle: submit re-exports this module (AGENTS 6.2)
    )
    from dendrux.runtime.durability import _ownership_handoff

    raw = await store.get_pause_state(run_id)
    if raw is None or raw.get("schema_version") != 3:
        return False
    governance = await _admit_managed_parent(agent, store, run_id)
    wait = PauseState.from_managed_dict(raw, run_id=run_id)
    assert wait.pause_id is not None
    descriptor = managed_submission_descriptor(wait, call_id)
    submitted: dict[str, Any] = {"submitted_tool_results": _submit.serialize_tool_results([result])}
    if blocked is not None:
        # Safe guardrail text only; recovery reapplies the stop, never the policy.
        submitted["managed_blocked"] = True
    operation = partial(
        store.submit_and_claim,
        run_id,
        expected_status=RunStatus.PAUSED.value,
        submitted_data=submitted,
        submission=descriptor,
        validate=_managed_adoption_validator(run_id, wait.pause_id, call_id, child_run_id),
    )
    from dendrux.runtime._continuation_guardrails import _submission_evidence

    # A blocked report marks the parent in its claim transaction, as a blocked submission does.
    async with _submission_evidence(store, run_id, blocked is not None):
        won: ClaimResult | bool = await _ownership_handoff(
            store, operation, partial(_submit._cancel_claim, store, run_id), run_id=run_id
        )
    if won and not isinstance(won, bool):
        from dendrux.runtime.state import claimed_store

        await _submit.report_governance_change(
            agent, claimed_store(store, run_id, won), run_id, governance
        )
    return won


def _refuse_tokenless_managed(run_id: str, checkpoint: dict[str, Any] | None) -> None:
    """A managed run's pause is claimed only with its explicit token (0.7).

    Decided at claim admission from the run's persisted managed identity, not
    the caller's Agent: an Agent rebuilt without registrations cannot apply
    the ordinary token rule to a managed root or child.
    """
    from dendrux.runtime import (
        submit as _submit,  # cycle: submit re-exports this module (AGENTS 6.2)
    )
    from dendrux.runtime.state import MANAGED_HEADER_KEY

    if (checkpoint or {}).get(MANAGED_HEADER_KEY) is not None:
        import logging

        _codes.log_code(
            logging.getLogger(__name__), _codes.MANAGED_PAUSE_ID_REQUIRED, run_id=run_id
        )
        raise _submit._pause_id_required()
