"""Shared race-safe helpers for the Agent submit methods.

The agent methods :meth:`Agent.submit_tool_results`,
:meth:`Agent.submit_input`, and :meth:`Agent.submit_approval` all follow
the same persist-first-then-CAS-claim pattern. This module keeps that
pattern in one place so the guarantees (persist before claim, atomic
transition, stable errors) stay consistent across methods.

Helpers raise public exceptions from :mod:`dendrux.errors`; callers do
not need to interpret store return values.
"""

from __future__ import annotations

import asyncio
import threading
import weakref
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Any, NoReturn, cast

from dendrux import _codes
from dendrux._lifecycle import PAUSE_STATUSES as _PAUSE_STATUSES
from dendrux._lifecycle import TERMINAL_STATUSES as _TERMINAL_STATUSES
from dendrux._lifecycle import DenialReason, FailureReason, GovernanceEventType, StoredEventType
from dendrux._meta_keys import (
    GOVERNANCE_KEY as GOVERNANCE_KEY,
)
from dendrux.errors import (
    InvalidToolResultError,
    PauseStatusMismatchError,
    RunAlreadyClaimedError,
    RunAlreadyTerminalError,
    RunControlError,
    RunNotFoundError,
    RunNotPausedError,
)
from dendrux.runtime._tenant_checks import _strict_tenancy_applies
from dendrux.types import (
    ULID_CHARS,
    ApprovalDecision,
    ImagePart,
    ImageRef,
    InputValidationError,
    PauseState,
    RunStatus,
    TextPart,
    ToolCall,
    ToolResult,
    ToolTarget,
    content_part_from_dict,
    content_part_to_dict,
)

_HOOK_CACHE_SIZE = 128
_MAX_DECISIONS = 1000
_GOVERNANCE_SNAPSHOT_BYTES = 65536
_MAX_RESULTS = 1000
_MAX_RESULT_ID_CHARS = 255
_MAX_RESULT_IMAGES = 100

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Sequence

    from dendrux.agent import Agent
    from dendrux.llm.base import LLMProvider
    from dendrux.media._catalog import BlobEdge
    from dendrux.media._policy import ImagePolicy, Origin
    from dendrux.media._runtime import ImageRuntime
    from dendrux.runtime.state import ClaimResult, EventWrite, RunRecord, StateStore
    from dendrux.types import ContentPart

__all__ = [
    "DEFAULT_DENIAL_REASON",
    "approval_gates_of",
    "build_rejection_results",
    "check_new_image_capability",
    "claim_or_raise",
    "deserialize_decisions",
    "deserialize_tool_results",
    "effective_image_capability",
    "IngestGroup",
    "ingest_groups",
    "ingest_tool_result_images",
    "ingest_user_items",
    "merge_user_items",
    "user_items_group",
    "parts_edges",
    "tool_result_edges",
    "is_client_call",
    "normalize_decisions",
    "preflight_status",
    "raise_for_non_paused_status",
    "serialize_decisions",
    "serialize_tool_results",
    "validate_approval_decisions",
    "validate_tool_results",
]

DEFAULT_DENIAL_REASON = DenialReason.USER_DECLINED.value
RESUME_FAILURE_KINDS = frozenset(
    {"setup_error", "validation_error", "persistence_error", "cancelled"}
)


async def raise_for_non_paused_status(store: StateStore, run_id: str) -> NoReturn:
    """Raise the most specific exception for a run that has no pause state."""
    record = await store.get_run(run_id)
    if record is None:
        raise RunNotFoundError(run_id)
    current = RunStatus(record.status)
    if current in _TERMINAL_STATUSES:
        raise RunAlreadyTerminalError(run_id, current)
    raise RunNotPausedError(run_id, current)


def _invalid_tool_result(run_id: str, reason: str) -> InvalidToolResultError:
    import logging

    _codes.log_code(logging.getLogger(__name__), _codes.INVALID_TOOL_RESULT, run_id=run_id)
    return InvalidToolResultError(run_id, reason)


async def validate_tool_results(
    store: StateStore, run_id: str, results: list[ToolResult]
) -> PauseState:
    """Verify ``results`` align with the pending tool calls on ``run_id``.

    Raises on any mismatch — no partial acceptance. Rejects duplicate
    ``call_id`` values (one result per pending call) as well as
    missing/unexpected ids. Returns the validated pause.
    """
    raw_pause = await store.get_pause_state(run_id)
    if raw_pause is None:
        await raise_for_non_paused_status(store, run_id)

    pause_state = PauseState.from_dict(raw_pause)
    _check_result_identities(run_id, results)
    pending_ids = {tc.id for tc in pause_state.pending_tool_calls}
    provided_counts = Counter(r.call_id for r in results)
    duplicates = sorted(k for k, v in provided_counts.items() if v > 1)
    provided_ids = set(provided_counts)

    parts: list[str] = []
    if duplicates:
        parts.append(f"duplicate call id count={len(duplicates)}")
    missing = sorted(pending_ids - provided_ids)
    extra = sorted(provided_ids - pending_ids)
    if missing:
        parts.append(f"missing call id count={len(missing)}")
    if extra:
        parts.append(f"unexpected call id count={len(extra)}")

    if parts:
        raise _invalid_tool_result(run_id, "; ".join(parts))
    validate_result_names(run_id, pause_state, results)
    refuse_managed_results(run_id, pause_state.pending_targets, results)
    return pause_state


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


def validate_result_names(run_id: str, pause: PauseState, results: list[ToolResult]) -> None:
    """Bind result IDs/names and require a string payload; text is normalized later."""

    _check_result_identities(run_id, results)
    pending = {call.id: call.name for call in pause.pending_tool_calls}
    if (
        len(results) != len(pending)
        or len({r.call_id for r in results}) != len(results)
        or {r.call_id for r in results} != set(pending)
    ):
        raise _invalid_tool_result(run_id, "tool result call ids do not match pending calls")
    for result in results:
        if pending.get(result.call_id) != result.name:
            raise _invalid_tool_result(run_id, "tool result name does not match pending call")
        if not isinstance(result.payload, str):
            raise _invalid_tool_result(run_id, "tool result payload must be a JSON string")


def pause_validator(
    run_id: str,
    expected_pause_id: str | None,
    *,
    agent_name: str | None = None,
    call_results: list[ToolResult] | None = None,
    client_results: bool = False,
) -> Callable[[dict[str, Any]], None]:
    """Validation performed while the claim transaction holds the run row.

    ``client_results``: ``call_results`` were supplied by the caller (not runtime
    denials), so none may answer a generated subagent call.
    """

    from dendrux.types import _validate_identifier

    if expected_pause_id is not None:
        _validate_identifier(expected_pause_id, "expected_pause_id", maximum=ULID_CHARS)

    def validate(current: dict[str, Any]) -> None:
        current_id = current.get("pause_id")
        if expected_pause_id is not None and expected_pause_id != current_id:
            import logging

            _codes.log_code(logging.getLogger(__name__), _codes.PAUSE_ID_MISMATCH, run_id=run_id)
            raise PauseStatusMismatchError(
                run_id,
                RunStatus(current.get("status", RunStatus.PAUSED.value)),
                RunStatus(current.get("status", RunStatus.PAUSED.value)),
                expected_pause_id=expected_pause_id,
                current_pause_id=current_id,
                reason="pause_id",
            )
        if agent_name is not None and current.get("agent_name") not in (None, agent_name):
            raise ValueError("Agent name mismatch on resume")
        if call_results is not None:
            _check_call_results(run_id, current, call_results, client_results=client_results)

    return validate


def _check_call_results(
    run_id: str, current: dict[str, Any], call_results: list[ToolResult], *, client_results: bool
) -> None:
    """Results answer exactly the pending calls; caller results never a subagent call."""
    pending = {call["id"]: call["name"] for call in current.get("pending_tool_calls", [])}
    if (
        len(pending) != len(call_results)
        or len({r.call_id for r in call_results}) != len(call_results)
        or any(pending.get(result.call_id) != result.name for result in call_results)
    ):
        raise _invalid_tool_result(run_id, "tool results do not match pending calls")
    if client_results:
        refuse_managed_results(run_id, current.get("pending_targets"), call_results)


_RESUME_CLAIM_MESSAGE = (
    "Run '{run_id}' is not in status '{expected}' — "
    "cannot resume. It may have been claimed by another caller."
)


@dataclass(frozen=True)
class _ClaimFailureMapping:
    submit_error: Callable[[str, RunStatus, RunStatus], RunControlError | RunNotFoundError]
    resume_message: str = _RESUME_CLAIM_MESSAGE


_CLAIM_FAILURES: dict[str | tuple[str, str], _ClaimFailureMapping] = {
    "missing": _ClaimFailureMapping(lambda run_id, current, expected: RunNotFoundError(run_id)),
    "terminal": _ClaimFailureMapping(
        lambda run_id, current, expected: RunAlreadyTerminalError(run_id, current)
    ),
    (RunStatus.ERROR.value, FailureReason.ABANDONED_WAITING.value): _ClaimFailureMapping(
        lambda run_id, current, expected: RunAlreadyTerminalError(run_id, current),
        "Run '{run_id}' was abandoned (swept as expired waiting run). It cannot be resumed.",
    ),
    (RunStatus.ERROR.value, FailureReason.STALE_RUNNING.value): _ClaimFailureMapping(
        lambda run_id, current, expected: RunAlreadyTerminalError(run_id, current),
        "Run '{run_id}' was swept as stale ({reason}). It cannot be resumed.",
    ),
    (RunStatus.ERROR.value, FailureReason.NEVER_STARTED.value): _ClaimFailureMapping(
        lambda run_id, current, expected: RunAlreadyTerminalError(run_id, current),
        "Run '{run_id}' was swept as stale ({reason}). It cannot be resumed.",
    ),
    "claimed": _ClaimFailureMapping(
        lambda run_id, current, expected: RunAlreadyClaimedError(run_id)
    ),
    "pause": _ClaimFailureMapping(
        lambda run_id, current, expected: PauseStatusMismatchError(run_id, current, expected)
    ),
    "not_paused": _ClaimFailureMapping(
        lambda run_id, current, expected: RunNotPausedError(run_id, current)
    ),
}


def _claim_failure_mapping(
    record: RunRecord | None, *, after_preflight: bool
) -> _ClaimFailureMapping:
    if record is None:
        return _CLAIM_FAILURES["missing"]
    if record.status in _TERMINAL_STATUSES:
        if record.status == RunStatus.ERROR.value:
            return _CLAIM_FAILURES.get(
                (record.status, getattr(record, "failure_reason", None) or ""),
                _CLAIM_FAILURES["terminal"],
            )
        return _CLAIM_FAILURES["terminal"]
    if after_preflight and record.status == RunStatus.RUNNING.value:
        return _CLAIM_FAILURES["claimed"]
    if record.status in _PAUSE_STATUSES:
        return _CLAIM_FAILURES["pause"]
    return _CLAIM_FAILURES["not_paused"]


def _raise_submit_claim_failure(
    record: RunRecord | None, run_id: str, expected: RunStatus, *, after_preflight: bool
) -> NoReturn:
    current = RunStatus(record.status) if record is not None else RunStatus.PENDING
    rule = _claim_failure_mapping(record, after_preflight=after_preflight)
    raise rule.submit_error(run_id, current, expected)


async def preflight_status(store: StateStore, run_id: str, expected_status: RunStatus) -> None:
    """Raise the most specific error unless ``run_id`` is in ``expected_status``.

    Also refuses a run whose cancellation was requested but not yet observed:
    it is on the path to CANCELLED and resuming it would bounce off the next
    checkpoint.
    """
    record = await store.get_run(run_id)
    if record is not None and RunStatus(record.status) == expected_status:
        if record.cancel_requested:
            raise _CLAIM_FAILURES["terminal"].submit_error(
                run_id, RunStatus.CANCELLED, expected_status
            )
        return
    _raise_submit_claim_failure(record, run_id, expected_status, after_preflight=False)


@dataclass
class _ClaimValidationSupport:
    atomic: bool
    logged: bool = False


@lru_cache(maxsize=_HOOK_CACHE_SIZE)
def _claim_validation_support(method: Any) -> _ClaimValidationSupport:
    import inspect

    params = inspect.signature(method).parameters
    return _ClaimValidationSupport(
        "validate" in params
        or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    )


async def claim_paused_validated(
    store: StateStore,
    run_id: str,
    *,
    expected_status: str,
    validate: Callable[[dict[str, Any]], None],
) -> ClaimResult | int | bool:
    """Use atomic validation when supported; preserve legacy claims best effort.

    The cached key is the unbound implementation, not a store instance. Legacy
    post-claim validation cannot eliminate a race; it stops execution on mismatch.
    """
    import logging

    method = store.claim_paused_run
    implementation = getattr(method, "__func__", method)
    try:
        hash(implementation)
    except TypeError:
        implementation = type(implementation).__call__
    support = _claim_validation_support(implementation)
    if support.atomic:
        return await method(run_id, expected_status=expected_status, validate=validate)
    if not support.logged:
        support.logged = True
        _codes.log_code(logging.getLogger(__name__), _codes.LEGACY_CLAIM_VALIDATION, run_id=run_id)
    current = await store.get_pause_state(run_id)
    if current is None:
        await raise_for_non_paused_status(store, run_id)
    validate(current)
    epoch = await method(run_id, expected_status=expected_status)
    if epoch:
        try:
            current = await store.get_pause_state(run_id)
            if current is None:
                raise ValueError("Claimed run has no pause state")
            validate(current)
        except BaseException as exc:
            await _release_claim_result(store, run_id, expected_status, epoch, exc)
            raise
    return epoch


@dataclass(frozen=True)
class _ClaimRequest:
    raw: dict[str, Any]
    statuses: tuple[str, ...]
    generation: str | None
    explicit: bool = True
    """The caller named the generation; False when it was taken from the read pause."""


def _pause_generation(raw: dict[str, Any], expected_pause_id: str | None) -> str | None:
    return expected_pause_id if expected_pause_id is not None else raw.get("pause_id")


def _claim_request(
    raw: dict[str, Any],
    expected_status: str,
    expected_pause_id: str | None,
    *,
    fallbacks: tuple[str, ...] = (),
    caller_token: bool | None = None,
) -> _ClaimRequest:
    """``caller_token``: whether the caller supplied the token (default: it was given
    when one is passed). A token the runtime pinned from the validated pause is not
    the caller's."""
    return _ClaimRequest(
        raw,
        (expected_status, *fallbacks),
        _pause_generation(raw, expected_pause_id),
        explicit=_caller_supplied(expected_pause_id, caller_token),
    )


def _caller_supplied(expected_pause_id: str | None, caller_token: bool | None) -> bool:
    """Whether the caller gave the token (default: one was passed)."""
    return expected_pause_id is not None if caller_token is None else caller_token


async def _claim_submission(
    store: StateStore,
    run_id: str,
    request: _ClaimRequest,
    status: str,
    *,
    validate: Callable[[dict[str, Any]], None],
    payload: dict[str, Any] | None,
    checkpointing: bool,
    submit_empty: bool = False,
    approval: bool | None = None,
    checkpoint: dict[str, Any] | None = None,
    blob_refs: Sequence[BlobEdge] | None = None,
    blob_refs_scope: str | None = None,
) -> tuple[ClaimResult | int | bool, dict[str, Any] | None, dict[str, Any] | None]:
    from functools import partial

    from dendrux.runtime._continuation_guardrails import _project_submission, _submission_evidence
    from dendrux.runtime.durability import _ownership_handoff

    payload, block = await _project_submission(run_id, payload)
    blocked = block is not None
    descriptor, payload = await _described_submission(
        store,
        run_id,
        request,
        status,
        payload,
        checkpointing=checkpointing,
        submit_empty=submit_empty,
        approval=approval,
        checkpoint=checkpoint,
    )
    operation = _claim_operation(
        store, run_id, status, validate, payload, descriptor, blob_refs, blob_refs_scope
    )
    async with _submission_evidence(store, run_id, blocked):
        won = await _ownership_handoff(
            store, operation, partial(_cancel_claim, store, run_id), run_id=run_id
        )
    if won:
        from dendrux.runtime._blocked_content import _remember_claim

        _remember_claim(store, run_id, block)
    return won, descriptor, payload


async def _described_submission(
    store: StateStore,
    run_id: str,
    request: _ClaimRequest,
    status: str,
    payload: dict[str, Any] | None,
    *,
    checkpointing: bool,
    submit_empty: bool,
    approval: bool | None,
    checkpoint: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """A format-2 claim's submission descriptor and payload; refuses a tokenless
    continuation of a managed run (0.7)."""
    format_two = request.raw.get("schema_version") == 2 and checkpointing
    described = format_two and (payload is not None or submit_empty)
    tokenless = format_two and not request.explicit and request.raw.get("pause_id") is not None
    if (described or tokenless) and checkpoint is None:
        checkpoint = await store.get_loop_state(run_id) or {}
    if tokenless:
        _refuse_tokenless_managed(run_id, checkpoint)
    if not described:
        return None, payload
    payload = dict(payload or {})
    if approval is not None:
        payload["approved"] = approval
    return submission_descriptor(request.raw, checkpoint or {}, payload, status), payload


def _claim_operation(
    store: StateStore,
    run_id: str,
    status: str,
    validate: Callable[[dict[str, Any]], None],
    payload: dict[str, Any] | None,
    descriptor: dict[str, Any] | None,
    blob_refs: Sequence[BlobEdge] | None,
    blob_refs_scope: str | None,
) -> Callable[[], Awaitable[ClaimResult | int | bool]]:
    """The claim the handoff runs: a plain validated claim, or submit-and-claim."""
    from functools import partial

    if payload is None:
        return partial(
            claim_paused_validated, store, run_id, expected_status=status, validate=validate
        )
    kwargs: dict[str, Any] = {"validate": validate}
    if descriptor is not None:
        kwargs["submission"] = descriptor
    if blob_refs:
        kwargs.update(blob_refs=list(blob_refs), blob_refs_scope=blob_refs_scope)
    return partial(
        store.submit_and_claim, run_id, expected_status=status, submitted_data=payload, **kwargs
    )


async def _cancel_claim(store: StateStore, run_id: str, result: ClaimResult | int | bool) -> None:
    from dendrux.runtime.runner import _persist_task_cancellation
    from dendrux.runtime.state import claimed_store

    if not result or isinstance(result, bool):
        import logging

        if result:
            _codes.log_code(
                logging.getLogger(__name__), _codes.CANCEL_PERSIST_FAILED, run_id=run_id
            )
        return
    await _persist_task_cancellation(claimed_store(store, run_id, result), run_id)


def _claimed_submission_view(
    raw: dict[str, Any], descriptor: dict[str, Any] | None, payload: dict[str, Any] | None
) -> dict[str, Any]:
    if descriptor is None:
        return raw
    return {
        **raw,
        "submission": {
            **{k: descriptor[k] for k in ("submission_id", "kind", "pause_id", "call_ids")},
            "payload": payload,
        },
    }


def submission_descriptor(
    raw: dict[str, Any], checkpoint: dict[str, Any], payload: dict[str, Any], status: str
) -> dict[str, Any]:
    """Describe a validated format-2 submission without embedding its payload."""
    from dendrux.types import generate_ulid

    kind = (
        "approval"
        if status == RunStatus.WAITING_APPROVAL.value
        else "input"
        if "submitted_user_input" in payload
        else "tool_results"
    )
    pending = checkpoint.get("pending", [])
    ids = (
        [r["call_id"] for r in payload["submitted_tool_results"]]
        if "submitted_tool_results" in payload
        else [r["call_id"] for r in payload["submitted_decisions"]]
        if "submitted_decisions" in payload
        else [p["call_id"] for p in pending]
    )
    return {
        "submission_id": generate_ulid(),
        "kind": kind,
        "pause_id": raw.get("pause_id"),
        "payload_ref": {"pause_data_key": "submission"},
        "call_ids": ids,
        "reserved_order_index": checkpoint.get("trace_order_offset", 0),
    }


def _submission_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """Accept only the closed format-2 submission envelope (§12.1)."""
    if "submitted" in raw:
        raise ValueError("Invalid format-2 submission envelope")
    envelope = raw.get("submission")
    if envelope is None:
        return {}
    keys = {"submission_id", "kind", "pause_id", "call_ids", "payload"}
    if not isinstance(envelope, dict) or set(envelope) != keys:
        raise ValueError("Invalid format-2 submission envelope")
    if (
        not all(isinstance(envelope[k], str) and envelope[k] for k in ("submission_id", "pause_id"))
        or envelope["kind"] not in ("approval", "input", "tool_results")
        or not isinstance(envelope["payload"], dict)
        or not isinstance(envelope["call_ids"], list)
        or not all(isinstance(call_id, str) for call_id in envelope["call_ids"])
    ):
        raise ValueError("Invalid format-2 submission envelope")
    return envelope["payload"]


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
    from dendrux.skills._session import SkillSession

    record = await store.get_run(run_id)
    if record is None:
        raise RunNotFoundError(run_id)
    if record.agent_name != agent.name:
        raise RunNotReclaimableError(run_id, "agent_mismatch")
    saved = {GOVERNANCE_KEY: (record.meta or {}).get(GOVERNANCE_KEY)}
    if saved[GOVERNANCE_KEY] is None:
        saved.pop(GOVERNANCE_KEY)
    change = _governance_change(agent, saved)
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

    from dendrux.runtime.durability import _ownership_handoff

    raw = await store.get_pause_state(run_id)
    if raw is None or raw.get("schema_version") != 3:
        return False
    governance = await _admit_managed_parent(agent, store, run_id)
    wait = PauseState.from_managed_dict(raw, run_id=run_id)
    assert wait.pause_id is not None
    descriptor = managed_submission_descriptor(wait, call_id)
    submitted: dict[str, Any] = {"submitted_tool_results": serialize_tool_results([result])}
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
            store, operation, partial(_cancel_claim, store, run_id), run_id=run_id
        )
    if won and not isinstance(won, bool):
        from dendrux.runtime.state import claimed_store

        await report_governance_change(agent, claimed_store(store, run_id, won), run_id, governance)
    return won


def submitted_view(raw: dict[str, Any]) -> dict[str, Any]:
    """Project a validated v2 envelope for legacy parsers without mutating storage."""
    return {**raw, **_submission_payload(raw)} if raw.get("schema_version") == 2 else raw


async def claim_or_raise(
    store: StateStore,
    run_id: str,
    *,
    expected_status: RunStatus,
    submitted_data: dict[str, Any] | None = None,
    agent: Agent | None = None,
    expected_pause_id: str | None = None,
    approval: bool = False,
    call_results: list[ToolResult] | None = None,
    validate: Callable[[dict[str, Any]], None] | None = None,
    blob_refs: Sequence[BlobEdge] | None = None,
    blob_refs_scope: str | None = None,
    caller_token: bool | None = None,
) -> ClaimResult | int:
    """Perform a status preflight, then persist-first handoff + CAS claim.

    ``caller_token=False`` marks an ``expected_pause_id`` the runtime pinned from
    the pause it validated (not supplied by the caller): the claim still compares
    it, and a managed run then refuses the claim for its missing token.

    The preflight reads the current status before attempting the CAS so
    that a never-paused running row does not get misclassified as a
    concurrent claim race. If the preflight observes ``expected_status``,
    the CAS runs; any subsequent failure is attributable to a genuine
    race.

    If ``submitted_data`` is provided, merges it into ``pause_data`` and
    transitions from ``expected_status`` to RUNNING atomically. Otherwise
    just claims (used for approval-approve). ``validate`` (with
    ``submitted_data``) runs on the pause_data being claimed inside the
    claim transaction; if it raises, nothing is claimed and it propagates.
    ``blob_refs`` (submission edges of submitted images) are written in the
    claim transaction; they are passed to the store only when non-empty.

    Raises:
        RunNotFoundError: ``run_id`` does not exist.
        RunAlreadyTerminalError: run is already in a terminal state.
        RunAlreadyClaimedError: run was paused at preflight but another
            submit won the CAS.
        PauseStatusMismatchError: run is paused for a different reason.
        RunNotPausedError: run is in some other non-paused state (e.g.
            already running, never paused).
    """
    await preflight_status(store, run_id, expected_status)

    raw = await store.get_pause_state(run_id)
    if raw is None:
        await raise_for_non_paused_status(store, run_id)
    if agent is not None:
        require_pause_id(
            agent,
            expected_pause_id,
            pause_id=raw.get("pause_id"),
            approval=approval,
            store=store,
            run_id=run_id,
        )
    request = _claim_request(
        raw, expected_status.value, expected_pause_id, caller_token=caller_token
    )
    validate = submission_validator(
        run_id,
        raw,
        agent=agent,
        expected_pause_id=request.generation,
        call_results=call_results,
        extra=validate,
        client_results=expected_status == RunStatus.WAITING_CLIENT_TOOL,
    )

    from dendrux.runtime._capabilities import capabilities

    won, _, _ = await _claim_submission(
        store,
        run_id,
        request,
        request.statuses[0],
        validate=validate,
        payload=submitted_data,
        checkpointing=raw.get("schema_version") == 2 and capabilities(store).checkpointing,
        submit_empty=approval,
        approval=approval if expected_status == RunStatus.WAITING_APPROVAL else None,
        blob_refs=blob_refs,
        blob_refs_scope=blob_refs_scope,
    )

    if won:
        if agent is not None:
            try:
                await report_governance_change(agent, store, run_id, raw)
            except BaseException as exc:
                await _release_claim_result(store, run_id, expected_status.value, won, exc)
                raise
        return won

    # The successful preflight distinguishes a competing claim from a never-paused run.
    record = await store.get_run(run_id)
    _raise_submit_claim_failure(record, run_id, expected_status, after_preflight=True)


def _release_supported(store: StateStore, *, claim_epoch: int | None = None) -> bool:
    epoch = claim_epoch if claim_epoch is not None else getattr(store, "claim_epoch", None)
    return (
        isinstance(epoch, int)
        and not isinstance(epoch, bool)
        and callable(getattr(store, "release_run_claim", None))
    )


async def _release_claim_result(
    store: StateStore,
    run_id: str,
    status: str,
    result: ClaimResult | int | bool,
    failure: BaseException | None,
) -> bool:
    if isinstance(failure, asyncio.CancelledError):
        from dendrux.runtime.tasks import _drain_cleanup

        await _drain_cleanup(_cancel_claim(store, run_id, result), run_id=run_id)
        return False
    from dendrux.runtime.state import claimed_store

    view = claimed_store(store, run_id, result)
    if _release_supported(view) or status == RunStatus.RUNNING.value:
        return await release_claim(view, run_id, status, failure=failure)
    if isinstance(failure, Exception):
        from dendrux.runtime.runner import _persist_run_failure

        await _persist_run_failure(view, run_id, failure)
    return False


def _resume_failure_kind(failure: BaseException | None) -> str:
    from dendrux.errors import PersistenceError

    if isinstance(failure, PersistenceError):
        return "persistence_error"
    if isinstance(failure, (ValueError, KeyError, TypeError)):
        return "validation_error"
    return "setup_error"


def _resume_aborted_event(status: str, failure: BaseException | None) -> EventWrite | None:
    from dendrux.runtime.state import EventWrite
    from dendrux.types import generate_ulid

    if RunStatus(status) not in _PAUSE_STATUSES:
        return None
    return EventWrite(
        generate_ulid(),
        StoredEventType.RUN_RESUME_ABORTED.value,
        0,
        {"failure_kind": _resume_failure_kind(failure)},
        None,
        None,
    )


async def _release_with_event(
    store: StateStore, release: Any, run_id: str, status: str, epoch: int, event: EventWrite | None
) -> bool:
    import inspect

    from dendrux.runtime.state import unclaimed_store

    implementation = getattr(unclaimed_store(store), "release_run_claim", release)
    params = inspect.signature(implementation).parameters
    atomic = "event" in params
    kwargs = {"event": event} if event is not None and atomic else {}
    won = await release(run_id, status=status, claim_epoch=epoch, **kwargs)
    if won and event is not None and not atomic:
        await store.save_run_event(
            run_id,
            event_type=event.event_type,
            data=event.data,
            event_id=event.event_id,
            claim_epoch=epoch,
        )
    return bool(won)


async def release_claim(
    store: StateStore,
    run_id: str,
    status: str,
    *,
    claim_epoch: int | None = None,
    failure: BaseException | None = None,
) -> bool:
    """Drain claim release and report whether it succeeded."""
    import logging

    from dendrux.runtime.tasks import _drain_cleanup

    epoch = claim_epoch if claim_epoch is not None else getattr(store, "claim_epoch", None)
    release = getattr(store, "release_run_claim", None)
    if not _release_supported(store, claim_epoch=epoch):
        _codes.log_code(logging.getLogger(__name__), _codes.PERSISTENCE_ERROR, run_id=run_id)
        return False
    assert isinstance(epoch, int)
    event = _resume_aborted_event(status, failure)

    async def cleanup() -> bool:
        try:
            if await _release_with_event(store, release, run_id, status, epoch, event):
                return True
        except (Exception, asyncio.CancelledError):
            pass
        _codes.log_code(logging.getLogger(__name__), _codes.PERSISTENCE_ERROR, run_id=run_id)
        return False

    released, cancelled = await _drain_cleanup(
        cleanup(), name=f"release-claim:{run_id}", run_id=run_id
    )
    if cancelled is not None:
        raise cancelled
    return released


def build_rejection_results(
    pause_state: PauseState, rejection_reason: str | None = None
) -> list[ToolResult]:
    """Build synthetic failed :class:`ToolResult` entries for every pending call.

    Used by :meth:`Agent.submit_approval` when ``approved=False`` so the
    rejection is visible to the LLM as failed tool output and the agent
    decides the next step (retry, alternative, give up).

    The rejection reason is encoded in ``payload`` as a JSON string so the
    LLM sees it as tool-message content. ``error`` carries the same value
    for observability.
    """
    import json

    reason = rejection_reason or DEFAULT_DENIAL_REASON
    payload = json.dumps(reason)
    return [
        ToolResult(
            name=tc.name,
            call_id=tc.id,
            payload=payload,
            success=False,
            error=reason,
            duration_ms=0,
        )
        for tc in pause_state.pending_tool_calls
    ]


def validate_denial_results(agent: Agent, run_id: str, results: list[ToolResult]) -> None:
    """Reject oversized synthetic denials before claiming the pause."""
    if any(
        len(result.payload.encode("utf-8")) > agent._limits.max_tool_result_bytes
        for result in results
    ):
        raise _invalid_tool_result(run_id, "denial result exceeds the result limit")
    _validated_client_results(agent, run_id, results)


def decision_denial_results(
    pause: PauseState, decisions: Mapping[str, ApprovalDecision]
) -> list[ToolResult]:
    import json

    return [
        ToolResult(
            name=call.name,
            call_id=call.id,
            payload=json.dumps({"denied": decisions[call.id].reason or DEFAULT_DENIAL_REASON}),
            success=False,
            error=decisions[call.id].reason or DEFAULT_DENIAL_REASON,
        )
        for call in pause.pending_tool_calls
        if call.id in decisions and not decisions[call.id].approved
    ]


def serialize_tool_results(results: list[ToolResult]) -> list[dict[str, Any]]:
    """Match the wire shape expected by ``submitted_tool_results`` in pause_data.

    ``images`` (descriptor dicts) is added only for results with images, which
    must already be ingested.
    """
    out: list[dict[str, Any]] = []
    for r in results:
        entry: dict[str, Any] = {
            "name": r.name,
            "call_id": r.call_id,
            "payload": r.payload,
            "success": r.success,
            "error": r.error,
            "duration_ms": r.duration_ms,
        }
        if r.images:
            entry["images"] = [content_part_to_dict(_as_part(img)) for img in r.images]
        out.append(entry)
    return out


def deserialize_tool_results(raw: list[dict[str, Any]]) -> list[ToolResult]:
    """Rebuild ``submitted_tool_results`` entries (images as stored refs)."""
    results: list[ToolResult] = []
    for r in raw:
        images = tuple(
            part
            for part in (content_part_from_dict(d) for d in r.get("images") or [])
            if isinstance(part, ImagePart)
        )
        results.append(
            ToolResult(
                name=r["name"],
                call_id=r["call_id"],
                payload=r["payload"],
                success=r.get("success", True),
                error=r.get("error"),
                duration_ms=r.get("duration_ms", 0),
                images=images,
            )
        )
    return results


def _as_part(img: object) -> ImagePart:
    if isinstance(img, ImagePart):
        return img
    if isinstance(img, ImageRef):
        return ImagePart(ref=img)
    raise TypeError("tool result images must be ingested before they are persisted")


# ---------------------------------------------------------------------------
# Images at the entry points (ingest before the first durable write)
# ---------------------------------------------------------------------------


async def effective_image_capability(provider: LLMProvider, model: str, policy: ImagePolicy) -> str:
    """``"yes" | "no" | "unknown"`` for new images on ``model`` (§5.6).

    Policy overrides win, then a ``"no"`` the provider instance learned from
    an image-unsupported response, then ``provider.image_input_support``.
    ``"unknown"`` with ``unknown_capability="error"`` counts as ``"no"``. A
    provider without ``image_input_support`` reports from
    ``capabilities.supports_multimodal``.
    """
    override = policy.capability_overrides.get(model)
    if override in ("yes", "no"):
        return override
    learned = getattr(provider, "learned_image_support", None)
    if learned is not None and learned(model) == "no":
        return "no"
    support = getattr(provider, "image_input_support", None)
    if support is not None:
        value = await support(model)
    else:
        caps = getattr(provider, "capabilities", None)
        value = "unknown" if getattr(caps, "supports_multimodal", False) else "no"
    if value == "unknown" and policy.unknown_capability == "error":
        return "no"
    return str(value)


async def check_new_image_capability(
    provider: LLMProvider,
    model: str,
    policy: ImagePolicy,
    *,
    image_count: int,
    origin: str,
) -> None:
    """Raise ``UnsupportedModalityError`` before any write when new images cannot be sent.

    Only ``on_unsupported="error"`` raises; a placeholder or describe policy
    lets the request view degrade the images instead.
    """
    if image_count <= 0 or policy.on_unsupported != "error":
        return
    if await effective_image_capability(provider, model, policy) != "no":
        return
    from dendrux.errors import UnsupportedModalityError

    source = "capability"
    if policy.capability_overrides.get(model) != "no":
        source_of = getattr(provider, "image_capability_source", None)
        if source_of is not None:
            source = str(source_of())
        elif type(provider).__name__.startswith("OpenRouter"):
            source = "catalog"
    raise UnsupportedModalityError(
        provider=type(provider).__name__,
        model=model,
        image_count=image_count,
        origin=origin,
        source=source,
    )


@dataclass(frozen=True)
class IngestGroup:
    """One message's (or tool result's) image inputs for :func:`ingest_groups`."""

    images: tuple[Any, ...]
    origin: Origin
    tool_name: str | None = None


async def ingest_groups(
    groups: Sequence[IngestGroup],
    *,
    runtime: ImageRuntime,
    before_store: Callable[[], Awaitable[None]] | None = None,
) -> list[tuple[ImagePart, ...]]:
    """Ingest several groups with ONE phase 1 for all of them before any phase 2 (§5.2).

    Each group is its own ``ingest()`` call (its own per-message cap, origin
    and tool name); the calls nest through ``before_store``, so every group
    is validated before any blob is written. ``before_store`` (the entry
    point's capability check) runs last, between the phases. Returns the
    canonical parts per group, in order.
    """
    from dendrux.media._ingest import ingest

    results: list[tuple[ImagePart, ...]] = [() for _ in groups]
    todo = [i for i, g in enumerate(groups) if g.images]

    async def _step(k: int) -> None:
        if k == len(todo):
            if before_store is not None:
                await before_store()
            return
        group = groups[todo[k]]

        async def _next() -> None:
            await _step(k + 1)

        results[todo[k]] = await ingest(
            list(group.images),
            policy=runtime.policy,
            runtime=runtime,
            origin=group.origin,
            tool_name=group.tool_name,
            before_store=_next,
        )

    await _step(0)
    return results


def user_items_group(items: Sequence[Any], origin: Origin = "user") -> IngestGroup:
    """The image inputs of a ``UserContent`` sequence."""
    return IngestGroup(tuple(item for item in items if not isinstance(item, str)), origin)


def merge_user_items(items: Sequence[Any], images: Sequence[ImagePart]) -> tuple[ContentPart, ...]:
    """Re-interleave ingested images with the text items, in caller order."""
    ingested = iter(images)
    return tuple(TextPart(item) if isinstance(item, str) else next(ingested) for item in items)


async def ingest_user_items(
    items: Sequence[Any],
    *,
    runtime: ImageRuntime,
    origin: Origin = "user",
    before_store: Callable[[], Awaitable[None]] | None = None,
) -> tuple[ContentPart, ...]:
    """Ingest the images of a ``UserContent`` sequence; return canonical parts in order."""
    (images,) = await ingest_groups(
        [user_items_group(items, origin)], runtime=runtime, before_store=before_store
    )
    return merge_user_items(items, images)


async def ingest_tool_result_images(
    results: Sequence[ToolResult],
    *,
    runtime: ImageRuntime,
    before_store: Callable[[], Awaitable[None]] | None = None,
) -> list[ToolResult]:
    """Ingest client-tool result images (``origin="client_tool"``), in order.

    All results are validated before any blob is written; ``before_store``
    runs once, between the phases. Results without images are unchanged.
    """
    from dataclasses import replace

    groups = [IngestGroup(tuple(r.images), "client_tool", r.name) for r in results]
    parts = await ingest_groups(groups, runtime=runtime, before_store=before_store)
    return [replace(r, images=p) if r.images else r for r, p in zip(results, parts, strict=True)]


def parts_edges(
    parts: Sequence[ContentPart], *, tenant_scope: str, owner_kind: str, owner_id: str
) -> list[BlobEdge]:
    """Edges for the image parts of one persisted parts list (ordinal = index in it)."""
    from dendrux.media._catalog import BlobEdge

    return [
        BlobEdge(tenant_scope, part.ref.sha256, owner_kind, owner_id, i)
        for i, part in enumerate(parts)
        if isinstance(part, ImagePart)
    ]


def tool_result_edges(
    results: Sequence[ToolResult], *, tenant_scope: str, iteration: int
) -> list[BlobEdge]:
    """Submission edges of client-tool results: ``f"{iteration}:{call_id}"`` / image index."""
    from dendrux.media._catalog import BlobEdge

    return [
        BlobEdge(
            tenant_scope, _as_part(img).ref.sha256, "submission", f"{iteration}:{r.call_id}", i
        )
        for r in results
        for i, img in enumerate(r.images)
    ]


# ---------------------------------------------------------------------------
# Per-call approval decisions
# ---------------------------------------------------------------------------


def is_client_call(
    pause_state: PauseState,
    tc: ToolCall,
    target_lookup: Mapping[str, ToolTarget] | None = None,
) -> bool:
    """True when a pending call must never execute on the server.

    Fail-safe routing: the call is client-side if either its persisted
    pause-time target or the current tool definition says non-server.
    """
    persisted = pause_state.pending_targets.get(tc.id)
    if persisted is not None and persisted != ToolTarget.SERVER.value:
        return True
    current = (target_lookup or {}).get(tc.name)
    return current is not None and current != ToolTarget.SERVER


def approval_gates_of(
    pause_state: PauseState,
    target_lookup: Mapping[str, ToolTarget] | None = None,
) -> dict[str, dict[str, str | None]]:
    """Approval-gated call ids of a WAITING_APPROVAL pause.

    Uses only the snapshot persisted at pause time; policy is never
    recomputed. Pauses persisted before per-call approvals carry no
    snapshot: every server-side pending call is then treated as gated
    (source ``"legacy"``), so nothing runs without an explicit decision.
    """
    if pause_state.approval_gates is not None:
        return {k: dict(v) for k, v in pause_state.approval_gates.items()}
    return {
        tc.id: {"source": "legacy", "reason": None}
        for tc in pause_state.pending_tool_calls
        if not is_client_call(pause_state, tc, target_lookup)
    }


def _check_decisions_size(decisions: Mapping[Any, Any]) -> None:
    if len(decisions) > _MAX_DECISIONS:
        raise ValueError("decisions exceeds its count limit")


def normalize_decisions(decisions: object) -> list[tuple[str, ApprovalDecision]]:
    """Type-check a public ``decisions`` mapping into ``(call_id, decision)`` pairs."""
    if not isinstance(decisions, Mapping):
        raise ValueError(
            "decisions must be a mapping of tool call id to ApprovalDecision, "
            f"got {type(decisions).__name__}."
        )
    from dendrux.runtime._entry_checks import require_boolean
    from dendrux.types import _validate_identifier, _validate_persistable

    _check_decisions_size(decisions)
    pairs: list[tuple[str, ApprovalDecision]] = []
    for call_id, decision in decisions.items():
        _validate_identifier(call_id, "call_id")
        if not isinstance(call_id, str):
            raise ValueError(f"decisions keys must be tool call id strings, got {call_id!r}.")
        if not isinstance(decision, ApprovalDecision):
            raise ValueError(
                f"decisions[{call_id!r}] must be an ApprovalDecision, "
                f"got {type(decision).__name__}."
            )
        require_boolean(decision.approved, "approved")
        _validate_persistable(decision.reason, "reason")
        pairs.append((call_id, decision))
    return pairs


def validate_approval_decisions(
    run_id: str,
    pause_state: PauseState,
    pairs: Iterable[tuple[str, ApprovalDecision]],
    target_lookup: Mapping[str, ToolTarget] | None = None,
) -> dict[str, ApprovalDecision]:
    """Check decisions cover exactly the approval-gated calls of the pause.

    Raises ``ValueError`` on duplicate, missing or unknown call ids — no
    partial acceptance. Returns the decisions keyed by call id.
    """
    pair_list = list(pairs)
    gated = set(approval_gates_of(pause_state, target_lookup))
    counts = Counter(call_id for call_id, _ in pair_list)
    problems: list[str] = []
    duplicates = sorted(k for k, v in counts.items() if v > 1)
    if duplicates:
        problems.append(f"duplicate call_ids={duplicates}")
    missing = sorted(gated - set(counts))
    if missing:
        problems.append(f"missing call_ids={missing}")
    unknown = sorted(set(counts) - gated)
    if unknown:
        problems.append(f"unknown call_ids={unknown}")
    if problems:
        raise ValueError(
            f"Approval decisions for run '{run_id}' must cover exactly the "
            f"approval-gated calls: {'; '.join(problems)}."
        )
    return dict(pair_list)


def serialize_decisions(
    pause_state: PauseState, decisions: Mapping[str, ApprovalDecision]
) -> list[dict[str, Any]]:
    """Wire shape of ``submitted_decisions`` in pause_data, in pending order."""
    order = {tc.id: i for i, tc in enumerate(pause_state.pending_tool_calls)}
    return [
        {"call_id": call_id, "approved": d.approved, "reason": d.reason}
        for call_id, d in sorted(decisions.items(), key=lambda kv: order.get(kv[0], len(order)))
    ]


def deserialize_decisions(raw: object) -> list[tuple[str, ApprovalDecision]]:
    """Parse persisted ``submitted_decisions`` back into pairs (duplicates kept)."""
    if not isinstance(raw, list):
        raise ValueError("submitted_decisions must be a list.")
    pairs: list[tuple[str, ApprovalDecision]] = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("call_id"), str):
            raise ValueError("submitted_decisions entries need a string call_id.")
        reason = item.get("reason")
        pairs.append(
            (
                item["call_id"],
                ApprovalDecision(
                    approved=item.get("approved") is True,
                    reason=reason if isinstance(reason, str) else None,
                ),
            )
        )
    return pairs


def _invalid_json_constant(value: str) -> NoReturn:
    raise ValueError("non-finite JSON number")


def _governed_limits(limits: Any) -> dict[str, Any]:
    from dendrux.types import Limits

    defaults = Limits()
    recovery_bounds = {
        "max_pending_calls",
        "max_rebuild_bytes",
        "max_rebuild_rows",
    }
    # Default recovery bounds preserve older governance snapshots; overrides remain governed.
    return {
        key: value
        for key, value in asdict(limits).items()
        if key != "max_pause_summary_bytes"
        and (key not in recovery_bounds or value != getattr(defaults, key))
    }


def governance_snapshot(agent: Agent) -> dict[str, Any]:
    """Versioned digests of stable governance configuration, never object reprs.

    Budget amounts are intentionally excluded so a caller can raise them when
    resuming a budget pause. Opaque custom settings opt in with a fingerprint.
    """
    import hashlib
    import json

    def qualified(value: Any) -> str:
        return (
            str(getattr(value, "__module__", type(value).__module__))
            + "."
            + str(getattr(value, "__qualname__", type(value).__qualname__))
        )

    def digest(value: Any) -> str:
        wire = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(wire.encode("utf-8")) > _GOVERNANCE_SNAPSHOT_BYTES:
            raise ValueError("Governance fingerprint exceeds 64 KiB")
        return hashlib.sha256(wire.encode("utf-8")).hexdigest()

    guards = []
    for guard in agent.guardrails or ():
        detectors = getattr(guard, "_detectors", None)
        if detectors is None:
            detectors = getattr(getattr(guard, "_scanner", None), "_detectors", ())
        entities = getattr(guard, "entities", None)
        if entities is None:
            entities = [det.name for det in detectors or ()]
        fingerprint = getattr(guard, "governance_fingerprint", None)
        guards.append(
            {
                "class": qualified(type(guard)),
                "action": guard.action,
                "entities": sorted(set(entities)),
                "engine": getattr(guard, "engine", None),
                "restore": getattr(guard, "restore", None),
                "fingerprint": fingerprint() if fingerprint is not None else None,
            }
        )
    budget = agent.budget
    limits = _governed_limits(agent._limits)
    parts = {
        "guardrails": guards,
        "before_tool": qualified(agent.before_tool) if agent.before_tool else None,
        "deny": sorted(agent.deny),
        "require_approval": sorted(agent.require_approval),
        "limits": limits,
        "budget": (
            {
                "on_exceed": budget.on_exceed,
                "include_children": budget.include_children,
                "count_cache_tokens": bool(budget.count_cache_tokens)
                if budget.max_tokens is not None
                else False,
            }
            if budget
            else None
        ),
    }
    hashed = {key: digest(value) for key, value in parts.items()}
    return {"version": 1, "digest": digest(hashed), "parts": hashed}


def _governance_change(agent: Agent, raw: dict[str, Any]) -> dict[str, Any] | None:
    saved = raw.get(GOVERNANCE_KEY)
    if saved is None:
        return None
    actual = governance_snapshot(agent)
    if saved.get("digest") == actual["digest"]:
        return None
    changed = sorted(
        key for key, value in actual["parts"].items() if saved.get("parts", {}).get(key) != value
    )
    return {
        "expected_digest": saved.get("digest"),
        "actual_digest": actual["digest"],
        "changed": changed,
    }


def submission_validator(
    run_id: str,
    raw: dict[str, Any],
    *,
    agent: Agent | None,
    expected_pause_id: str | None = None,
    call_results: list[ToolResult] | None = None,
    extra: Callable[[dict[str, Any]], None] | None = None,
    client_results: bool = False,
) -> Callable[[dict[str, Any]], None]:
    """Validate before claiming, then repeat under the claim transaction's lock."""
    from dendrux.errors import GovernanceMismatchError

    # Even an optional-token client is bound to the generation we observed.
    generation = _pause_generation(raw, expected_pause_id)
    base = pause_validator(
        run_id,
        generation,
        agent_name=agent.name if agent else None,
        call_results=call_results,
        client_results=client_results,
    )

    def validate(current: dict[str, Any]) -> None:
        from dendrux.types import _validate_provider_kwargs

        if current.get("schema_version") == 2:
            _validate_provider_kwargs(current.get("provider_kwargs") or {}, run_id=run_id)
        base(current)
        if agent is not None and agent._on_governance_change == "raise":
            change = _governance_change(agent, current)
            if change is not None:
                import logging

                _codes.log_code(
                    logging.getLogger(__name__), _codes.GOVERNANCE_MISMATCH, run_id=run_id
                )
                raise GovernanceMismatchError(run_id, **change)
        if extra is not None:
            extra(current)

    validate(raw)
    return validate


async def report_governance_change(
    agent: Agent,
    store: StateStore,
    run_id: str,
    raw: dict[str, Any],
) -> None:
    import logging

    log = logging.getLogger(__name__)
    if GOVERNANCE_KEY not in raw:
        _codes.log_code(log, _codes.GOVERNANCE_LEGACY, run_id=run_id)
        return
    change = _governance_change(agent, raw)
    if change is None:
        return
    _codes.log_code(log, _codes.GOVERNANCE_CHANGED, run_id=run_id)
    await store.save_run_event(
        run_id, event_type=GovernanceEventType.GOVERNANCE_CHANGED.value, data=change
    )


class _PauseIdRequiredError(InputValidationError):
    def __init__(self, field: str, code: str) -> None:
        super().__init__(field, code)
        self.args = (str(self) + " Send the pause_id from get_pending as expected_pause_id.",)


def _pause_id_required() -> ValueError:
    return _PauseIdRequiredError("expected_pause_id", "pause_id_required")


def _require_approval_token(approval: bool, expected: str | None) -> None:
    from dendrux.types import _validate_identifier

    if expected is not None:
        _validate_identifier(expected, "expected_pause_id", maximum=ULID_CHARS)
    if approval and expected is None:
        raise _pause_id_required()


def require_pause_id(
    agent: Agent,
    expected: str | None,
    *,
    pause_id: str | None,
    approval: bool = False,
    store: StateStore | None = None,
    run_id: str | None = None,
) -> None:
    """Require a token only when the stored pause has a durable generation."""
    _require_approval_token(approval and pause_id is not None, expected)
    if pause_id is not None and expected is None and _strict_tenancy_applies(agent, store):
        raise _pause_id_required()
    if run_id is not None:
        pause_validator(run_id, expected)({"pause_id": pause_id})


def _refuse_tokenless_managed(run_id: str, checkpoint: dict[str, Any] | None) -> None:
    """A managed run's pause is claimed only with its explicit token (0.7).

    Decided at claim admission from the run's persisted managed identity, not
    the caller's Agent: an Agent rebuilt without registrations cannot apply
    the ordinary token rule to a managed root or child.
    """
    from dendrux.runtime.state import MANAGED_HEADER_KEY

    if (checkpoint or {}).get(MANAGED_HEADER_KEY) is not None:
        import logging

        _codes.log_code(
            logging.getLogger(__name__), _codes.MANAGED_PAUSE_ID_REQUIRED, run_id=run_id
        )
        raise _pause_id_required()


def _client_json_payload(payload: str) -> str:
    """JSON form for caps and transform input, without changing model text."""
    import json

    try:
        json.loads(payload, parse_constant=_invalid_json_constant)
    except (ValueError, RecursionError):
        return json.dumps(payload)
    return payload


def _cap_client_result(result: ToolResult, limit: int) -> ToolResult:
    """Apply the tool cap to an already validated client result."""
    from dendrux.tools import cap_tool_result

    return cap_tool_result(result, limit)


async def prepare_client_results(
    agent: Agent, run_id: str, pause: PauseState, results: list[ToolResult]
) -> list[ToolResult]:
    """Validate identity and cap; preserve wire bytes unless a transform applies."""
    import json

    from dendrux.loops.react import _apply_result_transform

    results = await _validate_client_results(agent, run_id, results)
    validate_result_names(run_id, pause, results)
    try:
        capped = [
            _cap_client_result(result, agent._limits.max_tool_result_bytes) for result in results
        ]
    except UnicodeError as exc:
        raise _invalid_tool_result(run_id, "tool result payload is not valid UTF-8") from exc
    validate_result_names(run_id, pause, capped)
    lookups = await agent.get_tool_lookups()
    pending = {call.id: call for call in pause.pending_tool_calls}
    prepared = []
    for result in capped:
        if result.meta.get("code") == "size_limit":
            import logging

            _codes.log_code(logging.getLogger(__name__), _codes.SIZE_LIMIT, run_id=run_id)
        if result.success and agent.tool_result_transform is not None:
            try:
                normalized = _client_json_payload(result.payload)
                value = await _apply_result_transform(
                    agent.tool_result_transform,
                    pending[result.call_id],
                    lookups,
                    json.loads(normalized, parse_constant=_invalid_json_constant),
                )
                # A string returned for legacy plain text is already model text.
                # JSON inputs and structured outputs retain JSON serialization.
                payload = (
                    value
                    if normalized != result.payload and isinstance(value, str)
                    else json.dumps(value, allow_nan=False)
                )
                result = replace(result, payload=payload)
            except Exception as exc:
                raise _invalid_tool_result(run_id, "client tool result transform failed") from exc
            result = (await _validate_client_results(agent, run_id, [result]))[0]
            result = _cap_client_result(result, agent._limits.max_tool_result_bytes)
        prepared.append(result)
    return prepared


def legacy_approval_denial(
    run_id: str, pause: PauseState, results: list[ToolResult]
) -> list[ToolResult]:
    """Legacy rejection is a denial, never proof that server code executed."""
    validate_result_names(run_id, pause, results)
    if any(result.success is not False for result in results):
        raise _invalid_tool_result(run_id, "approval tool results must be failed denials")
    return build_rejection_results(pause)


async def prepare_resume_results(
    agent: Agent, store: StateStore, run_id: str, pause: PauseState, results: list[ToolResult]
) -> list[ToolResult]:
    if await _resume_status(store, run_id) == RunStatus.WAITING_APPROVAL.value:
        results = legacy_approval_denial(
            run_id, pause, await _validate_client_results(agent, run_id, results)
        )
    return await prepare_client_results(agent, run_id, pause, results)


async def _resume_status(store: StateStore, run_id: str) -> str | None:
    projected = getattr(store, "get_run_status", None)
    if projected is not None:
        return cast("str | None", await projected(run_id))
    record = await store.get_run(run_id)
    return record.status if record is not None else None


def _check_result_identities(run_id: str, results: list[ToolResult]) -> None:
    if not isinstance(results, list) or len(results) > _MAX_RESULTS:
        raise _invalid_tool_result(run_id, "tool results exceed the batch limit")
    for result in results:
        if not isinstance(result, ToolResult) or any(
            not isinstance(value, str) or not value or len(value) > _MAX_RESULT_ID_CHARS
            for value in (result.name, result.call_id)
        ):
            raise _invalid_tool_result(run_id, "invalid tool result identity")


_CLIENT_VALIDATION_THREAD_BYTES = 65536
_VALIDATION_GATES: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, weakref.ReferenceType[asyncio.Lock]
] = weakref.WeakKeyDictionary()
_VALIDATION_GATES_LOCK = threading.Lock()


def _validation_gate() -> asyncio.Lock:
    """The per-loop gate for off-loop client-result validation. Its wait queue
    is bounded by the caller's own concurrency (one waiter per in-flight submit
    call), so it needs no separate cap."""
    loop = asyncio.get_running_loop()
    with _VALIDATION_GATES_LOCK:
        reference = _VALIDATION_GATES.get(loop)
        gate = reference() if reference is not None else None
        if gate is None:
            gate = asyncio.Lock()
            _VALIDATION_GATES[loop] = weakref.ref(gate)
        return gate


async def _finish_validation(worker: asyncio.Task[list[ToolResult]]) -> None:
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    if not worker.cancelled():
        worker.exception()


async def _thread_client_results(
    agent: Agent, run_id: str, results: list[ToolResult]
) -> list[ToolResult]:
    async with _validation_gate():
        worker = asyncio.create_task(
            asyncio.to_thread(_validated_client_results, agent, run_id, results)
        )
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            # Hold the admission slot until the CPU job really ends. Cancelling
            # an asyncio waiter cannot stop a running default-executor thread.
            await _finish_validation(worker)
            raise


async def _validate_client_results(
    agent: Agent, run_id: str, results: list[ToolResult]
) -> list[ToolResult]:
    _check_result_identities(run_id, results)
    size = sum(
        len(result.payload) * (1 if result.payload.isascii() else 4)
        for result in results
        if isinstance(result.payload, str)
    )
    if size >= _CLIENT_VALIDATION_THREAD_BYTES:
        return await _thread_client_results(agent, run_id, results)
    return _validated_client_results(agent, run_id, results)


def _validated_client_results(
    agent: Agent, run_id: str, results: list[ToolResult]
) -> list[ToolResult]:
    _check_result_identities(run_id, results)
    checked = []
    total = 0
    for result in results:
        try:
            if len(result.images) > _MAX_RESULT_IMAGES:
                raise ValueError("image count limit")
            data = {
                "name": result.name,
                "call_id": result.call_id,
                "payload": result.payload,
                "success": result.success,
                "error": result.error,
                "duration_ms": result.duration_ms,
            }
            from dendrux._wire import tool_result_from_dict

            parsed = tool_result_from_dict(
                data, agent._limits, _validated_payload=result._validated_payload
            )
            total += len(parsed.payload.encode("utf-8"))
            if total > agent._limits.max_input_bytes:
                raise ValueError("batch limit")
        except (ValueError, TypeError, RecursionError):
            raise _invalid_tool_result(
                run_id, "tool result fields violate the client boundary"
            ) from None
        checked.append(replace(parsed, images=result.images))
    return checked
