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

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Any, NoReturn

from dendrux.errors import (
    InvalidToolResultError,
    PauseStatusMismatchError,
    RunAlreadyClaimedError,
    RunAlreadyTerminalError,
    RunNotFoundError,
    RunNotPausedError,
)
from dendrux.types import (
    ApprovalDecision,
    ImagePart,
    ImageRef,
    PauseState,
    RunStatus,
    TextPart,
    ToolCall,
    ToolResult,
    ToolTarget,
    content_part_from_dict,
    content_part_to_dict,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Sequence

    from dendrux.agent import Agent
    from dendrux.llm.base import LLMProvider
    from dendrux.media._catalog import BlobEdge
    from dendrux.media._policy import ImagePolicy, Origin
    from dendrux.media._runtime import ImageRuntime
    from dendrux.runtime.state import StateStore
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

DEFAULT_DENIAL_REASON = "User declined to run this tool."


_TERMINAL_STATUSES = frozenset(
    {
        RunStatus.SUCCESS,
        RunStatus.ERROR,
        RunStatus.CANCELLED,
        RunStatus.MAX_ITERATIONS,
    }
)

_PAUSE_STATUSES = frozenset(
    {
        RunStatus.WAITING_CLIENT_TOOL,
        RunStatus.WAITING_HUMAN_INPUT,
        RunStatus.WAITING_APPROVAL,
        RunStatus.PAUSED,
    }
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

    logging.getLogger(__name__).warning(
        "Invalid client tool result: run_id=%s code=invalid_tool_result",
        run_id,
        extra={"run_id": run_id, "code": "invalid_tool_result"},
    )
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
    pending_ids = {tc.id for tc in pause_state.pending_tool_calls}
    provided_counts = Counter(r.call_id for r in results)
    duplicates = sorted(k for k, v in provided_counts.items() if v > 1)
    provided_ids = set(provided_counts)

    parts: list[str] = []
    if duplicates:
        parts.append(f"duplicate call_ids={duplicates}")
    missing = sorted(pending_ids - provided_ids)
    extra = sorted(provided_ids - pending_ids)
    if missing:
        parts.append(f"missing call_ids={missing}")
    if extra:
        parts.append(f"unexpected call_ids={extra}")

    if parts:
        raise _invalid_tool_result(run_id, "; ".join(parts))
    validate_result_names(run_id, pause_state, results, json_payload=False)
    return pause_state


def validate_result_names(
    run_id: str, pause: PauseState, results: list[ToolResult], *, json_payload: bool = True
) -> None:
    """Bind result IDs/names and require a string payload; text is normalized later."""

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
) -> Callable[[dict[str, Any]], None]:
    """Validation performed while the claim transaction holds the run row."""

    def validate(current: dict[str, Any]) -> None:
        current_id = current.get("pause_id")
        if expected_pause_id is not None and expected_pause_id != current_id:
            import logging

            logging.getLogger(__name__).warning(
                "Pause generation mismatch: run_id=%s code=pause_id_mismatch",
                run_id,
                extra={"run_id": run_id, "code": "pause_id_mismatch"},
            )
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
            pending = {call["id"]: call["name"] for call in current.get("pending_tool_calls", [])}
            if (
                len(pending) != len(call_results)
                or len({r.call_id for r in call_results}) != len(call_results)
                or any(pending.get(result.call_id) != result.name for result in call_results)
            ):
                raise _invalid_tool_result(run_id, "tool results do not match pending calls")

    return validate


async def preflight_status(store: StateStore, run_id: str, expected_status: RunStatus) -> None:
    """Raise the most specific error unless ``run_id`` is in ``expected_status``.

    Also refuses a run whose cancellation was requested but not yet observed:
    it is on the path to CANCELLED and resuming it would bounce off the next
    checkpoint.
    """
    record = await store.get_run(run_id)
    if record is None:
        raise RunNotFoundError(run_id)
    current = RunStatus(record.status)
    if current != expected_status:
        if current in _TERMINAL_STATUSES:
            raise RunAlreadyTerminalError(run_id, current)
        if current in _PAUSE_STATUSES:
            raise PauseStatusMismatchError(run_id, current, expected_status)
        # RUNNING, PENDING, or anything else non-paused.
        raise RunNotPausedError(run_id, current)
    if record.cancel_requested:
        raise RunAlreadyTerminalError(run_id, RunStatus.CANCELLED)


@dataclass
class _ClaimValidationSupport:
    atomic: bool
    logged: bool = False


@lru_cache(maxsize=128)
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
) -> int:
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
        logging.getLogger(__name__).warning(
            "Custom store lacks atomic claim validation; using pre/post checks: "
            "run_id=%s code=legacy_claim_validation",
            run_id,
            extra={"run_id": run_id, "code": "legacy_claim_validation"},
        )
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
        except BaseException:
            release = getattr(store, "release_run_claim", None)
            if release is not None and type(epoch) is int:
                import anyio

                with anyio.CancelScope(shield=True):
                    await release(run_id, status=expected_status, claim_epoch=epoch)
            raise
    return epoch


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
) -> int:
    """Perform a status preflight, then persist-first handoff + CAS claim.

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
        warn_missing_pause_id(
            agent, expected_pause_id, pause_id=raw.get("pause_id"), approval=approval
        )
    validate = submission_validator(
        run_id,
        raw,
        agent=agent,
        expected_pause_id=expected_pause_id,
        call_results=call_results,
        extra=validate,
    )

    # --- Claim: run was expected_status at preflight, attempt CAS. ---
    if submitted_data is not None:
        claim_kwargs: dict[str, Any] = {} if validate is None else {"validate": validate}
        if blob_refs:
            claim_kwargs["blob_refs"] = list(blob_refs)
            claim_kwargs["blob_refs_scope"] = blob_refs_scope
        won = await store.submit_and_claim(
            run_id,
            expected_status=expected_status.value,
            submitted_data=submitted_data,
            **claim_kwargs,
        )
    else:
        won = await claim_paused_validated(
            store,
            run_id,
            expected_status=expected_status.value,
            validate=validate,
        )

    if won:
        if agent is not None:
            try:
                await report_governance_change(agent, store, run_id, raw)
            except BaseException:
                release = getattr(store, "release_run_claim", None)
                if release is not None and type(won) is int:
                    import anyio

                    with anyio.CancelScope(shield=True):
                        await release(run_id, status=expected_status.value, claim_epoch=won)
                raise
        return won

    # CAS lost after preflight succeeded — something transitioned the run
    # between the two reads. Re-check to pick the most specific error.
    record = await store.get_run(run_id)
    if record is None:
        # Extremely unlikely — run vanished mid-submit.
        raise RunNotFoundError(run_id)

    current = RunStatus(record.status)
    if current in _TERMINAL_STATUSES:
        raise RunAlreadyTerminalError(run_id, current)
    if current == RunStatus.RUNNING:
        # We just observed ``expected_status``; RUNNING now means a
        # concurrent submit claimed it.
        raise RunAlreadyClaimedError(run_id)
    if current in _PAUSE_STATUSES:
        # Unusual — paused status changed between preflight and CAS.
        raise PauseStatusMismatchError(run_id, current, expected_status)
    raise RunNotPausedError(run_id, current)


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
                payload=_client_json_payload(r["payload"]),
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


def normalize_decisions(decisions: object) -> list[tuple[str, ApprovalDecision]]:
    """Type-check a public ``decisions`` mapping into ``(call_id, decision)`` pairs."""
    if not isinstance(decisions, Mapping):
        raise ValueError(
            "decisions must be a mapping of tool call id to ApprovalDecision, "
            f"got {type(decisions).__name__}."
        )
    pairs: list[tuple[str, ApprovalDecision]] = []
    for call_id, decision in decisions.items():
        if not isinstance(call_id, str):
            raise ValueError(f"decisions keys must be tool call id strings, got {call_id!r}.")
        if not isinstance(decision, ApprovalDecision):
            raise ValueError(
                f"decisions[{call_id!r}] must be an ApprovalDecision, "
                f"got {type(decision).__name__}."
            )
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


GOVERNANCE_KEY = "dendrux.governance"


def _invalid_json_constant(value: str) -> NoReturn:
    raise ValueError("non-finite JSON number")


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
        if len(wire.encode("utf-8")) > 65536:
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
    parts = {
        "guardrails": guards,
        "before_tool": qualified(agent.before_tool) if agent.before_tool else None,
        "deny": sorted(agent.deny),
        "require_approval": sorted(agent.require_approval),
        "limits": asdict(agent._limits),
        "budget": (
            {
                "on_exceed": budget.on_exceed,
                "include_children": budget.include_children,
                "count_cache_tokens": bool(budget.count_cache_tokens),
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
) -> Callable[[dict[str, Any]], None]:
    """Validate before claiming, then repeat under the claim transaction's lock."""
    from dendrux.errors import GovernanceMismatchError

    # Even an optional-token client is bound to the generation we observed.
    generation = expected_pause_id if expected_pause_id is not None else raw.get("pause_id")
    base = pause_validator(
        run_id, generation, agent_name=agent.name if agent else None, call_results=call_results
    )

    def validate(current: dict[str, Any]) -> None:
        base(current)
        if agent is not None and agent._on_governance_change == "raise":
            change = _governance_change(agent, current)
            if change is not None:
                import logging

                logging.getLogger(__name__).warning(
                    "Governance mismatch: run_id=%s code=governance_mismatch",
                    run_id,
                    extra={"run_id": run_id, "code": "governance_mismatch"},
                )
                raise GovernanceMismatchError(run_id, **change)
        if extra is not None:
            extra(current)

    validate(raw)
    return validate


@dataclass(frozen=True)
class _WarningLocation:
    filename: str
    lineno: int
    module: str
    registry: dict[str | tuple[str, type[Warning], int], int]


def capture_warning_location() -> _WarningLocation | None:
    """Keep a lazy stream's creation site without retaining the caller frame."""
    import inspect

    frame = inspect.currentframe()
    while frame is not None and frame.f_globals.get("__name__", "").startswith("dendrux."):
        frame = frame.f_back
    if frame is None:
        return None
    return _WarningLocation(
        frame.f_code.co_filename,
        frame.f_lineno,
        frame.f_globals.get("__name__", ""),
        frame.f_globals.setdefault("__warningregistry__", {}),
    )


def _warn_at_location(message: str, location: _WarningLocation | None) -> None:
    import warnings

    if location is not None:
        warnings.warn_explicit(
            message,
            FutureWarning,
            location.filename,
            location.lineno,
            module=location.module,
            registry=location.registry,
        )
    else:
        warnings.warn(message, FutureWarning, stacklevel=_user_warning_stacklevel())


def _user_warning_stacklevel() -> int:
    import inspect

    frame = inspect.currentframe()
    frame = frame.f_back if frame is not None else None
    level = 1
    while frame is not None and frame.f_globals.get("__name__", "").startswith("dendrux."):
        level += 1
        frame = frame.f_back
    return level


async def report_governance_change(
    agent: Agent,
    store: StateStore,
    run_id: str,
    raw: dict[str, Any],
    *,
    warning_location: _WarningLocation | None = None,
) -> None:
    import logging

    log = logging.getLogger(__name__)
    if GOVERNANCE_KEY not in raw:
        log.info(
            "Skipping pre-0.5 governance check: run_id=%s code=governance_legacy",
            run_id,
            extra={"run_id": run_id, "code": "governance_legacy"},
        )
        return
    change = _governance_change(agent, raw)
    if change is None:
        return
    log.warning(
        "Governance changed: run_id=%s code=governance_changed",
        run_id,
        extra={"run_id": run_id, "code": "governance_changed"},
    )
    if not agent._on_governance_change_chosen:
        _warn_at_location(
            "Changed governance warns in 0.5; the default becomes 'raise' in 0.6. "
            "Choose on_governance_change explicitly.",
            warning_location,
        )
    await store.save_run_event(run_id, event_type="governance.changed", data=change)


def warn_missing_pause_id(
    agent: Agent,
    expected: str | None,
    *,
    pause_id: str | None,
    approval: bool = False,
    warning_location: _WarningLocation | None = None,
) -> None:
    # A pre-0.5 pause has no token the caller could provide.
    if pause_id is not None and expected is None and (agent._strict_tenancy or approval):
        _warn_at_location(
            "Pass expected_pause_id to bind this submission to its pause. "
            "It is planned to be required in 0.6 for approvals.",
            warning_location,
        )


def _client_json_payload(payload: str) -> str:
    """Keep JSON values; encode legacy plain text as a JSON string."""
    import json

    try:
        json.loads(payload, parse_constant=_invalid_json_constant)
    except (ValueError, RecursionError):
        return json.dumps(payload)
    return payload


async def prepare_client_results(
    agent: Agent, run_id: str, pause: PauseState, results: list[ToolResult]
) -> list[ToolResult]:
    """Validate identity, normalize text to JSON, cap, transform, then re-cap."""
    import json

    from dendrux.loops.react import _apply_result_transform
    from dendrux.tools import cap_tool_result

    validate_result_names(run_id, pause, results, json_payload=False)
    try:
        capped = [
            cap_tool_result(
                replace(result, payload=_client_json_payload(result.payload)),
                agent._limits.max_tool_result_bytes,
            )
            for result in results
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

            logging.getLogger(__name__).warning(
                "Client tool result capped: run_id=%s code=size_limit",
                run_id,
                extra={"run_id": run_id, "code": "size_limit"},
            )
        if result.success and agent.tool_result_transform is not None:
            try:
                value = await _apply_result_transform(
                    agent.tool_result_transform,
                    pending[result.call_id],
                    lookups,
                    json.loads(result.payload, parse_constant=_invalid_json_constant),
                )
                result = replace(result, payload=json.dumps(value, allow_nan=False))
            except Exception as exc:
                raise _invalid_tool_result(run_id, "client tool result transform failed") from exc
            result = cap_tool_result(result, agent._limits.max_tool_result_bytes)
        prepared.append(result)
    return prepared
