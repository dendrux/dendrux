"""Enforce SQL persistence value contracts."""

from __future__ import annotations

import datetime as _dt
import json
import logging
from typing import TYPE_CHECKING, Any, NoReturn

from sqlalchemy import and_, select

from dendrux import _codes
from dendrux._lifecycle import (
    STORAGE_FORMAT_CHECKPOINTED,
    GovernanceEventType,
    StoredEventType,
)
from dendrux._meta_keys import (
    BLOCKED_INSERT_SESSION_KEY,
    BLOCKED_META_KEY,
    BLOCKED_RUNS_SESSION_KEY,
)
from dendrux.runtime.state import (
    _TERMINAL_STATUS_VALUES,
    MANAGED_HEADER_KEY,
    TraceRecord,
    TraceWrite,
    _finalized_output_data,
    _to_aware_utc,
    _usage_totals,
)
from dendrux.types import (
    _PROVIDER_KWARGS_MAX_BYTES,
    ULID_CHARS,
    RunStatus,
    UsageStats,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from dendrux.runtime.stores.sql._sql_host import _StoreHost


_RECOVERY_DESCRIPTOR_BYTES = 32768


_RECOVERY_TRACE_BYTES = 1024 * 1024


_RECOVERY_MAX_MESSAGES = 999


PAUSE_INVALIDATION_EVENT_TYPES = (
    StoredEventType.RUN_RESUMED.value,
    StoredEventType.RUN_COMPLETED.value,
    StoredEventType.RUN_ERROR.value,
    StoredEventType.RUN_CANCELLED.value,
)


logger = logging.getLogger("dendrux.runtime.stores.sql.base")


def _status_values(status: Any, *, initial: bool = False) -> dict[str, Any]:
    from sqlalchemy import case

    from dendrux.db.models import AgentRun

    now = _dt.datetime.now(_dt.UTC)
    return {
        "status": status,
        "status_changed_at": now
        if initial
        else case((AgentRun.status != status, now), else_=AgentRun.status_changed_at),
    }


def _advancing_turn(trace: TraceWrite | None) -> bool:
    return (
        trace is not None
        and trace.advances_progress
        and trace.trace_kind == "turn"
        and trace.role in ("assistant", "tool")
    )


def _checkpoint_progress(trace: TraceWrite | None) -> dict[str, Any]:
    from sqlalchemy import case, exists

    from dendrux.db.models import AgentRun, ReactTrace

    if not _advancing_turn(trace):
        return {}
    assert trace is not None
    # The existing fence also checks freshness, so a retried trace cannot
    # erase crashes accumulated since its original commit.
    fresh = ~exists(select(ReactTrace.id).where(ReactTrace.id == trace.trace_id))
    return {"reclaim_attempts": case((fresh, 0), else_=AgentRun.reclaim_attempts)}


def _progress_values(advancing: bool) -> dict[str, Any]:
    return {"reclaim_attempts": 0} if advancing else {}


def _submission_progress(current: dict[str, Any], traces: Sequence[TraceWrite]) -> dict[str, Any]:
    advancing = (current.get("submission") or {}).get("kind") == "tool_results" and any(
        _advancing_turn(trace) for trace in traces
    )
    return _progress_values(advancing)


def _final_progress(status: str, answer: str | None, claim_epoch: int | None) -> dict[str, Any]:
    from sqlalchemy import case

    from dendrux.db.models import AgentRun

    if status != RunStatus.SUCCESS.value or answer is None or claim_epoch is None:
        return {}
    return {
        "reclaim_attempts": case(
            (
                and_(
                    AgentRun.storage_format == STORAGE_FORMAT_CHECKPOINTED,
                    AgentRun.status.notin_(_TERMINAL_STATUS_VALUES),
                ),
                0,
            ),
            else_=AgentRun.reclaim_attempts,
        )
    }


async def _final_output_values(
    session: AsyncSession,
    run_id: str,
    answer: str | None,
    error: str | None,
    iteration_count: int | None,
    total_usage: UsageStats | None,
    pii_mapping: dict[str, str] | None,
    failure_reason: str | None,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if iteration_count is not None:
        values["iteration_count"] = iteration_count
    if answer is not None or total_usage is not None:
        values["output_data"] = await _finalized_output_data(session, run_id, answer, total_usage)
    if error is not None:
        values["error"] = error
    if total_usage is not None:
        values.update(_usage_totals(total_usage))
    if pii_mapping is not None:
        values["pii_mapping"] = pii_mapping
    if failure_reason is not None:
        values["failure_reason"] = failure_reason
    return values


def _merge_public_meta(meta: dict[str, Any] | None, patch: dict[str, Any]) -> dict[str, Any]:
    return {
        **(meta or {}),
        **{key: value for key, value in patch.items() if key != BLOCKED_META_KEY},
    }


def _creation_mark_proven(store: _StoreHost, run_id: str, meta: dict[str, Any] | None) -> bool:
    """A blocked creation's readback must also show the mark it committed with."""
    from dendrux.runtime._blocked_content import _blocked_target

    return _blocked_target.get() != (store, run_id) or (meta or {}).get(BLOCKED_META_KEY) is True


def _note_blocked_event(session: AsyncSession, run_id: str, event_type: str) -> None:
    """A ``guardrail.blocked`` event marks its run in the same transaction."""
    if event_type == GovernanceEventType.GUARDRAIL_BLOCKED.value:
        session.info.setdefault(BLOCKED_RUNS_SESSION_KEY, set()).add(run_id)


def _refuse_direct_managed_child(run_id: str, loop_state: Any) -> None:
    """Managed children are reclaimed through their root (0.7); refuse before the claim."""
    from dendrux.errors import ManagedChildControlError
    from dendrux.runtime.context import managed_root_reclaim

    header = (loop_state or {}).get(MANAGED_HEADER_KEY)
    if isinstance(header, dict) and "invocation" in header and not managed_root_reclaim(run_id):
        raise ManagedChildControlError(run_id)


def _creation_metadata(
    session: AsyncSession, store: _StoreHost, run_id: str, meta: dict[str, Any] | None
) -> dict[str, Any] | None:
    from dendrux.runtime._blocked_content import _blocked_target

    cleaned = (
        {key: value for key, value in meta.items() if key != BLOCKED_META_KEY} if meta else meta
    )
    if _blocked_target.get() == (store, run_id):
        session.info[BLOCKED_INSERT_SESSION_KEY] = run_id
        return {**(cleaned or {}), BLOCKED_META_KEY: True}
    return cleaned


def _check_create_identifiers(
    run_id: str,
    agent_name: str,
    tenant_id: str | None,
    idempotency_key: str | None,
    model: str | None,
) -> None:
    from dendrux.types import _validate_identifier

    _validate_identifier(run_id, "run_id", maximum=ULID_CHARS)
    _validate_identifier(agent_name, "agent_name")
    for name, optional in (
        ("tenant_id", tenant_id),
        ("idempotency_key", idempotency_key),
        ("model", model),
    ):
        if optional is not None:
            _validate_identifier(optional, name)


def _checkpoint_error(run_id: str) -> NoReturn:
    from dendrux.errors import PersistenceError

    _codes.log_code(logger, _codes.CHECKPOINT_TOO_LARGE, run_id=run_id)
    raise PersistenceError(code="checkpoint_too_large")


def _json_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _validate_checkpoint(run_id: str, value: dict[str, Any]) -> None:
    from dendrux.types import _validate_persistable

    _validate_persistable(value, "loop_state")
    if (
        _json_bytes(value) > _RECOVERY_DESCRIPTOR_BYTES
        or _json_bytes(value.get("provider_kwargs", {})) > _PROVIDER_KWARGS_MAX_BYTES
    ):
        _checkpoint_error(run_id)


def _validate_rewrite_entries(run_id: str, value: Any) -> None:
    if not isinstance(value, list) or len(value) > _RECOVERY_MAX_MESSAGES:
        _checkpoint_error(run_id)
    if any(
        not isinstance(m, dict)
        or not isinstance(m.get("role"), str)
        or not isinstance(m.get("content"), str)
        for m in value
    ):
        raise ValueError("Rewrite entries require role and content")
    if any(not isinstance(m.get("meta", {}), dict) for m in value):
        _codes.log_code(logger, _codes.REWRITE_INVALID, run_id=run_id)
        raise ValueError("Rewrite entry metadata must be an object")


def _validate_trace_write(run_id: str, trace: TraceWrite) -> None:
    from dendrux.types import _validate_persistable

    _validate_persistable(trace.content, "content")
    _validate_persistable(trace.meta, "meta")
    if trace.trace_kind not in ("seed", "turn", "rewrite", "state"):
        raise ValueError("Unsupported trace kind")
    if trace.trace_kind in ("rewrite", "state"):
        if len(trace.content.encode("utf-8")) > _RECOVERY_TRACE_BYTES:
            _checkpoint_error(run_id)
        value = json.loads(trace.content)
        if trace.trace_kind == "rewrite":
            _validate_rewrite_entries(run_id, value)
        elif not isinstance(value, dict) or value.get("kind") not in (
            "image_view",
            "skills_session",
            "clarification",
        ):
            raise ValueError("State traces require a versioned state payload")


def _positive_page(limit: int) -> None:
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")


def _trace_record(row: Any) -> TraceRecord:
    return TraceRecord(
        id=row.id,
        role=row.role,
        content=row.content,
        order_index=row.order_index,
        meta=row.meta,
        created_at=_to_aware_utc(row.created_at),
        trace_kind=row.trace_kind,
    )


def _validate_lease_batch_size(*, batch_size: int, cap: int) -> None:
    _positive_page(batch_size)
    if batch_size > cap:
        _codes.log_code(logger, _codes.RECOVERY_SCAN_LIMIT, run_id=None)
        raise ValueError(f"batch_size must not exceed {cap}")
