"""Small builders shared by the contract modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dendrux.runtime.state import EventWrite, LoopStatePatch, TraceWrite
from dendrux.types import UsageStats, generate_ulid

if TYPE_CHECKING:
    from datetime import datetime

    from dendrux.testing.store_kit._kit import Kit

#: Events appended by hand never use these names, which the runtime emits itself.
EVENT = "kit.event"


def ulid() -> str:
    return generate_ulid()


def tool_call_kwargs(**overrides: Any) -> dict[str, Any]:
    """Keyword arguments for ``save_tool_call`` (the run id is passed positionally)."""
    values: dict[str, Any] = {
        "tool_call_id": ulid(),
        "provider_tool_call_id": None,
        "tool_name": "kit_tool",
        "target": "server",
        "params": {"q": "x"},
        "result_payload": '"ok"',
        "success": True,
        "duration_ms": 1,
        "iteration_index": 0,
    }
    values.update(overrides)
    return values


def usage(input_tokens: int = 1, output_tokens: int = 2) -> UsageStats:
    return UsageStats(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def trace_write(
    role: str, content: str, order: int, kind: str = "turn", meta: dict[str, Any] | None = None
) -> TraceWrite:
    return TraceWrite(
        trace_id=ulid(),
        role=role,
        content=content,
        order_index=order,
        trace_kind=kind,
        meta=meta,
    )


def event_write(kind: str, call_id: str, iteration: int = 0) -> EventWrite:
    return EventWrite(ulid(), kind, iteration, {"call_id": call_id}, call_id, None)


def start_call_kwargs(call_id: str, **overrides: Any) -> dict[str, Any]:
    """Keyword arguments for ``start_tool_call`` (run id positional, epoch added by the caller)."""
    values: dict[str, Any] = {
        "row_id": ulid(),
        "tool_call_id": call_id,
        "provider_tool_call_id": None,
        "tool_name": "kit_tool",
        "target": "server",
        "params": {},
        "iteration_index": 0,
        "idempotent": True,
        "event": event_write("tool.started", call_id),
    }
    values.update(overrides)
    return values


def complete_call_kwargs(call_id: str, order: int, **overrides: Any) -> dict[str, Any]:
    """Keyword arguments for ``complete_tool_call`` (run id positional, epoch by the caller)."""
    values: dict[str, Any] = {
        "tool_call_id": call_id,
        "outcome": "completed",
        "result_payload": '"ok"',
        "success": True,
        "duration_ms": 1,
        "error_message": None,
        "replayed": False,
        "observation": trace_write("tool", '"ok"', order, meta={"call_id": call_id}),
        "event": event_write("tool.completed", call_id),
        "loop_state_patch": LoopStatePatch(remove_pending=(call_id,)),
    }
    values.update(overrides)
    return values


RECLAIM: dict[str, Any] = {
    "governance": {},
    "governance_policy": "warn",
    "skill_pins": {},
    "loop_type": "react",
}


async def started_run(kit: Kit, tenant: str | None = "t") -> str:
    """A running format-2 run that has started and checkpointed, so it can be reclaimed."""
    run_id = await kit.new_run(tenant_id=tenant)
    await kit.store.save_run_event(run_id, event_type="run.started")
    await kit.store.checkpoint_step(
        run_id,
        claim_epoch=0,
        loop_state={"v": 1, "loop": "react", "phase": "dispatch"},
        pii_mapping=None,
        trace=None,
    )
    return run_id


async def lease_cutoff(kit: Kit) -> datetime:
    """Make what exists old; return the lease cutoff that spares anything created afterwards."""
    threshold = await kit.age()
    return await kit.now() - threshold


async def mark(kit: Kit, cutoff: datetime, **overrides: Any) -> Any:
    options: dict[str, Any] = {
        "cutoff": cutoff,
        "batch_size": 10,
        "max_runs": 100,
        "max_returned": 10,
        "reclaim_limit": 3,
        "cursor": None,
    }
    options.update(overrides)
    return await kit.store.mark_lease_expired(**options)
