"""Small builders shared by the contract modules."""

from __future__ import annotations

from typing import Any

from dendrux.runtime.state import EventWrite, LoopStatePatch, TraceWrite
from dendrux.types import UsageStats, generate_ulid

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
