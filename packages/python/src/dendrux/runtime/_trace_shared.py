"""Shared conversation-trace serialization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dendrux._meta_keys import MESSAGE_META_KEY, PARTS_META_KEY
from dendrux.runtime._blob_catalog import edges_for_parts
from dendrux.runtime.state import TraceWrite
from dendrux.types import content_part_to_dict, generate_ulid, tool_call_parse_meta

# isort: split
from dendrux.types import _message_to_dict

if TYPE_CHECKING:
    from dendrux.types import Message, ToolCall


def _tool_call_evidence(tc: ToolCall) -> dict[str, Any]:
    """Persisted shape of a tool call; never includes the raw argument text."""
    return {
        "id": tc.id,
        "provider_tool_call_id": tc.provider_tool_call_id,
        "name": tc.name,
        "params": tc.params,
        **tool_call_parse_meta(tc),
    }


def message_trace(
    message: Message,
    order: int,
    *,
    kind: str = "turn",
    tenant_scope: str | None = None,
) -> Any:
    """A lossless conversation envelope, with content stored only once."""
    return _message_trace(message, order, generate_ulid(), kind=kind, tenant_scope=tenant_scope)


def _message_trace(
    message: Message,
    order: int,
    trace_id: str,
    *,
    kind: str = "turn",
    tenant_scope: str | None = None,
) -> Any:
    envelope = _message_to_dict(message)
    envelope.pop("content", None)
    meta = dict(message.meta)
    meta[MESSAGE_META_KEY] = envelope
    if message.tool_calls:
        meta["tool_calls"] = [_tool_call_evidence(tc) for tc in message.tool_calls]
    if message.reasoning_blocks:
        meta["reasoning_blocks"] = message.reasoning_blocks
    if message.call_id:
        meta["call_id"] = message.call_id
    if message.name:
        meta["tool_name"] = message.name
    if message.parts is not None:
        meta[PARTS_META_KEY] = [content_part_to_dict(p) for p in message.parts]
    edges = (
        edges_for_parts(tenant_scope, "trace", str(order), meta.get(PARTS_META_KEY) or [])
        if tenant_scope
        else []
    )
    return TraceWrite(
        trace_id,
        message.role.value,
        message.content,
        order,
        trace_kind=kind,
        meta=meta,
        blob_refs=edges or None,
        blob_refs_scope=tenant_scope if edges else None,
    )
