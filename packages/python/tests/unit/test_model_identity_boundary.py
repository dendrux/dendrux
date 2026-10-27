"""Bounded identity mapping preserves valid provider protocol identities."""

from dendrux.llm._evidence import persistable_call, persistable_event
from dendrux.types import StreamEvent, StreamEventType, ToolCall


def test_provider_ids_within_column_width_are_unchanged_in_stream_events():
    provider_id = "call_" + "x" * 250
    call = ToolCall("read", provider_tool_call_id=provider_id)
    normalized = persistable_call(call)
    event = persistable_event(
        StreamEvent(StreamEventType.TOOL_USE_END, tool_call=call, tool_call_id=provider_id)
    )
    assert normalized.provider_tool_call_id == provider_id
    assert event.tool_call_id == provider_id
    assert event.tool_call.provider_tool_call_id == provider_id


def test_oversized_provider_identity_mapping_is_stable_across_stream_and_batch():
    provider_id = "call_" + "x" * 1000
    call = ToolCall("read", provider_tool_call_id=provider_id)
    normalized = persistable_call(call)
    event = persistable_event(
        StreamEvent(StreamEventType.TOOL_USE_END, tool_call=call, tool_call_id=provider_id)
    )
    assert event.tool_call_id == normalized.provider_tool_call_id
    assert event.tool_call.provider_tool_call_id == normalized.provider_tool_call_id
    assert persistable_call(normalized) == normalized
