"""Normalize model text and reject unsafe arguments before any callback."""

import hashlib
import math
import re
from dataclasses import replace
from typing import Any, overload

from dendrux.llm._usage import normalize_usage
from dendrux.types import (
    InputValidationError,
    LLMResponse,
    StreamEvent,
    ToolCall,
    _validate_persistable,
)

_BAD_TEXT = re.compile(r"[\x00\ud800-\udfff]")


@overload
def _text(value: str) -> str: ...


@overload
def _text(value: None) -> None: ...


def _text(value: str | None) -> str | None:
    return _BAD_TEXT.sub("\ufffd", value) if value is not None else None


def _evidence(value: Any) -> Any:
    """Bound opaque JSON traversal; never invoke arbitrary object serializers."""
    nodes = size = 0

    def copy(item: Any, depth: int = 0) -> Any:
        nonlocal nodes, size
        nodes += 1
        if depth > 32 or nodes > 10000:
            raise ValueError("Model evidence exceeds its traversal bound")
        if isinstance(item, str):
            if len(item) > 1048576:
                raise ValueError("Model evidence exceeds its byte bound")
            # The length check bounds the UTF-8 allocation to at most 4 MiB.
            # Surrogate replacement happens before encoding, in bounded text.
            cleaned = _text(item)
            size += len(cleaned.encode("utf-8"))
            if size > 1048576:
                raise ValueError("Model evidence exceeds its byte bound")
            return cleaned
        if item is None or isinstance(item, (bool, int)):
            return item
        if isinstance(item, float):
            return item if math.isfinite(item) else None
        if isinstance(item, dict):
            if len(item) > 5000:
                raise ValueError("Model evidence exceeds its mapping bound")
            return {
                copy(k, depth + 1): copy(v, depth + 1)
                for k, v in item.items()
                if isinstance(k, str)
            }
        if isinstance(item, (list, tuple)):
            return [copy(v, depth + 1) for v in item]
        return None

    try:
        return copy(value)
    except (ValueError, RecursionError):
        return None


@overload
def _bounded_id(value: str, maximum: int) -> str: ...


@overload
def _bounded_id(value: None, maximum: int) -> None: ...


def _bounded_id(value: str | None, maximum: int) -> str | None:
    cleaned = _text(value)
    if cleaned is not None and len(cleaned) > maximum:
        return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:maximum]
    return cleaned


@overload
def _tool_name(value: str) -> str: ...


@overload
def _tool_name(value: None) -> None: ...


def _tool_name(value: str | None) -> str | None:
    cleaned = _text(value)
    return "<unknown_tool>" if cleaned is not None and len(cleaned) > 255 else cleaned


class RejectedToolName(ToolCall):
    """A call whose model-supplied name exceeds the storage boundary."""


class RejectedArguments(ToolCall):
    """Persistable call identity whose arguments failed the text/JSON boundary."""


@overload
def persistable_call(call: ToolCall) -> ToolCall: ...


@overload
def persistable_call(call: None) -> None: ...


def persistable_call(call: ToolCall | None) -> ToolCall | None:
    if call is None:
        return None
    bounded = _call_identity(call)
    try:
        _validate_persistable(bounded.params, "params")
    except InputValidationError:
        return RejectedArguments(
            name=bounded.name,
            id=bounded.id,
            provider_tool_call_id=bounded.provider_tool_call_id,
            params={},
        )
    return bounded


def _call_identity(call: ToolCall) -> ToolCall:
    name = _tool_name(call.name)
    call_id = _bounded_id(call.id, 26)
    provider_id = _bounded_id(call.provider_tool_call_id, 255)
    if isinstance(call, RejectedToolName) or len(call.name) > 255:
        return RejectedToolName(name=name, id=call_id, provider_tool_call_id=provider_id, params={})
    return replace(call, name=name, id=call_id, provider_tool_call_id=provider_id)


def _response_artifacts(response: LLMResponse, calls: list[ToolCall]) -> dict[str, Any]:
    if any(isinstance(call, RejectedArguments) for call in calls):
        return dict(raw=None, provider_response=None, reasoning_blocks=None)
    return dict(
        raw=response.raw,
        provider_response=_evidence(response.provider_response),
        reasoning_blocks=_evidence(response.reasoning_blocks),
    )


def persistable_response(response: LLMResponse) -> LLMResponse:
    calls = [persistable_call(call) for call in response.tool_calls or []]
    return replace(
        response,
        text=_text(response.text),
        usage=normalize_usage(response.usage),
        model=_bounded_id(response.model, 255),
        reasoning=_text(response.reasoning),
        tool_calls=calls if response.tool_calls is not None else None,
        raw_stop_reason=_text(response.raw_stop_reason),
        **_response_artifacts(response, calls),
    )


def persistable_event(event: StreamEvent) -> StreamEvent:
    return replace(
        event,
        text=_text(event.text),
        tool_name=_tool_name(event.tool_name),
        tool_call_id=_bounded_id(event.tool_call_id, 255),
        tool_call=persistable_call(event.tool_call),
        raw=persistable_response(event.raw) if isinstance(event.raw, LLMResponse) else None,
    )
