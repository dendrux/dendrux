"""Normalize rejected arguments before callbacks receive model evidence."""

from dataclasses import replace
from typing import overload

from dendrux.types import InputValidationError, LLMResponse, ToolCall, _validate_persistable


class RejectedArguments(ToolCall):
    """Persistable call identity whose arguments failed the text/JSON boundary."""


@overload
def persistable_call(call: ToolCall) -> ToolCall: ...


@overload
def persistable_call(call: None) -> None: ...


def persistable_call(call: ToolCall | None) -> ToolCall | None:
    if call is None:
        return None
    try:
        _validate_persistable(call.params, "params")
    except InputValidationError:
        return RejectedArguments(
            name=call.name,
            id=call.id,
            provider_tool_call_id=call.provider_tool_call_id,
            params={},
        )
    return call


def persistable_response(response: LLMResponse) -> LLMResponse:
    calls = [persistable_call(call) for call in response.tool_calls or []]
    if not any(isinstance(call, RejectedArguments) for call in calls):
        return response
    return replace(
        response, tool_calls=calls, raw=None, provider_response=None, reasoning_blocks=None
    )
