"""Provider invocation boundaries, independent of how methods were installed."""

from collections.abc import AsyncGenerator
from typing import Any

from dendrux.errors import ProviderResponseError, provider_error_in
from dendrux.llm._evidence import persistable_event, persistable_response
from dendrux.llm.base import LLMProvider
from dendrux.types import LLMResponse, StreamEvent, StreamEventType


def _validate_response(provider: LLMProvider, response: LLMResponse, run_id: Any) -> LLMResponse:
    if not response.text and not response.tool_calls and response.stop_reason != "max_tokens":
        from dendrux.llm._resilience import log_provider_error

        error = ProviderResponseError(
            provider=type(provider).__name__,
            model=response.model or provider.model,
            provider_code="empty_completion",
            retryable=False,
            message="Provider returned neither text nor tool calls.",
        )
        log_provider_error(error, run_id if isinstance(run_id, str) else None)
        raise error
    return response


async def complete_call(provider: LLMProvider, *args: Any, **kwargs: Any) -> LLMResponse:
    try:
        response = await provider.complete(*args, **kwargs)
        return persistable_response(_validate_response(provider, response, kwargs.get("run_id")))
    except ExceptionGroup as group:
        failure = provider_error_in(group)
        if failure is not None:
            raise failure from None
        raise


async def stream_call(
    provider: LLMProvider, *args: Any, **kwargs: Any
) -> AsyncGenerator[StreamEvent, None]:
    source = None
    try:
        try:
            source = provider.complete_stream(*args, **kwargs)
            async for event in source:
                if event.type == StreamEventType.DONE and isinstance(event.raw, LLMResponse):
                    _validate_response(provider, event.raw, kwargs.get("run_id"))
                yield persistable_event(event)
        finally:
            if source is not None:
                await source.aclose()
    except ExceptionGroup as group:
        failure = provider_error_in(group)
        if failure is not None:
            raise failure from None
        raise
