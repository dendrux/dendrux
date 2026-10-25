"""Provider invocation boundaries, independent of how methods were installed."""

from collections.abc import AsyncGenerator
from typing import Any

from dendrux.errors import provider_error_in
from dendrux.llm._evidence import persistable_event, persistable_response
from dendrux.llm.base import LLMProvider
from dendrux.types import LLMResponse, StreamEvent


async def complete_call(provider: LLMProvider, *args: Any, **kwargs: Any) -> LLMResponse:
    try:
        return persistable_response(await provider.complete(*args, **kwargs))
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
                yield persistable_event(event)
        finally:
            if source is not None:
                await source.aclose()
    except ExceptionGroup as group:
        failure = provider_error_in(group)
        if failure is not None:
            raise failure from None
        raise
