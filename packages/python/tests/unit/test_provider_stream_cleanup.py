"""Stream finalizers preserve typed provider failures."""

import pytest

from dendrux.errors import ProviderTimeoutError
from dendrux.llm._calls import stream_call
from dendrux.llm.mock import MockLLM


async def test_grouped_stream_close_failure_is_typed():
    class Source:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self):
            raise ExceptionGroup("PRIVATE_CLOSE_GROUP", [ProviderTimeoutError(provider="fixture")])

    class Provider(MockLLM):
        def complete_stream(self, *args, **kwargs):
            return Source()

    provider = Provider([])
    with pytest.raises(ProviderTimeoutError):
        async for _ in stream_call(provider, []):
            pass
