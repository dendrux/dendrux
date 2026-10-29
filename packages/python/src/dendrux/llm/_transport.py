"""Request-local decoded byte guards, before either SDK's parser."""

from __future__ import annotations

import contextlib
import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

import httpx

from dendrux.errors import ProviderResponseError

logger = logging.getLogger(__name__)
_NEWLINE = re.compile(rb"\r\n|\r|\n")
_CONFIG = "dendrux.response_bound"
_STREAMING = "dendrux.streaming_response"


async def _deferred_send(
    self: httpx.AsyncClient, request: httpx.Request, **kwargs: Any
) -> httpx.Response:
    # Both SDKs catch and retry *every* Exception from send(). Their normal
    # response parser reads outside that retry block. Keep reads there, even
    # for non-streaming calls, while preserving the SDK's default client setup.
    request.extensions[_STREAMING] = bool(kwargs.get("stream", False))
    kwargs["stream"] = True
    return await httpx.AsyncClient.send(self, request, **kwargs)


def deferred_client_type(base: type[httpx.AsyncClient]) -> type[httpx.AsyncClient]:
    """Extend the SDK default client with deferred body reads."""
    return type("BoundedAsyncHttpxClient", (base,), {"send": _deferred_send})


def configure_request(
    request: httpx.Request, *, response_bytes: int, event_bytes: int, provider: str
) -> None:
    """Attach immutable guard configuration to this attempt, never the client."""
    request.extensions[_CONFIG] = (response_bytes, event_bytes, provider)


class _RawChunks(httpx.AsyncByteStream):
    def __init__(self, upstream: httpx.AsyncByteStream) -> None:
        self.upstream = upstream

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self.upstream:
            # httpx's gzip decoder expands a whole raw chunk in one allocation.
            # Restrict its input too: a single compressed transport chunk must
            # not allocate the full expanded body before we can count it.
            for start in range(0, len(chunk), 1024):
                yield chunk[start : start + 1024]

    async def aclose(self) -> None:
        await self.upstream.aclose()


class _Budget:
    def __init__(self, limit: int, *, streaming: bool, provider: str) -> None:
        self.limit = limit
        self.streaming = streaming
        self.provider = provider
        self.used = 0
        self.line = 0
        self.cr = False

    def _charge(self, size: int) -> None:
        self.used += size
        if self.used > self.limit:
            code = "stream_size_limit" if self.streaming else "response_size_limit"
            logger.warning(code, extra={"code": code, "run_id": None})
            raise ProviderResponseError(provider=self.provider, provider_code=code, retryable=False)

    def _newline(self, width: int) -> None:
        self._charge(width)
        if not self.line:
            self.used = 0
        self.line = 0

    def account(self, chunk: bytes) -> None:
        if not self.streaming:
            self._charge(len(chunk))
            return
        start = 0
        if self.cr:
            self.cr = False
            start = int(chunk.startswith(b"\n"))
            self._newline(start)
        for match in _NEWLINE.finditer(chunk, start):
            size = match.start() - start
            self._charge(size)
            self.line += size
            start = match.end()
            if match.group() == b"\r" and start == len(chunk):
                self._charge(1)
                self.cr = True
            else:
                self._newline(len(match.group()))
        size = len(chunk) - start
        self._charge(size)
        self.line += size

    def finish(self) -> None:
        if self.cr:
            self.cr = False
            self._newline(0)


def guard_response(response: httpx.Response) -> None:
    """Install one decoded iterator and counter on the response of this attempt."""
    config = response.request.extensions.get(_CONFIG)
    if config is None:
        return
    response_bytes, event_bytes, provider = config
    # The SDK chooses its parser from the request mode, not Content-Type.
    # Status errors are read as ordinary bodies even on streaming requests.
    streaming = bool(response.request.extensions.get(_STREAMING)) and response.is_success
    budget = _Budget(
        event_bytes if streaming else response_bytes, streaming=streaming, provider=provider
    )
    if response.headers.get("content-encoding") and isinstance(
        response.stream, httpx.AsyncByteStream
    ):
        response.stream = _RawChunks(response.stream)
    original = response.aiter_bytes

    async def bounded_bytes(chunk_size: int | None = None) -> AsyncIterator[bytes]:
        try:
            # Small decoded chunks also bound the SDK's input between checks.
            async for decoded in original():
                size = min(chunk_size or 65536, 65536)
                for start in range(0, len(decoded), size):
                    chunk = decoded[start : start + size]
                    budget.account(chunk)
                    yield chunk
            budget.finish()
        except BaseException:
            with contextlib.suppress(Exception):
                await response.aclose()
            raise

    response.aiter_bytes = bounded_bytes  # type: ignore[method-assign]
