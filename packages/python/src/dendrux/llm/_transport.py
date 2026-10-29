"""Request-local decoded byte guards, before either SDK's parser."""

from __future__ import annotations

import contextlib
import functools
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

import httpx

from dendrux.errors import ProviderResponseError

_CONFIG = "dendrux.response_bound"
_STREAMING = "dendrux.streaming_response"
_ERROR = "dendrux.response_error"


async def _deferred_send(
    send: Callable[..., Awaitable[httpx.Response]], request: httpx.Request, **kwargs: Any
) -> httpx.Response:
    streaming = bool(kwargs.get("stream", False))
    request.extensions[_STREAMING] = streaming
    kwargs["stream"] = True
    response = await send(request, **kwargs)
    if not streaming:
        try:
            # Transport failures must escape send() so the SDK still retries.
            await response.aread()
        except ProviderResponseError as error:
            # Only policy refusals cross the SDK's catch-all send retry block.
            response.extensions[_ERROR] = error
    return response


@functools.cache
def deferred_client_type(base: type[httpx.AsyncClient]) -> type[httpx.AsyncClient]:
    """Extend the SDK default client with deferred body reads."""

    async def send(
        self: httpx.AsyncClient, request: httpx.Request, **kwargs: Any
    ) -> httpx.Response:
        # mypy cannot model the runtime-generated subclass's MRO.
        parent_send = super(client_type, self).send  # type: ignore[misc]
        return await _deferred_send(parent_send, request, **kwargs)

    client_type: type[httpx.AsyncClient] = type("BoundedAsyncHttpxClient", (base,), {"send": send})
    return client_type


def configure_request(
    request: httpx.Request, *, response_bytes: int, event_bytes: int, provider: str
) -> None:
    """Attach immutable guard configuration to this attempt, never the client."""
    request.headers["accept-encoding"] = "gzip, deflate"
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
        self.tail = b""

    def _charge(self, size: int) -> None:
        self.used += size
        if self.used > self.limit:
            code = "stream_size_limit" if self.streaming else "response_size_limit"
            raise ProviderResponseError(provider=self.provider, provider_code=code, retryable=False)

    def account(self, chunk: bytes) -> None:
        if not self.streaming:
            self._charge(len(chunk))
            return
        # Match both SDKs' _aiter_chunks on the exact chunks we yield. Mixed
        # spec-valid blank lines need not end the SDK's pending event buffer.
        for line in chunk.splitlines(keepends=True):
            self._charge(len(line))
            suffix = self.tail + line[-4:]
            if suffix.endswith((b"\r\r", b"\n\n", b"\r\n\r\n")):
                self.used = 0
                self.tail = b""
            else:
                self.tail = suffix[-3:]


async def guard_response(response: httpx.Response) -> None:
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
    original_read = response.aread
    original_status = response.raise_for_status
    checked = False

    def raise_deferred() -> None:
        error = response.extensions.get(_ERROR)
        if error is not None:
            raise error

    def checked_status() -> httpx.Response:
        # SDK status handling precedes parsing. A refused error body must not
        # turn into a status retry (or an attempt to inspect an unread body).
        if not response.is_success:
            raise_deferred()
        return original_status()

    async def bounded_read() -> bytes:
        raise_deferred()
        if not checked and hasattr(response, "_content"):
            # Mock/custom transports can return an already-buffered response.
            async for _ in bounded_bytes():
                pass
        return await original_read()

    async def bounded_bytes(chunk_size: int | None = None) -> AsyncIterator[bytes]:
        nonlocal checked
        try:
            raise_deferred()
            if chunk_size is not None and chunk_size < 1:
                raise ValueError("chunk_size must be positive")
            pending = bytearray()
            # Small decoded chunks also bound the SDK's input between checks.
            # Both SDKs use chunk_size=None. Explicit callers receive fixed
            # size chunks and choose that output buffer size themselves.
            async for decoded in original():
                for start in range(0, len(decoded), 65536):
                    piece = decoded[start : start + 65536]
                    if not checked and not streaming:
                        budget.account(piece)
                    if chunk_size is None:
                        if not checked and streaming:
                            budget.account(piece)
                        yield piece
                        continue
                    offset = 0
                    while offset < len(piece):
                        take = min(chunk_size - len(pending), len(piece) - offset)
                        pending.extend(piece[offset : offset + take])
                        offset += take
                        if len(pending) == chunk_size:
                            chunk = bytes(pending)
                            pending.clear()
                            if not checked and streaming:
                                budget.account(chunk)
                            yield chunk
            if pending:
                chunk = bytes(pending)
                if not checked and streaming:
                    budget.account(chunk)
                yield chunk
            checked = True
        except BaseException:
            with contextlib.suppress(Exception):
                await response.aclose()
            raise

    response.aiter_bytes = bounded_bytes  # type: ignore[method-assign]
    response.aread = bounded_read  # type: ignore[method-assign]
    response.raise_for_status = checked_status  # type: ignore[method-assign]
    encodings = response.headers.get_list("content-encoding", split_commas=True)
    if encodings and (
        len(encodings) != 1 or encodings[0].strip().lower() not in {"identity", "gzip", "deflate"}
    ):
        response.extensions[_ERROR] = ProviderResponseError(
            provider=provider, provider_code="response_encoding_unsupported", retryable=False
        )
        with contextlib.suppress(Exception):
            await response.aclose()
