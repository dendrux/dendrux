"""Request-local response policies, before SDK parsing or rendering.

No over-limit decoded body/event reaches an SDK consumer. Decoder output and
lookahead are bounded; refusal stops further upstream pulls. This cannot undo
allocations a custom transport made before returning a chunk or cached body.
Unread streaming error bodies may be closed and retried by the SDK. A known
refusal is terminal across status, cached-body and raw/decoded read surfaces.
"""

from __future__ import annotations

import contextlib
import functools
import zlib
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterator

import httpx

from dendrux._meta_keys import (
    RESPONSE_BOUND_KEY as _CONFIG,
)
from dendrux._meta_keys import (
    STREAMING_RESPONSE_KEY as _STREAMING,
)
from dendrux.errors import ProviderResponseError

_WINDOW = 65536
_ClientT = TypeVar("_ClientT")


class _RequestScope:
    def __init__(self) -> None:
        self.active = True


_sdk_request: ContextVar[_RequestScope | None] = ContextVar("dendrux_sdk_request", default=None)


class _PolicyStop(BaseException):
    def __init__(self, provider: str, code: str) -> None:
        self.provider = provider
        self.code = code


@functools.cache
def _sdk_client_type(base: type[Any]) -> type[Any]:

    async def request(self: Any, *args: Any, **kwargs: Any) -> Any:
        scope = _RequestScope()
        token = _sdk_request.set(scope)
        try:
            return await super(client_type, self).request(*args, **kwargs)
        except _PolicyStop as stop:
            provider, code = stop.provider, stop.code
        finally:
            # Copied contexts must not retain an active conversion boundary.
            scope.active = False
            _sdk_request.reset(token)
        # Outside the handler: neither the carrier nor SDK error/body is chained.
        raise ProviderResponseError(provider=provider, provider_code=code, retryable=False)

    client_type: type[Any] = type("BoundedSDKClient", (base,), {"request": request})
    return client_type


def guard_sdk_client(client: _ClientT) -> _ClientT:
    """Add a request boundary without replacing SDK constructor or copy behavior."""
    client.__class__ = _sdk_client_type(cast("Any", type(client)))
    return client


async def _deferred_send(
    send: Callable[..., Awaitable[httpx.Response]], request: httpx.Request, **kwargs: Any
) -> httpx.Response:
    streaming = bool(kwargs.get("stream", False))
    request.extensions[_STREAMING] = streaming
    kwargs["stream"] = True
    response = await send(request, **kwargs)
    if not streaming:
        # Transport failures escape so the SDK still retries. Policy failures
        # persist on the response without retaining an exception traceback.
        with contextlib.suppress(ProviderResponseError):
            await response.aread()
    state = response.__dict__.get("_bound")
    if state is not None:
        # HTTPX's own post-hook status logging must finish inside send first.
        state.armed = True
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


class _Budget:
    def __init__(self, limit: int, *, streaming: bool, provider: str) -> None:
        self.limit = limit
        self.streaming = streaming
        self.provider = provider
        self.used = 0
        self.tail = b""
        self.wire_used = 0
        # Allow ordinary compression overhead, including incompressible bodies.
        self.wire_limit = limit + max(1024, limit // 100)

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
                self.wire_used = 0
                self.tail = b""
            else:
                self.tail = suffix[-3:]


class _ResponseState:
    def __init__(self, response: httpx.Response, budget: _Budget) -> None:
        self.response = response
        self.budget = budget
        self.code: str | None = None
        self.armed = False

    def refuse(self, error: ProviderResponseError) -> None:
        self.code = self.code or error.provider_code or "response_size_limit"
        self.response.__dict__.pop("_content", None)
        self.response.__dict__.pop("_text", None)

    def check(self) -> None:
        if self.code is not None:
            scope = _sdk_request.get()
            if self.armed and scope is not None and scope.active:
                # Policy control flow must bypass SDK catch-all Exception retries.
                raise _PolicyStop(self.budget.provider, self.code)
            raise ProviderResponseError(
                provider=self.budget.provider, provider_code=self.code, retryable=False
            ) from None

    def account(self, chunk: bytes | int) -> None:
        self.check()
        try:
            if isinstance(chunk, int):
                self.budget._charge(chunk)
            else:
                self.budget.account(chunk)
            if self.budget.wire_used > self.budget.wire_limit:
                self.budget._charge(self.budget.limit + 1)
        except ProviderResponseError as error:
            self.refuse(error)
        self.check()


class _Decoder:
    def __init__(self, encoding: str, window: int) -> None:
        self.encoding = encoding
        self.window = window
        self.prefix = b""
        self.first_attempt = True
        self.finished = False
        self.decoder: Any = None
        if encoding == "gzip":
            self.decoder = zlib.decompressobj(zlib.MAX_WBITS | 16)

    def decode(self, data: bytes) -> Iterator[tuple[bytes, bool]]:
        if self.finished:
            yield data, True
            return
        if self.encoding in ("", "identity"):
            for start in range(0, len(data), self.window):
                yield data[start : start + self.window], False
            return
        if self.decoder is None:
            data = self.prefix + data
            if len(data) < 2:
                self.prefix = data
                return
            self.prefix = b""
            # Deflate permits both zlib-wrapped and raw RFC 1951 streams.
            wrapped = data[0] & 15 == 8 and data[0] >> 4 <= 7 and int.from_bytes(data[:2]) % 31 == 0
            self.decoder = zlib.decompressobj(zlib.MAX_WBITS if wrapped else -zlib.MAX_WBITS)
        yield from self._inflate(data)

    def _inflate(self, data: bytes) -> Iterator[tuple[bytes, bool]]:
        while True:
            decoded = self._decompress(data)
            if decoded:
                yield decoded, False
            if self.decoder.eof:
                trailing = self.decoder.unused_data
                self.finished = True
                self.decoder = None
                if trailing:
                    yield trailing, True
                return
            data = self.decoder.unconsumed_tail
            if not data and len(decoded) < self.window:
                return
            # Empty-input draining is bounded too; flush(length) is not a cap.

    def _decompress(self, data: bytes) -> bytes:
        first = self.first_attempt
        self.first_attempt = False
        try:
            return cast("bytes", self.decoder.decompress(data, self.window))
        except zlib.error as error:
            if first and self.encoding == "deflate":
                self.decoder = zlib.decompressobj(-zlib.MAX_WBITS)
                return self._decompress(data)
            raise httpx.DecodingError("Invalid compressed response") from error


class _BoundedStream(httpx.AsyncByteStream):
    def __init__(
        self, upstream: httpx.AsyncByteStream, state: _ResponseState, encoding: str
    ) -> None:
        self.upstream = upstream
        self.state = state
        self.decoder = _Decoder(encoding, min(_WINDOW, state.budget.limit + 1))
        self.closed = state.response.is_closed

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async with contextlib.aclosing(self.iterate(raw=True)) as chunks:
            async for chunk in chunks:
                yield chunk

    def _pieces(self, chunk: bytes, *, raw: bool) -> Iterator[bytes]:
        for start in range(0, len(chunk), self.decoder.window):
            encoded = chunk[start : start + self.decoder.window]
            if self.decoder.encoding not in ("", "identity"):
                self.state.budget.wire_used += len(encoded)
            for decoded, trailing in self.decoder.decode(encoded):
                # Trailing wire bytes are charged, never parsed as SSE separators.
                self.state.account(len(decoded) if trailing else decoded)
                if not raw and not trailing:
                    yield decoded
            # Empty stored blocks produce no decoded bytes to trigger a check.
            self.state.account(0)
            if raw:
                yield encoded

    async def iterate(self, *, raw: bool) -> AsyncGenerator[bytes, None]:
        try:
            self.state.check()
            async for chunk in self.upstream:
                self.state.check()
                if not raw:
                    self.state.response._num_bytes_downloaded += len(chunk)
                for piece in self._pieces(chunk, raw=raw):
                    yield piece
        except BaseException:
            with contextlib.suppress(Exception):
                await self.aclose()
            raise
        else:
            await self.aclose()

    async def aclose(self) -> None:
        if not self.closed:
            self.closed = True
            self.state.response.is_closed = True
            await self.upstream.aclose()


_BODY_SURFACES = frozenset(
    {
        "content",
        "text",
        "json",
        "read",
        "aread",
        "iter_bytes",
        "iter_text",
        "iter_lines",
        "iter_raw",
        "aiter_bytes",
        "aiter_text",
        "aiter_lines",
        "aiter_raw",
        "raise_for_status",
    }
)


async def _rechunk(source: AsyncIterator[bytes], size: int | None) -> AsyncGenerator[bytes, None]:
    pending = bytearray()
    async for chunk in source:
        if size is None:
            yield chunk
            continue
        offset = 0
        while offset < len(chunk):
            take = min(size - len(pending), len(chunk) - offset)
            pending.extend(chunk[offset : offset + take])
            offset += take
            if len(pending) == size:
                yield bytes(pending)
                pending.clear()
    if pending:
        yield bytes(pending)


class _BoundedResponse(httpx.Response):
    __slots__ = ()
    _bound: _ResponseState

    def __getattribute__(self, name: str) -> Any:
        state = object.__getattribute__(self, "__dict__").get("_bound")
        if state is not None and (
            name in _BODY_SURFACES or (name == "status_code" and state.armed)
        ):
            state.check()
        return super().__getattribute__(name)

    def iter_raw(self, chunk_size: int | None = None) -> Iterator[bytes]:
        self._bound.check()
        yield from super().iter_raw(chunk_size)

    async def aiter_raw(self, chunk_size: int | None = None) -> AsyncIterator[bytes]:
        self._bound.check()
        try:
            async with contextlib.aclosing(
                cast("AsyncGenerator[bytes, None]", super().aiter_raw(chunk_size))
            ) as chunks:
                async for chunk in chunks:
                    yield chunk
        finally:
            with contextlib.suppress(Exception):
                await self.aclose()

    def _start_decoded(self) -> _BoundedStream:
        self._bound.check()
        if self.is_stream_consumed:
            raise httpx.StreamConsumed()
        if self.is_closed:
            raise httpx.StreamClosed()
        self.is_stream_consumed = True
        self._num_bytes_downloaded = 0
        assert isinstance(self.stream, _BoundedStream)
        return self.stream

    async def aiter_bytes(self, chunk_size: int | None = None) -> AsyncIterator[bytes]:
        self._bound.check()
        if chunk_size is not None and chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if "_content" in self.__dict__:
            async for chunk in super().aiter_bytes(chunk_size or _WINDOW):
                yield chunk
            return
        stream = self._start_decoded()
        async with (
            contextlib.aclosing(stream.iterate(raw=False)) as source,
            contextlib.aclosing(_rechunk(source, chunk_size)) as chunks,
        ):
            async for chunk in chunks:
                yield chunk


@functools.cache
def _response_type(base: Any) -> type[httpx.Response]:
    return type("BoundedResponse", (_BoundedResponse, base), {"__slots__": ()})


def _response_encoding(response: httpx.Response, state: _ResponseState) -> str:
    encodings = response.headers.get_list("content-encoding", split_commas=True)
    encoding = encodings[0].strip().lower() if encodings else ""
    if len(encodings) > 1 or encoding not in {"", "identity", "gzip", "deflate"}:
        state.refuse(
            ProviderResponseError(
                provider=state.budget.provider,
                provider_code="response_encoding_unsupported",
                retryable=False,
            )
        )
    return encoding


async def guard_response(response: httpx.Response) -> None:
    """Install a response-local terminal policy and bounded stream decoder."""
    config = response.request.extensions.get(_CONFIG)
    if config is None or "_bound" in response.__dict__:
        return
    response_bytes, event_bytes, provider = config
    streaming = bool(response.request.extensions.get(_STREAMING)) and response.is_success
    budget = _Budget(
        event_bytes if streaming else response_bytes, streaming=streaming, provider=provider
    )
    state = _ResponseState(response, budget)
    encoding = _response_encoding(response, state)
    response.__dict__["_bound"] = state
    response.__class__ = _response_type(cast("Any", type(response)))
    if isinstance(response.stream, httpx.AsyncByteStream):
        response.stream = _BoundedStream(response.stream, state, encoding)
    cached = response.__dict__.get("_content")
    if cached is not None:
        with contextlib.suppress(ProviderResponseError):
            for start in range(0, len(cached), _WINDOW):
                state.account(cached[start : start + _WINDOW])
    if state.code is not None:
        with contextlib.suppress(Exception):
            await response.aclose()
