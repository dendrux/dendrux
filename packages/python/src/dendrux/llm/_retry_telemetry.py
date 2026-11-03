"""Provider retry telemetry — observe vendor-SDK HTTP retries.

Both ``anthropic`` and ``openai`` Python SDKs use ``httpx`` underneath
and accept a custom ``http_client=`` constructor argument. We pass a
client with a response event hook attached. The hook fires for every
HTTP response — including the ones the SDK is about to retry on —
without changing any SDK behavior.

The hook is several call frames removed from where dendrux holds the
loop's iteration state. We carry that state through the async call tree
with ``contextvars``, set by the loop just before each
``provider.complete()`` call and reset after. Events leave through two
sinks the loop hands in (``TelemetrySinks``): this module never imports
the loops, so the provider layer stays below them.

Failure contract: the hook is observability only. Any exception inside
is logged and swallowed — an instrumentation bug must never break a
real LLM call.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import AsyncGenerator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any, NamedTuple

import httpx

from dendrux._lifecycle import GovernanceEventType

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Iterator

    from dendrux.types import StreamEvent

    # ``(event_type, iteration, data)``; the run is bound by whoever built it.
    GovernanceSink = Callable[[str, int, dict[str, Any]], Awaitable[None]]

_logger = logging.getLogger(__name__)


class TelemetrySinks(NamedTuple):
    """Where the hook reports ``provider.retry``: persistence, then notification.

    Each sink is an async ``(event_type, iteration, data)`` callable bound to
    one run, or None. The loop builds the pair from its recorder and notifier
    (``dendrux.loops._helpers.governance_sinks``).
    """

    record: GovernanceSink | None
    notify: GovernanceSink | None


# State carried into the httpx hook. Set by the loop around every
# ``provider.complete()`` / ``provider.complete_stream()`` call.
# The run the binding belongs to (the sinks carry their own run id).
_run_id: ContextVar[str | None] = ContextVar("dendrux_retry_run_id", default=None)
_iteration: ContextVar[int] = ContextVar("dendrux_retry_iteration", default=0)
_sinks: ContextVar[TelemetrySinks | None] = ContextVar("dendrux_retry_sinks", default=None)
_NO_SINKS = TelemetrySinks(None, None)

# Per-call attempt counter — set by the provider at the start of each
# ``complete()`` call so the hook can label retries 1..N within one
# logical call. List wrapper gives mutability through the contextvar.
_attempts: ContextVar[list[int] | None] = ContextVar("dendrux_retry_attempts", default=None)


@contextmanager
def telemetry_context(
    *,
    run_id: str | None,
    iteration: int,
    sinks: TelemetrySinks | None,
) -> Iterator[None]:
    """Bind retry-telemetry context for the duration of a ``with`` block.

    Wrap each ``provider.complete()`` / ``provider.complete_stream()``
    call so the httpx hook can attribute retries to the right run +
    iteration and route events through the run's sinks (``None``: no
    events).

    Example::

        with telemetry_context(
            run_id=run_id,
            iteration=iteration,
            sinks=governance_sinks(recorder, notifier, run_id),
        ):
            response = await provider.complete(...)
    """
    tokens = (
        _run_id.set(run_id),
        _iteration.set(iteration),
        _sinks.set(sinks),
    )
    try:
        yield
    finally:
        _run_id.reset(tokens[0])
        _iteration.reset(tokens[1])
        _sinks.reset(tokens[2])


@contextmanager
def call_attempt_tracking() -> Iterator[None]:
    """Reset the per-call attempt counter for one provider call.

    Set by the provider around its single ``self._client.<call>(...)``
    so the hook can label attempts 1..N within that one logical call.
    Counter resets so a subsequent call on the same task starts fresh.
    """
    token = begin_call_attempt_tracking()
    try:
        yield
    finally:
        end_call_attempt_tracking(token)


def begin_call_attempt_tracking() -> Token[list[int] | None]:
    """Token-based variant of :func:`call_attempt_tracking`.

    Use when the provider's call site mixes ``async with`` and other
    sync context managers in a way that does not nest cleanly. Always
    pair with :func:`end_call_attempt_tracking` in a try/finally.
    """
    return _attempts.set([0])


def end_call_attempt_tracking(token: Token[list[int] | None]) -> None:
    """Release the per-call attempt counter set by ``begin_call_attempt_tracking``."""
    _attempts.reset(token)


async def _on_response(response: httpx.Response, *, retryable_statuses: frozenset[int]) -> None:
    """httpx response hook — emit ``provider.retry`` on retryable failures.

    Fail-open: any exception inside is logged and swallowed so an
    observability bug never breaks an LLM call.
    """
    try:
        if response.status_code not in retryable_statuses:
            return

        sinks = _sinks.get()
        record, notify = sinks if sinks is not None else _NO_SINKS
        if record is None and notify is None:
            # Hook fired outside dendrux context (e.g. user shares the
            # provider with another framework) — nothing to write.
            return

        attempts = _attempts.get()
        attempt = (attempts[0] + 1) if attempts else 1
        if attempts is not None:
            attempts[0] = attempt

        iteration = _iteration.get()
        endpoint: str | None = None
        try:
            if response.request is not None:
                endpoint = response.request.url.path
        except Exception:
            endpoint = None

        data: dict[str, Any] = {
            "status_code": response.status_code,
            "endpoint": endpoint,
            "attempt": attempt,
            "retry_after": response.headers.get("retry-after"),
        }

        await _emit(record, notify, iteration, data)
    except Exception:
        _logger.warning("retry telemetry hook raised", exc_info=True)


async def _emit(
    record: GovernanceSink | None,
    notify: GovernanceSink | None,
    iteration: int,
    data: dict[str, Any],
) -> None:
    """Send one ``provider.retry`` to each sink; a failing sink never stops the other."""
    event_type = GovernanceEventType.PROVIDER_RETRY.value
    if record is not None:
        try:
            await record(event_type, iteration, data)
        except Exception:
            _logger.warning("provider.retry recorder write failed", exc_info=True)
    if notify is not None:
        try:
            await notify(event_type, iteration, data)
        except Exception:
            _logger.warning("provider.retry notifier broadcast failed", exc_info=True)


def telemetry_event_hooks(
    retryable_statuses: Iterable[int],
) -> dict[str, list[Callable[..., Awaitable[None]]]]:
    """Fresh httpx hooks for clients that also configure limits or deadlines.

    ``retryable_statuses`` are the HTTP statuses the SDK retries; only those
    emit ``provider.retry`` (the provider layer passes its one shared set).
    """
    hook = functools.partial(_on_response, retryable_statuses=frozenset(retryable_statuses))
    return {"response": [hook]}


class TelemetryStream(AsyncGenerator["StreamEvent", None]):
    """Own the provider immediately, including close before the first advance."""

    def __init__(
        self,
        stream: AsyncGenerator[StreamEvent, None],
        *,
        run_id: str,
        iteration: int,
        sinks: TelemetrySinks | None,
    ):
        self._stream = stream
        self._run_id, self._iteration = run_id, iteration
        self._sinks = sinks
        self._closed = False
        with self._binding():
            self._iterator = aiter(stream)

    def _binding(self) -> Any:
        return telemetry_context(run_id=self._run_id, iteration=self._iteration, sinks=self._sinks)

    async def __anext__(self) -> StreamEvent:
        if self._closed:
            raise StopAsyncIteration
        try:
            with self._binding():
                return await anext(self._iterator)
        except StopAsyncIteration:
            await self.aclose()
            raise

    async def asend(self, value: None) -> StreamEvent:
        return await self.__anext__()

    async def athrow(self, *args: Any) -> StreamEvent:
        with self._binding():
            return await self._stream.athrow(*args)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._binding():
            await self._stream.aclose()


def telemetry_stream(
    stream: AsyncGenerator[StreamEvent, None],
    *,
    run_id: str,
    iteration: int,
    sinks: TelemetrySinks | None,
) -> TelemetryStream:
    return TelemetryStream(stream, run_id=run_id, iteration=iteration, sinks=sinks)


def make_telemetry_http_client(
    timeout: httpx.Timeout,
    retryable_statuses: Iterable[int],
) -> httpx.AsyncClient:
    """Build an ``httpx.AsyncClient`` with the dendrux retry hook attached.

    Pass the returned client to the vendor SDK constructor via
    ``http_client=`` so the hook fires on every request the SDK makes,
    including its internal retries.
    """
    return httpx.AsyncClient(
        timeout=timeout,
        event_hooks=telemetry_event_hooks(retryable_statuses),
    )


__all__ = [
    "TelemetrySinks",
    "begin_call_attempt_tracking",
    "call_attempt_tracking",
    "end_call_attempt_tracking",
    "make_telemetry_http_client",
    "telemetry_context",
    "telemetry_event_hooks",
    "telemetry_stream",
]
