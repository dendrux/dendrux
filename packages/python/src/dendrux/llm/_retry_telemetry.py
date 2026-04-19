"""Provider retry telemetry — observe vendor-SDK HTTP retries.

Both ``anthropic`` and ``openai`` Python SDKs use ``httpx`` underneath
and accept a custom ``http_client=`` constructor argument. We pass a
client with a response event hook attached. The hook fires for every
HTTP response — including the ones the SDK is about to retry on —
without changing any SDK behavior.

The hook is several call frames removed from where dendrux holds the
loop's recorder/notifier/iteration state. We carry that state through
the async call tree with ``contextvars``, set by the loop just before
each ``provider.complete()`` call and reset after.

Failure contract: the hook is observability only. Any exception inside
is logged and swallowed — an instrumentation bug must never break a
real LLM call.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dendrux.loops.base import LoopNotifier, LoopRecorder

_logger = logging.getLogger(__name__)


# State carried into the httpx hook. Set by the loop around every
# ``provider.complete()`` / ``provider.complete_stream()`` call.
_run_id: ContextVar[str | None] = ContextVar("dendrux_retry_run_id", default=None)
_iteration: ContextVar[int] = ContextVar("dendrux_retry_iteration", default=0)
_recorder: ContextVar[LoopRecorder | None] = ContextVar("dendrux_retry_recorder", default=None)
_notifier: ContextVar[LoopNotifier | None] = ContextVar("dendrux_retry_notifier", default=None)

# Per-call attempt counter — set by the provider at the start of each
# ``complete()`` call so the hook can label retries 1..N within one
# logical call. List wrapper gives mutability through the contextvar.
_attempts: ContextVar[list[int] | None] = ContextVar("dendrux_retry_attempts", default=None)


# Status codes the Anthropic and OpenAI SDKs treat as retryable. The hook
# only emits for these — other 4xx (400, 401, 403, 404, 422) are caller
# errors not retried by the SDK and so not "retry" telemetry.
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


@contextmanager
def telemetry_context(
    *,
    run_id: str | None,
    iteration: int,
    recorder: LoopRecorder | None,
    notifier: LoopNotifier | None,
) -> Iterator[None]:
    """Bind retry-telemetry context for the duration of a ``with`` block.

    Wrap each ``provider.complete()`` / ``provider.complete_stream()``
    call so the httpx hook can attribute retries to the right run +
    iteration and route events through the loop's recorder/notifier.

    Example::

        with telemetry_context(
            run_id=run_id,
            iteration=iteration,
            recorder=recorder,
            notifier=notifier,
        ):
            response = await provider.complete(...)
    """
    tokens = (
        _run_id.set(run_id),
        _iteration.set(iteration),
        _recorder.set(recorder),
        _notifier.set(notifier),
    )
    try:
        yield
    finally:
        _run_id.reset(tokens[0])
        _iteration.reset(tokens[1])
        _recorder.reset(tokens[2])
        _notifier.reset(tokens[3])


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


async def _on_response(response: httpx.Response) -> None:
    """httpx response hook — emit ``provider.retry`` on retryable failures.

    Fail-open: any exception inside is logged and swallowed so an
    observability bug never breaks an LLM call.
    """
    try:
        if response.status_code not in _RETRYABLE_STATUS:
            return

        recorder = _recorder.get()
        notifier = _notifier.get()
        if recorder is None and notifier is None:
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

        # Fire through the standard governance plumbing. Imports are
        # local to avoid a circular import at module load.
        from dendrux.loops._helpers import (
            notify_governance,
            record_governance,
        )

        if recorder is not None:
            try:
                await record_governance(recorder, "provider.retry", iteration, data)
            except Exception:
                _logger.warning("provider.retry recorder write failed", exc_info=True)
        if notifier is not None:
            try:
                await notify_governance(notifier, "provider.retry", iteration, data)
            except Exception:
                _logger.warning("provider.retry notifier broadcast failed", exc_info=True)
    except Exception:
        _logger.warning("retry telemetry hook raised", exc_info=True)


def make_telemetry_http_client(
    timeout: httpx.Timeout,
) -> httpx.AsyncClient:
    """Build an ``httpx.AsyncClient`` with the dendrux retry hook attached.

    Pass the returned client to the vendor SDK constructor via
    ``http_client=`` so the hook fires on every request the SDK makes,
    including its internal retries.
    """
    return httpx.AsyncClient(
        timeout=timeout,
        event_hooks={"response": [_on_response]},
    )


__all__ = [
    "begin_call_attempt_tracking",
    "call_attempt_tracking",
    "end_call_attempt_tracking",
    "make_telemetry_http_client",
    "telemetry_context",
]
