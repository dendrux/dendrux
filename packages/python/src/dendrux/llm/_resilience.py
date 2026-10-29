"""Provider resilience: typed errors, call deadlines, per-loop clients, stream guards.

Internal module shared by every built-in provider. It owns:

- **Error mapping.** Every SDK and httpx failure becomes a
  :class:`~dendrux.errors.ProviderError` subclass with a category, a
  ``retryable`` flag and a fixed message. Vendor message text never goes into
  the error's message (``str(exc)`` is persisted); it stays only on the SDK
  exception chained as ``__cause__``.
- **Deadlines.** ``call_timeout`` bounds one non-streaming call including the
  SDK's own retries; ``stream_idle_timeout`` bounds the gap between semantic
  stream events; ``stream_total_timeout`` bounds a whole stream. Each is
  enforced with ``asyncio.timeout`` around single awaits (never across a
  ``yield``), so a fired deadline abandons the call and nothing retries it.
- **HTTP client.** The SDK's default client (its connection limits and socket
  keep-alive) with the retry-telemetry hook and a call-state hook, rebuilt per
  event loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import math
import re
import threading
import warnings
import weakref
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, ParamSpec, TypeVar

import httpx

from dendrux.errors import (
    ContentFilteredError,
    ContextLengthExceededError,
    ModelRefusalError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderStreamIncompleteError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from dendrux.llm._helpers import GatewayErrorKind, classify_gateway_error
from dendrux.llm._retry_telemetry import telemetry_event_hooks
from dendrux.llm._transport import configure_request, deferred_client_type, guard_response

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
    from types import ModuleType

    from dendrux.errors import TimeoutReason
    from dendrux.types import StreamEvent

logger = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")

# ------------------------------------------------------------------
# Deadlines
# ------------------------------------------------------------------

DEFAULT_CALL_TIMEOUT = 600.0
DEFAULT_STREAM_IDLE_TIMEOUT = 600.0
CONNECT_TIMEOUT = 10.0
# The httpx read timeout sits this far above the asyncio deadline, so the
# deadline (which never retries) always fires first.
_READ_TIMEOUT_MARGIN = 30.0
# Never "no read timeout": that brings back the multi-hour hang.
_FALLBACK_READ_TIMEOUT = 600.0


class _TimeoutUnset:
    """Marks ``timeout=`` as not passed (the deprecated shorthand)."""

    _instance: _TimeoutUnset | None = None

    def __new__(cls) -> _TimeoutUnset:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<unset>"


TIMEOUT_UNSET: Any = _TimeoutUnset()


def _check_seconds(name: str, value: object, owner: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{owner}: {name} must be a positive number of seconds or None.")
    return float(value)


@dataclass(frozen=True, slots=True)
class Deadlines:
    """Per-call deadlines of one provider instance (seconds; None = off)."""

    call_timeout: float | None
    stream_idle_timeout: float | None
    stream_total_timeout: float | None
    max_stream_bytes: int = 16 * 1024 * 1024
    max_stream_tool_calls: int = 1000
    max_response_bytes: int = 16 * 1024 * 1024

    @property
    def read_timeout(self) -> float:
        """httpx read timeout: the largest active deadline plus a margin."""
        active = [d for d in (self.call_timeout, self.stream_idle_timeout) if d]
        return max(active) + _READ_TIMEOUT_MARGIN if active else _FALLBACK_READ_TIMEOUT


def resolve_deadlines(
    owner: str,
    *,
    timeout: Any,
    call_timeout: float | None,
    stream_idle_timeout: float | None,
    stream_total_timeout: float | None,
    max_stream_bytes: int = 16 * 1024 * 1024,
    max_stream_tool_calls: int = 1000,
    max_response_bytes: int = 16 * 1024 * 1024,
    stacklevel: int = 3,
) -> Deadlines:
    """Validate the deadline kwargs and fold in the deprecated ``timeout=``.

    An explicit ``timeout=X`` sets both ``call_timeout`` and
    ``stream_idle_timeout`` to X, except where one was itself set to a
    non-default value (the new kwarg wins), and warns.
    """
    for name, value in (
        ("max_response_bytes", max_response_bytes),
        ("max_stream_bytes", max_stream_bytes),
        ("max_stream_tool_calls", max_stream_tool_calls),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{owner}: {name} must be a positive integer.")
    call = _check_seconds("call_timeout", call_timeout, owner)
    idle = _check_seconds("stream_idle_timeout", stream_idle_timeout, owner)
    total = _check_seconds("stream_total_timeout", stream_total_timeout, owner)
    if timeout is not TIMEOUT_UNSET:
        legacy = _check_seconds("timeout", timeout, owner)
        warnings.warn(
            f"{owner}(timeout=...) is deprecated: it now sets both call_timeout and "
            f"stream_idle_timeout. Pass call_timeout= and stream_idle_timeout= instead.",
            DeprecationWarning,
            stacklevel=stacklevel,
        )
        if call == DEFAULT_CALL_TIMEOUT:
            call = legacy
        if idle == DEFAULT_STREAM_IDLE_TIMEOUT:
            idle = legacy
    return Deadlines(
        call_timeout=call,
        stream_idle_timeout=idle,
        stream_total_timeout=total,
        max_response_bytes=max_response_bytes,
        max_stream_bytes=max_stream_bytes,
        max_stream_tool_calls=max_stream_tool_calls,
    )


# ------------------------------------------------------------------
# Call state (seen by the httpx hooks of the current call)
# ------------------------------------------------------------------

# Statuses the SDKs retry after a backoff sleep.
_BACKOFF_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


@dataclass
class _CallState:
    in_flight: bool = False
    last_status: int | None = None
    retry_after: float | None = None


_call_state: ContextVar[_CallState | None] = ContextVar("dendrux_provider_call", default=None)


def parse_retry_after(headers: Any) -> float | None:
    """Seconds from ``retry-after-ms`` / ``retry-after`` (numeric form only)."""
    if headers is None:
        return None
    try:
        raw_ms = headers.get("retry-after-ms")
        if raw_ms is not None:
            value = float(raw_ms) / 1000.0
            return value if math.isfinite(value) and value >= 0 else None
        raw = headers.get("retry-after")
        if raw is not None:
            value = float(raw)
            return value if math.isfinite(value) and value >= 0 else None
    except (TypeError, ValueError, AttributeError):
        return None
    return None


async def _on_request(
    request: httpx.Request, *, deadlines: Deadlines | None = None, provider: str = "unknown"
) -> None:
    if deadlines is not None:
        configure_request(
            request,
            response_bytes=deadlines.max_response_bytes,
            event_bytes=deadlines.max_stream_bytes,
            provider=provider,
        )
    state = _call_state.get()
    if state is not None:
        state.in_flight = True


async def _on_response(response: httpx.Response) -> None:
    await guard_response(response)
    try:
        state = _call_state.get()
        if state is not None:
            state.in_flight = False
            state.last_status = response.status_code
            state.retry_after = parse_retry_after(response.headers)
    except Exception:  # observability only; never break a call
        logger.debug("provider call-state hook failed", exc_info=True)


def make_http_client(
    sdk: ModuleType, deadlines: Deadlines, *, provider: str | None = None
) -> httpx.AsyncClient:
    """The SDK's default async client, with Dendrux's hooks and deadlines.

    ``sdk.DefaultAsyncHttpxClient`` keeps the SDK's connection limits
    (1000 / 100 keep-alive) and TCP keep-alive. Redirects stay off, as before.
    """
    client_type = deferred_client_type(sdk.DefaultAsyncHttpxClient)
    client = client_type(
        timeout=httpx.Timeout(deadlines.read_timeout, connect=CONNECT_TIMEOUT),
        follow_redirects=False,
        event_hooks={
            "request": [
                functools.partial(
                    _on_request, deadlines=deadlines, provider=provider or sdk.__name__
                )
            ],
            "response": [_on_response, *telemetry_event_hooks()["response"]],
        },
    )
    return client


# Live loops other than the owner that may hold their own client copy at once.
MAX_LOOP_CLIENTS = 64


class LoopClients(Generic[T]):
    """Keeps an SDK client usable across event loops, with a bounded cache.

    httpx connections belong to the loop that opened them. The first loop to
    use the provider owns its client. When that loop has closed (a later
    ``asyncio.run``), the client's HTTP transport is replaced with a fresh one
    built by ``new_http`` (same SDK object, same connection limits); the old
    transport is dropped, never awaited across loops.

    A different loop that is still alive gets its own copy of the SDK client.
    Copies are keyed by a weak reference to their loop. Every access first
    evicts the copies whose loop has closed or been collected: they cannot be
    awaited from another loop, so they are dropped and their sockets are
    released when the garbage collector finalises them. At most
    :data:`MAX_LOOP_CLIENTS` copies are kept; past that the oldest is evicted
    (a call already using it keeps its own reference and finishes normally).
    """

    def __init__(
        self,
        new_http: Callable[[], httpx.AsyncClient],
        fork: Callable[[T, httpx.AsyncClient], T],
    ) -> None:
        self._new_http = new_http
        self._fork = fork
        self._home_loop: weakref.ref[asyncio.AbstractEventLoop] | None = None
        # id(loop) -> (weak ref to the loop, client copy), oldest first.
        self._others: dict[int, tuple[weakref.ref[asyncio.AbstractEventLoop], T]] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        """Number of cached copies for loops other than the owner."""
        with self._lock:
            return len(self._others)

    def _evict_dead_locked(self) -> None:
        for key, (ref, _client) in list(self._others.items()):
            loop = ref()
            if loop is None or loop.is_closed():
                del self._others[key]
                logger.debug("provider.client_evicted reason=event_loop_closed")

    def resolve(self, home: T) -> T:
        """The client to use on the running loop."""
        loop = asyncio.get_running_loop()
        with self._lock:
            self._evict_dead_locked()
            bound = self._home_loop() if self._home_loop is not None else None
            if self._home_loop is None or bound is loop:
                self._home_loop = weakref.ref(loop)
                return home
            if bound is None or bound.is_closed():
                home._client = self._new_http()  # type: ignore[attr-defined]
                self._home_loop = weakref.ref(loop)
                logger.debug("provider.client_rebuilt reason=event_loop_changed")
                return home
            entry = self._others.get(id(loop))
            if entry is not None and entry[0]() is loop:
                return entry[1]
            client = self._fork(home, self._new_http())
            self._others[id(loop)] = (weakref.ref(loop), client)
            while len(self._others) > MAX_LOOP_CLIENTS:
                del self._others[next(iter(self._others))]
                logger.debug("provider.client_evicted reason=cache_full")
            return client

    def pop_current_loop_extra(self) -> T | None:
        """Detach the copy used by the running loop, if any (for ``close``).

        Copies of other live loops stay cached: they cannot be closed from
        this loop, and are evicted once their loop closes.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        with self._lock:
            self._evict_dead_locked()
            entry = self._others.get(id(loop))
            if entry is None or entry[0]() is not loop:
                return None
            del self._others[id(loop)]
            return entry[1]


# ------------------------------------------------------------------
# Error mapping
# ------------------------------------------------------------------

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.:\-]{1,64}$")

_AUTH_TYPES = frozenset({"authentication_error", "permission_error"})
_AUTH_CODES = frozenset(
    {"invalid_api_key", "unauthorized", "forbidden", "permission_denied", "access_denied"}
)
_CREDIT_TYPES = frozenset({"billing_error"})
_CREDIT_CODES = frozenset(
    {"insufficient_quota", "quota_exceeded", "billing_hard_limit_reached", "insufficient_credits"}
)
_RATE_TYPES = frozenset({"rate_limit_error"})
_RATE_CODES = frozenset({"rate_limit_exceeded", "rate_limit", "rate_limited", "too_many_requests"})
_UNAVAILABLE_TYPES = frozenset({"api_error", "overloaded_error", "server_error", "timeout_error"})
_UNAVAILABLE_CODES = frozenset(
    {"server_error", "overloaded", "service_unavailable", "internal_error", "server_overloaded"}
)
# 4xx statuses decided by the status alone (408/409/429 are retryable kinds).
_CLIENT_ERROR_STATUSES = frozenset(range(400, 500)) - {408, 409, 429}
_REJECTED_TYPES = frozenset(
    {"invalid_request_error", "not_found_error", "request_too_large", "invalid_request"}
)
_FILTER_CODES = frozenset(
    {"content_filter", "content_policy_violation", "content_filtered", "responsible_ai_policy"}
)

_MESSAGES = {
    "auth": "The provider rejected the credentials.",
    "credits": "The provider account has no remaining credits or quota.",
    "rate_limit": "The provider rate-limited the request.",
    "unavailable": "The provider is unavailable or overloaded.",
    "rejected": "The provider rejected the request.",
    "content_filter": "The provider's content filter stopped the request or response.",
    "unknown": "The provider reported an error.",
}


def safe_token(value: object) -> str | None:
    """``value`` when it looks like a code (``[A-Za-z0-9_.:-]{1,64}``), else None."""
    return value if isinstance(value, str) and _SAFE_TOKEN.match(value) else None


def context_length_error(
    *, provider: str, model: str | None, status_code: int | None, provider_code: str | None
) -> ContextLengthExceededError:
    where = f" for model '{model}'" if model else ""
    status = f" (HTTP {status_code})" if status_code else ""
    return ContextLengthExceededError(
        f"Request exceeds the model's context window{where}{status}. "
        "Reduce the conversation history, tool results or max_tokens.",
        status_code=status_code,
        provider_code=provider_code,
        model=model,
        provider=provider,
    )


def error_from_fields(
    *,
    provider: str,
    model: str | None,
    status_code: int | None,
    code: str | None,
    error_type: str | None,
    param: str | None = None,
    vendor_text: str = "",
    body: object = None,
    headers: Any = None,
    streaming: bool = False,
    endpoint: str | None = None,
    message: str | None = None,
) -> Exception:
    """The typed error for one vendor failure, described by its scalar fields.

    ``vendor_text`` feeds classification only. ``message`` overrides the fixed
    per-category message (image-bearing requests keep their sanitised vendor
    message there). Returns ``NotImplementedError`` for a streaming request the
    endpoint rejects as unsupported.
    """
    classified = classify_gateway_error(
        status_code=status_code,
        code=code,
        param=param,
        message=vendor_text,
        body=body if body is not None else ({"message": vendor_text} if vendor_text else None),
    )
    provider_code = classified.provider_code or code
    if classified.kind is GatewayErrorKind.CONTEXT_LENGTH:
        return context_length_error(
            provider=provider, model=model, status_code=status_code, provider_code=provider_code
        )
    if streaming and classified.kind is GatewayErrorKind.STREAMING_UNSUPPORTED:
        where = f" by endpoint at {endpoint}" if endpoint else " by the endpoint"
        return NotImplementedError(
            f"Streaming request rejected{where}. If this endpoint does not support "
            f"streaming, use agent.run() instead. (Provider detail is on __cause__.)"
        )
    lcode = (provider_code or "").lower()
    ltype = (error_type or "").lower()
    # OpenRouter puts an HTTP-like numeric code inside in-stream errors.
    effective_status = status_code
    if effective_status is None and lcode.isdigit() and len(lcode) == 3:
        effective_status = int(lcode)
    common: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "status_code": status_code,
        "provider_code": provider_code,
        "error_type": safe_token(error_type),
    }

    def text(category: str) -> str:
        return message if message is not None else _MESSAGES[category]

    if lcode in _FILTER_CODES:
        return ContentFilteredError(**common, message=text("content_filter"))
    if effective_status is not None and effective_status in _CLIENT_ERROR_STATUSES:
        # A definite client error: the status decides, never a 5xx-ish
        # type or code (some gateways label 4xx bodies ``api_error``), so it
        # is never retryable. Anthropic's "credit balance is too low" is a 400
        # ``invalid_request_error`` and lands in rejected.
        if effective_status in (401, 403):
            return ProviderAuthError(**common, message=text("auth"))
        if effective_status == 402 or ltype in _CREDIT_TYPES or lcode in _CREDIT_CODES:
            return ProviderError(**common, category="credits", message=text("credits"))
        return ProviderError(**common, category="rejected", message=text("rejected"))
    if effective_status in (401, 403) or ltype in _AUTH_TYPES or lcode in _AUTH_CODES:
        return ProviderAuthError(**common, message=text("auth"))
    if effective_status == 402 or ltype in _CREDIT_TYPES or lcode in _CREDIT_CODES:
        return ProviderError(**common, category="credits", message=text("credits"))
    if effective_status == 429 or ltype in _RATE_TYPES or lcode in _RATE_CODES:
        return ProviderRateLimitError(
            **common, message=text("rate_limit"), retry_after=parse_retry_after(headers)
        )
    if (
        (
            effective_status is not None
            and (effective_status >= 500 or effective_status in (408, 409))
        )
        or ltype in _UNAVAILABLE_TYPES
        or lcode in _UNAVAILABLE_CODES
    ):
        return ProviderUnavailableError(**common, message=text("unavailable"))
    if (effective_status is not None and 400 <= effective_status < 500) or ltype in _REJECTED_TYPES:
        return ProviderError(**common, category="rejected", message=text("rejected"))
    return ProviderError(**common, category="unknown", message=text("unknown"))


def _error_facts(exc: BaseException) -> dict[str, Any]:
    """Scalars of an SDK ``APIError`` (both SDK body shapes; never the request)."""
    status = getattr(exc, "status_code", None)
    status_code = status if isinstance(status, int) and status >= 300 else None
    body = getattr(exc, "body", None)
    inner: Any = body
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        inner = body["error"]
    vendor_text = ""
    code: str | None = None
    error_type: str | None = None
    param: str | None = None
    if isinstance(inner, dict):
        raw_text = inner.get("message")
        vendor_text = raw_text if isinstance(raw_text, str) else ""
        raw_code = inner.get("code")
        if isinstance(raw_code, str):
            code = raw_code or None
        elif isinstance(raw_code, int) and not isinstance(raw_code, bool) and raw_code != status:
            code = str(raw_code)
        raw_type = inner.get("type")
        error_type = raw_type if isinstance(raw_type, str) and raw_type else None
        raw_param = inner.get("param")
        param = raw_param if isinstance(raw_param, str) and raw_param else None
    elif isinstance(inner, str):
        vendor_text = inner
    if not vendor_text:
        sdk_text = getattr(exc, "message", None)
        vendor_text = sdk_text if isinstance(sdk_text, str) else ""
    if code is None:
        sdk_code = getattr(exc, "code", None)
        code = sdk_code if isinstance(sdk_code, str) and sdk_code else None
    if error_type is None:
        sdk_type = getattr(exc, "type", None)
        error_type = sdk_type if isinstance(sdk_type, str) and sdk_type else None
    if param is None:
        sdk_param = getattr(exc, "param", None)
        param = sdk_param if isinstance(sdk_param, str) and sdk_param else None
    response = getattr(exc, "response", None)
    return {
        "status_code": status_code,
        "code": code,
        "error_type": error_type,
        "param": param,
        "vendor_text": vendor_text,
        "body": inner if isinstance(inner, (dict, str)) else None,
        "headers": getattr(response, "headers", None) if status_code is not None else None,
    }


def _timeout_reason_for(exc: BaseException, *, streaming: bool) -> TimeoutReason | None:
    cause: BaseException | None = exc
    while cause is not None:
        if isinstance(cause, (httpx.ConnectTimeout, httpx.PoolTimeout)):
            return "connect"
        cause = cause.__cause__
    return "stream_idle" if streaming else None


def timeout_message(reason: TimeoutReason, limit: float | None = None) -> str:
    """Fixed, actionable message for a timeout (never vendor text)."""
    shown = f"={limit:g}s" if limit is not None else ""
    if reason == "connect":
        return "The connection to the provider timed out before the request was sent."
    if reason == "call_deadline":
        return (
            f"The LLM call timed out: it exceeded call_timeout{shown} and was abandoned "
            "(not retried). "
            "For long generations use streaming (agent.stream()), or raise call_timeout "
            "on the provider."
        )
    if reason == "stream_idle":
        return (
            f"The stream timed out: no event arrived within stream_idle_timeout{shown}; it was "
            "abandoned (not retried). Raise stream_idle_timeout on the provider for slow models."
        )
    if reason == "stream_total":
        return f"The stream timed out: it exceeded stream_total_timeout{shown} and was abandoned."
    return f"The call timed out: call_timeout{shown} expired while waiting to retry."


def map_sdk_error(
    exc: BaseException,
    *,
    sdk: ModuleType,
    provider: str,
    model: str | None,
    streaming: bool,
    endpoint: str | None = None,
) -> Exception | None:
    """Typed error for an SDK or httpx failure, or None to leave ``exc`` as is.

    ``streaming`` is True for any failure of a streaming call. Mid-stream
    transport failures (the SDKs do not wrap them) become
    ``ProviderStreamIncompleteError``; httpx timeouts become
    ``ProviderTimeoutError``.
    """
    if isinstance(exc, ProviderError):
        return None
    if isinstance(exc, (sdk.APITimeoutError, httpx.TimeoutException)):
        reason = _timeout_reason_for(exc, streaming=streaming)
        return ProviderTimeoutError(
            provider=provider,
            model=model,
            error_type="timeout",
            timeout_reason=reason,
            retryable=reason == "connect",
            message=(
                timeout_message(reason)
                if reason == "connect"
                else "The provider connection timed out reading the response (socket read "
                "timeout, after SDK retries). For long generations use streaming, or "
                "raise call_timeout / stream_idle_timeout on the provider."
            ),
        )
    if isinstance(exc, sdk.APIConnectionError):
        return ProviderUnavailableError(
            provider=provider,
            model=model,
            error_type="connection",
            message="Could not reach the provider (connection failed after retries).",
        )
    if isinstance(exc, sdk.APIError):
        facts = _error_facts(exc)
        return error_from_fields(
            provider=provider,
            model=model,
            streaming=streaming,
            endpoint=endpoint,
            **facts,
        )
    if isinstance(exc, httpx.TransportError):
        if streaming:
            return ProviderStreamIncompleteError(
                provider=provider,
                model=model,
                error_type="transport",
                message="The response stream broke before its terminal event.",
            )
        return ProviderUnavailableError(
            provider=provider,
            model=model,
            error_type="transport",
            message="The connection to the provider failed.",
        )
    return None


def stream_incomplete(provider: str, model: str | None) -> ProviderStreamIncompleteError:
    return ProviderStreamIncompleteError(
        provider=provider,
        model=model,
        message="The response stream ended without its terminal event.",
    )


def refusal_error(
    provider: str, model: str | None, *, refusal_text: str | None, code: str | None = None
) -> ModelRefusalError:
    return ModelRefusalError(
        provider=provider,
        model=model,
        provider_code=code,
        message="The model refused to answer.",
        refusal_text=refusal_text or None,
    )


def content_filter_error(
    provider: str, model: str | None, *, code: str | None = "content_filter"
) -> ContentFilteredError:
    return ContentFilteredError(
        provider=provider,
        model=model,
        provider_code=code,
        message=_MESSAGES["content_filter"],
    )


# ------------------------------------------------------------------
# Deadline enforcement
# ------------------------------------------------------------------


def _deadline_error(
    state: _CallState,
    *,
    reason: TimeoutReason,
    limit: float | None,
    provider: str,
    model: str | None,
) -> ProviderError:
    status = state.last_status
    if status in _BACKOFF_STATUSES and not state.in_flight:
        # The deadline fired while the SDK slept before a retry: keep the
        # category of the failure being retried.
        message = timeout_message("deadline_during_backoff", limit)
        if status == 429:
            return ProviderRateLimitError(
                provider=provider,
                model=model,
                status_code=status,
                retryable=False,
                timeout_reason="deadline_during_backoff",
                retry_after=state.retry_after,
                message=message,
            )
        return ProviderUnavailableError(
            provider=provider,
            model=model,
            status_code=status,
            retryable=False,
            timeout_reason="deadline_during_backoff",
            message=message,
        )
    return ProviderTimeoutError(
        provider=provider,
        model=model,
        timeout_reason=reason,
        retryable=False,
        message=timeout_message(reason, limit),
    )


def log_provider_error(error: ProviderError, run_id: str | None) -> None:
    """One structured log line per typed provider error (codes only, no vendor text)."""
    logger.warning(
        "provider.error run_id=%s provider=%s model=%s category=%s status=%s code=%s "
        "timeout_reason=%s retryable=%s",
        run_id,
        error.provider,
        error.model,
        error.category,
        error.status_code,
        error.safe_provider_code,
        error.timeout_reason,
        error.retryable,
        extra={"run_id": run_id, "code": error.safe_provider_code or error.category},
    )


def logs_provider_errors(
    func: Callable[P, Coroutine[Any, Any, T]],
) -> Callable[P, Coroutine[Any, Any, T]]:
    """Log every :class:`ProviderError` a ``complete`` raises, with its ``run_id``."""

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return await func(*args, **kwargs)
        except ProviderError as error:
            run_id = kwargs.get("run_id")
            log_provider_error(error, run_id if isinstance(run_id, str) else None)
            raise

    return wrapper


def logs_provider_stream_errors(
    func: Callable[P, AsyncGenerator[StreamEvent, None]],
) -> Callable[P, AsyncGenerator[StreamEvent, None]]:
    """Log every :class:`ProviderError` a ``complete_stream`` raises, with its ``run_id``."""

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> AsyncGenerator[StreamEvent, None]:
        try:
            async with contextlib.aclosing(func(*args, **kwargs)) as events:
                async for event in events:
                    yield event
        except ProviderError as error:
            run_id = kwargs.get("run_id")
            log_provider_error(error, run_id if isinstance(run_id, str) else None)
            raise

    return wrapper


async def run_with_deadline(
    make_call: Callable[[], Awaitable[T]],
    *,
    when: float | None,
    reason: TimeoutReason,
    limit: float | None,
    provider: str,
    model: str | None,
) -> T:
    """Await ``make_call()`` (one SDK call, its retries included) under ``when``.

    ``when`` is an absolute ``loop.time()``; None means no deadline. On
    expiry the call is abandoned and a typed error is raised (not chained).
    """
    state = _CallState()
    token = _call_state.set(state)
    try:
        if when is None:
            return await make_call()
        timer = asyncio.timeout_at(when)
        try:
            async with timer:
                return await make_call()
        except TimeoutError:
            if not timer.expired():
                raise
        raise _deadline_error(state, reason=reason, limit=limit, provider=provider, model=model)
    finally:
        _call_state.reset(token)


async def call_with_timeout(
    make_call: Callable[[], Awaitable[T]],
    *,
    deadlines: Deadlines,
    provider: str,
    model: str | None,
) -> T:
    """A non-streaming call bounded by ``call_timeout`` (SDK retries included)."""
    limit = deadlines.call_timeout
    when = asyncio.get_running_loop().time() + limit if limit is not None else None
    return await run_with_deadline(
        make_call, when=when, reason="call_deadline", limit=limit, provider=provider, model=model
    )


class StreamClock:
    """Idle and total deadlines for one streaming call.

    ``open`` bounds the request that opens the stream (SDK retries included);
    ``step`` bounds one pull from the SDK iterator. ``mark_semantic`` resets
    the idle clock; call it after handing a semantic event to the consumer,
    so time the consumer spends is never counted against the vendor.
    """

    def __init__(self, deadlines: Deadlines, *, provider: str, model: str | None) -> None:
        self._loop = asyncio.get_running_loop()
        self._idle = deadlines.stream_idle_timeout
        self._total = deadlines.stream_total_timeout
        self._remaining = deadlines.max_stream_bytes
        self._slot_limit = deadlines.max_stream_tool_calls
        self._slots: set[object] = set()
        now = self._loop.time()
        self._idle_at = now + self._idle if self._idle is not None else None
        self._total_at = now + self._total if self._total is not None else None
        self.provider = provider
        self.model = model

    def _reject(self, code: str) -> None:
        raise ProviderResponseError(
            provider=self.provider, model=self.model, provider_code=code, retryable=False
        )

    def tool_slot(self, key: object) -> None:
        """Bound distinct tool-call buffers before an adapter creates one."""
        if key not in self._slots:
            if len(self._slots) >= self._slot_limit:
                self._reject("stream_tool_call_limit")
            self._slots.add(key)

    def check_tool_count(self, count: int) -> None:
        """Bound completed calls even when upstream indices or ids are reused."""
        if count > self._slot_limit:
            self._reject("stream_tool_call_limit")

    def account(self, value: Any = None, *, credited_bytes: int = 0) -> int:
        """Charge retained fragments and eight bytes per accounting event.

        Adapters supply only newly retained data, never an SDK event or its
        cumulative snapshot. A final argument value credits its prior deltas.
        """
        self._remaining -= 8
        if self._remaining < 0:
            self._reject("stream_size_limit")
        size = self._retained_size(value, self._remaining + credited_bytes)
        self._remaining -= max(0, size - credited_bytes)
        if self._remaining < 0:
            self._reject("stream_size_limit")
        return size

    def _retained_size(self, value: Any, maximum: int, depth: int = 0) -> int:
        if maximum < 0 or depth > 64:
            self._reject("stream_size_limit")
        if isinstance(value, str):
            if len(value) > maximum:
                self._reject("stream_size_limit")
            size = 0
            for start in range(0, len(value), 65536):
                size += len(value[start : start + 65536].encode("utf-8"))
                if size > maximum:
                    self._reject("stream_size_limit")
            return size
        if isinstance(value, (dict, list, tuple)):
            size = 8
            values = value.items() if isinstance(value, dict) else value
            for item in values:
                size += self._retained_size(item, maximum - size, depth + 1)
            if size > maximum:
                self._reject("stream_size_limit")
            return size
        return 8 if value is not None else 0

    def mark_semantic(self) -> None:
        if self._idle is not None:
            self._idle_at = self._loop.time() + self._idle

    def _next(self) -> tuple[float | None, TimeoutReason, float | None]:
        if self._idle_at is None and self._total_at is None:
            return None, "stream_idle", None
        total_first = self._idle_at is None or (
            self._total_at is not None and self._total_at <= self._idle_at
        )
        if self._total_at is not None and total_first:
            return self._total_at, "stream_total", self._total
        return self._idle_at, "stream_idle", self._idle

    async def open(self, make_call: Callable[[], Awaitable[T]]) -> T:
        when, reason, limit = self._next()
        return await run_with_deadline(
            make_call,
            when=when,
            reason=reason,
            limit=limit,
            provider=self.provider,
            model=self.model,
        )

    async def step(self, awaitable: Awaitable[T]) -> T:
        when, reason, limit = self._next()
        if when is None:
            return await awaitable
        timer = asyncio.timeout_at(when)
        try:
            async with timer:
                return await awaitable
        except TimeoutError:
            if not timer.expired():
                raise
        raise ProviderTimeoutError(
            provider=self.provider,
            model=self.model,
            timeout_reason=reason,
            retryable=False,
            message=timeout_message(reason, limit),
        )


async def retry_unstarted_stream(
    open_attempt: Callable[[], AsyncGenerator[StreamEvent, None]],
    *,
    provider: str,
    model: str | None,
    run_id: str | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    """Re-yield one stream attempt; retry once if it broke before yielding anything.

    Only :class:`ProviderStreamIncompleteError` is retried, and only when no
    event (text, reasoning or tool) has reached the consumer. The attempt
    generator is always closed, so early ``aclose()`` and cancellation close
    the upstream response.
    """
    attempts = 0
    while True:
        attempts += 1
        attempt = open_attempt()
        yielded = False
        retry = False
        try:
            async for event in attempt:
                yielded = True
                yield event
            return
        except ProviderStreamIncompleteError:
            if yielded or attempts > 1:
                raise
            retry = True
        finally:
            await attempt.aclose()
        if retry:
            logger.warning(
                "provider.stream_retry run_id=%s provider=%s model=%s reason=stream_incomplete",
                run_id,
                provider,
                model,
            )


class WireRequest:
    """One built request, carried through a stream attempt.

    Its ``repr`` shows nothing: traceback frame locals (as error trackers
    capture them) must never expose request bodies, which may hold images.
    """

    __slots__ = ("api_kwargs", "captured_request", "legacy", "registry")

    def __init__(
        self,
        api_kwargs: dict[str, Any],
        captured_request: dict[str, Any],
        registry: Any,
        legacy: list[dict[str, Any]] | None = None,
    ) -> None:
        self.api_kwargs = api_kwargs
        self.captured_request = captured_request
        self.registry = registry
        self.legacy = legacy or []

    def __repr__(self) -> str:
        return "<WireRequest>"

    @property
    def model(self) -> str:
        return str(self.api_kwargs["model"])


async def close_quietly(resource: Any) -> None:
    """Close an SDK stream or response; never raises."""
    for name in ("aclose", "close"):
        close = getattr(resource, name, None)
        if close is None:
            continue
        try:
            result = close()
            if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
                await result
        except Exception:
            logger.debug("closing a provider stream failed", exc_info=True)
        return
