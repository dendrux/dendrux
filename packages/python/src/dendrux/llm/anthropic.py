"""Anthropic Messages API provider.

Thin adapter between Dendrux's universal types and the Anthropic Messages API.
Handles bidirectional conversion:
  - Outbound: Message/ToolDef → Anthropic MessageParam/ToolParam
  - Inbound:  Anthropic response → LLMResponse/ToolCall

No business logic, no agent loop awareness. Just shape translation.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import hashlib
import json
import logging
import os
import re
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Literal

import anthropic

from dendrux.errors import ProviderError
from dendrux.llm._helpers import (
    build_call_index,
    build_tool_call,
    replayable_reasoning_blocks,
    resolve_tool_message_call,
)
from dendrux.llm._images import (
    WireImageRegistry,
    anthropic_image,
    anthropic_image_support,
    anthropic_text,
    convert_parts,
    normalize_base_url,
    sanitize_vendor_message,
    sdk_image_request_error,
)
from dendrux.llm._resilience import (
    DEFAULT_CALL_TIMEOUT,
    DEFAULT_STREAM_IDLE_TIMEOUT,
    TIMEOUT_UNSET,
    LoopClients,
    StreamClock,
    WireRequest,
    call_with_timeout,
    logs_provider_errors,
    logs_provider_stream_errors,
    make_http_client,
    map_sdk_error,
    refusal_error,
    resolve_deadlines,
    retry_unstarted_stream,
    stream_incomplete,
)
from dendrux.llm._retry_telemetry import call_attempt_tracking
from dendrux.llm._schema import (
    ANTHROPIC_STRICT_MAX_OPTIONAL,
    ANTHROPIC_STRICT_MAX_TOOLS,
    ANTHROPIC_STRICT_MAX_UNIONS,
    anthropic_strict_budget,
    prepare_anthropic_strict,
)
from dendrux.llm._stop_reason import anthropic_stop_reason
from dendrux.llm.base import LLMProvider
from dendrux.types import (
    REPLAYED_META_KEY,
    LLMResponse,
    ProviderCapabilities,
    Role,
    StreamEvent,
    StreamEventType,
    ToolCall,
    UsageStats,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    import httpx

    from dendrux.llm.base import ImageSupport
    from dendrux.types import Message, ToolDef

logger = logging.getLogger(__name__)

# Provider name on typed errors.
_PROVIDER = "anthropic"

_STRUCTURED_TOOL_NAME = "structured_output"
_STRUCTURED_TOOL_DESCRIPTION = "Return your response in this exact format."

# Strict-mode fallback. Any 400 on a request that carried a strict entry is
# retried once with every strict entry reverted to its non-strict (0.4.0)
# payload. Only a 400 that clearly points at a strict schema (a ``tools.N.``
# path on a strict entry, or a request-level grammar/limit message; wording
# recorded in live probes 2026-10-18) is remembered, per (model, entry), so
# later requests skip the failing round trip. The memory is a bounded LRU; an
# evicted entry is simply tried strict again.
_STRICT_ERROR_PATH = re.compile(r"^tools\.(\d+)\.")
_STRICT_ERROR_MARKERS = (
    "grammar",
    "too complex",
    "schemas contains too many",
    "strict tools",
)
_STRICT_REJECTED_MAX = 1024
_STRICT_REJECTED: OrderedDict[tuple[str, str], None] = OrderedDict()
_STRICT_WARNED_MAX = 1024
_STRICT_WARNED: set[str] = set()
_STRICT_LOCK = threading.Lock()


def _warn_strict_once(key: str, message: str, *args: Any) -> None:
    """WARNING the first time ``key`` is seen in this process, DEBUG after."""
    with _STRICT_LOCK:
        first = key not in _STRICT_WARNED
        if first:
            if len(_STRICT_WARNED) >= _STRICT_WARNED_MAX:
                _STRICT_WARNED.clear()
            _STRICT_WARNED.add(key)
    (logger.warning if first else logger.debug)(message, *args)


def _message_ref(message: str) -> str:
    """A short digest of a vendor error message: correlates log lines, shows no text."""
    return hashlib.sha256(message.encode()).hexdigest()[:12]


def _strict_entry_key(entry: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()


def _is_strict_rejected(model: str, entry: dict[str, Any]) -> bool:
    key = (model, _strict_entry_key(entry))
    with _STRICT_LOCK:
        if key not in _STRICT_REJECTED:
            return False
        _STRICT_REJECTED.move_to_end(key)
        return True


def _remember_strict_rejected(model: str, entry: dict[str, Any]) -> None:
    key = (model, _strict_entry_key(entry))
    with _STRICT_LOCK:
        _STRICT_REJECTED[key] = None
        _STRICT_REJECTED.move_to_end(key)
        while len(_STRICT_REJECTED) > _STRICT_REJECTED_MAX:
            _STRICT_REJECTED.popitem(last=False)


def _schema_digest(schema: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()[:12]


def _strict_failure(exc: anthropic.APIError, sent_tools: Any) -> tuple[str, bool] | None:
    """``(message, points_at_strict)`` for a 400 on a request with strict entries.

    None when the request carried no strict entry or the error is not a 400
    (then it is raised unchanged). ``points_at_strict`` is True only for the
    recorded strict-schema wordings; any other 400 is still retried once
    without strict but is not remembered.
    """
    if not isinstance(exc, anthropic.BadRequestError) or not isinstance(sent_tools, list):
        return None
    strict_indices = {
        i for i, t in enumerate(sent_tools) if isinstance(t, dict) and t.get("strict") is True
    }
    if not strict_indices:
        return None
    error = exc.body.get("error") if isinstance(exc.body, dict) else None
    raw = error.get("message") if isinstance(error, dict) else None
    message = sanitize_vendor_message(raw if isinstance(raw, str) else str(exc))
    if not isinstance(error, dict) or error.get("type") != "invalid_request_error":
        return message, False
    path = _STRICT_ERROR_PATH.match(message)
    if path is not None:
        return message, int(path.group(1)) in strict_indices
    lowered = message.lower()
    return message, any(m in lowered for m in _STRICT_ERROR_MARKERS)


# Anthropic-specific kwargs that complete() will forward to the API.
# Anything not in this set is silently ignored to prevent leaking
# unsupported params into the API call.
_SUPPORTED_KWARGS = frozenset(
    {"temperature", "top_p", "top_k", "stop_sequences", "model", "max_tokens"}
)

# Dendrux effort vocabulary → Anthropic. "extra" is the consumer label for
# "xhigh"; every other value passes through (Anthropic validates it).
_EFFORT_ALIASES = {"extra": "xhigh"}

# Content block types replayed verbatim on assistant turns, and never given a
# cache_control marker.
_THINKING_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})

# Sent when thinking blocks replayed from an earlier run are on the wire, so an
# account enforcing preserved-thinking prefix binding drops a mismatched block
# instead of failing the request.
_THINKING_BINDING_BETA = "thinking-binding-controls-2026-08-01"


def _normalize_effort(effort: str) -> str:
    """Map a Dendrux effort alias to Anthropic's vocabulary."""
    return _EFFORT_ALIASES.get(effort, effort)


def _reasoning_tokens_from_details(details: Any) -> int | None:
    """Read thinking_tokens from output_tokens_details (dict or typed object)."""
    if isinstance(details, dict):
        return details.get("thinking_tokens")
    if details is not None:
        return getattr(details, "thinking_tokens", None)
    return None


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API provider.

    Converts Dendrux's universal Message/ToolDef types to Anthropic's native
    format and normalizes responses back. Supports both batch (complete())
    and streaming (complete_stream()) with token-by-token text deltas and
    tool call events.

    Capabilities reflect what this adapter *implements*, not what the
    Anthropic API can theoretically do. Image parts are sent as base64
    ``image`` blocks; images in tool results go inside the ``tool_result``
    block (native mapping).

    Usage:
        provider = AnthropicProvider(api_key="sk-...", model="claude-sonnet-4-6")
        response = await provider.complete(messages, tools=tools)
    """

    tool_result_images = "native"

    capabilities = ProviderCapabilities(
        supports_native_tools=True,
        supports_tool_call_ids=True,
        supports_streaming=True,
        supports_streaming_tool_deltas=True,
        supports_thinking=True,
        supports_multimodal=True,
        supports_system_prompt=True,
        supports_parallel_tool_calls=True,
        supports_structured_output=True,
        max_context_tokens=200_000,
    )

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 16_000,
        temperature: float | None = None,
        thinking: bool = False,
        effort: str | None = None,
        show_thinking: bool = True,
        thinking_budget: int | None = None,
        timeout: Any = TIMEOUT_UNSET,
        max_retries: int = 3,
        cache_ttl: Literal["5m", "1h"] | None = None,
        strict: bool = True,
        call_timeout: float | None = DEFAULT_CALL_TIMEOUT,
        stream_idle_timeout: float | None = DEFAULT_STREAM_IDLE_TIMEOUT,
        stream_total_timeout: float | None = None,
    ) -> None:
        """Create an Anthropic provider.

        Args:
            model: Model identifier (e.g. "claude-sonnet-4-6").
            api_key: API key. Defaults to ANTHROPIC_API_KEY env var.
            max_tokens: Maximum output tokens per call. Override per-call via kwargs.
            temperature: Sampling temperature. None = model default. Override per-call.
            thinking: Enable extended/adaptive thinking. Default False (off).
                When on, ``temperature`` is omitted (incompatible with thinking).
            effort: Thinking effort — low|medium|high|xhigh|max ("extra"→xhigh).
                Sent as ``output_config.effort`` when thinking is on.
            show_thinking: When thinking is on, ``True`` returns summarized
                thinking text (display=summarized); ``False`` omits it
                (display=omitted, faster first token). Default True.
            thinking_budget: Force legacy manual thinking with a fixed
                ``budget_tokens`` (older models / precise cost). Default None
                → adaptive thinking.
            timeout: Deprecated shorthand: sets both ``call_timeout`` and
                ``stream_idle_timeout`` (with a ``DeprecationWarning``).
            max_retries: Number of automatic SDK retries on transient errors
                (connection resets, 408/409/429/5xx). They all count against
                ``call_timeout``.
            cache_ttl: Anthropic prompt-cache TTL. ``None`` (default) omits the
                field so Anthropic uses its 5-minute default. Set ``"1h"`` for
                long-running workflows where iterations span more than 5 minutes
                (costs 2× base input on cache creation, reads stay cheap).
            strict: Use Anthropic strict mode (grammar-constrained sampling).
                ``True`` (default): ``@tool(strict=True)`` tools, and structured
                output whose schema allows it, are sent with ``"strict": true``
                and a schema adapted to Anthropic's strict subset; a 400 on such
                a request is retried once without strict. ``False``: never send
                strict; tool and structured-output payloads are exactly as in
                0.4.0. Anthropic caches strict schemas server-side for up to 24
                hours (see its data-retention notes), so keep sensitive data
                such as PHI out of schemas or pass ``False``.
            call_timeout: Wall-clock limit in seconds for one non-streaming
                call, SDK retries included. When it fires the call is
                abandoned and never retried (``ProviderTimeoutError``).
                ``None`` turns it off, but the socket read timeout still
                applies (the largest deadline + 30 s, or 600 s) and the SDK
                retries a read timeout, so a slow
                generation can be re-run (and paid for) up to
                ``max_retries + 1`` times. For long generations stream, or
                set a large ``call_timeout`` instead.
            stream_idle_timeout: Maximum gap in seconds between meaningful
                stream events (``ping`` does not count), and before the first
                one. ``None`` turns it off.
            stream_total_timeout: Wall-clock limit in seconds for a whole
                stream. ``None`` (default) means no limit.
        """
        if api_key is None and not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError(
                "AnthropicProvider needs an API key. "
                "Pass api_key='sk-...' or set the ANTHROPIC_API_KEY environment variable."
            )
        self._deadlines = resolve_deadlines(
            "AnthropicProvider",
            timeout=timeout,
            call_timeout=call_timeout,
            stream_idle_timeout=stream_idle_timeout,
            stream_total_timeout=stream_total_timeout,
        )
        deadlines = self._deadlines

        def new_http() -> httpx.AsyncClient:
            return make_http_client(anthropic, deadlines)

        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            http_client=new_http(),
            max_retries=max_retries,
        )
        self._clients: LoopClients[anthropic.AsyncAnthropic] = LoopClients(
            new_http, lambda client, http: client.copy(http_client=http)
        )
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._thinking = thinking
        self._effort = effort
        self._show_thinking = show_thinking
        self._thinking_budget = thinking_budget
        self._cache_ttl = cache_ttl
        self._strict = strict

    @property
    def model(self) -> str:
        """The model identifier this provider is configured to use."""
        return self._model

    def __repr__(self) -> str:
        return f"AnthropicProvider(model={self._model!r})"

    async def close(self) -> None:
        """Close the underlying HTTP client and release connections."""
        # Close only what belongs to the running loop: after a loop change the
        # home transport is first rebuilt, never awaited across loops.
        current = self._clients.resolve(self._client)
        self._clients.pop_current_loop_extra()
        await current.close()

    def _sdk(self) -> anthropic.AsyncAnthropic:
        """The SDK client for the running event loop (rebuilt after a loop change)."""
        return self._clients.resolve(self._client)

    # ------------------------------------------------------------------
    # Image input (0.4.0)
    # ------------------------------------------------------------------

    async def image_input_support(self, model: str) -> ImageSupport:
        """``"yes"`` for ``claude-*`` models; ``"unknown"`` for any other name."""
        return anthropic_image_support(model)

    @property
    def thinking_enabled(self) -> bool:
        """True when the provider was configured with ``thinking=True``."""
        return bool(self._thinking)

    def many_image_rule(self, model: str) -> tuple[int, int] | None:
        """More than 20 images in a request => every image's long side <= 2000 px."""
        return (20, 2000)

    def image_capability_key(self, model: str) -> tuple[str, str, str]:
        return ("anthropic", normalize_base_url(self._client.base_url), model)

    async def __aenter__(self) -> AnthropicProvider:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    def _cache_control_marker(self) -> dict[str, Any]:
        """Build the cache_control marker for system + last-message breakpoints.

        Includes ``ttl`` only when the provider was configured with one;
        otherwise Anthropic applies its default 5m.
        """
        marker: dict[str, Any] = {"type": "ephemeral"}
        if self._cache_ttl is not None:
            marker["ttl"] = self._cache_ttl
        return marker

    def _apply_cache_control(
        self,
        system_prompt: str,
        api_messages: list[dict[str, Any]],
        *,
        mark_first: bool = False,
        mark_indices: list[int] | None = None,
    ) -> tuple[Any, list[dict[str, Any]]]:
        """Apply cache_control to system block and selected messages.

        Always marks the last message (warms the next iteration's cache — the
        call that writes it pays full creation, the next reads it). When
        ``mark_first`` is set (a stable context head is present), also marks the
        first user message so the frozen ``tools+system+stable`` prefix is
        cached across runs. ``mark_indices`` names further messages to mark
        (the stable head and the end of a replayed run prefix); at most the
        first and last of them are used, keeping within Anthropic's four
        breakpoints.

        Returns ``(system, messages)`` where ``system`` is a block list with
        cache_control (or ``NOT_GIVEN``) and ``messages`` is a fresh list with
        only the marked messages deep-copied. The caller's input is never mutated.
        """
        marker = self._cache_control_marker()

        if system_prompt:
            system_blocks: Any = [{"type": "text", "text": system_prompt, "cache_control": marker}]
        else:
            system_blocks = anthropic.NOT_GIVEN

        if not api_messages:
            return system_blocks, api_messages

        new_messages = list(api_messages)
        indices = {len(new_messages) - 1}
        if mark_first:
            indices.add(0)
        if mark_indices:
            indices.update({min(mark_indices), max(mark_indices)})
        for i in indices:
            marked = copy.deepcopy(new_messages[i])
            self._mark_last_block(marked, marker)
            new_messages[i] = marked
        return system_blocks, new_messages

    @staticmethod
    def _mark_last_block(message: dict[str, Any], marker: dict[str, Any]) -> None:
        content = message["content"]
        if isinstance(content, str):
            message["content"] = [{"type": "text", "text": content, "cache_control": marker}]
        elif isinstance(content, list):
            # Thinking blocks are replayed unmodified, so the marker goes on
            # the last block that may carry one.
            for block in reversed(content):
                if block.get("type") not in _THINKING_BLOCK_TYPES:
                    block["cache_control"] = marker
                    return

    def _build_api_kwargs(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None,
        kwargs: dict[str, Any],
        *,
        registry: WireImageRegistry | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build Anthropic API kwargs from Dendrux messages and caller kwargs.

        Returns (api_kwargs, captured_request) where captured_request is the
        serializable subset for the evidence layer. On an image-bearing request
        every base64 payload in captured_request is replaced by
        ``"dendrux-image:sha256:<hex>"`` (``registry`` records the images).
        """
        if registry is None:
            registry = WireImageRegistry()
        # Model + thinking resolve first: they decide whether stored thinking
        # blocks may be replayed on assistant turns.
        model = kwargs.pop("model", self._model)
        thinking_on = kwargs.pop("thinking", self._thinking)
        stable_indices: list[int] = []
        system_prompt, api_messages = self._convert_messages(
            messages,
            replay_reasoning_for=model if thinking_on else None,
            stable_indices=stable_indices,
            registry=registry,
        )
        api_tools = self._convert_tools(tools, model=model) if tools else anthropic.NOT_GIVEN
        # Messages tagged placement="stable" (a stable context head, the end of
        # a replayed run prefix) get their own cross-run cache breakpoints.
        system_blocks, cached_messages = self._apply_cache_control(
            system_prompt, api_messages, mark_indices=stable_indices
        )

        api_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": kwargs.pop("max_tokens", self._max_tokens),
            "messages": cached_messages,
            "system": system_blocks,
            "tools": api_tools,
        }

        # Resolve thinking controls — per-call kwargs override constructor.
        effort = kwargs.pop("effort", self._effort)
        show_thinking = kwargs.pop("show_thinking", self._show_thinking)
        thinking_budget = kwargs.pop("thinking_budget", self._thinking_budget)
        if thinking_on:
            self._apply_thinking(api_kwargs, effort, show_thinking, thinking_budget)
            if self._replays_run_reasoning(messages, model):
                api_kwargs["thinking"]["block_binding"] = {"prefix_mismatch_behavior": "drop_block"}
                api_kwargs["extra_headers"] = {"anthropic-beta": _THINKING_BINDING_BETA}

        # Temperature is incompatible with thinking — only send it when off.
        if thinking_on:
            kwargs.pop("temperature", None)
        elif "temperature" in kwargs:
            api_kwargs["temperature"] = kwargs.pop("temperature")
        elif self._temperature is not None:
            api_kwargs["temperature"] = self._temperature

        for key in _SUPPORTED_KWARGS - {"model", "max_tokens", "temperature"}:
            if key in kwargs:
                api_kwargs[key] = kwargs.pop(key)

        captured_request = {k: v for k, v in api_kwargs.items() if v is not anthropic.NOT_GIVEN}
        if registry.has_images:
            captured_request = registry.sanitize(captured_request)
        return api_kwargs, captured_request

    @staticmethod
    def _replays_run_reasoning(messages: list[Message], model: str) -> bool:
        """True when thinking blocks replayed from an earlier run go on the wire."""
        return any(
            m.meta.get(REPLAYED_META_KEY)
            and replayable_reasoning_blocks(m, model=model, block_types=_THINKING_BLOCK_TYPES)
            for m in messages
        )

    def _apply_thinking(
        self,
        api_kwargs: dict[str, Any],
        effort: str | None,
        show_thinking: bool,
        thinking_budget: int | None,
    ) -> None:
        """Build the ``thinking`` + ``output_config`` API params.

        Adaptive mode by default (the only mode on current 4.6–4.8 / Fable /
        Mythos models); a ``thinking_budget`` switches to legacy manual mode
        for older models or fixed-cost workloads.
        """
        display = "summarized" if show_thinking else "omitted"
        if thinking_budget is not None:
            api_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
                "display": display,
            }
        else:
            api_kwargs["thinking"] = {"type": "adaptive", "display": display}
        if effort is not None:
            api_kwargs["output_config"] = {"effort": _normalize_effort(effort)}

    @logs_provider_errors
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        *,
        output_schema: dict[str, Any] | None = None,
        run_id: str | None = None,
        cache_key_prefix: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send messages to Claude and return a normalized response.

        kwargs override constructor defaults (e.g. model, max_tokens).
        Only Anthropic-supported kwargs are forwarded; unknown keys are ignored.

        ``run_id`` and ``cache_key_prefix`` are accepted for protocol
        compatibility with OpenAI providers but unused — Anthropic caching
        is keyed by byte-stable prefix and ``cache_control`` markers, not
        by an explicit cache key.

        When output_schema is provided without thinking, injects a synthetic
        tool with the schema and forces tool_choice. The tool is sent with
        ``strict: true`` (schema normalized for Anthropic strict mode) when
        the schema allows it, otherwise exactly as before, non-strict. The
        tool_use response is extracted and normalized into LLMResponse.text
        as a JSON string, with tool_calls set to None (preserving the
        SingleCall invariant). With thinking on (which rejects a forced
        tool_choice) the schema goes in ``output_config.format`` instead and
        the JSON answer comes back as text.

        If a request carrying strict entries gets a 400, it is retried once with
        every strict entry sent exactly as in 0.4.0; ``provider_request``
        records the request that succeeded. A second failure, any non-400
        error, and errors on requests without strict entries are raised
        as typed :class:`~dendrux.errors.ProviderError` subclasses (the SDK
        exception on ``__cause__``).

        The whole call, SDK retries included, is bounded by ``call_timeout``.
        A refusal (``stop_reason="refusal"``) raises
        :class:`~dendrux.errors.ModelRefusalError`.
        """
        registry = WireImageRegistry()
        api_kwargs, captured_request = self._build_api_kwargs(
            messages, tools, kwargs, registry=registry
        )
        model = api_kwargs["model"]

        structured: Literal["tool", "format"] | None = None
        if output_schema is not None and "thinking" in api_kwargs:
            self._apply_output_format(api_kwargs, captured_request, output_schema)
            structured = "format"
        elif output_schema is not None:
            # Structured output: inject synthetic tool + forced tool_choice
            api_kwargs["tools"] = [self._structured_output_tool(output_schema, model=model)]
            api_kwargs["tool_choice"] = {"type": "tool", "name": _STRUCTURED_TOOL_NAME}
            # Update captured_request so evidence reflects the actual API call
            captured_request["tools"] = api_kwargs["tools"]
            captured_request["tool_choice"] = api_kwargs["tool_choice"]
            structured = "tool"

        legacy = (
            [self._structured_output_payload(output_schema)]
            if structured == "tool" and output_schema is not None
            else [self._legacy_tool(td) for td in tools or []]
        )
        request = WireRequest(api_kwargs, captured_request, registry, legacy)
        del api_kwargs, captured_request, legacy
        client = self._sdk()

        # Image-bearing requests never chain SDK errors (they carry the request
        # body): the typed error is built in the handler, raised after it.
        failure: Exception | None = None
        try:
            response = await call_with_timeout(
                functools.partial(self._create_with_strict_fallback, client, request),
                deadlines=self._deadlines,
                provider=_PROVIDER,
                model=model,
            )
        except ProviderError:
            raise
        except Exception as exc:
            mapped = self._map_failure(exc, request.registry, model, streaming=False)
            if mapped is None:
                raise
            if not request.registry.has_images:
                raise mapped from exc
            failure = mapped
        if failure is not None:
            # Frame locals reach tracebacks (and error trackers): drop the
            # request and registry, which hold the base64 payloads.
            del request, registry
            raise failure

        self._raise_on_refusal(response, model)
        if structured == "tool":
            llm_response = self._normalize_structured_response(response)
        else:
            llm_response = self._normalize_response(response)

        # Attach adapter-boundary payloads for evidence layer
        llm_response.provider_request = request.captured_request
        llm_response.model = model
        llm_response.provider_response = response.model_dump()

        return llm_response

    @logs_provider_stream_errors
    async def complete_stream(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        *,
        output_schema: dict[str, Any] | None = None,
        run_id: str | None = None,
        cache_key_prefix: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream LLM response as events, token by token.

        Text deltas are yielded immediately as they arrive.
        Tool call arguments are accumulated internally and yielded as
        complete TOOL_USE_END events when each content block finishes.

        ``run_id`` / ``cache_key_prefix`` accepted for protocol parity but
        unused (Anthropic caching is byte-based via ``cache_control``).

        At stream end, yields a DONE event carrying the full LLMResponse
        (with usage stats and provider payloads) for the loop to consume.

        A 400 when the stream opens (before any event) on a request carrying
        strict entries is retried once without strict, as in ``complete``.

        A stream that ends without its terminal event raises
        :class:`~dendrux.errors.ProviderStreamIncompleteError`; it is retried
        once only when nothing has been yielded yet. ``stream_idle_timeout``
        bounds the gap between events (``ping`` does not count) and
        ``stream_total_timeout`` the whole stream. Closing the generator
        closes the upstream response.
        """
        registry = WireImageRegistry()
        api_kwargs, captured_request = self._build_api_kwargs(
            messages, tools, kwargs, registry=registry
        )
        request = WireRequest(
            api_kwargs, captured_request, registry, [self._legacy_tool(td) for td in tools or []]
        )
        model = request.model
        del api_kwargs, captured_request, registry
        client = self._sdk()
        clock = StreamClock(self._deadlines, provider=_PROVIDER, model=model)
        # Closing this generator (aclose, cancel) closes the upstream response.
        async with contextlib.aclosing(
            retry_unstarted_stream(
                functools.partial(self._stream_attempt, client, clock, request),
                provider=_PROVIDER,
                model=model,
                run_id=run_id,
            )
        ) as events:
            async for event in events:
                yield event

    async def _stream_attempt(
        self,
        client: anthropic.AsyncAnthropic,
        clock: StreamClock,
        request: WireRequest,
    ) -> AsyncGenerator[StreamEvent, None]:
        """One streaming request: open, relay events, require the terminal event."""
        model = request.model
        # Accumulators for building the final LLMResponse
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        # Thinking-token count arrives on the message_delta event, not on the
        # streamed final message's usage — capture it here.
        stream_reasoning_tokens: int | None = None
        # Fallback when the final message carries no stop_reason.
        stream_stop_reason: str | None = None
        saw_stop = False

        # Per-block state for tool call assembly
        _current_tool_name: str | None = None
        _current_tool_id: str | None = None
        _current_tool_json_parts: list[str] = []

        failure: Exception | None = None
        final_message: Any = None
        try:
            async with contextlib.AsyncExitStack() as stack:
                stream = await clock.open(
                    functools.partial(self._enter_stream, client, stack, request)
                )
                iterator = stream.__aiter__()
                while True:
                    try:
                        event = await clock.step(iterator.__anext__())
                    except StopAsyncIteration:
                        break
                    if event.type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            _current_tool_name = block.name
                            _current_tool_id = block.id
                            _current_tool_json_parts = []
                            yield StreamEvent(
                                type=StreamEventType.TOOL_USE_START,
                                tool_name=block.name,
                                tool_call_id=block.id,
                            )

                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            text_parts.append(delta.text)
                            yield StreamEvent(
                                type=StreamEventType.TEXT_DELTA,
                                text=delta.text,
                            )
                        elif delta.type == "thinking_delta":
                            yield StreamEvent(
                                type=StreamEventType.REASONING_DELTA,
                                text=delta.thinking,
                            )
                        elif delta.type == "signature_delta":
                            # Carried on the final message's blocks; nothing to
                            # surface live.
                            pass
                        elif delta.type == "input_json_delta":
                            _current_tool_json_parts.append(delta.partial_json)

                    elif event.type == "content_block_stop":
                        if _current_tool_name is not None:
                            raw_json = "".join(_current_tool_json_parts)
                            tc = build_tool_call(
                                _current_tool_name,
                                raw_json,
                                provider_tool_call_id=_current_tool_id,
                                provider="anthropic",
                                model=self._model,
                            )
                            tool_calls.append(tc)
                            yield StreamEvent(
                                type=StreamEventType.TOOL_USE_END,
                                tool_call=tc,
                                tool_name=tc.name,
                                tool_call_id=_current_tool_id,
                            )
                            _current_tool_name = None
                            _current_tool_id = None
                            _current_tool_json_parts = []

                    elif event.type == "message_delta":
                        delta_stop = getattr(getattr(event, "delta", None), "stop_reason", None)
                        if delta_stop:
                            stream_stop_reason = delta_stop
                        details = getattr(
                            getattr(event, "usage", None), "output_tokens_details", None
                        )
                        captured = _reasoning_tokens_from_details(details)
                        if captured is not None:
                            stream_reasoning_tokens = captured

                    elif event.type == "message_stop":
                        saw_stop = True

                    # The SDK drops ``ping``: every event here is semantic.
                    clock.mark_semantic()

                # The final message carries usage and the provider response.
                final_message = await self._final_message(clock, stream)
        except ProviderError:
            raise
        except Exception as exc:
            mapped = self._map_failure(exc, request.registry, model, streaming=True)
            if mapped is None:
                raise
            if not request.registry.has_images:
                raise mapped from exc
            failure = mapped
        if failure is not None:
            # Frame locals reach tracebacks (and error trackers): the request
            # (with its base64 payloads) is never bound here.
            raise failure

        final_stop = getattr(final_message, "stop_reason", None) if final_message else None
        if final_message is None or not (saw_stop or stream_stop_reason or final_stop):
            raise stream_incomplete(_PROVIDER, model)
        stop_reason_raw = final_stop or stream_stop_reason
        if stop_reason_raw == "refusal":
            raise refusal_error(_PROVIDER, model, refusal_text="".join(text_parts))

        usage = self._usage_from_response(final_message.usage)
        if usage.reasoning_tokens is None and stream_reasoning_tokens is not None:
            usage.reasoning_tokens = stream_reasoning_tokens
        reasoning, reasoning_blocks = self._extract_reasoning(final_message.content)
        stop_reason, raw_stop_reason = anthropic_stop_reason(stop_reason_raw)

        llm_response = LLMResponse(
            text="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls if tool_calls else None,
            raw=final_message,
            usage=usage,
            reasoning=reasoning,
            reasoning_blocks=reasoning_blocks,
            stop_reason=stop_reason,
            raw_stop_reason=raw_stop_reason,
        )
        llm_response.provider_request = request.captured_request
        llm_response.model = model
        llm_response.provider_response = final_message.model_dump()

        yield StreamEvent(type=StreamEventType.DONE, raw=llm_response)

    @staticmethod
    async def _final_message(clock: StreamClock, stream: Any) -> Any:
        """The accumulated message, or None when the stream carried no message."""
        try:
            return await clock.step(stream.get_final_message())
        except AssertionError:
            # The SDK asserts a snapshot exists; none means no message_start.
            return None

    def _map_failure(
        self, exc: BaseException, registry: WireImageRegistry, model: str, *, streaming: bool
    ) -> Exception | None:
        """Typed error for a failed call, or None to re-raise ``exc`` unchanged."""
        if (
            registry.has_images
            and isinstance(exc, anthropic.APIError)
            and not isinstance(exc, anthropic.APIConnectionError)
        ):
            return self._image_error(exc, registry, model, streaming=streaming)
        return map_sdk_error(
            exc, sdk=anthropic, provider=_PROVIDER, model=model, streaming=streaming
        )

    @staticmethod
    def _raise_on_refusal(response: Any, model: str) -> None:
        if getattr(response, "stop_reason", None) != "refusal":
            return
        text = "".join(
            getattr(block, "text", "") or ""
            for block in getattr(response, "content", None) or []
            if getattr(block, "type", None) == "text"
        )
        raise refusal_error(_PROVIDER, model, refusal_text=text)

    def _apply_output_format(
        self,
        api_kwargs: dict[str, Any],
        captured_request: dict[str, Any],
        output_schema: dict[str, Any],
    ) -> None:
        """Send the schema as ``output_config.format`` (merged with any effort).

        Used when thinking is on, which rejects the forced tool_choice of the
        synthetic-tool path. The schema is adapted to Anthropic's strict
        subset when it allows; otherwise it is sent as given.
        """
        schema, problem = prepare_anthropic_strict(output_schema)
        if problem is not None:
            logger.debug("Anthropic output_config.format sent with the given schema: %s", problem)
            schema = output_schema
        output_config = dict(api_kwargs.get("output_config") or {})
        output_config["format"] = {"type": "json_schema", "schema": schema}
        api_kwargs["output_config"] = output_config
        captured_request["output_config"] = output_config

    # ------------------------------------------------------------------
    # Outbound conversions: Dendrux → Anthropic
    # ------------------------------------------------------------------

    def _convert_messages(
        self,
        messages: list[Message],
        *,
        replay_reasoning_for: str | None = None,
        stable_indices: list[int] | None = None,
        registry: WireImageRegistry | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Convert Dendrux messages to Anthropic format.

        Returns (system_prompt, api_messages) where:
        - SYSTEM messages are extracted and joined into a single string
        - ASSISTANT messages with tool_calls become content block arrays
        - TOOL messages with native-tool correlation (call_id → provider_tool_call_id)
          become tool_result blocks in user messages
        - TOOL messages without native correlation become plain user messages
          (strategy is responsible for formatting these as text observations)
        - USER/ASSISTANT (text-only) pass through normally
        - ASSISTANT ``reasoning_blocks`` (thinking/redacted_thinking) lead the
          content array, unmodified, when ``replay_reasoning_for`` names the
          request model and it matches the model that produced them

        When ``stable_indices`` is given, the API-message index of every
        ``placement="stable"`` message is appended to it.

        Messages with image ``parts`` become content-block lists in part order
        (text blocks + base64 ``image`` blocks); a TOOL message's parts go inside
        its ``tool_result`` block. Each image's wire path is recorded in
        ``registry``. Text-only messages take the unchanged v0.3.0 path.
        """
        if registry is None:
            registry = WireImageRegistry()
        system_parts: list[str] = []
        api_messages: list[dict[str, Any]] = []
        call_index = build_call_index(messages)

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_parts.append(msg.content)

            elif msg.role == Role.USER:
                if msg.parts is None:
                    api_messages.append({"role": "user", "content": msg.content})
                else:
                    api_messages.append(self._user_parts_message(msg, api_messages, registry))

            elif msg.role == Role.ASSISTANT:
                thinking = replayable_reasoning_blocks(
                    msg, model=replay_reasoning_for, block_types=_THINKING_BLOCK_TYPES
                )
                if msg.tool_calls:
                    # Assistant with tool calls → content blocks
                    content: list[dict[str, Any]] = list(thinking)
                    if msg.content:
                        content.append({"type": "text", "text": msg.content})
                    for tc in msg.tool_calls:
                        content.append(
                            {
                                "type": "tool_use",
                                "id": tc.provider_tool_call_id or tc.id,
                                "name": tc.name,
                                "input": tc.params,
                            }
                        )
                    api_messages.append({"role": "assistant", "content": content})
                elif thinking:
                    if msg.content:
                        thinking.append({"type": "text", "text": msg.content})
                    api_messages.append({"role": "assistant", "content": thinking})
                else:
                    api_messages.append({"role": "assistant", "content": msg.content})

            elif msg.role == Role.TOOL:
                original_call = resolve_tool_message_call(msg, call_index)

                if original_call.provider_tool_call_id:
                    if msg.parts is None:
                        # Native tool correlation path — emit tool_result block
                        self._append_tool_result(
                            api_messages,
                            tool_use_id=original_call.provider_tool_call_id,
                            content=msg.content,
                            is_error=bool(msg.meta.get("is_error")),
                        )
                    else:
                        # Native tool-result images: text + image blocks nested
                        # in the tool_result, in part order.
                        i, j = self._tool_result_position(api_messages)
                        blocks = convert_parts(
                            msg.parts,
                            registry=registry,
                            prefix=("messages", i, "content", j, "content"),
                            text_block=anthropic_text,
                            image_block=anthropic_image,
                        )
                        self._append_tool_result(
                            api_messages,
                            tool_use_id=original_call.provider_tool_call_id,
                            content=blocks,
                            is_error=bool(msg.meta.get("is_error")),
                        )
                elif msg.parts is None:
                    # No native correlation (e.g. PromptBasedReAct via
                    # NativeToolCalling — shouldn't happen, but be safe).
                    # Emit as plain user message so the strategy's text
                    # formatting is preserved.
                    api_messages.append({"role": "user", "content": msg.content})
                else:
                    api_messages.append(self._user_parts_message(msg, api_messages, registry))

            if stable_indices is not None and msg.placement == "stable" and api_messages:
                stable_indices.append(len(api_messages) - 1)

        system_prompt = "\n\n".join(system_parts)
        return system_prompt, api_messages

    @staticmethod
    def _user_parts_message(
        msg: Message, api_messages: list[dict[str, Any]], registry: WireImageRegistry
    ) -> dict[str, Any]:
        """A user message whose content is the parts as blocks, in caller order."""
        assert msg.parts is not None
        content = convert_parts(
            msg.parts,
            registry=registry,
            prefix=("messages", len(api_messages), "content"),
            text_block=anthropic_text,
            image_block=anthropic_image,
        )
        return {"role": "user", "content": content}

    @staticmethod
    def _tool_result_position(api_messages: list[dict[str, Any]]) -> tuple[int, int]:
        """``(message index, block index)`` the next tool_result block will take.

        Mirrors the merge rule of :meth:`_append_tool_result`.
        """
        if (
            api_messages
            and api_messages[-1]["role"] == "user"
            and isinstance(api_messages[-1]["content"], list)
            and api_messages[-1]["content"]
            and api_messages[-1]["content"][0].get("type") == "tool_result"
        ):
            return len(api_messages) - 1, len(api_messages[-1]["content"])
        return len(api_messages), 0

    def _image_error(
        self,
        exc: anthropic.APIError,
        registry: WireImageRegistry,
        model: str,
        *,
        streaming: bool,
    ) -> Exception:
        """Typed, sanitised error for a failed image-bearing request (not raised)."""
        return sdk_image_request_error(
            exc,
            provider=self,
            provider_name="anthropic",
            model=model,
            registry=registry,
            streaming=streaming,
        )

    def _append_tool_result(
        self,
        api_messages: list[dict[str, Any]],
        *,
        tool_use_id: str,
        content: str | list[dict[str, Any]],
        is_error: bool,
    ) -> None:
        """Append a tool_result block, merging with the previous user message if consecutive."""
        tool_result_block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }
        if is_error:
            tool_result_block["is_error"] = True

        # Merge consecutive tool results into a single user message
        if (
            api_messages
            and api_messages[-1]["role"] == "user"
            and isinstance(api_messages[-1]["content"], list)
            and api_messages[-1]["content"]
            and api_messages[-1]["content"][0].get("type") == "tool_result"
        ):
            api_messages[-1]["content"].append(tool_result_block)
        else:
            api_messages.append({"role": "user", "content": [tool_result_block]})

    def _convert_tools(
        self, tools: list[ToolDef], *, model: str | None = None
    ) -> list[dict[str, Any]]:
        """Convert Dendrux ToolDefs to Anthropic tool format.

        Non-strict tools are sent exactly as stored (byte-stable for prompt
        caching). A ``strict`` tool is sent with ``"strict": true`` and a
        copy of its schema normalized for Anthropic strict mode, unless the
        schema cannot be strict, the request-wide limits (strict tools,
        optional and union-typed parameters, counted in tool order) are
        spent, or Anthropic already rejected it for ``model`` in this
        process; then it is sent non-strict (WARNING once per tool name,
        DEBUG after). With ``strict=False`` on the provider every tool is
        sent non-strict.
        """
        model = model or self._model
        converted: list[dict[str, Any]] = []
        strict_tools = optional_total = union_total = 0
        for td in tools:
            payload = self._legacy_tool(td)
            if td.strict and self._strict:
                schema, problem = prepare_anthropic_strict(td.parameters)
                candidate = {**payload, "input_schema": schema, "strict": True}
                optional = unions = 0
                if problem is None and _is_strict_rejected(model, candidate):
                    logger.debug(
                        "Anthropic tool %r sent without strict: rejected by Anthropic "
                        "earlier in this process",
                        td.name,
                    )
                    converted.append(payload)
                    continue
                if problem is None:
                    optional, unions = anthropic_strict_budget(schema)
                    if strict_tools + 1 > ANTHROPIC_STRICT_MAX_TOOLS:
                        problem = f"over {ANTHROPIC_STRICT_MAX_TOOLS} strict tools in one request"
                    elif optional_total + optional > ANTHROPIC_STRICT_MAX_OPTIONAL:
                        problem = (
                            f"over {ANTHROPIC_STRICT_MAX_OPTIONAL} optional parameters "
                            f"across strict tools in one request"
                        )
                    elif union_total + unions > ANTHROPIC_STRICT_MAX_UNIONS:
                        problem = (
                            f"over {ANTHROPIC_STRICT_MAX_UNIONS} union-typed parameters "
                            f"across strict tools in one request"
                        )
                if problem is None:
                    strict_tools += 1
                    optional_total += optional
                    union_total += unions
                    payload = candidate
                else:
                    _warn_strict_once(
                        f"downgrade:{td.name}",
                        "Anthropic tool %r sent without strict: %s",
                        td.name,
                        problem,
                    )
            converted.append(payload)
        return converted

    @staticmethod
    def _legacy_tool(td: ToolDef) -> dict[str, Any]:
        """The non-strict tool payload (unchanged since 0.4.0)."""
        return {"name": td.name, "description": td.description, "input_schema": td.parameters}

    @staticmethod
    def _structured_output_payload(output_schema: dict[str, Any]) -> dict[str, Any]:
        """The non-strict structured-output tool payload (unchanged since 0.4.0)."""
        return {
            "name": _STRUCTURED_TOOL_NAME,
            "description": _STRUCTURED_TOOL_DESCRIPTION,
            "input_schema": output_schema,
        }

    def _structured_output_tool(
        self, output_schema: dict[str, Any], *, model: str
    ) -> dict[str, Any]:
        """The synthetic structured-output tool, strict when the schema allows it."""
        payload = self._structured_output_payload(output_schema)
        if not self._strict:
            return payload
        schema, problem = prepare_anthropic_strict(output_schema)
        candidate = {**payload, "input_schema": schema, "strict": True}
        if problem is None and _is_strict_rejected(model, candidate):
            problem = "rejected by Anthropic earlier in this process"
        if problem is not None:
            logger.debug("Anthropic structured output sent without strict: %s", problem)
            return payload
        return candidate

    @staticmethod
    def _revert_strict(
        api_kwargs: dict[str, Any],
        captured_request: dict[str, Any],
        legacy: list[dict[str, Any]],
        message: str,
        *,
        points_at_strict: bool,
    ) -> list[str]:
        """Swap every strict entry for its non-strict payload; return their labels.

        When the 400 points at strict, the entries are remembered for the
        request model and a WARNING is logged once per label.
        """
        reverted: list[dict[str, Any]] = []
        labels: list[str] = []
        for entry, original in zip(api_kwargs["tools"], legacy, strict=True):
            if entry.get("strict") is not True:
                reverted.append(entry)
                continue
            label = entry["name"]
            if label == _STRUCTURED_TOOL_NAME:
                label = f"{label} (schema {_schema_digest(original['input_schema'])})"
            labels.append(label)
            if points_at_strict:
                _remember_strict_rejected(api_kwargs["model"], entry)
                _warn_strict_once(
                    f"rejected:{label}",
                    "Anthropic rejected strict mode for %s; retrying without strict "
                    "(and sending it non-strict for the rest of this process); error ref %s",
                    label,
                    _message_ref(message),
                )
            reverted.append(original)
        api_kwargs["tools"] = reverted
        captured_request["tools"] = reverted
        return labels

    @staticmethod
    def _warn_unmatched_fallback(labels: list[str], message: str) -> None:
        names = ", ".join(labels)
        ref = _message_ref(message)
        _warn_strict_once(
            f"unmatched:{names}:{ref}",
            "Anthropic returned a 400 for a request with strict entries (%s); it succeeded "
            "without strict, so strict stays on for later requests; error ref %s",
            names,
            ref,
        )

    async def _create_with_strict_fallback(
        self, client: anthropic.AsyncAnthropic, request: WireRequest
    ) -> Any:
        """``messages.create``, retried once without strict on a 400."""
        try:
            with call_attempt_tracking():
                return await client.messages.create(**request.api_kwargs)
        except anthropic.APIError as exc:
            failure = _strict_failure(exc, request.api_kwargs.get("tools"))
            if failure is None:
                raise
        # Retried outside the handler so a second failure never chains the first.
        message, points_at_strict = failure
        labels = self._revert_strict(
            request.api_kwargs,
            request.captured_request,
            request.legacy,
            message,
            points_at_strict=points_at_strict,
        )
        with call_attempt_tracking():
            response = await client.messages.create(**request.api_kwargs)
        if not points_at_strict:
            self._warn_unmatched_fallback(labels, message)
        return response

    async def _enter_stream(
        self,
        client: anthropic.AsyncAnthropic,
        stack: contextlib.AsyncExitStack,
        request: WireRequest,
    ) -> Any:
        """``messages.stream`` entered on ``stack``, retried once without strict on a 400.

        The request is made on enter, before any event exists.
        """
        with call_attempt_tracking():
            failure: tuple[str, bool] | None = None
            try:
                return await stack.enter_async_context(client.messages.stream(**request.api_kwargs))
            except anthropic.APIError as exc:
                failure = _strict_failure(exc, request.api_kwargs.get("tools"))
                if failure is None:
                    raise
            message, points_at_strict = failure
            labels = self._revert_strict(
                request.api_kwargs,
                request.captured_request,
                request.legacy,
                message,
                points_at_strict=points_at_strict,
            )
            stream = await stack.enter_async_context(client.messages.stream(**request.api_kwargs))
            if not points_at_strict:
                self._warn_unmatched_fallback(labels, message)
            return stream

    # ------------------------------------------------------------------
    # Inbound conversions: Anthropic → Dendrux
    # ------------------------------------------------------------------

    @staticmethod
    def _usage_from_response(usage: Any) -> UsageStats:
        """Build UsageStats from an Anthropic Usage object.

        ``reasoning_tokens`` is read from ``output_tokens_details.thinking_tokens``
        when the SDK reports it (None otherwise) — these are billed within
        ``output_tokens``.
        """
        details = getattr(usage, "output_tokens_details", None)
        reasoning_tokens = _reasoning_tokens_from_details(details)
        return UsageStats(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", None),
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", None),
            reasoning_tokens=reasoning_tokens,
        )

    @staticmethod
    def _extract_reasoning(content: list[Any]) -> tuple[str | None, list[Any] | None]:
        """Pull reasoning summary text + replay blocks from message content.

        Keeps the full thinking/redacted_thinking blocks (incl. signature) for
        multi-turn replay; surfaces the summary text when present (an
        ``omitted`` display yields blocks with empty thinking text).
        """
        reasoning_parts: list[str] = []
        reasoning_blocks: list[Any] = []
        for block in content:
            if block.type in ("thinking", "redacted_thinking"):
                reasoning_blocks.append(
                    block.model_dump() if hasattr(block, "model_dump") else block
                )
                thinking_text = getattr(block, "thinking", None)
                if thinking_text:
                    reasoning_parts.append(thinking_text)
        return (
            "".join(reasoning_parts) if reasoning_parts else None,
            reasoning_blocks if reasoning_blocks else None,
        )

    def _normalize_structured_response(self, response: anthropic.types.Message) -> LLMResponse:
        """Extract structured output from a forced tool_use response.

        The synthetic "structured_output" tool's input is the structured
        data. We serialize it to JSON and put it in LLMResponse.text,
        with tool_calls=None so SingleCall's invariant is preserved.

        The forced call ends with ``stop_reason="tool_use"``; it is reported
        as ``"end_turn"`` (the answer finished), with the vendor string kept
        in ``raw_stop_reason``.
        """
        structured_data: dict[str, Any] | None = None
        for block in response.content:
            if block.type == "tool_use" and block.name == "structured_output":
                structured_data = dict(block.input) if block.input else {}
                break

        if structured_data is None:
            raise RuntimeError(
                "Anthropic structured output: expected a tool_use block named "
                "'structured_output' but none was found in the response."
            )

        usage = self._usage_from_response(response.usage)
        stop_reason, raw_stop_reason = anthropic_stop_reason(getattr(response, "stop_reason", None))
        if stop_reason == "tool_use":
            # The forced structured_output call is the answer, not a tool request.
            stop_reason = "end_turn"

        return LLMResponse(
            text=json.dumps(structured_data),
            tool_calls=None,
            raw=response,
            usage=usage,
            stop_reason=stop_reason,
            raw_stop_reason=raw_stop_reason,
        )

    def _normalize_response(self, response: anthropic.types.Message) -> LLMResponse:
        """Convert Anthropic response to Dendrux LLMResponse.

        Text blocks are concatenated with no separator — each block is
        already self-contained. This avoids injecting content that doesn't
        exist in the provider output.
        """
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        name=block.name,
                        params=dict(block.input) if block.input else {},
                        provider_tool_call_id=block.id,
                    )
                )

        reasoning, reasoning_blocks = self._extract_reasoning(response.content)
        usage = self._usage_from_response(response.usage)
        stop_reason, raw_stop_reason = anthropic_stop_reason(getattr(response, "stop_reason", None))

        return LLMResponse(
            text="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls if tool_calls else None,
            raw=response,
            usage=usage,
            reasoning=reasoning,
            reasoning_blocks=reasoning_blocks,
            stop_reason=stop_reason,
            raw_stop_reason=raw_stop_reason,
        )
