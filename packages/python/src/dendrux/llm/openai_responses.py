"""OpenAI Responses API provider.

Thin adapter between Dendrux's universal types and the OpenAI Responses API.
Use this provider when you need OpenAI's built-in tools (web_search,
code_interpreter, file_search) alongside custom Dendrux tools.

For standard function calling without built-in tools, use OpenAIProvider
(Chat Completions) instead — it works with any OpenAI-compatible API.

No business logic, no agent loop awareness. Just shape translation.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
from typing import TYPE_CHECKING, Any, Literal

import openai

from dendrux.errors import ProviderError
from dendrux.llm._helpers import (
    build_call_index,
    build_tool_call,
    replayable_reasoning_blocks,
    resolve_tool_message_call,
)
from dendrux.llm._images import (
    WireImageRegistry,
    convert_parts,
    image_request_error,
    normalize_base_url,
    openai_image_support,
    portable_framing,
    portable_tool_text,
    responses_image,
    responses_text,
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
    close_quietly,
    content_filter_error,
    error_from_fields,
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
from dendrux.llm._stop_reason import responses_stop_reason
from dendrux.llm._usage import openai_wire_usage
from dendrux.llm.base import LLMProvider
from dendrux.types import (
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
    from dendrux.types import ImagePart, Message, ToolDef

logger = logging.getLogger(__name__)


_EFFORT_ALIASES = {"extra": "xhigh"}

_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1/"

_TOOL_RESULT_IMAGE_MODES = ("native", "portable")

_REASONING_ITEM_TYPES = frozenset({"reasoning"})

# Assistant message phases the Responses API accepts on input items.
_REPLAYABLE_PHASES = frozenset({"commentary", "final_answer"})


def _assistant_item(msg: Message) -> dict[str, Any]:
    """Build an assistant message input item, resending its recorded phase.

    OpenAI asks callers replaying history to preserve each assistant
    message's original ``phase``; dropping it can degrade later turns.
    """
    item: dict[str, Any] = {"role": "assistant", "content": msg.content}
    phase = msg.meta.get("phase")
    if phase in _REPLAYABLE_PHASES:
        item["phase"] = phase
    return item


def _normalize_effort(effort: str) -> str:
    """Map a Dendrux effort alias to OpenAI's vocabulary."""
    return _EFFORT_ALIASES.get(effort, effort)


def _build_usage_with_cache(usage: Any) -> UsageStats:
    """Build UsageStats from an OpenAI Responses API usage object.

    Reads ``input_tokens_details.cached_tokens`` (cache reads) and
    ``input_tokens_details.cache_write_tokens`` (cache writes; untyped in
    SDK 2.30, so read from ``model_extra`` too). ``input_tokens`` is
    fresh-only: the vendor's input count minus reads and writes, clamped at
    zero. An omitted field stays ``None``.
    """
    return openai_wire_usage(
        usage,
        input_field="input_tokens",
        output_field="output_tokens",
        details_field="input_tokens_details",
        output_details_field="output_tokens_details",
    )


# Stream lifecycle events that do not show the model producing anything;
# they never reset the idle deadline.
_NON_SEMANTIC_EVENTS = frozenset({"response.created", "response.in_progress", "response.queued"})
_PROVIDER = "openai_responses"


def _field(obj: Any, name: str) -> str | None:
    value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
    return value if isinstance(value, str) and value else None


def _refusal_text(response: Any) -> str | None:
    """Joined ``refusal`` content of a response's message items, or None."""
    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", None) or []:
            if getattr(content, "type", None) == "refusal":
                parts.append(getattr(content, "refusal", "") or "")
    return "".join(parts) if parts else None


# Responses-API kwargs that complete() will forward.
_SUPPORTED_KWARGS = frozenset(
    {
        "temperature",
        "top_p",
        "max_output_tokens",
        "model",
        "tool_choice",
    }
)

# Built-in tool types supported via builtin_tools parameter.
# Only web_search_preview works with a bare {"type": "..."} dict.
# code_interpreter and file_search require additional config (container,
# vector_store_ids) — pass them directly via complete() kwargs when needed.
_BUILTIN_TOOL_TYPES = frozenset({"web_search_preview"})


class OpenAIResponsesProvider(LLMProvider):
    """OpenAI Responses API provider.

    Use this when you need OpenAI's built-in tools (web search)
    alongside custom Dendrux function tools.

    For standard function calling without built-in tools, use
    OpenAIProvider (Chat Completions) instead.

    Reasoning models with tool calling:
        Reasoning items returned alongside function calls are stored on the
        assistant message and replayed before its function_call items on the
        next request, but only to the same model that produced them.

    Usage:
        # With built-in web search + custom tools
        provider = OpenAIResponsesProvider(
            model="gpt-4o",
            builtin_tools=["web_search_preview"],
        )

        # With reasoning effort (o-series / GPT-5)
        provider = OpenAIResponsesProvider(
            model="gpt-5",
            reasoning_effort="medium",
        )
    """

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
        builtin_tools: list[str] | None = None,
        max_output_tokens: int = 16_000,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        thinking: bool = False,
        effort: str | None = None,
        show_thinking: bool = True,
        timeout: Any = TIMEOUT_UNSET,
        max_retries: int = 3,
        prompt_cache_retention: Literal["in-memory", "24h"] | None = None,
        cache_key_prefix: str | None = None,
        tool_result_images: Literal["native", "portable"] = "native",
        call_timeout: float | None = DEFAULT_CALL_TIMEOUT,
        stream_idle_timeout: float | None = DEFAULT_STREAM_IDLE_TIMEOUT,
        stream_total_timeout: float | None = None,
        max_stream_bytes: int = 16 * 1024 * 1024,
        max_stream_tool_calls: int = 1000,
    ) -> None:
        """Create an OpenAI Responses API provider.

        Args:
            model: Model identifier (e.g. "gpt-4o", "gpt-5").
            api_key: API key. Defaults to OPENAI_API_KEY env var.
            builtin_tools: OpenAI built-in tools to enable (e.g. "web_search_preview").
            max_output_tokens: Maximum output tokens per call. Override per-call.
            temperature: Sampling temperature. None = model default.
            reasoning_effort: Reasoning depth ("low", "medium", "high").
            thinking: Enable reasoning + surface its summary. Default False.
                When on (or when effort is set), the ``reasoning`` param is sent.
            effort: Cross-vendor reasoning effort (low|medium|high|xhigh;
                "extra"→xhigh). Overrides ``reasoning_effort``.
            show_thinking: When reasoning is on, request a summary
                (reasoning.summary="auto") so reasoning text is returned.
                Default True. Sent only with the reasoning param, so
                non-reasoning models stay unaffected unless you opt in.
            timeout: Deprecated shorthand: sets both ``call_timeout`` and
                ``stream_idle_timeout`` (with a ``DeprecationWarning``).
            max_retries: Number of automatic SDK retries on transient errors
                (connection resets, 408/409/429/5xx). They all count against
                ``call_timeout``.
            prompt_cache_retention: Responses API prompt-cache retention.
                ``None`` (default) omits the field so OpenAI uses its in-memory
                default. Set ``"24h"`` for long-running workflows on supported
                models. Hyphenated literal per the SDK.
            cache_key_prefix: Stable cache-routing identifier, sent as
                ``prompt_cache_key="dendrux:{cache_key_prefix}"``. OpenAI
                recommends one key per user/customer (not per request); on
                GPT-5.6+ the key is unnecessary. Takes precedence over the
                per-call prefix the agent loop derives (``"{agent}:{model}"``).
                ``None`` (default) leaves routing to the per-call prefix, and
                sends no key when there is none.
            tool_result_images: How images returned by tools reach the model.
                ``"native"`` (default) puts them inside ``function_call_output``
                as ``input_image`` items; ``"portable"`` sends a text tool output
                plus one framed user message after the tool outputs (the Chat
                Completions mapping).
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
                stream events (``response.in_progress`` does not count), and
                before the first one. ``None`` turns it off.
            max_stream_bytes: Cumulative UTF-8 and structural stream budget.
                Default 16 MiB. Exceeding it closes the upstream response.
            max_stream_tool_calls: Maximum retained tool-call slots (default 1,000).
            stream_total_timeout: Wall-clock limit in seconds for a whole
                stream. ``None`` (default) means no limit.
        """
        if tool_result_images not in _TOOL_RESULT_IMAGE_MODES:
            raise ValueError(
                f"tool_result_images must be 'native' or 'portable', got {tool_result_images!r}"
            )
        for t in builtin_tools or []:
            if t not in _BUILTIN_TOOL_TYPES:
                raise ValueError(
                    f"Unknown built-in tool '{t}'. "
                    f"Supported: {', '.join(sorted(_BUILTIN_TOOL_TYPES))}"
                )

        if api_key is None and not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OpenAIResponsesProvider needs an API key. "
                "Pass api_key='sk-...' or set the OPENAI_API_KEY environment variable."
            )

        self._deadlines = resolve_deadlines(
            "OpenAIResponsesProvider",
            timeout=timeout,
            call_timeout=call_timeout,
            stream_idle_timeout=stream_idle_timeout,
            stream_total_timeout=stream_total_timeout,
            max_stream_bytes=max_stream_bytes,
            max_stream_tool_calls=max_stream_tool_calls,
        )
        deadlines = self._deadlines

        def new_http() -> httpx.AsyncClient:
            return make_http_client(openai, deadlines)

        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            http_client=new_http(),
            max_retries=max_retries,
        )
        self._clients: LoopClients[openai.AsyncOpenAI] = LoopClients(
            new_http, lambda client, http: client.copy(http_client=http)
        )
        self._model = model
        self._builtin_tools = builtin_tools or []
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
        self._thinking = thinking
        self._effort = effort
        self._show_thinking = show_thinking
        self._prompt_cache_retention = prompt_cache_retention
        self._cache_key_prefix = cache_key_prefix
        self.tool_result_images = tool_result_images

    @property
    def model(self) -> str:
        """The model identifier this provider is configured to use."""
        return self._model

    def __repr__(self) -> str:
        parts = [f"model={self._model!r}"]
        if self._builtin_tools:
            parts.append(f"builtin_tools={self._builtin_tools!r}")
        return f"OpenAIResponsesProvider({', '.join(parts)})"

    async def close(self) -> None:
        """Close the underlying HTTP client and release connections."""
        # Close only what belongs to the running loop: after a loop change the
        # home transport is first rebuilt, never awaited across loops.
        current = self._clients.resolve(self._client)
        self._clients.pop_current_loop_extra()
        await current.close()

    def _sdk(self) -> openai.AsyncOpenAI:
        """The SDK client for the running event loop (rebuilt after a loop change)."""
        return self._clients.resolve(self._client)

    def _map_failure(
        self, exc: BaseException, registry: WireImageRegistry, model: str, *, streaming: bool
    ) -> Exception | None:
        """Typed error for a failed call, or None to re-raise ``exc`` unchanged."""
        if (
            registry.has_images
            and isinstance(exc, openai.APIError)
            and not isinstance(exc, openai.APIConnectionError)
        ):
            return self._image_error(exc, registry, model, streaming=streaming)
        return map_sdk_error(
            exc,
            sdk=openai,
            provider=_PROVIDER,
            model=model,
            streaming=streaming,
            endpoint=str(self._client.base_url),
        )

    @staticmethod
    def _event_error(error: Any, model: str) -> Exception:
        """Typed error for an ``error`` / ``response.failed`` payload (text-only)."""
        error_type = _field(error, "type")
        if error_type == "error":
            error_type = None
        return error_from_fields(
            provider=_PROVIDER,
            model=model,
            status_code=None,
            code=_field(error, "code"),
            error_type=error_type,
            param=_field(error, "param"),
            vendor_text=_field(error, "message") or "",
        )

    def _check_response(self, response: Any, model: str) -> None:
        """Raise for a failed, cancelled, filtered or refused response."""
        status = getattr(response, "status", None)
        if status == "failed":
            raise self._event_error(getattr(response, "error", None), model)
        if status == "cancelled":
            raise ProviderError(
                provider=_PROVIDER,
                model=model,
                category="unknown",
                message="The provider cancelled the response.",
            )
        if status == "incomplete":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
            if reason == "content_filter":
                raise content_filter_error(_PROVIDER, model)
        refusal = _refusal_text(response)
        if refusal is not None:
            raise refusal_error(_PROVIDER, model, refusal_text=refusal)

    # ------------------------------------------------------------------
    # Image input (0.4.0)
    # ------------------------------------------------------------------

    async def image_input_support(self, model: str) -> ImageSupport:
        """Static table on the official OpenAI endpoint; ``"unknown"`` elsewhere."""
        if str(self._client.base_url) != _OPENAI_DEFAULT_BASE_URL:
            return "unknown"
        return openai_image_support(model)

    @property
    def thinking_enabled(self) -> bool:
        """True when the provider was configured with ``thinking=True``."""
        return bool(self._thinking)

    def image_capability_key(self, model: str) -> tuple[str, str, str]:
        return ("responses", normalize_base_url(self._client.base_url), model)

    def _image_error(
        self, exc: openai.APIError, registry: WireImageRegistry, model: str, *, streaming: bool
    ) -> Exception:
        """Typed, sanitised error for a failed image-bearing request (not raised)."""
        return sdk_image_request_error(
            exc,
            provider=self,
            provider_name="openai_responses",
            model=model,
            registry=registry,
            streaming=streaming,
        )

    def _stream_event_image_error(
        self, error: Any, registry: WireImageRegistry, model: str
    ) -> Exception:
        """Typed error for an in-stream ``error`` / ``response.failed`` event."""

        def field(name: str) -> str | None:
            value = error.get(name) if isinstance(error, dict) else getattr(error, name, None)
            return value if isinstance(value, str) and value else None

        # An ``error`` stream event's ``type`` is the event name, not an error type.
        error_type = field("type")
        if error_type == "error":
            error_type = None
        return image_request_error(
            provider=self,
            provider_name="openai_responses",
            model=model,
            registry=registry,
            status_code=None,
            code=field("code"),
            error_type=error_type,
            param=field("param"),
            message=field("message") or "",
            streaming=True,
        )

    async def __aenter__(self) -> OpenAIResponsesProvider:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    def _build_api_kwargs(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None,
        kwargs: dict[str, Any],
        *,
        run_id: str | None = None,
        cache_key_prefix: str | None = None,
        registry: WireImageRegistry | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build Responses API kwargs from Dendrux messages and caller kwargs.

        Returns (api_kwargs, captured_request) where captured_request is the
        serializable subset for the evidence layer. On an image-bearing request
        every data URL in captured_request is replaced by
        ``"dendrux-image:sha256:<hex>"`` (``registry`` records the images).

        Cache routing: ``prompt_cache_key`` is sent only when a prefix is
        known — the constructor ``cache_key_prefix`` first, then the per-call
        one (the agent loop passes ``"{agent_name}:{model}"``). ``run_id`` is
        accepted for interface compatibility and never used as a key.
        """
        if registry is None:
            registry = WireImageRegistry()
        model = kwargs.pop("model", self._model)
        instructions, input_items = self._convert_messages(
            messages, replay_reasoning_for=model, registry=registry
        )
        api_tools = self._build_tools(tools)

        max_output_tokens = kwargs.pop(
            "max_output_tokens",
            kwargs.pop("max_tokens", self._max_output_tokens),
        )

        api_kwargs: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "max_output_tokens": max_output_tokens,
        }

        # Only pass instructions if non-empty
        if instructions:
            api_kwargs["instructions"] = instructions

        # Only pass tools if non-empty
        if api_tools:
            api_kwargs["tools"] = api_tools

        # Apply constructor defaults for optional params (per-call kwargs override)
        for attr, key in [
            (self._temperature, "temperature"),
        ]:
            if key in kwargs:
                api_kwargs[key] = kwargs.pop(key)
            elif attr is not None:
                api_kwargs[key] = attr

        # Reasoning — Responses API uses a nested object. `effort` (cross-vendor)
        # overrides reasoning_effort; `show_thinking` requests a summary so the
        # reasoning text is returned. We send the reasoning param only when
        # reasoning is wanted (effort set or thinking on), so non-reasoning
        # models stay unaffected by default.
        thinking_on = kwargs.pop("thinking", self._thinking)
        show_thinking = kwargs.pop("show_thinking", self._show_thinking)
        effort = kwargs.pop("effort", self._effort)
        reasoning_effort = kwargs.pop("reasoning_effort", self._reasoning_effort)
        effective_effort = _normalize_effort(effort) if effort is not None else reasoning_effort
        if effective_effort is not None or thinking_on:
            reasoning: dict[str, Any] = {}
            if effective_effort is not None:
                reasoning["effort"] = effective_effort
            # Summary is requested only when thinking is explicitly opted into,
            # so legacy reasoning_effort callers keep their exact request shape.
            if thinking_on and show_thinking:
                reasoning["summary"] = "auto"
            if reasoning:
                api_kwargs["reasoning"] = reasoning

        # Forward remaining supported kwargs
        already_handled = {
            "model",
            "max_output_tokens",
            "max_tokens",
            "temperature",
        }
        for key in _SUPPORTED_KWARGS - already_handled:
            if key in kwargs:
                api_kwargs[key] = kwargs.pop(key)

        # Cache routing — never keyed on run_id: a per-run key scatters one
        # conversation's runs across cache lanes and defeats cross-run hits.
        cache_key_source = self._cache_key_prefix or cache_key_prefix
        if cache_key_source:
            api_kwargs["prompt_cache_key"] = f"dendrux:{cache_key_source}"
        if self._prompt_cache_retention is not None:
            api_kwargs["prompt_cache_retention"] = self._prompt_cache_retention

        captured_request = dict(api_kwargs)
        if registry.has_images:
            captured_request = registry.sanitize(captured_request)
        return api_kwargs, captured_request

    def _inject_structured_output(
        self,
        api_kwargs: dict[str, Any],
        captured_request: dict[str, Any],
        output_schema: dict[str, Any],
    ) -> None:
        """Inject text.format for structured output and update evidence."""
        from dendrux.llm._schema import normalize_for_openai_strict

        strict_schema = normalize_for_openai_strict(output_schema)
        schema_name = output_schema.get("title", "structured_output")
        text_format = {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": strict_schema,
            },
        }
        api_kwargs["text"] = text_format
        captured_request["text"] = text_format

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
        """Send messages to the model via the Responses API.

        kwargs override constructor defaults (e.g. model, max_output_tokens).
        Only supported kwargs are forwarded; unknown keys are ignored.

        ``cache_key_prefix`` routes requests to an OpenAI cache pool via
        ``prompt_cache_key`` (a constructor ``cache_key_prefix`` takes
        precedence). ``run_id`` is never used as a key.

        When output_schema is provided, sets text.format to json_schema
        with strict mode.

        The whole call, SDK retries included, is bounded by ``call_timeout``.
        A ``status="failed"`` reply raises a typed error carrying its code;
        ``cancelled`` raises; a content-filtered or refused answer raises
        :class:`~dendrux.errors.ContentFilteredError` /
        :class:`~dendrux.errors.ModelRefusalError`.
        """
        registry = WireImageRegistry()
        api_kwargs, captured_request = self._build_api_kwargs(
            messages,
            tools,
            kwargs,
            run_id=run_id,
            cache_key_prefix=cache_key_prefix,
            registry=registry,
        )

        if output_schema is not None:
            self._inject_structured_output(api_kwargs, captured_request, output_schema)
        request = WireRequest(api_kwargs, captured_request, registry)
        model = request.model
        del api_kwargs, captured_request
        client = self._sdk()

        # Image-bearing requests never chain SDK errors (they carry the request
        # body): the typed error is built in the handler, raised after it.
        failure: Exception | None = None
        try:
            response = await call_with_timeout(
                functools.partial(self._create, client, request),
                deadlines=self._deadlines,
                provider=_PROVIDER,
                model=model,
            )
        except ProviderError:
            raise
        except Exception as exc:
            mapped = self._map_failure(exc, registry, model, streaming=False)
            if mapped is None:
                raise
            if not registry.has_images:
                raise mapped from exc
            failure = mapped
        if failure is not None:
            # Frame locals reach tracebacks (and error trackers): drop the
            # request and registry, which hold the base64 payloads.
            del request, registry
            raise failure

        self._check_response(response, model)
        llm_response = self._normalize_response(response)

        # Attach adapter-boundary payloads for evidence layer
        llm_response.provider_request = request.captured_request
        llm_response.model = model
        llm_response.provider_response = response.model_dump()

        return llm_response

    @staticmethod
    async def _create(client: openai.AsyncOpenAI, request: WireRequest) -> Any:
        """One ``responses.create`` (SDK retries included)."""
        with call_attempt_tracking():
            return await client.responses.create(**request.api_kwargs)

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
        """Stream LLM response as events via the Responses API.

        The Responses API has explicit event types for each phase of output
        (content deltas, function call starts/args/done), making the mapping
        to StreamEvent straightforward.

        ``cache_key_prefix`` routes requests to an OpenAI cache pool via
        ``prompt_cache_key`` (a constructor ``cache_key_prefix`` takes
        precedence). ``run_id`` is never used as a key.

        At stream end, yields a DONE event carrying the full LLMResponse.

        Raises NotImplementedError if the endpoint rejects streaming. An
        ``error`` event or ``response.failed`` raises a typed error with its
        real code. A stream that ends without ``response.completed`` /
        ``response.incomplete`` raises
        :class:`~dendrux.errors.ProviderStreamIncompleteError` (retried once
        when nothing has been yielded yet). Closing the generator closes the
        upstream response.
        """
        registry = WireImageRegistry()
        api_kwargs, captured_request = self._build_api_kwargs(
            messages,
            tools,
            kwargs,
            run_id=run_id,
            cache_key_prefix=cache_key_prefix,
            registry=registry,
        )

        if output_schema is not None:
            self._inject_structured_output(api_kwargs, captured_request, output_schema)
        api_kwargs["stream"] = True
        request = WireRequest(api_kwargs, captured_request, registry)
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
        client: openai.AsyncOpenAI,
        clock: StreamClock,
        request: WireRequest,
    ) -> AsyncGenerator[StreamEvent, None]:
        """One streaming request: open, relay events, require the terminal event."""
        model = request.model
        # Accumulators for building the final LLMResponse
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        refusal_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        usage = UsageStats(usage_reported=False)

        # Track active function calls: item_id → (name, call_id)
        _active_calls: dict[str, tuple[str, str]] = {}
        argument_bytes: dict[str, int] = {}
        # Native message phase ("commentary" | "final_answer"): item_id → phase
        _message_phase: dict[str, str | None] = {}
        _completed_response: Any = None

        failure: Exception | None = None
        stream: Any = None
        try:
            try:
                stream = await clock.open(functools.partial(self._create, client, request))
                iterator = stream.__aiter__()
                while True:
                    try:
                        event = await clock.step(iterator.__anext__())
                    except StopAsyncIteration:
                        break
                    event_type = getattr(event, "type", None)
                    if event_type in {
                        "response.output_text.delta",
                        "response.refusal.delta",
                        "response.reasoning_summary_text.delta",
                    }:
                        clock.account(event.delta)
                    elif event_type == "response.function_call_arguments.delta":
                        clock.tool_slot(event.item_id)
                        size = clock.account(event.delta)
                        argument_bytes[event.item_id] = argument_bytes.get(event.item_id, 0) + size
                    elif event_type == "response.function_call_arguments.done":
                        clock.account(
                            event.arguments, credited_bytes=argument_bytes.pop(event.item_id, 0)
                        )
                    else:
                        clock.account()

                    # --- Failures (they may follow deltas already yielded) ---
                    if event_type in ("error", "response.failed"):
                        error = (
                            event
                            if event_type == "error"
                            else getattr(getattr(event, "response", None), "error", None)
                        )
                        if request.registry.has_images:
                            failure = self._stream_event_image_error(error, request.registry, model)
                        else:
                            failure = self._event_error(error, model)
                        break

                    # --- Text deltas ---
                    if event_type == "response.output_text.delta":
                        delta_text = event.delta
                        if delta_text:
                            text_parts.append(delta_text)
                            yield StreamEvent(
                                type=StreamEventType.TEXT_DELTA,
                                text=delta_text,
                                phase=_message_phase.get(getattr(event, "item_id", "")),
                            )

                    elif event_type == "response.refusal.delta":
                        delta_text = getattr(event, "delta", None)
                        if delta_text:
                            refusal_parts.append(delta_text)

                    # --- Reasoning summary deltas (arrive before output text) ---
                    elif event_type == "response.reasoning_summary_text.delta":
                        delta_text = event.delta
                        if delta_text:
                            reasoning_parts.append(delta_text)
                            yield StreamEvent(type=StreamEventType.REASONING_DELTA, text=delta_text)

                    # --- Function call lifecycle (ordered: added → delta → done) ---
                    elif event_type == "response.output_item.added":
                        item = event.item
                        if getattr(item, "type", None) == "message":
                            clock.account(item.id)
                            _message_phase[item.id] = getattr(item, "phase", None)
                        elif getattr(item, "type", None) == "function_call":
                            clock.tool_slot(item.id)
                            clock.account([item.name, item.id, getattr(item, "call_id", None)])
                            item_id = item.id
                            name = item.name
                            call_id = getattr(item, "call_id", item_id)
                            _active_calls[item_id] = (name, call_id)
                            yield StreamEvent(
                                type=StreamEventType.TOOL_USE_START,
                                tool_name=name,
                                tool_call_id=call_id,
                            )

                    elif event_type == "response.function_call_arguments.done":
                        clock.check_tool_count(len(tool_calls) + 1)
                        item_id = event.item_id
                        raw_args = event.arguments
                        name, call_id = _active_calls.pop(item_id, ("unknown", item_id))
                        if name == "unknown":
                            logger.warning(
                                "Tool call completed with no matching start event — "
                                "provider=%s model=%s item_id=%s",
                                "openai_responses",
                                self._model,
                                item_id,
                            )
                        tc = build_tool_call(
                            name,
                            raw_args or "",
                            provider_tool_call_id=call_id,
                            provider="openai_responses",
                            model=self._model,
                        )
                        tool_calls.append(tc)
                        yield StreamEvent(
                            type=StreamEventType.TOOL_USE_END,
                            tool_call=tc,
                            tool_name=tc.name,
                            tool_call_id=call_id,
                        )

                    # --- Response finished — extract usage. A truncated response
                    # ends with ``response.incomplete`` instead of ``completed``.
                    elif event_type in ("response.completed", "response.incomplete"):
                        _completed_response = event.response
                        if getattr(_completed_response, "usage", None):
                            usage = _build_usage_with_cache(_completed_response.usage)

                    if event_type not in _NON_SEMANTIC_EVENTS:
                        clock.mark_semantic()
            finally:
                if stream is not None:
                    await close_quietly(stream)
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
            # Raised outside any handler: it chains nothing that holds the
            # request body, and this frame never binds the request payload.
            del stream
            raise failure

        if _completed_response is None:
            raise stream_incomplete(_PROVIDER, model)
        if refusal_parts:
            raise refusal_error(_PROVIDER, model, refusal_text="".join(refusal_parts))
        self._check_response(_completed_response, model)

        # Reasoning items (for multi-turn replay) come from the completed response.
        blocks = [
            item.model_dump() if hasattr(item, "model_dump") else item
            for item in getattr(_completed_response, "output", None) or []
            if getattr(item, "type", None) == "reasoning"
        ]
        reasoning_blocks: list[Any] | None = blocks or None

        stop_reason, raw_stop_reason = responses_stop_reason(_completed_response)

        # Assemble the final LLMResponse
        llm_response = LLMResponse(
            text="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls if tool_calls else None,
            raw=None,
            usage=usage,
            reasoning="".join(reasoning_parts) if reasoning_parts else None,
            reasoning_blocks=reasoning_blocks,
            stop_reason=stop_reason,
            raw_stop_reason=raw_stop_reason,
        )
        llm_response.provider_request = request.captured_request
        llm_response.model = model
        if hasattr(_completed_response, "model_dump"):
            llm_response.provider_response = _completed_response.model_dump()

        yield StreamEvent(type=StreamEventType.DONE, raw=llm_response)

    # ------------------------------------------------------------------
    # Outbound conversions: Dendrux → OpenAI Responses API
    # ------------------------------------------------------------------

    def _convert_messages(
        self,
        messages: list[Message],
        *,
        replay_reasoning_for: str | None = None,
        registry: WireImageRegistry | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Convert Dendrux messages to Responses API input format.

        Returns (instructions, input_items) where:
          - SYSTEM messages → joined into instructions string
          - USER → {role: "user", content: ...}
          - ASSISTANT → {role: "assistant", content: ..., phase?: ...} — the
            ``meta["phase"]`` recorded for the message is resent when set
          - ASSISTANT with tool_calls → function_call items
          - TOOL → function_call_output items
          - ASSISTANT ``reasoning_blocks`` → reasoning items ahead of that
            message's text/function_call items, when ``replay_reasoning_for``
            names the request model and it matches the producing model

        Image parts: a USER message becomes ``input_text`` / ``input_image``
        items in caller order. TOOL images follow ``tool_result_images``:
        ``"native"`` emits them inside ``function_call_output.output``;
        ``"portable"`` emits a text output and, after the tool block, ONE framed
        user message (as Chat Completions). Text-only messages take the
        unchanged v0.3.0 path.
        """
        if registry is None:
            registry = WireImageRegistry()
        native = self.tool_result_images == "native"
        instructions_parts: list[str] = []
        input_items: list[dict[str, Any]] = []
        call_index = build_call_index(messages)
        pending: list[tuple[str, str, tuple[ImagePart, ...]]] = []

        for msg in messages:
            if pending and msg.role != Role.TOOL:
                self._flush_tool_images(input_items, pending, registry)

            if msg.role == Role.SYSTEM:
                instructions_parts.append(msg.content)

            elif msg.role == Role.USER:
                if msg.parts is None:
                    input_items.append({"role": "user", "content": msg.content})
                else:
                    content = convert_parts(
                        msg.parts,
                        registry=registry,
                        prefix=("input", len(input_items), "content"),
                        text_block=responses_text,
                        image_block=responses_image,
                    )
                    input_items.append({"role": "user", "content": content})

            elif msg.role == Role.ASSISTANT:
                # The API rejects null-valued fields (e.g. ``status: None``
                # from the SDK's model_dump), so only those are dropped.
                for item in replayable_reasoning_blocks(
                    msg, model=replay_reasoning_for, block_types=_REASONING_ITEM_TYPES
                ):
                    input_items.append({k: v for k, v in item.items() if v is not None})
                if msg.tool_calls:
                    # Emit text content if present
                    if msg.content:
                        input_items.append(_assistant_item(msg))
                    # Emit function_call items
                    for tc in msg.tool_calls:
                        input_items.append(
                            {
                                "type": "function_call",
                                "call_id": tc.provider_tool_call_id or tc.id,
                                "name": tc.name,
                                "arguments": json.dumps(tc.params),
                            }
                        )
                else:
                    input_items.append(_assistant_item(msg))

            elif msg.role == Role.TOOL:
                original_call = resolve_tool_message_call(msg, call_index)
                if msg.parts is None:
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": original_call.provider_tool_call_id or original_call.id,
                            "output": msg.content,
                        }
                    )
                elif native:
                    output = convert_parts(
                        msg.parts,
                        registry=registry,
                        prefix=("input", len(input_items), "output"),
                        text_block=responses_text,
                        image_block=responses_image,
                    )
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": original_call.provider_tool_call_id or original_call.id,
                            "output": output,
                        }
                    )
                else:
                    wire_call_id = original_call.provider_tool_call_id or original_call.id
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": wire_call_id,
                            "output": portable_tool_text(msg),
                        }
                    )
                    if msg.images:
                        pending.append((original_call.name, wire_call_id, msg.images))

        if pending:
            self._flush_tool_images(input_items, pending, registry)
        instructions = "\n\n".join(instructions_parts)
        return instructions, input_items

    @staticmethod
    def _flush_tool_images(
        input_items: list[dict[str, Any]],
        pending: list[tuple[str, str, tuple[ImagePart, ...]]],
        registry: WireImageRegistry,
    ) -> None:
        """Append the framed user message for a tool block's images (portable mode)."""
        content = portable_framing(
            pending,
            registry=registry,
            prefix=("input", len(input_items), "content"),
            text_block=responses_text,
            image_block=responses_image,
        )
        input_items.append({"role": "user", "content": content})
        pending.clear()

    def _build_tools(self, dendrux_tools: list[ToolDef] | None) -> list[dict[str, Any]]:
        """Build the tools array: built-in tools + Dendrux function tools."""
        api_tools: list[dict[str, Any]] = []

        # Add built-in tools (web_search, code_interpreter, etc.)
        for builtin in self._builtin_tools:
            api_tools.append({"type": builtin})

        # Add Dendrux function tools (flatter format than Chat Completions)
        if dendrux_tools:
            for td in dendrux_tools:
                function_tool: dict[str, Any] = {
                    "type": "function",
                    "name": td.name,
                    "description": td.description,
                    "parameters": td.parameters,
                }
                if td.strict:
                    function_tool["strict"] = True
                api_tools.append(function_tool)

        return api_tools

    # ------------------------------------------------------------------
    # Inbound conversions: OpenAI Responses → Dendrux
    # ------------------------------------------------------------------

    def _normalize_response(self, response: Any) -> LLMResponse:
        """Convert Responses API response to Dendrux LLMResponse.

        The response.output is a list of items. We extract:
          - output_text items → text
          - function_call items → ToolCall list
          - Built-in tool outputs (web_search results, etc.) are consumed
            by the model internally — they don't appear as ToolCalls.
        """
        text = response.output_text if hasattr(response, "output_text") else None

        # Extract function calls + reasoning items from output
        tool_calls: list[ToolCall] = []
        reasoning_parts: list[str] = []
        reasoning_blocks: list[Any] = []
        for item in response.output:
            item_type = getattr(item, "type", None)
            if item_type == "function_call":
                tool_calls.append(
                    build_tool_call(
                        item.name,
                        item.arguments or "",
                        provider_tool_call_id=item.call_id,
                        provider="openai_responses",
                        model=self._model,
                    )
                )
            elif item_type == "reasoning":
                # Keep the reasoning item for multi-turn replay; surface the
                # summary text when present.
                reasoning_blocks.append(item.model_dump() if hasattr(item, "model_dump") else item)
                for part in getattr(item, "summary", None) or []:
                    part_text = getattr(part, "text", None)
                    if part_text:
                        reasoning_parts.append(part_text)

        # Extract usage
        usage = UsageStats(usage_reported=False)
        if hasattr(response, "usage") and response.usage:
            usage = _build_usage_with_cache(response.usage)
        stop_reason, raw_stop_reason = responses_stop_reason(response)

        return LLMResponse(
            text=text if text else None,
            tool_calls=tool_calls if tool_calls else None,
            raw=response,
            usage=usage,
            reasoning="".join(reasoning_parts) if reasoning_parts else None,
            reasoning_blocks=reasoning_blocks if reasoning_blocks else None,
            stop_reason=stop_reason,
            raw_stop_reason=raw_stop_reason,
        )
