"""OpenAI Chat Completions API provider.

Thin adapter between Dendrux's universal types and the OpenAI Chat Completions API.
Handles bidirectional conversion:
  - Outbound: Message/ToolDef → OpenAI ChatCompletionMessageParam/tool
  - Inbound:  OpenAI ChatCompletion → LLMResponse/ToolCall

Works with any OpenAI-compatible API (vLLM, SGLang, Groq, Together, Ollama, etc.)
by setting base_url.

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
    resolve_tool_message_call,
)
from dendrux.llm._images import (
    WireImageRegistry,
    chat_image,
    chat_text,
    convert_parts,
    normalize_base_url,
    openai_image_support,
    portable_framing,
    portable_tool_text,
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
from dendrux.llm._stop_reason import chat_stop_reason
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


_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1/"

_EFFORT_ALIASES = {"extra": "xhigh"}


def _normalize_effort(effort: str) -> str:
    """Map a Dendrux effort alias to OpenAI's vocabulary."""
    return _EFFORT_ALIASES.get(effort, effort)


def _extra_attr(obj: Any, field: str) -> Any:
    """Read a field that may live in a Pydantic model's ``model_extra``.

    OpenAI-compatible backends (notably OpenRouter) return reasoning fields
    the OpenAI schema doesn't define; ``extra='allow'`` keeps them, reachable
    via attribute access or ``model_extra``. Pure OpenAI returns none of them,
    so every reader below is inert there.
    """
    value = getattr(obj, field, None)
    if value is not None:
        return value
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict):
        return extra.get(field)
    return None


def _reasoning_text_from(obj: Any) -> str | None:
    """Reasoning summary text from a message or stream delta.

    OpenRouter surfaces it as ``reasoning`` (``reasoning_content`` is an
    accepted alias). Empty strings collapse to ``None`` so "no text exposed"
    stays distinct from "reasoning happened" (which ``reasoning_tokens`` marks).
    """
    for field in ("reasoning", "reasoning_content"):
        value = _extra_attr(obj, field)
        if isinstance(value, str) and value:
            return value
    return None


def _reasoning_details_from(obj: Any) -> list[Any] | None:
    """Opaque reasoning artifacts (``reasoning_details``) to replay verbatim
    across tool-call iterations, per OpenRouter's guidance."""
    value = _extra_attr(obj, "reasoning_details")
    return value if isinstance(value, list) and value else None


def _build_usage_with_cache(usage: Any) -> UsageStats:
    """Build UsageStats from an OpenAI Chat Completions usage object.

    Reads ``prompt_tokens_details.cached_tokens`` (cache reads) and
    ``prompt_tokens_details.cache_write_tokens`` (cache writes) when present
    (compatible backends may omit them; an omitted field stays ``None``).
    ``input_tokens`` is fresh-only: ``prompt_tokens`` minus reads and
    writes, clamped at zero, matching Anthropic's native semantics.
    """
    return openai_wire_usage(
        usage,
        input_field="prompt_tokens",
        output_field="completion_tokens",
        details_field="prompt_tokens_details",
        output_details_field="completion_tokens_details",
    )


# OpenAI-specific kwargs that complete() will forward to the API.
_SUPPORTED_KWARGS = frozenset(
    {
        "temperature",
        "top_p",
        "stop",
        "max_tokens",
        "max_completion_tokens",
        "model",
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "tool_choice",
        "reasoning_effort",
    }
)


def _track_sse_done(stream: Any) -> list[bool] | None:
    """Watch an SDK chat stream for the ``data: [DONE]`` sentinel.

    The SDK stops at ``[DONE]`` without saying so, so a clean ``[DONE]`` end
    and a connection a proxy closed early look the same. This wraps the
    stream's SSE event iterator (before the first pull) to record the
    sentinel. Returns the flag holder, or None when the stream cannot be
    observed (then ``finish_reason`` or a usage chunk counts as the end).
    """
    original = getattr(stream, "_iter_events", None)
    if not callable(original):
        return None
    seen = [False]

    async def tracked() -> AsyncGenerator[Any, None]:
        async for sse in original():
            data = getattr(sse, "data", None)
            if isinstance(data, str) and data.startswith("[DONE]"):
                seen[0] = True
            yield sse

    try:
        stream._iter_events = tracked
    except (AttributeError, TypeError):
        return None
    return seen


class _ToolCallBuffer:
    """Accumulator for one streaming tool call, keyed by chunk index.

    Handles any arrival order of name, id, and argument fragments —
    some OpenAI-compatible backends don't guarantee name-first delivery.
    """

    __slots__ = ("name", "tool_call_id", "json_parts", "start_emitted")

    def __init__(self) -> None:
        self.name: str | None = None
        self.tool_call_id: str | None = None
        self.json_parts: list[str] = []
        self.start_emitted: bool = False


class OpenAIProvider(LLMProvider):
    """OpenAI Chat Completions API provider.

    Works with OpenAI and any OpenAI-compatible API (vLLM, SGLang, Groq,
    Together, Ollama, LM Studio) via base_url.

    Usage:
        # OpenAI
        provider = OpenAIProvider(model="gpt-4o")

        # vLLM / SGLang
        provider = OpenAIProvider(
            model="meta-llama/Llama-3-70B",
            base_url="http://localhost:8000/v1",
        )

        # Groq
        provider = OpenAIProvider(
            model="llama-3.3-70b",
            base_url="https://api.groq.com/openai/v1",
            api_key="gsk-...",
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
        max_context_tokens=128_000,
    )

    # Images in tool results go in ONE framed user message after the tool
    # block: Chat tool messages accept text only (live probe: 400 or silently
    # dropped).
    tool_result_images = "portable"

    # Name reported on typed image errors, and whether a rejected image may be
    # located from the vendor error's request path.
    _image_provider_name = "openai"
    _locate_rejected_images = True
    # Frames between the user's call and the ``timeout=`` deprecation warning.
    _deprecation_stacklevel = 3

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 16_000,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        effort: str | None = None,
        timeout: Any = TIMEOUT_UNSET,
        max_retries: int = 3,
        prompt_cache_retention: Literal["in-memory", "24h"] | None = None,
        default_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        cache_key_prefix: str | None = None,
        call_timeout: float | None = DEFAULT_CALL_TIMEOUT,
        stream_idle_timeout: float | None = DEFAULT_STREAM_IDLE_TIMEOUT,
        stream_total_timeout: float | None = None,
        max_stream_bytes: int = 16 * 1024 * 1024,
        max_stream_tool_calls: int = 1000,
    ) -> None:
        """Create an OpenAI Chat Completions provider.

        Args:
            model: Model identifier (e.g. "gpt-4o", "gpt-4o-mini").
            api_key: API key. Defaults to OPENAI_API_KEY env var.
            base_url: Override for compatible APIs (vLLM, SGLang, Groq, etc.).
            max_tokens: Maximum output tokens per call. Override per-call via kwargs.
            temperature: Sampling temperature. None = model default. For GPT-4o family.
            reasoning_effort: Reasoning depth ("low", "medium", "high"). For o-series/GPT-5.
            effort: Cross-vendor reasoning effort (low|medium|high|xhigh; "extra"→xhigh).
                Overrides ``reasoning_effort`` when both set. Only sent when set, so
                non-reasoning models (gpt-4o) and compatible backends are unaffected.
                OpenAI Chat returns no reasoning text — only ``reasoning_tokens``; use
                OpenAIResponsesProvider to surface reasoning summaries.
            timeout: Deprecated shorthand: sets both ``call_timeout`` and
                ``stream_idle_timeout`` (with a ``DeprecationWarning``).
            max_retries: Number of automatic SDK retries on transient errors
                (connection resets, 408/409/429/5xx). They all count against
                ``call_timeout``.
            prompt_cache_retention: OpenAI prompt-cache retention. ``None``
                (default) omits the field so OpenAI uses its in-memory default
                (5–10 min idle, up to 1 h). Set ``"24h"`` for long-running
                workflows on supported models. Hyphenated literal per the SDK.
            default_headers: Extra HTTP headers sent with every request
                (e.g. OpenRouter attribution headers).
            extra_body: Extra JSON fields merged into every request body —
                for vendor extensions of the Chat Completions API (e.g.
                OpenRouter's ``provider`` routing block). A per-call
                ``extra_body`` kwarg shallow-merges over this, per-call keys
                winning.
            cache_key_prefix: Stable cache-routing identifier, sent as
                ``prompt_cache_key="dendrux:{cache_key_prefix}"`` on the
                official OpenAI endpoint only. OpenAI recommends one key per
                user/customer (not per request); on GPT-5.6+ the key is
                unnecessary. Takes precedence over the per-call prefix the
                agent loop derives (``"{agent}:{model}"``). ``None`` (default)
                leaves routing to the per-call prefix, and sends no key when
                there is none.
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
                stream chunks, and before the first one. ``None`` turns it off.
            max_stream_bytes: Cumulative UTF-8 and structural stream budget.
                Default 16 MiB. Exceeding it closes the upstream response.
            max_stream_tool_calls: Maximum retained tool-call slots (default 1,000).
            stream_total_timeout: Wall-clock limit in seconds for a whole
                stream. ``None`` (default) means no limit.
        """
        if api_key is None and not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OpenAIProvider needs an API key. "
                "Pass api_key='sk-...' or set the OPENAI_API_KEY environment variable. "
                "For local OpenAI-compatible servers (vLLM, Ollama, LM Studio), "
                "pass any non-empty placeholder, e.g. api_key='not-needed'."
            )
        self._deadlines = resolve_deadlines(
            type(self).__name__,
            timeout=timeout,
            call_timeout=call_timeout,
            stream_idle_timeout=stream_idle_timeout,
            stream_total_timeout=stream_total_timeout,
            max_stream_bytes=max_stream_bytes,
            max_stream_tool_calls=max_stream_tool_calls,
            stacklevel=self._deprecation_stacklevel,
        )
        deadlines = self._deadlines

        def new_http() -> httpx.AsyncClient:
            return make_http_client(openai, deadlines)

        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=new_http(),
            max_retries=max_retries,
            default_headers=default_headers,
        )
        self._clients: LoopClients[openai.AsyncOpenAI] = LoopClients(
            new_http, lambda client, http: client.copy(http_client=http)
        )
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
        self._effort = effort
        self._prompt_cache_retention = prompt_cache_retention
        self._extra_body = extra_body
        self._cache_key_prefix = cache_key_prefix

    @property
    def model(self) -> str:
        """The model identifier this provider is configured to use."""
        return self._model

    def __repr__(self) -> str:
        base = str(self._client.base_url)
        if base != "https://api.openai.com/v1/":
            return f"OpenAIProvider(model={self._model!r}, base_url={base!r})"
        return f"OpenAIProvider(model={self._model!r})"

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

    # ------------------------------------------------------------------
    # Image input (0.4.0)
    # ------------------------------------------------------------------

    async def image_input_support(self, model: str) -> ImageSupport:
        """Static table on the official OpenAI endpoint; ``"unknown"`` elsewhere."""
        if str(self._client.base_url) != _OPENAI_DEFAULT_BASE_URL:
            return "unknown"
        return openai_image_support(model)

    def image_capability_key(self, model: str) -> tuple[str, str, str]:
        return ("chat", normalize_base_url(self._client.base_url), model)

    def _image_error(
        self, exc: openai.APIError, registry: WireImageRegistry, model: str, *, streaming: bool
    ) -> Exception:
        """Typed, sanitised error for a failed image-bearing request (not raised)."""
        return sdk_image_request_error(
            exc,
            provider=self,
            provider_name=self._image_provider_name,
            model=model,
            registry=registry,
            streaming=streaming,
            locate=self._locate_rejected_images,
        )

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
            provider=self._image_provider_name,
            model=model,
            streaming=streaming,
            endpoint=str(self._client.base_url),
        )

    async def __aenter__(self) -> OpenAIProvider:
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
        """Build OpenAI API kwargs from Dendrux messages and caller kwargs.

        Returns (api_kwargs, captured_request) where captured_request is the
        serializable subset for the evidence layer. On an image-bearing request
        every data URL in captured_request is replaced by
        ``"dendrux-image:sha256:<hex>"`` (``registry`` records the images).

        Cache routing: ``prompt_cache_key`` is sent only when a prefix is
        known — the constructor ``cache_key_prefix`` first, then the per-call
        one (the agent loop passes ``"{agent_name}:{model}"``). ``run_id`` is
        accepted for interface compatibility and never used as a key. The
        provider's ``prompt_cache_retention`` is forwarded only when set, so
        vendor defaults apply otherwise.
        """
        if registry is None:
            registry = WireImageRegistry()
        api_messages = self._convert_messages(messages, registry=registry)
        api_tools = self._convert_tools(tools) if tools else openai.NOT_GIVEN

        # Resolve the output-token cap: per-call kwargs override constructor default.
        # The official OpenAI endpoint requires `max_completion_tokens` (gpt-5 and
        # o-series reject `max_tokens` with a 400, and it's deprecated for gpt-4o
        # too); OpenAI-compatible backends (vLLM, Groq, Ollama, etc.) still expect
        # `max_tokens`. Gate on base_url, mirroring the prompt_cache_key routing below.
        max_output_tokens = kwargs.pop(
            "max_completion_tokens",
            kwargs.pop("max_tokens", self._max_tokens),
        )
        is_openai_endpoint = str(self._client.base_url) == _OPENAI_DEFAULT_BASE_URL
        token_param = "max_completion_tokens" if is_openai_endpoint else "max_tokens"

        api_kwargs: dict[str, Any] = {
            "model": kwargs.pop("model", self._model),
            "messages": api_messages,
            "tools": api_tools,
            token_param: max_output_tokens,
        }

        # `effort` is the cross-vendor reasoning knob; it overrides
        # reasoning_effort. `thinking` is accepted for API uniformity but does
        # not by itself add/remove reasoning_effort — sending it to a
        # non-reasoning model (gpt-4o) or compatible backend would break the
        # call, so we only send a value when one is explicitly given.
        kwargs.pop("thinking", None)
        effort = kwargs.pop("effort", self._effort)
        if effort is not None:
            kwargs["reasoning_effort"] = _normalize_effort(effort)

        # Apply constructor defaults for optional params (per-call kwargs override)
        for attr, key in [
            (self._temperature, "temperature"),
            (self._reasoning_effort, "reasoning_effort"),
        ]:
            if key in kwargs:
                api_kwargs[key] = kwargs.pop(key)
            elif attr is not None:
                api_kwargs[key] = attr

        # Forward remaining supported kwargs — ignore the rest
        already_handled = {
            "model",
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "reasoning_effort",
        }
        for key in _SUPPORTED_KWARGS - already_handled:
            if key in kwargs:
                api_kwargs[key] = kwargs.pop(key)

        # Cache routing — never keyed on run_id: a per-run key scatters one
        # conversation's runs across cache lanes and defeats cross-run hits.
        # Skip on non-OpenAI base_urls (Groq/Together/vLLM/etc.) since some
        # compatible backends reject unknown OpenAI-specific request fields.
        if is_openai_endpoint:
            cache_key_source = self._cache_key_prefix or cache_key_prefix
            if cache_key_source:
                api_kwargs["prompt_cache_key"] = f"dendrux:{cache_key_source}"
            if self._prompt_cache_retention is not None:
                api_kwargs["prompt_cache_retention"] = self._prompt_cache_retention

        # Vendor extensions of the request body (e.g. OpenRouter's `provider`
        # routing block). Per-call extra_body shallow-merges over the
        # constructor default, per-call keys winning.
        call_extra_body = kwargs.pop("extra_body", None)
        if self._extra_body or call_extra_body:
            api_kwargs["extra_body"] = {**(self._extra_body or {}), **(call_extra_body or {})}

        captured_request = {k: v for k, v in api_kwargs.items() if v is not openai.NOT_GIVEN}
        if registry.has_images:
            captured_request = registry.sanitize(captured_request)
        return api_kwargs, captured_request

    def _inject_structured_output(
        self,
        api_kwargs: dict[str, Any],
        captured_request: dict[str, Any],
        output_schema: dict[str, Any],
    ) -> None:
        """Inject response_format for structured output and update evidence."""
        from dendrux.llm._schema import normalize_for_openai_strict

        strict_schema = normalize_for_openai_strict(output_schema)
        schema_name = output_schema.get("title", "structured_output")
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": strict_schema,
            },
        }
        api_kwargs["response_format"] = response_format
        captured_request["response_format"] = response_format

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
        """Send messages to the model and return a normalized response.

        kwargs override constructor defaults (e.g. model, max_tokens, temperature).
        Only supported kwargs are forwarded; unknown keys are ignored.

        ``cache_key_prefix`` (typically ``"{agent_name}:{model}"``) drives the
        ``prompt_cache_key`` sent to OpenAI (a constructor
        ``cache_key_prefix`` takes precedence). ``run_id`` is never used as
        a key.

        When output_schema is provided, sets response_format to json_schema
        with strict mode. The response is a JSON string in message.content.

        The whole call, SDK retries included, is bounded by ``call_timeout``.
        Failures raise typed :class:`~dendrux.errors.ProviderError`
        subclasses (SDK exception on ``__cause__``); a refusal raises
        :class:`~dendrux.errors.ModelRefusalError` and a content filter
        :class:`~dendrux.errors.ContentFilteredError`.
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
                provider=self._image_provider_name,
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

        llm_response = self._normalize_response(response, model=model)

        # Attach adapter-boundary payloads for evidence layer
        llm_response.provider_request = request.captured_request
        llm_response.model = model
        llm_response.provider_response = response.model_dump()

        return llm_response

    @staticmethod
    async def _create(client: openai.AsyncOpenAI, request: WireRequest) -> Any:
        """One ``chat.completions.create`` (SDK retries included)."""
        with call_attempt_tracking():
            return await client.chat.completions.create(**request.api_kwargs)

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
        Tool call arguments are buffered by index and flushed as complete
        TOOL_USE_END events when the terminal finish_reason arrives, or at
        end of stream if the backend never sent one.

        ``cache_key_prefix`` drives ``prompt_cache_key`` for OpenAI cache
        pool routing (a constructor ``cache_key_prefix`` takes precedence).
        ``run_id`` is never used as a key.

        At stream end, yields a DONE event carrying the full LLMResponse.

        Raises NotImplementedError if the endpoint says streaming is not
        supported (common with some OpenAI-compatible backends), and
        ContextLengthExceededError if the request overflows the context window.
        Other failures raise typed :class:`~dendrux.errors.ProviderError`
        subclasses. A stream that ends without ``[DONE]``, a ``finish_reason``
        or a usage chunk raises :class:`~dendrux.errors.ProviderStreamIncompleteError`
        (retried once when nothing has been yielded yet). Closing the
        generator closes the upstream response.
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
        api_kwargs["stream_options"] = {"include_usage": True}
        request = WireRequest(api_kwargs, captured_request, registry)
        model = request.model
        del api_kwargs, captured_request, registry
        client = self._sdk()
        clock = StreamClock(self._deadlines, provider=self._image_provider_name, model=model)
        # Closing this generator (aclose, cancel) closes the upstream response.
        async with contextlib.aclosing(
            retry_unstarted_stream(
                functools.partial(self._stream_attempt, client, clock, request),
                provider=self._image_provider_name,
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
        """One streaming request: open, relay chunks, require the terminal chunk."""
        model = request.model
        # Accumulators for building the final LLMResponse
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        reasoning_parts: list[str] = []
        reasoning_block_acc: list[Any] = []
        refusal_parts: list[str] = []
        usage = UsageStats(usage_reported=False)
        finish_reason: str | None = None
        saw_usage = False

        # Tool call buffers keyed by chunk index — no ordering assumptions.
        # Name, id, and arguments may arrive in any order for a given index
        # (some OpenAI-compatible backends don't guarantee name-first).
        _tool_buffers: dict[int, _ToolCallBuffer] = {}

        failure: Exception | None = None
        stream: Any = None
        done_seen: list[bool] | None = None
        try:
            try:
                stream = await clock.open(functools.partial(self._create, client, request))
                done_seen = _track_sse_done(stream)
                chunk_iter = stream.__aiter__()
                while True:
                    try:
                        chunk = await clock.step(chunk_iter.__anext__())
                    except StopAsyncIteration:
                        break
                    clock.account(chunk)
                    semantic = False

                    # Usage arrives in the final chunk (``include_usage``: no
                    # choices). Some backends repeat usage on every content
                    # chunk; only the choice-less shape marks the end.
                    if chunk.usage:
                        usage = self._normalize_usage(chunk.usage)
                        saw_usage = saw_usage or not chunk.choices
                        semantic = True

                    if not chunk.choices:
                        if semantic:
                            clock.mark_semantic()
                        continue

                    choice = chunk.choices[0]
                    delta = choice.delta

                    # --- Reasoning deltas (OpenRouter/compatible; inert on pure OpenAI) ---
                    # Reasoning streams before content, so emit it first.
                    reasoning_delta = _reasoning_text_from(delta)
                    if reasoning_delta:
                        semantic = True
                        reasoning_parts.append(reasoning_delta)
                        yield StreamEvent(
                            type=StreamEventType.REASONING_DELTA, text=reasoning_delta
                        )
                    reasoning_detail_delta = _reasoning_details_from(delta)
                    if reasoning_detail_delta:
                        semantic = True
                        reasoning_block_acc.extend(reasoning_detail_delta)

                    refusal_delta = getattr(delta, "refusal", None)
                    if isinstance(refusal_delta, str) and refusal_delta:
                        semantic = True
                        refusal_parts.append(refusal_delta)

                    # --- Text deltas ---
                    if delta.content:
                        semantic = True
                        text_parts.append(delta.content)
                        yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=delta.content)

                    # --- Tool call deltas ---
                    if delta.tool_calls:
                        semantic = True
                        for delta_tc in delta.tool_calls:
                            idx = delta_tc.index

                            # Ensure buffer exists for this index
                            if idx not in _tool_buffers:
                                clock.tool_slot((len(tool_calls), idx))
                                _tool_buffers[idx] = _ToolCallBuffer()

                            buf = _tool_buffers[idx]

                            # Capture id (first occurrence)
                            if delta_tc.id and buf.tool_call_id is None:
                                buf.tool_call_id = delta_tc.id

                            if delta_tc.function:
                                # Capture name (first occurrence)
                                if delta_tc.function.name and buf.name is None:
                                    buf.name = delta_tc.function.name

                                # Accumulate argument fragments
                                if delta_tc.function.arguments:
                                    buf.json_parts.append(delta_tc.function.arguments)

                            # Emit TOOL_USE_START once we know the name (deferred
                            # if args arrived before name for this index).
                            if buf.name and not buf.start_emitted:
                                buf.start_emitted = True
                                yield StreamEvent(
                                    type=StreamEventType.TOOL_USE_START,
                                    tool_name=buf.name,
                                    tool_call_id=buf.tool_call_id,
                                )

                    # --- Flush all buffered tool calls on terminal finish_reason ---
                    if choice.finish_reason is not None:
                        semantic = True
                        finish_reason = choice.finish_reason
                        clock.check_tool_count(len(tool_calls) + len(_tool_buffers))
                        for flush_event in self._drain_tool_buffers(_tool_buffers, tool_calls):
                            yield flush_event

                    if semantic:
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
            # Frame locals reach tracebacks (and error trackers): the request
            # (with its base64 payloads) is never bound here.
            del stream
            raise failure

        if finish_reason is None and not saw_usage and not (done_seen and done_seen[0]):
            raise stream_incomplete(self._image_provider_name, model)
        if refusal_parts:
            raise refusal_error(
                self._image_provider_name, model, refusal_text="".join(refusal_parts)
            )
        if finish_reason == "content_filter":
            raise content_filter_error(self._image_provider_name, model)

        # Some compatible backends end the stream ([DONE]) without ever sending
        # a finish_reason; flush whatever is still buffered so calls are not
        # lost. No-op when the finish_reason chunk drained the buffers.
        for flush_event in self._drain_tool_buffers(_tool_buffers, tool_calls):
            yield flush_event

        # Assemble the final LLMResponse
        stop_reason, raw_stop_reason = chat_stop_reason(finish_reason)
        llm_response = LLMResponse(
            text="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls if tool_calls else None,
            raw=None,
            usage=usage,
            reasoning="".join(reasoning_parts) if reasoning_parts else None,
            reasoning_blocks=reasoning_block_acc if reasoning_block_acc else None,
            stop_reason=stop_reason,
            raw_stop_reason=raw_stop_reason,
        )
        llm_response.provider_request = request.captured_request
        llm_response.model = model
        llm_response.provider_response = {
            "object": "chat.completion.chunked",
            "finish_reason": finish_reason,
            "usage": {
                "prompt_tokens": usage.input_tokens,
                "completion_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
            },
        }

        yield StreamEvent(type=StreamEventType.DONE, raw=llm_response)

    def _drain_tool_buffers(
        self,
        buffers: dict[int, _ToolCallBuffer],
        tool_calls: list[ToolCall],
    ) -> list[StreamEvent]:
        """Turn buffered tool-call fragments into START/END events and clear them.

        Appends each completed call to ``tool_calls``. Emptying ``buffers``
        makes a second drain a no-op, so a call is never emitted twice.
        """
        events: list[StreamEvent] = []
        for idx in sorted(buffers):
            buf = buffers[idx]
            raw_json = "".join(buf.json_parts)

            # Ensure START was emitted even if name arrived late
            if buf.name and not buf.start_emitted:
                buf.start_emitted = True
                events.append(
                    StreamEvent(
                        type=StreamEventType.TOOL_USE_START,
                        tool_name=buf.name,
                        tool_call_id=buf.tool_call_id,
                    )
                )

            if buf.name is None:
                logger.warning(
                    "Tool call completed with no name — provider=%s model=%s call_id=%s",
                    "openai",
                    self._model,
                    buf.tool_call_id,
                )

            tc = build_tool_call(
                buf.name or "unknown",
                raw_json,
                provider_tool_call_id=buf.tool_call_id,
                provider="openai",
                model=self._model,
            )
            tool_calls.append(tc)
            events.append(
                StreamEvent(
                    type=StreamEventType.TOOL_USE_END,
                    tool_call=tc,
                    tool_name=tc.name,
                    tool_call_id=buf.tool_call_id,
                )
            )
        buffers.clear()
        return events

    # ------------------------------------------------------------------
    # Outbound conversions: Dendrux → OpenAI
    # ------------------------------------------------------------------

    def _convert_messages(
        self, messages: list[Message], *, registry: WireImageRegistry | None = None
    ) -> list[dict[str, Any]]:
        """Convert Dendrux messages to OpenAI Chat Completions format.

        Role mapping:
          - SYSTEM → "system" (or "developer" — both work, "system" is universal)
          - USER → "user"
          - ASSISTANT → "assistant", with tool_calls array if present
          - TOOL → "tool" with tool_call_id correlation

        Image parts (portable mapping): a USER message becomes a content-part
        list (``text`` / ``image_url``) in caller order. A TOOL message with
        images keeps text-only content (payload, notes, placeholders, then the
        image note); after the last TOOL message of the block ONE framed user
        message carries every visible tool image in call order. Text-only
        messages take the unchanged v0.3.0 path.
        """
        if registry is None:
            registry = WireImageRegistry()
        api_messages: list[dict[str, Any]] = []
        call_index = build_call_index(messages)
        # Visible tool images of the current tool block, flushed as one framed
        # user message before the next non-TOOL message (or at the end).
        pending: list[tuple[str, str, tuple[ImagePart, ...]]] = []

        for msg in messages:
            if pending and msg.role != Role.TOOL:
                self._flush_tool_images(api_messages, pending, registry)

            if msg.role == Role.SYSTEM:
                api_messages.append({"role": "system", "content": msg.content})

            elif msg.role == Role.USER:
                if msg.parts is None:
                    api_messages.append({"role": "user", "content": msg.content})
                else:
                    content = convert_parts(
                        msg.parts,
                        registry=registry,
                        prefix=("messages", len(api_messages), "content"),
                        text_block=chat_text,
                        image_block=chat_image,
                    )
                    api_messages.append({"role": "user", "content": content})

            elif msg.role == Role.ASSISTANT:
                if msg.tool_calls:
                    # Assistant with tool calls
                    api_msg: dict[str, Any] = {"role": "assistant"}
                    if msg.content:
                        api_msg["content"] = msg.content
                    api_msg["tool_calls"] = [
                        {
                            "id": tc.provider_tool_call_id or tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.params),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                else:
                    api_msg = {"role": "assistant", "content": msg.content}
                # Replay provider reasoning so multi-step tool calls keep the
                # model's reasoning context, per OpenRouter's guidance. Prefer
                # the structured blocks (verbatim — required for encrypted/
                # summarized types); fall back to the plaintext string for
                # models that only return raw reasoning. Inert for pure OpenAI,
                # which never populates either.
                if msg.reasoning_blocks:
                    api_msg["reasoning_details"] = msg.reasoning_blocks
                elif msg.reasoning:
                    api_msg["reasoning"] = msg.reasoning
                api_messages.append(api_msg)

            elif msg.role == Role.TOOL:
                original_call = resolve_tool_message_call(msg, call_index)

                if msg.parts is None:
                    api_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": original_call.provider_tool_call_id or original_call.id,
                            "content": msg.content,
                        }
                    )
                else:
                    wire_call_id = original_call.provider_tool_call_id or original_call.id
                    api_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": wire_call_id,
                            "content": portable_tool_text(msg),
                        }
                    )
                    if msg.images:
                        pending.append((original_call.name, wire_call_id, msg.images))

        if pending:
            self._flush_tool_images(api_messages, pending, registry)
        return api_messages

    @staticmethod
    def _flush_tool_images(
        api_messages: list[dict[str, Any]],
        pending: list[tuple[str, str, tuple[ImagePart, ...]]],
        registry: WireImageRegistry,
    ) -> None:
        """Append the framed user message for a tool block's images; clear ``pending``."""
        content = portable_framing(
            pending,
            registry=registry,
            prefix=("messages", len(api_messages), "content"),
            text_block=chat_text,
            image_block=chat_image,
        )
        api_messages.append({"role": "user", "content": content})
        pending.clear()

    def _convert_tools(self, tools: list[ToolDef]) -> list[dict[str, Any]]:
        """Convert Dendrux ToolDefs to OpenAI function tool format."""
        converted: list[dict[str, Any]] = []
        for td in tools:
            function: dict[str, Any] = {
                "name": td.name,
                "description": td.description,
                "parameters": td.parameters,
            }
            if td.strict:
                function["strict"] = True
            converted.append({"type": "function", "function": function})
        return converted

    # ------------------------------------------------------------------
    # Inbound conversions: OpenAI → Dendrux
    # ------------------------------------------------------------------

    def _normalize_usage(self, usage: Any) -> UsageStats:
        """Map a provider usage object to :class:`UsageStats`.

        Overridable seam for OpenAI-compatible presets that return extra
        usage fields on the same wire format (e.g. OpenRouter's ``cost`` and
        ``cache_write_tokens``). The base implementation covers the standard
        Chat Completions usage shape; both the streaming and non-streaming
        paths route through here so an override enriches both at once.
        """
        return _build_usage_with_cache(usage)

    def _normalize_response(self, response: Any, *, model: str | None = None) -> LLMResponse:
        """Convert OpenAI ChatCompletion to Dendrux LLMResponse.

        A ``message.refusal`` raises :class:`~dendrux.errors.ModelRefusalError`;
        ``finish_reason="content_filter"`` raises
        :class:`~dendrux.errors.ContentFilteredError`.
        """
        model = model or self._model
        if not response.choices:
            raise ProviderError(
                provider=self._image_provider_name,
                model=model,
                category="unknown",
                message="The provider returned a completion with no choices.",
            )
        choice = response.choices[0]
        message = choice.message
        refusal = getattr(message, "refusal", None)
        if isinstance(refusal, str) and refusal:
            raise refusal_error(self._image_provider_name, model, refusal_text=refusal)
        if getattr(choice, "finish_reason", None) == "content_filter":
            raise content_filter_error(self._image_provider_name, model)

        # Extract text
        text = message.content

        # Extract tool calls
        tool_calls: list[ToolCall] | None = None
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                tool_calls.append(
                    build_tool_call(
                        tc.function.name,
                        tc.function.arguments or "",
                        provider_tool_call_id=tc.id,
                        provider="openai",
                        model=self._model,
                    )
                )

        # Extract usage
        usage = UsageStats(usage_reported=False)
        if response.usage:
            usage = self._normalize_usage(response.usage)
        stop_reason, raw_stop_reason = chat_stop_reason(getattr(choice, "finish_reason", None))

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            raw=response,
            usage=usage,
            reasoning=_reasoning_text_from(message),
            reasoning_blocks=_reasoning_details_from(message),
            stop_reason=stop_reason,
            raw_stop_reason=raw_stop_reason,
        )
