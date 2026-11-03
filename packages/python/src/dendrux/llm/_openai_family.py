"""Plumbing shared by the OpenAI-family adapters (Chat Completions, Responses, OpenRouter).

Internal module, not part of the public API. It holds what the two OpenAI
adapters did identically: the client lifecycle, the request pipeline around
the SDK call (build, call, map the failure, attach the evidence), the
failure tail of one stream attempt, cache routing, the structured-output
step and the tool-image flow of message conversion. Everything that is wire
format stays in each adapter: message and tool shapes, stream event loops,
response normalisation and reasoning.

Its own module (not ``_helpers``) because these pieces only make sense for
the OpenAI wire family, and (not ``openai``) because the Responses adapter
must not depend on the Chat adapter. It never imports the vendor SDK: an
adapter names its SDK module in ``_sdk_module``, so the SDK stays imported
by the adapters alone.
"""

from __future__ import annotations

import contextlib
import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self

from dendrux.errors import ProviderError
from dendrux.llm._images import (
    WireImageRegistry,
    convert_parts,
    normalize_base_url,
    openai_image_support,
    portable_framing,
    portable_tool_text,
    sdk_image_request_error,
)
from dendrux.llm._resilience import (
    StreamClock,
    WireRequest,
    call_with_timeout,
    close_quietly,
    map_sdk_error,
    retry_unstarted_stream,
)
from dendrux.llm.base import LLMProvider

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        AsyncIterator,
        Awaitable,
        Callable,
        Iterable,
        Sequence,
    )
    from types import ModuleType

    from dendrux.llm._resilience import Deadlines, LoopClients
    from dendrux.llm.base import ImageSupport
    from dendrux.types import (
        ContentPart,
        ImagePart,
        LLMResponse,
        Message,
        StreamEvent,
        ToolCall,
        ToolDef,
    )

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1/"

PendingImages = list[tuple[str, str, "tuple[ImagePart, ...]"]]


# ------------------------------------------------------------------
# Small shared readers and request steps
# ------------------------------------------------------------------


def extra_field(obj: Any, field: str) -> Any:
    """Read a field that may live in a Pydantic model's ``model_extra``.

    OpenAI-compatible backends (notably OpenRouter) return fields the OpenAI
    schema does not define; ``extra='allow'`` keeps them, reachable via
    attribute access or ``model_extra``. Pure OpenAI returns none of them.
    """
    value = getattr(obj, field, None)
    if value is not None:
        return value
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict):
        return extra.get(field)
    return None


def usage_extra_field(obj: Any, field: str) -> Any:
    """:func:`extra_field` that also accepts ``None`` and a raw ``dict``.

    Kept apart from :func:`extra_field` on purpose: the two differ on a plain
    dict (this one reads it, :func:`extra_field` does not), and each caller's
    behaviour is pinned.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(field)
    return extra_field(obj, field)


def apply_call_defaults(
    api_kwargs: dict[str, Any], kwargs: dict[str, Any], defaults: Iterable[tuple[Any, str]]
) -> None:
    """Per-call kwargs win; otherwise a non-None constructor default is sent."""
    for attr, key in defaults:
        if key in kwargs:
            api_kwargs[key] = kwargs.pop(key)
        elif attr is not None:
            api_kwargs[key] = attr


def forward_kwargs(api_kwargs: dict[str, Any], kwargs: dict[str, Any], keys: Iterable[str]) -> None:
    """Move each supported per-call kwarg that is present into the request."""
    for key in keys:
        if key in kwargs:
            api_kwargs[key] = kwargs.pop(key)


def apply_cache_routing(
    api_kwargs: dict[str, Any],
    *,
    own_prefix: str | None,
    call_prefix: str | None,
    retention: str | None,
) -> None:
    """``prompt_cache_key`` from the constructor prefix, else the per-call one.

    Never keyed on run_id: a per-run key scatters one conversation's runs
    across cache lanes and defeats cross-run hits. ``prompt_cache_retention``
    is sent only when set, so vendor defaults apply otherwise.
    """
    cache_key_source = own_prefix or call_prefix
    if cache_key_source:
        api_kwargs["prompt_cache_key"] = f"dendrux:{cache_key_source}"
    if retention is not None:
        api_kwargs["prompt_cache_retention"] = retention


def inject_structured_output(
    api_kwargs: dict[str, Any],
    captured_request: dict[str, Any],
    output_schema: dict[str, Any],
    *,
    key: str,
    envelope: Callable[[str, dict[str, Any]], dict[str, Any]],
) -> None:
    """Set a strict JSON-schema output format under ``key`` in the request and its evidence.

    ``envelope(name, strict_schema)`` builds the adapter's wire shape.
    """
    from dendrux.llm._schema import normalize_for_openai_strict

    value = envelope(
        output_schema.get("title", "structured_output"),
        normalize_for_openai_strict(output_schema),
    )
    api_kwargs[key] = value
    captured_request[key] = value


def raise_or_defer(
    provider: OpenAIFamilyProvider,
    exc: Exception,
    registry: WireImageRegistry,
    model: str,
    *,
    streaming: bool,
) -> Exception:
    """Map a failed SDK call; call this inside the ``except`` that caught ``exc``.

    Re-raises ``exc`` when it maps to nothing, and raises the typed error
    chained to ``exc`` for a text-only request. For an image-bearing request
    it returns the typed error instead: the caller raises it after its
    handler (so nothing that holds the request body is chained) and after
    deleting its own locals that hold the payload.
    """
    mapped = provider._map_failure(exc, registry, model, streaming=streaming)
    if mapped is None:
        raise exc
    if not registry.has_images:
        raise mapped from exc
    return mapped


@contextlib.asynccontextmanager
async def opened_stream(
    clock: StreamClock, make_call: Callable[[], Awaitable[Any]]
) -> AsyncIterator[Any]:
    """Open one SDK stream within the clock's deadlines; always close it quietly."""
    stream = await clock.open(make_call)
    try:
        yield stream
    finally:
        await close_quietly(stream)


# ------------------------------------------------------------------
# Message conversion: the tool-image flow
# ------------------------------------------------------------------


@dataclass(frozen=True)
class WireBlocks:
    """Where image parts sit in a request and how text and image blocks look."""

    prefix: str
    text_block: Callable[[str], dict[str, Any]]
    image_block: Callable[[WireImageRegistry, ImagePart], dict[str, Any]]

    def parts(
        self,
        parts: Sequence[ContentPart],
        registry: WireImageRegistry,
        index: int,
        field: str = "content",
    ) -> list[dict[str, Any]]:
        """``parts`` as wire blocks, each image registered at its wire path."""
        return convert_parts(
            parts,
            registry=registry,
            prefix=(self.prefix, index, field),
            text_block=self.text_block,
            image_block=self.image_block,
        )

    def user_item(self, msg: Message, registry: WireImageRegistry, index: int) -> dict[str, Any]:
        """A USER message: plain content, or content parts in caller order."""
        if msg.parts is None:
            return {"role": "user", "content": msg.content}
        return {"role": "user", "content": self.parts(msg.parts, registry, index)}

    def flush_tool_images(
        self, items: list[dict[str, Any]], pending: PendingImages, registry: WireImageRegistry
    ) -> None:
        """Append ONE framed user message carrying a tool block's images; clear ``pending``."""
        content = portable_framing(
            pending,
            registry=registry,
            prefix=(self.prefix, len(items), "content"),
            text_block=self.text_block,
            image_block=self.image_block,
        )
        items.append({"role": "user", "content": content})
        pending.clear()


def portable_tool_item(
    msg: Message,
    call: ToolCall,
    pending: PendingImages,
    make_item: Callable[[str, str], dict[str, Any]],
) -> dict[str, Any]:
    """A TOOL message under the portable mapping: text only, images queued in ``pending``.

    ``make_item(wire_call_id, text)`` builds the adapter's tool-result item.
    """
    wire_call_id = call.provider_tool_call_id or call.id
    if msg.parts is None:
        return make_item(wire_call_id, msg.content)
    if msg.images:
        pending.append((call.name, wire_call_id, msg.images))
    return make_item(wire_call_id, portable_tool_text(msg))


# ------------------------------------------------------------------
# The shared provider base
# ------------------------------------------------------------------


class OpenAIFamilyProvider(LLMProvider):
    """Lifecycle, failure mapping and the request pipeline of the OpenAI adapters.

    Subclasses set the class attributes and implement ``_build_api_kwargs``,
    ``_inject_structured_output``, ``_create``, ``_finish_response`` and
    ``_stream_attempt``; ``complete`` / ``complete_stream`` stay on the
    adapter (decorated, with its own docstring) and call ``_complete_wire`` /
    ``_stream_wire``.
    """

    # Name reported on typed errors and logs ("openai", "openai_responses", ...).
    _image_provider_name: ClassVar[str]
    # First element of ``image_capability_key``.
    _image_capability_family: ClassVar[str]
    # Whether a rejected image may be located from the vendor error's path.
    _locate_rejected_images: ClassVar[bool] = True
    # The vendor SDK module (``openai``), named by the adapter that imports it.
    _sdk_module: ModuleType
    # Chat Completions asks for a final usage chunk on streams.
    _stream_include_usage: ClassVar[bool] = False

    _client: Any
    _clients: LoopClients[Any]
    _deadlines: Deadlines
    _model: str

    @property
    def model(self) -> str:
        """The model identifier this provider is configured to use."""
        return self._model

    async def close(self) -> None:
        """Close the underlying HTTP client and release connections."""
        # Close only what belongs to the running loop: after a loop change the
        # home transport is first rebuilt, never awaited across loops.
        current = self._clients.resolve(self._client)
        self._clients.pop_current_loop_extra()
        await current.close()

    def _sdk(self) -> Any:
        """The SDK client for the running event loop (rebuilt after a loop change)."""
        return self._clients.resolve(self._client)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def image_input_support(self, model: str) -> ImageSupport:
        """Static table on the official OpenAI endpoint; ``"unknown"`` elsewhere."""
        if str(self._client.base_url) != OPENAI_DEFAULT_BASE_URL:
            return "unknown"
        return openai_image_support(model)

    def image_capability_key(self, model: str) -> tuple[str, str, str]:
        return (self._image_capability_family, normalize_base_url(self._client.base_url), model)

    def _image_error(
        self, exc: BaseException, registry: WireImageRegistry, model: str, *, streaming: bool
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
        sdk = self._sdk_module
        if (
            registry.has_images
            and isinstance(exc, sdk.APIError)
            and not isinstance(exc, sdk.APIConnectionError)
        ):
            return self._image_error(exc, registry, model, streaming=streaming)
        return map_sdk_error(
            exc,
            sdk=sdk,
            provider=self._image_provider_name,
            model=model,
            streaming=streaming,
            endpoint=str(self._client.base_url),
        )

    # -- adapter hooks (each adapter defines them) ------------------------
    # _build_api_kwargs(messages, tools, kwargs, *, run_id, cache_key_prefix,
    #     registry) -> (api_kwargs, captured_request)
    _build_api_kwargs: Callable[..., tuple[dict[str, Any], dict[str, Any]]]
    # _inject_structured_output(api_kwargs, captured_request, output_schema)
    _inject_structured_output: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], None]
    # _create(client, request): one SDK create call (SDK retries included)
    _create: Callable[[Any, WireRequest], Awaitable[Any]]
    # _finish_response(response, model): check and normalise a non-streaming reply
    _finish_response: Callable[[Any, str], LLMResponse]
    # _stream_attempt(client, clock, request): one streaming request
    _stream_attempt: Callable[[Any, StreamClock, WireRequest], AsyncGenerator[StreamEvent, None]]

    # -- the shared pipeline ---------------------------------------------

    def _wire_request(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None,
        kwargs: dict[str, Any],
        output_schema: dict[str, Any] | None,
        run_id: str | None,
        cache_key_prefix: str | None,
        *,
        stream: bool = False,
    ) -> WireRequest:
        """Build the SDK kwargs and their evidence copy for one call."""
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
        if stream:
            api_kwargs["stream"] = True
            if self._stream_include_usage:
                api_kwargs["stream_options"] = {"include_usage": True}
        return WireRequest(api_kwargs, captured_request, registry)

    async def _complete_wire(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None,
        kwargs: dict[str, Any],
        output_schema: dict[str, Any] | None,
        run_id: str | None,
        cache_key_prefix: str | None,
    ) -> LLMResponse:
        """One non-streaming call: build, call (bounded by ``call_timeout``), map, attach."""
        request = self._wire_request(
            messages, tools, kwargs, output_schema, run_id, cache_key_prefix
        )
        model = request.model
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
            failure = raise_or_defer(self, exc, request.registry, model, streaming=False)
        if failure is not None:
            # Frame locals reach tracebacks (and error trackers): drop the
            # request, which holds the base64 payloads.
            del request
            raise failure

        llm_response = self._finish_response(response, model)

        # Attach adapter-boundary payloads for evidence layer
        llm_response.provider_request = request.captured_request
        llm_response.model = model
        llm_response.provider_response = response.model_dump()
        return llm_response

    def _stream_wire(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None,
        kwargs: dict[str, Any],
        output_schema: dict[str, Any] | None,
        run_id: str | None,
        cache_key_prefix: str | None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """The events of one streaming call (one retry while nothing was yielded).

        Call it from the adapter's ``complete_stream`` body, so the request is
        built on the first pull, and close the result with ``aclosing``:
        closing it closes the upstream response.
        """
        request = self._wire_request(
            messages, tools, kwargs, output_schema, run_id, cache_key_prefix, stream=True
        )
        model = request.model
        client = self._sdk()
        clock = StreamClock(self._deadlines, provider=self._image_provider_name, model=model)
        return retry_unstarted_stream(
            functools.partial(self._stream_attempt, client, clock, request),
            provider=self._image_provider_name,
            model=model,
            run_id=run_id,
        )
