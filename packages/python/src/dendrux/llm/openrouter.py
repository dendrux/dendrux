"""OpenRouter provider preset.

OpenRouter is OpenAI-Chat-Completions wire-compatible, so this is a thin
preset over :class:`OpenAIProvider` — same transport, same conversion layer,
no extra SDK. It adds:

  - The OpenRouter base URL and ``OPENROUTER_API_KEY`` resolution
  - Optional attribution headers (``HTTP-Referer`` / ``X-Title``)
  - ``provider.require_parameters=true`` routing by default, so OpenRouter
    only routes to upstreams that honor the parameters we send (notably
    ``tools`` — the same model slug fans out to upstreams with different
    tool fidelity)
  - A call-time native-tools capability guard (see below)

**The native-tools guard.** A model that lacks native function calling does
not error when handed tools — it returns text and silently never calls them.
That cannot be detected from a response (a capable model choosing not to
call a tool looks identical), so the guard keys on OpenRouter's per-model
``supported_parameters`` metadata instead. It fires only when ``tools`` are
actually passed — tool-free use of any catalog model is unrestricted — and
raises by default; pass ``require_native_tools=False`` to downgrade to a
warning. Metadata fetch failures and unknown slugs degrade to a soft
warning, never a broken run.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

from dendrux.llm.openai import OpenAIProvider

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from dendrux.types import LLMResponse, Message, StreamEvent, ToolDef

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True, slots=True)
class OpenRouterModel:
    """A model in the OpenRouter catalog — an immutable snapshot.

    Rich typed data instead of filter flags: compose any selection with
    plain Python, e.g.
    ``[m for m in models if m.is_free and m.supports_tools]``.
    """

    id: str
    """OpenRouter model slug, e.g. ``"deepseek/deepseek-chat"``."""
    name: str
    """Human-readable name, e.g. ``"DeepSeek V3"``."""
    context_length: int | None
    """Maximum context window in tokens, when advertised."""
    supported_parameters: tuple[str, ...]
    """Request parameters the model supports (``"tools"``, ``"temperature"``, ...)."""
    prompt_price: float | None
    """Input price in USD per token; ``0.0`` for free models, None if unparseable."""
    completion_price: float | None
    """Output price in USD per token; ``0.0`` for free models, None if unparseable."""
    input_modalities: tuple[str, ...]
    """Accepted input modalities (``"text"``, ``"image"``, ...); empty if unknown."""
    output_modalities: tuple[str, ...]
    """Produced output modalities; empty if unknown."""

    @property
    def supports_tools(self) -> bool:
        """True when the model advertises native function calling."""
        return "tools" in self.supported_parameters

    @property
    def is_free(self) -> bool:
        """True when both prompt and completion prices are zero."""
        return self.prompt_price == 0 and self.completion_price == 0

    @property
    def is_multimodal(self) -> bool:
        """True when the model accepts any non-text input (image, audio, ...)."""
        return any(m != "text" for m in self.input_modalities)


def _parse_price(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_modalities(arch: dict[str, Any], key: str, side: int) -> tuple[str, ...]:
    """Read modalities from architecture metadata.

    Prefers the explicit ``input_modalities``/``output_modalities`` lists;
    falls back to parsing the legacy ``modality`` string
    (e.g. ``"text+image->text"``, side 0 = input, 1 = output).
    """
    explicit = arch.get(key)
    if isinstance(explicit, list):
        return tuple(m for m in explicit if isinstance(m, str))
    modality = arch.get("modality")
    if isinstance(modality, str) and "->" in modality:
        parts = modality.split("->")
        if len(parts) == 2:
            return tuple(m.strip() for m in parts[side].split("+") if m.strip())
    return ()


def _parse_model_entry(entry: dict[str, Any]) -> OpenRouterModel | None:
    model_id = entry.get("id")
    if not isinstance(model_id, str):
        return None
    params = entry.get("supported_parameters") or []
    pricing = entry.get("pricing") or {}
    arch = entry.get("architecture") or {}
    context_length = entry.get("context_length")
    return OpenRouterModel(
        id=model_id,
        name=entry.get("name") or model_id,
        context_length=int(context_length) if isinstance(context_length, int | float) else None,
        supported_parameters=tuple(p for p in params if isinstance(p, str)),
        prompt_price=_parse_price(pricing.get("prompt")),
        completion_price=_parse_price(pricing.get("completion")),
        input_modalities=_parse_modalities(arch, "input_modalities", 0),
        output_modalities=_parse_modalities(arch, "output_modalities", 1),
    )


# Per-process catalog cache keyed by models URL — one fetch serves both the
# native-tools guard and list_models(), so they can never disagree. A failed
# fetch is cached as None so a flaky metadata endpoint costs at most one
# request per process on the guard path (which degrades to a warning, never
# a broken run); list_models() refetches and raises instead.
_catalog_cache: dict[str, dict[str, OpenRouterModel] | None] = {}


async def _fetch_catalog(models_url: str) -> dict[str, OpenRouterModel]:
    """Fetch {model_id -> OpenRouterModel} from OpenRouter. Raises on failure."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(models_url)
        resp.raise_for_status()
        payload = resp.json()
    catalog: dict[str, OpenRouterModel] = {}
    for entry in payload.get("data", []):
        model = _parse_model_entry(entry)
        if model is not None:
            catalog[model.id] = model
    return catalog


async def _get_catalog(models_url: str) -> dict[str, OpenRouterModel] | None:
    """Cached catalog for the guard path — swallows fetch failures as None."""
    if models_url not in _catalog_cache:
        try:
            _catalog_cache[models_url] = await _fetch_catalog(models_url)
        except Exception as exc:  # noqa: BLE001 — degrade path, never break runs
            logger.warning(
                "Could not fetch OpenRouter model catalog from %s (%s). "
                "Native-tool support cannot be verified; proceeding without the check.",
                models_url,
                exc,
            )
            _catalog_cache[models_url] = None
    return _catalog_cache[models_url]


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter provider — open-source and premium models via one API.

    Usage:
        provider = OpenRouterProvider(model="deepseek/deepseek-chat")

        # With attribution (shows up on openrouter.ai rankings)
        provider = OpenRouterProvider(
            model="meta-llama/llama-3.3-70b-instruct",
            app_url="https://myapp.example",
            app_name="MyApp",
        )

    Accepts all :class:`OpenAIProvider` keyword arguments (``max_tokens``,
    ``temperature``, ``timeout``, ``extra_body``, ...) as passthrough.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        app_url: str | None = None,
        app_name: str | None = None,
        require_native_tools: bool = True,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Create an OpenRouter provider.

        Args:
            model: OpenRouter model slug (e.g. "deepseek/deepseek-chat",
                "meta-llama/llama-3.3-70b-instruct").
            api_key: OpenRouter API key. Defaults to OPENROUTER_API_KEY env var.
            app_url: Optional attribution URL, sent as ``HTTP-Referer``.
            app_name: Optional attribution name, sent as ``X-Title``.
            require_native_tools: When True (default), passing tools to a model
                whose catalog metadata lacks native tool support raises instead
                of silently returning text that ignores the tools. Set False to
                downgrade to a warning.
            extra_body: Extra JSON fields for the request body. Merged with the
                default ``provider`` routing block; keys you set win, and your
                ``provider`` entries merge over ``require_parameters=True``.
            **kwargs: Passed through to :class:`OpenAIProvider` (max_tokens,
                temperature, timeout, max_retries, ...).
        """
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouterProvider needs an API key. Pass api_key='sk-or-...' "
                "or set the OPENROUTER_API_KEY environment variable "
                "(create one at https://openrouter.ai/keys)."
            )

        headers: dict[str, str] = {}
        if app_url:
            headers["HTTP-Referer"] = app_url
        if app_name:
            headers["X-Title"] = app_name

        # Only route to upstreams that honor the parameters we send — without
        # this, OpenRouter may fall back to an upstream that drops `tools`.
        user_provider_block = (extra_body or {}).get("provider") or {}
        merged_extra_body = {
            **(extra_body or {}),
            "provider": {"require_parameters": True, **user_provider_block},
        }

        base_url = kwargs.pop("base_url", OPENROUTER_BASE_URL)
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            default_headers=headers or None,
            extra_body=merged_extra_body,
            **kwargs,
        )
        self._require_native_tools = require_native_tools
        self._models_url = base_url.rstrip("/") + "/models"
        self._tools_verified: set[str] = set()
        self._tool_warnings_emitted: set[tuple[str, str]] = set()

    def __repr__(self) -> str:
        return f"OpenRouterProvider(model={self._model!r})"

    async def list_models(self, *, refresh: bool = False) -> list[OpenRouterModel]:
        """List the OpenRouter model catalog as typed snapshots.

        Serves from the same per-process cache the native-tools guard uses,
        so a model picked from this list is guaranteed to pass the guard.
        Pass ``refresh=True`` to force a refetch — anything beyond
        process-lifetime caching (TTLs, persistence) is application policy
        and stays in your hands.

        Unlike the guard (which degrades to a warning), this raises on fetch
        failure — an explicit listing request deserves a real error, not an
        empty catalog.

        Usage:
            models = await provider.list_models()
            free_tool_models = [m for m in models if m.is_free and m.supports_tools]
            text_only = [m for m in models if not m.is_multimodal]

        Returns:
            All catalog models as :class:`OpenRouterModel` snapshots.
        """
        catalog = _catalog_cache.get(self._models_url)
        if refresh or catalog is None:
            catalog = await _fetch_catalog(self._models_url)
            _catalog_cache[self._models_url] = catalog
        return list(catalog.values())

    def _warn_once(self, model: str, reason: str, message: str) -> None:
        key = (model, reason)
        if key not in self._tool_warnings_emitted:
            self._tool_warnings_emitted.add(key)
            logger.warning(message)

    async def _ensure_native_tool_support(self, model: str) -> None:
        """Verify the model advertises native tools; raise or warn if not.

        Keys on catalog metadata, never on an observed response — an empty
        ``tool_calls`` from a capable model (chose not to call) is
        indistinguishable from one emitted by an incapable model.
        """
        if model in self._tools_verified:
            return
        catalog = await _get_catalog(self._models_url)
        if catalog is None:
            self._warn_once(
                model,
                "catalog-unavailable",
                f"OpenRouter model catalog unavailable — cannot verify that "
                f"{model!r} supports native tool calling. Proceeding.",
            )
            return
        entry = catalog.get(model)
        if entry is None:
            self._warn_once(
                model,
                "unknown-model",
                f"Model {model!r} not found in the OpenRouter catalog — cannot "
                f"verify native tool support. Proceeding.",
            )
            return
        if entry.supports_tools:
            self._tools_verified.add(model)
            return
        message = (
            f"Model {model!r} (via OpenRouter) does not advertise native tool "
            f"support (supported_parameters has no 'tools'). It will return "
            f"plain text and never call your tools. Choose a tool-capable "
            f"model (see https://openrouter.ai/models?supported_parameters=tools), "
            f"or run this agent without tools. Pass require_native_tools=False "
            f"to downgrade this error to a warning."
        )
        if self._require_native_tools:
            raise ValueError(message)
        self._warn_once(model, "no-native-tools", message)

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
        """Send messages via OpenRouter; guards native tool support when tools are passed."""
        if tools:
            await self._ensure_native_tool_support(kwargs.get("model", self._model))
        return await super().complete(
            messages,
            tools,
            output_schema=output_schema,
            run_id=run_id,
            cache_key_prefix=cache_key_prefix,
            **kwargs,
        )

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
        """Stream via OpenRouter; guards native tool support when tools are passed."""
        if tools:
            await self._ensure_native_tool_support(kwargs.get("model", self._model))
        async for event in super().complete_stream(
            messages,
            tools,
            output_schema=output_schema,
            run_id=run_id,
            cache_key_prefix=cache_key_prefix,
            **kwargs,
        ):
            yield event
