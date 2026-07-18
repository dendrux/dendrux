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
from typing import TYPE_CHECKING, Any

import httpx

from dendrux.llm.openai import OpenAIProvider

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from dendrux.types import LLMResponse, Message, StreamEvent, ToolDef

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Per-process catalog cache keyed by models URL. A failed fetch is cached as
# None so a flaky metadata endpoint costs at most one request per process —
# the guard degrades to a warning, it never breaks runs.
_catalog_cache: dict[str, dict[str, frozenset[str]] | None] = {}


async def _fetch_catalog(models_url: str) -> dict[str, frozenset[str]] | None:
    """Fetch {model_id -> supported_parameters} from OpenRouter, None on failure."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(models_url)
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:  # noqa: BLE001 — degrade path, never break runs
        logger.warning(
            "Could not fetch OpenRouter model catalog from %s (%s). "
            "Native-tool support cannot be verified; proceeding without the check.",
            models_url,
            exc,
        )
        return None
    catalog: dict[str, frozenset[str]] = {}
    for entry in payload.get("data", []):
        model_id = entry.get("id")
        params = entry.get("supported_parameters") or []
        if isinstance(model_id, str):
            catalog[model_id] = frozenset(p for p in params if isinstance(p, str))
    return catalog


async def _get_catalog(models_url: str) -> dict[str, frozenset[str]] | None:
    if models_url not in _catalog_cache:
        _catalog_cache[models_url] = await _fetch_catalog(models_url)
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
        supported = catalog.get(model)
        if supported is None:
            self._warn_once(
                model,
                "unknown-model",
                f"Model {model!r} not found in the OpenRouter catalog — cannot "
                f"verify native tool support. Proceeding.",
            )
            return
        if "tools" in supported:
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
