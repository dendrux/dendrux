"""Tests for the OpenRouter provider preset.

Covers the native-tools capability guard (the crux — an incapable model
returns text and silently never calls tools, which is indistinguishable
from a capable model choosing not to call one), the extra_body/routing
merge, attribution headers, and the dict-typed-arguments parser hardening.

All unit tests are fully mocked — no network.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock

import pytest

openai = pytest.importorskip("openai", reason="openai extra not installed")

import dendrux.llm.openrouter as openrouter_mod  # noqa: E402
from dendrux.llm._helpers import (  # noqa: E402
    parse_tool_json_lossy,
    parse_tool_json_strict,
)
from dendrux.llm.openai import OpenAIProvider  # noqa: E402
from dendrux.llm.openrouter import (  # noqa: E402
    OPENROUTER_BASE_URL,
    OpenRouterModel,
    OpenRouterProvider,
    _parse_model_entry,
)
from dendrux.types import Message, Role, StreamEventType, ToolDef, UsageStats  # noqa: E402


def _model(
    model_id: str,
    params: tuple[str, ...] = (),
    *,
    prompt_price: float | None = 0.000002,
    completion_price: float | None = 0.000006,
    input_modalities: tuple[str, ...] = ("text",),
) -> OpenRouterModel:
    return OpenRouterModel(
        id=model_id,
        name=model_id,
        context_length=131_072,
        supported_parameters=params,
        prompt_price=prompt_price,
        completion_price=completion_price,
        input_modalities=input_modalities,
        output_modalities=("text",),
    )


CATALOG = {
    "deepseek/deepseek-chat": _model(
        "deepseek/deepseek-chat", ("tools", "tool_choice", "temperature")
    ),
    "tiny/no-tools-model": _model("tiny/no-tools-model", ("temperature", "top_p")),
}


@pytest.fixture(autouse=True)
def _clear_catalog_cache() -> None:
    openrouter_mod._catalog_cache.clear()


@pytest.fixture
def catalog_ok(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    fetch = AsyncMock(return_value=CATALOG)
    monkeypatch.setattr(openrouter_mod, "_fetch_catalog", fetch)
    return fetch


def _provider(**kwargs: Any) -> OpenRouterProvider:
    kwargs.setdefault("model", "deepseek/deepseek-chat")
    kwargs.setdefault("api_key", "sk-or-test")
    return OpenRouterProvider(**kwargs)


def _tools() -> list[ToolDef]:
    return [ToolDef(name="echo", description="Echo", parameters={"type": "object"})]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
class TestConstruction:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            OpenRouterProvider(model="deepseek/deepseek-chat")

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
        provider = OpenRouterProvider(model="deepseek/deepseek-chat")
        assert provider.model == "deepseek/deepseek-chat"

    def test_base_url_is_openrouter(self) -> None:
        provider = _provider()
        assert str(provider._client.base_url).rstrip("/") == OPENROUTER_BASE_URL

    def test_attribution_headers(self) -> None:
        provider = _provider(app_url="https://myapp.example", app_name="MyApp")
        headers = provider._client.default_headers
        assert headers["HTTP-Referer"] == "https://myapp.example"
        assert headers["X-Title"] == "MyApp"

    def test_require_parameters_routing_default(self) -> None:
        provider = _provider()
        assert provider._extra_body == {"provider": {"require_parameters": True}}

    def test_user_provider_block_merges_over_default(self) -> None:
        provider = _provider(extra_body={"provider": {"order": ["deepseek"]}})
        assert provider._extra_body == {
            "provider": {"require_parameters": True, "order": ["deepseek"]}
        }

    def test_user_extra_body_keys_preserved(self) -> None:
        provider = _provider(extra_body={"transforms": ["middle-out"]})
        assert provider._extra_body is not None
        assert provider._extra_body["transforms"] == ["middle-out"]
        assert provider._extra_body["provider"] == {"require_parameters": True}

    def test_repr(self) -> None:
        assert repr(_provider()) == "OpenRouterProvider(model='deepseek/deepseek-chat')"


# ---------------------------------------------------------------------------
# Native-tools capability guard
# ---------------------------------------------------------------------------
class TestNativeToolsGuard:
    async def test_capable_model_passes(self, catalog_ok: AsyncMock) -> None:
        provider = _provider()
        await provider._ensure_native_tool_support("deepseek/deepseek-chat")
        assert "deepseek/deepseek-chat" in provider._tools_verified

    async def test_incapable_model_raises_by_default(self, catalog_ok: AsyncMock) -> None:
        provider = _provider(model="tiny/no-tools-model")
        with pytest.raises(ValueError, match="does not advertise native tool"):
            await provider._ensure_native_tool_support("tiny/no-tools-model")

    async def test_incapable_model_warns_when_opted_out(
        self, catalog_ok: AsyncMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = _provider(model="tiny/no-tools-model", require_native_tools=False)
        with caplog.at_level(logging.WARNING):
            await provider._ensure_native_tool_support("tiny/no-tools-model")
        assert "does not advertise native tool" in caplog.text

    async def test_warning_emitted_once_per_model(
        self, catalog_ok: AsyncMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = _provider(model="tiny/no-tools-model", require_native_tools=False)
        with caplog.at_level(logging.WARNING):
            await provider._ensure_native_tool_support("tiny/no-tools-model")
            await provider._ensure_native_tool_support("tiny/no-tools-model")
        assert caplog.text.count("does not advertise native tool") == 1

    async def test_catalog_fetch_failure_proceeds_with_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(
            openrouter_mod, "_fetch_catalog", AsyncMock(side_effect=ConnectionError("boom"))
        )
        provider = _provider()
        with caplog.at_level(logging.WARNING):
            await provider._ensure_native_tool_support("deepseek/deepseek-chat")
        assert "catalog" in caplog.text and "cannot verify" in caplog.text.lower()

    async def test_unknown_model_proceeds_with_warning(
        self, catalog_ok: AsyncMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = _provider(model="brand/new-model")
        with caplog.at_level(logging.WARNING):
            await provider._ensure_native_tool_support("brand/new-model")
        assert "not found in the OpenRouter catalog" in caplog.text

    async def test_catalog_fetched_once_per_process(self, catalog_ok: AsyncMock) -> None:
        provider = _provider()
        await provider._ensure_native_tool_support("deepseek/deepseek-chat")
        await provider._ensure_native_tool_support("deepseek/deepseek-chat")
        # Second provider instance shares the process-level cache.
        other = _provider()
        await other._ensure_native_tool_support("deepseek/deepseek-chat")
        assert catalog_ok.await_count == 1

    async def test_failed_fetch_cached_not_retried_per_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fetch = AsyncMock(side_effect=ConnectionError("boom"))
        monkeypatch.setattr(openrouter_mod, "_fetch_catalog", fetch)
        provider = _provider()
        await provider._ensure_native_tool_support("deepseek/deepseek-chat")
        await provider._ensure_native_tool_support("deepseek/deepseek-chat")
        assert fetch.await_count == 1


# ---------------------------------------------------------------------------
# Guard wiring in complete() / complete_stream()
# ---------------------------------------------------------------------------
class TestGuardWiring:
    async def test_complete_with_tools_raises_before_request(
        self, catalog_ok: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = _provider(model="tiny/no-tools-model")
        create = AsyncMock()
        monkeypatch.setattr(provider._client.chat.completions, "create", create)
        with pytest.raises(ValueError, match="does not advertise native tool"):
            await provider.complete([Message(role=Role.USER, content="hi")], _tools())
        create.assert_not_awaited()

    async def test_complete_without_tools_skips_guard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fetch = AsyncMock(return_value=CATALOG)
        monkeypatch.setattr(openrouter_mod, "_fetch_catalog", fetch)
        sentinel = object()
        monkeypatch.setattr(OpenAIProvider, "complete", AsyncMock(return_value=sentinel))
        provider = _provider(model="tiny/no-tools-model")  # incapable, but no tools passed
        result = await provider.complete([Message(role=Role.USER, content="hi")])
        assert result is sentinel
        fetch.assert_not_awaited()

    async def test_complete_stream_with_tools_raises_before_request(
        self, catalog_ok: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = _provider(model="tiny/no-tools-model")
        create = AsyncMock()
        monkeypatch.setattr(provider._client.chat.completions, "create", create)
        with pytest.raises(ValueError, match="does not advertise native tool"):
            stream = provider.complete_stream([Message(role=Role.USER, content="hi")], _tools())
            async for _ in stream:
                pass
        create.assert_not_awaited()

    async def test_agent_run_surfaces_guard_as_raised_error(self, catalog_ok: AsyncMock) -> None:
        """Full stack: agent.run() re-raises the guard's ValueError."""
        from dendrux import Agent, tool

        @tool()
        async def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        agent = Agent(
            provider=_provider(model="tiny/no-tools-model"),
            prompt="You are a calculator.",
            tools=[add],
        )
        async with agent:
            with pytest.raises(ValueError, match="does not advertise native tool"):
                await agent.run("What is 1 + 1?")

    async def test_agent_stream_surfaces_guard_as_run_error_event(
        self, catalog_ok: AsyncMock
    ) -> None:
        """Full stack: agent.stream() surfaces the guard in-band as run_error.

        Streams report failures as events, not exceptions — the guard message
        must arrive on the run_error event so stream consumers see it.
        """
        from dendrux import Agent, tool
        from dendrux.types import RunEventType

        @tool()
        async def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        agent = Agent(
            provider=_provider(model="tiny/no-tools-model"),
            prompt="You are a calculator.",
            tools=[add],
        )
        events = []
        async with agent:
            stream = agent.stream("What is 1 + 1?")
            async with stream:
                async for event in stream:
                    events.append(event)
        error_events = [e for e in events if e.type == RunEventType.RUN_ERROR]
        assert len(error_events) == 1
        assert "does not advertise native tool" in (error_events[0].error or "")

    async def test_guard_checks_per_call_model_override(self, catalog_ok: AsyncMock) -> None:
        """A per-call model= override is guarded, not the constructor model."""
        provider = _provider(model="deepseek/deepseek-chat")
        with pytest.raises(ValueError, match="does not advertise native tool"):
            await provider.complete(
                [Message(role=Role.USER, content="hi")],
                _tools(),
                model="tiny/no-tools-model",
            )


# ---------------------------------------------------------------------------
# Optional model — discovery-only construction
# ---------------------------------------------------------------------------
class TestOptionalModel:
    def test_construct_without_model(self) -> None:
        provider = OpenRouterProvider(api_key="sk-or-test")
        assert provider._model is None

    def test_repr_without_model(self) -> None:
        assert repr(OpenRouterProvider(api_key="sk-or-test")) == "OpenRouterProvider(model=None)"

    async def test_list_models_works_without_model(self, catalog_ok: AsyncMock) -> None:
        """The whole point: discovery needs no model."""
        provider = OpenRouterProvider(api_key="sk-or-test")
        models = await provider.list_models()
        assert {m.id for m in models} == set(CATALOG)

    async def test_complete_without_model_raises_before_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = OpenRouterProvider(api_key="sk-or-test")
        create = AsyncMock()
        monkeypatch.setattr(provider._client.chat.completions, "create", create)
        with pytest.raises(ValueError, match="without a model"):
            await provider.complete([Message(role=Role.USER, content="hi")])
        create.assert_not_awaited()

    async def test_complete_stream_without_model_raises_before_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = OpenRouterProvider(api_key="sk-or-test")
        create = AsyncMock()
        monkeypatch.setattr(provider._client.chat.completions, "create", create)
        with pytest.raises(ValueError, match="without a model"):
            stream = provider.complete_stream([Message(role=Role.USER, content="hi")])
            async for _ in stream:
                pass
        create.assert_not_awaited()

    async def test_per_call_model_enables_inference_without_constructor_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model= override lets a discovery-only provider run a one-off call."""
        sentinel = object()
        monkeypatch.setattr(OpenAIProvider, "complete", AsyncMock(return_value=sentinel))
        provider = OpenRouterProvider(api_key="sk-or-test")
        result = await provider.complete(
            [Message(role=Role.USER, content="hi")], model="deepseek/deepseek-chat"
        )
        assert result is sentinel


# ---------------------------------------------------------------------------
# extra_body flows into the request kwargs (OpenAIProvider passthrough)
# ---------------------------------------------------------------------------
class TestExtraBodyPassthrough:
    def test_constructor_extra_body_in_api_kwargs(self) -> None:
        provider = _provider()
        api_kwargs, captured = provider._build_api_kwargs(
            [Message(role=Role.USER, content="hi")], None, {}
        )
        assert api_kwargs["extra_body"] == {"provider": {"require_parameters": True}}
        assert captured["extra_body"] == api_kwargs["extra_body"]

    def test_per_call_extra_body_merges_over_constructor(self) -> None:
        provider = _provider()
        api_kwargs, _ = provider._build_api_kwargs(
            [Message(role=Role.USER, content="hi")],
            None,
            {"extra_body": {"transforms": ["middle-out"]}},
        )
        assert api_kwargs["extra_body"] == {
            "provider": {"require_parameters": True},
            "transforms": ["middle-out"],
        }

    def test_plain_openai_provider_omits_extra_body_by_default(self) -> None:
        provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
        api_kwargs, _ = provider._build_api_kwargs(
            [Message(role=Role.USER, content="hi")], None, {}
        )
        assert "extra_body" not in api_kwargs


# ---------------------------------------------------------------------------
# Parser hardening — dict-typed `arguments` from some upstreams
# ---------------------------------------------------------------------------
class TestDictArguments:
    def test_strict_accepts_dict(self) -> None:
        params = parse_tool_json_strict({"a": 1}, tool_name="add", call_id="c1")
        assert params == {"a": 1}

    def test_lossy_accepts_dict(self) -> None:
        params = parse_tool_json_lossy(
            {"a": 1}, provider="openrouter", model="m", tool_name="add", call_id="c1"
        )
        assert params == {"a": 1}

    def test_strict_still_parses_strings(self) -> None:
        assert parse_tool_json_strict('{"a": 1}', tool_name="add", call_id="c1") == {"a": 1}

    def test_strict_still_raises_on_malformed_string(self) -> None:
        with pytest.raises(ValueError, match="invalid JSON"):
            parse_tool_json_strict("{broken", tool_name="add", call_id="c1")


# ---------------------------------------------------------------------------
# list_models() — public catalog listing
# ---------------------------------------------------------------------------
class TestListModels:
    async def test_returns_typed_snapshots(self, catalog_ok: AsyncMock) -> None:
        provider = _provider()
        models = await provider.list_models()
        assert {m.id for m in models} == set(CATALOG)
        assert all(isinstance(m, OpenRouterModel) for m in models)

    async def test_served_from_shared_cache(self, catalog_ok: AsyncMock) -> None:
        """list_models and the guard share one fetch — they can never disagree."""
        provider = _provider()
        await provider._ensure_native_tool_support("deepseek/deepseek-chat")
        models = await provider.list_models()
        assert catalog_ok.await_count == 1
        picked = next(m for m in models if m.supports_tools)
        assert picked.id in provider._tools_verified or picked.id == "deepseek/deepseek-chat"

    async def test_refresh_forces_refetch(self, catalog_ok: AsyncMock) -> None:
        provider = _provider()
        await provider.list_models()
        await provider.list_models()
        assert catalog_ok.await_count == 1
        await provider.list_models(refresh=True)
        assert catalog_ok.await_count == 2

    async def test_raises_on_fetch_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unlike the guard's soft-warn, an explicit listing raises."""
        monkeypatch.setattr(
            openrouter_mod, "_fetch_catalog", AsyncMock(side_effect=ConnectionError("boom"))
        )
        provider = _provider()
        with pytest.raises(ConnectionError):
            await provider.list_models()

    async def test_refetches_after_guard_path_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cached guard-path failure (None) does not poison list_models."""
        fetch = AsyncMock(side_effect=[ConnectionError("boom"), CATALOG])
        monkeypatch.setattr(openrouter_mod, "_fetch_catalog", fetch)
        provider = _provider()
        await provider._ensure_native_tool_support("deepseek/deepseek-chat")  # caches None
        models = await provider.list_models()  # refetches instead of returning nothing
        assert {m.id for m in models} == set(CATALOG)

    def test_filtering_composes_with_plain_python(self) -> None:
        models = [
            _model("free/tools:free", ("tools",), prompt_price=0.0, completion_price=0.0),
            _model("paid/tools", ("tools",)),
            _model("free/chat-only:free", (), prompt_price=0.0, completion_price=0.0),
            _model("paid/vision", ("tools",), input_modalities=("text", "image")),
        ]
        assert [m.id for m in models if m.is_free and m.supports_tools] == ["free/tools:free"]
        assert [m.id for m in models if m.is_multimodal] == ["paid/vision"]
        assert [m.id for m in models if not m.is_multimodal and not m.is_free] == ["paid/tools"]


# ---------------------------------------------------------------------------
# OpenRouterModel properties
# ---------------------------------------------------------------------------
class TestModelProperties:
    def test_supports_tools(self) -> None:
        assert _model("a", ("tools",)).supports_tools
        assert not _model("b", ("temperature",)).supports_tools

    def test_is_free(self) -> None:
        assert _model("a", prompt_price=0.0, completion_price=0.0).is_free
        assert not _model("b").is_free
        # Unknown pricing is not "free"
        assert not _model("c", prompt_price=None, completion_price=None).is_free

    def test_is_multimodal(self) -> None:
        assert _model("a", input_modalities=("text", "image")).is_multimodal
        assert not _model("b", input_modalities=("text",)).is_multimodal
        # Unknown modalities are not claimed multimodal
        assert not _model("c", input_modalities=()).is_multimodal


# ---------------------------------------------------------------------------
# Catalog entry parsing
# ---------------------------------------------------------------------------
class TestParseModelEntry:
    def test_full_entry(self) -> None:
        model = _parse_model_entry(
            {
                "id": "deepseek/deepseek-chat",
                "name": "DeepSeek V3",
                "context_length": 163840,
                "pricing": {"prompt": "0.0000002", "completion": "0.0000008"},
                "supported_parameters": ["tools", "temperature"],
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                },
            }
        )
        assert model is not None
        assert model.name == "DeepSeek V3"
        assert model.context_length == 163840
        assert model.prompt_price == pytest.approx(2e-7)
        assert model.supports_tools and not model.is_free and not model.is_multimodal

    def test_free_variant(self) -> None:
        model = _parse_model_entry(
            {
                "id": "meta-llama/llama-3.3-70b-instruct:free",
                "pricing": {"prompt": "0", "completion": "0"},
                "supported_parameters": [],
            }
        )
        assert model is not None
        assert model.is_free
        assert model.name == "meta-llama/llama-3.3-70b-instruct:free"  # falls back to id

    def test_legacy_modality_string_fallback(self) -> None:
        model = _parse_model_entry(
            {"id": "x/vision", "architecture": {"modality": "text+image->text"}}
        )
        assert model is not None
        assert model.input_modalities == ("text", "image")
        assert model.output_modalities == ("text",)
        assert model.is_multimodal

    def test_missing_fields_degrade_to_none_or_empty(self) -> None:
        model = _parse_model_entry({"id": "bare/model"})
        assert model is not None
        assert model.context_length is None
        assert model.prompt_price is None
        assert model.input_modalities == ()

    def test_entry_without_id_skipped(self) -> None:
        assert _parse_model_entry({"name": "no id"}) is None

    def test_negative_price_sentinel_is_none(self) -> None:
        """OpenRouter reports '-1' for variable pricing (auto-routers)."""
        model = _parse_model_entry(
            {"id": "openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"}}
        )
        assert model is not None
        assert model.prompt_price is None
        assert model.completion_price is None
        assert not model.is_free


# ---------------------------------------------------------------------------
# Usage accounting — OpenRouter cost + cache-write mapping
# ---------------------------------------------------------------------------
def _usage(**fields: Any) -> Any:
    """A real OpenAI CompletionUsage carrying OpenRouter's extra fields.

    Built from the actual SDK type (``extra='allow'``) so the test exercises
    the same extra-field retention production relies on — not a stand-in that
    would pass regardless of whether the SDK keeps unknown fields.
    """
    from openai.types import CompletionUsage

    base: dict[str, Any] = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    base.update(fields)
    return CompletionUsage.model_validate(base)


@dataclass
class _FakeDelta:
    content: str | None = None
    tool_calls: Any = None


@dataclass
class _FakeStreamChoice:
    delta: _FakeDelta = field(default_factory=_FakeDelta)
    index: int = 0
    finish_reason: str | None = None


@dataclass
class _FakeChunk:
    choices: list[Any] = field(default_factory=list)
    usage: Any = None


@dataclass
class _FakeMessage:
    role: str = "assistant"
    content: str | None = "ok"
    tool_calls: Any = None


@dataclass
class _FakeChoice:
    message: _FakeMessage = field(default_factory=_FakeMessage)
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class _FakeCompletion:
    choices: list[Any] = field(default_factory=lambda: [_FakeChoice()])
    usage: Any = None

    def model_dump(self) -> dict[str, Any]:
        return {"fake": True}


async def _astream(chunks: list[Any]) -> Any:
    for chunk in chunks:
        yield chunk


class TestUsageAccounting:
    """OpenRouter maps its extra usage fields onto the cross-provider UsageStats.

    ``cost`` -> ``cost_usd`` (USD; what OpenRouter charged the account) and
    ``prompt_tokens_details.cache_write_tokens`` -> ``cache_creation_input_tokens``.
    Populating these at the source is enough — the run-level accumulator and
    RunStore already carry both.
    """

    def test_cost_mapped_to_cost_usd(self) -> None:
        stats = _provider()._normalize_usage(_usage(cost=0.000123))
        assert stats.cost_usd == pytest.approx(0.000123)

    def test_cache_write_mapped_to_cache_creation(self) -> None:
        stats = _provider()._normalize_usage(
            _usage(prompt_tokens_details={"cached_tokens": 40, "cache_write_tokens": 10})
        )
        assert stats.cache_creation_input_tokens == 10
        # base mapping still holds: cached-read + fresh-input normalization
        assert stats.cache_read_input_tokens == 40
        assert stats.input_tokens == 60

    def test_missing_cost_leaves_none(self) -> None:
        """A compatible backend that omits cost must not fabricate a number."""
        stats = _provider()._normalize_usage(_usage())
        assert stats.cost_usd is None
        assert stats.cache_creation_input_tokens is None

    def test_zero_cost_preserved(self) -> None:
        """Free models report cost 0.0 — a reported zero survives, not -> None."""
        stats = _provider()._normalize_usage(_usage(cost=0.0))
        assert stats.cost_usd == 0.0

    def test_base_reasoning_tokens_preserved_through_override(self) -> None:
        stats = _provider()._normalize_usage(
            _usage(completion_tokens_details={"reasoning_tokens": 5})
        )
        assert stats.reasoning_tokens == 5

    async def test_non_streaming_complete_surfaces_cost(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = _provider()
        completion = _FakeCompletion(
            usage=_usage(cost=0.0005, prompt_tokens_details={"cache_write_tokens": 8})
        )
        monkeypatch.setattr(
            provider._client.chat.completions, "create", AsyncMock(return_value=completion)
        )
        result = await provider.complete([Message(role=Role.USER, content="hi")])
        assert result.usage.cost_usd == pytest.approx(0.0005)
        assert result.usage.cache_creation_input_tokens == 8

    async def test_streaming_final_chunk_surfaces_cost(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = _provider()
        chunks = [
            _FakeChunk(choices=[_FakeStreamChoice(delta=_FakeDelta(content="hi"))]),
            # OpenRouter attaches usage to the final SSE chunk (empty choices)
            _FakeChunk(
                choices=[],
                usage=_usage(cost=0.0009, prompt_tokens_details={"cache_write_tokens": 3}),
            ),
        ]
        monkeypatch.setattr(
            provider._client.chat.completions,
            "create",
            AsyncMock(return_value=_astream(chunks)),
        )
        done = None
        async for event in provider.complete_stream([Message(role=Role.USER, content="hi")]):
            if event.type == StreamEventType.DONE:
                done = event
        assert done is not None
        assert done.raw.usage.cost_usd == pytest.approx(0.0009)
        assert done.raw.usage.cache_creation_input_tokens == 3

    def test_multi_iteration_aggregation(self) -> None:
        """Per-call cost + cache-write sum into the run-level total."""
        from dendrux.loops.react import _accumulate_usage

        provider = _provider()
        total = UsageStats()
        _accumulate_usage(
            total,
            provider._normalize_usage(
                _usage(cost=0.001, prompt_tokens_details={"cache_write_tokens": 10})
            ),
        )
        _accumulate_usage(
            total,
            provider._normalize_usage(
                _usage(cost=0.002, prompt_tokens_details={"cache_write_tokens": 5})
            ),
        )
        assert total.cost_usd == pytest.approx(0.003)
        assert total.cache_creation_input_tokens == 15
