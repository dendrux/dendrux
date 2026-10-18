"""Live provider matrix for 0.4.0 image input (spec section 9.3).

Pass rule: the model's answer must contain the code word drawn in the fixture
image (and the colour where one is asked for). An HTTP 200 alone never passes.

Fixtures (tests/fixtures/images/):
    zebra42.png  white "ZEBRA42" on a red background
    kiwi77.png   white "KIWI77" on a green background

Scenarios
    S1   user image
    S2   server tool returns an image
    S3   client tool image, pause, resume in a fresh Agent and engine
    S4   cache: iteration 2 and a runs=[R] follow-up read the stable prefix (>= 5k tokens)
    S5   two images in one message are told apart
    S7   drop policy advances (Anthropic)
    S8a  text-only model, new image: typed error before any completion request
    S8b  text-only model forced to "yes": provider 404 becomes a typed error
    S8c  history image replayed into a text-only model becomes a placeholder
    S9a  Anthropic many-image dimension rule (no request is sent)
    L06  text-only two-call run (cache, usage and request-hash baseline for the v0.3.0 pairing)

Run (the coordinator does this; keys come from the environment only):
    DENDRUX_LIVE=1 DENDRUX_LIVE_RESULTS=/path/results.json \
        pytest tests/live/test_images_live.py -v

Select with -k, e.g. -k "anthropic and S4" or -k "S8".
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from dendrux import Agent, Image, ToolOutput, tool
from dendrux.errors import ImageLimitError, UnsupportedModalityError
from dendrux.media import ImageDropPolicy, ImagePolicy
from dendrux.store import LLMCall, RunStore
from dendrux.types import RunStatus, ToolResult

from .conftest import scrub

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from dendrux.llm.base import LLMProvider

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "images"

DEEPSEEK = "deepseek/deepseek-chat"
INLINE_NEEDLES = ("data:image", "iVBORw0K", "/9j/4AAQ", "UklGR", "R0lGOD")

SHORT_PROMPT = "You read images carefully and answer briefly and exactly as asked."


# --------------------------------------------------------------------------- routes


@dataclass(frozen=True)
class Route:
    route_id: str
    family: str  # anthropic | openai-chat | openai-responses | openrouter
    model: str
    key_env: str
    mode: str = ""  # responses: native | portable; openrouter: vision | text
    strict_cache: bool = False  # cache reads are deterministic on this route

    def make_provider(self) -> LLMProvider:
        if self.family == "anthropic":
            from dendrux.llm.anthropic import AnthropicProvider

            return AnthropicProvider(model=self.model)
        if self.family == "openai-chat":
            from dendrux.llm.openai import OpenAIProvider

            return OpenAIProvider(model=self.model)
        if self.family == "openai-responses":
            from dendrux.llm.openai_responses import OpenAIResponsesProvider

            return OpenAIResponsesProvider(
                model=self.model,
                tool_result_images="native" if self.mode == "native" else "portable",
            )
        from dendrux.llm.openrouter import OpenRouterProvider

        if self.mode == "vision":
            return OpenRouterProvider(
                model=self.model,
                extra_body={"provider": {"order": ["anthropic"], "allow_fallbacks": False}},
            )
        return OpenRouterProvider(model=self.model)


SONNET = Route(
    "anthropic/sonnet-4-6", "anthropic", "claude-sonnet-4-6", "ANTHROPIC_API_KEY", "", True
)
HAIKU = Route("anthropic/haiku-4-5", "anthropic", "claude-haiku-4-5", "ANTHROPIC_API_KEY", "", True)
CHAT_5 = Route("openai-chat/gpt-5-mini", "openai-chat", "gpt-5-mini", "OPENAI_API_KEY")
CHAT_4O = Route("openai-chat/gpt-4o-mini", "openai-chat", "gpt-4o-mini", "OPENAI_API_KEY")
RESP_N5 = Route(
    "openai-responses-native/gpt-5-mini",
    "openai-responses",
    "gpt-5-mini",
    "OPENAI_API_KEY",
    "native",
)
RESP_N4 = Route(
    "openai-responses-native/gpt-4o-mini",
    "openai-responses",
    "gpt-4o-mini",
    "OPENAI_API_KEY",
    "native",
)
RESP_P5 = Route(
    "openai-responses-portable/gpt-5-mini",
    "openai-responses",
    "gpt-5-mini",
    "OPENAI_API_KEY",
    "portable",
)
RESP_P4 = Route(
    "openai-responses-portable/gpt-4o-mini",
    "openai-responses",
    "gpt-4o-mini",
    "OPENAI_API_KEY",
    "portable",
)
OR_VISION = Route(
    "openrouter-vision/claude-haiku-4.5",
    "openrouter",
    "anthropic/claude-haiku-4.5",
    "OPENROUTER_API_KEY",
    "vision",
    True,
)
OR_TEXT = Route(
    "openrouter-text/deepseek-chat", "openrouter", DEEPSEEK, "OPENROUTER_API_KEY", "text"
)


@dataclass(frozen=True)
class Case:
    route: Route
    scenario: str

    @property
    def family(self) -> str:
        return self.route.family

    @property
    def route_id(self) -> str:
        return self.route.route_id

    @property
    def model(self) -> str:
        return self.route.model

    @property
    def id(self) -> str:
        return f"{self.route.route_id}::{self.scenario}"


def _cases() -> list[Case]:
    matrix: list[tuple[Route, tuple[str, ...]]] = [
        (SONNET, ("S1", "S2", "S3", "S4", "S5", "S7", "S9a", "L06")),
        (HAIKU, ("S1", "S2", "S3", "S4", "S5", "S7", "S9a", "L06")),
        (CHAT_5, ("S1", "S2", "S3", "S4", "S5", "L06")),
        (CHAT_4O, ("S1", "S2", "S3", "S4", "S5", "L06")),
        (RESP_N5, ("S1", "S2", "S3", "S4", "S5", "L06")),
        (RESP_N4, ("S1", "S2", "S3", "S4", "S5", "L06")),
        (RESP_P5, ("S2", "S3")),
        (RESP_P4, ("S2", "S3")),
        (OR_VISION, ("S1", "S2", "S3", "S4", "L06")),
        (OR_TEXT, ("S8a", "S8b", "S8c")),
    ]
    return [Case(route, scenario) for route, scenarios in matrix for scenario in scenarios]


# --------------------------------------------------------------------------- tools


def img(name: str) -> Image:
    return Image.from_bytes((FIXTURES / name).read_bytes())


@tool()
async def show_card() -> ToolOutput:
    """Show the card. Returns an image with a code word on it."""
    return ToolOutput("Card rendered.", images=[img("kiwi77.png")])


CARD_SEQUENCE = ("kiwi77.png", "zebra42.png", "zebra42.jpg", "zebra42.webp")


@tool()
async def next_card(step: int) -> ToolOutput:
    """Show card number `step` (1 to 4).

    Args:
        step: The card number, starting at 1.
    """
    index = max(1, min(step, len(CARD_SEQUENCE))) - 1
    if step < len(CARD_SEQUENCE):
        text = f"Card {step} of {len(CARD_SEQUENCE)} shown. Call next_card with step={step + 1}."
    else:
        text = f"Card {step} of {len(CARD_SEQUENCE)} shown. That was the last card."
    return ToolOutput(text, images=[img(CARD_SEQUENCE[index])])


@tool(target="client")
async def take_screenshot() -> str:
    """Capture the user's screen. Runs in the client app."""
    return ""


@tool()
async def lookup(key: str) -> str:
    """Look up a reference value.

    Args:
        key: The reference key.
    """
    return f"value-for-{key}: 4711"


# --------------------------------------------------------------------------- helpers


def norm(text: str | None) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text or "").upper()


def expect_word(answer: str | None, word: str) -> None:
    assert answer, "empty answer"
    assert norm(word) in norm(answer), f"code word {word} not in answer: {answer[:200]!r}"


def expect_colour(answer: str | None, colour: str) -> None:
    assert answer and colour in answer.lower(), f"colour {colour} not in answer: {answer!r}"


def long_prompt(nonce: str) -> str:
    """A stable system prompt of roughly 9k tokens, unique per test so call 1 writes the cache."""
    lines = [
        f"Rule {i:03d}: When handling case number {i}, check the ledger, confirm the "
        "customer reference, and record the outcome in the audit log."
        for i in range(1, 301)
    ]
    return (
        f"[session {nonce}] You are a careful assistant. Follow the tools' instructions "
        "exactly and answer briefly. Reference rules follow; none of them change how you "
        "answer the user.\n" + "\n".join(lines)
    )


def assert_clean_exception(exc: BaseException) -> None:
    """No chained exceptions and no image bytes anywhere reachable from the error."""
    assert exc.__cause__ is None, "error has a __cause__"
    assert exc.__context__ is None, "error has a __context__"
    blob = " ".join([str(exc), repr(exc), repr(exc.args), repr(vars(exc))])
    for needle in INLINE_NEEDLES:
        assert needle not in blob, f"image bytes reachable from the error ({needle})"


class Ctx:
    def __init__(self, case: Case, tmp_path: Path, rec: dict[str, Any]) -> None:
        self.case = case
        self.rec = rec
        self.url = f"sqlite+aiosqlite:///{tmp_path / 'live.db'}"
        self.nonce = uuid.uuid4().hex[:10]
        self.rec.setdefault("calls", [])

    def agent(
        self,
        *,
        tools: list[Any] | None = None,
        prompt: str = SHORT_PROMPT,
        images: ImagePolicy | None = None,
        provider: LLMProvider | None = None,
        name: str = "LiveImages",
    ) -> Agent:
        return Agent(
            name=name,
            provider=provider or self.case.route.make_provider(),
            prompt=prompt,
            tools=tools or [],
            database_url=self.url,
            images=images,
        )

    def store(self) -> RunStore:
        return RunStore.from_database_url(self.url)

    async def audit(self, run_id: str, label: str = "run") -> list[LLMCall]:
        """Per-call usage and request hash; fail on inline bytes or more than 4 cache markers."""
        async with self.store() as store:
            calls = await store.get_llm_calls(run_id)
        for call in calls:
            request = call.provider_request or {}
            text = json.dumps(request, sort_keys=True, default=str)
            for needle in INLINE_NEEDLES:
                assert needle not in text, (
                    f"inline image bytes ({needle}) in provider_request: {label} #{call.iteration}"
                )
            if self.case.family == "anthropic":
                markers = text.count('"cache_control"')
                assert markers <= 4, (
                    f"{markers} cache_control markers, {label} call {call.iteration}"
                )
            response = call.provider_response or {}
            self.rec["calls"].append(
                {
                    "label": label,
                    "run_id": run_id,
                    "iteration": call.iteration,
                    "model": call.model,
                    "input_tokens": call.input_tokens,
                    "output_tokens": call.output_tokens,
                    "cache_read_input_tokens": call.cache_read_input_tokens,
                    "cache_creation_input_tokens": call.cache_creation_input_tokens,
                    "reasoning_tokens": call.reasoning_tokens,
                    "request_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "prompt_cache_key": request.get("prompt_cache_key"),
                    "upstream": response.get("provider") if isinstance(response, dict) else None,
                }
            )
        upstream = {
            str(c["upstream"]).lower()
            for c in self.rec["calls"]
            if c["run_id"] == run_id and c.get("upstream")
        }
        if (
            self.case.route.mode == "vision"
            and upstream
            and not any("anthropic" in u for u in upstream)
        ):
            pytest.skip(
                f"OpenRouter routing was not pinned to anthropic (served by {sorted(upstream)})"
            )
        return calls

    async def events(self, run_id: str, kinds: list[str]) -> list[Any]:
        async with self.store() as store:
            return list(await store.get_events(run_id, event_types=kinds))


# --------------------------------------------------------------------------- scenarios


async def s1_user_image(ctx: Ctx) -> None:
    async with ctx.agent() as agent:
        result = await agent.run(
            [
                "What code word is written in this image, and what is its background colour? "
                "Answer as: WORD, colour.",
                img("zebra42.png"),
            ]
        )
    assert result.status is RunStatus.SUCCESS, result.error
    expect_word(result.answer, "ZEBRA42")
    expect_colour(result.answer, "red")
    ctx.rec["answer"] = (result.answer or "")[:300]
    await ctx.audit(result.run_id)


async def s2_server_tool_image(ctx: Ctx) -> None:
    async with ctx.agent(tools=[show_card]) as agent:
        result = await agent.run(
            "Call show_card once. Then tell me the code word in the image it returns and "
            "the image's background colour. Answer as: WORD, colour."
        )
    assert result.status is RunStatus.SUCCESS, result.error
    expect_word(result.answer, "KIWI77")
    expect_colour(result.answer, "green")
    completed = await ctx.events(result.run_id, ["tool.completed"])
    assert any((e.data or {}).get("image_count") == 1 for e in completed), "no tool image recorded"
    ctx.rec["answer"] = (result.answer or "")[:300]
    await ctx.audit(result.run_id)


async def s3_client_tool_resume(ctx: Ctx) -> None:
    tenant = "live-acme"
    async with ctx.agent(tools=[take_screenshot]) as first:
        paused = await first.run(
            "Take a screenshot and tell me the code word shown in it.", tenant_id=tenant
        )
    assert paused.status is RunStatus.WAITING_CLIENT_TOOL, paused.status
    async with ctx.store() as store:
        call = (await store.get_pauses(paused.run_id))[-1].pending_tool_calls[0]

    # A different Agent over a new engine, as a separate worker would do.
    async with ctx.agent(tools=[take_screenshot]) as second:
        result = await second.submit_tool_results(
            paused.run_id,
            [
                ToolResult(
                    name=call["name"],
                    call_id=call["id"],
                    payload=json.dumps("Captured."),
                    images=(img("zebra42.png"),),
                )
            ],
        )
    assert result.status is RunStatus.SUCCESS, result.error
    expect_word(result.answer, "ZEBRA42")
    resumed = await ctx.events(paused.run_id, ["run.resumed"])
    assert any(
        (r.get("image_count") == 1)
        for e in resumed
        for r in (e.data or {}).get("submitted_results", [])
    ), "run.resumed did not record the image"
    ctx.rec["answer"] = (result.answer or "")[:300]
    await ctx.audit(paused.run_id)


async def s4_cache(ctx: Ctx) -> None:
    route = ctx.case.route
    strict = route.strict_cache
    prompt = long_prompt(ctx.nonce)
    question = (
        "The user's image is attached. Call show_card once, then answer as: "
        "user=<WORD> card=<WORD>, using the code words in the user's image and in the card."
    )
    first_run: str | None = None
    iter2_read = 0
    for attempt in range(1 if strict else 2):
        async with ctx.agent(tools=[show_card], prompt=prompt) as agent:
            result = await agent.run([question, img("zebra42.png")])
        assert result.status is RunStatus.SUCCESS, result.error
        first_run = result.run_id
        expect_word(result.answer, "ZEBRA42")
        expect_word(result.answer, "KIWI77")
        calls = await ctx.audit(result.run_id, label=f"s4-attempt{attempt + 1}")
        assert len(calls) >= 2, "expected a tool iteration and an answer iteration"
        if attempt == 0 and strict and calls[0].cache_creation_input_tokens is not None:
            assert calls[0].cache_creation_input_tokens > 0, "call 1 wrote nothing to the cache"
        iter2_read = calls[1].cache_read_input_tokens or 0
        if iter2_read > 0:
            break
    assert iter2_read > 0, "iteration 2 read nothing from the cache"
    assert first_run is not None

    async with ctx.agent(tools=[show_card], prompt=prompt) as agent:
        follow = await agent.run(
            "Which code word did my image show? Answer with the word only.",
            runs=[first_run],
            replay_last="all",
        )
    assert follow.status is RunStatus.SUCCESS, follow.error
    expect_word(follow.answer, "ZEBRA42")
    follow_calls = await ctx.audit(follow.run_id, label="s4-followup")
    follow_read = follow_calls[0].cache_read_input_tokens or 0
    ctx.rec["follow_up_first_call_cache_read"] = follow_read
    if strict:
        assert follow_read > 0, (
            "the runs=[R] follow-up read nothing from the cache on its first call"
        )

    if ctx.case.family in ("openai-chat", "openai-responses"):
        keys = {c["prompt_cache_key"] for c in ctx.rec["calls"] if c["prompt_cache_key"]}
        sha = hashlib.sha256((FIXTURES / "zebra42.png").read_bytes()).hexdigest()
        assert len(keys) <= 2, f"prompt_cache_key varied across calls: {keys}"
        for key in keys:
            assert sha[:12] not in key and "img_" not in key, "image identity in prompt_cache_key"


async def s5_two_images(ctx: Ctx) -> None:
    async with ctx.agent() as agent:
        result = await agent.run(
            [
                "Each of these two images has a code word and a background colour. Answer "
                "exactly as: first=<WORD> <colour>, second=<WORD> <colour>.",
                img("zebra42.png"),
                img("kiwi77.png"),
            ]
        )
    assert result.status is RunStatus.SUCCESS, result.error
    answer = result.answer or ""
    expect_word(answer, "ZEBRA42")
    expect_word(answer, "KIWI77")
    assert norm(answer).index("ZEBRA42") < norm(answer).index("KIWI77"), f"order lost: {answer!r}"
    lowered = answer.lower()
    assert (
        "red" in lowered and "green" in lowered and lowered.index("red") < lowered.index("green")
    ), f"colours not matched to images: {answer!r}"
    ctx.rec["answer"] = answer[:300]
    await ctx.audit(result.run_id)


async def s7_drop_advance(ctx: Ctx) -> None:
    policy = ImagePolicy(drop=ImageDropPolicy(keep_recent=1, interval=1))
    async with ctx.agent(tools=[next_card], prompt=long_prompt(ctx.nonce), images=policy) as agent:
        result = await agent.run(
            "Call next_card with step=1 and keep calling it with the next step, one call per "
            "turn, waiting for each result, until the last card. Then give the code word of "
            "the LAST card only.",
        )
    assert result.status is RunStatus.SUCCESS, result.error
    expect_word(result.answer, "ZEBRA42")
    changed = await ctx.events(result.run_id, ["image.view_changed"])
    assert changed, "drop policy never advanced"
    calls = await ctx.audit(result.run_id)
    placeholder = "omitted from context"
    assert any(placeholder in json.dumps(c.provider_request, default=str) for c in calls), (
        "no drop placeholder reached the provider"
    )
    advance_iteration = min((e.iteration_index or 0) for e in changed)
    ctx.rec["advance_iteration"] = advance_iteration
    ctx.rec["cache_reads_after_advance"] = [
        c.cache_read_input_tokens for c in calls if c.iteration >= advance_iteration
    ]
    assert any((c.cache_read_input_tokens or 0) > 0 for c in calls), (
        "no cache read in the whole run"
    )


async def s9a_many_image_rule(ctx: Ctx) -> None:
    big = Image.from_bytes((FIXTURES / "photo_4032x3024.jpg").read_bytes())
    inputs: list[Any] = ["Describe these.", *[img("zebra42.png") for _ in range(20)], big]
    policy = ImagePolicy(max_images_per_message=25)
    async with ctx.agent(images=policy) as agent:
        try:
            result = await agent.run(inputs)
        except ImageLimitError as exc:
            assert exc.scope == "many_image_dimension", exc.scope
            assert_clean_exception(exc)
            ctx.rec["error"] = (
                f"ImageLimitError({exc.scope}, limit={exc.limit}, actual={exc.actual})"
            )
            return
    assert result.status is RunStatus.ERROR, result.status
    assert "many" in (result.error or "").lower() or "2000" in (result.error or ""), result.error
    async with ctx.store() as store:
        assert not await store.get_llm_calls(result.run_id), "a request was sent"


async def l06_text_only(ctx: Ctx) -> None:
    async with ctx.agent(tools=[lookup], prompt=long_prompt(ctx.nonce)) as agent:
        result = await agent.run("Look up the key 'alpha' and tell me the number in its value.")
    assert result.status is RunStatus.SUCCESS, result.error
    assert "4711" in (result.answer or "")
    calls = await ctx.audit(result.run_id)
    assert len(calls) >= 2
    ctx.rec["answer"] = (result.answer or "")[:200]
    ctx.rec["note"] = (
        "pair with the same scenario on the v0.3.0 tag: request hash differs by nonce only"
    )


class _CompletionRequests:
    """Counts chat-completions HTTP requests on an OpenAI-compatible provider."""

    def __init__(self, provider: LLMProvider) -> None:
        self.count = 0
        client = getattr(getattr(provider, "_client", None), "_client", None)
        hooks = getattr(client, "event_hooks", None)
        self.attached = hooks is not None
        if hooks is not None:
            hooks["request"].append(self._on_request)
        for name in ("complete", "complete_stream"):
            original = getattr(provider, name)
            setattr(provider, name, self._wrap(original))

    def _wrap(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def counted(*args: Any, **kwargs: Any) -> Any:
            self.count += 1
            return original(*args, **kwargs)

        return counted

    async def _on_request(self, request: Any) -> None:
        if str(request.url.path).endswith("/chat/completions"):
            self.count += 1


async def s8a_text_only_catalog(ctx: Ctx) -> None:
    provider = ctx.case.route.make_provider()
    spy = _CompletionRequests(provider)
    async with ctx.agent(provider=provider) as agent:
        with pytest.raises(UnsupportedModalityError) as caught:
            await agent.run(["Describe this image.", img("zebra42.png")])
    exc = caught.value
    assert exc.source == "catalog", exc.source
    assert_clean_exception(exc)
    assert spy.count == 0, f"{spy.count} completion request(s) were sent"
    async with ctx.store() as store:
        assert await store.count_runs() == 0, "a run row was created"
    ctx.rec["error"] = f"UnsupportedModalityError(source={exc.source})"
    ctx.rec["http_hook_attached"] = spy.attached


async def s8b_forced_yes_404(ctx: Ctx) -> None:
    policy = ImagePolicy(capability_overrides={DEEPSEEK: "yes"})
    async with ctx.agent(images=policy) as agent:
        with pytest.raises(UnsupportedModalityError) as caught:
            await agent.run(["Describe this image.", img("zebra42.png")])
    exc = caught.value
    assert exc.source == "provider_response", exc.source
    assert exc.status_code in (400, 404), exc.status_code
    assert_clean_exception(exc)
    ctx.rec["error"] = (
        f"UnsupportedModalityError(source={exc.source}, status={exc.status_code}, "
        f"code={exc.provider_code})"
    )


async def s8c_history_placeholder(ctx: Ctx) -> None:
    vision = OR_VISION.make_provider()
    async with ctx.agent(provider=vision) as agent:
        first = await agent.run(["Remember this image. Reply OK.", img("zebra42.png")])
    assert first.status is RunStatus.SUCCESS, first.error
    await ctx.audit(first.run_id, label="s8c-vision")

    async with ctx.agent() as agent:  # text-only model, same database
        second = await agent.run(
            "Reply with the single word DONE.", runs=[first.run_id], replay_last="all"
        )
    assert second.status is RunStatus.SUCCESS, second.error
    calls = await ctx.audit(second.run_id, label="s8c-text-only")
    sent = json.dumps([c.provider_request for c in calls], default=str)
    assert "does not accept images" in sent, "no unsupported placeholder in the request"
    assert "image_url" not in sent, "an image part reached the text-only model"
    ctx.rec["answer"] = (second.answer or "")[:200]


SCENARIOS: dict[str, Callable[[Ctx], Awaitable[None]]] = {
    "S1": s1_user_image,
    "S2": s2_server_tool_image,
    "S3": s3_client_tool_resume,
    "S4": s4_cache,
    "S5": s5_two_images,
    "S7": s7_drop_advance,
    "S8a": s8a_text_only_catalog,
    "S8b": s8b_forced_yes_404,
    "S8c": s8c_history_placeholder,
    "S9a": s9a_many_image_rule,
    "L06": l06_text_only,
}


# --------------------------------------------------------------------------- the test


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.id)
async def test_live(case: Case, tmp_path: Path, rec: dict[str, Any]) -> None:
    if not os.environ.get(case.route.key_env):
        pytest.skip(f"{case.route.key_env} is not set in the environment")
    if case.scenario == "S8c" and not os.environ.get(OR_VISION.key_env):
        pytest.skip(f"{OR_VISION.key_env} is not set in the environment")
    ctx = Ctx(case, tmp_path, rec)
    try:
        await SCENARIOS[case.scenario](ctx)
    except (AssertionError, pytest.skip.Exception):
        raise
    except Exception as exc:
        tail = "".join(traceback.format_exception(exc)[-6:])
        raise AssertionError(f"{type(exc).__name__}: {scrub(tail)}") from None
