"""Provider wire mapping for image parts.

Goldens U-04..U-07: Anthropic native, OpenAI Chat portable, OpenAI Responses
native + portable override, OpenRouter (Chat portable). Covers user images,
image-only content, text/image order, detail handling, parallel tool calls,
mixed visible / placeholder / note rendering (N-7), the exact portable framing
bytes, cache-control count, sanitised captured_request and no caller mutation.
No network.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from dendrux.llm._images import (
    TOOL_IMAGE_NOTE,
    TOOL_OUTPUT_CLOSE,
    TOOL_OUTPUT_OPEN,
    WireImageRegistry,
    framing_token,
)
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.llm.openai import OpenAIProvider
from dendrux.llm.openai_responses import OpenAIResponsesProvider
from dendrux.llm.openrouter import OpenRouterProvider
from dendrux.media import inspect_image
from dendrux.media._ids import placeholder
from dendrux.types import ImagePart, ImageRef, Message, Role, TextPart, ToolCall

FIXTURES = Path(__file__).parent.parent / "fixtures" / "images"
ZEBRA = (FIXTURES / "zebra42.png").read_bytes()
KIWI = (FIXTURES / "kiwi77.png").read_bytes()
ZEBRA_B64 = base64.b64encode(ZEBRA).decode("ascii")
KIWI_B64 = base64.b64encode(KIWI).decode("ascii")
ZEBRA_URL = "data:image/png;base64," + ZEBRA_B64
KIWI_URL = "data:image/png;base64," + KIWI_B64


def image_part(data: bytes, detail: Any = None) -> ImagePart:
    header = inspect_image(data)
    ref = ImageRef(
        sha256=hashlib.sha256(data).hexdigest(),
        media_type=header.media_type,
        byte_length=len(data),
        width=header.width,
        height=header.height,
    )
    return ImagePart(ref=ref, detail=detail, data=data)


Z = image_part(ZEBRA)
K = image_part(KIWI)
ZID = Z.image_id
KID = K.image_id
ZSHA = Z.ref.sha256
KSHA = K.ref.sha256


def sentinel(sha: str) -> str:
    return f"dendrux-image:sha256:{sha}"


def user_parts(*parts: Any) -> Message:
    return Message.with_parts(Role.USER, [TextPart(p) if isinstance(p, str) else p for p in parts])


def tool_parts(call: ToolCall, *parts: Any) -> Message:
    return Message.with_parts(
        Role.TOOL,
        [TextPart(p) if isinstance(p, str) else p for p in parts],
        name=call.name,
        call_id=call.id,
    )


def screenshot_turn(*calls: ToolCall) -> list[Message]:
    return [
        Message(role=Role.USER, content="Take a screenshot and tell me the code word."),
        Message(role=Role.ASSISTANT, content="", tool_calls=list(calls)),
    ]


def call(provider_id: str = "toolu_P1", name: str = "take_screenshot") -> ToolCall:
    return ToolCall(name=name, params={}, provider_tool_call_id=provider_id)


def count_key(obj: Any, key: str) -> int:
    if isinstance(obj, dict):
        return (key in obj) + sum(count_key(v, key) for v in obj.values())
    if isinstance(obj, list):
        return sum(count_key(v, key) for v in obj)
    return 0


def assert_no_image_bytes(obj: Any) -> None:
    dumped = json.dumps(obj, default=str)
    assert ZEBRA_B64[:40] not in dumped
    assert KIWI_B64[:40] not in dumped
    assert "data:image" not in dumped
    assert "iVBORw0K" not in dumped


@pytest.fixture
def anthropic_provider() -> AnthropicProvider:
    return AnthropicProvider(model="claude-sonnet-4-6", api_key="test-key")


@pytest.fixture
def chat_provider() -> OpenAIProvider:
    return OpenAIProvider(model="gpt-5-mini", api_key="test-key")


@pytest.fixture
def responses_provider() -> OpenAIResponsesProvider:
    return OpenAIResponsesProvider(model="gpt-5-mini", api_key="test-key")


# ---------------------------------------------------------------------------
# Anthropic (native)
# ---------------------------------------------------------------------------
class TestAnthropicWire:
    def test_user_image_golden(self, anthropic_provider: AnthropicProvider) -> None:
        msgs = [user_parts("What does this say?", Z)]
        _, api = anthropic_provider._convert_messages(msgs)
        assert api == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What does this say?"},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": ZEBRA_B64},
                    },
                ],
            }
        ]

    def test_text_image_text_order_and_image_only(
        self, anthropic_provider: AnthropicProvider
    ) -> None:
        reg = WireImageRegistry()
        msgs = [user_parts("a", Z, "b", K), Message(role=Role.ASSISTANT, content="ok")]
        msgs.append(user_parts(K))
        _, api = anthropic_provider._convert_messages(msgs, registry=reg)
        assert [b["type"] for b in api[0]["content"]] == ["text", "image", "text", "image"]
        assert api[0]["content"][3]["source"]["data"] == KIWI_B64
        assert api[2]["content"] == [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": KIWI_B64},
            }
        ]
        assert reg.paths == {
            ("messages", 0, "content", 1): ZID,
            ("messages", 0, "content", 3): KID,
            ("messages", 2, "content", 0): KID,
        }

    def test_detail_dropped(self, anthropic_provider: AnthropicProvider) -> None:
        _, api = anthropic_provider._convert_messages([user_parts("x", image_part(ZEBRA, "low"))])
        assert "detail" not in json.dumps(api)

    def test_tool_result_native_golden_with_cache_marker(
        self, anthropic_provider: AnthropicProvider
    ) -> None:
        c = call()
        msgs = [*screenshot_turn(c), tool_parts(c, '"Captured."', Z)]
        api_kwargs, captured = anthropic_provider._build_api_kwargs(msgs, None, {})
        assert api_kwargs["messages"][2] == {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_P1",
                    "content": [
                        {"type": "text", "text": '"Captured."'},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": ZEBRA_B64,
                            },
                        },
                    ],
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
        nested = api_kwargs["messages"][2]["content"][0]["content"]
        assert all("cache_control" not in b for b in nested)
        assert captured["messages"][2]["content"][0]["content"][1]["source"]["data"] == sentinel(
            ZSHA
        )
        assert_no_image_bytes(captured)

    def test_user_image_last_block_carries_marker(
        self, anthropic_provider: AnthropicProvider
    ) -> None:
        api_kwargs, _ = anthropic_provider._build_api_kwargs([user_parts("look", Z)], None, {})
        blocks = api_kwargs["messages"][0]["content"]
        assert "cache_control" not in blocks[0]
        assert blocks[1]["cache_control"] == {"type": "ephemeral"}

    def test_parallel_tools_merge_and_paths(self, anthropic_provider: AnthropicProvider) -> None:
        c1, c2 = call("toolu_A"), call("toolu_B")
        msgs = [*screenshot_turn(c1, c2), tool_parts(c1, '"one"', Z), tool_parts(c2, '"two"', K)]
        reg = WireImageRegistry()
        _, api = anthropic_provider._convert_messages(msgs, registry=reg)
        assert len(api) == 3
        results = api[2]["content"]
        assert [r["tool_use_id"] for r in results] == ["toolu_A", "toolu_B"]
        assert results[1]["content"][1]["source"]["data"] == KIWI_B64
        assert reg.paths == {
            ("messages", 2, "content", 0, "content", 1): ZID,
            ("messages", 2, "content", 1, "content", 1): KID,
        }

    def test_mixed_visible_placeholder_note(self, anthropic_provider: AnthropicProvider) -> None:
        c = call()
        drop = placeholder("drop", KID)
        note = "[image from MCP could not be used: corrupt]"
        msgs = [*screenshot_turn(c), tool_parts(c, '"p"', note, Z, drop, Z)]
        _, api = anthropic_provider._convert_messages(msgs)
        content = api[2]["content"][0]["content"]
        assert content[:2] == [{"type": "text", "text": '"p"'}, {"type": "text", "text": note}]
        assert content[2]["type"] == "image"
        assert content[3] == {"type": "text", "text": drop}
        assert content[4]["type"] == "image"

    def test_all_placeholders_is_plain_string(self, anthropic_provider: AnthropicProvider) -> None:
        # prepare_request emits a parts-free message when no image is left.
        c = call()
        text = '"p"\n\n' + placeholder("rejected", ZID)
        tool = Message(role=Role.TOOL, content=text, name=c.name, call_id=c.id)
        _, api = anthropic_provider._convert_messages([*screenshot_turn(c), tool])
        assert api[2]["content"][0]["content"] == text

    def test_second_of_two_tool_messages_visible(
        self, anthropic_provider: AnthropicProvider
    ) -> None:
        c1, c2 = call("toolu_A"), call("toolu_B")
        t1 = Message(role=Role.TOOL, content='"a"', name=c1.name, call_id=c1.id)
        msgs = [*screenshot_turn(c1, c2), t1, tool_parts(c2, '"b"', Z)]
        _, api = anthropic_provider._convert_messages(msgs)
        assert api[2]["content"][0]["content"] == '"a"'
        assert api[2]["content"][1]["content"][1]["type"] == "image"

    def test_is_error_and_no_native_correlation(
        self, anthropic_provider: AnthropicProvider
    ) -> None:
        c = call()
        err = Message.with_parts(
            Role.TOOL,
            [TextPart('"e"'), Z],
            name=c.name,
            call_id=c.id,
            meta={"is_error": True},
        )
        _, api = anthropic_provider._convert_messages([*screenshot_turn(c), err])
        assert api[2]["content"][0]["is_error"] is True
        c2 = ToolCall(name="t", params={})
        plain = tool_parts(c2, '"x"', Z)
        _, api = anthropic_provider._convert_messages([*screenshot_turn(c2), plain])
        assert api[2]["role"] == "user"
        assert api[2]["content"][1]["type"] == "image"

    @pytest.mark.parametrize("stable", [False, True])
    def test_cache_control_at_most_four(
        self, anthropic_provider: AnthropicProvider, stable: bool
    ) -> None:
        c1, c2 = call("toolu_A"), call("toolu_B")
        head = Message(role=Role.USER, content="context head", placement="stable")
        msgs = [
            Message(role=Role.SYSTEM, content="sys"),
            *([head] if stable else []),
            user_parts("look", Z, K),
            Message(role=Role.ASSISTANT, content="", tool_calls=[c1, c2]),
            tool_parts(c1, '"one"', Z),
            tool_parts(c2, '"two"', K),
            Message(role=Role.ASSISTANT, content="seen"),
            user_parts(Z),
        ]
        api_kwargs, captured = anthropic_provider._build_api_kwargs(msgs, None, {})
        total = count_key(api_kwargs["messages"], "cache_control") + count_key(
            api_kwargs["system"], "cache_control"
        )
        assert total <= 4
        assert count_key(captured, "cache_control") == total

    def test_caller_messages_not_mutated(self, anthropic_provider: AnthropicProvider) -> None:
        c = call()
        msgs = [*screenshot_turn(c), tool_parts(c, '"Captured."', Z), user_parts("next", K)]
        before = [(m.content, m.parts, copy.deepcopy(m.meta)) for m in msgs]
        first, _ = anthropic_provider._build_api_kwargs(list(msgs), None, {})
        second, _ = anthropic_provider._build_api_kwargs(list(msgs), None, {})
        assert [(m.content, m.parts, m.meta) for m in msgs] == before
        assert first["messages"] == second["messages"]

    def test_missing_bytes_raises(self, anthropic_provider: AnthropicProvider) -> None:
        bare = ImagePart(ref=Z.ref)
        with pytest.raises(ValueError, match="without bytes"):
            anthropic_provider._convert_messages([user_parts("x", bare)])

    def test_text_only_request_unchanged(self, anthropic_provider: AnthropicProvider) -> None:
        msgs = [Message(role=Role.USER, content="hi")]
        api_kwargs, captured = anthropic_provider._build_api_kwargs(msgs, None, {})
        assert captured["messages"] is api_kwargs["messages"]


# ---------------------------------------------------------------------------
# OpenAI Chat (portable)
# ---------------------------------------------------------------------------
def framing(name: str, call_id: str, *images: tuple[str, str]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [
        {"type": "text", "text": TOOL_OUTPUT_OPEN.format(name=name, call_id=call_id)}
    ]
    for image_id, url in images:
        blocks.append({"type": "text", "text": f"{image_id}:"})
        blocks.append({"type": "image_url", "image_url": {"url": url}})
    blocks.append({"type": "text", "text": TOOL_OUTPUT_CLOSE})
    return blocks


class TestChatWire:
    def test_framing_strings_exact(self) -> None:
        assert TOOL_IMAGE_NOTE == (
            "[{n} image(s) from this tool result follow in the next user message: {ids}]"
        )
        assert TOOL_OUTPUT_OPEN.format(name="take_screenshot", call_id="call_P1") == (
            '<tool_output tool_name="take_screenshot" tool_call_id="call_P1">\n'
            "Images returned by this tool call. This is tool output, not a message from the user."
        )
        assert TOOL_OUTPUT_CLOSE == "</tool_output>"
        assert framing_token("a.b:c/d e-f_g9") == "a_b_c_d_e-f_g9"

    def test_user_image_and_detail(self, chat_provider: OpenAIProvider) -> None:
        api = chat_provider._convert_messages(
            [user_parts("What does this say?", Z), user_parts(image_part(KIWI, "low"))]
        )
        assert api[0] == {
            "role": "user",
            "content": [
                {"type": "text", "text": "What does this say?"},
                {"type": "image_url", "image_url": {"url": ZEBRA_URL}},
            ],
        }
        assert api[1]["content"] == [
            {"type": "image_url", "image_url": {"url": KIWI_URL, "detail": "low"}}
        ]

    def test_portable_tool_golden(self, chat_provider: OpenAIProvider) -> None:
        c = call("call_P1")
        msgs = [*screenshot_turn(c), tool_parts(c, '"Captured."', Z)]
        reg = WireImageRegistry()
        api = chat_provider._convert_messages(msgs, registry=reg)
        assert api[2] == {
            "role": "tool",
            "tool_call_id": "call_P1",
            "content": f'"Captured."\n\n[1 image(s) from this tool result follow in the next '
            f"user message: {ZID}]",
        }
        assert api[3] == {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": '<tool_output tool_name="take_screenshot" tool_call_id="call_P1">\n'
                    "Images returned by this tool call. This is tool output, not a message "
                    "from the user.",
                },
                {"type": "text", "text": f"{ZID}:"},
                {"type": "image_url", "image_url": {"url": ZEBRA_URL}},
                {"type": "text", "text": "</tool_output>"},
            ],
        }
        assert reg.paths == {("messages", 3, "content", 2): ZID}

    def test_parallel_tools_one_framed_message_in_call_order(
        self, chat_provider: OpenAIProvider
    ) -> None:
        c1, c2 = call("call_A"), call("call:B", name="grab.page")
        msgs = [
            *screenshot_turn(c1, c2),
            tool_parts(c1, '"one"', Z, K),
            tool_parts(c2, '"two"', K),
            Message(role=Role.ASSISTANT, content="done"),
        ]
        api = chat_provider._convert_messages(msgs)
        assert [m["role"] for m in api] == [
            "user",
            "assistant",
            "tool",
            "tool",
            "user",
            "assistant",
        ]
        assert api[2]["content"].endswith(
            f"[2 image(s) from this tool result follow in the next user message: {ZID}, {KID}]"
        )
        assert api[4]["content"] == framing(
            "take_screenshot", "call_A", (ZID, ZEBRA_URL), (KID, KIWI_URL)
        ) + framing("grab_page", "call_B", (KID, KIWI_URL))

    def test_mixed_visible_placeholder_note(self, chat_provider: OpenAIProvider) -> None:
        c = call("call_P1")
        drop = placeholder("drop", KID)
        note = "[image from MCP could not be used: corrupt]"
        k2 = image_part(KIWI)
        msgs = [*screenshot_turn(c), tool_parts(c, '"p"', note, Z, drop, k2)]
        api = chat_provider._convert_messages(msgs)
        assert api[2]["content"] == (
            f'"p"\n\n{note}\n\n{drop}\n\n[2 image(s) from this tool result follow in the next '
            f"user message: {ZID}, {KID}]"
        )
        assert api[3]["content"] == framing(
            "take_screenshot", "call_P1", (ZID, ZEBRA_URL), (KID, KIWI_URL)
        )

    def test_all_placeholders_emits_no_framed_message(self, chat_provider: OpenAIProvider) -> None:
        c = call("call_P1")
        text = '"p"\n\n' + placeholder("unsupported", ZID)
        tool = Message(role=Role.TOOL, content=text, name=c.name, call_id=c.id)
        api = chat_provider._convert_messages([*screenshot_turn(c), tool])
        assert api[-1] == {"role": "tool", "tool_call_id": "call_P1", "content": text}

    def test_only_second_tool_message_visible(self, chat_provider: OpenAIProvider) -> None:
        c1, c2 = call("call_A"), call("call_B")
        t1 = Message(role=Role.TOOL, content='"a"', name=c1.name, call_id=c1.id)
        msgs = [*screenshot_turn(c1, c2), t1, tool_parts(c2, '"b"', Z), user_parts("next?", K)]
        api = chat_provider._convert_messages(msgs)
        assert api[2] == {"role": "tool", "tool_call_id": "call_A", "content": '"a"'}
        assert api[4]["content"] == framing("take_screenshot", "call_B", (ZID, ZEBRA_URL))
        assert api[5]["content"][0] == {"type": "text", "text": "next?"}

    def test_captured_request_sanitised_and_cache_key_unchanged(
        self, chat_provider: OpenAIProvider
    ) -> None:
        c = call("call_P1")
        msgs = [*screenshot_turn(c), tool_parts(c, '"Captured."', Z), user_parts("again", Z)]
        api_kwargs, captured = chat_provider._build_api_kwargs(
            msgs, None, {}, cache_key_prefix="agent:gpt-5-mini"
        )
        assert_no_image_bytes(captured)
        assert captured["messages"][3]["content"][2]["image_url"]["url"] == sentinel(ZSHA)
        assert api_kwargs["messages"][3]["content"][2]["image_url"]["url"] == ZEBRA_URL
        text_kwargs, _ = chat_provider._build_api_kwargs(
            [Message(role=Role.USER, content="x")], None, {}, cache_key_prefix="agent:gpt-5-mini"
        )
        assert api_kwargs["prompt_cache_key"] == text_kwargs["prompt_cache_key"]
        assert count_key(api_kwargs, "cache_control") == 0

    def test_openrouter_same_portable_mapping(self) -> None:
        provider = OpenRouterProvider(model="anthropic/claude-haiku-4.5", api_key="test-key")
        c = call("toolu_1")
        msgs = [*screenshot_turn(c), tool_parts(c, '"Captured."', Z)]
        api_kwargs, captured = provider._build_api_kwargs(msgs, None, {})
        assert api_kwargs["messages"][3]["content"] == framing(
            "take_screenshot", "toolu_1", (ZID, ZEBRA_URL)
        )
        assert count_key(api_kwargs["messages"], "cache_control") == 0
        assert_no_image_bytes(captured)


# ---------------------------------------------------------------------------
# OpenAI Responses (native default, portable override)
# ---------------------------------------------------------------------------
class TestResponsesWire:
    def test_user_image_detail_always_sent(
        self, responses_provider: OpenAIResponsesProvider
    ) -> None:
        reg = WireImageRegistry()
        _, items = responses_provider._convert_messages(
            [user_parts("What?", Z, image_part(KIWI, "high"))], registry=reg
        )
        assert items == [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What?"},
                    {"type": "input_image", "image_url": ZEBRA_URL, "detail": "auto"},
                    {"type": "input_image", "image_url": KIWI_URL, "detail": "high"},
                ],
            }
        ]
        assert reg.paths == {("input", 0, "content", 1): ZID, ("input", 0, "content", 2): KID}

    def test_native_tool_output_golden(self, responses_provider: OpenAIResponsesProvider) -> None:
        assert responses_provider.tool_result_images == "native"
        c = call("call_P1")
        reg = WireImageRegistry()
        _, items = responses_provider._convert_messages(
            [*screenshot_turn(c), tool_parts(c, '"Captured."', Z)], registry=reg
        )
        assert items[-1] == {
            "type": "function_call_output",
            "call_id": "call_P1",
            "output": [
                {"type": "input_text", "text": '"Captured."'},
                {"type": "input_image", "image_url": ZEBRA_URL, "detail": "auto"},
            ],
        }
        assert reg.paths == {("input", 2, "output", 1): ZID}

    def test_native_mixed_rendering(self, responses_provider: OpenAIResponsesProvider) -> None:
        c = call("call_P1")
        drop = placeholder("drop", KID)
        _, items = responses_provider._convert_messages(
            [*screenshot_turn(c), tool_parts(c, '"p"', Z, drop)]
        )
        assert items[-1]["output"][2] == {"type": "input_text", "text": drop}

    def test_portable_override(self) -> None:
        provider = OpenAIResponsesProvider(
            model="gpt-5-mini", api_key="test-key", tool_result_images="portable"
        )
        c1, c2 = call("call_A"), call("call_B")
        msgs = [*screenshot_turn(c1, c2), tool_parts(c1, '"a"', Z), tool_parts(c2, '"b"', K)]
        reg = WireImageRegistry()
        _, items = provider._convert_messages(msgs, registry=reg)
        assert items[3]["output"] == (
            f'"a"\n\n[1 image(s) from this tool result follow in the next user message: {ZID}]'
        )
        assert items[5]["role"] == "user"
        assert items[5]["content"] == [
            {
                "type": "input_text",
                "text": TOOL_OUTPUT_OPEN.format(name="take_screenshot", call_id="call_A"),
            },
            {"type": "input_text", "text": f"{ZID}:"},
            {"type": "input_image", "image_url": ZEBRA_URL, "detail": "auto"},
            {"type": "input_text", "text": TOOL_OUTPUT_CLOSE},
            {
                "type": "input_text",
                "text": TOOL_OUTPUT_OPEN.format(name="take_screenshot", call_id="call_B"),
            },
            {"type": "input_text", "text": f"{KID}:"},
            {"type": "input_image", "image_url": KIWI_URL, "detail": "auto"},
            {"type": "input_text", "text": TOOL_OUTPUT_CLOSE},
        ]
        assert reg.paths == {("input", 5, "content", 2): ZID, ("input", 5, "content", 6): KID}

    def test_invalid_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="tool_result_images"):
            OpenAIResponsesProvider(
                model="gpt-5-mini",
                api_key="test-key",
                tool_result_images="inline",  # type: ignore[arg-type]
            )

    def test_captured_request_sanitised(self, responses_provider: OpenAIResponsesProvider) -> None:
        c = call("call_P1")
        api_kwargs, captured = responses_provider._build_api_kwargs(
            [*screenshot_turn(c), tool_parts(c, '"Captured."', Z)],
            None,
            {},
            cache_key_prefix="a:m",
        )
        assert_no_image_bytes(captured)
        assert captured["input"][2]["output"][1]["image_url"] == sentinel(ZSHA)
        assert api_kwargs["prompt_cache_key"] == "dendrux:a:m"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class TestRegistry:
    def test_locate_prefix_ancestor_and_ambiguity(self) -> None:
        reg = WireImageRegistry()
        reg.register(("messages", 2, "content", 0, "content", 1), Z)
        reg.register(("messages", 2, "content", 1, "content", 1), K)
        reg.register(("messages", 4, "content", 1), Z)
        assert reg.locate(("messages", 2, "content", 1, "content", 1, "source", "data")) == KID
        assert reg.locate(("messages", 4)) == ZID  # ancestor holding one image
        assert reg.locate(("messages", 2)) is None  # ancestor holding two
        assert reg.locate(("messages", 3, "content", 0)) is None
        assert reg.locate(None) is None

    def test_sanitize_does_not_mutate(self) -> None:
        reg = WireImageRegistry()
        url = reg.data_url(Z)
        payload = {"a": [url, {"b": (reg.base64(Z), "keep")}]}
        before = copy.deepcopy(payload)
        out = reg.sanitize(payload)
        assert payload == before
        assert out == {"a": [sentinel(ZSHA), {"b": (sentinel(ZSHA), "keep")}]}


# ---------------------------------------------------------------------------
# Fix round (review-C)
# ---------------------------------------------------------------------------
class TestFixRoundWire:
    def test_responses_native_all_placeholders_is_string(
        self, responses_provider: OpenAIResponsesProvider
    ) -> None:
        c = call("call_P1")
        text = '"p"\n\n' + placeholder("drop", ZID)
        tool = Message(role=Role.TOOL, content=text, name=c.name, call_id=c.id)
        _, items = responses_provider._convert_messages([*screenshot_turn(c), tool])
        assert items[-1] == {"type": "function_call_output", "call_id": "call_P1", "output": text}

    def test_responses_native_only_second_visible(
        self, responses_provider: OpenAIResponsesProvider
    ) -> None:
        c1, c2 = call("call_A"), call("call_B")
        t1 = Message(role=Role.TOOL, content='"a"', name=c1.name, call_id=c1.id)
        reg = WireImageRegistry()
        _, items = responses_provider._convert_messages(
            [*screenshot_turn(c1, c2), t1, tool_parts(c2, '"b"', Z)], registry=reg
        )
        assert items[3] == {"type": "function_call_output", "call_id": "call_A", "output": '"a"'}
        assert items[4]["output"] == [
            {"type": "input_text", "text": '"b"'},
            {"type": "input_image", "image_url": ZEBRA_URL, "detail": "auto"},
        ]
        assert reg.paths == {("input", 4, "output", 1): ZID}

    def test_locate_never_descends_into_another_block(
        self, anthropic_provider: AnthropicProvider
    ) -> None:
        # Reviewer probe: consecutive user turns; an index that would name B after
        # a vendor-side merge must not resolve to the user image A.
        c = call("toolu_B")
        msgs = [
            Message(role=Role.USER, content="stable head", placement="stable"),
            Message(role=Role.USER, content="dynamic context"),
            user_parts(Z, "what is this?"),
            Message(role=Role.ASSISTANT, content="", tool_calls=[c]),
            tool_parts(c, '"shot"', K),
        ]
        reg = WireImageRegistry()
        anthropic_provider._convert_messages(msgs, registry=reg)
        assert reg.paths[("messages", 2, "content", 0)] == ZID
        probe = ("messages", 2, "content", 0, "content", 1, "image", "source", "base64")
        assert reg.locate(probe) is None
        assert reg.locate(("messages", 2, "content", 0, "source", "base64", "data")) == ZID
        assert reg.locate(("messages", 4, "content", 0, "content", 1, "image")) == KID

    def test_data_url_registry_keeps_one_string_per_image(self) -> None:
        reg = WireImageRegistry()
        url = reg.data_url(Z)
        assert reg.data_url(Z) is url
        assert reg._b64 == {}
        assert list(reg._by_value) == [url]
