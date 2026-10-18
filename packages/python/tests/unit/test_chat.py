"""Tests for the public chat-history surface (dendrux.chat)."""

from __future__ import annotations

import dataclasses

import pytest

from dendrux.chat import (
    ChatMessage,
    ChatRole,
    normalize_chat_history,
    validate_chat_history,
)
from dendrux.types import Message, Role


class TestChatRole:
    def test_only_user_and_assistant(self) -> None:
        assert {r.value for r in ChatRole} == {"user", "assistant"}

    def test_is_str_enum(self) -> None:
        assert ChatRole.USER == "user"
        assert ChatRole.ASSISTANT == "assistant"
        assert f"role={ChatRole.USER}" == "role=user"

    def test_excludes_system_and_tool(self) -> None:
        with pytest.raises(ValueError):
            ChatRole("system")
        with pytest.raises(ValueError):
            ChatRole("tool")


class TestChatMessage:
    def test_construct_directly(self) -> None:
        msg = ChatMessage(role=ChatRole.USER, content="hi")
        assert msg.role is ChatRole.USER
        assert msg.content == "hi"

    def test_user_factory(self) -> None:
        msg = ChatMessage.user("hello there")
        assert msg.role is ChatRole.USER
        assert msg.content == "hello there"

    def test_assistant_factory(self) -> None:
        msg = ChatMessage.assistant("how can I help?")
        assert msg.role is ChatRole.ASSISTANT
        assert msg.content == "how can I help?"

    def test_equality_for_same_content(self) -> None:
        a = ChatMessage.user("hi")
        b = ChatMessage(role=ChatRole.USER, content="hi")
        assert a == b
        assert hash(a) == hash(b)

    def test_inequality_when_role_differs(self) -> None:
        assert ChatMessage.user("hi") != ChatMessage.assistant("hi")

    def test_inequality_when_content_differs(self) -> None:
        assert ChatMessage.user("hi") != ChatMessage.user("hello")

    def test_is_frozen(self) -> None:
        msg = ChatMessage.user("hi")
        with pytest.raises(dataclasses.FrozenInstanceError):
            msg.content = "changed"  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            msg.role = ChatRole.ASSISTANT  # type: ignore[misc]

    def test_preserves_unicode_content_verbatim(self) -> None:
        msg = ChatMessage.user("héllo 🙂  с пробелами")
        assert msg.content == "héllo 🙂  с пробелами"

    def test_preserves_whitespace_verbatim(self) -> None:
        msg = ChatMessage.assistant("  leading and trailing  \n")
        assert msg.content == "  leading and trailing  \n"


class TestValidateChatHistory:
    def test_none_is_noop(self) -> None:
        validate_chat_history(None)

    def test_empty_list_is_noop(self) -> None:
        validate_chat_history([])

    def test_minimal_valid_pair(self) -> None:
        validate_chat_history(
            [
                ChatMessage.user("hi"),
                ChatMessage.assistant("hello"),
            ]
        )

    def test_longer_alternation_is_valid(self) -> None:
        validate_chat_history(
            [
                ChatMessage.user("a"),
                ChatMessage.assistant("b"),
                ChatMessage.user("c"),
                ChatMessage.assistant("d"),
                ChatMessage.user("e"),
                ChatMessage.assistant("f"),
            ]
        )

    def test_starts_with_assistant_raises(self) -> None:
        with pytest.raises(ValueError, match="must start with a user message"):
            validate_chat_history(
                [
                    ChatMessage.assistant("oops"),
                    ChatMessage.user("hi"),
                    ChatMessage.assistant("hello"),
                ]
            )

    def test_ends_with_user_raises(self) -> None:
        with pytest.raises(ValueError, match="pass it as user_input instead"):
            validate_chat_history(
                [
                    ChatMessage.user("hi"),
                    ChatMessage.assistant("hello"),
                    ChatMessage.user("forgot to put me in user_input"),
                ]
            )

    def test_single_user_raises_ends_with_user(self) -> None:
        with pytest.raises(ValueError, match="pass it as user_input instead"):
            validate_chat_history([ChatMessage.user("hi")])

    def test_single_assistant_raises_starts_with_assistant(self) -> None:
        with pytest.raises(ValueError, match="must start with a user message"):
            validate_chat_history([ChatMessage.assistant("hi")])

    def test_consecutive_user_raises(self) -> None:
        with pytest.raises(ValueError, match=r"consecutive user messages at index 3"):
            validate_chat_history(
                [
                    ChatMessage.user("a"),
                    ChatMessage.assistant("b"),
                    ChatMessage.user("c"),
                    ChatMessage.user("d"),
                    ChatMessage.assistant("e"),
                ]
            )

    def test_consecutive_assistant_raises(self) -> None:
        with pytest.raises(ValueError, match=r"consecutive assistant messages at index 2"):
            validate_chat_history(
                [
                    ChatMessage.user("a"),
                    ChatMessage.assistant("b"),
                    ChatMessage.assistant("c"),
                    ChatMessage.user("d"),
                    ChatMessage.assistant("e"),
                ]
            )

    def test_empty_user_content_raises(self) -> None:
        with pytest.raises(ValueError, match=r"index 0 has empty content"):
            validate_chat_history(
                [
                    ChatMessage.user(""),
                    ChatMessage.assistant("hi"),
                ]
            )

    def test_empty_assistant_content_raises(self) -> None:
        with pytest.raises(ValueError, match=r"index 1 has empty content"):
            validate_chat_history(
                [
                    ChatMessage.user("hi"),
                    ChatMessage.assistant(""),
                ]
            )

    def test_starts_check_fires_before_consecutive_check(self) -> None:
        with pytest.raises(ValueError, match="must start with a user message"):
            validate_chat_history(
                [
                    ChatMessage.assistant("a"),
                    ChatMessage.assistant("b"),
                    ChatMessage.user("c"),
                    ChatMessage.assistant("d"),
                ]
            )


class TestNormalizeChatHistory:
    def test_none_returns_empty_list(self) -> None:
        assert normalize_chat_history(None) == []

    def test_empty_list_returns_empty_list(self) -> None:
        assert normalize_chat_history([]) == []

    def test_converts_roles_to_internal_role(self) -> None:
        result = normalize_chat_history(
            [
                ChatMessage.user("hi"),
                ChatMessage.assistant("hello"),
            ]
        )
        assert [m.role for m in result] == [Role.USER, Role.ASSISTANT]

    def test_preserves_content_verbatim(self) -> None:
        content = "  héllo 🙂  с пробелами  \n  "
        result = normalize_chat_history(
            [
                ChatMessage.user(content),
                ChatMessage.assistant("ok"),
            ]
        )
        assert result[0].content == content

    def test_output_messages_have_no_runtime_fields(self) -> None:
        result = normalize_chat_history(
            [
                ChatMessage.user("hi"),
                ChatMessage.assistant("hello"),
            ]
        )
        for msg in result:
            assert msg.name is None
            assert msg.tool_calls is None
            assert msg.call_id is None
            assert msg.meta == {}

    def test_deterministic_across_calls(self) -> None:
        history = [
            ChatMessage.user("a"),
            ChatMessage.assistant("b"),
            ChatMessage.user("c"),
            ChatMessage.assistant("d"),
        ]
        first = normalize_chat_history(history)
        second = normalize_chat_history(history)
        assert first == second
        for a, b in zip(first, second, strict=True):
            assert a == b
            assert a.role == b.role
            assert a.content == b.content
            assert a.name == b.name
            assert a.tool_calls == b.tool_calls
            assert a.call_id == b.call_id
            assert a.meta == b.meta

    def test_equivalent_chatmessages_normalize_identically(self) -> None:
        via_factory = normalize_chat_history(
            [
                ChatMessage.user("hi"),
                ChatMessage.assistant("hello"),
            ]
        )
        via_direct = normalize_chat_history(
            [
                ChatMessage(role=ChatRole.USER, content="hi"),
                ChatMessage(role=ChatRole.ASSISTANT, content="hello"),
            ]
        )
        assert via_factory == via_direct

    def test_returns_internal_message_type(self) -> None:
        result = normalize_chat_history(
            [
                ChatMessage.user("hi"),
                ChatMessage.assistant("hello"),
            ]
        )
        for msg in result:
            assert isinstance(msg, Message)

    def test_validates_before_converting(self) -> None:
        with pytest.raises(ValueError, match="must start with a user message"):
            normalize_chat_history(
                [
                    ChatMessage.assistant("bad"),
                    ChatMessage.user("hi"),
                    ChatMessage.assistant("hello"),
                ]
            )

    def test_validates_empty_content(self) -> None:
        with pytest.raises(ValueError, match=r"index 0 has empty content"):
            normalize_chat_history(
                [
                    ChatMessage.user(""),
                    ChatMessage.assistant("hi"),
                ]
            )


class TestChatMessageImagesAndTypes:
    """Spec §2.3: assistant turns never carry images; content is always a str."""

    @staticmethod
    def _image():
        from pathlib import Path

        from dendrux.media import Image

        png = Path(__file__).resolve().parents[1] / "fixtures" / "images" / "zebra42.png"
        return Image.from_bytes(png.read_bytes())

    def test_assistant_factory_with_images_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="assistant chat messages cannot carry images"):
            ChatMessage.assistant(["x", self._image()])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="cannot carry images"):
            ChatMessage.assistant(self._image())  # type: ignore[arg-type]

    def test_assistant_with_parts_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="cannot carry images"):
            ChatMessage(role="assistant", content="x", parts=("x", self._image()))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="cannot carry images"):
            ChatMessage(ChatRole.ASSISTANT, "x", parts=("x", self._image()))

    def test_system_and_unknown_roles_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="'user' or 'assistant'"):
            ChatMessage(role="system", content="x", parts=("x", self._image()))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="'user' or 'assistant'"):
            ChatMessage(role="tool", content="x")  # type: ignore[arg-type]

    def test_string_role_is_normalised(self) -> None:
        msg = ChatMessage(role="assistant", content="hi")  # type: ignore[arg-type]
        assert msg.role is ChatRole.ASSISTANT
        assert msg == ChatMessage.assistant("hi")

    @pytest.mark.parametrize("content", [None, 3, b"bytes", ["a", "b"], {"a": 1}])
    def test_non_str_content_raises_type_error(self, content) -> None:
        with pytest.raises(TypeError, match="content must be a str"):
            ChatMessage.assistant(content)
        with pytest.raises(TypeError, match="content must be a str"):
            ChatMessage(role=ChatRole.USER, content=content)

    def test_user_list_content_hints_at_user_factory(self) -> None:
        with pytest.raises(TypeError, match=r"ChatMessage\.user"):
            ChatMessage(role=ChatRole.USER, content=["x", self._image()])  # type: ignore[arg-type]

    def test_bad_part_element_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match=r"ChatMessage\.parts\[1\]"):
            ChatMessage(ChatRole.USER, "x", parts=("x", 42))  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            ChatMessage.user(["x", 42])  # type: ignore[list-item]

    def test_user_parts_still_fine(self) -> None:
        img = self._image()
        msg = ChatMessage.user(["look", img])
        assert msg.role is ChatRole.USER and msg.content == "look" and msg.parts == ("look", img)

    def test_text_only_unchanged(self) -> None:
        msg = ChatMessage.user("hello")
        assert (msg.role, msg.content, msg.parts) == (ChatRole.USER, "hello", None)
        assert ChatMessage.assistant("") == ChatMessage(ChatRole.ASSISTANT, "")

    async def test_assistant_images_never_reach_a_run(self) -> None:
        from dendrux.agent import Agent
        from dendrux.llm.mock import MockLLM
        from dendrux.types import LLMResponse

        agent = Agent(prompt="p", provider=MockLLM([LLMResponse(text="x")]))
        with pytest.raises(ValueError, match="cannot carry images"):
            await agent.run(
                "q", history=[ChatMessage.user("hi"), ChatMessage.assistant(["x", self._image()])]
            )
        assert agent.provider.calls_made == 0  # type: ignore[union-attr]
