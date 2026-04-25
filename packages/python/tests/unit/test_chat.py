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
