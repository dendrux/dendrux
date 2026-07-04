"""Tests for the public context-block surface (text-folding assembly)."""

import pytest

from dendrux.context_blocks import (
    ContextBlock,
    fold_context,
    render_band,
    validate_context,
)
from dendrux.types import Message, Role


def _history() -> list[Message]:
    return [
        Message(role=Role.USER, content="Q1"),
        Message(role=Role.ASSISTANT, content="A1"),
    ]


class TestValidateContext:
    def test_none_and_empty_are_noops(self) -> None:
        validate_context(None)
        validate_context([])

    def test_rejects_non_block(self) -> None:
        with pytest.raises(TypeError, match="context\\[0\\]"):
            validate_context(["not a block"])  # type: ignore[list-item]

    def test_rejects_empty_content(self) -> None:
        with pytest.raises(ValueError, match="empty content"):
            validate_context([ContextBlock(content="")])

    def test_rejects_bad_placement(self) -> None:
        with pytest.raises(ValueError, match="placement"):
            validate_context([ContextBlock(content="x", placement="sticky")])  # type: ignore[arg-type]


class TestRenderBand:
    def test_filters_by_placement_and_preserves_order(self) -> None:
        ctx = [
            ContextBlock("instructions", placement="stable"),
            ContextBlock("doc", placement="dynamic"),
            ContextBlock("memory", placement="stable"),
        ]
        assert render_band(ctx, "stable") == "instructions\n\nmemory"
        assert render_band(ctx, "dynamic") == "doc"

    def test_empty(self) -> None:
        assert render_band(None, "stable") == ""
        assert render_band([], "dynamic") == ""


class TestFoldNoHistory:
    def test_no_history_no_context_returns_none(self) -> None:
        assert fold_context([], None, "hi") is None

    def test_stable_only(self) -> None:
        msgs = fold_context([], [ContextBlock("spec", placement="stable")], "hi")
        assert msgs is not None
        assert len(msgs) == 1
        # Uses the first-message wrapper so it stays byte-stable once it
        # becomes history[0] next run.
        assert msgs[0].content == "STABLE CONTEXT:\nspec\n\nUSER MESSAGE:\nhi"
        assert msgs[0].role == Role.USER

    def test_dynamic_only(self) -> None:
        msgs = fold_context([], [ContextBlock("doc")], "hi")
        assert msgs is not None
        assert msgs[0].content == "ADDITIONAL CONTEXT:\ndoc\n\nUSER INPUT:\nhi"

    def test_stable_and_dynamic(self) -> None:
        ctx = [
            ContextBlock("spec", placement="stable"),
            ContextBlock("doc", placement="dynamic"),
        ]
        msgs = fold_context([], ctx, "hi")
        assert msgs is not None
        assert msgs[0].content == (
            "STABLE CONTEXT:\nspec\n\nUSER MESSAGE:\nADDITIONAL CONTEXT:\ndoc\n\nUSER INPUT:\nhi"
        )


class TestFoldWithHistory:
    def test_no_context_is_plain_assembly(self) -> None:
        # first message unchanged; current message is the raw input.
        msgs = fold_context(_history(), None, "Q2")
        assert msgs is not None
        assert [m.content for m in msgs] == ["Q1", "A1", "Q2"]
        assert msgs[0].role == Role.USER
        assert msgs[-1].role == Role.USER

    def test_stable_folds_into_first_message(self) -> None:
        msgs = fold_context(_history(), [ContextBlock("spec", placement="stable")], "Q2")
        assert msgs is not None
        assert msgs[0].content == "STABLE CONTEXT:\nspec\n\nUSER MESSAGE:\nQ1"
        assert msgs[0].role == Role.USER
        assert msgs[-1].content == "Q2"  # no dynamic → raw input

    def test_dynamic_folds_into_current_message(self) -> None:
        msgs = fold_context(_history(), [ContextBlock("doc")], "Q2")
        assert msgs is not None
        assert msgs[0].content == "Q1"  # first untouched
        assert msgs[-1].content == "ADDITIONAL CONTEXT:\ndoc\n\nUSER INPUT:\nQ2"

    def test_stable_and_dynamic_fold_into_respective_messages(self) -> None:
        ctx = [
            ContextBlock("spec", placement="stable"),
            ContextBlock("doc", placement="dynamic"),
        ]
        msgs = fold_context(_history(), ctx, "Q2")
        assert msgs is not None
        assert msgs[0].content == "STABLE CONTEXT:\nspec\n\nUSER MESSAGE:\nQ1"
        assert msgs[-1].content == "ADDITIONAL CONTEXT:\ndoc\n\nUSER INPUT:\nQ2"

    def test_within_band_order_preserved(self) -> None:
        ctx = [
            ContextBlock("instructions", placement="stable"),
            ContextBlock("memory", placement="stable"),
        ]
        msgs = fold_context(_history(), ctx, "Q2")
        assert msgs is not None
        assert msgs[0].content == ("STABLE CONTEXT:\ninstructions\n\nmemory\n\nUSER MESSAGE:\nQ1")

    def test_history_middle_untouched(self) -> None:
        hist = [
            Message(role=Role.USER, content="Q1"),
            Message(role=Role.ASSISTANT, content="A1"),
            Message(role=Role.USER, content="Q2"),
            Message(role=Role.ASSISTANT, content="A2"),
        ]
        msgs = fold_context(hist, [ContextBlock("spec", placement="stable")], "Q3")
        assert msgs is not None
        assert [m.content for m in msgs[1:-1]] == ["A1", "Q2", "A2"]
