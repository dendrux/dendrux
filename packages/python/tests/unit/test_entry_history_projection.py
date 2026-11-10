"""Entry chat history preserves roles and images while projecting rejected text."""

import pytest

from dendrux.chat import ChatMessage
from dendrux.guardrails import GuardrailEngine, SecretDetection
from dendrux.runtime._entry_guardrails import _EntryProjection
from dendrux.types import ImagePart, ImageRef

SECRET = "AKIAIOSFODNN7EXAMPLE"
IMAGE = ImagePart(ImageRef("0" * 64, "image/png", 1, 1, 1))


@pytest.mark.parametrize("history", [None, [], [ChatMessage.user("clean")]])
async def test_history_without_engine_retains_original_object(history):
    assert await _EntryProjection(None).history(history) is history


@pytest.mark.parametrize("history", [None, []])
async def test_empty_history_with_engine_retains_original_object(history):
    assert await _EntryProjection(GuardrailEngine([SecretDetection()])).history(history) is history


@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.parametrize("images", [False, True])
async def test_history_projects_text_without_mutating_caller_messages(blocked, images):
    text = SECRET if blocked else "clean"
    original = ChatMessage.user([text, IMAGE, text] if images else text)
    history = [ChatMessage.assistant("earlier"), original]
    entry = _EntryProjection(GuardrailEngine([SecretDetection()]))
    projected = await entry.history(history)
    assert projected is not history
    assert projected[0] == history[0]
    assert projected[1].role == original.role
    assert history[1] is original and text in original.content
    if blocked:
        assert entry.error and entry.location == "message 2 (role=user)"
        assert SECRET not in repr(projected)
        assert projected[1].content.startswith("[blocked:")
    else:
        assert entry.error is None and projected == history
    if images:
        assert projected[1].parts[1] is IMAGE
    else:
        assert projected[1].parts is None
