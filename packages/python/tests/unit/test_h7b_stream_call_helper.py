"""Direct checks of the H7b streamed-call pieces the loop pins cannot reach."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dendrux.loops._llm_call import EventRelay
from dendrux.loops.react import _StreamDeltas
from dendrux.loops.single import _require_single_done, _single_delta
from dendrux.types import (
    LLMResponse,
    RunEvent,
    RunEventType,
    StreamEvent,
    StreamEventType,
    ToolCall,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


def test_translators_skip_done() -> None:
    event = StreamEvent(type=StreamEventType.DONE, raw=LLMResponse(text="x"))
    assert _StreamDeltas(1)(event) is None
    assert _single_delta(event) is None


def test_single_done_rejects_tool_calls() -> None:
    response = LLMResponse(tool_calls=[ToolCall(name="t", params={})])
    with pytest.raises(RuntimeError, match=r"unexpected tool_calls from provider \(1 calls\)"):
        _require_single_done(response, 1)


async def test_relay_ends_when_the_call_absorbs_a_thrown_exception() -> None:
    seen: list[str] = []

    async def events() -> AsyncGenerator[RunEvent, None]:
        try:
            yield RunEvent(type=RunEventType.TEXT_DELTA, text="a")
        except RuntimeError as exc:
            seen.append(str(exc))

    relay = EventRelay(events())
    first = await relay.next()
    assert first is not None and first.text == "a"
    relay.throw(RuntimeError("thrown"))
    assert await relay.next() is None
    assert seen == ["thrown"] and relay.thrown is None
