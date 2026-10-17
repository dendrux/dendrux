"""A batch provider with scripted responses and an optional gate.

Used by run-control tests to hold a run inside ``complete()``
deterministically: when call ``gate_at`` starts, ``entered`` is set and the
call blocks on ``gate``. ``cancelled`` flips if the blocked call receives a
``CancelledError``. ``messages`` records what each call was sent.
"""

from __future__ import annotations

import asyncio
from typing import Any

from dendrux.llm.base import LLMProvider
from dendrux.types import LLMResponse, ProviderCapabilities, ToolCall


def tool_call(name: str, params: dict[str, Any] | None = None, *, pid: str) -> LLMResponse:
    """An LLM response requesting one tool call."""
    return LLMResponse(
        tool_calls=[ToolCall(name=name, params=params or {}, provider_tool_call_id=pid)]
    )


def tool_calls(*calls: tuple[str, dict[str, Any], str]) -> LLMResponse:
    """An LLM response requesting several tool calls ``(name, params, provider id)``."""
    return LLMResponse(
        tool_calls=[ToolCall(name=n, params=p, provider_tool_call_id=pid) for n, p, pid in calls]
    )


class ScriptedLLM(LLMProvider):
    capabilities = ProviderCapabilities(supports_native_tools=True, supports_tool_call_ids=True)

    def __init__(self, responses: list[LLMResponse], *, gate_at: int | None = None) -> None:
        self._responses = list(responses)
        self.gate_at = gate_at
        self.gate = asyncio.Event()
        self.entered = asyncio.Event()
        self.cancelled = False
        self.calls = 0
        self.messages: list[list[Any]] = []

    @property
    def model(self) -> str:
        return "scripted"

    async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> LLMResponse:
        index = self.calls
        self.calls += 1
        self.messages.append(list(messages))
        if self.gate_at is not None and index == self.gate_at:
            self.entered.set()
            try:
                await self.gate.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        if index >= len(self._responses):
            raise IndexError(f"ScriptedLLM exhausted after {len(self._responses)} responses")
        response = self._responses[index]
        response.model = self.model
        return response

    def contents(self, call: int) -> list[str]:
        """Message contents sent on call ``call``."""
        return [m.content for m in self.messages[call]]
