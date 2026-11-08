"""Scripted sales conversation and file-based effect receipts for example tests."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any

import pytest

from dendrux.llm._retry_telemetry import begin_call_attempt_tracking, end_call_attempt_tracking
from dendrux.types import (
    LLMResponse,
    ProviderCapabilities,
    StreamEvent,
    StreamEventType,
    UsageStats,
)
from examples._sales_sheet import SalesSheet
from tests._helpers.scripted_provider import ScriptedLLM, tool_call, tool_calls

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable
    from pathlib import Path


class SalesProvider(ScriptedLLM):
    """Read, request a region write and notice, then explain the approval decision."""

    capabilities = ProviderCapabilities(
        supports_native_tools=True, supports_tool_call_ids=True, supports_streaming=True
    )

    def __init__(self, *, approved: bool = True) -> None:
        self.final_answer = (
            "Normalized the region to NORTH, preserved amounts, and posted the review notice."
            if approved
            else "Cleanup declined; no sales cells changed and no notice posted."
        )
        super().__init__(
            [
                tool_call("read_sheet", pid="sales-read"),
                tool_calls(
                    ("write_region", {"row_number": 1, "region": "NORTH"}, "sales-write"),
                    ("post_review_notice", {}, "sales-notice"),
                ),
                LLMResponse(text=self.final_answer),
            ]
        )
        self.closed_streams = 0

    async def complete(self, messages: Any, tools: Any = None, **kwargs: Any) -> LLMResponse:
        """Consume the provider generator on this task, including its finalization."""
        response = None
        async with aclosing(self.complete_stream(messages, tools, **kwargs)) as events:
            async for event in events:
                if event.type == StreamEventType.DONE:
                    response = event.raw
        assert isinstance(response, LLMResponse)
        return response

    async def complete_stream(
        self, messages: Any, tools: Any = None, **kwargs: Any
    ) -> AsyncGenerator[StreamEvent, None]:
        """Emit scripted events inside the real provider attempt ContextVar scope."""
        task = asyncio.current_task()
        token = begin_call_attempt_tracking()
        try:
            response = await super().complete(messages, tools, **kwargs)
            response.usage = UsageStats(input_tokens=10, output_tokens=5, total_tokens=15)
            if response.text:
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=response.text)
            for call in response.tool_calls or []:
                yield StreamEvent(
                    type=StreamEventType.TOOL_USE_START,
                    tool_name=call.name,
                    tool_call_id=call.provider_tool_call_id,
                )
                yield StreamEvent(
                    type=StreamEventType.TOOL_USE_END,
                    tool_call=call,
                    tool_name=call.name,
                    tool_call_id=call.provider_tool_call_id,
                )
            yield StreamEvent(type=StreamEventType.DONE, raw=response)
        finally:
            end_call_attempt_tracking(token)
            assert asyncio.current_task() is task
            self.closed_streams += 1


@dataclass
class EffectLedger:
    """Append effect receipts to an app-owned file outside Dendrux transactions."""

    path: Path

    def record(self, name: str) -> None:
        """Record a completed adapter effect without database coupling."""
        with self.path.open("a", encoding="utf-8") as file:
            file.write(name + "\n")

    def entries(self) -> list[str]:
        """Return the small fixture's receipts in execution order."""
        return self.path.read_text(encoding="utf-8").splitlines() if self.path.exists() else []

    def wrap(self, function: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        """Retain tool metadata and signature while recording successful calls."""

        @wraps(function)
        async def recorded(*args: Any, **kwargs: Any) -> Any:
            result = await function(*args, **kwargs)
            self.record(function.__name__)
            return result

        return recorded


@dataclass
class SalesFixture:
    """The example adapter and its independently persisted effect ledger."""

    sheet: SalesSheet
    ledger: EffectLedger

    @property
    def tools(self) -> list[Callable[..., Awaitable[Any]]]:
        """Use the real example tools with receipts around the two effects."""
        return [
            self.sheet.read_sheet,
            self.ledger.wrap(self.sheet.write_region),
            self.ledger.wrap(self.sheet.post_review_notice),
        ]


@pytest.fixture
def sales_sheet(tmp_path: Path) -> SalesFixture:
    """Initialize an isolated three-row sales directory for a normal journey."""
    sheet = SalesSheet(tmp_path)
    sheet.initialize()
    return SalesFixture(sheet, EffectLedger(tmp_path / "effects.log"))
