"""PersistenceObserver — bridges the loop's observer hooks to the StateStore.

Implements LoopObserver by writing each event to the StateStore.
The runner creates this and passes it to the loop when persistence is active.

Failure policy: log and continue. Observer failures must not kill agent runs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dendrite.loops.base import LoopObserver

if TYPE_CHECKING:
    from dendrite.runtime.state import StateStore
    from dendrite.types import LLMResponse, Message, ToolCall, ToolResult, ToolTarget

logger = logging.getLogger(__name__)


class PersistenceObserver(LoopObserver):
    """Writes loop events to a StateStore for persistence.

    Tracks order_index internally to ensure react_traces are ordered
    correctly within a single run.
    """

    def __init__(
        self,
        state_store: StateStore,
        run_id: str,
        *,
        model: str | None = None,
        provider_name: str | None = None,
        target_lookup: dict[str, ToolTarget] | None = None,
    ) -> None:
        self._store = state_store
        self._run_id = run_id
        self._model = model
        self._provider_name = provider_name
        self._target_lookup = target_lookup or {}
        self._order_index = 0

    async def on_message_appended(self, message: Message, iteration: int) -> None:
        """Persist a message to react_traces."""
        # Build meta with tool_calls info for assistant messages
        meta = dict(message.meta) if message.meta else {}
        meta["iteration"] = iteration

        if message.tool_calls:
            meta["tool_calls"] = [
                {
                    "id": tc.id,
                    "provider_tool_call_id": tc.provider_tool_call_id,
                    "name": tc.name,
                    "params": tc.params,
                }
                for tc in message.tool_calls
            ]

        if message.call_id:
            meta["call_id"] = message.call_id
        if message.name:
            meta["tool_name"] = message.name

        # Serialize content — for tool messages the content is already
        # a JSON string (the tool result payload)
        content = message.content

        await self._store.save_trace(
            self._run_id,
            message.role.value,
            content,
            order_index=self._order_index,
            meta=meta,
        )
        self._order_index += 1

    async def on_llm_call_completed(self, response: LLMResponse, iteration: int) -> None:
        """Persist token usage from an LLM call."""
        await self._store.save_usage(
            self._run_id,
            iteration_index=iteration,
            usage=response.usage,
            model=self._model,
            provider=self._provider_name,
        )

    async def on_tool_completed(
        self, tool_call: ToolCall, tool_result: ToolResult, iteration: int
    ) -> None:
        """Persist a tool call record."""
        # Parse params for JSON column
        params = tool_call.params if tool_call.params else None

        await self._store.save_tool_call(
            self._run_id,
            tool_call_id=tool_call.id,
            provider_tool_call_id=tool_call.provider_tool_call_id,
            tool_name=tool_call.name,
            target=self._target_lookup.get(tool_call.name, "server"),
            params=params,
            result_payload=tool_result.payload,
            success=tool_result.success,
            duration_ms=tool_result.duration_ms,
            iteration_index=iteration,
            error_message=tool_result.error,
        )
