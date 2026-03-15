"""PersistenceObserver — bridges the loop's observer hooks to the StateStore.

Implements LoopObserver by writing each event to the StateStore.
The runner creates this and passes it to the loop when persistence is active.

Failure policy: log and continue. Observer failures must not kill agent runs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from dendrite.loops.base import LoopObserver

if TYPE_CHECKING:
    from collections.abc import Callable

    from dendrite.runtime.state import StateStore
    from dendrite.types import LLMResponse, Message, ToolCall, ToolResult, ToolTarget

logger = logging.getLogger(__name__)


def _identity(s: str) -> str:
    return s


def _redact_value(v: Any, redact: Callable[[str], str]) -> Any:
    """Recursively apply a string redactor to all string values."""
    if isinstance(v, str):
        return redact(v)
    if isinstance(v, dict):
        return {k: _redact_value(val, redact) for k, val in v.items()}
    if isinstance(v, list):
        return [_redact_value(item, redact) for item in v]
    return v


def _redact_dict(d: dict[str, Any], redact: Callable[[str], str]) -> dict[str, Any]:
    """Recursively apply a string redactor to all string values in a dict."""
    return {k: _redact_value(v, redact) for k, v in d.items()}


class PersistenceObserver(LoopObserver):
    """Writes loop events to a StateStore for persistence.

    Tracks order_index internally to ensure react_traces are ordered
    correctly within a single run.

    Args:
        redact: Optional callable that receives a string and returns a
            scrubbed/sanitized string. Applied to trace content, tool
            params (string values), tool result payloads, and error
            messages before persistence. The function receives plain
            strings and must return plain strings (not JSON).
    """

    def __init__(
        self,
        state_store: StateStore,
        run_id: str,
        *,
        model: str | None = None,
        provider_name: str | None = None,
        target_lookup: dict[str, ToolTarget] | None = None,
        redact: Callable[[str], str] | None = None,
        initial_order_index: int = 0,
    ) -> None:
        self._store = state_store
        self._run_id = run_id
        self._model = model
        self._provider_name = provider_name
        self._target_lookup = target_lookup or {}
        self._redact = redact or _identity
        self._order_index = initial_order_index

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
                    "params": _redact_dict(tc.params, self._redact) if tc.params else tc.params,
                }
                for tc in message.tool_calls
            ]

        if message.call_id:
            meta["call_id"] = message.call_id
        if message.name:
            meta["tool_name"] = message.name

        # Apply redaction before persistence
        content = self._redact(message.content)

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
        """Persist a tool call record with optional redaction."""
        # Redact params (string values only) before persistence
        params = tool_call.params if tool_call.params else None
        if params is not None:
            params = _redact_dict(params, self._redact)

        # Redact tool result payload and error message
        redacted_payload = self._redact(tool_result.payload)
        redacted_error = self._redact(tool_result.error) if tool_result.error else None

        await self._store.save_tool_call(
            self._run_id,
            tool_call_id=tool_call.id,
            provider_tool_call_id=tool_call.provider_tool_call_id,
            tool_name=tool_call.name,
            target=self._target_lookup.get(tool_call.name, "server"),
            params=params,
            result_payload=redacted_payload,
            success=tool_result.success,
            duration_ms=tool_result.duration_ms,
            iteration_index=iteration,
            error_message=redacted_error,
        )
