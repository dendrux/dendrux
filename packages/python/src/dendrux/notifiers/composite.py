"""Composite notifier — fans out to multiple notifiers.

Lives in the core notifiers package (no FastAPI dependency) so it
can be used in script-mode runs without the ``http`` extra.
"""

from __future__ import annotations

import inspect
import logging
import traceback
from itertools import islice
from typing import TYPE_CHECKING, Any

from dendrux.errors import safe_exception_info
from dendrux.loops.base import BaseNotifier, LoopNotifier
from dendrux.runtime._tracelink import validate_trace_link

if TYPE_CHECKING:
    from dendrux.types import LLMResponse, Message, RunResult, ToolCall, ToolDef, ToolResult

logger = logging.getLogger(__name__)


class CompositeNotifier(BaseNotifier):
    """Fans out loop events to multiple notifiers.

    The loop sees one notifier; CompositeNotifier dispatches to all
    registered notifiers. If one fails, the others still fire.
    """

    def __init__(self, notifiers: list[LoopNotifier]) -> None:
        self._notifiers = list(notifiers)

    def trace_link(self, run_id: str) -> str | None:
        """Return the first valid run link exposed by a child notifier."""
        for notifier in self._notifiers:
            link = self._link(notifier, "trace_link", run_id)
            if link is not None:
                return link
        return None

    def current_span_link(self, run_id: str, tool_call_id: str) -> str | None:
        """Forward a single child's tool link; mixed composites use the run link."""
        if len(self._notifiers) != 1:
            return None
        return self._link(self._notifiers[0], "current_span_link", run_id, tool_call_id)

    @staticmethod
    def _link(notifier: LoopNotifier, name: str, *args: str) -> str | None:
        try:
            hook = getattr(notifier, name, None)
            return validate_trace_link(hook(*args)) if callable(hook) else None
        except Exception as exc:
            logger.warning("CompositeNotifier.%s failed: %s", name, type(exc).__name__)
            return None

    async def on_run_started(
        self,
        run_id: str,
        *,
        agent_name: str | None = None,
        agent_model: str | None = None,
        trace_link: str | None = None,
    ) -> None:
        for notifier in self._notifiers:
            try:
                kwargs: dict[str, Any] = {"agent_name": agent_name, "agent_model": agent_model}
                try:
                    params = dict(inspect.signature(notifier.on_run_started).parameters)
                except (TypeError, ValueError):
                    params = {}
                parameter = params.get("trace_link")
                if (
                    parameter is not None and parameter.kind != inspect.Parameter.POSITIONAL_ONLY
                ) or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                    kwargs["trace_link"] = trace_link
                await notifier.on_run_started(run_id, **kwargs)
            except Exception as exc:
                frames = "".join(
                    f"  File {frame.f_code.co_filename!r}, line {line}, in {frame.f_code.co_name}\n"
                    for frame, line in islice(traceback.walk_tb(exc.__traceback__), 32)
                )
                diagnostic = RuntimeError(f"{type(exc).__name__}\n{frames}")
                logger.warning(
                    "CompositeNotifier.on_run_started failed: %s",
                    type(exc).__name__,
                    exc_info=(RuntimeError, diagnostic, None),
                )

    async def on_run_finished(self, run_id: str, result: RunResult) -> None:
        for notifier in self._notifiers:
            try:
                await notifier.on_run_finished(run_id, result)
            except Exception:
                logger.warning(
                    "CompositeNotifier: on_run_finished failed", exc_info=safe_exception_info()
                )

    async def on_run_failed(
        self,
        run_id: str,
        error: BaseException,
        *,
        iteration: int | None = None,
    ) -> None:
        for notifier in self._notifiers:
            try:
                await notifier.on_run_failed(run_id, error, iteration=iteration)
            except Exception:
                logger.warning(
                    "CompositeNotifier: on_run_failed failed", exc_info=safe_exception_info()
                )

    async def on_message_appended(self, run_id: str, message: Message, iteration: int) -> None:
        for notifier in self._notifiers:
            try:
                await notifier.on_message_appended(run_id, message, iteration)
            except Exception:
                logger.warning(
                    "CompositeNotifier: on_message_appended failed", exc_info=safe_exception_info()
                )

    async def on_llm_call_started(
        self,
        run_id: str,
        iteration: int,
        *,
        semantic_messages: list[Message] | None = None,
        semantic_tools: list[ToolDef] | None = None,
    ) -> None:
        for notifier in self._notifiers:
            try:
                await notifier.on_llm_call_started(
                    run_id,
                    iteration,
                    semantic_messages=semantic_messages,
                    semantic_tools=semantic_tools,
                )
            except Exception:
                logger.warning(
                    "CompositeNotifier: on_llm_call_started failed", exc_info=safe_exception_info()
                )

    async def on_llm_call_completed(
        self,
        run_id: str,
        response: LLMResponse,
        iteration: int,
        *,
        semantic_messages: list[Message] | None = None,
        semantic_tools: list[ToolDef] | None = None,
        duration_ms: int | None = None,
        guardrail_findings: dict[str, Any] | None = None,
    ) -> None:
        for notifier in self._notifiers:
            try:
                await notifier.on_llm_call_completed(
                    run_id,
                    response,
                    iteration,
                    semantic_messages=semantic_messages,
                    semantic_tools=semantic_tools,
                    duration_ms=duration_ms,
                    guardrail_findings=guardrail_findings,
                )
            except Exception:
                logger.warning(
                    "CompositeNotifier: on_llm_call_completed failed",
                    exc_info=safe_exception_info(),
                )

    async def on_llm_call_failed(
        self,
        run_id: str,
        iteration: int,
        error: BaseException,
        *,
        duration_ms: int | None = None,
    ) -> None:
        for notifier in self._notifiers:
            try:
                await notifier.on_llm_call_failed(run_id, iteration, error, duration_ms=duration_ms)
            except Exception:
                logger.warning(
                    "CompositeNotifier: on_llm_call_failed failed", exc_info=safe_exception_info()
                )

    async def on_tool_started(self, run_id: str, tool_call: ToolCall, iteration: int) -> None:
        for notifier in self._notifiers:
            try:
                await notifier.on_tool_started(run_id, tool_call, iteration)
            except Exception:
                logger.warning(
                    "CompositeNotifier: on_tool_started failed", exc_info=safe_exception_info()
                )

    async def on_tool_completed(
        self,
        run_id: str,
        tool_call: ToolCall,
        tool_result: ToolResult,
        iteration: int,
    ) -> None:
        for notifier in self._notifiers:
            try:
                await notifier.on_tool_completed(run_id, tool_call, tool_result, iteration)
            except Exception:
                logger.warning(
                    "CompositeNotifier: on_tool_completed failed", exc_info=safe_exception_info()
                )

    async def on_governance_event(
        self,
        run_id: str,
        event_type: str,
        iteration: int,
        data: dict[str, Any],
        *,
        correlation_id: str | None = None,
    ) -> None:
        for notifier in self._notifiers:
            try:
                await notifier.on_governance_event(
                    run_id, event_type, iteration, data, correlation_id=correlation_id
                )
            except Exception:
                logger.warning(
                    "CompositeNotifier: on_governance_event failed", exc_info=safe_exception_info()
                )
