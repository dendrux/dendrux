"""OpenTelemetry notifier for Dendrux.

Emits spans on the developer's existing OTel tracer following the
GenAI semantic conventions. Dendrux runs become visible in whichever
OTel backend the host application already exports to — no extra
setup, no exporter ownership.

Three-level span tree::

    invoke_agent <agent_name>          (run-level root)
    ├── chat <model>                   (per LLM call)
    └── execute_tool <tool_name>       (per tool call)

If a span is already active when the run starts (e.g. a FastAPI
request span), the ``invoke_agent`` span attaches as its child
automatically via OTel context propagation.

Failure policy: this notifier is fail-open. Any exception raised by
the OTel SDK or tracer is swallowed and logged as a warning — runs
must never die because observability is broken.

By default no prompt content, completion text, or tool arguments are
captured. Use ``include_messages=True`` / ``include_tool_params=True``
to opt in (this bypasses Dendrux's PII guardrail redaction — only flip
in trusted environments).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from dendrux.loops.base import BaseNotifier

if TYPE_CHECKING:
    from dendrux.types import LLMResponse, Message, RunResult, ToolCall, ToolDef, ToolResult

try:
    from opentelemetry import context as otel_context
    from opentelemetry import trace
    from opentelemetry.trace import Span, Status, StatusCode, Tracer

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - import guard
    _OTEL_AVAILABLE = False
    Span = Any  # type: ignore[assignment,misc]
    Tracer = Any  # type: ignore[assignment,misc]


_LOG = logging.getLogger(__name__)

_FRAMEWORK = "dendrux"
_OP_INVOKE = "invoke_agent"
_OP_CHAT = "chat"
_OP_TOOL = "execute_tool"


class OpenTelemetryNotifier(BaseNotifier):
    """LoopNotifier that emits OpenTelemetry spans for Dendrux runs.

    Args:
        tracer: An OTel ``Tracer`` instance. If ``None``, a tracer is
            obtained from the global provider via ``trace.get_tracer("dendrux")``.
            Pass an explicit tracer when you want to control the
            instrumentation scope name or version.
        include_tool_params: When ``True``, serialize tool call ``params``
            into ``dendrux.tool.params`` as a JSON string. Bypasses
            guardrail redaction. Defaults to ``False``.
        include_messages: When ``True``, attach the LLM's response text
            as ``gen_ai.completion`` on chat spans. Bypasses guardrail
            redaction. Defaults to ``False``.

    Example::

        from dendrux.notifiers.otel import OpenTelemetryNotifier

        result = await agent.run(
            "summarize this PDF",
            extra_notifier=OpenTelemetryNotifier(),
        )
    """

    def __init__(
        self,
        tracer: Tracer | None = None,
        *,
        include_tool_params: bool = False,
        include_messages: bool = False,
    ) -> None:
        if not _OTEL_AVAILABLE:
            msg = (
                "OpenTelemetryNotifier requires the OpenTelemetry API. "
                "Install it with: pip install dendrux[otel]"
            )
            raise ImportError(msg)

        self._tracer: Tracer = tracer if tracer is not None else trace.get_tracer(_FRAMEWORK)
        self._include_tool_params = include_tool_params
        self._include_messages = include_messages

        # Per-run span tracking. Concurrent runs are disambiguated by run_id;
        # we never rely on contextvars for our own bookkeeping.
        self._run_spans: dict[str, Span] = {}
        self._llm_spans: dict[tuple[str, int], Span] = {}
        self._tool_spans: dict[str, Span] = {}
        self._run_models: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Run-level (root span)
    # ------------------------------------------------------------------

    async def on_run_started(
        self,
        run_id: str,
        *,
        agent_name: str | None = None,
        agent_model: str | None = None,
    ) -> None:
        try:
            span_name = f"{_OP_INVOKE} {agent_name}" if agent_name else _OP_INVOKE
            attributes: dict[str, Any] = {
                "gen_ai.operation.name": _OP_INVOKE,
                "dendrux.framework": _FRAMEWORK,
                "dendrux.run.id": run_id,
            }
            if agent_name:
                attributes["gen_ai.agent.name"] = agent_name
            if agent_model:
                attributes["gen_ai.request.model"] = agent_model
                self._run_models[run_id] = agent_model

            span = self._tracer.start_span(span_name, attributes=attributes)
            self._run_spans[run_id] = span
        except Exception as exc:  # noqa: BLE001 - fail open
            _LOG.warning("OpenTelemetryNotifier.on_run_started failed: %s", exc)

    async def on_run_finished(self, run_id: str, result: RunResult) -> None:
        span = self._run_spans.pop(run_id, None)
        self._run_models.pop(run_id, None)
        if span is None:
            return
        try:
            status_value = getattr(result.status, "value", str(result.status))
            span.set_attribute("dendrux.run.status", status_value)
            span.set_status(Status(StatusCode.OK))
            span.end()
        except Exception as exc:  # noqa: BLE001 - fail open
            _LOG.warning("OpenTelemetryNotifier.on_run_finished failed: %s", exc)

    async def on_run_failed(
        self,
        run_id: str,
        error: BaseException,
        *,
        iteration: int | None = None,
    ) -> None:
        span = self._run_spans.pop(run_id, None)
        self._run_models.pop(run_id, None)
        if span is None:
            return
        try:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            if iteration is not None:
                span.set_attribute("dendrux.run.iteration", iteration)
            span.end()
        except Exception as exc:  # noqa: BLE001 - fail open
            _LOG.warning("OpenTelemetryNotifier.on_run_failed failed: %s", exc)

    # ------------------------------------------------------------------
    # LLM call (chat span)
    # ------------------------------------------------------------------

    async def on_llm_call_started(
        self,
        run_id: str,
        iteration: int,
        *,
        semantic_messages: list[Message] | None = None,
        semantic_tools: list[ToolDef] | None = None,
    ) -> None:
        try:
            model = self._run_models.get(run_id)
            span_name = f"{_OP_CHAT} {model}" if model else _OP_CHAT
            attributes: dict[str, Any] = {
                "gen_ai.operation.name": _OP_CHAT,
                "dendrux.run.id": run_id,
                "dendrux.iteration": iteration,
            }
            if model:
                attributes["gen_ai.request.model"] = model

            span = self._tracer.start_span(
                span_name,
                context=self._child_context(run_id),
                attributes=attributes,
            )
            self._llm_spans[(run_id, iteration)] = span
        except Exception as exc:  # noqa: BLE001 - fail open
            _LOG.warning("OpenTelemetryNotifier.on_llm_call_started failed: %s", exc)

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
        span = self._llm_spans.pop((run_id, iteration), None)
        if span is None:
            return
        try:
            usage = getattr(response, "usage", None)
            if usage is not None:
                input_tokens = getattr(usage, "input_tokens", None)
                output_tokens = getattr(usage, "output_tokens", None)
                if input_tokens is not None:
                    span.set_attribute("gen_ai.usage.input_tokens", int(input_tokens))
                if output_tokens is not None:
                    span.set_attribute("gen_ai.usage.output_tokens", int(output_tokens))

            if duration_ms is not None:
                span.set_attribute("dendrux.llm.duration_ms", int(duration_ms))

            if guardrail_findings:
                span.set_attribute("dendrux.guardrails.hit_count", len(guardrail_findings))

            if self._include_messages and getattr(response, "text", None):
                span.set_attribute("gen_ai.completion", str(response.text))

            span.set_status(Status(StatusCode.OK))
            span.end()
        except Exception as exc:  # noqa: BLE001 - fail open
            _LOG.warning("OpenTelemetryNotifier.on_llm_call_completed failed: %s", exc)

    async def on_llm_call_failed(
        self,
        run_id: str,
        iteration: int,
        error: BaseException,
        *,
        duration_ms: int | None = None,
    ) -> None:
        span = self._llm_spans.pop((run_id, iteration), None)
        if span is None:
            return
        try:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            if duration_ms is not None:
                span.set_attribute("dendrux.llm.duration_ms", int(duration_ms))
            span.end()
        except Exception as exc:  # noqa: BLE001 - fail open
            _LOG.warning("OpenTelemetryNotifier.on_llm_call_failed failed: %s", exc)

    # ------------------------------------------------------------------
    # Tool call (execute_tool span)
    # ------------------------------------------------------------------

    async def on_tool_started(self, run_id: str, tool_call: ToolCall, iteration: int) -> None:
        try:
            span_name = f"{_OP_TOOL} {tool_call.name}"
            attributes: dict[str, Any] = {
                "gen_ai.operation.name": _OP_TOOL,
                "dendrux.run.id": run_id,
                "dendrux.iteration": iteration,
                "dendrux.tool.name": tool_call.name,
                "dendrux.tool.call_id": tool_call.id,
            }
            if self._include_tool_params and tool_call.params is not None:
                attributes["dendrux.tool.params"] = json.dumps(tool_call.params, default=str)

            span = self._tracer.start_span(
                span_name,
                context=self._child_context(run_id),
                attributes=attributes,
            )
            self._tool_spans[tool_call.id] = span
        except Exception as exc:  # noqa: BLE001 - fail open
            _LOG.warning("OpenTelemetryNotifier.on_tool_started failed: %s", exc)

    async def on_tool_completed(
        self,
        run_id: str,
        tool_call: ToolCall,
        tool_result: ToolResult,
        iteration: int,
    ) -> None:
        span = self._tool_spans.pop(tool_call.id, None)
        if span is None:
            return
        try:
            success = bool(getattr(tool_result, "success", True))
            span.set_attribute("dendrux.tool.success", success)
            duration_ms = getattr(tool_result, "duration_ms", None)
            if duration_ms is not None:
                span.set_attribute("dendrux.tool.duration_ms", int(duration_ms))

            if not success:
                err_msg = getattr(tool_result, "error", None) or "tool returned success=false"
                span.set_status(Status(StatusCode.ERROR, str(err_msg)))
            else:
                span.set_status(Status(StatusCode.OK))
            span.end()
        except Exception as exc:  # noqa: BLE001 - fail open
            _LOG.warning("OpenTelemetryNotifier.on_tool_completed failed: %s", exc)

    # ------------------------------------------------------------------
    # Governance events (span events on the run-level span)
    # ------------------------------------------------------------------

    async def on_governance_event(
        self,
        run_id: str,
        event_type: str,
        iteration: int,
        data: dict[str, Any],
        *,
        correlation_id: str | None = None,
    ) -> None:
        span = self._run_spans.get(run_id)
        if span is None:
            return
        try:
            attrs: dict[str, Any] = {"dendrux.iteration": iteration}
            if correlation_id:
                attrs["dendrux.correlation_id"] = correlation_id
            for k, v in data.items():
                if isinstance(v, (str, bool, int, float)):
                    attrs[f"dendrux.governance.{k}"] = v
            span.add_event(f"governance.{event_type}", attributes=attrs)
        except Exception as exc:  # noqa: BLE001 - fail open
            _LOG.warning("OpenTelemetryNotifier.on_governance_event failed: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _child_context(self, run_id: str) -> Any:
        """Return an OTel context with the run span set as parent.

        Falls back to the current ambient context if the run span is
        unknown (e.g. lifecycle ordering bug or a stray emission after
        cleanup) so we never lose the span entirely.
        """
        parent = self._run_spans.get(run_id)
        if parent is None:
            return otel_context.get_current()
        return trace.set_span_in_context(parent)
