"""Managed subagents: turning one terminal child's saved outcome into its parent's report.

The parent admits a child's outcome as it admits any tool result: the report
is built from the saved terminal record (codes and counts only on failure),
passed through the parent's ``tool_result_transform``, size-checked and then
projected through the parent's guardrails. The claim and the completion
transaction that make the report durable stay in :mod:`dendrux.runtime._managed`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from dendrux import _codes
from dendrux.runtime._managed_route import _Waiting  # noqa: TC001 - move-proof binding
from dendrux.types import RunStatus, ToolResult

if TYPE_CHECKING:
    from dendrux.agent import Agent
    from dendrux.runtime.state import RunRecord
    from dendrux.types import ToolCall

logger = logging.getLogger("dendrux.runtime._managed")

REPORT_TOO_LARGE = "managed_report_too_large"
CHILD_FAILED = "managed_child_failed"
CHILD_CANCELLED = "managed_child_cancelled"
REPORT_INVALID = "managed_report_invalid"


def child_report(record: RunRecord, child: Agent, *, limit: int) -> ToolResult:
    """The parent's tool result for a terminal child; codes and counts only on failure."""
    status = record.status
    if status == RunStatus.CANCELLED.value:
        return _failed_report(child.name, CHILD_CANCELLED, record.id)
    if status != RunStatus.SUCCESS.value:
        return _failed_report(child.name, CHILD_FAILED, record.id)
    try:
        payload = _report_payload(record.answer, getattr(child, "_output_type", None))
    except (TypeError, ValueError):
        return _failed_report(child.name, REPORT_INVALID, record.id)
    size = len(payload.encode("utf-8"))
    if size > limit:
        _codes.log_code(
            logger, _codes.MANAGED_REPORT_TOO_LARGE, run_id=record.id, bytes=size, limit=limit
        )
        return _failed_report(child.name, REPORT_TOO_LARGE, record.id, bytes=size, limit=limit)
    return ToolResult(name=child.name, call_id="", payload=payload)


def _report_payload(answer: str | None, output_type: Any) -> str:
    if output_type is None:
        return json.dumps(answer or "", ensure_ascii=False)
    value = json.loads(answer or "")
    output_type.model_validate(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


_FAILURE_TEXT = {
    CHILD_FAILED: "The subagent did not finish successfully.",
    CHILD_CANCELLED: "The subagent was cancelled.",
    REPORT_INVALID: "The subagent's output did not match its declared type.",
    REPORT_TOO_LARGE: "The subagent's report is larger than the tool result limit.",
}


_FAILURE_CODES = {
    CHILD_FAILED: _codes.MANAGED_CHILD_FAILED,
    CHILD_CANCELLED: _codes.MANAGED_CHILD_CANCELLED,
    REPORT_INVALID: _codes.MANAGED_REPORT_INVALID,
}


def _failed_report(name: str, code: str, child_run_id: str, **counts: int) -> ToolResult:
    message = _FAILURE_TEXT[code]
    if code in _FAILURE_CODES:
        _codes.log_code(logger, _FAILURE_CODES[code], run_id=child_run_id)
    body = {"error": message, "code": code, "child_run_id": child_run_id, **counts}
    return ToolResult(
        name=name,
        call_id="",
        payload=json.dumps(body),
        success=False,
        error=message,
        meta={"code": code},
    )


@dataclass(frozen=True)
class AcceptedReport:
    """The parent-approved report: transformed, size-checked and projected (in memory)."""

    result: ToolResult
    observation: Any
    pii_mapping: dict[str, str] | None
    blocked: str | None


async def accept_report(waiting: _Waiting, record: RunRecord, store: Any) -> AcceptedReport:
    """Admit a child's saved outcome as the parent admits any tool result.

    The parent's ``tool_result_transform`` runs once with the original call;
    the size refusal applies to the transformed serialized payload; the
    parent's guardrails project it (PII placeholders) or block it. Only this
    representation becomes durable; recovery never repeats the transform.
    """
    parent = waiting.parent_agent
    limit = parent._limits.max_tool_result_bytes
    report = replace(child_report(record, waiting.child, limit=limit), call_id=waiting.call.id)
    report = await _transformed(parent, waiting.call, report, limit, record.id)
    return await project_report(parent, store, waiting.parent_run_id, report)


async def _transformed(
    parent: Agent, call: ToolCall, report: ToolResult, limit: int, child_run_id: str
) -> ToolResult:
    from dendrux.loops.react import _apply_result_transform

    transform = parent.tool_result_transform
    if transform is None or not report.success:
        return report
    try:
        lookups = await parent.get_tool_lookups()
        value = await _apply_result_transform(transform, call, lookups, json.loads(report.payload))
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except Exception:
        return replace(_failed_report(report.name, REPORT_INVALID, child_run_id), call_id=call.id)
    size = len(payload.encode("utf-8"))
    if size > limit:
        _codes.log_code(
            logger, _codes.MANAGED_REPORT_TOO_LARGE, run_id=child_run_id, bytes=size, limit=limit
        )
        failed = _failed_report(
            report.name, REPORT_TOO_LARGE, child_run_id, bytes=size, limit=limit
        )
        return replace(failed, call_id=call.id)
    return replace(report, payload=payload)


async def project_report(
    parent: Agent, store: Any, run_id: str, report: ToolResult
) -> AcceptedReport:
    """The parent's guardrail projection of ``report`` (also used by recovery)."""
    from dendrux.loops._helpers import _project_tool_message
    from dendrux.types import tool_result_message

    message = tool_result_message(report)
    if not parent.guardrails:
        return AcceptedReport(report, message, None, None)
    from dendrux.guardrails import GuardrailEngine

    engine = GuardrailEngine(
        parent.guardrails, pii_mapping=await store.get_pii_mapping(run_id), run_id=run_id
    )
    projected, _findings, blocked = await _project_tool_message(engine, report, message)
    if blocked is not None:
        report = replace(report, payload=projected.content, error=None, meta={}, notes=())
    return AcceptedReport(report, projected, engine.get_pii_mapping(), blocked)


AcceptedReport.__module__ = "dendrux.runtime._managed"
