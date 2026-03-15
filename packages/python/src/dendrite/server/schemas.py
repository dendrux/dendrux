"""Pydantic models for the Dendrite server API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class CreateRunRequest(BaseModel):
    """POST /runs request body."""

    agent_name: str
    input: str
    tenant_id: str | None = None


class CreateRunResponse(BaseModel):
    """POST /runs response body."""

    run_id: str
    status: str
    token: str | None = None  # Run-scoped HMAC token (Group 4)


class RunStatusResponse(BaseModel):
    """GET /runs/{run_id} response body."""

    run_id: str
    status: str
    answer: str | None = None
    error: str | None = None
    iteration_count: int = 0
    pending_tool_calls: list[PendingToolCall] | None = None


class PendingToolCall(BaseModel):
    """A tool call waiting for client execution."""

    tool_call_id: str
    tool_name: str
    params: dict[str, Any] | None = None
    target: str = "client"


class ToolResultSubmission(BaseModel):
    """One tool result in a POST /runs/{run_id}/tool-results request."""

    tool_call_id: str
    tool_name: str
    result: str  # JSON string payload
    success: bool = True
    error: str | None = None


class SubmitToolResultsRequest(BaseModel):
    """POST /runs/{run_id}/tool-results request body."""

    tool_results: list[ToolResultSubmission]


class SubmitInputRequest(BaseModel):
    """POST /runs/{run_id}/input request body."""

    user_input: str


class ResumeResponse(BaseModel):
    """Response after resuming a paused run."""

    run_id: str
    status: str


# Rebuild forward refs for nested models
RunStatusResponse.model_rebuild()
