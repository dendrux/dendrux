"""SQLAlchemy models — the 6 core tables.

Tables:
    agent_runs       — One row per run() call. The anchor for everything.
    react_traces     — Canonical conversation history for one agent run.
    tool_calls       — Every tool invocation with params, result, timing.
    token_usage      — Per-LLM-call token counts and cost (legacy, kept for backcompat).
    llm_interactions — Per-LLM-call record with semantic + provider payloads (Sprint 3.5).
    run_events       — Append-only state transition log for observability.

Each sub-agent spawn gets its own agent_run linked via parent_run_id.
No instance_id or delegation columns on react_traces — isolation is
structural (different agent_run_id = different traces).

Deferred tables (added via migration in their sprint):
    sandbox_runs  — Sprint 6 (Sandbox)
"""

from __future__ import annotations

import datetime  # noqa: TC003 — needed at runtime for SQLAlchemy Mapped[] resolution
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all Dendrux models."""

    type_annotation_map = {
        dict[str, Any]: JSON,
    }


class AgentRun(Base):
    """Individual agent execution. The heartbeat of Dendrux.

    One row per run() call. Sub-agent spawns get their own row
    linked via parent_run_id. Developer links their world to ours
    via meta (opaque JSON blob — Dendrux stores it, never reads it).
    """

    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    tenant_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    agent_name: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(50), default="pending")
    input_data: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    output_data: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    iteration_count: Mapped[int] = mapped_column(Integer, default=0)

    model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    strategy: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Sub-agent tree structure
    parent_run_id: Mapped[str | None] = mapped_column(
        String(26), ForeignKey("agent_runs.id", ondelete="SET NULL"), nullable=True
    )
    delegation_level: Mapped[int] = mapped_column(Integer, default=0)

    # Retry lineage — separate from delegation tree
    retry_of_run_id: Mapped[str | None] = mapped_column(
        String(26), ForeignKey("agent_runs.id", ondelete="SET NULL"), nullable=True
    )

    # Aggregate token usage (rolled up from token_usage table)
    total_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_cost_usd: Mapped[float | None] = mapped_column(Numeric(10, 6), nullable=True)
    total_cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_cache_creation_tokens: Mapped[int] = mapped_column(Integer, default=0)

    # Developer's linking data — Dendrux stores it, never reads it
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Pause state — execution state for resume (D1: not observable, not redacted)
    # Cleared on finalize. Contains unredacted history for correct LLM resume.
    pause_data: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Guardrail PII mapping — audit-first, NOT cleared on finalize
    pii_mapping: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Error details (populated on ERROR status)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Sweep / recovery columns
    last_progress_at: Mapped[datetime.datetime | None] = mapped_column(DateTime, nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Idempotency columns — duplicate run prevention
    idempotency_key: Mapped[str | None] = mapped_column(String(255), nullable=True, unique=True)
    idempotency_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Cooperative cancellation flag — runner observes at checkpoints,
    # cleared on terminal finalize so a stale True never lingers.
    cancel_requested: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="0"
    )

    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    traces: Mapped[list[ReactTrace]] = relationship(
        back_populates="agent_run", cascade="all, delete-orphan"
    )
    tool_call_records: Mapped[list[ToolCallRecord]] = relationship(
        back_populates="agent_run", cascade="all, delete-orphan"
    )
    token_usages: Mapped[list[TokenUsage]] = relationship(
        back_populates="agent_run", cascade="all, delete-orphan"
    )
    llm_interactions: Mapped[list[LLMInteraction]] = relationship(
        back_populates="agent_run", cascade="all, delete-orphan"
    )
    run_events: Mapped[list[RunEvent]] = relationship(
        back_populates="agent_run", cascade="all, delete-orphan"
    )
    child_runs: Mapped[list[AgentRun]] = relationship(
        back_populates="parent_run",
        foreign_keys=[parent_run_id],
    )
    parent_run: Mapped[AgentRun | None] = relationship(
        back_populates="child_runs", remote_side=[id], foreign_keys=[parent_run_id]
    )

    __table_args__ = (
        Index("ix_agent_runs_parent_run_id", "parent_run_id"),
        Index("ix_agent_runs_status", "status"),
        Index("ix_agent_runs_created_at", "created_at"),
    )


class ReactTrace(Base):
    """Internal reasoning trace for one agent run.

    The canonical conversation history — USER, ASSISTANT, and TOOL messages
    in order.

    Each sub-agent's traces are naturally isolated: different agent_run_id
    = different traces. No instance_id or delegation columns needed.
    """

    __tablename__ = "react_traces"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    agent_run_id: Mapped[str] = mapped_column(
        String(26), ForeignKey("agent_runs.id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(String(20))  # user/assistant/tool
    content: Mapped[str] = mapped_column(Text)
    order_index: Mapped[int] = mapped_column(Integer)
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    agent_run: Mapped[AgentRun] = relationship(back_populates="traces")

    __table_args__ = (Index("ix_react_traces_agent_run_id_order", "agent_run_id", "order_index"),)


class ToolCallRecord(Base):
    """Every tool invocation with full context.

    Records both Dendrux's stable correlation ID (tool_call_id) and
    the provider's native ID (provider_tool_call_id) for round-tripping.
    """

    __tablename__ = "tool_calls"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    agent_run_id: Mapped[str] = mapped_column(
        String(26), ForeignKey("agent_runs.id", ondelete="CASCADE")
    )
    tool_call_id: Mapped[str] = mapped_column(String(26))  # Dendrux ULID
    provider_tool_call_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    tool_name: Mapped[str] = mapped_column(String(255))
    target: Mapped[str] = mapped_column(String(50), default="server")
    params: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    iteration_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    agent_run: Mapped[AgentRun] = relationship(back_populates="tool_call_records")

    __table_args__ = (Index("ix_tool_calls_agent_run_id", "agent_run_id"),)


class TokenUsage(Base):
    """Per-LLM-call token counts and cost.

    One row per provider.complete() call — iteration-level granularity.
    Powers cost dashboards and optimization decisions.
    """

    __tablename__ = "token_usage"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    agent_run_id: Mapped[str] = mapped_column(
        String(26), ForeignKey("agent_runs.id", ondelete="CASCADE")
    )
    iteration_index: Mapped[int] = mapped_column(Integer)
    model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    provider: Mapped[str | None] = mapped_column(String(100), nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float | None] = mapped_column(Numeric(10, 6), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cache_read_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cache_creation_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    agent_run: Mapped[AgentRun] = relationship(back_populates="token_usages")

    __table_args__ = (Index("ix_token_usage_agent_run_id", "agent_run_id"),)


class LLMInteraction(Base):
    """Per-LLM-call record with full semantic and provider payloads.

    The authoritative per-call record for the evidence layer. Stores both
    Dendrux's normalized view (semantic_request/response) and the exact
    vendor-specific wire format (provider_request/response) captured at
    the adapter boundary.

    token_usage is kept for backwards compatibility; this table supersedes it.
    """

    __tablename__ = "llm_interactions"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    agent_run_id: Mapped[str] = mapped_column(
        String(26), ForeignKey("agent_runs.id", ondelete="CASCADE")
    )
    iteration_index: Mapped[int] = mapped_column(Integer)
    model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    provider: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Dendrux-normalized payloads
    semantic_request: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    semantic_response: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Exact vendor API payloads (opaque JSON)
    provider_request: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    provider_response: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Token usage
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float | None] = mapped_column(Numeric(10, 6), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cache_read_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cache_creation_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Guardrail findings — best-effort enrichment (not authoritative audit)
    guardrail_findings: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    agent_run: Mapped[AgentRun] = relationship(back_populates="llm_interactions")

    __table_args__ = (Index("ix_llm_interactions_agent_run_id", "agent_run_id"),)


class RunEvent(Base):
    """Append-only state transition log for observability.

    Records durable timestamps for every lifecycle event in a run:
    started, LLM calls, tool calls, paused, resumed, completed, etc.

    This table exists specifically so the dashboard can reconstruct
    exact pause/resume timing without inferring from mutable columns
    or cleared pause_data. Every event has an immutable created_at.

    Ordering: sequence_index is the stable ordering key within a run.
    Timestamps alone are not safe (concurrent events within the same ms).
    The notifier increments sequence_index monotonically.

    Correlation: correlation_id links related events (e.g. tool.completed
    back to the tool_call_id, run.resumed to the original run.paused).
    This avoids ambiguity in multi-tool, multi-pause runs.

    Privacy: dashboard renders only observable data from this table.
    pause_data (unredacted execution state) is never exposed here.

    Event types:
        run.started, run.completed, run.error, run.cancelled,
        run.paused, run.resumed,
        llm.completed, tool.completed
    """

    __tablename__ = "run_events"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    agent_run_id: Mapped[str] = mapped_column(
        String(26), ForeignKey("agent_runs.id", ondelete="CASCADE")
    )
    event_type: Mapped[str] = mapped_column(String(50))
    sequence_index: Mapped[int] = mapped_column(Integer, default=0)
    iteration_index: Mapped[int] = mapped_column(Integer, default=0)
    correlation_id: Mapped[str | None] = mapped_column(String(26), nullable=True)
    data: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    agent_run: Mapped[AgentRun] = relationship(back_populates="run_events")

    __table_args__ = (
        Index("ix_run_events_agent_run_id_seq", "agent_run_id", "sequence_index"),
        Index("ix_run_events_event_type", "event_type"),
    )
