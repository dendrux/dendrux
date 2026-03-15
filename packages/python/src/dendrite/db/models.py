"""SQLAlchemy models — the 4 Sprint 2 tables.

Tables:
    agent_runs    — One row per run() call. The anchor for everything.
    react_traces  — Canonical conversation history for one agent run.
    tool_calls    — Every tool invocation with params, result, timing.
    token_usage   — Per-LLM-call token counts and cost.

Each sub-agent spawn gets its own agent_run linked via parent_run_id.
No instance_id or delegation columns on react_traces — isolation is
structural (different agent_run_id = different traces).

Deferred tables (added via migration in their sprint):
    event_logs    — Sprint 4 (Workers & Crash Recovery)
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
    """Base class for all Dendrite models."""

    type_annotation_map = {
        dict[str, Any]: JSON,
    }


class AgentRun(Base):
    """Individual agent execution. The heartbeat of Dendrite.

    One row per run() call. Sub-agent spawns get their own row
    linked via parent_run_id. Developer links their world to ours
    via meta (opaque JSON blob — Dendrite stores it, never reads it).
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

    # Aggregate token usage (rolled up from token_usage table)
    total_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_cost_usd: Mapped[float | None] = mapped_column(Numeric(10, 6), nullable=True)

    # Developer's linking data — Dendrite stores it, never reads it
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Pause state — execution state for resume (D1: not observable, not redacted)
    # Cleared on finalize. Contains unredacted history for correct LLM resume.
    pause_data: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Error details (populated on ERROR status)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

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
    child_runs: Mapped[list[AgentRun]] = relationship(
        back_populates="parent_run",
    )
    parent_run: Mapped[AgentRun | None] = relationship(
        back_populates="child_runs", remote_side=[id]
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

    Records both Dendrite's stable correlation ID (tool_call_id) and
    the provider's native ID (provider_tool_call_id) for round-tripping.
    """

    __tablename__ = "tool_calls"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)
    agent_run_id: Mapped[str] = mapped_column(
        String(26), ForeignKey("agent_runs.id", ondelete="CASCADE")
    )
    tool_call_id: Mapped[str] = mapped_column(String(26))  # Dendrite ULID
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
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    agent_run: Mapped[AgentRun] = relationship(back_populates="token_usages")

    __table_args__ = (Index("ix_token_usage_agent_run_id", "agent_run_id"),)
