"""Task-local run context shared by the runner and loop admission checks."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dendrux.runtime.context import DelegationContext

current_delegation: ContextVar[DelegationContext | None] = ContextVar(
    "dendrux_delegation", default=None
)


def start_callback_delivered(run_id: str) -> bool:
    ctx = current_delegation.get()
    return ctx is not None and ctx.run_id == run_id and ctx.start_callback_delivered
