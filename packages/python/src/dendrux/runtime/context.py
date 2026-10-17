"""Delegation context — automatic parent-child run linking.

Propagates run ancestry through nested agent.run() calls via
contextvars. When a tool function calls agent.run() for a sub-agent,
the child run automatically knows its parent's run_id and delegation
level — zero developer code needed.

Linking is best-effort:
  - Both persisted + same store → parent_run_id set on child
  - Parent ephemeral or different store → link skipped, warning emitted
  - Delegation always works regardless — linking is observability, not control

Store identity: compared by database URL (via store_identity property),
not by Python object identity. Two stores pointing at the same database
match even if they are different objects. Custom StateStore implementations
can define a store_identity property; otherwise falls back to object id.

Limitation: automatic linking works for in-process nested async calls.
Distributed/worker boundaries require explicit propagation (future work).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable

    from dendrux.skills._session import SkillSession

T = TypeVar("T")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DelegationDepthExceededError(RuntimeError):
    """Raised when a child run would exceed max_delegation_depth.

    Attributes:
        delegation_level: The level the child would have been at.
        max_depth: The configured maximum depth.
    """

    def __init__(self, delegation_level: int, max_depth: int) -> None:
        self.delegation_level = delegation_level
        self.max_depth = max_depth
        super().__init__(
            f"Delegation depth {delegation_level} exceeds "
            f"max_delegation_depth={max_depth}. "
            f"Increase the limit or restructure the agent tree."
        )


# ---------------------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DelegationContext:
    """Runtime context for a running agent — set by the runner, read by child runs.

    Attributes:
        run_id: This run's ID.
        delegation_level: Logical nesting depth (0 for root).
        persisted: Whether this run was written to a state store.
        store_identity: Comparable identity for the state store (database URL
            for SQLAlchemy stores, str(id()) for custom stores). None if not
            persisted.
        max_delegation_depth: Inherited ceiling on delegation depth.
            None means unbounded.
        warned_mismatches: Mismatch types already warned about for this
            parent run. Used to deduplicate warnings when an orchestrator
            fans out to many child runs. Mutable set on a frozen dataclass —
            frozen prevents field reassignment, not mutation of the set.
        tenant_id: This run's tenant. Delegated runs inherit it.
        state_store: The store this run persists to (None if ephemeral);
            used by :func:`heartbeat`.
        loop: The event loop the run executes on; lets :func:`heartbeat`
            schedule its write from a sync tool's worker thread.
    """

    run_id: str
    delegation_level: int = 0
    persisted: bool = False
    store_identity: str | None = None
    max_delegation_depth: int | None = None
    warned_mismatches: set[str] = field(default_factory=set)
    tenant_id: str | None = None
    state_store: Any = field(default=None, compare=False, repr=False)
    loop: asyncio.AbstractEventLoop | None = field(default=None, compare=False, repr=False)
    _heartbeat_at: list[float] = field(default_factory=lambda: [0.0], compare=False, repr=False)

    def __hash__(self) -> int:
        return hash(self.run_id)


# ---------------------------------------------------------------------------
# ContextVar — the propagation mechanism
# ---------------------------------------------------------------------------

_current_delegation: ContextVar[DelegationContext | None] = ContextVar(
    "dendrux_delegation", default=None
)


def get_delegation_context() -> DelegationContext | None:
    """Read the current delegation context (None if this is a root run)."""
    return _current_delegation.get()


def set_delegation_context(ctx: DelegationContext | None) -> Token[DelegationContext | None]:
    """Set the delegation context for the current async task.

    Returns a token for resetting in finally blocks.
    """
    return _current_delegation.set(ctx)


def reset_delegation_context(token: Token[DelegationContext | None]) -> None:
    """Reset the delegation context to its previous value."""
    _current_delegation.reset(token)


# ---------------------------------------------------------------------------
# Store identity — extract comparable identity from any StateStore
# ---------------------------------------------------------------------------


def get_store_identity(store: object | None) -> str | None:
    """Extract a comparable identity from a state store.

    Resolution order:
      1. store.store_identity property (SQLAlchemyStateStore returns DB URL)
      2. Fallback: str(id(store)) for custom stores without the property

    Returns None if store is None.
    """
    if store is None:
        return None
    identity = getattr(store, "store_identity", None)
    if identity is not None:
        return str(identity)
    return str(id(store))


# ---------------------------------------------------------------------------
# Tenant inheritance — delegated runs belong to their parent's tenant
# ---------------------------------------------------------------------------


def resolve_child_tenant(parent_ctx: DelegationContext | None, tenant_id: str | None) -> str | None:
    """Effective tenant for a new run.

    A root run keeps ``tenant_id``. A delegated run inherits the parent's
    tenant when ``tenant_id`` is None; an explicit different tenant raises
    ``ValueError`` before anything is created.
    """
    if parent_ctx is None:
        return tenant_id
    if tenant_id is None:
        return parent_ctx.tenant_id
    if tenant_id != parent_ctx.tenant_id:
        raise ValueError(
            f"Delegated run cannot use tenant_id={tenant_id!r}: its parent run "
            f"{parent_ctx.run_id} belongs to tenant {parent_ctx.tenant_id!r}. "
            "Child runs inherit the parent's tenant; omit tenant_id."
        )
    return tenant_id


# ---------------------------------------------------------------------------
# Heartbeat — keep a long-running tool's run from being swept as stale
# ---------------------------------------------------------------------------

_HEARTBEAT_MIN_INTERVAL_S = 1.0
_heartbeat_tasks: set[asyncio.Task[None]] = set()


async def _touch(store: Any, run_id: str) -> None:
    try:
        await store.touch_progress(run_id)
    except Exception:
        logger.warning("Heartbeat failed for run %s", run_id, exc_info=True)


def _heartbeat_target() -> DelegationContext | None:
    """The persisted run executing here, if any, rate-limited per run."""
    ctx = get_delegation_context()
    if ctx is None or not ctx.persisted or ctx.state_store is None:
        return None
    now = time.monotonic()
    if now - ctx._heartbeat_at[0] < _HEARTBEAT_MIN_INTERVAL_S:
        return None
    ctx._heartbeat_at[0] = now
    return ctx


def heartbeat() -> None:
    """Record forward progress for the run executing the current tool.

    Fire-and-forget and thread-safe: callable from async tools and from sync
    tools (which run in a worker thread). It schedules a ``last_progress_at``
    update on the run's event loop and returns immediately, so a long wait
    inside a tool is not swept as ``stale_running``. Calls closer than one
    second apart for the same run are coalesced. No-op outside a persisted
    run.
    """
    ctx = _heartbeat_target()
    if ctx is None or ctx.loop is None:
        return
    loop, store, run_id = ctx.loop, ctx.state_store, ctx.run_id

    def _schedule() -> None:
        task = loop.create_task(_touch(store, run_id))
        _heartbeat_tasks.add(task)
        task.add_done_callback(_heartbeat_tasks.discard)

    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        _schedule()
        return
    # A closed loop raises RuntimeError: the run is gone, nothing to record.
    with contextlib.suppress(RuntimeError):
        loop.call_soon_threadsafe(_schedule)


async def aheartbeat() -> None:
    """Awaitable :func:`heartbeat`: writes ``last_progress_at`` before returning.

    Must be awaited on the run's event loop (inside an async tool). Failures
    are logged, never raised. No-op outside a persisted run.
    """
    ctx = _heartbeat_target()
    if ctx is not None:
        await _touch(ctx.state_store, ctx.run_id)


# ---------------------------------------------------------------------------
# Linking logic — called by the runner when creating a child run
# ---------------------------------------------------------------------------


def resolve_parent_link(
    parent_ctx: DelegationContext | None,
    child_store: object | None,
) -> tuple[str | None, int]:
    """Determine parent_run_id and delegation_level for a new run.

    Returns:
        (parent_run_id, delegation_level) to pass to create_run().
        parent_run_id is None when linking is not safe.

    Raises:
        DelegationDepthExceededError: If the child's delegation level would
            exceed the inherited max_delegation_depth.
    """
    if parent_ctx is None:
        # Root run — no parent
        return None, 0

    delegation_level = parent_ctx.delegation_level + 1

    # Depth guard — check before doing anything else
    max_depth = parent_ctx.max_delegation_depth
    if max_depth is not None and delegation_level > max_depth:
        raise DelegationDepthExceededError(delegation_level, max_depth)

    if child_store is None:
        # Child is ephemeral — no DB row, no linking needed
        return None, delegation_level

    if not parent_ctx.persisted:
        # Case 2: parent ephemeral, child persisted
        if "ephemeral_parent" not in parent_ctx.warned_mismatches:
            logger.warning(
                "Delegation link skipped for child of run %s: parent run is not persisted.",
                parent_ctx.run_id,
            )
            parent_ctx.warned_mismatches.add("ephemeral_parent")
        return None, delegation_level

    child_identity = get_store_identity(child_store)
    if parent_ctx.store_identity != child_identity:
        # Case 4: different stores
        if "different_store" not in parent_ctx.warned_mismatches:
            logger.warning(
                "Delegation link skipped for child of run %s: "
                "parent and child do not share the same state store.",
                parent_ctx.run_id,
            )
            parent_ctx.warned_mismatches.add("different_store")
        return None, delegation_level

    # Case 1: same store, both persisted — link
    return parent_ctx.run_id, delegation_level


class StreamContext:
    """Bind run state only while a stream advances or closes, never across a yield.

    The inner generator may discover its session/delegation on the first step.
    Setters update that step's context and retain the values for subsequent steps;
    the wrapper owns all reset tokens, including when closed from another task.
    """

    def __init__(self, session: SkillSession | None = None):
        self.session = session
        self.delegation: DelegationContext | None = None
        self.started = False

    def set_session(self, session: SkillSession) -> None:
        from dendrux.skills._session import set_session

        self.session = session
        set_session(session)

    def set_delegation(self, delegation: DelegationContext) -> None:
        self.delegation = delegation
        set_delegation_context(delegation)

    async def _step(self, operation: Awaitable[T]) -> T:
        from dendrux.skills._session import reset_session, set_session

        if not self.started:
            self.delegation = get_delegation_context()
            self.started = True
        skill_token = set_session(self.session)
        delegation_token = set_delegation_context(self.delegation)
        try:
            return await operation
        finally:
            reset_delegation_context(delegation_token)
            reset_session(skill_token)

    async def wrap(self, generator: AsyncGenerator[T, None]) -> AsyncGenerator[T, None]:
        try:
            while True:
                try:
                    event = await self._step(anext(generator))
                except StopAsyncIteration:
                    return
                yield event
        finally:
            await self._step(generator.aclose())
