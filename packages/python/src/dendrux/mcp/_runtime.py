"""Process-local managed MCP runtime: shared connections, per-Agent leases."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import re
from collections import deque
from dataclasses import dataclass, field, replace
from enum import StrEnum
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from dendrux.mcp._client import MCPClientAdapter
from dendrux.mcp._errors import (
    MCPBindingConflictError,
    MCPCallCapacityError,
    MCPConnectionCapacityError,
    MCPConnectionError,
    MCPConnectionEvictingError,
    MCPOutcomeUnknownError,
    MCPRuntimeClosedError,
    MCPStaleConnectionError,
    MCPToolCallError,
)
from dendrux.mcp._server import MCPServer, build_mcp_tool_defs, create_mcp_executor
from dendrux.mcp._source import MCPSource

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

    from dendrux.types import ToolDef

logger = logging.getLogger(__name__)

_NAMESPACE_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

# Post-drain grace for connection and cleanup tasks: enough for cooperative
# cancellation/close to finish, but bounded when a transport misbehaves.
_FORCED_CANCEL_GRACE = 0.05


MCPEvictionMode = Literal["drain", "force"]


class MCPRuntimeState(StrEnum):
    """Application-owned managed runtime lifecycle state."""

    OPEN = "open"
    DRAINING = "draining"
    CLOSED = "closed"


@runtime_checkable
class MCPCredentialProvider(Protocol):
    """Application callback that returns ephemeral MCP authentication."""

    async def get_auth(self) -> Any:
        """Return authentication suitable for the configured MCP transport."""
        ...


def _copy_tool_names(
    value: Collection[str] | None,
    field_name: str,
) -> frozenset[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        raise ValueError(f"MCPToolPolicy {field_name} must be a collection of tool names.")

    copied = frozenset(value)
    valid = all(
        isinstance(name, str)
        and bool(name.strip())
        and not any(ord(char) < 32 or ord(char) == 127 for char in name)
        for name in copied
    )
    if not valid:
        raise ValueError(
            f"MCPToolPolicy {field_name} must contain non-empty tool names "
            "without control characters."
        )
    return copied


@dataclass(frozen=True, slots=True, init=False)
class MCPToolPolicy:
    """Immutable tool exposure and safety policy for one Agent view."""

    allowed_tools: frozenset[str] | None
    force_serial_tools: frozenset[str]

    def __init__(
        self,
        *,
        allowed_tools: Collection[str] | None = None,
        force_serial_tools: Collection[str] = (),
    ) -> None:
        object.__setattr__(
            self,
            "allowed_tools",
            _copy_tool_names(allowed_tools, "allowed_tools"),
        )
        object.__setattr__(
            self,
            "force_serial_tools",
            _copy_tool_names(force_serial_tools, "force_serial_tools") or frozenset(),
        )


def _validate_identity_key(value: str | None, field_name: str, *, optional: bool) -> None:
    if value is None:
        if optional:
            return
        raise ValueError(f"MCP connection {field_name} key is required.")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"MCP connection {field_name} key must be a non-empty string.")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"MCP connection {field_name} key cannot contain control characters.")


def _validate_namespace(value: str) -> None:
    if not value or not _NAMESPACE_RE.fullmatch(value) or "__" in value:
        raise ValueError(
            f"MCP tool view namespace '{value}' is invalid. It must match "
            "[a-zA-Z0-9_-]+ and cannot contain '__'."
        )


@dataclass(frozen=True, slots=True, repr=False)
class MCPConnection:
    """Lazy logical handle for one runtime-managed connection identity.

    Holds the registry identity, source configuration, and credential
    provider. It is not an Agent tool source and does not imply that a
    physical connection is currently open; call :meth:`tools` to create
    the view an Agent consumes.
    """

    runtime: MCPRuntime
    tenant_key: str | None
    connection_key: str
    source: MCPSource
    credentials: MCPCredentialProvider | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    generation: int = 0

    @property
    def identity(self) -> tuple[str | None, str]:
        """Return the process-local registry identity for this connection."""
        return self.tenant_key, self.connection_key

    def tools(
        self,
        *,
        namespace: str | None = None,
        allowed_tools: Collection[str] | None = None,
        force_serial_tools: Collection[str] = (),
    ) -> MCPToolView:
        """Return an Agent-consumable tool view over this connection.

        Creating a view performs no connection or discovery work. Calling
        with no arguments exposes every server tool under the source-name
        namespace; that is an explicit governance decision.
        """
        resolved_namespace = self.source.name if namespace is None else namespace
        _validate_namespace(resolved_namespace)
        return MCPToolView(
            connection=self,
            namespace=resolved_namespace,
            policy=MCPToolPolicy(
                allowed_tools=allowed_tools,
                force_serial_tools=force_serial_tools,
            ),
        )

    def __repr__(self) -> str:
        return (
            "MCPConnection("
            f"tenant_key={self.tenant_key!r}, "
            f"connection_key={self.connection_key!r}, "
            f"source={self.source!r}"
            ")"
        )


@dataclass(frozen=True, slots=True, repr=False)
class MCPToolView:
    """Immutable namespace and policy for one Agent over one connection.

    This is the managed-runtime object intended for Agent tool sources.
    The live runtime integration acquires a per-Agent lease when consuming
    it; the view never owns or directly closes the shared connection.
    """

    connection: MCPConnection
    namespace: str
    policy: MCPToolPolicy

    @property
    def identity(self) -> tuple[str | None, str]:
        """Return the registry identity of the underlying connection."""
        return self.connection.identity

    def __repr__(self) -> str:
        return (
            "MCPToolView("
            f"connection={self.connection!r}, "
            f"namespace={self.namespace!r}, "
            f"policy={self.policy!r}"
            ")"
        )


class _ViewToolSource(MCPServer):
    """Per-Agent lease adapter exposing one MCPToolView as a tool source.

    Acquires its lease lazily on first discovery and releases it exactly
    once on ``close()``. The shared physical connection is owned by the
    runtime; closing this source never closes the transport.
    """

    def __init__(self, view: MCPToolView) -> None:
        self._view = view
        self.source = view.connection.source
        self.name = view.namespace
        self.failure_mode = self.source.failure_mode
        self._url = self.source.url
        self._command = list(self.source.command) if self.source.command is not None else None
        self.last_error: str | None = None
        self._entry: _ConnectionEntry | None = None
        self._leased = False
        self._close_generation = 0
        # Queue ownership is per Agent view, not per physical connection. It
        # lets close() invalidate only this Agent's waiting calls when several
        # Agents share one entry.
        self._call_owner = object()

    async def _discover(self) -> list[ToolDef]:
        connection = self._view.connection
        close_generation = self._close_generation
        self.last_error = None
        try:
            if not self._leased:
                self._entry = await connection.runtime._lease(connection)
                self._leased = True
                if self._close_generation != close_generation:
                    # The Agent was closed from another task while we were
                    # connecting. Its close() could not release a lease we did
                    # not hold yet, so hand it back here or this connection
                    # stays pinned for the life of the runtime.
                    raise RuntimeError(f"MCP tool view '{self.name}' was closed during discovery.")
            entry = self._entry
            assert entry is not None
            policy = self._view.policy
            available = {tool.name for tool in entry.raw_tools}
            for field_name, requested in (
                ("allowed_tools", policy.allowed_tools),
                ("force_serial_tools", policy.force_serial_tools),
            ):
                unknown = sorted(set(requested or ()) - available)
                if unknown:
                    raise ValueError(
                        f"MCP tool view '{self.name}' {field_name} references tools "
                        f"unknown to source '{self.source.name}': {', '.join(unknown)}."
                    )
            exposed = entry.raw_tools
            if policy.allowed_tools is not None:
                exposed = [tool for tool in exposed if tool.name in policy.allowed_tools]
            return build_mcp_tool_defs(
                source=self.source,
                namespace=self.name,
                tools=exposed,
                connection_info=entry.info,
                force_serial_tools=policy.force_serial_tools,
            )
        except BaseException as exc:
            self.last_error = str(exc)
            # Agent cleanup only closes sources that discovered successfully,
            # so a failed view must release its own lease.
            await self.close()
            raise

    def _create_executor(self, mcp_tool_name: str) -> Callable[..., Any]:
        entry = self._entry
        if not self._leased or entry is None or entry.adapter is None:
            raise RuntimeError("Cannot create an MCP executor before discovery.")
        executor = create_mcp_executor(
            entry.adapter,
            namespace=self.name,
            mcp_tool_name=mcp_tool_name,
            max_result_bytes=self.source.max_result_bytes,
        )
        connection = self._view.connection
        qualified_name = f"{self.name}__{mcp_tool_name}"

        def unknown_outcome() -> MCPOutcomeUnknownError:
            return MCPOutcomeUnknownError(
                f"MCP tool '{qualified_name}' was interrupted by a forced eviction. "
                "The server may already have applied it, so it must not be retried "
                "automatically."
            )

        def check_lease() -> None:
            """Fail unless this view still holds this exact lease."""
            if not self._leased or self._entry is not entry:
                raise MCPToolCallError(
                    f"MCP tool view '{self.name}' is no longer active. "
                    "Create or discover tools from an active Agent before calling it."
                )

        async def guarded_executor(**params: Any) -> Any:
            permit = await connection.runtime._begin_call(
                connection.identity,
                entry,
                tool=qualified_name,
                owner=self._call_owner,
                # Re-checked on every wake, so one message covers both a stale
                # executor and an Agent that closed while its call was queued.
                check_lease=check_lease,
            )
            try:
                result = await executor(**params)
            except asyncio.CancelledError as exc:
                if not permit.interrupted:
                    raise
                # Absorb only the cancellation issued by MCPRuntime. If the
                # application also cancelled this Agent task, its independent
                # request remains and cancellation must keep propagating.
                remaining = permit.task.uncancel()
                permit.interrupted = False
                if remaining:
                    raise
                raise unknown_outcome() from exc
            except BaseException as exc:
                if entry.force_evicted:
                    if permit.interrupted:
                        remaining = permit.task.uncancel()
                        permit.interrupted = False
                        # A non-zero remainder is an application cancellation.
                        # Raise it now instead of letting the unrelated tool
                        # failure hide it until some later suspension point.
                        if remaining:
                            raise asyncio.CancelledError from exc
                    raise unknown_outcome() from exc
                raise
            else:
                if permit.interrupted:
                    remaining = permit.task.uncancel()
                    permit.interrupted = False
                    if remaining:
                        raise asyncio.CancelledError
                    # A normal return is a definitive server response. Forced
                    # teardown prevents future work; it does not make this
                    # already-completed call's outcome uncertain.
                return result
            finally:
                connection.runtime._end_call(connection.identity, permit)

        return guarded_executor

    async def close(self) -> None:
        """Release this view's lease; the shared connection stays open."""
        self._close_generation += 1
        if self._leased:
            entry = self._entry
            connection = self._view.connection
            if entry is not None:
                connection.runtime._release(
                    connection.identity,
                    entry,
                    call_owner=self._call_owner,
                )
            # Keep ownership intact when release rejects (for example, from
            # the wrong event loop) so the lease can still be released from
            # its owning loop.
            self._leased = False
            self._entry = None


class _ConnectionEntry:
    """Mutable per-identity connection record owned by MCPRuntime.

    ``task`` is the shared connect-and-discovery task. Every lease awaits it
    shielded, so cancelling one caller never cancels connection establishment
    for the others, and the task itself owns adapter cleanup on failure.
    """

    __slots__ = (
        "active_calls",
        "adapter",
        "call_permits",
        "close_started",
        "fenced",
        "force_evicted",
        "idle_handle",
        "idle_seq",
        "info",
        "leases",
        "raw_tools",
        "retire_task",
        "source",
        "task",
    )

    def __init__(self, source: MCPSource) -> None:
        self.source = source
        self.adapter: MCPClientAdapter | None = None
        self.raw_tools: list[Any] = []
        self.info: Any = None
        self.leases = 0
        self.active_calls = 0
        self.call_permits: set[_CallPermit] = set()
        self.task: asyncio.Task[None] | None = None
        # fenced: no new leases or calls may be taken on this entry. Set by
        # eviction, idle retirement, and shutdown alike. force_evicted is
        # narrower: work was interrupted, so outcomes are unknown.
        self.fenced = False
        self.force_evicted = False
        self.close_started = False
        # Idle retirement: a pending timer, then the task closing this exact
        # physical entry. Both are scoped to this entry so a stale callback can
        # never touch a replacement connection.
        self.idle_handle: asyncio.TimerHandle | None = None
        self.retire_task: asyncio.Task[None] | None = None
        # Monotonic stamp of when this entry last became idle. Ordering, not
        # time: capacity pressure retires the smallest sequence first. Tracked
        # even when idle_timeout is None, which only disables the timer.
        self.idle_seq: int | None = None


class _CallPermit:
    """Exact-once ownership of one admitted runtime-wide call slot."""

    __slots__ = ("entry", "interrupted", "release_event", "released", "task")

    def __init__(self, entry: _ConnectionEntry, task: asyncio.Task[Any]) -> None:
        self.entry = entry
        self.task = task
        self.interrupted = False
        self.release_event = asyncio.Event()
        self.released = False


class _CallWaiter:
    """One FIFO position in the queue for a global tool-call slot.

    Unlike connection admissions, positions are never shared: two calls on one
    connection want two slots, so each queues for itself. The entry is kept so
    an eviction can reject exactly the calls queued against it.
    """

    __slots__ = ("entry", "event", "owner")

    def __init__(self, entry: _ConnectionEntry, owner: object) -> None:
        self.entry = entry
        self.owner = owner
        self.event = asyncio.Event()


class _Admission:
    """One FIFO position in the queue for opening a physical connection.

    Every caller waiting on the same missing identity shares one position, so
    a hundred Agents racing for one server never crowd out other identities.
    """

    __slots__ = ("event", "identity", "waiters")

    def __init__(self, identity: tuple[str | None, str]) -> None:
        self.identity = identity
        self.event = asyncio.Event()
        self.waiters = 0


@dataclass(slots=True)
class _Registration:
    """Canonical configuration for one connection identity.

    ``generation`` increments whenever an identity is bound after an
    eviction, which is what invalidates handles issued before it.
    """

    source: MCPSource
    generation: int


def _same_connection_config(registered: MCPSource, candidate: MCPSource) -> bool:
    """Compare connection configuration, ignoring presentation.

    The name is only the default view namespace. Auth is compared by identity:
    one mutable provider may refresh itself in place, while replacing the
    provider requires eviction before an authenticated session can be reused.
    """
    return (
        registered.auth is candidate.auth and replace(registered, name=candidate.name) == candidate
    )


class MCPRuntime:
    """Process-local owner of shared MCP connections and per-Agent leases.

    One physical connection exists per ``(tenant_key, connection_key)``.
    Connect and discovery are lazy and single-flight; Agents lease the
    shared connection through tool views and release on ``Agent.close()``.

    A connection leaves the runtime in one of three ways:

    * **Idle retirement** — after ``idle_timeout`` seconds with no leases and
      no in-flight calls, the transport closes but the registration and its
      generation survive, so existing handles reconnect transparently. Pass
      ``idle_timeout=None`` to disable timed retirement; capacity pressure may
      still retire an idle connection. ``0`` retires as soon as the last lease
      is released.
    * **Eviction** — :meth:`evict` closes the transport *and* forgets the
      registration, permanently invalidating every handle issued for it.
    * **Shutdown** — :meth:`close` drains, then tears everything down.

    All three are bounded: a transport that resists cancellation or closing
    is handed to background tracking rather than waited on, so no caller can
    be blocked by a misbehaving server.

    Tool calls are bounded separately by ``max_in_flight_calls``, one budget
    shared by every connection: any number of Agents may hold leases, but only
    that many MCP operations are admitted at once. Calls over the limit queue
    FIFO for up to ``call_wait_timeout`` and are then shed, having never been
    sent. Forced cleanup releases logical capacity after a bounded grace even
    if a hostile transport keeps its already-started operation alive.
    """

    def __init__(
        self,
        *,
        max_connections: int = 100,
        max_in_flight_calls: int = 100,
        idle_timeout: float | None = 300.0,
        connection_wait_timeout: float = 10.0,
        call_wait_timeout: float = 10.0,
        shutdown_timeout: float = 30.0,
    ) -> None:
        self.max_connections = _positive_int(max_connections, "max_connections")
        self.max_in_flight_calls = _positive_int(
            max_in_flight_calls,
            "max_in_flight_calls",
        )
        # None disables idle retirement entirely: connections stay warm until
        # they are evicted or the runtime shuts down. Useful for stdio servers
        # whose subprocess startup costs more than holding the connection.
        self.idle_timeout: float | None = (
            None if idle_timeout is None else _non_negative_float(idle_timeout, "idle_timeout")
        )
        # Bounds only the wait for a free connection slot, never the MCP
        # handshake itself. 0 fails fast instead of queueing.
        self.connection_wait_timeout = _non_negative_float(
            connection_wait_timeout,
            "connection_wait_timeout",
        )
        # Bounds only the wait for a free call slot, never the tool call
        # itself. 0 sheds load immediately instead of queueing.
        self.call_wait_timeout = _non_negative_float(call_wait_timeout, "call_wait_timeout")
        self.shutdown_timeout = _positive_float(shutdown_timeout, "shutdown_timeout")
        self._state = MCPRuntimeState.OPEN
        # The registered source is the canonical configuration for an
        # identity. evict() must remove entries here, or an evicted
        # connection key could never be rebound to a changed endpoint.
        self._registrations: dict[tuple[str | None, str], _Registration] = {}
        self._entries: dict[tuple[str | None, str], _ConnectionEntry] = {}
        self._evictions: dict[tuple[str | None, str], asyncio.Task[None]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._drain_waiters: set[asyncio.Event] = set()
        self._close_task: asyncio.Task[None] | None = None
        self._abandoned_tasks: set[asyncio.Task[Any]] = set()
        self._generation_seq = 0
        # Insertion into _entries *is* the slot reservation, so capacity is
        # always len(self._entries) with no counter that could drift.
        self._admission_queue: deque[_Admission] = deque()
        self._admissions: dict[tuple[str | None, str], _Admission] = {}
        self._idle_seq = 0
        # Exact permits, rather than a free-standing counter, bound aggregate
        # process load without drifting across cancellation and forced cleanup.
        self._active_call_permits: set[_CallPermit] = set()
        self._call_queue: deque[_CallWaiter] = deque()
        self._lock = Lock()

    @property
    def _in_flight_calls(self) -> int:
        return len(self._active_call_permits)

    def _bind_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError(
                "MCPRuntime is bound to the event loop that first used it. "
                "Create one runtime per event loop."
            )

    @staticmethod
    def _consume_task_exception(task: asyncio.Task[Any]) -> BaseException | None:
        try:
            return task.exception()
        except asyncio.CancelledError:
            return None

    def _abandoned_task_done(self, task: asyncio.Task[Any]) -> None:
        exception = self._consume_task_exception(task)
        with self._lock:
            self._abandoned_tasks.discard(task)
        if exception is not None and not isinstance(exception, MCPConnectionError):
            logger.warning(
                "MCPRuntime background cleanup failed",
                exc_info=(type(exception), exception, exception.__traceback__),
            )

    def _track_abandoned_task(self, task: asyncio.Task[Any]) -> None:
        with self._lock:
            self._abandoned_tasks.add(task)
        task.add_done_callback(self._abandoned_task_done)

    def _notify_drain(self) -> None:
        """Wake every drain waiter (runtime shutdown and evictions)."""
        with self._lock:
            waiters = list(self._drain_waiters)
        for waiter in waiters:
            waiter.set()

    async def _wait_for_quiet(
        self,
        deadline: float,
        is_quiet: Callable[[], bool],
    ) -> bool:
        """Wait until ``is_quiet()`` holds or the deadline passes.

        Returns True when the drain completed, False when it timed out.
        ``is_quiet`` is evaluated under the runtime lock.
        """
        loop = asyncio.get_running_loop()
        while True:
            waiter = asyncio.Event()
            with self._lock:
                if is_quiet():
                    return True
                self._drain_waiters.add(waiter)
            try:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return False
                await asyncio.wait_for(waiter.wait(), timeout=remaining)
            except TimeoutError:
                return False
            finally:
                with self._lock:
                    self._drain_waiters.discard(waiter)

    def _cancel_idle(self, entry: _ConnectionEntry) -> None:
        """Clear idle state on an entry. Caller must hold the runtime lock."""
        handle = entry.idle_handle
        if handle is not None:
            handle.cancel()
            entry.idle_handle = None
        entry.idle_seq = None

    def _discard_entry(
        self,
        identity: tuple[str | None, str],
        entry: _ConnectionEntry,
    ) -> None:
        """Relinquish ownership of one entry and free its slot.

        The single place a physical entry leaves the runtime, so capacity and
        the admission queue can never disagree with ``_entries``. Caller must
        hold the runtime lock.
        """
        if self._entries.get(identity) is entry:
            self._cancel_idle(entry)
            del self._entries[identity]
        self._pump_admissions()

    def _pump_admissions(self) -> None:
        """Admit the queue head, or start one retirement to make room.

        Caller must hold the runtime lock.
        """
        while self._admission_queue:
            head = self._admission_queue[0]
            if head.waiters == 0:
                # Every caller for this identity left; drop the position so it
                # cannot wedge the queue behind an absent waiter.
                self._admission_queue.popleft()
                if self._admissions.get(head.identity) is head:
                    del self._admissions[head.identity]
                continue
            if len(self._entries) < self.max_connections:
                head.event.set()
            else:
                # Only the head may provoke reclamation, and only one at a
                # time: the retirement frees a slot and pumps again.
                self._try_pressure_retire()
            return

    def _can_admit(self, admission: _Admission | None) -> bool:
        """Whether this caller may open a new connection right now.

        A free slot is not enough: queued identities must not be barged past,
        so only the queue head (or any caller when nothing is queued) wins.
        Caller must hold the runtime lock.
        """
        if len(self._entries) >= self.max_connections:
            return False
        if not self._admission_queue:
            return True
        return self._admission_queue[0] is admission

    def _enqueue_admission(
        self,
        identity: tuple[str | None, str],
        admission: _Admission | None,
    ) -> _Admission:
        """Join (or create) this identity's single FIFO position.

        Caller must hold the runtime lock.
        """
        if admission is not None:
            return admission
        existing = self._admissions.get(identity)
        if existing is None:
            existing = _Admission(identity)
            self._admissions[identity] = existing
            self._admission_queue.append(existing)
        existing.waiters += 1
        return existing

    def _release_admission(self, admission: _Admission | None) -> None:
        """Drop one waiter from a FIFO position, retiring it when empty.

        Caller must hold the runtime lock.
        """
        if admission is None:
            return
        if admission.waiters > 0:
            admission.waiters -= 1
        if admission.waiters:
            return
        if self._admissions.get(admission.identity) is admission:
            del self._admissions[admission.identity]
        with contextlib.suppress(ValueError):
            self._admission_queue.remove(admission)
        # A departing head must hand the queue to the next caller.
        self._pump_admissions()

    def _try_pressure_retire(self) -> None:
        """Retire the longest-idle connection to free a slot, if one exists.

        Capacity pressure retires regardless of ``idle_timeout``; that option
        only disables the *timer*. Caller must hold the runtime lock.
        """
        if self._state is not MCPRuntimeState.OPEN or self._loop is None:
            return
        if any(entry.retire_task is not None for entry in self._entries.values()):
            return  # a reclamation is already in flight
        candidate: tuple[tuple[str | None, str], _ConnectionEntry] | None = None
        for identity, entry in self._entries.items():
            if entry.idle_seq is None or not self._is_retirable(identity, entry):
                continue
            if candidate is None or entry.idle_seq < candidate[1].idle_seq:  # type: ignore[operator]
                candidate = (identity, entry)
        if candidate is None:
            return  # every connection is busy; a release will pump again
        identity, entry = candidate
        self._cancel_idle(entry)
        entry.retire_task = self._loop.create_task(self._retire(identity, entry))

    def _is_retirable(
        self,
        identity: tuple[str | None, str],
        entry: _ConnectionEntry,
    ) -> bool:
        """Whether this exact entry is still owned, unreferenced, and open.

        Arming, firing, and performing retirement all gate on this one
        predicate so the three stages can never disagree. Caller must hold
        the runtime lock.
        """
        return (
            self._state is MCPRuntimeState.OPEN
            and self._entries.get(identity) is entry
            and not entry.fenced
            and not entry.leases
            and not entry.active_calls
        )

    def _maybe_schedule_idle(
        self,
        identity: tuple[str | None, str],
        entry: _ConnectionEntry,
    ) -> None:
        """Arm idle retirement for a now-unreferenced connection.

        One event-loop timer per idle connection, no sweeper. Caller must hold
        the runtime lock.
        """
        loop = self._loop
        if loop is None or entry.retire_task is not None or not self._is_retirable(identity, entry):
            return
        if entry.idle_seq is None:
            self._idle_seq += 1
            entry.idle_seq = self._idle_seq
            # Newly reclaimable: a waiting caller can now retire this entry
            # even though len(_entries) has not changed yet.
            self._pump_admissions()
            if entry.retire_task is not None:
                return
        idle_timeout = self.idle_timeout
        if idle_timeout is not None and entry.idle_handle is None:
            entry.idle_handle = loop.call_later(
                idle_timeout,
                self._on_idle_expired,
                identity,
                entry,
            )

    def _on_idle_expired(
        self,
        identity: tuple[str | None, str],
        entry: _ConnectionEntry,
    ) -> None:
        """Timer callback: start retiring this exact entry if still idle."""
        with self._lock:
            entry.idle_handle = None
            loop = self._loop
            if (
                loop is None
                or entry.retire_task is not None
                or not self._is_retirable(identity, entry)
            ):
                # Busy again or already going away; the next release re-arms.
                return
            entry.retire_task = loop.create_task(self._retire(identity, entry))

    async def _retire(
        self,
        identity: tuple[str | None, str],
        entry: _ConnectionEntry,
    ) -> None:
        """Close an idle connection while keeping its identity bindable.

        Unlike eviction this never touches the registration or its generation:
        existing handles stay valid and transparently reconnect on next use.
        """
        loop = asyncio.get_running_loop()
        with self._lock:
            if not self._is_retirable(identity, entry):
                # Lost the entry to an eviction, a new lease, or shutdown; it
                # belongs to whoever fenced it, so leave it entirely alone.
                entry.retire_task = None
                return
            entry.fenced = True
        try:
            await self._teardown_entry(
                entry,
                loop=loop,
                cancel_warning=(
                    "MCP source '%s' connect task ignored idle cancellation; abandoning it."
                ),
                close_warning=(
                    "MCP source '%s' idle cleanup exceeded its grace; "
                    "allowing it to finish in the background."
                ),
            )
        finally:
            with self._lock:
                self._discard_entry(identity, entry)
                entry.retire_task = None
            self._notify_drain()

    async def _settle_teardown_tasks(
        self,
        tasks: dict[asyncio.Task[None], _ConnectionEntry],
        *,
        timeout: float,
        warning: str,
    ) -> None:
        """Await teardown tasks, then hand resistant ones to the background.

        Every finished task's result is consumed so a late failure never
        surfaces as an unobserved exception, and anything still running is
        tracked rather than waited on, keeping every caller bounded.
        """
        if not tasks:
            return
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        for task in done:
            self._consume_task_exception(task)
        for task in pending:
            logger.warning(warning, tasks[task].source.name)
            self._track_abandoned_task(task)

    async def _teardown_entry(
        self,
        entry: _ConnectionEntry,
        *,
        loop: asyncio.AbstractEventLoop,
        cancel_warning: str,
        close_warning: str,
    ) -> None:
        """Abort establishment and close one fenced entry's transport.

        Shared by idle retirement and explicit eviction; runtime shutdown
        batches the same two steps across every entry at once.
        """
        task = entry.task
        if task is not None and not task.done():
            task.cancel()
            await self._settle_teardown_tasks(
                {task: entry},
                timeout=_FORCED_CANCEL_GRACE,
                warning=cancel_warning,
            )
        await self._settle_teardown_tasks(
            {loop.create_task(self._close_adapter(entry)): entry},
            timeout=_FORCED_CANCEL_GRACE,
            warning=close_warning,
        )

    async def _close_adapter(self, entry: _ConnectionEntry) -> None:
        """Close an entry's transport at most once across all callers."""
        with self._lock:
            adapter = entry.adapter
            if adapter is None or entry.close_started:
                return
            # Guards against runtime shutdown and eviction both closing the
            # same transport when they run concurrently.
            entry.close_started = True
        try:
            await adapter.close()
        except Exception:
            logger.warning(
                "MCP source '%s' connection cleanup failed",
                entry.source.name,
                exc_info=True,
            )

    async def _lease(self, connection: MCPConnection) -> _ConnectionEntry:
        """Lease the shared physical connection, opening it single-flight.

        Opening a *new* physical connection consumes one of
        ``max_connections`` slots and queues FIFO when the pool is full;
        reusing a live one never waits.
        """
        identity = connection.identity
        admission: _Admission | None = None
        deadline: float | None = None
        loop = asyncio.get_running_loop()
        try:
            while True:
                with self._lock:
                    if self._state is not MCPRuntimeState.OPEN:
                        raise MCPRuntimeClosedError(
                            "MCPRuntime is closed and cannot open MCP connections."
                        )
                    self._bind_loop()
                    if identity in self._evictions:
                        raise MCPConnectionEvictingError(
                            identity,
                            f"MCP connection {identity!r} is being evicted and cannot accept "
                            "new leases. Bind again once eviction completes.",
                        )
                    registration = self._registrations.get(identity)
                    if registration is None:
                        raise MCPStaleConnectionError(
                            identity,
                            f"MCP connection handle {identity!r} is no longer registered; it was "
                            "evicted or never bound. Bind again and use the new handle.",
                        )
                    if (
                        registration.generation != connection.generation
                        or not _same_connection_config(registration.source, connection.source)
                    ):
                        raise MCPStaleConnectionError(
                            identity,
                            f"MCP connection handle {identity!r} was superseded by a rebind "
                            "with a different configuration. Bind again and use the new handle.",
                        )
                    entry = self._entries.get(identity)
                    retire_task = entry.retire_task if entry is not None else None
                    if retire_task is None:
                        if entry is not None:
                            # Reusing a live identity needs no slot, so it never
                            # queues behind callers opening other connections.
                            self._cancel_idle(entry)
                            entry.leases += 1
                            task = entry.task
                            break
                        if self._can_admit(admission):
                            entry = _ConnectionEntry(registration.source)
                            entry.task = loop.create_task(self._open_connection(identity, entry))
                            # Insertion into _entries is the slot reservation.
                            self._entries[identity] = entry
                            self._release_admission(admission)
                            admission = None
                            entry.leases += 1
                            task = entry.task
                            break
                        if deadline is None:
                            deadline = loop.time() + self.connection_wait_timeout
                        if loop.time() >= deadline:
                            raise MCPConnectionCapacityError(
                                identity,
                                limit=self.max_connections,
                                timeout=self.connection_wait_timeout,
                            )
                        admission = self._enqueue_admission(identity, admission)
                        admission.event.clear()
                        waiter = admission.event
                        remaining = deadline - loop.time()
                        self._pump_admissions()
                if retire_task is not None:
                    # This socket is being retired for idleness. Retirement is
                    # bounded, so wait it out and reconnect rather than fail.
                    await asyncio.shield(retire_task)
                    continue
                try:
                    await asyncio.wait_for(waiter.wait(), timeout=remaining)
                except TimeoutError:
                    continue  # re-check under the lock, then raise the capacity error
        finally:
            if admission is not None:
                with self._lock:
                    self._release_admission(admission)

        assert task is not None
        try:
            # Shielded: cancelling this caller must not cancel connection
            # establishment for concurrent leases of the same identity.
            await asyncio.shield(task)
        except BaseException:
            self._release(identity, entry)
            raise
        return entry

    async def _open_connection(
        self,
        identity: tuple[str | None, str],
        entry: _ConnectionEntry,
    ) -> None:
        """Connect and discover once for an entry; owns cleanup on failure."""
        adapter: MCPClientAdapter | None = None
        try:
            adapter = MCPClientAdapter(entry.source)
            await adapter.connect()
            raw_tools = await adapter.list_tools()
        except BaseException:
            with self._lock:
                if self._entries.get(identity) is entry and not entry.fenced:
                    # Discard so the identity can be retried with a fresh entry,
                    # freeing its slot and any idle timer a departed waiter armed.
                    self._discard_entry(identity, entry)
            if adapter is not None:
                try:
                    await adapter.close()
                except Exception:
                    logger.warning(
                        "MCP source '%s' cleanup failed after a connect error",
                        entry.source.name,
                        exc_info=True,
                    )
            raise
        with self._lock:
            # A force-closed or evicted entry must never receive a live
            # adapter: a transport that suppressed cancellation could
            # otherwise publish into an already-closed runtime.
            still_current = self._entries.get(identity) is entry and not entry.fenced
            if still_current:
                entry.adapter = adapter
                entry.raw_tools = list(raw_tools)
                entry.info = adapter.info
        if not still_current:
            try:
                await adapter.close()
            except Exception:
                logger.warning(
                    "MCP source '%s' cleanup failed after late establishment",
                    entry.source.name,
                    exc_info=True,
                )
            raise MCPConnectionError(
                f"MCP source '{entry.source.name}' connection was released before "
                "establishment completed."
            )
        if not raw_tools:
            logger.warning(
                "MCP source '%s' discovered zero tools. This may indicate a configuration problem.",
                entry.source.name,
            )

    def _release(
        self,
        identity: tuple[str | None, str],
        entry: _ConnectionEntry,
        *,
        call_owner: object | None = None,
    ) -> None:
        """Release one lease held on an exact entry.

        The entry token guards against a stale waiter unwinding after its
        failed entry was discarded and decrementing an unrelated replacement
        entry that is now live under the same identity.
        """
        with self._lock:
            self._bind_loop()
            if entry.leases > 0:
                entry.leases -= 1
            # A failed lease acquisition has no call owner. An Agent release
            # wakes only calls belonging to that view; other Agents can share
            # this physical entry and must remain asleep in FIFO order.
            if call_owner is not None:
                self._wake_call_waiters_for_owner(call_owner)
            self._maybe_schedule_idle(identity, entry)
        self._notify_drain()

    def _can_start_call(self, waiter: _CallWaiter | None) -> bool:
        """Whether this caller may occupy a call slot right now.

        A free slot is not enough: queued callers must not be barged past, so
        only the queue head (or any caller when nothing is queued) wins.
        Caller must hold the runtime lock.
        """
        if self._in_flight_calls >= self.max_in_flight_calls:
            return False
        if not self._call_queue:
            return True
        return self._call_queue[0] is waiter

    def _pump_calls(self) -> None:
        """Hand a freed call slot to the queue head.

        Caller must hold the runtime lock.
        """
        if self._call_queue and self._in_flight_calls < self.max_in_flight_calls:
            self._call_queue[0].event.set()

    def _wake_call_waiters_for_owner(self, owner: object) -> None:
        """Wake only queued calls owned by one closing Agent view."""
        for waiter in self._call_queue:
            if waiter.owner is owner:
                waiter.event.set()

    def _wake_call_waiters_for_entry(self, entry: _ConnectionEntry) -> None:
        """Wake every queued call on one fenced physical connection."""
        for waiter in self._call_queue:
            if waiter.entry is entry:
                waiter.event.set()

    def _wake_all_call_waiters(self) -> None:
        """Wake every queued call during runtime shutdown."""
        for waiter in self._call_queue:
            waiter.event.set()

    def _dequeue_call(self, waiter: _CallWaiter) -> None:
        """Drop one FIFO position and hand the queue on.

        Caller must hold the runtime lock.
        """
        with contextlib.suppress(ValueError):
            self._call_queue.remove(waiter)
        self._pump_calls()

    def _check_call_state(
        self,
        identity: tuple[str | None, str],
        entry: _ConnectionEntry,
        check_lease: Callable[[], None],
    ) -> None:
        """Reject a call whose lease, runtime, or connection is gone.

        Re-evaluated on every wake, so a queued call can never start against a
        connection that was evicted meanwhile or an Agent that has closed.
        Runtime shutdown wins over all narrower states; otherwise the view
        owns its lease-invalid message. Caller must hold the runtime lock.
        """
        if self._state is not MCPRuntimeState.OPEN:
            raise MCPRuntimeClosedError("MCPRuntime is closed and cannot start MCP tool calls.")
        check_lease()
        if identity in self._evictions:
            raise MCPConnectionEvictingError(
                identity,
                f"MCP connection {identity!r} is being evicted and cannot start tool calls.",
            )
        if self._entries.get(identity) is not entry or entry.adapter is None or entry.fenced:
            raise MCPStaleConnectionError(
                identity,
                f"MCP connection {identity!r} no longer owns the entry for this tool view. "
                "Bind again and use a new view.",
            )

    async def _begin_call(
        self,
        identity: tuple[str | None, str],
        entry: _ConnectionEntry,
        *,
        tool: str,
        owner: object,
        check_lease: Callable[[], None],
    ) -> _CallPermit:
        """Occupy one runtime-wide call slot for an exact live connection.

        Queues FIFO while every slot is busy. ``call_wait_timeout`` bounds only
        the wait for admission; once admitted a call runs for as long as it
        needs. A caller that leaves before admission — shed, cancelled, or
        invalidated — never counted against either budget.
        """
        loop = asyncio.get_running_loop()
        waiter: _CallWaiter | None = None
        deadline: float | None = None
        try:
            while True:
                with self._lock:
                    self._bind_loop()
                    self._check_call_state(identity, entry, check_lease)
                    if self._can_start_call(waiter):
                        task = asyncio.current_task()
                        assert task is not None
                        permit = _CallPermit(entry, task)
                        self._active_call_permits.add(permit)
                        entry.call_permits.add(permit)
                        entry.active_calls += 1
                        if waiter is not None:
                            self._dequeue_call(waiter)
                            waiter = None
                        return permit
                    if deadline is None:
                        deadline = loop.time() + self.call_wait_timeout
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise MCPCallCapacityError(
                            identity,
                            tool=tool,
                            limit=self.max_in_flight_calls,
                            timeout=self.call_wait_timeout,
                        )
                    if waiter is None:
                        waiter = _CallWaiter(entry, owner)
                        self._call_queue.append(waiter)
                    waiter.event.clear()
                    event = waiter.event
                try:
                    await asyncio.wait_for(event.wait(), timeout=remaining)
                except TimeoutError:
                    continue  # re-check under the lock, then shed
        finally:
            if waiter is not None:
                with self._lock:
                    self._dequeue_call(waiter)

    def _end_call(
        self,
        identity: tuple[str | None, str],
        permit: _CallPermit,
    ) -> None:
        """Release one exact call permit and wake a draining runtime."""
        with self._lock:
            self._bind_loop()
            released = self._release_call_permit(identity, permit)
        if released:
            self._notify_drain()

    def _release_call_permit(
        self,
        identity: tuple[str | None, str],
        permit: _CallPermit,
    ) -> bool:
        """Release a permit once, including after forced abandonment.

        Caller must hold the runtime lock. A late executor ``finally`` sees
        ``released`` and becomes a no-op, so it cannot decrement a newer call.
        """
        if permit.released:
            return False
        permit.released = True
        permit.release_event.set()
        entry = permit.entry
        self._active_call_permits.discard(permit)
        entry.call_permits.discard(permit)
        if entry.active_calls > 0:
            entry.active_calls -= 1
        self._pump_calls()
        self._maybe_schedule_idle(identity, entry)
        return True

    async def _interrupt_active_calls(
        self,
        entries: dict[tuple[str | None, str], _ConnectionEntry],
    ) -> None:
        """Cancel forced calls and reclaim resistant permits after one grace.

        Physical execution may outlive the grace when a transport suppresses
        cancellation. The connection is already fenced and its result is
        outcome-unknown, so the runtime releases logical capacity; the caller
        remains responsible for its task because the runtime did not create it.
        """
        with self._lock:
            owned = [
                (identity, permit)
                for identity, entry in entries.items()
                for permit in tuple(entry.call_permits)
            ]
        if not owned:
            return

        # No await may be introduced between this snapshot and cancel(): every
        # permit still belongs to a task suspended inside its executor, so the
        # runtime-issued cancellation is attributable without racing release.
        for _, permit in owned:
            if permit.interrupted:
                # A forced eviction and a forced shutdown can both reach the
                # same permit inside one grace. At most one runtime-issued
                # cancellation may be outstanding, or the executor's uncancel()
                # would read the second one back as an application request and
                # re-raise CancelledError — telling the caller that nothing
                # happened when the outcome is in fact unknown.
                continue
            permit.interrupted = True
            permit.task.cancel()

        release_waiters = [asyncio.create_task(permit.release_event.wait()) for _, permit in owned]
        _, pending_waiters = await asyncio.wait(
            release_waiters,
            timeout=_FORCED_CANCEL_GRACE,
        )
        for waiter in pending_waiters:
            waiter.cancel()
        if pending_waiters:
            await asyncio.gather(*pending_waiters, return_exceptions=True)

        released = False
        with self._lock:
            resistant = [permit for _, permit in owned if not permit.released]
            for identity, permit in owned:
                released = self._release_call_permit(identity, permit) or released
        if released:
            self._notify_drain()
        if resistant:
            # Tenant and connection keys are opaque application identifiers
            # and may contain PII. Only validated source labels are safe for
            # automatic library logs.
            sources = ", ".join(sorted({permit.entry.source.name for permit in resistant}))
            logger.warning(
                "%d MCP tool call(s) on source(s) %s ignored forced cancellation; "
                "releasing their runtime slots while their callers remain responsible "
                "for eventual task completion.",
                len(resistant),
                sources,
            )

    @property
    def state(self) -> MCPRuntimeState:
        """Return the current application-owned lifecycle state."""
        with self._lock:
            return self._state

    def bind(
        self,
        *,
        connection_key: str,
        source: MCPSource,
        tenant_key: str | None = None,
        credentials: MCPCredentialProvider | None = None,
    ) -> MCPConnection:
        """Register a connection identity and return its lazy handle."""
        with self._lock:
            if self._state is not MCPRuntimeState.OPEN:
                raise MCPRuntimeClosedError("MCPRuntime is closed and cannot bind new connections.")
            _validate_identity_key(tenant_key, "tenant", optional=True)
            _validate_identity_key(connection_key, "connection", optional=False)
            if not isinstance(source, MCPSource):
                raise ValueError("MCPRuntime source must be an MCPSource instance.")
            if credentials is not None and not isinstance(credentials, MCPCredentialProvider):
                raise ValueError(
                    "MCPRuntime credentials must implement MCPCredentialProvider.get_auth()."
                )
            _validate_namespace(source.name)

            identity = tenant_key, connection_key
            if identity in self._evictions:
                raise MCPConnectionEvictingError(
                    identity,
                    f"MCP connection {identity!r} is being evicted. "
                    "Bind again once eviction completes.",
                )
            registration = self._registrations.get(identity)
            if registration is None:
                # First bind, or the first bind after an eviction: a new
                # generation invalidates every handle issued before it.
                self._generation_seq += 1
                registration = _Registration(source=source, generation=self._generation_seq)
                self._registrations[identity] = registration
            else:
                registered_physical = registration.source.physical_identity
                physical_identity = source.physical_identity
                if registered_physical != physical_identity:
                    mismatch: Literal["transport", "endpoint"] = (
                        "transport"
                        if registered_physical[0] != physical_identity[0]
                        else "endpoint"
                    )
                    raise MCPBindingConflictError(identity, mismatch=mismatch)
                if (
                    not _same_connection_config(registration.source, source)
                    and identity in self._entries
                ):
                    raise MCPBindingConflictError(identity, mismatch="configuration")
                # The newest bind becomes canonical: a non-live rebind may
                # rotate configuration outright, and a config-equivalent one
                # refreshes the auth object for the next open. The generation
                # is unchanged, so existing handles stay valid.
                registration.source = source

            return MCPConnection(
                runtime=self,
                tenant_key=tenant_key,
                connection_key=connection_key,
                source=source,
                credentials=credentials,
                generation=registration.generation,
            )

    async def evict(
        self,
        *,
        connection_key: str,
        tenant_key: str | None = None,
        mode: MCPEvictionMode = "drain",
        timeout: float | None = None,
    ) -> None:
        """Close and forget one managed connection.

        ``mode="drain"`` stops new leases and calls, waits up to ``timeout``
        seconds for active ones to finish, then starts transport cleanup.
        ``mode="force"`` fences the connection immediately and starts bounded
        best-effort cleanup. If a transport resists closing, cleanup continues
        in the background and a replacement connection may open after this
        method returns; Dendrux will not route new work to the fenced transport.
        Interrupted tool calls raise :class:`MCPOutcomeUnknownError` because
        the server may already have applied their effect, and they must never
        be retried automatically.

        Only the exact ``(tenant_key, connection_key)`` partition is affected.
        Handles issued before the eviction are invalid afterwards, and the key
        may be rebound with a new endpoint or credentials. Concurrent and
        repeated calls share one operation; a ``force`` call escalates an
        in-flight drain.
        """
        if mode not in ("drain", "force"):
            raise ValueError("MCPRuntime evict mode must be 'drain' or 'force'.")
        resolved_timeout = (
            self.shutdown_timeout if timeout is None else _positive_float(timeout, "evict timeout")
        )
        _validate_identity_key(tenant_key, "tenant", optional=True)
        _validate_identity_key(connection_key, "connection", optional=False)

        identity = tenant_key, connection_key
        shutdown_task: asyncio.Task[None] | None = None
        with self._lock:
            if self._state is MCPRuntimeState.CLOSED:
                return
            self._bind_loop()
            if self._state is MCPRuntimeState.DRAINING:
                entry = self._entries.get(identity)
                escalate = mode == "force" and entry is not None and not entry.force_evicted
                if escalate:
                    assert entry is not None
                    entry.fenced = True
                    entry.force_evicted = True
                    self._cancel_idle(entry)
                shutdown_task = self._close_task
                task = None
            else:
                task = self._evictions.get(identity)
                if task is not None:
                    entry = self._entries.get(identity)
                    escalate = mode == "force" and entry is not None and not entry.force_evicted
                    if escalate:
                        assert entry is not None
                        entry.force_evicted = True
                else:
                    escalate = False
                    entry = self._entries.get(identity)
                    if entry is None:
                        # Nothing live: drop the registration so the key can be
                        # rebound with a different endpoint or credentials. A
                        # caller already queued for this identity must re-check
                        # immediately instead of sleeping until its timeout.
                        self._registrations.pop(identity, None)
                        admission = self._admissions.get(identity)
                        if admission is not None:
                            admission.event.set()
                        return
                    entry.fenced = True
                    if mode == "force":
                        entry.force_evicted = True
                    # Eviction outranks idle retirement: it also forgets the
                    # registration, so the handle becomes permanently stale.
                    self._cancel_idle(entry)
                    # Calls queued for a slot on this connection will never get
                    # to run; reject them now rather than after their budget.
                    self._wake_call_waiters_for_entry(entry)
                    registration = self._registrations.get(identity)
                    task = asyncio.get_running_loop().create_task(
                        self._evict_impl(
                            identity,
                            entry,
                            generation=registration.generation if registration else None,
                            timeout=resolved_timeout,
                        )
                    )
                    self._evictions[identity] = task
        if escalate:
            # Wake the in-flight drain so it stops waiting for live work.
            self._notify_drain()
        if shutdown_task is not None:
            await asyncio.shield(shutdown_task)
            return
        assert task is not None
        # Shielded so one cancelled caller cannot abort a shared eviction.
        await asyncio.shield(task)

    async def _evict_impl(
        self,
        identity: tuple[str | None, str],
        entry: _ConnectionEntry,
        *,
        generation: int | None,
        timeout: float,
    ) -> None:
        loop = asyncio.get_running_loop()
        try:
            deadline = loop.time() + timeout

            def quiet() -> bool:
                return entry.force_evicted or not (entry.leases or entry.active_calls)

            drained = await self._wait_for_quiet(deadline, quiet)
            with self._lock:
                if not drained or entry.leases or entry.active_calls:
                    # Escalation: the transport closes underneath work that is
                    # still running, so those calls have an unknown outcome.
                    if not entry.force_evicted:
                        logger.warning(
                            "MCP source '%s' eviction drain timed out; forcing close.",
                            entry.source.name,
                        )
                    entry.force_evicted = True

            if entry.force_evicted:
                await self._interrupt_active_calls({identity: entry})
            await self._teardown_entry(
                entry,
                loop=loop,
                cancel_warning=(
                    "MCP source '%s' connect task ignored eviction cancellation; abandoning it."
                ),
                close_warning=(
                    "MCP source '%s' connection cleanup exceeded the eviction grace; "
                    "allowing it to finish in the background."
                ),
            )
        finally:
            with self._lock:
                self._discard_entry(identity, entry)
                registration = self._registrations.get(identity)
                # Remove the registration only if this eviction still owns it,
                # so a newer rebind is never deleted by an older cleanup.
                if registration is not None and registration.generation == generation:
                    del self._registrations[identity]
                self._evictions.pop(identity, None)
            self._notify_drain()

    async def close(self) -> None:
        """Drain active leases, then start bounded transport cleanup.

        Idempotent and safe to call concurrently: every caller awaits the
        same shutdown task. Leases still held after ``shutdown_timeout`` are
        force-evicted. A transport that resists cancellation or close is
        retained and observed until its cleanup task eventually finishes.
        """
        with self._lock:
            self._bind_loop()
            if self._close_task is None:
                # Fence synchronously before yielding to _close_impl. Otherwise
                # a ready lease task can reserve and fully open a connection
                # after shutdown has already been requested.
                self._state = MCPRuntimeState.DRAINING
                for admission in self._admission_queue:
                    admission.event.set()
                self._wake_all_call_waiters()
                self._close_task = asyncio.get_running_loop().create_task(self._close_impl())
            task = self._close_task
        # Shielded so one cancelled caller cannot abort the shared shutdown.
        await asyncio.shield(task)

    async def _close_impl(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.shutdown_timeout

        def quiet() -> bool:
            return not any(
                (entry.leases or entry.active_calls) and not entry.force_evicted
                for entry in self._entries.values()
            )

        drained = await self._wait_for_quiet(deadline, quiet)

        with self._lock:
            entries = list(self._entries.values())
            for entry in entries:
                self._cancel_idle(entry)
                entry.fenced = True
                if not drained and (entry.leases or entry.active_calls):
                    # Forced shutdown closes the transport underneath a running
                    # call, so its outcome is unknown just as in a forced evict.
                    entry.force_evicted = True
            pending_tasks = [
                *self._evictions.values(),
                *(entry.retire_task for entry in entries if entry.retire_task is not None),
            ]
            forced_entries = {
                identity: entry for identity, entry in self._entries.items() if entry.force_evicted
            }
        self._notify_drain()
        await self._interrupt_active_calls(forced_entries)
        connect_tasks = {
            entry.task: entry
            for entry in entries
            if entry.task is not None and not entry.task.done()
        }
        for task in connect_tasks:
            task.cancel()
        # Unlike per-entry teardown, shutdown shares one grace window across
        # every connection. Applying it once per entry would make forced
        # shutdown scale linearly with the number of resistant transports.
        await self._settle_teardown_tasks(
            connect_tasks,
            timeout=max(deadline - loop.time(), _FORCED_CANCEL_GRACE),
            warning=(
                "MCP source '%s' connect task ignored cancellation within the "
                "shutdown budget; abandoning it."
            ),
        )
        await self._settle_teardown_tasks(
            {
                loop.create_task(self._close_adapter(entry)): entry
                for entry in entries
                if entry.adapter is not None
            },
            timeout=_FORCED_CANCEL_GRACE,
            warning=(
                "MCP source '%s' connection cleanup exceeded the shutdown grace; "
                "allowing it to finish in the background."
            ),
        )
        if pending_tasks:
            # Evictions and idle retirements own entries too; let them finish
            # before the runtime declares itself closed.
            await asyncio.gather(
                *(asyncio.shield(task) for task in pending_tasks),
                return_exceptions=True,
            )
            for task in pending_tasks:
                exception = self._consume_task_exception(task)
                if exception is not None:
                    logger.warning(
                        "MCPRuntime connection teardown failed during shutdown",
                        exc_info=(type(exception), exception, exception.__traceback__),
                    )
        with self._lock:
            self._entries.clear()
            self._registrations.clear()
            self._state = MCPRuntimeState.CLOSED

    async def __aenter__(self) -> MCPRuntime:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


def _positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"MCPRuntime {field_name} must be a positive integer.")
    return value


def _non_negative_float(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"MCPRuntime {field_name} must be a non-negative number.")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0:
        raise ValueError(f"MCPRuntime {field_name} must be a finite non-negative number.")
    return resolved


def _positive_float(value: float, field_name: str) -> float:
    resolved = _non_negative_float(value, field_name)
    if resolved == 0:
        raise ValueError(f"MCPRuntime {field_name} must be greater than zero.")
    return resolved
