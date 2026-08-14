"""Immutable operational telemetry types for the managed MCP runtime.

The runtime-level equivalent of connection-pool metrics: point-in-time
snapshots (:meth:`MCPRuntime.snapshot`) and typed lifecycle events
(``MCPRuntime(observer=...)``). Everything here is process-local and
storage-free — no database, no exporter, no hosted monitoring. Developers
bridge these signals to Prometheus, OpenTelemetry, Datadog, logs, or
their own systems.

Every field is value-free by construction: identities, validated source
names, class names, counts, and durations. Credentials, URLs, tool
arguments, results, and error text never appear. Tenant and connection
keys are carried on the typed objects for the application to use, but the
runtime itself never writes them to its own logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable


class MCPRuntimeState(StrEnum):
    """Application-owned managed runtime lifecycle state.

    Defined here, in the dependency-free types module, so every public
    snapshot annotation resolves at runtime for reflection-based tooling.
    """

    OPEN = "open"
    DRAINING = "draining"
    CLOSED = "closed"


class MCPConnectionStatus(StrEnum):
    """Point-in-time state of one physical managed connection."""

    CONNECTING = "connecting"
    ACTIVE = "active"
    IDLE = "idle"
    BROKEN = "broken"
    CLOSING = "closing"


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPConnectionSnapshot:
    """One physical connection as it existed at snapshot time.

    ``instance_id`` is the runtime-wide monotonic id of this exact physical
    connection; a reconnect under the same identity gets a new one.
    """

    tenant_key: str | None
    connection_key: str
    source_name: str
    instance_id: int
    status: MCPConnectionStatus
    leases: int
    active_calls: int

    @property
    def identity(self) -> tuple[str | None, str]:
        """Return the registry identity of this connection."""
        return self.tenant_key, self.connection_key


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPOpenCircuitSnapshot:
    """One registered identity whose circuit breaker is currently open.

    ``retry_after`` is the remaining cooldown in seconds; ``0.0`` means the
    cooldown has elapsed and the next acquisition becomes the probe.
    """

    tenant_key: str | None
    connection_key: str
    source_name: str
    failure_count: int
    last_failure: str | None
    retry_after: float

    @property
    def identity(self) -> tuple[str | None, str]:
        """Return the registry identity of this circuit."""
        return self.tenant_key, self.connection_key


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPRuntimeSnapshot:
    """Immutable point-in-time view of runtime operational health.

    Taken atomically under one lock acquisition, so the numbers are
    mutually consistent: every admitted call belongs to a listed
    connection, and waiters count callers queued at snapshot time.

    ``in_flight_calls`` counts admitted calls occupying call capacity.
    A forcibly interrupted call whose transport ignores cancellation has
    its capacity reclaimed and moves to ``detached_calls`` until its
    operation actually exits, so reclaimed-but-running work stays visible.
    """

    state: MCPRuntimeState
    connections: tuple[MCPConnectionSnapshot, ...]
    in_flight_calls: int
    detached_calls: int
    connection_waiters: int
    call_waiters: int
    open_circuits: tuple[MCPOpenCircuitSnapshot, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPRuntimeEvent:
    """Base class for managed-runtime lifecycle events."""


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPConnectionEvent(MCPRuntimeEvent):
    """Base class for events scoped to one connection identity."""

    tenant_key: str | None
    connection_key: str
    source_name: str

    @property
    def identity(self) -> tuple[str | None, str]:
        """Return the registry identity this event belongs to."""
        return self.tenant_key, self.connection_key


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPPhysicalConnectionEvent(MCPConnectionEvent):
    """Base class for events tied to one exact physical connection.

    ``instance_id`` is a runtime-wide monotonic id assigned when the
    physical connection is created. It distinguishes overlapping lifetimes
    under one identity — a resistant old transport still closing while a
    replacement (possibly after a rebind) is already open and serving.
    """

    instance_id: int


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPConnectionOpening(MCPPhysicalConnectionEvent):
    """A physical connection attempt (connect + discovery) started.

    Ends in exactly one of :class:`MCPConnectionOpened`,
    :class:`MCPConnectionFailed`, or :class:`MCPConnectionAborted`.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPConnectionOpened(MCPPhysicalConnectionEvent):
    """A physical connection established and discovered its tools."""

    tool_count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPConnectionFailed(MCPPhysicalConnectionEvent):
    """A physical connection attempt failed before becoming usable.

    ``cause`` is the failure's class name only; transport error text is
    never carried.
    """

    cause: str


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPConnectionAborted(MCPPhysicalConnectionEvent):
    """A connection attempt was abandoned without a server verdict.

    Emitted when establishment is cancelled by shutdown, eviction, or
    retirement, or completes only after losing its slot. Not a server
    failure: it never counts toward the circuit breaker.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPConnectionClosed(MCPPhysicalConnectionEvent):
    """The runtime released one physical connection and attempted close.

    ``clean`` is False when close raised or did not finish within the bounded
    cleanup grace. Ownership is released either way, and cleanup detail goes
    only to value-free logs. Emitted exactly once per opened connection.
    """

    reason: Literal["idle", "broken", "evicted", "shutdown"]
    clean: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPEvictionStarted(MCPConnectionEvent):
    """An explicit eviction began for a live connection."""

    mode: Literal["drain", "force"]


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPEvictionCompleted(MCPConnectionEvent):
    """An explicit eviction finished; the identity is bindable again.

    ``forced`` is True when running work was interrupted — either
    ``mode="force"`` or a drain that timed out and escalated.
    """

    forced: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPCircuitOpened(MCPConnectionEvent):
    """Consecutive failures reached the threshold; acquisitions fail fast.

    Also emitted when a failed probe re-arms an already-open circuit.
    """

    failure_count: int
    last_failure: str | None
    reset_timeout: float


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPCircuitProbing(MCPConnectionEvent):
    """The cooldown elapsed and one half-open probe connection started."""


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPCircuitClosed(MCPConnectionEvent):
    """A previously open circuit closed after a successful connection."""


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPConnectionCapacityRejected(MCPConnectionEvent):
    """An acquisition was rejected: no connection slot within its budget."""

    limit: int


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPCallCapacityRejected(MCPConnectionEvent):
    """A tool call was shed: no call slot became free within its budget."""

    tool: str
    limit: int


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPToolCallStarted(MCPPhysicalConnectionEvent):
    """A tool call was admitted and handed to the transport.

    ``tool`` is the Agent-visible canonical name. Every started call ends
    in exactly one of :class:`MCPToolCallCompleted`,
    :class:`MCPToolCallFailed`, :class:`MCPToolCallOutcomeUnknown`, or
    :class:`MCPToolCallCancelled`.
    """

    tool: str


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPToolCallCompleted(MCPPhysicalConnectionEvent):
    """A tool call returned a definitive server response."""

    tool: str
    duration: float


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPToolCallFailed(MCPPhysicalConnectionEvent):
    """A tool call failed over a functioning connection.

    ``cause`` is the failure's class name only.
    """

    tool: str
    duration: float
    cause: str


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPToolCallOutcomeUnknown(MCPPhysicalConnectionEvent):
    """A tool call was interrupted with an unknown outcome.

    The server may already have applied the call; it is never retried
    automatically. A cancellation-resistant operation is terminalized here
    when the runtime reclaims its capacity; its caller may still observe the
    transport eventually exit, but no second telemetry terminal is emitted.
    """

    tool: str
    duration: float
    reason: Literal["forced", "connection_lost"]


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPToolCallCancelled(MCPPhysicalConnectionEvent):
    """The caller abandoned a tool call before a server response.

    Emitted when the application cancels the calling task — or
    process-level control flow unwinds it — while the call is in flight.
    The runtime did not interrupt it and no server verdict was observed.
    """

    tool: str
    duration: float


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPRuntimeShutdownStarted(MCPRuntimeEvent):
    """Runtime shutdown began draining leases and calls."""


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPRuntimeShutdownCompleted(MCPRuntimeEvent):
    """Runtime shutdown finished; the runtime is permanently closed.

    ``forced`` is True when the drain timed out and running work was
    interrupted. It is the final event emitted by this runtime, including
    when resistant transport cleanup continues without runtime ownership.
    """

    forced: bool


@runtime_checkable
class MCPRuntimeObserver(Protocol):
    """Application callback receiving managed-runtime lifecycle events.

    ``on_event`` must be a plain synchronous callable — coroutine
    functions and async callable objects are rejected at construction;
    awaitables returned through a synchronous wrapper are disposed and the
    event is dropped. It is invoked on the
    runtime's event loop, always outside runtime locks, so it may safely
    call :meth:`MCPRuntime.snapshot`. It must be fast and non-blocking —
    hand events to a metrics client or queue of the application's choosing
    rather than doing I/O inline. Exceptions it raises are swallowed and
    logged without detail: telemetry never breaks MCP work, but a failing
    observer silently drops its own events. Every ``BaseException`` is
    isolated, including cancellation and process-level control flow.
    """

    def on_event(self, event: MCPRuntimeEvent) -> None:
        """Handle one immutable lifecycle event."""
        ...
