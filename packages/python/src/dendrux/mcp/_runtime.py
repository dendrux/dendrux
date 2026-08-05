"""Process-local managed MCP runtime: shared connections, per-Agent leases."""

from __future__ import annotations

import asyncio
import logging
import math
import re
from dataclasses import dataclass, field, replace
from enum import StrEnum
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from dendrux.mcp._client import MCPClientAdapter
from dendrux.mcp._errors import MCPBindingConflictError, MCPConnectionError, MCPToolCallError
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

    async def _discover(self) -> list[ToolDef]:
        connection = self._view.connection
        self.last_error = None
        try:
            if not self._leased:
                self._entry = await connection.runtime._lease(connection)
                self._leased = True
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

        async def guarded_executor(**params: Any) -> Any:
            if not self._leased or self._entry is not entry:
                raise MCPToolCallError(
                    f"MCP tool view '{self.name}' is no longer active. "
                    "Create or discover tools from an active Agent before calling it."
                )
            connection.runtime._begin_call(connection.identity, entry)
            try:
                return await executor(**params)
            finally:
                connection.runtime._end_call(entry)

        return guarded_executor

    async def close(self) -> None:
        """Release this view's lease; the shared connection stays open."""
        if self._leased:
            entry = self._entry
            connection = self._view.connection
            if entry is not None:
                connection.runtime._release(connection.identity, entry)
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
        "info",
        "leases",
        "raw_tools",
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
        self.task: asyncio.Task[None] | None = None


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
    ``close()`` drains active leases before closing transports. Once
    ``shutdown_timeout`` elapses, resistant transport cleanup is retained
    and observed in the background so shutdown remains bounded.
    """

    def __init__(
        self,
        *,
        max_connections: int = 100,
        max_in_flight_calls: int = 100,
        idle_timeout: float = 300.0,
        shutdown_timeout: float = 30.0,
    ) -> None:
        self.max_connections = _positive_int(max_connections, "max_connections")
        self.max_in_flight_calls = _positive_int(
            max_in_flight_calls,
            "max_in_flight_calls",
        )
        self.idle_timeout = _non_negative_float(idle_timeout, "idle_timeout")
        self.shutdown_timeout = _positive_float(shutdown_timeout, "shutdown_timeout")
        self._state = MCPRuntimeState.OPEN
        # The registered source is the canonical configuration for an
        # identity. evict() must remove entries here, or an evicted
        # connection key could never be rebound to a changed endpoint.
        self._registrations: dict[tuple[str | None, str], MCPSource] = {}
        self._entries: dict[tuple[str | None, str], _ConnectionEntry] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._drain_event: asyncio.Event | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._abandoned_tasks: set[asyncio.Task[None]] = set()
        self._lock = Lock()

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
    def _consume_task_exception(task: asyncio.Task[None]) -> BaseException | None:
        try:
            return task.exception()
        except asyncio.CancelledError:
            return None

    def _abandoned_task_done(self, task: asyncio.Task[None]) -> None:
        exception = self._consume_task_exception(task)
        with self._lock:
            self._abandoned_tasks.discard(task)
        if exception is not None and not isinstance(exception, MCPConnectionError):
            logger.warning(
                "MCPRuntime background cleanup failed",
                exc_info=(type(exception), exception, exception.__traceback__),
            )

    def _track_abandoned_task(self, task: asyncio.Task[None]) -> None:
        with self._lock:
            self._abandoned_tasks.add(task)
        task.add_done_callback(self._abandoned_task_done)

    async def _close_adapter(self, entry: _ConnectionEntry) -> None:
        adapter = entry.adapter
        if adapter is None:
            return
        try:
            await adapter.close()
        except Exception:
            logger.warning(
                "MCP source '%s' connection cleanup failed",
                entry.source.name,
                exc_info=True,
            )

    async def _lease(self, connection: MCPConnection) -> _ConnectionEntry:
        """Lease the shared physical connection, opening it single-flight."""
        identity = connection.identity
        with self._lock:
            if self._state is not MCPRuntimeState.OPEN:
                raise RuntimeError("MCPRuntime is closed and cannot open MCP connections.")
            self._bind_loop()
            registered_source = self._registrations.get(identity)
            if registered_source is None or not _same_connection_config(
                registered_source, connection.source
            ):
                raise RuntimeError(
                    f"MCP connection handle {identity!r} was superseded by a rebind "
                    "with a different configuration. Bind again and use the new handle."
                )
            entry = self._entries.get(identity)
            if entry is None:
                entry = _ConnectionEntry(registered_source)
                entry.task = asyncio.get_running_loop().create_task(
                    self._open_connection(identity, entry)
                )
                self._entries[identity] = entry
            entry.leases += 1
            task = entry.task

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
                if self._entries.get(identity) is entry:
                    # Discard so the identity can be retried with a fresh entry.
                    del self._entries[identity]
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
            still_current = self._entries.get(identity) is entry
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
                f"MCP source '{entry.source.name}' connection was evicted before "
                "establishment completed."
            )
        if not raw_tools:
            logger.warning(
                "MCP source '%s' discovered zero tools. This may indicate a configuration problem.",
                entry.source.name,
            )

    def _release(self, identity: tuple[str | None, str], entry: _ConnectionEntry) -> None:
        """Release one lease held on an exact entry.

        The entry token guards against a stale waiter unwinding after its
        failed entry was discarded and decrementing an unrelated replacement
        entry that is now live under the same identity.
        """
        with self._lock:
            self._bind_loop()
            if self._entries.get(identity) is entry and entry.leases > 0:
                entry.leases -= 1
            event = self._drain_event
        if event is not None:
            event.set()

    def _begin_call(
        self,
        identity: tuple[str | None, str],
        entry: _ConnectionEntry,
    ) -> None:
        """Account for one tool call against an exact live connection entry."""
        with self._lock:
            self._bind_loop()
            if (
                self._state is not MCPRuntimeState.OPEN
                or self._entries.get(identity) is not entry
                or entry.adapter is None
            ):
                raise MCPToolCallError(
                    "MCP tool view is no longer active on its managed connection."
                )
            entry.active_calls += 1

    def _end_call(self, entry: _ConnectionEntry) -> None:
        """Release one in-flight call token and wake a draining runtime."""
        with self._lock:
            self._bind_loop()
            if entry.active_calls > 0:
                entry.active_calls -= 1
            event = self._drain_event
        if event is not None:
            event.set()

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
                raise RuntimeError("MCPRuntime is closed and cannot bind new connections.")
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
            registered_source = self._registrations.get(identity)
            if registered_source is None:
                self._registrations[identity] = source
            else:
                registered_physical = registered_source.physical_identity
                physical_identity = source.physical_identity
                if registered_physical != physical_identity:
                    mismatch: Literal["transport", "endpoint"] = (
                        "transport"
                        if registered_physical[0] != physical_identity[0]
                        else "endpoint"
                    )
                    raise MCPBindingConflictError(identity, mismatch=mismatch)
                if (
                    not _same_connection_config(registered_source, source)
                    and identity in self._entries
                ):
                    raise MCPBindingConflictError(identity, mismatch="configuration")
                # The newest bind becomes canonical: a non-live rebind may
                # rotate configuration outright, and a config-equivalent one
                # refreshes the auth object for the next open.
                self._registrations[identity] = source

            return MCPConnection(
                runtime=self,
                tenant_key=tenant_key,
                connection_key=connection_key,
                source=source,
                credentials=credentials,
            )

    async def close(self) -> None:
        """Drain active leases, then start bounded transport cleanup.

        Idempotent and safe to call concurrently: every caller awaits the
        same shutdown task. Leases still held after ``shutdown_timeout`` are
        force-evicted. A transport that resists cancellation or close is
        retained and observed until its cleanup task eventually finishes.
        """
        with self._lock:
            if self._state is MCPRuntimeState.CLOSED and self._close_task is None:
                return
            self._bind_loop()
            if self._close_task is None:
                self._close_task = asyncio.get_running_loop().create_task(self._close_impl())
            task = self._close_task
        # Shielded so one cancelled caller cannot abort the shared shutdown.
        await asyncio.shield(task)

    async def _close_impl(self) -> None:
        with self._lock:
            self._state = MCPRuntimeState.DRAINING

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.shutdown_timeout
        while True:
            with self._lock:
                if not any(entry.leases or entry.active_calls for entry in self._entries.values()):
                    break
                event = asyncio.Event()
                self._drain_event = event
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                await asyncio.wait_for(event.wait(), timeout=remaining)
            except TimeoutError:
                break

        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
            self._registrations.clear()
            self._drain_event = None
        connect_tasks = {
            entry.task: entry
            for entry in entries
            if entry.task is not None and not entry.task.done()
        }
        for task in connect_tasks:
            task.cancel()
        if connect_tasks:
            # All connection tasks share one grace window. Applying it once
            # per entry would make forced shutdown scale linearly with the
            # number of cancellation-resistant transports.
            grace = max(deadline - loop.time(), _FORCED_CANCEL_GRACE)
            done, pending = await asyncio.wait(connect_tasks, timeout=grace)
            for task in done:
                self._consume_task_exception(task)
            for task in pending:
                entry = connect_tasks[task]
                logger.warning(
                    "MCP source '%s' connect task ignored cancellation within the "
                    "shutdown budget; abandoning it.",
                    entry.source.name,
                )
                # The task still owns its partially opened adapter. Retain it
                # until completion and consume its result so a late eviction
                # error never becomes an unobserved task exception.
                self._track_abandoned_task(task)

        close_tasks = {
            loop.create_task(self._close_adapter(entry)): entry
            for entry in entries
            if entry.adapter is not None
        }
        if close_tasks:
            done, pending = await asyncio.wait(
                close_tasks,
                timeout=_FORCED_CANCEL_GRACE,
            )
            for task in done:
                self._consume_task_exception(task)
            for task in pending:
                entry = close_tasks[task]
                logger.warning(
                    "MCP source '%s' connection cleanup exceeded the shutdown grace; "
                    "allowing it to finish in the background.",
                    entry.source.name,
                )
                self._track_abandoned_task(task)
        with self._lock:
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
