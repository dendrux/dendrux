"""Inert contracts for the process-local managed MCP runtime."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import StrEnum
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from dendrux.mcp._errors import MCPBindingConflictError
from dendrux.mcp._source import MCPPhysicalIdentity, MCPSource

if TYPE_CHECKING:
    from collections.abc import Collection

_NAMESPACE_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


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
        raise ValueError(f"MCP binding {field_name} key is required.")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"MCP binding {field_name} key must be a non-empty string.")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"MCP binding {field_name} key cannot contain control characters.")


def _validate_namespace(value: str) -> None:
    if not value or not _NAMESPACE_RE.fullmatch(value) or "__" in value:
        raise ValueError(
            f"MCP binding namespace '{value}' is invalid. It must match "
            "[a-zA-Z0-9_-]+ and cannot contain '__'."
        )


@dataclass(frozen=True, slots=True, repr=False)
class MCPBinding:
    """Immutable, non-live description of one runtime-managed Agent tool view."""

    runtime: MCPRuntime
    tenant_key: str | None
    connection_key: str
    source: MCPSource
    namespace: str
    policy: MCPToolPolicy
    credentials: MCPCredentialProvider | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def identity(self) -> tuple[str | None, str]:
        """Return the process-local registry identity for this connection."""
        return self.tenant_key, self.connection_key

    def with_policy(
        self,
        *,
        namespace: str | None = None,
        allowed_tools: Collection[str] | None = None,
        force_serial_tools: Collection[str] = (),
    ) -> MCPBinding:
        """Return a new Agent view over the same physical connection identity."""
        resolved_namespace = self.namespace if namespace is None else namespace
        _validate_namespace(resolved_namespace)
        return MCPBinding(
            runtime=self.runtime,
            tenant_key=self.tenant_key,
            connection_key=self.connection_key,
            source=self.source,
            namespace=resolved_namespace,
            policy=MCPToolPolicy(
                allowed_tools=allowed_tools,
                force_serial_tools=force_serial_tools,
            ),
            credentials=self.credentials,
        )

    def __repr__(self) -> str:
        return (
            "MCPBinding("
            f"tenant_key={self.tenant_key!r}, "
            f"connection_key={self.connection_key!r}, "
            f"source={self.source!r}, "
            f"namespace={self.namespace!r}, "
            f"policy={self.policy!r}"
            ")"
        )


class MCPRuntime:
    """Process-local owner for future managed MCP connections and leases.

    This initial contract validates configuration and creates inert bindings.
    It intentionally owns no SDK clients, tasks, timers, or subprocesses yet.
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
        # evict() must remove entries here, or an evicted connection key could
        # never be rebound to a changed endpoint.
        self._registrations: dict[tuple[str | None, str], MCPPhysicalIdentity] = {}
        self._lock = Lock()

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
    ) -> MCPBinding:
        """Create an inert binding without opening an MCP connection."""
        with self._lock:
            if self._state is not MCPRuntimeState.OPEN:
                raise RuntimeError("MCPRuntime is closed and cannot create new bindings.")
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
            physical_identity = source.physical_identity
            registered_identity = self._registrations.get(identity)
            if registered_identity is None:
                self._registrations[identity] = physical_identity
            elif registered_identity != physical_identity:
                mismatch: Literal["transport", "endpoint"] = (
                    "transport" if registered_identity[0] != physical_identity[0] else "endpoint"
                )
                raise MCPBindingConflictError(identity, mismatch=mismatch)

            return MCPBinding(
                runtime=self,
                tenant_key=tenant_key,
                connection_key=connection_key,
                source=source,
                namespace=source.name,
                policy=MCPToolPolicy(),
                credentials=credentials,
            )

    async def close(self) -> None:
        """Close the inert runtime contract idempotently."""
        with self._lock:
            if self._state is MCPRuntimeState.CLOSED:
                return
            self._state = MCPRuntimeState.DRAINING
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
