"""Credential contracts shared by the managed runtime and MCP OAuth."""

from __future__ import annotations

# Public protocol annotations are resolved at runtime by documentation and
# dependency-injection tooling.
from collections.abc import Mapping  # noqa: TC003
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


def _validate_key_part(value: object, field_name: str, *, optional: bool) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"MCPAuthKey {field_name} must be a non-empty string.")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"MCPAuthKey {field_name} cannot contain control characters.")


@dataclass(frozen=True, slots=True)
class MCPAuthKey:
    """Identity of one user's authorization for one MCP connection.

    Mirrors the runtime's ``(tenant_key, connection_key)`` identity. Keys are
    application identifiers and may be personal data; Dendrux never writes
    them to its own logs or error messages.
    """

    connection_key: str
    tenant_key: str | None = None

    def __post_init__(self) -> None:
        _validate_key_part(self.connection_key, "connection_key", optional=False)
        _validate_key_part(self.tenant_key, "tenant_key", optional=True)


@runtime_checkable
class MCPCredentialProvider(Protocol):
    """Application callback that returns ephemeral MCP authentication.

    Invoked lazily, once per physical connection attempt — never per Agent,
    per lease, or per tool call — and bounded by its own ``connect_timeout``
    budget, separate from the transport handshake's identical budget (a
    connect behind a slow provider can take up to twice ``connect_timeout``).
    The provider must return a non-empty mapping: request headers for HTTP
    sources, environment variables for stdio sources. Only mappings are
    accepted because their exact values double as redaction needles. Managed
    connections reject opaque static auth objects for the same reason; use
    static headers or this provider boundary instead. The resolved value
    exists only to establish that one transport: it is never persisted,
    logged, or rendered into errors or representations.
    """

    async def get_auth(self) -> Mapping[str, str]:
        """Return authentication suitable for the configured MCP transport."""
        ...


@runtime_checkable
class MCPRefreshingCredentialProvider(MCPCredentialProvider, Protocol):
    """Credential provider supporting explicit rejection of a cached token.

    The runtime calls this once for concurrent rejected calls, then calls
    ``get_auth()`` while reconnecting. The application owns refresh-token
    exchange, durable storage, and synchronization across worker processes,
    unless the provider is Dendrux's own :class:`MCPOAuth` credential.
    """

    async def on_auth_rejected(self) -> None:
        """Invalidate or refresh rejected credentials before the next get_auth."""
        ...
