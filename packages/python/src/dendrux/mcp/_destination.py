"""Public destination policy contract for MCP HTTP connections."""

from __future__ import annotations

from dataclasses import dataclass
from ipaddress import IPv4Address, IPv6Address  # noqa: TC003
from typing import Protocol


@dataclass(frozen=True, slots=True)
class MCPDestination:
    """One resolved TCP destination, checked before opening its socket.

    ``host`` is the requested hostname; ``ip`` is the numeric address that
    will actually be dialed. IPv4-mapped IPv6 addresses are normalized to IPv4.
    """

    host: str
    port: int
    ip: IPv4Address | IPv6Address


class MCPDestinationPolicy(Protocol):
    """Application-owned async policy; only an explicit True permits dialing.

    Called for each candidate on every new TCP connection, within the source's
    connect timeout. False, invalid return values, and exceptions fail closed.
    Existing pooled sockets do not perform DNS resolution again. Rebind the
    source when changing policy to retire existing connections.
    """

    async def __call__(self, destination: MCPDestination) -> bool:
        """Return True to allow this exact destination address."""
        ...
