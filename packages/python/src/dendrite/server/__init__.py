"""Dendrite server — mountable FastAPI application for hosted agents.

Requires the [server] extra: pip install dendrite[server]
"""

from __future__ import annotations

try:
    import fastapi as _fastapi  # noqa: F401
except ImportError as e:
    raise ImportError(
        "Dendrite server requires FastAPI. Install with: pip install dendrite[server]"
    ) from e

from dendrite.registry import AgentRegistry, HostedAgentConfig
from dendrite.server.app import create_app

__all__ = ["create_app", "AgentRegistry", "HostedAgentConfig"]
