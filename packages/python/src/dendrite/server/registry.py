"""Backward-compatible re-export. Canonical location: dendrite.registry."""

from dendrite.registry import AgentRegistry, HostedAgentConfig

__all__ = ["AgentRegistry", "HostedAgentConfig"]
