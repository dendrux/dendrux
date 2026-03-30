"""Dendrite — The runtime for agents that act in the real world."""

__version__ = "0.1.0a1"

from dendrite.agent import Agent
from dendrite.bridge import bridge
from dendrite.runtime.runner import run
from dendrite.tool import tool

__all__ = ["Agent", "bridge", "run", "tool"]
