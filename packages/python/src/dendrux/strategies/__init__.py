"""Agent strategies — how the agent communicates with the LLM."""

from dendrux.strategies.base import Strategy
from dendrux.strategies.native import NativeToolCalling

__all__ = ["NativeToolCalling", "Strategy"]
