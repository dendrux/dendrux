"""What a store declares, and which author fixtures that declaration obliges."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dendrux.runtime.state import StoreCapabilities

#: Capability flags in the order the report prints them.
CAPABILITIES = (
    "fenced_writes",
    "retry_ids",
    "checkpointing",
    "atomic_completion",
    "recovery",
)
#: Flags that together make a store create format-2 runs (``state.py`` creation rule).
FORMAT_2_FLAGS = ("fenced_writes", "retry_ids", "checkpointing", "atomic_completion")
DEPLOYMENTS = ("single_process", "single_host", "multi_server")

FACTORY = "dendrux_store_factory"
REOPEN = "dendrux_store_reopen"
SECOND_INSTANCE = "dendrux_store_second_instance"
CLOCK = "dendrux_clock"
FAULT_INJECTOR = "dendrux_fault_injector"
LIVE_FACTORY = "dendrux_live_store_factory"


def required_fixtures_for(declared: tuple[str, ...], deployment: str | None) -> dict[str, str]:
    """Author fixtures obliged by declared capabilities and a deployment, with the reason."""
    needed = {FACTORY: "every store"}
    if "recovery" in declared:
        needed[REOPEN] = "recovery is declared"
        needed[CLOCK] = "recovery (leases) is declared"
    if deployment in ("single_host", "multi_server"):
        needed[SECOND_INSTANCE] = f"deployment is {deployment}"
    return needed


@dataclass(frozen=True)
class Declaration:
    """A store's declared capabilities and deployment, read once."""

    store_name: str
    capabilities: StoreCapabilities
    module: str = ""
    qualname: str = ""

    @classmethod
    def of(cls, store: Any) -> Declaration:
        from dendrux.runtime._capabilities import capabilities

        kind = type(store)
        return cls(kind.__name__, capabilities(store), kind.__module__, kind.__qualname__)

    @property
    def qualified_name(self) -> str:
        """The class's import path: tells apart two implementations that share a class name."""
        return f"{self.module}.{self.store_name}" if self.module else self.store_name

    @property
    def identity(self) -> str:
        """Unique per implementation: two classes with one name in different places differ."""
        return f"{self.module}.{self.qualname or self.store_name}"

    @property
    def deployment(self) -> str | None:
        return self.capabilities.deployment

    @property
    def declared(self) -> tuple[str, ...]:
        return tuple(name for name in CAPABILITIES if getattr(self.capabilities, name))

    def has(self, capability: str) -> bool:
        return bool(getattr(self.capabilities, capability))

    @property
    def creates_format_2(self) -> bool:
        return all(self.has(flag) for flag in FORMAT_2_FLAGS)

    @property
    def needs_second_instance(self) -> bool:
        return self.deployment in ("single_host", "multi_server")

    def required_fixtures(self) -> dict[str, str]:
        """Author fixtures this declaration obliges, each with the reason."""
        return required_fixtures_for(self.declared, self.deployment)

    def label(self, *, qualified: bool = False) -> str:
        flags = ",".join(self.declared) or "none"
        name = self.qualified_name if qualified else self.store_name
        return f"{name} (deployment={self.deployment}, capabilities={flags})"


def validate_declaration(store: Any) -> None:
    """Raise ``StoreCapabilityError`` when a declared promise lacks its methods.

    The legacy ``deployment=None`` warning is silenced here: the kit refuses that case itself.
    """
    from dendrux.runtime._capabilities import validate_capabilities

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        validate_capabilities(store)
