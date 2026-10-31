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
    "notify",
)
#: Flags that together make a store create format-2 runs (``state.py`` creation rule).
FORMAT_2_FLAGS = ("fenced_writes", "retry_ids", "checkpointing", "atomic_completion")
DEPLOYMENTS = ("single_process", "single_host", "multi_server")

FACTORY = "dendrux_store_factory"
REOPEN = "dendrux_store_reopen"
SECOND_INSTANCE = "dendrux_store_second_instance"
CLOCK = "dendrux_clock"
FAULT_INJECTOR = "dendrux_fault_injector"


@dataclass(frozen=True)
class Declaration:
    """A store's declared capabilities and deployment, read once."""

    store_name: str
    capabilities: StoreCapabilities

    @classmethod
    def of(cls, store: Any) -> Declaration:
        from dendrux.runtime._capabilities import capabilities

        return cls(type(store).__name__, capabilities(store))

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
        needed = {FACTORY: "every store"}
        if self.has("recovery"):
            needed[REOPEN] = "recovery is declared"
            needed[CLOCK] = "recovery (leases) is declared"
        if self.needs_second_instance:
            needed[SECOND_INSTANCE] = f"deployment is {self.deployment}"
        return needed

    def label(self) -> str:
        flags = ",".join(self.declared) or "none"
        return f"{self.store_name} (deployment={self.deployment}, capabilities={flags})"


def validate_declaration(store: Any) -> None:
    """Raise ``StoreCapabilityError`` when a declared promise lacks its methods.

    The legacy ``deployment=None`` warning is silenced here: the kit refuses that case itself.
    """
    from dendrux.runtime._capabilities import validate_capabilities

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        validate_capabilities(store)
