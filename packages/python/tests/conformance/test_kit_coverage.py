"""The kit's coverage tables stay equal to the protocol they claim to cover."""

from __future__ import annotations

import inspect

from dendrux.runtime.state import StateStore
from dendrux.testing.store_kit._plugin import contract_inventory
from dendrux.testing.store_kit.contracts.test_fenced_writes import FENCED_WRITER_CONTRACTS


def test_every_fenced_writer_of_the_protocol_has_a_stale_epoch_contract() -> None:
    fenced = {
        name
        for name, function in inspect.getmembers(StateStore, inspect.isfunction)
        if "claim_epoch" in inspect.signature(function).parameters
    }
    assert fenced == set(FENCED_WRITER_CONTRACTS), "a protocol writer has no stale-epoch contract"
    shipped = {name for names in contract_inventory().values() for name in names}
    assert set(FENCED_WRITER_CONTRACTS.values()) <= shipped, "a named contract does not exist"


#: Protocol methods the contracts do not name, each with the reason. Keep this empty if possible:
#: a method with no direct contract is a method a store can get wrong unseen.
UNCOVERED_BY_DESIGN: dict[str, str] = {}


def test_every_protocol_method_is_named_by_a_contract_or_excluded_with_a_reason() -> None:
    import re
    from pathlib import Path

    import dendrux.testing.store_kit.contracts as contracts

    source = "\n".join(path.read_text() for path in Path(contracts.__path__[0]).glob("test_*.py"))
    methods = {
        name
        for name, _ in inspect.getmembers(StateStore, inspect.isfunction)
        if not name.startswith("_")
    }
    unnamed = {m for m in methods if not re.search(rf"\b{m}\b", source)}
    assert unnamed == set(UNCOVERED_BY_DESIGN), (
        f"protocol methods no contract calls: {sorted(unnamed - set(UNCOVERED_BY_DESIGN))}; "
        f"excluded but now covered: {sorted(set(UNCOVERED_BY_DESIGN) - unnamed)}"
    )
