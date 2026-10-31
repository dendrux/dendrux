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
