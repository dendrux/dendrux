"""Switching the kit on. Imports nothing optional: this runs in every pytest session."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

PLUGIN_NAME = "dendrux.testing.store_kit"
#: The real plugin (contract collection, fixtures, report), registered only when switched on.
LOADED_NAME = "dendrux.testing.store_kit._plugin"


def add_options(parser: pytest.Parser) -> None:
    group = parser.getgroup("dendrux-store-kit", "Dendrux store test kit")
    group.addoption(
        "--dendrux-store-kit",
        action="store_true",
        dest="dendrux_store_kit",
        default=False,
        help="Collect and run the Dendrux store contracts against your dendrux_* fixtures.",
    )
    parser.addini(
        "dendrux_store_kit",
        type="bool",
        default=False,
        help="Collect the Dendrux store contracts (same as --dendrux-store-kit).",
    )


def activated(config: pytest.Config) -> bool:
    return bool(
        config.getoption("dendrux_store_kit", default=False)
        or config.getini("dendrux_store_kit")
        or PLUGIN_NAME in (config.getoption("plugins", default=None) or [])
    )
