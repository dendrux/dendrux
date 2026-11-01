"""Registers the Dendrux store test kit with pytest without importing Dendrux.

``pip install dendrux`` registers this module as a pytest plugin (the ``pytest11`` entry point),
so it runs in every pytest session of that environment. It therefore imports nothing from
``dendrux`` and nothing optional: it adds the kit's options and, only when the kit is switched
on (``-p dendrux.testing.store_kit``, ``--dendrux-store-kit`` or ``dendrux_store_kit = true``),
loads the real plugin, which needs ``pytest-asyncio`` (``pip install 'dendrux[testkit]'``).
"""

from __future__ import annotations

import importlib
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


def pytest_addoption(parser: pytest.Parser) -> None:
    add_options(parser)


def pytest_configure(config: pytest.Config) -> None:
    """Load the real plugin when the kit is switched on; do nothing otherwise."""
    if not activated(config):
        return
    import pytest

    try:
        importlib.import_module("pytest_asyncio")
    except ImportError:
        raise pytest.UsageError(
            "The Dendrux store kit needs pytest-asyncio: pip install 'dendrux[testkit]'"
        ) from None
    plugin = importlib.import_module(LOADED_NAME)
    if not config.pluginmanager.has_plugin(LOADED_NAME):
        config.pluginmanager.register(plugin, LOADED_NAME)
