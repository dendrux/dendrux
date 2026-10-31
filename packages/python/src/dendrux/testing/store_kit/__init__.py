"""Dendrux store test kit: a pytest plugin that certifies a ``StateStore``.

Run it against your store with ``pytest -p dendrux.testing.store_kit`` (or ``--dendrux-store-kit``)
after providing the ``dendrux_*`` fixtures documented in the guide; see ``FaultInjector``,
``Clock`` and ``StoreFactory`` for their shapes. ``pip install dendrux[testkit]`` brings pytest.

This module is registered as a pytest plugin by every environment that has ``dendrux``
installed, so it imports nothing optional. The real plugin (``_plugin``) is loaded only when the
kit is switched on, and then needs ``pytest-asyncio`` (``dendrux[testkit]``).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from dendrux.testing.store_kit._activation import LOADED_NAME, activated, add_options

if TYPE_CHECKING:
    import pytest

_LAZY = {
    "CertificationFailure": "dendrux.testing.store_kit._kit",
    "Kit": "dendrux.testing.store_kit._kit",
    "Declaration": "dendrux.testing.store_kit._declaration",
    "Clock": "dendrux.testing.store_kit._protocols",
    "FaultInjector": "dendrux.testing.store_kit._protocols",
    "FaultKind": "dendrux.testing.store_kit._protocols",
    "FaultPoint": "dendrux.testing.store_kit._protocols",
    "StoreDerive": "dendrux.testing.store_kit._protocols",
    "StoreFactory": "dendrux.testing.store_kit._protocols",
}

__all__ = [
    "CertificationFailure",
    "Clock",
    "Declaration",
    "FaultInjector",
    "FaultKind",
    "FaultPoint",
    "Kit",
    "StoreDerive",
    "StoreFactory",
    "pytest_addoption",
    "pytest_configure",
]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


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
