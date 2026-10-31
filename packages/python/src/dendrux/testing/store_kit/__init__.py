"""Dendrux store test kit: a pytest plugin that certifies a ``StateStore``.

Run it against your store with ``pytest -p dendrux.testing.store_kit`` (or ``--dendrux-store-kit``)
after providing the ``dendrux_*`` fixtures documented in the guide; see ``FaultInjector``,
``Clock`` and ``StoreFactory`` for their shapes. ``pip install dendrux[testkit]`` brings pytest.

The pytest11 entry point registers the top-level ``_dendrux_store_kit_bootstrap`` module, not this
one, so an inactive kit never imports ``dendrux``. This module re-exports its hooks so that
``-p dendrux.testing.store_kit`` and ``pytest_plugins`` behave the same. The real plugin
(``_plugin``) is loaded only when the kit is switched on, and then needs ``pytest-asyncio``
(``dendrux[testkit]``).
"""

from __future__ import annotations

import importlib
from typing import Any

# One implementation of the options and the switch, shared with the pytest11 entry point.
from _dendrux_store_kit_bootstrap import pytest_addoption, pytest_configure

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
