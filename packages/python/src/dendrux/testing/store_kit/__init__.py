"""Dendrux store test kit: a pytest plugin that certifies a ``StateStore``.

Run it against your store with ``pytest -p dendrux.testing.store_kit`` (or ``--dendrux-store-kit``)
after providing the ``dendrux_*`` fixtures documented in the guide; see ``FaultInjector``,
``Clock`` and ``StoreFactory`` for their shapes. ``pip install dendrux[testkit]`` brings pytest.
"""

from dendrux.testing.store_kit._declaration import Declaration
from dendrux.testing.store_kit._kit import CertificationFailure, Kit
from dendrux.testing.store_kit._plugin import (
    dendrux_kit,
    dendrux_kit_provided,
    pytest_addoption,
    pytest_configure,
    pytest_runtest_makereport,
)
from dendrux.testing.store_kit._protocols import (
    Clock,
    FaultInjector,
    FaultKind,
    FaultPoint,
    StoreDerive,
    StoreFactory,
)

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
    "dendrux_kit",
    "dendrux_kit_provided",
    "pytest_addoption",
    "pytest_configure",
    "pytest_runtest_makereport",
]
