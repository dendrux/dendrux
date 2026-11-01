"""What the store declares is complete and backed by what the author provided."""

from __future__ import annotations

from typing import Any

from dendrux.testing.store_kit._declaration import DEPLOYMENTS
from dendrux.testing.store_kit._kit import CertificationFailure, Kit
from dendrux.testing.store_kit.contracts._marks import contract


@contract("declaration")
async def test_deployment_is_declared(dendrux_kit: Kit) -> None:
    deployment = dendrux_kit.declaration.deployment
    if deployment is None:
        raise CertificationFailure(
            "capabilities.deployment is not declared; set it to one of "
            f"{', '.join(DEPLOYMENTS)} (a store without it cannot be certified)"
        )
    assert deployment in DEPLOYMENTS


@contract("declaration")
async def test_declared_capabilities_have_their_fixtures(
    dendrux_kit: Kit, dendrux_kit_provided: dict[str, Any]
) -> None:
    missing = [
        f"{name} ({reason})"
        for name, reason in dendrux_kit.declaration.required_fixtures().items()
        if name not in dendrux_kit_provided
    ]
    if missing:
        raise CertificationFailure("declared, but the fixture is missing: " + "; ".join(missing))


@contract("declaration")
async def test_declared_methods_match_the_declaration(dendrux_kit: Kit) -> None:
    # Building the kit already refused a declaration whose methods lack the promised keywords.
    assert dendrux_kit.declaration.capabilities.version == 1


@contract("declaration", needs=("notify",))
async def test_declared_notify_is_implemented(dendrux_kit: Kit) -> None:
    """The kit cannot test event hints, but it can refuse a notify promise nobody implemented.

    ``SQLStateStore`` declares ``notify`` and ships a hook that does nothing. A subclass that
    keeps both has promised hints it never sends. Delivery itself is not tested.
    """
    import inspect

    from dendrux.runtime.stores.sql.base import SQLStateStore

    hook = inspect.getattr_static(type(dendrux_kit.store), "_after_event_insert", None)
    if hook is SQLStateStore.__dict__["_after_event_insert"]:
        raise CertificationFailure(
            f"{dendrux_kit.declaration.store_name} declares notify but inherits the base "
            "_after_event_insert, which sends nothing: implement _after_event_insert or "
            "declare notify=False (the kit does not test notification delivery)"
        )
