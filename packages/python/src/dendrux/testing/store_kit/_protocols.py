"""Shapes of the objects the store author's fixtures hand to the kit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Awaitable
    from datetime import datetime, timedelta

#: Where an armed fault fires inside the next write transaction of a store.
FaultPoint = Literal["before_commit", "after_commit", "mid_write"]
#: ``error`` raises an exception; ``cancel`` raises ``asyncio.CancelledError``.
FaultKind = Literal["error", "cancel"]
FAULT_POINTS: tuple[FaultPoint, ...] = ("before_commit", "after_commit", "mid_write")


class StoreFactory(Protocol):
    """``dendrux_store_factory``: a new store over empty data, sync or async."""

    def __call__(self) -> Any | Awaitable[Any]: ...


class StoreDerive(Protocol):
    """``dendrux_store_reopen`` / ``dendrux_store_second_instance``.

    Takes a store the factory made and returns another instance over the same data. A reopened
    instance models a restarted process; a second instance is live next to the first, with its
    own connection(s), and both stay usable.
    """

    def __call__(self, store: Any) -> Any | Awaitable[Any]: ...


class Clock(Protocol):
    """``dendrux_clock``: moves the world forward.

    After ``advance(delta)`` everything the store wrote earlier must read as at least ``delta``
    older to its time-based features (sweeps, leases). How is up to the author: patch the
    store's time source, or age the stored timestamps.

    ``now()`` is optional: the store's current time as this clock sees it. A clock that moves the
    store's time source forward must provide it, because the kit computes lease cutoffs from it;
    a clock that ages stored timestamps may omit it (the wall clock is used).
    """

    def advance(self, delta: timedelta) -> None | Awaitable[None]: ...

    def now(self) -> datetime | Awaitable[datetime]: ...


class FaultInjector(Protocol):
    """``dendrux_fault_injector``: optional. Fails write transactions of a store.

    ``before_commit``: raise before anything is committed (nothing persists).
    ``after_commit``: raise after the commit (the write persists, the caller sees a failure: a
    lost acknowledgement). ``mid_write``: raise after some, but not all, of the transaction's
    writes ran (they must not persist; a one-write transaction is cut before its commit).

    ``kind="cancel"`` raises ``asyncio.CancelledError`` instead of an exception. ``after=n``
    lets the next ``n`` write transactions through first, so an operation made of several
    transactions can be cut between them (an atomic operation is one transaction and is
    unaffected). Without ``sticky`` the fault fires once; with it, on every write transaction
    until ``disarm``, so a store that retries internally still cannot succeed.

    ``fired(store)`` is how many times the fault fired since the last ``arm``. The kit asserts
    that every cut it asked for happened, so an injector that does nothing fails certification.
    """

    def arm(
        self,
        store: Any,
        point: FaultPoint,
        *,
        kind: FaultKind = "error",
        after: int = 0,
        sticky: bool = False,
    ) -> None: ...

    def fired(self, store: Any) -> int: ...

    def disarm(self, store: Any | None = None) -> None: ...
