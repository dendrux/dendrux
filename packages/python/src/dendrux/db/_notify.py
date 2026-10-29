"""Optional wake-up hints; the database remains the durable event log."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import asyncio
    from contextlib import AbstractAsyncContextManager


@dataclass
class NotifierState:
    """Shared health and monotonically increasing health generation."""

    healthy: bool
    generation: int


@dataclass(frozen=True)
class Subscription:
    """Coalescing wake-up signal and the notifier's shared health."""

    event: asyncio.Event
    state: NotifierState


class EventNotifier(Protocol):
    """Developer-owned notification lifecycle and per-run subscriptions."""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def subscribe(self, run_id: str) -> AbstractAsyncContextManager[Subscription]: ...
