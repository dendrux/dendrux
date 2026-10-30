"""Optional wake-up hints; the database remains the durable event log."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import asyncio
    from contextlib import AbstractAsyncContextManager


def validate_channel(channel: str) -> str:
    """Validate a PostgreSQL notification channel without echoing its value."""
    if (
        not isinstance(channel, str)
        or not channel
        or "\x00" in channel
        or len(channel.encode()) >= 64
    ):
        raise ValueError("channel must contain 1–63 UTF-8 bytes and no NUL.")
    return channel


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
