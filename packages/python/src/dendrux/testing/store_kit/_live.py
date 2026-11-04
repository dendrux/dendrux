"""Live-update support for the store kit: a recording ``LiveUpdates`` and what it saw.

A store that announces changes hands run ids to a ``LiveUpdates`` after its commit was issued.
``RecordingLiveUpdates`` is the ``LiveUpdates`` the kit gives such a store: it keeps every id it
is asked to send, in order, and, when given a probe, asks it what a *second connection* can see at
the moment the announcement is sent. An announcement that goes out before its commit is visible
elsewhere shows up as a row that does not exist yet.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dendrux.db._live_updates import LiveUpdates

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

    from dendrux.db._live_updates import _Gap


@dataclass(frozen=True)
class RunView:
    """One run as a reader on another connection sees it."""

    exists: bool
    status: str | None
    events: int


@dataclass(frozen=True)
class RecordedBatch:
    """The ids of one sent message and what the probe saw when it was sent."""

    ids: tuple[str, ...]
    seen: dict[str, RunView] = field(default_factory=dict)
    #: The kind (exception class name) of a probe that failed; never its message.
    probe_error: str | None = None


class RecordingLiveUpdates(LiveUpdates):
    """A ``LiveUpdates`` that records what a store sends. Never delivers anything.

    The send and flush timeouts are roomy by default (5 s and 10 s): the probe reads a database.
    """

    def __init__(
        self,
        *,
        probe: Callable[[tuple[str, ...]], Awaitable[dict[str, RunView]]] | None = None,
        **limits: Any,
    ) -> None:
        limits.setdefault("send_timeout_s", 5.0)
        limits.setdefault("flush_deadline_s", 10.0)
        super().__init__(**limits)
        self.probe = probe
        self.batches: list[RecordedBatch] = []

    @property
    def ids(self) -> list[str]:
        """Every id sent so far, in send order (an id announced twice appears twice)."""
        return [run_id for batch in self.batches for run_id in batch.ids]

    def clear(self) -> None:
        self.batches.clear()

    async def send(self, messages: Sequence[str]) -> None:
        for message in messages:
            ids = self._decode(message)
            if ids is not None:
                self.batches.append(await self._record(ids))

    async def _record(self, ids: tuple[str, ...]) -> RecordedBatch:
        if self.probe is None:
            return RecordedBatch(ids)
        try:
            return RecordedBatch(ids, await self.probe(ids))
        except Exception as error:
            return RecordedBatch(ids, probe_error=type(error).__name__)

    async def receive(self) -> AsyncIterator[str | None | _Gap]:
        yield None
        await asyncio.Event().wait()  # subscribed, and nothing ever arrives
