"""A developer-owned PostgreSQL event listener, separate from the writer pool."""

from __future__ import annotations

import asyncio
import logging
import math
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any

from dendrux import _codes
from dendrux.db._notify import NotifierState, Subscription, validate_channel
from dendrux.errors import NotifierUnavailableError
from dendrux.store._follow import FollowCapacityError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


class PostgresEventNotifier:
    """One listener connection shared by bounded per-run subscriptions.

    Call ``start`` and ``stop`` on the same event loop. ``listen_url`` must
    name a direct or session-pooled endpoint: this is a host assertion.
    The startup LISTEN/NOTIFY self-test is best effort and cannot establish
    that a PgBouncer transaction pool will preserve later deliveries.
    Notifications carry run ids only; readers always fetch durable events.
    """

    def __init__(
        self,
        listen_url: str,
        *,
        channel: str = "dendrux_events",
        reconnect_backoff: tuple[float, float] = (1.0, 30.0),
        max_subscriptions: int = 10_000,
        connect_timeout_s: float = 5.0,
        reconnect_attempt_timeout_s: float = 5.0,
        stop_timeout_s: float = 5.0,
        self_test_timeout_s: float = 2.0,
        health_check_interval_s: float = 15.0,
        health_check_timeout_s: float = 5.0,
    ) -> None:
        from sqlalchemy.engine import make_url

        parsed = make_url(listen_url)
        if parsed.get_backend_name() not in ("postgres", "postgresql"):
            raise ValueError("listen_url must be a PostgreSQL URL.")
        validate_channel(channel)
        if (
            isinstance(max_subscriptions, bool)
            or not isinstance(max_subscriptions, int)
            or max_subscriptions <= 0
        ):
            raise ValueError("max_subscriptions must be a positive int.")
        for value in (
            *reconnect_backoff,
            connect_timeout_s,
            reconnect_attempt_timeout_s,
            stop_timeout_s,
            self_test_timeout_s,
            health_check_interval_s,
            health_check_timeout_s,
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("Notifier deadlines and backoff must be finite and positive.")
        if len(reconnect_backoff) != 2 or reconnect_backoff[0] > reconnect_backoff[1]:
            raise ValueError("reconnect_backoff must be (initial, maximum), initial <= maximum.")
        self._dsn = parsed.set(drivername="postgresql").render_as_string(hide_password=False)
        self.channel = channel
        self.reconnect_backoff = reconnect_backoff
        self.max_subscriptions = max_subscriptions
        self.connect_timeout_s = connect_timeout_s
        self.reconnect_attempt_timeout_s = reconnect_attempt_timeout_s
        self.stop_timeout_s = stop_timeout_s
        self.self_test_timeout_s = self_test_timeout_s
        self.health_check_interval_s = health_check_interval_s
        self.health_check_timeout_s = health_check_timeout_s
        self.state = NotifierState(False, 0)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._connection: Any = None
        self._task: asyncio.Task[None] | None = None
        self._connecting: asyncio.Task[None] | None = None
        self._start_result: asyncio.Task[None] | None = None
        self._start_waiters = 0
        self._lock = asyncio.Lock()
        self._active = False
        self._lifecycle_generation = 0
        self._lost = asyncio.Event()
        self._self_test: tuple[str, asyncio.Event] | None = None
        self._subscriptions: dict[str, set[asyncio.Event]] = {}
        self._count = 0

    def _check_loop(self) -> None:
        if self._loop is not None and asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("The notifier belongs to another event loop.")

    def _wake_all(self) -> None:
        for events in self._subscriptions.values():
            for event in events:
                event.set()

    def _health(self, healthy: bool, code: _codes.Code | None = None) -> None:
        if self.state.healthy == healthy:
            return
        self.state.healthy = healthy
        self.state.generation += 1
        self._wake_all()
        if code is not None:
            _codes.log_code(logger, code, run_id=None)

    def _on_notification(self, connection: Any, pid: int, channel: str, payload: str) -> None:
        if connection is not self._connection:
            return
        test = self._self_test
        if test is not None and payload == test[0]:
            test[1].set()
        for event in self._subscriptions.get(payload, ()):
            event.set()

    def _on_termination(self, connection: Any) -> None:
        if self._active and connection is self._connection:
            self._health(False, _codes.NOTIFIER_DEGRADED)
            self._lost.set()

    async def _connect(self, *, self_test: bool) -> None:
        import asyncpg  # type: ignore[import-untyped]

        connection = None
        try:
            # The delivery wait has its own deadline; connection and LISTEN
            # setup share the connection deadline.
            async with asyncio.timeout(self.connect_timeout_s):
                connection = await asyncpg.connect(
                    self._dsn,
                    timeout=self.connect_timeout_s,
                    server_settings={"application_name": "dendrux_event_notifier"},
                )
                self._connection = connection
                connection.add_termination_listener(self._on_termination)
                await connection.add_listener(self.channel, self._on_notification)
                if self_test:
                    import secrets

                    token, received = "self_test_" + secrets.token_hex(16), asyncio.Event()
                    self._self_test = (token, received)
                    await connection.execute("SELECT pg_notify($1, $2)", self.channel, token)
            if self_test:
                async with asyncio.timeout(self.self_test_timeout_s):
                    await received.wait()
            if connection.is_closed():
                raise ConnectionError("Listener closed during setup.")
        except BaseException:
            if connection is not None:
                connection.terminate()
            if self._connection is connection:
                self._connection = None
            raise
        finally:
            self._self_test = None

    async def start(self) -> None:
        """Connect and self-test in one shared attempt.

        Cancelling one caller leaves other callers waiting on that attempt.
        If every caller cancels, clean it up. Failure leaves no connection
        or reconnect task.
        """
        self._check_loop()
        if self._start_result is None:
            self._loop = asyncio.get_running_loop()
            self._start_result = self._loop.create_task(self._start(self._lifecycle_generation))
            self._start_result.add_done_callback(
                lambda done: None if done.cancelled() else done.exception()
            )
        attempt = self._start_result
        self._start_waiters += 1
        try:
            await asyncio.shield(attempt)
        finally:
            self._start_waiters -= 1
            if self._start_waiters == 0:
                self._start_result = None
                if not attempt.done():
                    attempt.cancel()
                    await asyncio.gather(attempt, return_exceptions=True)

    async def _start(self, generation: int) -> None:
        assert self._loop is not None
        async with self._lock:
            if generation != self._lifecycle_generation:
                raise NotifierUnavailableError("stopped")
            if self._active:
                return
            self._active = True
            self._lost.clear()
            self._connecting = self._loop.create_task(self._connect(self_test=True))
            try:
                await self._connecting
                if not self._active:
                    raise NotifierUnavailableError("stopped")
                self._health(True)
                self._task = self._loop.create_task(
                    self._reconnect(), name="dendrux-event-notifier"
                )
            except BaseException as exc:
                self._active = False
                self._health(False)
                if isinstance(exc, asyncio.CancelledError):
                    task = asyncio.current_task()
                    if task is not None and task.cancelling():
                        raise
                    raise NotifierUnavailableError("stopped") from None
                if isinstance(exc, NotifierUnavailableError):
                    raise
                raise NotifierUnavailableError("start_failed") from exc
            finally:
                self._connecting = None

    async def _reconnect(self) -> None:
        while self._active:
            try:
                async with asyncio.timeout(self.health_check_interval_s):
                    await self._lost.wait()
            except TimeoutError:
                await self._check_health()
                continue
            self._lost.clear()
            delay, maximum = self.reconnect_backoff
            while self._active and not self.state.healthy:
                await asyncio.sleep(delay)
                try:
                    async with asyncio.timeout(self.reconnect_attempt_timeout_s):
                        await self._connect(self_test=False)
                except Exception:
                    delay = min(delay * 2, maximum)
                    continue
                self._health(True, _codes.NOTIFIER_RECOVERED)

    async def _check_health(self) -> None:
        connection = self._connection
        try:
            async with asyncio.timeout(self.health_check_timeout_s):
                await connection.execute("SELECT 1")
        except Exception:
            self._on_termination(connection)
            if connection is not None:
                connection.terminate()

    async def stop(self) -> None:
        """Wake all waiters and close, forcibly after ``stop_timeout_s``."""
        self._check_loop()
        self._lifecycle_generation += 1
        self._active = False
        was_healthy = self.state.healthy
        self._health(False)
        if not was_healthy:
            self._wake_all()  # Also wake subscribers when already degraded/stopped.
        # Cancel an in-flight start before waiting for its lifecycle lock.
        if self._connecting is not None:
            self._connecting.cancel()
        async with self._lock:
            await self._stop()

    async def _stop(self) -> None:
        tasks = [t for t in (self._connecting, self._task) if t is not None]
        for task in tasks:
            task.cancel()
        connection, self._connection = self._connection, None

        async def cleanup() -> None:
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if connection is not None and not connection.is_closed():
                await connection.close(timeout=self.stop_timeout_s)

        closing = asyncio.create_task(cleanup())
        try:
            done, _ = await asyncio.wait({closing}, timeout=self.stop_timeout_s)
            if done:
                with suppress(Exception):
                    closing.result()  # The finally block forcibly closes on failure.
        finally:
            if connection is not None and not connection.is_closed():
                connection.terminate()
            if not closing.done():
                closing.cancel()
            await asyncio.gather(closing, return_exceptions=True)
            self._task = None

    @asynccontextmanager
    async def subscribe(self, run_id: str) -> AsyncIterator[Subscription]:
        """Subscribe until context exit; refuse beyond ``max_subscriptions``."""
        self._check_loop()
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        if self._count >= self.max_subscriptions:
            raise FollowCapacityError(self.max_subscriptions)
        event = asyncio.Event()
        self._subscriptions.setdefault(run_id, set()).add(event)
        self._count += 1
        try:
            yield Subscription(event, self.state)
        finally:
            self._check_loop()
            events = self._subscriptions[run_id]
            events.remove(event)
            if not events:
                del self._subscriptions[run_id]
            self._count -= 1
