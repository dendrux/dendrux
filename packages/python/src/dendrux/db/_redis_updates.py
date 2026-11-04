"""Redis Pub/Sub transport for live updates.

``RedisUpdates`` implements the two transport hooks of ``LiveUpdates`` on ``redis.asyncio``:

* ``send`` publishes the base's encoded messages in one pipelined round trip on a connection
  that only the sender owns (a one-slot pool, no retries, no idle second client);
* ``receive`` subscribes on a dedicated connection, reports healthy only after the server
  acknowledged the subscription, and proves liveness with a token-matched ``PING`` on that same
  connection. It never reconnects by itself: redis-py would resubscribe silently and hide a gap.

The module imports without the ``redis`` package; constructing ``RedisUpdates`` is what needs
``pip install dendrux[redis]``. Nothing here logs. Errors that leave the transport are plain
``TimeoutError`` / ``ConnectionError`` / ``RuntimeError`` objects with a fixed text and no
chained original, because redis-py puts host and port into its messages.
"""

from __future__ import annotations

import asyncio
import secrets
from contextlib import suppress
from time import monotonic
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from dendrux.db._live_updates import LiveUpdates

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Mapping, Sequence

    from dendrux.db._live_updates import _Gap

INSTALL_HINT = "RedisUpdates needs the Redis client: pip install dendrux[redis]"
MAX_CHANNEL_LENGTH = 512
_SCHEMES = ("redis://", "rediss://", "unix://")
# Options that would defeat a guarantee of this transport, or move the destination away from url.
_FORBIDDEN_OPTIONS = frozenset(
    {
        "socket_timeout",
        "socket_connect_timeout",
        "retry",
        "retry_on_timeout",
        "retry_on_error",
        "max_connections",
        "connection_pool",
        "connection_class",
        "single_connection_client",
        "health_check_interval",
        "host",
        "port",
        "unix_socket_path",
    }
)
_POOL_ATTRIBUTES = ("_in_use_connections", "_available_connections")


def _load_api() -> Any:
    """The parts of ``redis.asyncio`` this module uses; the one place that imports redis."""
    try:
        from redis import exceptions
        from redis.asyncio import ConnectionPool, Redis
        from redis.asyncio.retry import Retry
        from redis.backoff import NoBackoff
    except ImportError:
        raise ImportError(INSTALL_HINT) from None
    return SimpleNamespace(
        ConnectionPool=ConnectionPool,
        Redis=Redis,
        Retry=Retry,
        NoBackoff=NoBackoff,
        ConnectionError=exceptions.ConnectionError,
        TimeoutError=exceptions.TimeoutError,
        RedisError=exceptions.RedisError,
    )


class _Redacted:
    """Holds a value whose text must never reach a repr, a log line or a traceback."""

    __slots__ = ("_value",)

    def __init__(self, value: Any) -> None:
        self._value = value

    def reveal(self) -> Any:
        return self._value

    def __repr__(self) -> str:
        return "<redacted>"

    __str__ = __repr__

    def __reduce__(self) -> Any:
        raise TypeError("a RedisUpdates connection setting cannot be pickled")


def _checked_url(url: object) -> str:
    if not isinstance(url, str) or not url.startswith(_SCHEMES):
        raise ValueError("url must be a string starting with redis://, rediss:// or unix://")
    return url


def _checked_channel(channel: object) -> str:
    if (
        not isinstance(channel, str)
        or not 0 < len(channel) <= MAX_CHANNEL_LENGTH
        or not channel.isprintable()
        or " " in channel
    ):
        raise ValueError(
            f"channel must be 1 to {MAX_CHANNEL_LENGTH} printable characters without spaces"
        )
    return channel


def _checked_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    chosen = dict(options or {})
    refused = sorted(key for key in chosen if key in _FORBIDDEN_OPTIONS)
    if refused:
        raise ValueError(
            "client_options may not set " + ", ".join(refused) + ": RedisUpdates fixes the "
            "timeouts, retries and single connection, and url gives the destination"
        )
    return chosen


def _abort_transports(pool: Any, extra: Iterable[Any] = ()) -> None:
    """Synchronously drop the sockets of every connection a pool holds, plus ``extra``."""
    connections = [conn for conn in extra if conn is not None]
    if pool is not None:
        for name in _POOL_ATTRIBUTES:
            connections.extend(getattr(pool, name, None) or ())
    for connection in connections:
        with suppress(Exception):
            connection._writer.transport.abort()


def _text(data: object) -> str:
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data if isinstance(data, str) else ""


class _Subscription:
    """One listen connection: its pool, its Pub/Sub object and its liveness state."""

    def __init__(self, owner: RedisUpdates) -> None:
        self.owner = owner
        self.pool = owner._new_pool(owner.send_timeout_s * 4)
        self.pubsub = owner._api.Redis(connection_pool=self.pool).pubsub()
        self._idle_until = monotonic() + owner.health_check_s
        self._token: str | None = None
        self._pong_by = 0.0
        owner._subscription = self

    def abort(self) -> None:
        _abort_transports(self.pool, [getattr(self.pubsub, "connection", None)])

    async def open(self) -> None:
        owner = self.owner
        async with asyncio.timeout(owner.send_timeout_s * 4) as scope:
            await self.pubsub.subscribe(owner._channel)
            while True:
                remaining = max(0.0, (scope.when() or 0.0) - asyncio.get_running_loop().time())
                reply = await self.pubsub.get_message(
                    ignore_subscribe_messages=False, timeout=remaining
                )
                if self._acknowledges(reply):
                    break
        self._idle_until = monotonic() + owner.health_check_s

    def _acknowledges(self, reply: Any) -> bool:
        return (
            isinstance(reply, dict)
            and reply.get("type") == "subscribe"
            and _text(reply.get("channel")) == self.owner._channel
        )

    async def step(self) -> str | _Gap | None:
        """Read at most one reply; the next message or gap to yield, or ``None`` to go on."""
        deadline = self._idle_until if self._token is None else self._pong_by
        wait = deadline - monotonic()
        if wait <= 0:
            await self._on_silence()
            return None
        reply = await self.pubsub.get_message(ignore_subscribe_messages=False, timeout=wait)
        return self._interpret(reply) if isinstance(reply, dict) else None

    def _interpret(self, reply: dict[str, Any]) -> str | _Gap | None:
        kind = reply.get("type")
        if kind == "message":
            self._idle_until = monotonic() + self.owner.health_check_s
            return _text(reply.get("data"))
        if kind == "pong" and self._token is not None and _text(reply.get("data")) == self._token:
            self._token = None
            self._idle_until = monotonic() + self.owner.health_check_s
        elif self._acknowledges(reply):
            # a second subscribe acknowledgement means the client resubscribed behind our back
            return LiveUpdates.GAP
        return None

    async def _on_silence(self) -> None:
        if self._token is not None:
            raise TimeoutError
        self._token = secrets.token_hex(8)
        self._pong_by = monotonic() + self.owner.send_timeout_s * 4
        await self.pubsub.ping(self._token)

    async def close(self) -> None:
        self.abort()  # first, synchronously: nothing below may leave the socket open
        if self.owner._subscription is self:
            self.owner._subscription = None
        with suppress(Exception):
            await self.pubsub.aclose()
        with suppress(Exception):
            await self.pool.aclose()


class RedisUpdates(LiveUpdates):
    """Live updates over Redis Pub/Sub (``pip install dendrux[redis]``).

    Pub/Sub ignores the Redis database number, so channel names must be unique per application
    and environment; the default is a starting point, not a namespace. Every subscriber receives
    every message on the channel: Redis moves the fan-out off the database, it does not remove it.
    """

    def __init__(
        self,
        url: str,
        *,
        channel: str = "dendrux:events",
        client_options: Mapping[str, Any] | None = None,
        **limits: Any,
    ) -> None:
        self._api = _load_api()
        super().__init__(**limits)
        self._channel = _checked_channel(channel)
        self._url = _Redacted(_checked_url(url))
        self._options = _Redacted(_checked_options(client_options))
        self._pool: Any = None
        self._client: Any = None
        self._pipeline: Any = None
        self._subscription: _Subscription | None = None
        self._check_options()

    def __repr__(self) -> str:
        return f"RedisUpdates(channel={self._channel!r})"

    def _new_pool(self, socket_timeout_s: float) -> Any:
        api = self._api
        options = {
            **self._options.reveal(),
            "max_connections": 1,
            "socket_timeout": socket_timeout_s,
            "socket_connect_timeout": self.send_timeout_s,
            "retry": api.Retry(api.NoBackoff(), 0),
        }
        return api.ConnectionPool.from_url(self._url.reveal(), **options)

    def _check_options(self) -> None:
        """Refuse a bad url or option now, without echoing either."""
        valid = True
        try:
            self._new_pool(self.send_timeout_s).make_connection()
        except Exception:
            valid = False
        if not valid:
            raise ValueError("url or client_options are not accepted by the Redis client")

    def _plain_error(self, error: BaseException) -> Exception:
        if isinstance(error, (TimeoutError, self._api.TimeoutError)):
            return TimeoutError("Redis live updates timed out")
        if isinstance(error, (OSError, self._api.ConnectionError)):
            return ConnectionError("Redis live updates connection failed")
        return RuntimeError("Redis live updates failed")

    # -- send ------------------------------------------------------------------------------

    def _sender(self) -> Any:
        if self._client is None:
            self._pool = self._new_pool(self.send_timeout_s)
            self._client = self._api.Redis(connection_pool=self._pool)
        return self._client

    async def send(self, messages: Sequence[str]) -> None:
        pipeline = self._sender().pipeline(transaction=False)
        for message in messages:
            pipeline.publish(self._channel, message)
        self._pipeline = pipeline
        failure: Exception | None = None
        try:
            await pipeline.execute()
        except Exception as error:
            failure = self._plain_error(error)
        finally:
            self._pipeline = None
        if failure is not None:
            raise failure

    def abort_sender(self) -> None:
        pipeline, pool = self._pipeline, self._pool
        self._pipeline = self._pool = self._client = None  # the next send starts from scratch
        _abort_transports(pool, [getattr(pipeline, "connection", None)])

    async def aclose_sender(self) -> None:
        client, pool = self._client, self._pool
        self._pipeline = self._pool = self._client = None
        if client is not None:
            with suppress(Exception):
                await client.aclose()
        if pool is not None:
            with suppress(Exception):
                await pool.aclose()

    # -- receive ---------------------------------------------------------------------------

    async def receive(self) -> AsyncIterator[str | None | _Gap]:
        subscription = _Subscription(self)
        failure: Exception | None = None
        try:
            await subscription.open()
            yield None
            while True:
                item = await subscription.step()
                if item is not None:
                    yield item
        except Exception as error:
            failure = self._plain_error(error)
        finally:
            await subscription.close()
        if failure is not None:
            raise failure

    def abort_listener(self) -> None:
        if self._subscription is not None:
            self._subscription.abort()
