"""A scriptable stand-in for ``redis.asyncio`` used by the RedisUpdates unit tests.

It models only what ``dendrux.db._redis_updates`` is allowed to use: a connection pool built from
a URL, a client over that pool, a pipeline that publishes, and a Pub/Sub object that reads
replies. The real server is exercised in ``tests/integration/test_redis_updates.py``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any


class FakeRedisError(Exception):
    """Stands in for ``redis.exceptions.RedisError``."""


class FakeConnectionError(FakeRedisError):
    """Stands in for ``redis.exceptions.ConnectionError`` (not an ``OSError``)."""


class FakeTimeoutError(FakeRedisError):
    """Stands in for ``redis.exceptions.TimeoutError``."""


class FakeTransport:
    def __init__(self) -> None:
        self.aborted = False

    def abort(self) -> None:
        self.aborted = True


class FakeWriter:
    def __init__(self) -> None:
        self.transport = FakeTransport()


class FakeConnection:
    def __init__(self, pool: FakePool) -> None:
        self.pool = pool
        self._writer = FakeWriter()
        self.disconnected = False

    @property
    def aborted(self) -> bool:
        return self._writer.transport.aborted


class FakePool:
    def __init__(self, world: FakeWorld, url: str, kwargs: dict[str, Any]) -> None:
        self.world = world
        self.url = url
        self.kwargs = kwargs
        self.max_connections = kwargs.get("max_connections")
        self._in_use_connections: set[FakeConnection] = set()
        self._available_connections: list[FakeConnection] = []
        self.connections: list[FakeConnection] = []
        self.closed = False

    def make_connection(self) -> FakeConnection:
        """Builds a connection object without connecting, like redis-py (and checks the options)."""
        for key in self.world.reject_keys:
            if key in self.kwargs:
                raise TypeError(f"unexpected keyword argument {key!r}")
        return FakeConnection(self)

    async def get_connection(self) -> FakeConnection:
        if self not in self.world.pools:  # a pool counts once it is used, not once it is built
            self.world.pools.append(self)
        if self._available_connections:
            conn = self._available_connections.pop()
        else:
            if self.max_connections is not None and len(self.connections) >= self.max_connections:
                raise AssertionError("pool exhausted: more than max_connections in use")
            conn = self.make_connection()
            self.connections.append(conn)
            self.world.connections.append(conn)
        self._in_use_connections.add(conn)
        return conn

    def release(self, conn: FakeConnection) -> None:
        # like redis-py, a released connection goes back to the pool even when it is dead
        self._in_use_connections.discard(conn)
        self._available_connections.append(conn)

    async def aclose(self) -> None:
        self.closed = True

    async def disconnect(self) -> None:
        self.closed = True


class FakePipeline:
    def __init__(self, client: FakeRedis, transaction: bool) -> None:
        self.client = client
        self.transaction = transaction
        self.commands: list[tuple[str, str]] = []
        self.connection: FakeConnection | None = None

    def publish(self, channel: str, message: str) -> FakePipeline:
        self.commands.append((channel, message))
        return self

    async def execute(self, raise_on_error: bool = True) -> list[int]:
        pool = self.client.pool
        world = pool.world
        conn = await pool.get_connection()
        self.connection = conn
        world.pipelines.append(self)
        try:
            await world.on_execute(self)
            world.published.extend(self.commands)
            return [1] * len(self.commands)
        except BaseException:
            conn.disconnected = True
            raise
        finally:
            pool.release(conn)
            self.connection = None


class FakePubSub:
    def __init__(self, client: FakeRedis) -> None:
        self.client = client
        self.world = client.pool.world
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.connection: FakeConnection | None = None
        self.subscribed: list[str] = []
        self.pings: list[str] = []
        self.closed = False
        self.world.pubsubs.append(self)

    async def subscribe(self, *channels: str) -> None:
        self.connection = await self.client.pool.get_connection()
        self.subscribed.extend(channels)
        world = self.world
        if world.ack_mode == "never":
            return
        for channel in channels:
            ack = {"type": "subscribe", "pattern": None, "channel": channel.encode(), "data": 1}
            if world.ack_mode == "wrong_channel":
                ack["channel"] = b"some:other:channel"
            if world.ack_delay:
                asyncio.get_running_loop().call_later(world.ack_delay, self.queue.put_nowait, ack)
            else:
                self.queue.put_nowait(ack)

    async def ping(self, message: str | None = None) -> None:
        assert message is not None
        self.pings.append(message)
        mode = self.world.pong_mode
        if mode == "silent":
            return
        token = message if mode == "echo" else message + "-stale"
        self.queue.put_nowait(
            {"type": "pong", "pattern": None, "channel": None, "data": token.encode()}
        )

    async def get_message(
        self, ignore_subscribe_messages: bool = False, timeout: float | None = 0.0
    ) -> Any:
        try:
            if timeout is None:
                item = await self.queue.get()
            elif timeout <= 0:
                item = self.queue.get_nowait()
            else:
                item = await asyncio.wait_for(self.queue.get(), timeout)
        except (TimeoutError, asyncio.QueueEmpty):
            return None
        if isinstance(item, BaseException):
            raise item
        return item

    def push(self, data: str | bytes, channel: str | None = None) -> None:
        raw = data.encode() if isinstance(data, str) else data
        name = (channel or self.subscribed[-1]).encode()
        self.queue.put_nowait({"type": "message", "pattern": None, "channel": name, "data": raw})

    def fail(self, error: BaseException) -> None:
        self.queue.put_nowait(error)

    async def aclose(self) -> None:
        self.closed = True
        if self.connection is not None:
            self.client.pool.release(self.connection)


class FakeRedis:
    def __init__(self, connection_pool: FakePool) -> None:
        self.pool = connection_pool
        self.closed = False
        connection_pool.world.clients.append(self)

    def pipeline(self, transaction: bool = True) -> FakePipeline:
        return FakePipeline(self, transaction)

    def pubsub(self) -> FakePubSub:
        return FakePubSub(self)

    async def aclose(self, close_connection_pool: bool | None = None) -> None:
        self.closed = True
        if close_connection_pool is not False:
            await self.pool.aclose()


class FakeRetry:
    def __init__(self, backoff: object, retries: int) -> None:
        self.backoff = backoff
        self.retries = retries


class FakeNoBackoff:
    pass


class FakeWorld:
    """Everything the fake client saw, and the switches that make it misbehave."""

    def __init__(self) -> None:
        self.pools: list[FakePool] = []
        self.clients: list[FakeRedis] = []
        self.connections: list[FakeConnection] = []
        self.pipelines: list[FakePipeline] = []
        self.pubsubs: list[FakePubSub] = []
        self.published: list[tuple[str, str]] = []
        self.ack_mode = "ok"  # ok | never | wrong_channel
        self.ack_delay = 0.0
        self.pong_mode = "echo"  # echo | silent | wrong
        self.execute_hook: Any = None  # async callable(pipeline): may stall or raise
        self.reject_keys: tuple[str, ...] = ("bogus_option",)  # pool options the client refuses

    async def on_execute(self, pipeline: FakePipeline) -> None:
        if self.execute_hook is not None:
            await self.execute_hook(pipeline)

    def api(self) -> SimpleNamespace:
        world = self

        class ConnectionPool(FakePool):
            @classmethod
            def from_url(cls, url: str, **kwargs: Any) -> FakePool:
                return FakePool(world, url, kwargs)

        return SimpleNamespace(
            ConnectionPool=ConnectionPool,
            Redis=FakeRedis,
            Retry=FakeRetry,
            NoBackoff=FakeNoBackoff,
            ConnectionError=FakeConnectionError,
            TimeoutError=FakeTimeoutError,
            RedisError=FakeRedisError,
        )
