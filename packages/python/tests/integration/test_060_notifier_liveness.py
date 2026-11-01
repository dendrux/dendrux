"""P1-2: a test-owned TCP proxy drops no socket but freezes delivery."""

from __future__ import annotations

import asyncio
import contextlib
import logging

import pytest
from sqlalchemy.engine import make_url

from dendrux.store import PostgresEventNotifier
from dendrux.types import generate_ulid


class Proxy:
    def __init__(self, host, port):
        self.host, self.port = host, port
        self.gate = asyncio.Event()
        self.gate.set()
        self.tasks = set()
        self.writers = set()

    async def __aenter__(self):
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        self.port_number = self.server.sockets[0].getsockname()[1]
        return self

    async def pipe(self, reader, writer):
        while chunk := await reader.read(65536):
            await self.gate.wait()
            writer.write(chunk)
            await writer.drain()

    async def handle(self, reader, writer):
        task = asyncio.current_task()
        self.tasks.add(task)
        self.writers.add(writer)
        pipes = []
        try:
            remote_reader, remote = await asyncio.open_connection(self.host, self.port)
            self.writers.add(remote)
            pipes = [
                asyncio.create_task(self.pipe(reader, remote)),
                asyncio.create_task(self.pipe(remote_reader, writer)),
            ]
            await asyncio.wait(pipes, return_when=asyncio.FIRST_COMPLETED)
        except (ConnectionError, OSError):
            pass
        finally:
            for pipe in pipes:
                pipe.cancel()
            await asyncio.gather(*pipes, return_exceptions=True)
            writer.close()
            if "remote" in locals():
                remote.close()
            self.tasks.discard(task)

    async def __aexit__(self, *exc):
        self.server.close()
        await self.server.wait_closed()
        self.gate.set()
        for task in list(self.tasks):
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        for writer in self.writers:
            writer.close()
            with contextlib.suppress(ConnectionError, TimeoutError):
                await asyncio.wait_for(writer.wait_closed(), 1)


class Blackhole(Proxy):
    async def handle(self, reader, writer):
        task = asyncio.current_task()
        self.tasks.add(task)
        self.writers.add(writer)
        try:
            while await reader.read(65536):
                pass
        finally:
            writer.close()
            self.tasks.discard(task)


async def test_p1_2_frozen_tcp_path_degrades_then_recovers(
    pg_test_url, engine, store, caplog, record_property
):
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres LISTEN liveness")
    parsed = make_url(pg_test_url)
    caplog.set_level(logging.INFO)
    async with Proxy(parsed.host, parsed.port or 5432) as proxy:
        url = parsed.set(
            host="127.0.0.1", port=proxy.port_number, query={**parsed.query, "sslmode": "disable"}
        ).render_as_string(hide_password=False)
        notifier = PostgresEventNotifier(
            url,
            health_check_interval_s=0.05,
            health_check_timeout_s=0.1,
            reconnect_backoff=(0.1, 0.2),
        )
        await notifier.start()
        try:
            async with notifier.subscribe(generate_ulid()) as sub:
                proxy.gate.clear()
                started = asyncio.get_running_loop().time()
                async with asyncio.timeout(1):
                    while sub.state.healthy:
                        await asyncio.sleep(0.001)
                elapsed = asyncio.get_running_loop().time() - started
                assert sub.event.is_set()
                record_property("silent_loss_detection_s", elapsed)
                print(f"SILENT_LOSS_DETECTION_S={elapsed:.4f}")
                proxy.gate.set()
                async with asyncio.timeout(2):
                    while not sub.state.healthy:
                        await asyncio.sleep(0.001)
                assert sub.state.generation == 3
                codes = [getattr(r, "code", None) for r in caplog.records]
                assert codes.count("notifier_degraded") == codes.count("notifier_recovered") == 1
        finally:
            proxy.gate.set()
            await notifier.stop()


async def test_p1_3_ten_follows_poll_when_owned_listener_is_blackholed(
    pg_test_url, engine, store, caplog, record_property
):
    if engine.dialect.name != "postgresql":
        pytest.skip("Postgres read path with unavailable listener")
    from dendrux.store import RunStore, StoredEvent

    caplog.set_level(logging.INFO)
    run = generate_ulid()
    await store.create_run(run, "owned-herd")
    await store.save_run_event(run, event_type="llm.completed")
    parsed = make_url(pg_test_url)
    async with Blackhole(parsed.host, parsed.port or 5432) as proxy:
        proxy.gate.clear()
        url = parsed.set(
            host="127.0.0.1", port=proxy.port_number, query={**parsed.query, "sslmode": "disable"}
        ).render_as_string(hide_password=False)
        reader = RunStore.from_database_url(pg_test_url, listen_url=url)
        reader._owned_notifier.connect_timeout_s = 0.05
        reader._owned_notifier.reconnect_backoff = (1, 1)
        follows = [reader.follow(run, tenant_id=None, until_terminal=False) for _ in range(10)]
        started = asyncio.get_running_loop().time()
        try:
            results = await asyncio.gather(
                *(anext(follow) for follow in follows), return_exceptions=True
            )
            elapsed = asyncio.get_running_loop().time() - started
            assert all(isinstance(item, StoredEvent) for item in results), results
            codes = [getattr(r, "code", None) for r in caplog.records]
            assert codes.count("notifier_unavailable") == 1
            assert elapsed < 1
            record_property("owned_herd_delivery_s", elapsed)
            print(f"OWNED_HERD_DELIVERY_S={elapsed:.4f}; CALLERS=10")
        finally:
            for follow in follows:
                await follow.aclose()
            await reader.close()
