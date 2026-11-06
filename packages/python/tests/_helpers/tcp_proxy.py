"""Test-owned TCP proxy that can freeze traffic without closing sockets."""

import asyncio
import contextlib


class Proxy:
    def __init__(self, host, port):
        self.host, self.port = host, port
        self.gate = asyncio.Event()
        self.gate.set()
        self.blocked = asyncio.Event()
        self.tasks = set()
        self.writers = set()

    async def __aenter__(self):
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        self.port_number = self.server.sockets[0].getsockname()[1]
        return self

    async def pipe(self, reader, writer):
        while chunk := await reader.read(65536):
            if not self.gate.is_set():
                self.blocked.set()
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
