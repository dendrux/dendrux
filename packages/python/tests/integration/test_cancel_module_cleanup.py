"""An owned ``database_url`` engine is disposed even when the caller cancels during disposal."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from importlib import import_module
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from dendrux import cancel_run
from dendrux.db import _schema_check, _sqlite_sync  # bind before any patch: see schema_hook
from dendrux.errors import PersistenceError, SchemaOutOfDateError
from dendrux.runtime import _cancel, sweep
from dendrux.types import RunResult, RunStatus

if TYPE_CHECKING:
    from pathlib import Path


def url_for(engine: Any, tmp_path: Path) -> str:
    if engine.dialect.name == "sqlite":
        return f"sqlite+aiosqlite:///{tmp_path / 'owned.db'}"
    return str(engine.url.render_as_string(hide_password=False))


def schema_hook(engine: Any) -> tuple[Any, str]:
    """The schema step to patch. ``_sqlite_sync`` binds ``ensure_schema_current`` when first
    imported, so both modules are imported at the top, before any test patches one."""
    if engine.dialect.name == "sqlite":
        return _sqlite_sync, "create_sqlite_schema"
    return _schema_check, "ensure_schema_current"


@pytest.mark.parametrize("exit_phase", ["success", "service_error", "schema_error"])
async def test_cancellation_during_owned_dispose_still_closes(
    engine: Any, tmp_path: Path, monkeypatch: Any, exit_phase: str
) -> None:
    sqlalchemy_asyncio = import_module("sqlalchemy.ext.asyncio")
    schema_module, schema_name = schema_hook(engine)
    real_schema = getattr(schema_module, schema_name)
    real_factory, real_dispose = sqlalchemy_asyncio.create_async_engine, AsyncEngine.dispose
    opened: list[AsyncEngine] = []
    completed: list[int] = []
    disposing, release = asyncio.Event(), asyncio.Event()

    def factory(*args: Any, **kwargs: Any) -> AsyncEngine:
        temp = real_factory(*args, **kwargs)
        opened.append(temp)
        return temp

    async def schema(temp: Any) -> None:
        await real_schema(temp)
        if exit_phase == "schema_error":
            raise SchemaOutOfDateError()

    async def service(*args: Any, **kwargs: Any) -> RunResult:
        if exit_phase == "service_error":
            raise PersistenceError()
        return RunResult(run_id="r", status=RunStatus.CANCELLED)

    async def dispose(self: AsyncEngine, *args: Any, **kwargs: Any) -> None:
        if self not in opened:
            await real_dispose(self, *args, **kwargs)
            return
        disposing.set()
        await release.wait()  # a slow asynchronous connection close
        await real_dispose(self, *args, **kwargs)
        completed.append(id(self))

    try:
        with monkeypatch.context() as patch:
            patch.setattr(sqlalchemy_asyncio, "create_async_engine", factory)
            patch.setattr(schema_module, schema_name, schema)
            patch.setattr(_cancel, "cancel_run", service)
            patch.setattr(AsyncEngine, "dispose", dispose)
            url = url_for(engine, tmp_path)
            task = asyncio.create_task(cancel_run("r", tenant_id="t1", database_url=url))
            await asyncio.wait_for(disposing.wait(), 10)
            task.cancel()
            await asyncio.sleep(0)
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 10)
            assert completed == [id(opened[0])], "cancellation abandoned owned-engine cleanup"
    finally:
        release.set()
        for temp in opened:
            await real_dispose(temp)


@pytest.mark.parametrize("exit_phase", ["success", "scan_error"])
async def test_sweep_cancellation_during_owned_dispose_still_closes(
    engine: Any, tmp_path: Path, monkeypatch: Any, exit_phase: str
) -> None:
    sqlalchemy_asyncio = import_module("sqlalchemy.ext.asyncio")
    real_factory, real_dispose = sqlalchemy_asyncio.create_async_engine, AsyncEngine.dispose
    real_scan = sweep._sweep_running
    opened: list[AsyncEngine] = []
    completed: list[int] = []
    scanned, disposing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    def factory(*args: Any, **kwargs: Any) -> AsyncEngine:
        temp = real_factory(*args, **kwargs)
        opened.append(temp)
        return temp

    async def scan(*args: Any, **kwargs: Any) -> Any:
        scanned.set()
        if exit_phase == "scan_error":
            raise PersistenceError()
        return await real_scan(*args, **kwargs)

    async def dispose(self: AsyncEngine, *args: Any, **kwargs: Any) -> None:
        if self not in opened or not scanned.is_set():
            await real_dispose(self, *args, **kwargs)
            return
        disposing.set()
        await release.wait()  # a slow asynchronous connection close
        await real_dispose(self, *args, **kwargs)
        completed.append(id(self))

    try:
        with monkeypatch.context() as patch:
            patch.setattr(sqlalchemy_asyncio, "create_async_engine", factory)
            patch.setattr(sweep, "_sweep_running", scan)
            patch.setattr(AsyncEngine, "dispose", dispose)
            url = url_for(engine, tmp_path)
            task = asyncio.create_task(
                sweep.sweep(database_url=url, stale_running=timedelta(minutes=5))
            )
            await asyncio.wait_for(disposing.wait(), 10)
            task.cancel()
            await asyncio.sleep(0)
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 10)
            assert completed == [id(opened[0])], "cancellation abandoned owned-engine cleanup"
    finally:
        release.set()
        for temp in opened:
            await real_dispose(temp)
