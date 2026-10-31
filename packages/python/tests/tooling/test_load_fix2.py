"""Caller cancellation and load-specific MySQL database isolation."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tools import load
from tools import load_monitor as monitor


@pytest.mark.parametrize("phase", ["sample", "rollback", "close"])
async def test_cancelling_stop_propagates_after_connection_cleanup(monkeypatch, phase):
    sampler = monitor.ResourceMonitor(None)
    entered = asyncio.Event()
    release_close = asyncio.Event()
    closed = asyncio.Event()
    close_cancelled = []

    async def rollback():
        if phase == "rollback":
            entered.set()
            await asyncio.Event().wait()

    async def sample():
        if phase == "sample":
            entered.set()
            await asyncio.Event().wait()
        sampler.buffer.add({"db.connections": 1})

    async def close():
        try:
            if phase == "close":
                entered.set()
                await release_close.wait()
            closed.set()
        except asyncio.CancelledError:
            close_cancelled.append(True)
            raise

    sampler.conn = SimpleNamespace(rollback=rollback, close=close)
    monkeypatch.setattr(sampler, "sample", sample)
    task = asyncio.create_task(sampler.stop(1))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        if phase == "close":
            await asyncio.sleep(0.01)
            assert not task.done(), "stop must finish connection cleanup before cancellation exits"
        release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
        assert closed.is_set()
        assert sampler.conn is None
        assert not close_cancelled
    finally:
        release_close.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("driver", ["mysql", "mysql+asyncmy", "mysql+aiomysql", "mariadb"])
def test_other_tracks_private_test_names_are_refused(driver):
    url = f"{driver}://root:dendrux_test@127.0.0.1:53306/dendrux_p5_b_x_test"
    with pytest.raises(ValueError, match="_load_"):
        load.check_private_url(url)
    with pytest.raises(ValueError, match="_load_"):
        load.check_private_url(url, allow_port=True, allow_remote_host=True)


@pytest.mark.parametrize("driver, port", [("mysql+asyncmy", 53306), ("mariadb", 53307)])
def test_mysql_load_database_names_are_accepted(driver, port):
    parsed = load.check_private_url(
        f"{driver}://root:dendrux_test@127.0.0.1:{port}/dendrux_load_mysql_test"
    )
    assert parsed.database == "dendrux_load_mysql_test"
    assert parsed.drivername == "mysql+asyncmy"


def test_postgres_private_names_do_not_require_load_marker():
    parsed = load.check_private_url("postgresql://u:p@localhost:5432/dendrux_p5_b_x_test")
    assert parsed.database == "dendrux_p5_b_x_test"


async def test_scenario_runner_does_not_absorb_caller_cancellation(monkeypatch):
    async def noop(*args, **kwargs):
        pass

    async def metadata(*args):
        return {"version": "test server"}

    async def cancelled(*args):
        raise asyncio.CancelledError

    monkeypatch.setattr(load, "make_engine", lambda *a, **kw: SimpleNamespace(dispose=noop))
    monkeypatch.setattr(load, "make_store", lambda *a, **kw: None)
    monkeypatch.setattr(load, "reset_db", noop)
    monkeypatch.setattr(load, "server_meta", metadata)
    monkeypatch.setattr(load, "machine_meta", lambda: {"dendrux_file": "test checkout"})
    monkeypatch.setattr(load, "monitored_job", cancelled)
    args = SimpleNamespace(
        allow_remote_host=False,
        allow_port=False,
        monitor_interval=1,
        url="postgresql://u:p@localhost:5432/task_test",
        scale="tiny",
        scenarios="runs",
        baseline=None,
        workers=1,
        leak_cycles=1,
        leak_thresholds=monitor.DEFAULT_THRESHOLDS,
        statement_budgets={},
        json=None,
        no_explain=True,
        write_baseline=None,
    )
    with pytest.raises(asyncio.CancelledError):
        await load.run_all(args)
