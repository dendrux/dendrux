"""Closed-loop pools detach safely, warn once and release orphaned sessions after GC."""

import asyncio
import json
import sys
import uuid

import pytest

from dendrux import Agent


async def test_shared_pool_has_no_idle_background_task(engine, tmp_path):
    before = asyncio.all_tasks()
    url = (
        engine.url.render_as_string(hide_password=False)
        if engine.dialect.name == "postgresql"
        else f"sqlite+aiosqlite:///{tmp_path}/idle.db"
    )
    agent = Agent(prompt="p", database_url=url, strict_tenancy=False)
    try:
        await agent._resolve_state_store()
        assert not (asyncio.all_tasks() - before), "an idle pool must not own a background task"
    finally:
        await agent.close()


_SCRIPT = r"""
import asyncio, gc, json, logging, sys, warnings
from contextlib import AsyncExitStack
import asyncpg
from dendrux import Agent
from dendrux.db._shared_pool import release
warnings.simplefilter("error")
# The owner explicitly permits ResourceWarning from orphaned driver connections.
warnings.filterwarnings("ignore", category=ResourceWarning)
url, tag, trigger, style = sys.argv[1:]
unraisable, logged, agents, engines, counts, records = [], [], [], [], [], []
sys.unraisablehook = lambda event: unraisable.append(str(event.exc_value))
class Capture(logging.Handler):
    def emit(self, record):
        records.append(record)
logging.getLogger().addHandler(Capture())
logging.getLogger().setLevel(logging.WARNING)

def run(coro):
    if style == "run":
        return asyncio.run(coro)
    loop = asyncio.new_event_loop()
    loop.set_exception_handler(lambda loop, context: logged.append(context.get("message")))
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

async def count():
    conn = await asyncpg.connect(url.replace("postgresql+asyncpg:", "postgresql:"))
    try:
        return await conn.fetchval(
            "SELECT count(*) FROM pg_stat_activity "
            "WHERE datname=current_database() AND application_name=$1", tag
        )
    finally:
        await conn.close()

async def acquire():
    for _ in range(2):
        agent = Agent(
            name="lazy", prompt="p", strict_tenancy=False, database_url=url,
            database_options=dict(pool_size=2, max_overflow=0,
                connect_args=dict(server_settings=dict(application_name=tag))),
        )
        agents.append(agent)
        await agent._resolve_state_store()
        engines.append(agent._private_engine)
    assert engines[-1] is engines[-2], "holders must share one pool"
    gc.collect()  # reclaim pools detached by this acquisition, not the current pool
    async with AsyncExitStack() as stack:
        for _ in range(2):
            await stack.enter_async_context(agent._private_engine.connect())
        counts.append(await count())

baseline = run(count())
try:
    for _ in range(4):
        run(acquire())
        if trigger == "close":
            run(agents[-1].close())
        elif trigger == "release":
            run(release(engines[-1]))
        else:
            continue  # the next acquire detaches the previous entry
        gc.collect()
        assert run(count()) == baseline, "GC must release the retired PG sessions"
    for agent in agents:
        run(agent.close())
    gc.collect()
    assert run(count()) == baseline, "GC must return PG backend count to baseline"
    assert max(counts) == baseline + 2, counts
    assert len(records) == 4, "each retired entry must emit one structured warning"
    for record in records:
        assert record.levelno == logging.WARNING
        assert getattr(record, "code", None) == "shared_engine_loop_closed"
        assert "agent.close()" in record.getMessage() and "async with Agent" in record.getMessage()
        assert url not in str(record.__dict__) and tag not in str(record.__dict__)
        assert not record.exc_info
    del agents[:], engines[:]
    gc.collect()
    assert not logged, logged
    assert not unraisable, unraisable
    print(json.dumps(dict(loops=4, peak=max(counts), connections=baseline, warnings=len(records))))
finally:
    for agent in agents:
        run(agent.close())
    gc.collect()
"""


@pytest.mark.parametrize("trigger", ["acquire", "release", "close"])
@pytest.mark.parametrize("style", ["manual", "run"])
async def test_closed_loop_retirement_warns_once_and_gc_restores_pg_baseline(
    engine, trigger, style, record_property
):
    if engine.dialect.name != "postgresql":
        pytest.skip("real asyncpg connections after owner loop close")
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-W",
        "error",
        "-c",
        _SCRIPT,
        engine.url.render_as_string(hide_password=False),
        "t7_lazy_" + uuid.uuid4().hex,
        trigger,
        style,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), 45)
    assert proc.returncode == 0, stderr.decode()
    assert not stderr, stderr.decode()
    metrics = json.loads(stdout)
    assert metrics == dict(loops=4, peak=2, connections=0, warnings=4)
    record_property("lazy_closed_loop", json.dumps(metrics))
