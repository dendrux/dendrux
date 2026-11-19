"""C9: a process dies mid-stream while a fan-out's owner tasks run; a fresh one recovers.

Phase one runs ``root.stream()`` in a child interpreter and dies with ``os._exit`` while
k of 3 children are finished and the rest are inside their provider calls (owner tasks
mid-flight), with the consumer still iterating. Phase two, a new interpreter, recovers
through the public ``sweep()`` + ``reclaim(root)``. Finished children never run again:
their provider calls, tool runs and saved rows are unchanged; the others run once more;
every call becomes exactly one row and one observation; nothing is left running.
Runs on file SQLite and Postgres.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.runtime.stores._factory import store_for_engine
from dendrux.types import RunEventType
from tests._helpers.fanout_effects import FanoutEffects
from tests.integration.test_fanout_nested_unwind import (
    TENANT,
    CrashSchedule,
    assert_crash_cut,
    assert_crash_effects,
    crash_workflow,
    fenced_checkpoint,
    install_crash_cut,
    records,
    recover_public,
)

pytestmark = pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)

CUTS = ["0", "1", "2", "nested"]
"""k of 3 children finished (k = 0, 1, n-1), and a nested fan-out below a sibling."""


def _env(engine, tmp_path):
    return {
        **os.environ,
        "PYTHONPATH": "src:.",
        "T6_CRASH_URL": engine.url.render_as_string(hide_password=False),
        "T6_EFFECTS": str(tmp_path / "effects.jsonl"),
        "T7_PHASE": str(tmp_path / "phase.json"),
    }


def _phase(env, mode, cut):
    return subprocess.run(
        [sys.executable, "-m", __name__, mode, cut],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("cut", CUTS)
async def test_c9_a_crash_mid_stream_recovers_without_rerunning_finished_children(
    store, engine, tmp_path, cut
):
    env = _env(engine, tmp_path)
    crashed = _phase(env, "crash", cut)
    assert crashed.returncode == 17, crashed.stderr[-3000:]
    seen = json.loads(Path(env["T7_PHASE"]).read_text())
    # The consumer was mid-stream: the workflow started, no terminal event was seen.
    assert seen["first"] == RunEventType.RUN_STARTED.value and not seen["terminal"]

    runs = await records(engine)
    root = next(row for row in runs if row.parent_run_id is None)
    bound_epoch = root.claim_epoch
    saved = {row.id: await store.get_run(row.id) for row in runs if row.status == "success"}
    running = {row.id: row.claim_epoch for row in runs if row.status == "running"}
    before = FanoutEffects(Path(env["T6_EFFECTS"])).counts()
    assert_crash_cut(runs, before, cut)
    assert running  # owners died holding their claims
    if cut in ("1", "2"):
        # Finished children's progress reached the stream before the crash.
        assert seen["child_events"] > 0

    recovered = _phase(env, "recover", cut)
    assert recovered.returncode == 0, recovered.stderr[-3000:]
    assert json.loads(Path(env["T7_PHASE"]).read_text()) == {"leaked_tasks": 0}

    assert (await store.get_run(root.id)).status == "success"
    for run_id, record in saved.items():
        assert await store.get_run(run_id) == record  # finished children untouched
    after = FanoutEffects(Path(env["T6_EFFECTS"])).counts()
    for row in runs:
        if row.id in saved:
            assert before.for_child(row.agent_name) == after.for_child(row.agent_name)
    for run_id, epoch in running.items():
        # The dead owners' claims are retired: their old epoch can write nothing.
        assert (await store.get_run(run_id)).claim_epoch > epoch
        await fenced_checkpoint(store, run_id, epoch)
    await fenced_checkpoint(store, root.id, bound_epoch)
    assert_crash_effects(after, cut)
    assert all(row.status != "running" for row in await records(engine))


async def _crash_phase(mode: str, cut: str) -> None:
    engine = create_async_engine(os.environ["T6_CRASH_URL"])
    store = store_for_engine(engine)
    effects = FanoutEffects(Path(os.environ["T6_EFFECTS"]))
    schedule = CrashSchedule(cut)
    root = crash_workflow(store, effects, mode, cut, schedule)
    phase = Path(os.environ["T7_PHASE"])
    if mode == "crash":
        install_crash_cut(store, schedule, cut)
        seen = {"first": None, "terminal": False, "child_events": 0}
        async with root.stream("go", tenant_id=TENANT) as stream:
            async for event in stream:
                kind = getattr(event.type, "value", event.type)
                seen["first"] = seen["first"] or kind
                seen["terminal"] = (
                    seen["terminal"]
                    or kind.startswith("run_")
                    and kind
                    not in (
                        "run_started",
                        "run_resumed",
                    )
                )
                if event.run_id not in (None, stream.run_id) and kind.startswith("tool_"):
                    seen["child_events"] += 1
                # The record is rewritten after every event; the cut ends the process.
                phase.write_text(json.dumps(seen))
        raise RuntimeError("crash cut was not reached")
    await recover_public(root, store, engine, cut)
    leaked = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    phase.write_text(json.dumps({"leaked_tasks": len(leaked)}))
    await root.close()
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(_crash_phase(sys.argv[1], sys.argv[2]))
