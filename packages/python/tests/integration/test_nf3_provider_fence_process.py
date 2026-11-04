"""Provider fencing after multiple processes actually acquired the same run."""

import asyncio
import multiprocessing

import pytest

from dendrux.types import RunStatus
from tests._fixtures.storage import pin_storage_format
from tests._helpers.provider_fence_process import fence_agent, provider_fence_worker
from tests._helpers.scripted_provider import ScriptedLLM, tool_call


async def _receive(pipe):
    assert await asyncio.to_thread(pipe.poll, 30), "worker missed provider boundary"
    message = pipe.recv()
    assert "error" not in message, message.get("error")
    return message


@pytest.mark.parametrize("engine", ["postgres"], indirect=True)
async def test_only_current_owner_dispatches_among_four_processes(store, engine):
    await pin_storage_format(engine, 1)
    seed = fence_agent(store, ScriptedLLM([tool_call("step", pid="pause")]))
    workers = []
    try:
        paused = await seed.run("go", tenant_id="tenant")
        assert paused.status == RunStatus.PAUSED
        pause = await store.get_pause_state(paused.run_id)
        original = await store.get_run_control(paused.run_id)
        context = multiprocessing.get_context("spawn")
        ready = []
        for index in range(4):
            parent, child = context.Pipe()
            process = context.Process(
                target=provider_fence_worker,
                args=(
                    engine.url.render_as_string(hide_password=False),
                    paused.run_id,
                    paused.pause_id,
                    child,
                ),
            )
            process.start()
            child.close()
            workers.append((process, parent))
            message = await _receive(parent)
            assert message["phase"] == "ready"
            assert message["epoch"] == original.claim_epoch + index + 1
            ready.append(message)
            if index < 3:
                # A real committed handoff makes this blocked process stale.
                assert await store.try_pause_run(
                    paused.run_id,
                    status="paused",
                    pause_data=pause,
                    claim_epoch=message["epoch"],
                )
        assert len({item["pid"] for item in ready}) == 4
        assert len({item["backend_pid"] for item in ready}) == 4
        for _, pipe in workers:
            pipe.send("release")
        results = await asyncio.gather(*(_receive(pipe) for _, pipe in workers))
        for process, _ in workers:
            await asyncio.to_thread(process.join, 10)
            assert process.exitcode == 0
        assert all(result["phase"] == "finished" for result in results)
        assert [result["calls"] for result in results] == [0, 0, 0, 1]
        assert [result["fenced"] for result in results] == [True, True, True, False]
        assert results[-1]["status"] == RunStatus.SUCCESS.value
        final = await store.get_run(paused.run_id)
        assert final.claim_epoch == ready[-1]["epoch"]
        assert final.status == RunStatus.SUCCESS.value
        events = await store.get_run_events(paused.run_id)
        assert sum(event.event_type == "run.completed" for event in events) == 1
        assert sum(event.event_type == "llm.completed" for event in events) == 2
    finally:
        for process, pipe in workers:
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, 10)
            pipe.close()
        await seed.close()
