"""Cancellation cannot discard a coroutine that owns a durable resume claim."""

import asyncio

from dendrux import Agent, tool
from dendrux.types import LLMResponse, RunResult, RunStatus, ToolResult
from tests._helpers.scripted_provider import ScriptedLLM, tool_call


@tool(target="client")
async def read_range() -> str:
    return ""


async def test_cancel_at_submit_spawn_never_strands_claim_in_200_trials(store, monkeypatch):
    # Keep the newly created task off the scheduler until cancel_run has
    # delivered both its durable request and its in-process signal.
    for trial in range(200):
        provider = ScriptedLLM([tool_call("read_range", pid="read"), LLMResponse(text="done")])
        async with Agent(
            prompt="p",
            tools=[read_range],
            provider=provider,
            state_store=store,
            strict_tenancy=False,
        ) as agent:
            paused = await agent.run("read", tenant_id="A")
            manager = agent._task_manager
            original_spawn = manager.spawn
            release = asyncio.Event()
            spawned = []

            def gated_spawn(
                run_id, coro, release=release, original_spawn=original_spawn, spawned=spawned
            ):
                # Register the task immediately but gate its first execution,
                # matching the scheduler window after a committed claim.
                async def gated():
                    await release.wait()
                    return await coro

                task = original_spawn(run_id, gated())
                spawned.append((task, coro))
                return task

            monkeypatch.setattr(manager, "spawn", gated_spawn)
            submission = asyncio.create_task(
                agent.submit_tool_results(
                    paused.run_id,
                    [ToolResult("read_range", paused.pending.tool_calls[0].id, '"ok"')],
                    tenant_id="A",
                )
            )
            while not spawned and not submission.done():
                await asyncio.sleep(0)
            cancelled = await agent.cancel_run(paused.run_id, tenant_id="A")
            assert isinstance(cancelled, RunResult)
            release.set()
            result = (await asyncio.gather(submission, return_exceptions=True))[0]
            try:
                assert isinstance(result, RunResult), (trial, type(result).__name__)
                assert result.status == RunStatus.CANCELLED
                row = await store.get_run(paused.run_id)
                assert row.status == "cancelled", trial
                events = await store.get_run_events(paused.run_id)
                assert sum(e.event_type == "run.cancelled" for e in events) == 1
            finally:
                for task, coro in spawned:
                    await asyncio.gather(task, return_exceptions=True)
                    coro.close()
