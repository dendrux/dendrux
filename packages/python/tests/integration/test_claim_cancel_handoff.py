"""Cancellation cannot discard a coroutine that owns a durable resume claim."""

import asyncio
import sys
from io import StringIO

import pytest

from dendrux import Agent, tool
from dendrux.types import LLMResponse, RunResult, RunStatus, ToolResult
from tests._helpers.scripted_provider import ScriptedLLM, tool_call

TRIAL_TIMEOUT = 20.0
CLEANUP_TIMEOUT = 5.0


async def _fail_stalled_trial(trial, wait, trial_tasks, spawned, release, error=None):
    stacks = StringIO()
    for task in sorted(asyncio.all_tasks(), key=lambda task: task.get_name()):
        task.print_stack(file=stacks)

    release.set()
    tasks = {*trial_tasks, *(task for task, _ in spawned)}
    for task in tasks:
        if not task.done():
            task.cancel()
    done, pending = await asyncio.wait(tasks, timeout=CLEANUP_TIMEOUT)
    await asyncio.gather(*done, return_exceptions=True)
    for task, coro in spawned:
        if task.done():
            coro.close()
        else:
            task.add_done_callback(lambda _done, coro=coro: coro.close())
    message = (
        f"trial {trial}: {wait} exceeded the {TRIAL_TIMEOUT:g}s trial deadline; "
        f"cleanup pending: {len(pending)}\n{stacks.getvalue()}"
    )
    if error is not None:
        # Keep the original failure while retaining cleanup's task diagnostics.
        error.add_note(message)
    else:
        pytest.fail(message, pytrace=False)


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
            loop = asyncio.get_running_loop()
            deadline = loop.time() + TRIAL_TIMEOUT
            stalled = False
            trial_tasks = {submission}

            async def fail(
                wait,
                error=None,
                trial=trial,
                trial_tasks=trial_tasks,
                spawned=spawned,
                release=release,
            ):
                nonlocal stalled
                stalled = True
                await _fail_stalled_trial(trial, wait, trial_tasks, spawned, release, error)

            async def wait_for_tasks(
                tasks, wait, error=None, deadline=deadline, loop=loop, fail=fail
            ):
                _, pending = await asyncio.wait(tasks, timeout=max(0, deadline - loop.time()))
                if pending:
                    await fail(wait, error)

            try:
                while not spawned and not submission.done():
                    if loop.time() >= deadline:
                        await fail("pre-spawn poll")
                    await asyncio.sleep(0)
                cancel_task = asyncio.create_task(agent.cancel_run(paused.run_id, tenant_id="A"))
                trial_tasks.add(cancel_task)
                await wait_for_tasks({cancel_task}, "cancel_run")
                cancelled = cancel_task.result()
                assert isinstance(cancelled, RunResult)
                release.set()
                await wait_for_tasks({submission}, "submission completion")
                result = (await asyncio.gather(submission, return_exceptions=True))[0]
                assert isinstance(result, RunResult), (trial, type(result).__name__)
                assert result.status == RunStatus.CANCELLED
                row = await store.get_run(paused.run_id)
                assert row.status == "cancelled", trial
                events = await store.get_run_events(paused.run_id)
                assert sum(e.event_type == "run.cancelled" for e in events) == 1
            finally:
                release.set()
                if not stalled:
                    await wait_for_tasks(
                        {*trial_tasks, *(task for task, _ in spawned)},
                        "spawned-task completion",
                        error=sys.exc_info()[1],
                    )
                for task, coro in spawned:
                    if task.done():
                        await asyncio.gather(task, return_exceptions=True)
                        coro.close()
