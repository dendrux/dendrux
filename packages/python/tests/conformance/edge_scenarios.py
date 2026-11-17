"""Additional format parity cells, separate from the immutable release corpus."""

from datetime import timedelta

from sqlalchemy import update

from dendrux.db.models import AgentRun
from dendrux.guardrails import PII
from dendrux.types import Clarification
from tests.conformance.harness import Cell, calls, text
from tests.conformance.scenarios import (
    QuestionStrategy,
    _paused_client,
    add,
    call_ids,
    client_result,
    multiply,
    read_range,
)


class OptionsQuestion(QuestionStrategy):
    def parse_response(self, response):
        step = super().parse_response(response)
        if isinstance(step.action, Clarification):
            step.action.options[:] = ["one", "two"]
        return step


def _flow(kind, streaming):
    async def flow(ctx):
        start = ctx.stream if streaming else ctx.run
        if kind == "question_options":
            agent = ctx.agent(ctx.provider(text("QUESTION: Which?"), text("done")))
            paused = await start(agent, "go", strategy=OptionsQuestion())
            await ctx.submit_input(agent, paused.run_id, "one", expected_pause_id=paused.pause_id)
            return
        if kind == "indexed_block":
            agent = ctx.agent(
                ctx.provider(calls(("read_range", {"sheet": "s"}, "c"))),
                tools=[read_range],
                guardrails=[PII(action="block")],
            )
            paused = await start(agent, "go")
            await ctx.resume(
                agent,
                paused.run_id,
                tool_results=[
                    client_result("read_range", call_ids(paused)[0], '"person@example.com"')
                ],
                expected_pause_id=paused.pause_id,
            )
            return
        script = {
            "unknown_multi": [
                ("add", {"a": 1, "b": 2}, "a"),
                ("multiply", {"a": 2, "b": 3}, "s"),
                ("nope", {}, "n"),
                ("add", {"a": 3, "b": 4}, "b"),
            ],
            "mixed_invalid": [
                ("add", {"a": 1, "b": 2}, "a"),
                ("nope", {}, "n"),
                ("multiply", {"a": 2, "b": 3}, "s"),
                ("add", {"bad": 1}, "bad"),
            ],
            "client_first": [("read_range", {"sheet": "s"}, "c"), ("add", {"a": 1, "b": 2}, "a")],
        }[kind]
        agent = ctx.agent(
            ctx.provider(calls(*script), text("done")), tools=[add, multiply, read_range]
        )
        result = await start(agent, "go")
        if result.pending:
            await ctx.resume(
                agent,
                result.run_id,
                tool_results=[client_result("read_range", call_ids(result)[0])],
                expected_pause_id=result.pause_id,
            )

    return flow


EDGE_CELLS = [
    Cell(
        id=f"edge_{kind}__{'stream' if streaming else 'run'}",
        scenario=kind,
        flow=_flow(kind, streaming),
        paths=("stream" if streaming else "run",),
        tags=(),
    )
    for kind in (
        "unknown_multi",
        "mixed_invalid",
        "client_first",
        "question_options",
        "indexed_block",
    )
    for streaming in (False, True)
    if kind != "indexed_block" or not streaming
]


def shared(actual):
    result = {k: actual[k] for k in ("steps", "provider_calls", "public_traces")}
    result["runs"] = [
        {k: r[k] for k in ("record", "tool_calls", "notes", "paused_with_state")}
        for r in actual["runs"]
    ]
    return result


async def _default_sweep(ctx):
    """Since 0.7.0 an omitted ``on_stale`` is ``"reclaim"`` (the 0.6 cells pin ``"error"``).

    A crashed format-2 ReAct run is marked for ``agent.reclaim()`` and stays running; on a
    format-1 database the same row is finalised as ``stale_running_legacy``. Abandoned
    waiting runs are reaped exactly as before.
    """
    agent, paused = await _paused_client(ctx, "run")
    orphan = await ctx.orphan_running_run()
    async with ctx.engine.begin() as conn:
        await conn.execute(
            update(AgentRun).where(AgentRun.id == orphan).values(loop_state={"loop": "react"})
        )
    await ctx.run(ctx.agent(ctx.provider(text("fresh"))), "stays untouched")
    await ctx.age_runs(paused.run_id, orphan)
    result = await ctx.sweep(
        stale_running=timedelta(minutes=20), abandoned_waiting=timedelta(minutes=30)
    )
    marked = [r.run_id for r in result.reclaimable]
    finalised = [r.run_id for r in result.stale_running]
    assert marked + finalised == [orphan], (marked, finalised)
    await ctx.cancel_run(agent, paused.run_id, "after sweep")


#: Default-policy cells: format 1 and format 2 differ by design, so there is no parity check.
DEFAULT_CELLS = [
    Cell(
        id="sweep_default__reclaims",
        scenario="sweep_default",
        flow=_default_sweep,
        paths=("run", "sweep", "cancel_run"),
    )
]
