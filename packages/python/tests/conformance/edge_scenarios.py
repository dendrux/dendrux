"""Additional format parity cells, separate from the immutable release corpus."""

from dendrux.guardrails import PII
from dendrux.types import Clarification
from tests.conformance.harness import Cell, calls, text
from tests.conformance.scenarios import (
    QuestionStrategy,
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
