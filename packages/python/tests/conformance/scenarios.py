"""Scenario flows pinning 0.5.0 behaviour on every public entry path.

Each flow drives public entry points through a ``Ctx`` and asserts nothing: the recording
(``harness.run_cell``) is compared with the committed golden and corpus. Scripts are
deterministic (scripted providers, fixed tools). Cell ids are ``<scenario>__<flow>``.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

from dendrux import Agent, ApprovalDecision, ContextBlock, Image, SingleCall, ToolOutput
from dendrux.chat import ChatMessage
from dendrux.errors import ProviderRateLimitError
from dendrux.guardrails import PII
from dendrux.strategies.native import NativeToolCalling
from dendrux.tool import tool
from dendrux.types import AgentStep, Budget, Clarification, ToolCall, ToolResult
from tests.conformance.harness import Cell, Ctx, calls, text, usage

IMAGES = Path(__file__).resolve().parents[1] / "fixtures" / "images"
PNG = (IMAGES / "zebra42.png").read_bytes()
PNG2 = (IMAGES / "kiwi77.png").read_bytes()

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool()
async def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool()
async def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool(target="client")
async def read_range(sheet: str) -> str:
    """Read a range on the client."""
    return ""


@tool(target="client")
async def read_cell(cell: str) -> str:
    """Read a cell on the client."""
    return ""


@tool()
async def refund(order_id: int) -> str:
    """Refund an order (needs approval)."""
    return f"refunded {order_id}"


@tool()
async def explode() -> str:
    """Always fails."""
    raise RuntimeError("scripted failure")


@tool()
async def email_to(address: str) -> str:
    """Send mail to an address."""
    return f"mail sent to {address}"


@tool()
async def screenshot(n: int) -> ToolOutput:
    """Server screenshot."""
    return ToolOutput(f"shot {n}", images=[Image.from_bytes(PNG if n % 2 else PNG2)])


class QuestionStrategy(NativeToolCalling):
    """Native tool calling that turns ``QUESTION: ...`` model text into a clarification pause."""

    def parse_response(self, response: Any) -> AgentStep:
        body = response.text or ""
        if not response.tool_calls and body.startswith("QUESTION:"):
            return AgentStep(
                reasoning=None,
                action=Clarification(question=body[len("QUESTION:") :].strip()),
                raw_response=body,
            )
        return super().parse_response(response)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CELLS: list[Cell] = []


def cell(id_: str, scenario: str, *paths: str, tags: tuple[str, ...] = ()) -> Any:
    def register(flow: Any) -> Any:
        CELLS.append(Cell(id=id_, scenario=scenario, flow=flow, paths=paths, tags=tags))
        return flow

    return register


def call_ids(result: Any) -> list[str]:
    assert result is not None and result.pending is not None, "expected a paused result"
    return [c.id for c in result.pending.tool_calls]


def client_result(name: str, call_id: str, payload: str = '"ok"', **kw: Any) -> ToolResult:
    return ToolResult(name=name, call_id=call_id, payload=payload, **kw)


# ---------------------------------------------------------------------------
# Plain answer, server tool, parallel tools, tool error, max iterations
# ---------------------------------------------------------------------------


@cell("plain_answer__run", "plain_answer", "run")
async def _plain_run(ctx: Ctx) -> None:
    await ctx.run(ctx.agent(ctx.provider(text("Hello there."))), "Say hello")


@cell("plain_answer__stream", "plain_answer", "stream")
async def _plain_stream(ctx: Ctx) -> None:
    await ctx.stream(ctx.agent(ctx.provider(text("Hello there."))), "Say hello")


def _server_tool_script() -> list[Any]:
    return [calls(("add", {"a": 2, "b": 3}, "t1"), body="Adding."), text("The sum is 5.")]


@cell("server_tool__run", "server_tool", "run")
async def _server_run(ctx: Ctx) -> None:
    await ctx.run(ctx.agent(ctx.provider(*_server_tool_script()), tools=[add]), "2+3?")


@cell("server_tool__stream", "server_tool", "stream")
async def _server_stream(ctx: Ctx) -> None:
    await ctx.stream(ctx.agent(ctx.provider(*_server_tool_script()), tools=[add]), "2+3?")


def _parallel_script() -> list[Any]:
    return [
        calls(("add", {"a": 1, "b": 2}, "t1"), ("multiply", {"a": 3, "b": 4}, "t2")),
        text("3 and 12."),
    ]


@cell("parallel_tools__run", "parallel_tools", "run")
async def _parallel_run(ctx: Ctx) -> None:
    agent = ctx.agent(ctx.provider(*_parallel_script()), tools=[add, multiply])
    await ctx.run(agent, "do both")


@cell("parallel_tools__stream", "parallel_tools", "stream")
async def _parallel_stream(ctx: Ctx) -> None:
    agent = ctx.agent(ctx.provider(*_parallel_script()), tools=[add, multiply])
    await ctx.stream(agent, "do both")


@cell("tool_error__run", "tool_error", "run")
async def _tool_error(ctx: Ctx) -> None:
    agent = ctx.agent(
        ctx.provider(calls(("explode", {}, "t1")), text("It failed, moving on.")), tools=[explode]
    )
    await ctx.run(agent, "try it")


def _spin() -> list[Any]:
    return [
        calls(("add", {"a": 1, "b": 1}, "t1")),
        calls(("add", {"a": 2, "b": 2}, "t2")),
        text("never reached"),
    ]


@cell("max_iterations__run", "max_iterations", "run")
async def _max_run(ctx: Ctx) -> None:
    await ctx.run(ctx.agent(ctx.provider(*_spin()), tools=[add], max_iterations=2), "loop")


@cell("max_iterations__stream", "max_iterations", "stream")
async def _max_stream(ctx: Ctx) -> None:
    await ctx.stream(ctx.agent(ctx.provider(*_spin()), tools=[add], max_iterations=2), "loop")


# ---------------------------------------------------------------------------
# Malformed tool arguments
# ---------------------------------------------------------------------------


@cell("malformed_tool_args__run", "malformed_tool_args", "run")
async def _malformed_args(ctx: Ctx) -> None:
    script = [
        calls(("add", {"a": "x"}, "t1"), ("nope", {}, "t2")),
        text("recovered"),
    ]
    await ctx.run(ctx.agent(ctx.provider(*script), tools=[add]), "bad args")


@cell("malformed_tool_args__stream", "malformed_tool_args", "stream")
async def _malformed_args_stream(ctx: Ctx) -> None:
    script = [calls(("add", {"a": 1}, "t1")), text("recovered")]
    await ctx.stream(ctx.agent(ctx.provider(*script), tools=[add]), "bad args")


@cell("malformed_tool_json__run", "malformed_tool_json", "run")
async def _malformed_json(ctx: Ctx) -> None:
    bad = calls(("add", {}, "t1"))
    bad.tool_calls = [
        ToolCall(
            name="add",
            params={},
            provider_tool_call_id="t1",
            raw_arguments='{"a": 1,',
            parse_error="arguments must be valid JSON",
        )
    ]
    await ctx.run(ctx.agent(ctx.provider(bad, text("recovered")), tools=[add]), "bad json")


# ---------------------------------------------------------------------------
# Client-tool pause and every continuation path
# ---------------------------------------------------------------------------


def _client_script(second: str = "The sheet has 3 rows.") -> list[Any]:
    return [calls(("read_range", {"sheet": "S1"}, "t1"), body="Reading."), text(second)]


async def _paused_client(ctx: Ctx, start: str, **run_kwargs: Any) -> tuple[Any, Any]:
    agent = ctx.agent(ctx.provider(*_client_script()), tools=[read_range])
    starter = ctx.run if start == "run" else ctx.stream
    paused = await starter(agent, "read the sheet", **run_kwargs)
    return agent, paused


@cell("client_tool__run_resume", "client_tool", "run", "resume")
async def _client_run_resume(ctx: Ctx) -> None:
    agent, paused = await _paused_client(ctx, "run")
    await ctx.resume(
        agent,
        paused.run_id,
        tool_results=[client_result("read_range", call_ids(paused)[0], '"3 rows"')],
        expected_pause_id=paused.pause_id,
    )


@cell("client_tool__run_resume_stream", "client_tool", "run", "resume_stream")
async def _client_run_resume_stream(ctx: Ctx) -> None:
    agent, paused = await _paused_client(ctx, "run")
    await ctx.resume_stream(
        agent,
        paused.run_id,
        tool_results=[client_result("read_range", call_ids(paused)[0], '"3 rows"')],
        expected_pause_id=paused.pause_id,
    )


@cell("client_tool__run_submit_tool_results", "client_tool", "run", "submit_tool_results")
async def _client_run_submit(ctx: Ctx) -> None:
    agent, paused = await _paused_client(ctx, "run")
    await ctx.submit_tool_results(
        agent,
        paused.run_id,
        [client_result("read_range", call_ids(paused)[0], '"3 rows"')],
        expected_pause_id=paused.pause_id,
    )


@cell("client_tool__stream_resume", "client_tool", "stream", "resume")
async def _client_stream_resume(ctx: Ctx) -> None:
    agent, paused = await _paused_client(ctx, "stream")
    await ctx.resume(
        agent,
        paused.run_id,
        tool_results=[client_result("read_range", call_ids(paused)[0], '"3 rows"')],
        expected_pause_id=paused.pause_id,
    )


@cell("client_tool__stream_resume_stream", "client_tool", "stream", "resume_stream")
async def _client_stream_resume_stream(ctx: Ctx) -> None:
    agent, paused = await _paused_client(ctx, "stream")
    await ctx.resume_stream(
        agent,
        paused.run_id,
        tool_results=[client_result("read_range", call_ids(paused)[0], '"3 rows"')],
        expected_pause_id=paused.pause_id,
    )


@cell("client_tool__stream_submit_tool_results", "client_tool", "stream", "submit_tool_results")
async def _client_stream_submit(ctx: Ctx) -> None:
    agent, paused = await _paused_client(ctx, "stream")
    await ctx.submit_tool_results(
        agent,
        paused.run_id,
        [client_result("read_range", call_ids(paused)[0], '"3 rows"')],
        expected_pause_id=paused.pause_id,
    )


@cell("client_tool__invalid_then_valid_submit", "client_tool", "run", "submit_tool_results")
async def _client_invalid_then_valid(ctx: Ctx) -> None:
    agent, paused = await _paused_client(ctx, "run")
    await ctx.submit_tool_results(
        agent,
        paused.run_id,
        [client_result("read_range", "wrong-call-id")],
        "unknown call id",
        expected_pause_id=paused.pause_id,
    )
    await ctx.submit_tool_results(
        agent,
        paused.run_id,
        [client_result("read_range", call_ids(paused)[0])],
        "valid",
        expected_pause_id=paused.pause_id,
    )
    await ctx.submit_tool_results(
        agent,
        paused.run_id,
        [client_result("read_range", call_ids(paused)[0])],
        "already finished",
    )


@cell("client_tool__failed_result", "client_tool", "run", "submit_tool_results")
async def _client_failed_result(ctx: Ctx) -> None:
    agent, paused = await _paused_client(ctx, "run")
    await ctx.submit_tool_results(
        agent,
        paused.run_id,
        [client_result("read_range", call_ids(paused)[0], '"sheet missing"', success=False)],
        expected_pause_id=paused.pause_id,
    )


@cell("client_tool__two_pauses", "client_tool", "run", "resume", "submit_tool_results")
async def _client_two_pauses(ctx: Ctx) -> None:
    script = [
        calls(("read_range", {"sheet": "S1"}, "t1")),
        calls(("read_range", {"sheet": "S2"}, "t2")),
        text("Both sheets read."),
    ]
    agent = ctx.agent(ctx.provider(*script), tools=[read_range])
    first = await ctx.run(agent, "read both")
    second = await ctx.resume(
        agent,
        first.run_id,
        tool_results=[client_result("read_range", call_ids(first)[0], '"a"')],
        expected_pause_id=first.pause_id,
    )
    await ctx.submit_tool_results(
        agent,
        second.run_id,
        [client_result("read_range", call_ids(second)[0], '"b"')],
        expected_pause_id=second.pause_id,
    )


@cell("client_tool__mixed_server_and_client", "client_tool", "run", "resume")
async def _client_mixed(ctx: Ctx) -> None:
    script = [
        calls(("add", {"a": 1, "b": 2}, "t1"), ("read_range", {"sheet": "S1"}, "t2")),
        text("Done with both."),
    ]
    agent = ctx.agent(ctx.provider(*script), tools=[add, read_range])
    paused = await ctx.run(agent, "mix")
    await ctx.resume(
        agent,
        paused.run_id,
        tool_results=[client_result("read_range", call_ids(paused)[0], '"x"')],
        expected_pause_id=paused.pause_id,
    )


@cell("client_tool__parallel_client_calls", "client_tool", "run", "submit_tool_results")
async def _client_parallel(ctx: Ctx) -> None:
    script = [
        calls(("read_range", {"sheet": "S1"}, "t1"), ("read_cell", {"cell": "A1"}, "t2")),
        text("Read both."),
    ]
    agent = ctx.agent(ctx.provider(*script), tools=[read_range, read_cell])
    paused = await ctx.run(agent, "read")
    names = {c.id: c.name for c in paused.pending.tool_calls}
    await ctx.submit_tool_results(
        agent,
        paused.run_id,
        [client_result(name, cid, f'"{name}"') for cid, name in names.items()],
        expected_pause_id=paused.pause_id,
    )


# ---------------------------------------------------------------------------
# Governance changed between the pause and the continuation
# ---------------------------------------------------------------------------


async def _gov_paused(ctx: Ctx) -> tuple[Agent, Any]:
    agent = ctx.agent(ctx.provider(*_client_script("Resumed.")), tools=[read_range, add])
    return agent, await ctx.run(agent, "read the sheet")


def _gov_changed(ctx: Ctx, mode: str) -> Agent:
    return ctx.agent(
        ctx.provider(text("Resumed under new rules.")),
        tools=[read_range, add],
        deny=["add"],
        on_governance_change=mode,
    )


@cell("governance__changed_warn", "governance", "run", "resume")
async def _gov_warn(ctx: Ctx) -> None:
    _, paused = await _gov_paused(ctx)
    await ctx.resume(
        _gov_changed(ctx, "warn"),
        paused.run_id,
        tool_results=[client_result("read_range", call_ids(paused)[0])],
        expected_pause_id=paused.pause_id,
    )


@cell("governance__changed_raise", "governance", "run", "resume", "submit_tool_results")
async def _gov_raise(ctx: Ctx) -> None:
    original, paused = await _gov_paused(ctx)
    results = [client_result("read_range", call_ids(paused)[0])]
    await ctx.resume(
        _gov_changed(ctx, "raise"),
        paused.run_id,
        "changed governance, raise",
        tool_results=results,
        expected_pause_id=paused.pause_id,
    )
    await ctx.submit_tool_results(
        original,
        paused.run_id,
        results,
        "original governance",
        expected_pause_id=paused.pause_id,
    )


# ---------------------------------------------------------------------------
# Human input
# ---------------------------------------------------------------------------


async def _asked(ctx: Ctx, start: str) -> tuple[Any, Any]:
    agent = ctx.agent(
        ctx.provider(text("QUESTION: Which file?"), text("Reading report.csv.")),
    )
    starter = ctx.run if start == "run" else ctx.stream
    return agent, await starter(agent, "Analyse something", strategy=QuestionStrategy())


@cell("human_input__run_submit_input", "human_input", "run", "submit_input")
async def _human_submit(ctx: Ctx) -> None:
    agent, paused = await _asked(ctx, "run")
    await ctx.submit_input(agent, paused.run_id, "report.csv", expected_pause_id=paused.pause_id)


@cell("human_input__run_resume", "human_input", "run", "resume")
async def _human_resume(ctx: Ctx) -> None:
    agent, paused = await _asked(ctx, "run")
    await ctx.resume(
        agent, paused.run_id, user_input="report.csv", expected_pause_id=paused.pause_id
    )


@cell("human_input__stream_resume_stream", "human_input", "stream", "resume_stream")
async def _human_stream(ctx: Ctx) -> None:
    agent, paused = await _asked(ctx, "stream")
    await ctx.resume_stream(
        agent, paused.run_id, user_input="report.csv", expected_pause_id=paused.pause_id
    )


@cell("human_input__wrong_submit_kind", "human_input", "run", "submit_tool_results", "submit_input")
async def _human_wrong_kind(ctx: Ctx) -> None:
    agent, paused = await _asked(ctx, "run")
    await ctx.submit_tool_results(
        agent, paused.run_id, [client_result("read_range", "x")], "wrong kind"
    )
    await ctx.submit_input(agent, paused.run_id, "report.csv", "right kind")


# ---------------------------------------------------------------------------
# Approval pause: approve, reject, per-call decisions
# ---------------------------------------------------------------------------


def _approval_agent(ctx: Ctx, *responses: Any) -> Agent:
    return ctx.agent(ctx.provider(*responses), tools=[refund], require_approval=["refund"])


async def _approval_paused(ctx: Ctx, start: str = "run") -> tuple[Agent, Any]:
    agent = _approval_agent(
        ctx, calls(("refund", {"order_id": 7}, "t1")), text("Refund processed.")
    )
    starter = ctx.run if start == "run" else ctx.stream
    return agent, await starter(agent, "refund order 7")


@cell("approval__submit_approve", "approval", "run", "submit_approval")
async def _approval_approve(ctx: Ctx) -> None:
    agent, paused = await _approval_paused(ctx)
    await ctx.submit_approval(
        agent, paused.run_id, approved=True, expected_pause_id=paused.pause_id
    )


@cell("approval__submit_reject", "approval", "run", "submit_approval")
async def _approval_reject(ctx: Ctx) -> None:
    agent, paused = await _approval_paused(ctx)
    await ctx.submit_approval(
        agent,
        paused.run_id,
        approved=False,
        rejection_reason="Not authorised.",
        expected_pause_id=paused.pause_id,
    )


@cell("approval__stream_submit_approve", "approval", "stream", "submit_approval")
async def _approval_stream_approve(ctx: Ctx) -> None:
    agent, paused = await _approval_paused(ctx, "stream")
    await ctx.submit_approval(
        agent, paused.run_id, approved=True, expected_pause_id=paused.pause_id
    )


@cell("approval__run_resume_approves", "approval", "run", "resume")
async def _approval_resume(ctx: Ctx) -> None:
    agent, paused = await _approval_paused(ctx)
    await ctx.resume(agent, paused.run_id, expected_pause_id=paused.pause_id)


@cell("approval__run_resume_stream_approves", "approval", "run", "resume_stream")
async def _approval_resume_stream(ctx: Ctx) -> None:
    agent, paused = await _approval_paused(ctx)
    await ctx.resume_stream(agent, paused.run_id, expected_pause_id=paused.pause_id)


@cell("approval__decisions_mixed", "approval", "run", "submit_approval")
async def _approval_decisions(ctx: Ctx) -> None:
    agent = _approval_agent(
        ctx,
        calls(("refund", {"order_id": 1}, "t1"), ("refund", {"order_id": 2}, "t2")),
        text("One refunded, one declined."),
    )
    paused = await ctx.run(agent, "refund both")
    ids = call_ids(paused)
    await ctx.submit_approval(
        agent,
        paused.run_id,
        decisions={ids[0]: ApprovalDecision.approve(), ids[1]: ApprovalDecision.deny("too large")},
        expected_pause_id=paused.pause_id,
    )


@cell("approval__stale_pause_id", "approval", "run", "submit_approval")
async def _approval_stale(ctx: Ctx) -> None:
    agent, paused = await _approval_paused(ctx)
    await ctx.submit_approval(
        agent,
        paused.run_id,
        approved=True,
        label="stale pause id",
        expected_pause_id="not-the-pause",
    )
    await ctx.submit_approval(
        agent, paused.run_id, approved=True, expected_pause_id=paused.pause_id
    )


# ---------------------------------------------------------------------------
# Budget pause with an unpriced model
# ---------------------------------------------------------------------------


def _budget_agent(ctx: Ctx, *responses: Any) -> Agent:
    return ctx.agent(
        ctx.provider(*responses, reports_cost=True),
        tools=[add],
        budget=Budget(max_cost_usd=5.0, on_exceed="pause", count_cache_tokens=False),
    )


@cell("budget_unpriced__run_resume", "budget_unpriced", "run", "resume")
async def _budget_run_resume(ctx: Ctx) -> None:
    first = _budget_agent(ctx, calls(("add", {"a": 1, "b": 1}, "t1"), cost=None))
    paused = await ctx.run(first, "work")
    second = _budget_agent(ctx, text("Finished.", cost=0.01))
    await ctx.resume(second, paused.run_id)


@cell(
    "budget_unpriced__stream_resume_stream_repauses", "budget_unpriced", "stream", "resume_stream"
)
async def _budget_stream_repause(ctx: Ctx) -> None:
    first = _budget_agent(ctx, calls(("add", {"a": 1, "b": 1}, "t1"), cost=None))
    paused = await ctx.stream(first, "work")
    again = _budget_agent(ctx, calls(("add", {"a": 2, "b": 2}, "t2"), cost=None), text("never"))
    await ctx.resume_stream(again, paused.run_id, "unpriced again")


@cell("budget_unpriced__cancel_on_exceed", "budget_unpriced", "run")
async def _budget_cancel(ctx: Ctx) -> None:
    agent = ctx.agent(
        ctx.provider(
            calls(("add", {"a": 1, "b": 1}, "t1"), cost=None), text("x"), reports_cost=True
        ),
        tools=[add],
        budget=Budget(max_cost_usd=5.0, on_exceed="cancel"),
    )
    await ctx.run(agent, "work")


# ---------------------------------------------------------------------------
# Guardrails: block, redact + restore
# ---------------------------------------------------------------------------


@cell("guardrail_block__run", "guardrail_block", "run")
async def _guardrail_block(ctx: Ctx) -> None:
    agent = ctx.agent(ctx.provider(text("unused")), guardrails=[PII(action="block")])
    await ctx.run(agent, "Mail jane.doe@example.com the report")


@cell("guardrail_block__stream_rejects_guardrails", "guardrail_block", "stream")
async def _guardrail_stream(ctx: Ctx) -> None:
    agent = ctx.agent(ctx.provider(text("unused")), guardrails=[PII(action="block")])
    await ctx.stream(agent, "hello", "stream with guardrails")


@cell("pii_restore__run", "pii_restore", "run")
async def _pii_run(ctx: Ctx) -> None:
    script = [
        calls(("email_to", {"address": "<<EMAIL_ADDRESS_1>>"}, "t1")),
        text("Told <<EMAIL_ADDRESS_1>> it is done."),
    ]
    agent = ctx.agent(ctx.provider(*script), tools=[email_to], guardrails=[PII()])
    await ctx.run(agent, "Mail jane.doe@example.com the report")


@cell("pii_restore__client_tool_resume", "pii_restore", "run", "resume")
async def _pii_resume(ctx: Ctx) -> None:
    script = [
        calls(("read_range", {"sheet": "<<EMAIL_ADDRESS_1>>"}, "t1")),
        text("Looked up <<EMAIL_ADDRESS_1>>."),
    ]
    agent = ctx.agent(ctx.provider(*script), tools=[read_range], guardrails=[PII()])
    paused = await ctx.run(agent, "Find jane.doe@example.com")
    await ctx.resume(
        agent,
        paused.run_id,
        tool_results=[
            client_result("read_range", call_ids(paused)[0], '"found jane.doe@example.com"')
        ],
        expected_pause_id=paused.pause_id,
    )


# ---------------------------------------------------------------------------
# Delegation (parent and child)
# ---------------------------------------------------------------------------


def _delegation(ctx: Ctx) -> Agent:
    child = ctx.agent(ctx.provider(text("child says hi")), name="Child", prompt="You are a helper.")

    @tool()
    async def delegate(task: str) -> str:
        """Delegate a task to the helper agent."""
        result = await child.run(task)
        return result.answer or ""

    return ctx.agent(
        ctx.provider(calls(("delegate", {"task": "help me"}, "t1")), text("Parent done.")),
        tools=[delegate],
        name="Parent",
    )


@cell("delegation__run", "delegation", "run")
async def _delegation_run(ctx: Ctx) -> None:
    await ctx.run(_delegation(ctx), "go")


@cell("delegation__stream", "delegation", "stream")
async def _delegation_stream(ctx: Ctx) -> None:
    await ctx.stream(_delegation(ctx), "go")


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


@cell("images__run", "images", "run", tags=("blob_catalog",))
async def _images_run(ctx: Ctx) -> None:
    agent = ctx.agent(ctx.provider(text("A zebra.")))
    await ctx.run(agent, ["What is this?", Image.from_bytes(PNG)])


@cell("images__server_tool_image_stream", "images", "stream", tags=("blob_catalog",))
async def _images_tool_stream(ctx: Ctx) -> None:
    script = [calls(("screenshot", {"n": 1}, "t1")), text("I see the screenshot.")]
    await ctx.stream(ctx.agent(ctx.provider(*script), tools=[screenshot]), "shoot")


@cell(
    "images__client_result_resume", "images", "run", "submit_tool_results", tags=("blob_catalog",)
)
async def _images_client(ctx: Ctx) -> None:
    script = [calls(("read_range", {"sheet": "S1"}, "t1")), text("Saw the picture.")]
    agent = ctx.agent(ctx.provider(*script), tools=[read_range])
    paused = await ctx.run(agent, ["look", Image.from_bytes(PNG)])
    await ctx.submit_tool_results(
        agent,
        paused.run_id,
        [
            client_result(
                "read_range", call_ids(paused)[0], '"see image"', images=[Image.from_bytes(PNG2)]
            )
        ],
        expected_pause_id=paused.pause_id,
    )


# ---------------------------------------------------------------------------
# Seeded history, context blocks, compaction summary
# ---------------------------------------------------------------------------

_HISTORY = [
    ChatMessage.user("Hello"),
    ChatMessage.assistant("Hi, how can I help?"),
    ChatMessage.user("What is the capital of France?"),
    ChatMessage.assistant("Paris."),
]


@cell("seeded_history__run", "seeded_history", "run")
async def _history_run(ctx: Ctx) -> None:
    await ctx.run(
        ctx.agent(ctx.provider(text("About 2.1 million."))), "And its population?", history=_HISTORY
    )


@cell("seeded_history__stream", "seeded_history", "stream")
async def _history_stream(ctx: Ctx) -> None:
    await ctx.stream(
        ctx.agent(ctx.provider(text("About 2.1 million."))), "And its population?", history=_HISTORY
    )


@cell("seeded_history__client_tool_resume", "seeded_history", "run", "resume")
async def _history_resume(ctx: Ctx) -> None:
    agent = ctx.agent(ctx.provider(*_client_script("Done.")), tools=[read_range])
    paused = await ctx.run(
        agent,
        "read the sheet",
        history=_HISTORY,
        context=[
            ContextBlock("project spec v2", kind="spec", placement="stable"),
            ContextBlock("retrieved doc", kind="doc"),
        ],
    )
    await ctx.resume(
        agent,
        paused.run_id,
        tool_results=[client_result("read_range", call_ids(paused)[0])],
        expected_pause_id=paused.pause_id,
    )


_CONTEXT = [
    ContextBlock("project spec v2", kind="spec", placement="stable"),
    ContextBlock("retrieved doc about Paris", kind="doc", placement="dynamic"),
]


@cell("context_blocks__run", "context_blocks", "run")
async def _context_run(ctx: Ctx) -> None:
    await ctx.run(
        ctx.agent(ctx.provider(text("Paris."))), "Capital?", history=_HISTORY[:2], context=_CONTEXT
    )


@cell("context_blocks__stream", "context_blocks", "stream")
async def _context_stream(ctx: Ctx) -> None:
    await ctx.stream(
        ctx.agent(ctx.provider(text("Paris."))), "Capital?", history=_HISTORY[:2], context=_CONTEXT
    )


@cell("context_blocks__no_history", "context_blocks", "run")
async def _context_no_history(ctx: Ctx) -> None:
    await ctx.run(ctx.agent(ctx.provider(text("Paris."))), "Capital?", context=_CONTEXT)


@cell("compaction_summary__run", "compaction_summary", "run")
async def _compaction_run(ctx: Ctx) -> None:
    """The caller-side compaction recipe: old turns replaced by a summary context block."""
    summary = ContextBlock(
        "Summary of the earlier conversation: greeting, then the capital of France was asked "
        "and answered (Paris).",
        kind="conversation_summary",
        placement="stable",
    )
    await ctx.run(
        ctx.agent(ctx.provider(text("About 2.1 million."))),
        "And its population?",
        history=_HISTORY[-2:],
        context=[summary],
    )


# ---------------------------------------------------------------------------
# Replay and retry
# ---------------------------------------------------------------------------


@cell("replay__run_with_runs", "replay", "run")
async def _replay(ctx: Ctx) -> None:
    first = await ctx.run(
        ctx.agent(ctx.provider(*_server_tool_script()), tools=[add]), "2+3?", "source with a tool"
    )
    second = await ctx.run(ctx.agent(ctx.provider(text("Hello."))), "Say hello", "plain source")
    await ctx.run(
        ctx.agent(ctx.provider(text("5 and a greeting."))),
        "Recap",
        "replay both, last one in full",
        runs=[first.run_id, second.run_id],
        replay_last=1,
    )


@cell("replay__stream_with_runs", "replay", "run", "stream")
async def _replay_stream(ctx: Ctx) -> None:
    first = await ctx.run(ctx.agent(ctx.provider(text("Paris."))), "Capital of France?")
    await ctx.stream(
        ctx.agent(ctx.provider(text("Still Paris."))),
        "Again?",
        runs=[first.run_id],
        replay_last="all",
    )


@cell("retry__after_max_iterations", "retry", "run", "retry")
async def _retry_max(ctx: Ctx) -> None:
    first = await ctx.run(ctx.agent(ctx.provider(*_spin()), tools=[add], max_iterations=2), "loop")
    await ctx.retry(ctx.agent(ctx.provider(text("Now I can answer.")), tools=[add]), first.run_id)


@cell("retry__after_provider_error", "retry", "stream", "retry")
async def _retry_error(ctx: Ctx) -> None:
    provider = ctx.provider(calls(("add", {"a": 1, "b": 2}, "t1")), text("x"), hook=_fail_second)
    first = await ctx.stream(ctx.agent(provider, tools=[add]), "work")
    await ctx.retry(ctx.agent(ctx.provider(text("Recovered.")), tools=[add]), first.run_id)


@cell("retry__after_success", "retry", "run", "retry")
async def _retry_success(ctx: Ctx) -> None:
    first = await ctx.run(ctx.agent(ctx.provider(*_server_tool_script()), tools=[add]), "2+3?")
    await ctx.retry(ctx.agent(ctx.provider(text("Same again."))), first.run_id)


# ---------------------------------------------------------------------------
# Provider failure
# ---------------------------------------------------------------------------


async def _fail_second(index: int, kwargs: dict[str, Any]) -> None:
    if index == 1:
        raise ProviderRateLimitError(
            provider="scripted",
            model="mock",
            status_code=429,
            provider_code="rate_limit",
            message="slow down",
        )


async def _fail_first(index: int, kwargs: dict[str, Any]) -> None:
    raise ProviderRateLimitError(
        provider="scripted",
        model="mock",
        status_code=429,
        provider_code="rate_limit",
        message="slow down",
    )


@cell("provider_error__run", "provider_error", "run")
async def _provider_error(ctx: Ctx) -> None:
    await ctx.run(ctx.agent(ctx.provider(text("x"), hook=_fail_first)), "go")


@cell("provider_error__stream", "provider_error", "stream")
async def _provider_error_stream(ctx: Ctx) -> None:
    await ctx.stream(ctx.agent(ctx.provider(text("x"), hook=_fail_first)), "go")


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


async def _cancel_requested(ctx: Ctx, index: int, kwargs: dict[str, Any]) -> None:
    if index == 0:
        await ctx.store.request_cancel(kwargs["run_id"])


def _cancel_hook(ctx: Ctx) -> Any:
    async def hook(index: int, kwargs: dict[str, Any]) -> None:
        await _cancel_requested(ctx, index, kwargs)

    return hook


@cell("cancellation__mid_run_run", "cancellation", "run")
async def _cancel_mid_run(ctx: Ctx) -> None:
    provider = ctx.provider(
        calls(("add", {"a": 1, "b": 1}, "t1")), text("never"), hook=_cancel_hook(ctx)
    )
    await ctx.run(ctx.agent(provider, tools=[add]), "work")


@cell("cancellation__mid_run_stream", "cancellation", "stream")
async def _cancel_mid_stream(ctx: Ctx) -> None:
    provider = ctx.provider(
        calls(("add", {"a": 1, "b": 1}, "t1")), text("never"), hook=_cancel_hook(ctx)
    )
    await ctx.stream(ctx.agent(provider, tools=[add]), "work")


@cell("cancellation__cancel_run_waiting_client_tool", "cancellation", "run", "cancel_run")
async def _cancel_waiting_client(ctx: Ctx) -> None:
    agent, paused = await _paused_client(ctx, "run")
    await ctx.cancel_run(agent, paused.run_id)
    await ctx.cancel_run(agent, paused.run_id, "second cancel is a no-op")


@cell("cancellation__cancel_run_waiting_approval", "cancellation", "run", "cancel_run")
async def _cancel_waiting_approval(ctx: Ctx) -> None:
    agent, paused = await _approval_paused(ctx)
    await ctx.cancel_run(agent, paused.run_id)


@cell("cancellation__cancel_run_waiting_human_input", "cancellation", "run", "cancel_run")
async def _cancel_waiting_input(ctx: Ctx) -> None:
    agent, paused = await _asked(ctx, "run")
    await ctx.cancel_run(agent, paused.run_id)


@cell("cancellation__cancel_run_terminal_and_missing", "cancellation", "run", "cancel_run")
async def _cancel_terminal(ctx: Ctx) -> None:
    agent = ctx.agent(ctx.provider(text("done")))
    done = await ctx.run(agent, "hi")
    await ctx.cancel_run(agent, done.run_id, "terminal run")
    await ctx.cancel_run(agent, "01ARZ3NDEKTSV4RRFFQ69G5FAV", "unknown run")


@cell("cancellation__cancel_run_cascades_to_child", "cancellation", "run", "cancel_run")
async def _cancel_cascade(ctx: Ctx) -> None:
    """A real parent waiting on a client tool whose delegated child also waits."""
    child = ctx.agent(
        ctx.provider(calls(("read_range", {"sheet": "child"}, "c1"))),
        tools=[read_range],
        name="Child",
        prompt="You are a helper.",
    )

    @tool()
    async def delegate(task: str) -> str:
        """Delegate a task to the helper agent."""
        result = await child.run(task)
        return f"helper is {result.status.value}"

    parent = ctx.agent(
        ctx.provider(
            calls(("delegate", {"task": "look"}, "t1"), ("read_range", {"sheet": "S1"}, "t2")),
            text("never reached"),
        ),
        tools=[delegate, read_range],
        name="Parent",
    )
    paused = await ctx.run(parent, "read both")
    records = {r.id: r for r in await ctx.store.list_runs(limit=10)}
    assert len(records) == 2, "a parent and its delegated child must both exist"
    (child_record,) = [r for r in records.values() if r.parent_run_id == paused.run_id]
    assert paused.status.value == "waiting_client_tool"
    assert child_record.status == "waiting_client_tool" and child_record.delegation_level == 1
    await ctx.cancel_run(parent, paused.run_id, cascade=True)
    after = {r.id: r.status for r in await ctx.store.list_runs(limit=10)}
    assert set(after.values()) == {"cancelled"}, after


@cell("cancellation__request_pause_then_resume", "cancellation", "run", "resume")
async def _pause_requested(ctx: Ctx) -> None:
    async def hook(index: int, kwargs: dict[str, Any]) -> None:
        if index == 0:
            await ctx.store.request_pause(kwargs["run_id"])

    provider = ctx.provider(calls(("add", {"a": 1, "b": 1}, "t1")), text("never"), hook=hook)
    paused = await ctx.run(ctx.agent(provider, tools=[add]), "work")
    await ctx.resume(ctx.agent(ctx.provider(text("Resumed and done.")), tools=[add]), paused.run_id)


# ---------------------------------------------------------------------------
# Inbox notes and submit_resume
# ---------------------------------------------------------------------------


def _pause_after_first_call(ctx: Ctx) -> Any:
    async def hook(index: int, kwargs: dict[str, Any]) -> None:
        if index == 0:
            await ctx.store.request_pause(kwargs["run_id"])

    return hook


@cell(
    "run_control__pause_notes_submit_resume",
    "run_control",
    "run",
    "submit_note",
    "submit_resume",
)
async def _notes_submit_resume(ctx: Ctx) -> None:
    provider = ctx.provider(
        calls(("add", {"a": 1, "b": 1}, "t1")), text("never"), hook=_pause_after_first_call(ctx)
    )
    paused = await ctx.run(ctx.agent(provider, tools=[add]), "work")
    await ctx.submit_note(
        ctx.agent(ctx.provider(text("unused"))),
        paused.run_id,
        "Please double check the total.",
        "first",
    )
    await ctx.submit_note(
        ctx.agent(ctx.provider(text("unused"))),
        paused.run_id,
        "Use metric units.",
        "second, with metadata",
        metadata={"ticket": 7},
    )
    resumer = ctx.agent(ctx.provider(text("Checked, in metric.")), tools=[add])
    await ctx.submit_resume(resumer, paused.run_id, expected_pause_id=paused.pause_id)
    await ctx.submit_note(resumer, paused.run_id, "Too late.", "note on a finished run")
    await ctx.submit_resume(resumer, paused.run_id, "resume a finished run")


@cell(
    "run_control__stream_pause_note_resume_stream",
    "run_control",
    "stream",
    "submit_note",
    "resume_stream",
)
async def _notes_resume_stream(ctx: Ctx) -> None:
    provider = ctx.provider(
        calls(("add", {"a": 2, "b": 2}, "t1")), text("never"), hook=_pause_after_first_call(ctx)
    )
    paused = await ctx.stream(ctx.agent(provider, tools=[add]), "work")
    await ctx.submit_note(
        ctx.agent(ctx.provider(text("unused"))), paused.run_id, "Note while paused."
    )
    resumer = ctx.agent(ctx.provider(text("Saw the note.")), tools=[add])
    await ctx.resume_stream(resumer, paused.run_id, expected_pause_id=paused.pause_id)


@cell(
    "run_control__note_on_client_pause",
    "run_control",
    "run",
    "submit_note",
    "submit_tool_results",
    "submit_resume",
)
async def _note_on_client_pause(ctx: Ctx) -> None:
    agent, paused = await _paused_client(ctx, "run")
    await ctx.submit_note(agent, paused.run_id, "Remember the header row.")
    await ctx.submit_resume(
        agent, paused.run_id, "not a plain pause", expected_pause_id=paused.pause_id
    )
    await ctx.submit_tool_results(
        agent,
        paused.run_id,
        [client_result("read_range", call_ids(paused)[0], '"3 rows"')],
        expected_pause_id=paused.pause_id,
    )


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


@cell(
    "sweep__stale_running_and_abandoned_waiting",
    "sweep",
    "run",
    "sweep",
    "cancel_run",
    "submit_approval",
)
async def _sweep(ctx: Ctx) -> None:
    agent, paused = await _paused_client(ctx, "run")
    approval_agent, approval = await _approval_paused(ctx)
    orphan = await ctx.orphan_running_run()
    fresh = await ctx.run(ctx.agent(ctx.provider(text("fresh"))), "stays untouched")
    await ctx.age_runs(paused.run_id, approval.run_id, orphan)
    await ctx.sweep(
        stale_running=timedelta(minutes=20),
        abandoned_waiting=timedelta(minutes=30),
    )
    await ctx.cancel_run(agent, paused.run_id, "after sweep")
    await ctx.submit_approval(
        approval_agent,
        approval.run_id,
        approved=True,
        label="after sweep",
        expected_pause_id=approval.pause_id,
    )
    assert fresh is not None


@cell("sweep__nothing_due", "sweep", "run", "sweep")
async def _sweep_nothing(ctx: Ctx) -> None:
    await ctx.run(ctx.agent(ctx.provider(text("done"))), "hi")
    await ctx.sweep(stale_running=timedelta(minutes=20), abandoned_waiting=timedelta(minutes=30))


# ---------------------------------------------------------------------------
# Idempotency, tenancy, SingleCall
# ---------------------------------------------------------------------------


@cell("idempotency__run_repeat", "idempotency", "run")
async def _idem_run(ctx: Ctx) -> None:
    agent = ctx.agent(ctx.provider(text("First and only.")))
    await ctx.run(agent, "do it", "first", idempotency_key="key-1")
    await ctx.run(agent, "do it", "repeat returns the stored result", idempotency_key="key-1")
    await ctx.run(agent, "something else", "same key different input", idempotency_key="key-1")


@cell("idempotency__stream_repeat", "idempotency", "stream")
async def _idem_stream(ctx: Ctx) -> None:
    agent = ctx.agent(ctx.provider(text("First and only.")))
    await ctx.stream(agent, "do it", "first", idempotency_key="key-2")
    await ctx.stream(agent, "do it", "repeat", idempotency_key="key-2")


@cell("tenancy__client_tool_submit", "tenancy", "run", "submit_tool_results", "cancel_run")
async def _tenancy(ctx: Ctx) -> None:
    agent = ctx.agent(ctx.provider(*_client_script()), tools=[read_range])
    paused = await ctx.run(agent, "read the sheet", tenant_id="acme")
    await ctx.submit_tool_results(
        agent,
        paused.run_id,
        [client_result("read_range", call_ids(paused)[0])],
        "wrong tenant",
        tenant_id="other",
        expected_pause_id=paused.pause_id,
    )
    await ctx.cancel_run(agent, paused.run_id, "wrong tenant cancel", tenant_id="other")
    await ctx.submit_tool_results(
        agent,
        paused.run_id,
        [client_result("read_range", call_ids(paused)[0])],
        "right tenant",
        tenant_id="acme",
        expected_pause_id=paused.pause_id,
    )


@cell("single_call__run", "single_call", "run")
async def _single_run(ctx: Ctx) -> None:
    await ctx.run(ctx.agent(ctx.provider(text("One shot.")), loop=SingleCall()), "Summarise")


@cell("single_call__stream", "single_call", "stream")
async def _single_stream(ctx: Ctx) -> None:
    await ctx.stream(ctx.agent(ctx.provider(text("One shot.")), loop=SingleCall()), "Summarise")


__all__ = ["CELLS", "usage"]
