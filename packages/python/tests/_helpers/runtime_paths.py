"""Public runtime entry adapters shared by the RT and H0 safety nets."""

from dataclasses import dataclass, field

from dendrux import Agent, tool
from dendrux.llm.base import LLMProvider
from dendrux.types import LLMResponse, ToolResult
from tests._helpers.paused_runs import _pause
from tests._helpers.recovery_checkpoint import checkpoint
from tests._helpers.scripted_provider import ScriptedLLM, tool_call

SUBMIT_KINDS = ["tool_results", "input", "approve", "reject", "decisions", "submit_resume"]
PATHS = ["run", "run_stream", "resume", "resume_stream", "retry", "recovery", *SUBMIT_KINDS]


@tool(target="client")
async def client_read() -> str:
    """Read a value on the client."""
    return "client"


def make_agent(store, responses, **kwargs):
    return Agent(
        name="cross-path",
        prompt="test",
        provider=responses if isinstance(responses, LLMProvider) else ScriptedLLM(responses),
        state_store=store,
        strict_tenancy=kwargs.pop("strict_tenancy", False),
        **kwargs,
    )


def client_result(paused):
    return ToolResult(name="client_read", call_id=paused.pending.tool_calls[0].id, payload="ok")


async def submission_case(store, kind, *, before_tool=None, provider=None):
    from dendrux.types import ApprovalDecision

    paused = await _pause(
        "client"
        if kind == "tool_results"
        else "input"
        if kind == "input"
        else "paused"
        if kind == "submit_resume"
        else "approval",
        store,
        "tenant",
        before_tool=before_tool,
    )
    a, run_id = paused.agent, paused.run_id
    if provider is not None:
        a = Agent(
            name=a.name,
            prompt=a.prompt,
            tools=a.tools,
            require_approval=a.require_approval,
            before_tool=before_tool,
            provider=provider,
            state_store=store,
            strict_tenancy=False,
        )
    raw = await store.get_pause_state(run_id)
    scope = dict(tenant_id="tenant", expected_pause_id=raw["pause_id"])

    async def invoke(agent=a, tenant="tenant", notifier=None):
        call_scope = {**scope, "tenant_id": tenant, "notifier": notifier}
        if kind == "tool_results":
            return await agent.submit_tool_results(
                run_id,
                [
                    ToolResult(
                        name="read_range",
                        call_id=raw["pending_tool_calls"][0]["id"],
                        payload='"ok"',
                    )
                ],
                **call_scope,
            )
        if kind == "input":
            return await agent.submit_input(run_id, "answer", **call_scope)
        if kind == "submit_resume":
            return await agent.submit_resume(run_id, **call_scope)
        if kind == "decisions":
            return await agent.submit_approval(
                run_id,
                decisions={c["id"]: ApprovalDecision.approve() for c in raw["pending_tool_calls"]},
                **call_scope,
            )
        return await agent.submit_approval(run_id, approved=kind == "approve", **call_scope)

    return a, run_id, invoke


@dataclass
class Entry:
    agent: Agent
    run_id: str | None = None
    events: list = field(default_factory=list)
    invoke: object = None


async def path_case(store, engine, path, provider, *, before_tool=None):
    if path in SUBMIT_KINDS:
        a, run_id, submit = await submission_case(
            store, path, before_tool=before_tool, provider=provider
        )
        case = Entry(a, run_id)
        case.invoke = submit
        return case

    a = make_agent(store, provider, tools=[client_read], before_tool=before_tool)
    case = Entry(a)
    paused = None
    if path in {"resume", "resume_stream"}:
        seeder = make_agent(
            store, [tool_call("client_read", pid="c")], tools=[client_read], before_tool=before_tool
        )
        paused = await seeder.run("go", tenant_id="tenant")
        case.run_id = paused.run_id
    elif path == "retry":
        original = await make_agent(store, [LLMResponse(text="prior")]).run(
            "go", tenant_id="tenant"
        )
        case.run_id = original.run_id
    elif path == "recovery":
        item = await checkpoint(a, store, engine)
        case.run_id = item.run_id

    async def invoke(*, tenant="tenant"):
        if path == "run":
            return await a.run("go", tenant_id=tenant)
        if path == "run_stream":
            stream = a.stream("go", tenant_id=tenant)
        elif path == "resume_stream":
            stream = a.resume_stream(
                paused.run_id,
                tool_results=[client_result(paused)],
                tenant_id=tenant,
                expected_pause_id=paused.pause_id,
            )
        elif path == "resume":
            return await a.resume(
                paused.run_id,
                tool_results=[client_result(paused)],
                tenant_id=tenant,
                expected_pause_id=paused.pause_id,
            )
        elif path == "retry":
            return await a.retry(case.run_id, tenant_id=tenant)
        elif path == "recovery":
            return await a.reclaim(case.run_id, item=item, tenant_id=tenant)
        async with stream:
            async for event in stream:
                case.events.append(event)
        return stream.result

    case.invoke = invoke
    return case


class ProbeProvider(ScriptedLLM):
    def __init__(self, *, block=False, error=None):
        super().__init__([LLMResponse(text="done")], gate_at=0 if block else None)
        self.run_id = None
        self.error = error

    async def complete(self, *args, **kwargs):
        self.run_id = kwargs.get("run_id")
        if self.error:
            raise self.error
        return await super().complete(*args, **kwargs)
