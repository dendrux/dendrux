"""Optional hook compatibility and trusted MCP replay-safety hints."""

import pytest
from mcp.types import Tool, ToolAnnotations

from dendrux.loops._helpers import notify_run_started
from dendrux.loops.base import BaseNotifier
from dendrux.mcp import MCPSource
from dendrux.mcp._server import build_mcp_tool_defs


@pytest.mark.parametrize("modern", [False, True])
async def test_run_started_signature_gate(modern):
    seen = []

    class Legacy(BaseNotifier):
        async def on_run_started(self, run_id, *, agent_name=None, agent_model=None):
            seen.append("legacy")

    class Modern(BaseNotifier):
        async def on_run_started(self, run_id, **kwargs):
            seen.append(kwargs["trace_link"])

    await notify_run_started(Modern() if modern else Legacy(), "run", trace_link="link")
    assert seen == ["link" if modern else "legacy"]


@pytest.mark.parametrize("transport", ["stdio", "http"])
@pytest.mark.parametrize("hint", ["idempotent_hint", "read_only_hint", "neither", "missing"])
@pytest.mark.parametrize("trust", [None, False, True])
def test_idempotency_requires_trusted_source(transport, hint, trust):
    kwargs = {} if trust is None else {"trust_replay_hints": trust}
    source = (
        MCPSource.stdio("local", ["unused"], **kwargs)
        if transport == "stdio"
        else MCPSource.http("remote", "https://example.com/mcp", **kwargs)
    )
    annotations = (
        None
        if hint == "missing"
        else ToolAnnotations(
            **(
                {hint: True}
                if hint != "neither"
                else {"idempotent_hint": False, "read_only_hint": False}
            )
        )
    )
    definitions, skipped = build_mcp_tool_defs(
        source=source,
        namespace="n",
        tools=[Tool(name="work", input_schema={"type": "object"}, annotations=annotations)],
        connection_info=None,
    )
    assert skipped == []
    assert definitions[0].idempotent == (
        trust is True and hint in ("idempotent_hint", "read_only_hint")
    )


@pytest.mark.parametrize("behavior", ["missing", "raises", "object"])
def test_optional_link_hooks_are_best_effort(behavior):
    from dendrux.loops._helpers import notifier_span_link, notifier_trace_link

    class Custom:
        def trace_link(self, run_id):
            if behavior == "raises":
                raise ValueError("unavailable")
            return object()

        def current_span_link(self, run_id, tool_call_id):
            return self.trace_link(run_id)

    notifier = object() if behavior == "missing" else Custom()
    assert notifier_trace_link(notifier, "run") is None
    assert notifier_span_link(notifier, "run", "call") is None


def test_replay_event_inventory_hides_provider_chosen_call_ids():
    from dendrux.store._redact import _SCHEMAS, _apply

    assert _apply(
        _SCHEMAS["tool.started"],
        {"tool_name": "work", "call_id": "arbitrary-provider-text", "idempotent": True},
    ) == {"tool_name": "work", "idempotent": True}
    assert _apply(_SCHEMAS["tool.replayed"], {"call_id": "arbitrary-provider-text"}) == {}


@pytest.mark.parametrize("trust", [1, "true", None])
def test_replay_trust_requires_boolean(trust):
    with pytest.raises(ValueError):
        MCPSource(name="local", command=("unused",), trust_replay_hints=trust)


@pytest.mark.parametrize("hint", [1, "true", False, None])
def test_replay_hints_require_literal_true(hint):
    source = MCPSource.stdio("local", ["unused"], trust_replay_hints=True)
    annotations = ToolAnnotations.model_construct(idempotent_hint=hint, read_only_hint=False)
    definitions, _ = build_mcp_tool_defs(
        source=source,
        namespace="n",
        tools=[Tool(name="work", input_schema={"type": "object"}, annotations=annotations)],
        connection_info=None,
    )
    assert definitions[0].idempotent is False
