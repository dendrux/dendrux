"""Exercise the persisted phase producers used by the exhaustive pure table."""

from functools import wraps

import pytest

from dendrux import tool
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import AgentStep, Clarification, LLMResponse, RunStatus, ToolCall
from tests._fixtures.storage import pin_storage_format
from tests._helpers.h0_phases import LOOP_PHASES, PHASE_ACTIONS
from tests._helpers.runtime_paths import client_read, make_agent

pytestmark = pytest.mark.parametrize("engine", ["sqlite"], indirect=True)


@tool(idempotent=True)
async def effect() -> str:
    """Return a deterministic server observation."""
    return "ok"


class AskInput(NativeToolCalling):
    def parse_response(self, response):
        return AgentStep(reasoning=None, action=Clarification(question="Which value?"))


@pytest.mark.parametrize(
    "case,loop_phase,call_phase",
    [
        ("server", "dispatch", "dispatchable"),
        ("client", "awaiting_client", "client"),
        ("approval", "awaiting_approval", "gated"),
        ("approve", "dispatch", "approved"),
        ("reject", "model_next", "rejected"),
        ("malformed", "dispatch", "invalid"),
        ("unknown_tool", "dispatch", "invalid"),
        ("denied", "dispatch", "invalid"),
        ("input", "awaiting_input", None),
        ("final", "finalizing", None),
    ],
)
async def test_phase_writers_match_the_recovery_inventory(
    store, engine, monkeypatch, case, loop_phase, call_phase
):
    await pin_storage_format(engine, 2)
    states = []
    consumed_phases = []

    def capture(method):
        @wraps(method)
        async def recorded(run_id, *args, **kwargs):
            result = await method(run_id, *args, **kwargs)
            states.append(await store.get_loop_state(run_id))
            if method.__name__ == "consume_submission":
                consumed_phases.extend(kwargs["patch"].set_call_phase.values())
            return result

        return recorded

    monkeypatch.setattr(store, "checkpoint_step", capture(store.checkpoint_step))
    monkeypatch.setattr(store, "consume_submission", capture(store.consume_submission))
    call = ToolCall(
        "client_read" if case == "client" else "missing" if case == "unknown_tool" else "effect",
        parse_error="invalid_arguments" if case == "malformed" else None,
    )
    response = (
        LLMResponse(text="done") if case in {"final", "input"} else LLMResponse(tool_calls=[call])
    )
    gated = case in {"approval", "approve", "reject", "malformed"}
    agent = make_agent(
        store,
        [response, LLMResponse(text="done")],
        tools=[client_read, effect],
        require_approval=["effect"] if gated else [],
        deny=["effect"] if case == "denied" else [],
    )
    result = await agent.run(
        "go", tenant_id="tenant", strategy=AskInput() if case == "input" else NativeToolCalling()
    )
    if case in {"approve", "reject"}:
        assert result.status == RunStatus.WAITING_APPROVAL
        result = await agent.submit_approval(
            result.run_id,
            approved=case == "approve",
            tenant_id="tenant",
            expected_pause_id=result.pause_id,
        )
    states.append(await store.get_loop_state(result.run_id))
    assert any(state["phase"] == loop_phase for state in states)
    if call_phase is not None:
        assert call_phase in consumed_phases or any(
            pending["call_id"] == call.id and pending["phase"] == call_phase
            for state in states
            for pending in state.get("pending", [])
        )
    if case == "final":
        assert any(state["phase"] == "model_next" for state in states)
    assert all(state["phase"] in LOOP_PHASES for state in states)
    assert all(
        pending["phase"] in PHASE_ACTIONS
        for state in states
        for pending in state.get("pending", [])
    )
