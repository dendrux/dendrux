"""SIGKILL at real format-2 Agent/store boundaries on the private Postgres DB.

Phase 1 verifies durable cuts and resolution, including approval_replay refusal.
Phase-2a dispatch, replay and observe execution are out of scope; the recovery
worker only resolves actions and consumes submissions, then is killed again.
"""

import asyncio
import multiprocessing
import signal

import pytest

from dendrux.runtime.persistence import resolve_pending

CUTS = (
    "assistant",
    "intent",
    "approval_replay",
    "returned",
    "parallel",
    "gate",
    "approval_submit",
    "approval_consume",
    "rejected",
    "input_submit",
    "input_consume",
    "client_submit",
    "clarification",
    "final",
)


def _worker(url, cut, connection, effect_path, recovery_run=None):
    from sqlalchemy.ext.asyncio import create_async_engine

    from dendrux import Agent, tool
    from dendrux.llm.mock import MockLLM
    from dendrux.runtime.persistence import PersistenceRecorder
    from dendrux.runtime.state import ClaimResult, SQLAlchemyStateStore, claimed_store
    from dendrux.runtime.submit import deserialize_tool_results
    from dendrux.strategies.native import NativeToolCalling
    from dendrux.types import (
        AgentStep,
        Clarification,
        LLMResponse,
        Message,
        Role,
        ToolCall,
        ToolResult,
        tool_result_message,
    )

    def halt(run, stage):
        connection.send({"run": run, "stage": stage})
        connection.recv()  # parent sends SIGKILL; no child-side graceful cleanup
        raise AssertionError("kill handshake unexpectedly returned")

    class Store(SQLAlchemyStateStore):
        finalizing = False

        async def create_run(self, *args, **kwargs):
            result = await super().create_run(*args, **kwargs)
            if cut == "startup":
                halt(result.run_id, cut)
            return result

        async def checkpoint_step(self, run, *, claim_epoch, loop_state, pii_mapping, trace):
            await super().checkpoint_step(
                run,
                claim_epoch=claim_epoch,
                loop_state=loop_state,
                pii_mapping=pii_mapping,
                trace=trace,
            )
            if cut == "assistant" and trace is not None and (trace.meta or {}).get("tool_calls"):
                halt(run, cut)
            if cut == "clarification" and loop_state.get("phase") == "awaiting_input":
                halt(run, cut)
            if cut == "final" and trace is not None and trace.content == "final answer":
                halt(run, "final_committed_before_finalize")

        async def start_tool_call(self, run, *, claim_epoch, row_id, **kwargs):
            await super().start_tool_call(run, claim_epoch=claim_epoch, row_id=row_id, **kwargs)
            if cut == "intent":
                halt(run, cut)

        async def complete_tool_call(
            self, run, *, claim_epoch, observation, event, loop_state_patch, **kwargs
        ):
            if cut == "returned":
                halt(run, cut)
            await super().complete_tool_call(
                run,
                claim_epoch=claim_epoch,
                observation=observation,
                event=event,
                loop_state_patch=loop_state_patch,
                **kwargs,
            )
            if cut == "parallel":
                halt(run, cut)

        async def pause_with_event(self, run, *, claim_epoch=None, **kwargs):
            if cut == "gate":
                halt(run, cut)
            return await super().pause_with_event(run, claim_epoch=claim_epoch, **kwargs)

        async def submit_and_claim(self, run, *, submission=None, **kwargs):
            result = await super().submit_and_claim(run, submission=submission, **kwargs)
            if cut in ("approval_submit", "input_submit", "client_submit"):
                halt(run, cut)
            return result

        async def consume_submission(self, run, *, claim_epoch, traces, patch, **kwargs):
            result = await super().consume_submission(
                run, claim_epoch=claim_epoch, traces=traces, patch=patch, **kwargs
            )
            if cut in ("approval_consume", "rejected", "input_consume"):
                halt(run, cut)
            return result

        async def finalize_with_event(
            self, run, *, claim_epoch=None, trace=None, event=None, **kwargs
        ):
            self.finalizing = True
            return await super().finalize_with_event(
                run, claim_epoch=claim_epoch, trace=trace, event=event, **kwargs
            )

        async def _insert_trace(self, session, run, trace):
            await super()._insert_trace(session, run, trace)
            if cut == "final" and self.finalizing and trace.content == "final answer":
                halt(run, cut)

    async def main():
        engine = create_async_engine(url)
        if recovery_run is not None:
            store = SQLAlchemyStateStore(engine)
            run = await store.get_run(recovery_run)
            state = await store.get_loop_state(recovery_run)
            rows = await store.get_tool_calls(recovery_run)
            actions = resolve_pending(state, rows)
            if actions and actions[0].kind == "consume_submission":
                raw = await store.get_pause_state(recovery_run)
                payload = raw["submission"]["payload"]
                messages = []
                if "submitted_tool_results" in payload:
                    messages = [
                        tool_result_message(r)
                        for r in deserialize_tool_results(payload["submitted_tool_results"])
                    ]
                elif "submitted_user_input" in payload:
                    messages = [Message(Role.USER, payload["submitted_user_input"])]
                token = ClaimResult(run.claim_epoch, 2, state, state["submission"])
                recorder = PersistenceRecorder(
                    claimed_store(store, recovery_run, token),
                    recovery_run,
                    initial_order_index=state["trace_order_offset"],
                )
                await recorder.consume_resume(messages)
            halt(recovery_run, "recovery")
        store = Store(engine)
        entered = 0
        all_entered = asyncio.Event()
        never = asyncio.Event()

        @tool(idempotent=cut != "approval_replay")
        async def change() -> str:
            """Record a visible side effect."""
            with open(effect_path, "a") as output:
                output.write("effect\n")
            if cut == "approval_replay":
                from dendrux.runtime.context import get_delegation_context

                halt(get_delegation_context().run_id, cut)
            return "ok"

        @tool(target="client")
        async def client() -> str:
            """A client-owned call."""
            raise AssertionError("client executed")

        async def arrive():
            nonlocal entered
            entered += 1
            if entered == 3:
                all_entered.set()
            await all_entered.wait()

        @tool(idempotent=True)
        async def slow() -> str:
            """Hold the first call in flight."""
            await arrive()
            await never.wait()
            return "slow"

        @tool(idempotent=True)
        async def middle() -> str:
            """Hold the second call in flight."""
            await arrive()
            await never.wait()
            return "middle"

        @tool(idempotent=True)
        async def fast() -> str:
            """Finish the third call first."""
            await arrive()
            return "fast"

        class Ask(NativeToolCalling):
            def parse_response(self, response):
                return AgentStep(reasoning=None, action=Clarification(question="Which value?"))

        approval = cut in ("gate", "approval_submit", "approval_consume", "rejected")
        human = cut in ("input_submit", "input_consume", "clarification")
        tools = (
            [slow, middle, fast]
            if cut == "parallel"
            else [client]
            if cut == "client_submit"
            else [change]
        )
        calls = [ToolCall(t.__name__) for t in tools]
        first = (
            LLMResponse(text="Which value?")
            if human
            else LLMResponse(text="final answer")
            if cut == "final"
            else LLMResponse(tool_calls=calls)
        )
        from dendrux.guardrails import PII, Pattern

        agent = Agent(
            prompt="code AB12; product AB12X" if cut == "startup" else "system",
            guardrails=[PII(extra_patterns=[Pattern("CODE", r"\bAB12\b")], restore="all")]
            if cut == "startup"
            else [],
            provider=MockLLM([first, LLMResponse(text="final answer")]),
            tools=tools,
            state_store=store,
            require_approval=["change"] if approval else [],
            strict_tenancy=False,
        )
        paused = await agent.run("go", strategy=Ask() if human else NativeToolCalling())
        if approval:
            await agent.submit_approval(
                paused.run_id, approved=cut != "rejected", expected_pause_id=paused.pause_id
            )
        elif human:
            await agent.submit_input(paused.run_id, "chosen", expected_pause_id=paused.pause_id)
        elif cut == "client_submit":
            state = await store.get_loop_state(paused.run_id)
            await agent.submit_tool_results(
                paused.run_id,
                [ToolResult(name="client", call_id=state["pending"][0]["call_id"], payload='"ok"')],
                expected_pause_id=paused.pause_id,
            )
        raise AssertionError("cut was never reached")

    try:
        asyncio.run(main())
    except BaseException:
        import traceback

        connection.send({"error": traceback.format_exc()})
        raise


async def _kill_at_cut(url, cut, effect_path, recovery_run=None):
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_worker, args=(url, cut, child, str(effect_path), recovery_run))
    process.start()
    child.close()
    try:
        assert await asyncio.to_thread(parent.poll, 20), "child did not reach the crash cut"
        message = parent.recv()
        assert "error" not in message, message.get("error")
        process.kill()
        await asyncio.to_thread(process.join, 5)
        assert process.exitcode == -signal.SIGKILL
        return message
    finally:
        if process.is_alive():
            process.kill()
            await asyncio.to_thread(process.join, 5)
        parent.close()


@pytest.mark.parametrize("cut", CUTS)
async def test_a5_sigkill_committed_boundaries(engine, store, tmp_path, cut):
    if engine.dialect.name != "postgresql":
        pytest.skip("SIGKILL crash cuts require Postgres")
    effect_path = tmp_path / "effects"
    result = await _kill_at_cut(engine.url.render_as_string(hide_password=False), cut, effect_path)
    run = result["run"]
    state = await store.get_loop_state(run)
    rows = await store.get_tool_calls(run)
    traces = await store.get_traces(run)
    kinds = [a.kind for a in resolve_pending(state, rows)]
    expected = {
        "assistant": ["dispatch"],
        "intent": ["replay"],
        "approval_replay": ["approval_replay"],
        "returned": ["replay"],
        "parallel": ["replay", "replay"],
        "gate": ["approval"],
        "approval_submit": ["consume_submission"],
        "approval_consume": ["dispatch"],
        "rejected": [],
        "input_submit": ["consume_submission"],
        "input_consume": [],
        "client_submit": ["consume_submission"],
        "clarification": [],
        "final": [],
    }
    assert kinds == expected[cut]
    if cut in ("returned", "approval_replay"):
        assert effect_path.read_text() == "effect\n"
    elif cut != "parallel":
        assert not effect_path.exists()
    if cut in ("rejected", "input_consume"):
        assert len([t for t in traces if t.role == "tool" or t.content == "chosen"]) == 1
    if cut == "parallel":
        assert [r.outcome for r in rows].count("completed") == 1
        assert [(t.meta or {}).get("tool_name") for t in traces if t.role == "tool"] == ["fast"]
    if cut == "clarification":
        assert state["phase"] == "awaiting_input" and state["clarification_ref"]
        assert await store.get_pause_state(run) is None
    if cut == "final":
        assert result["stage"] == "final"
        assert not any(t.content == "final answer" for t in traces)
        assert (await store.get_run(run)).status == "running"
        assert not any(e.event_type == "run.completed" for e in await store.get_run_events(run))
    # A fresh process is killed again during resolution/consumption. No tool
    # is executed by this phase-1 recovery harness; phase 2 owns the dispatcher.
    await _kill_at_cut(engine.url.render_as_string(hide_password=False), cut, effect_path, run)
    after = await store.get_loop_state(run)
    if cut == "approval_replay":
        assert effect_path.read_text() == "effect\n"
        assert [a.kind for a in resolve_pending(after, await store.get_tool_calls(run))] == [
            "approval_replay"
        ]
    if state.get("submission"):
        assert after["submission"] is None
        after_traces = await store.get_traces(run)
        await _kill_at_cut(engine.url.render_as_string(hide_password=False), cut, effect_path, run)
        assert await store.get_traces(run) == after_traces
    else:
        assert after == state


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
async def test_startup_sigkill_keeps_provider_seeds_mapping_and_marker(engine, store, tmp_path):
    from dendrux.runtime.replay import rebuild_history

    effect_path = tmp_path / "effects"
    result = await _kill_at_cut(
        engine.url.render_as_string(hide_password=False), "startup", effect_path
    )
    run_id = result["run"]
    assert await store.get_loop_state(run_id) == {"v": 1, "loop": "react"}
    mapping = await store.get_pii_mapping(run_id)
    assert mapping["<<CODE_1>>"] == "AB12"
    rebuilt = await rebuild_history(store, run_id)
    assert [m.content for m in rebuilt.messages] == ["code <<CODE_1>>; product AB12X", "go"]
    assert not await store.get_run_events(run_id)
    assert not effect_path.exists()
