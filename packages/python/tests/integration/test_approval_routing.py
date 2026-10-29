"""Approval resume: target routing (A-P0-1), per-call decisions (A1, A-P0-2,
A-P1-4), the resume barrier, and before_tool persistence — SQLite + Postgres.

A batch holding an approval-gated server tool and a client-target tool
pauses for approval as a whole. Resolving it must never run the client
function on the server: the client call stays pending and the run moves
to WAITING_CLIENT_TOOL.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import select

from dendrux import ApprovalDecision, ToolDecision
from dendrux.agent import Agent
from dendrux.db.models import ToolCallRecord
from dendrux.llm.mock import MockLLM
from dendrux.runtime.runner import resume_approval_approved_claim, resume_claimed
from dendrux.tool import tool
from dendrux.types import LLMResponse, RunEventType, RunStatus, ToolCall, ToolResult

if TYPE_CHECKING:
    from dendrux.runtime.state import StateStore

CLIENT_SIDE_EFFECTS: list[str] = []
SERVER_SIDE_EFFECTS: list[str] = []
CANCEL_HOOK: dict[str, Any] = {}


@tool()
async def refund(order_id: int) -> str:
    SERVER_SIDE_EFFECTS.append(f"refund:{order_id}")
    hook = CANCEL_HOOK.get("after_refund")
    if hook is not None:
        await hook()
    return f"Refunded {order_id}"


@tool()
async def search(query: str) -> str:
    SERVER_SIDE_EFFECTS.append(f"search:{query}")
    return f"results:{query}"


@tool()
async def send_email(to: str) -> str:
    SERVER_SIDE_EFFECTS.append(f"send_email:{to}")
    return "sent"


@tool(target="client")
async def read_range(sheet: str) -> str:
    CLIENT_SIDE_EFFECTS.append(f"read_range:{sheet}")
    return "SERVER EXECUTED A CLIENT TOOL"


@pytest.fixture(autouse=True)
def _reset() -> None:
    CLIENT_SIDE_EFFECTS.clear()
    SERVER_SIDE_EFFECTS.clear()
    CANCEL_HOOK.clear()


def _agent(llm: MockLLM, store: StateStore, **kw: Any) -> Agent:
    defaults: dict[str, Any] = {
        "provider": llm,
        "prompt": "Mixed agent.",
        "tools": [refund, search, send_email, read_range],
        "require_approval": ["refund"],
        "state_store": store,
    }
    defaults.update(kw)
    return Agent(**defaults)


def _mixed_batch() -> list[ToolCall]:
    return [
        ToolCall(name="refund", params={"order_id": 7}, provider_tool_call_id="p1"),
        ToolCall(name="read_range", params={"sheet": "S1"}, provider_tool_call_id="p2"),
    ]


async def _events(store: StateStore, run_id: str, name: str) -> list[dict[str, Any]]:
    return [e.data or {} for e in await store.get_run_events(run_id) if e.event_type == name]


def _tool_contents(llm: MockLLM, call_index: int) -> dict[str, Any]:
    msgs = llm.call_history[call_index]["messages"]
    return {m.call_id: m.content for m in msgs if m.role.value == "tool"}


async def _finish_client(agent: Agent, store: StateStore, run_id: str) -> Any:
    pause = await store.get_pause_state(run_id)
    results = [
        ToolResult(name=tc["name"], call_id=tc["id"], payload='"rows"')
        for tc in pause["pending_tool_calls"]
    ]
    return await agent.submit_tool_results(run_id, results)


# ---------------------------------------------------------------------------
# A-P0-1 — batch form routes by persisted target
# ---------------------------------------------------------------------------


class TestBatchApproveRoutesByTarget:
    async def test_approve_never_runs_client_tool_on_server(self, store) -> None:
        llm = MockLLM([LLMResponse(tool_calls=_mixed_batch()), LLMResponse(text="done")])
        agent = _agent(llm, store)
        paused = await agent.run("refund and read")
        assert paused.status == RunStatus.WAITING_APPROVAL

        result = await agent.submit_approval(
            paused.run_id, approved=True, expected_pause_id=paused.pause_id
        )

        assert CLIENT_SIDE_EFFECTS == []
        assert SERVER_SIDE_EFFECTS == ["refund:7"]
        assert result.status == RunStatus.WAITING_CLIENT_TOOL
        pause = await store.get_pause_state(paused.run_id)
        assert [tc["name"] for tc in pause["pending_tool_calls"]] == ["read_range"]
        assert "approval_gates" not in pause

        final = await _finish_client(agent, store, paused.run_id)
        assert final.status == RunStatus.SUCCESS
        assert CLIENT_SIDE_EFFECTS == []
        # Model saw both results: server output + client-submitted rows.
        contents = list(_tool_contents(llm, 1).values())
        assert contents == ['"Refunded 7"', '"rows"']

        [decided] = await _events(store, paused.run_id, "approval.decided")
        assert decided["decision"] == "approved"
        assert [d["decision"] for d in decided["decisions"]] == ["approved", "client_pending"]

    async def test_no_arg_resume_routes_by_target(self, store) -> None:
        llm = MockLLM([LLMResponse(tool_calls=_mixed_batch()), LLMResponse(text="done")])
        agent = _agent(llm, store)
        paused = await agent.run("refund and read")
        result = await agent.resume(paused.run_id, expected_pause_id=paused.pause_id)
        assert result.status == RunStatus.WAITING_CLIENT_TOOL
        assert CLIENT_SIDE_EFFECTS == []

    async def test_resume_stream_routes_by_target(self, store) -> None:
        llm = MockLLM([LLMResponse(tool_calls=_mixed_batch()), LLMResponse(text="done")])
        agent = _agent(llm, store)
        paused = await agent.run("refund and read")
        events = [
            e async for e in agent.resume_stream(paused.run_id, expected_pause_id=paused.pause_id)
        ]
        assert CLIENT_SIDE_EFFECTS == []
        assert SERVER_SIDE_EFFECTS == ["refund:7"]
        assert [e.type for e in events] == [
            RunEventType.RUN_RESUMED,
            RunEventType.TOOL_RESULT,
            RunEventType.RUN_PAUSED,
        ]
        assert events[-1].run_result.status == RunStatus.WAITING_CLIENT_TOOL
        record = await store.get_run(paused.run_id)
        assert record.status == RunStatus.WAITING_CLIENT_TOOL.value
        final = await _finish_client(agent, store, paused.run_id)
        assert final.status == RunStatus.SUCCESS

    async def test_legacy_reject_payload_unchanged(self, store) -> None:
        llm = MockLLM([LLMResponse(tool_calls=_mixed_batch()), LLMResponse(text="ok")])
        agent = _agent(llm, store)
        paused = await agent.run("refund and read")
        result = await agent.submit_approval(
            paused.run_id,
            approved=False,
            rejection_reason="Not authorized.",
            expected_pause_id=paused.pause_id,
        )
        assert result.status == RunStatus.SUCCESS
        assert SERVER_SIDE_EFFECTS == [] and CLIENT_SIDE_EFFECTS == []
        # Byte-for-byte legacy payload: the reason as a JSON string, every call.
        assert list(_tool_contents(llm, 1).values()) == [json.dumps("Not authorized.")] * 2
        [decided] = await _events(store, paused.run_id, "approval.decided")
        assert decided["decision"] == "rejected"
        assert decided["run_id"] == paused.run_id
        assert [(d["tool_name"], d["decision"], d["reason"]) for d in decided["decisions"]] == [
            ("refund", "denied", "Not authorized."),
            ("read_range", "denied", "Not authorized."),
        ]


# ---------------------------------------------------------------------------
# A1 — per-call decisions
# ---------------------------------------------------------------------------


class TestPerCallDecisions:
    async def _pause_three(self, store: StateStore, final: str = "done") -> tuple:
        calls = [
            ToolCall(name="refund", params={"order_id": 1}),
            ToolCall(name="search", params={"query": "q"}),
            ToolCall(name="read_range", params={"sheet": "S"}),
            ToolCall(name="refund", params={"order_id": 2}),
        ]
        llm = MockLLM([LLMResponse(tool_calls=calls), LLMResponse(text=final)])
        agent = _agent(llm, store)
        paused = await agent.run("go")
        assert paused.status == RunStatus.WAITING_APPROVAL
        return agent, llm, paused, calls

    @pytest.mark.parametrize("fmt", [1, 2])
    async def test_mixed_server_client(self, store, engine, fmt) -> None:
        from tests._fixtures.storage import pin_storage_format

        await pin_storage_format(engine, fmt)
        agent, llm, paused, calls = await self._pause_three(store)
        pause = await store.get_pause_state(paused.run_id)
        ids = [tc["id"] for tc in pause["pending_tool_calls"]]
        assert set(pause["approval_gates"]) == {ids[0], ids[3]}

        result = await agent.submit_approval(
            paused.run_id,
            decisions={
                ids[0]: ApprovalDecision.deny("Too large"),
                ids[3]: ApprovalDecision.approve(),
            },
            expected_pause_id=paused.pause_id,
        )
        assert result.status == RunStatus.WAITING_CLIENT_TOOL
        assert SERVER_SIDE_EFFECTS == ["search:q", "refund:2"]
        assert CLIENT_SIDE_EFFECTS == []

        [decided] = await _events(store, paused.run_id, "approval.decided")
        assert decided["decision"] == "mixed"
        assert decided["decisions"] == [
            {"call_id": ids[0], "tool_name": "refund", "decision": "denied", "reason": "Too large"},
            {
                "call_id": ids[1],
                "tool_name": "search",
                "decision": "not_gated_executed",
                "reason": None,
            },
            {
                "call_id": ids[2],
                "tool_name": "read_range",
                "decision": "client_pending",
                "reason": None,
            },
            {"call_id": ids[3], "tool_name": "refund", "decision": "approved", "reason": None},
        ]
        assert "params" not in json.dumps(decided)

        final = await _finish_client(agent, store, paused.run_id)
        assert final.status == RunStatus.SUCCESS
        contents = _tool_contents(llm, 1)
        # Format 2 follows assistant call order; format 1 preserves the
        # legacy server-then-client order.
        assert list(contents) == (ids if fmt == 2 else [ids[0], ids[1], ids[3], ids[2]])
        assert json.loads(contents[ids[0]]) == {"denied": "Too large"}
        assert json.loads(contents[ids[3]]) == "Refunded 2"

    async def test_all_denied_default_reason(self, store) -> None:
        agent, llm, paused, _ = await self._pause_three(store)
        pause = await store.get_pause_state(paused.run_id)
        gated = list(pause["approval_gates"])
        result = await agent.submit_approval(
            paused.run_id,
            decisions={g: ApprovalDecision.deny() for g in gated},
            expected_pause_id=paused.pause_id,
        )
        assert result.status == RunStatus.WAITING_CLIENT_TOOL
        assert SERVER_SIDE_EFFECTS == ["search:q"]
        [decided] = await _events(store, paused.run_id, "approval.decided")
        assert decided["decision"] == "rejected"
        rows = {r.tool_call_id: r for r in await store.get_tool_calls(paused.run_id)}
        assert rows[gated[0]].result == {"denied": "User declined to run this tool."}
        assert rows[gated[0]].success is False

    async def test_all_approved_server_only(self, store) -> None:
        calls = [ToolCall(name="refund", params={"order_id": 1})]
        llm = MockLLM([LLMResponse(tool_calls=calls), LLMResponse(text="done")])
        agent = _agent(llm, store)
        paused = await agent.run("go")
        pause = await store.get_pause_state(paused.run_id)
        [gid] = list(pause["approval_gates"])
        result = await agent.submit_approval(
            paused.run_id,
            decisions={gid: ApprovalDecision.approve()},
            expected_pause_id=paused.pause_id,
        )
        assert result.status == RunStatus.SUCCESS
        [decided] = await _events(store, paused.run_id, "approval.decided")
        assert decided["decision"] == "approved"

    @pytest.mark.parametrize("problem", ["missing", "unknown", "non_gated"])
    async def test_bad_ids_fail_before_claim(self, store, problem) -> None:
        agent, _, paused, _ = await self._pause_three(store)
        pause = await store.get_pause_state(paused.run_id)
        ids = [tc["id"] for tc in pause["pending_tool_calls"]]
        decisions = {ids[0]: ApprovalDecision.approve(), ids[3]: ApprovalDecision.approve()}
        if problem == "missing":
            del decisions[ids[3]]
        elif problem == "unknown":
            decisions["nope"] = ApprovalDecision.approve()
        else:
            decisions[ids[1]] = ApprovalDecision.approve()
        with pytest.raises(ValueError, match=problem if problem != "non_gated" else "unknown"):
            await agent.submit_approval(
                paused.run_id, decisions=decisions, expected_pause_id=paused.pause_id
            )
        record = await store.get_run(paused.run_id)
        assert record.status == RunStatus.WAITING_APPROVAL.value
        assert "submitted_decisions" not in await store.get_pause_state(paused.run_id)
        assert SERVER_SIDE_EFFECTS == []

    async def test_argument_validation(self, store) -> None:
        agent, _, paused, _ = await self._pause_three(store)
        with pytest.raises(ValueError, match="exactly one"):
            await agent.submit_approval(paused.run_id, expected_pause_id=paused.pause_id)
        with pytest.raises(ValueError, match="exactly one"):
            await agent.submit_approval(
                paused.run_id, approved=True, decisions={}, expected_pause_id=paused.pause_id
            )
        with pytest.raises(ValueError, match="rejection_reason"):
            await agent.submit_approval(
                paused.run_id, decisions={}, rejection_reason="x", expected_pause_id=paused.pause_id
            )
        with pytest.raises(ValueError, match="ApprovalDecision"):
            await agent.submit_approval(
                paused.run_id, decisions={"a": True}, expected_pause_id=paused.pause_id
            )
        with pytest.raises(ValueError, match="mapping"):
            await agent.submit_approval(
                paused.run_id, decisions=[("a", True)], expected_pause_id=paused.pause_id
            )
        record = await store.get_run(paused.run_id)
        assert record.status == RunStatus.WAITING_APPROVAL.value

    async def test_duplicate_ids_in_claimed_blob_release_claim(self, store) -> None:
        # This fault injection / pause blob exercises the format-1 write path.
        from tests._fixtures.storage import pin_storage_format

        await pin_storage_format(store._engine, 1)
        agent, _, paused, _ = await self._pause_three(store)
        pause = await store.get_pause_state(paused.run_id)
        gated = list(pause["approval_gates"])
        blob = [{"call_id": g, "approved": True, "reason": None} for g in gated]
        blob.append(dict(blob[0]))
        assert await store.submit_and_claim(
            paused.run_id,
            expected_status=RunStatus.WAITING_APPROVAL.value,
            submitted_data={"submitted_decisions": blob},
        )
        with pytest.raises(ValueError, match="duplicate"):
            await resume_claimed(
                paused.run_id,
                state_store=store,
                agent=agent,
                provider=agent.provider,
                pre_claim_status=RunStatus.WAITING_APPROVAL.value,
            )
        assert SERVER_SIDE_EFFECTS == []
        record = await store.get_run(paused.run_id)
        assert record.status == RunStatus.WAITING_APPROVAL.value
        assert not await _events(store, paused.run_id, "run.error")

    async def test_wrong_pause_status(self, store) -> None:
        from dendrux.errors import PauseStatusMismatchError

        llm = MockLLM(
            [LLMResponse(tool_calls=[ToolCall(name="read_range", params={"sheet": "S"})])]
        )
        agent = _agent(llm, store)
        paused = await agent.run("go")
        assert paused.status == RunStatus.WAITING_CLIENT_TOOL
        with pytest.raises(PauseStatusMismatchError):
            await agent.submit_approval(
                paused.run_id, decisions={}, expected_pause_id=paused.pause_id
            )

    async def test_legacy_pause_without_gate_snapshot(self, store) -> None:
        """Pauses persisted before 0.3.0: every server call needs a decision."""
        agent, _, paused, _ = await self._pause_three(store)
        pause = await store.get_pause_state(paused.run_id)
        pause.pop("approval_gates")
        await store.pause_run(
            paused.run_id, status=RunStatus.WAITING_APPROVAL.value, pause_data=pause
        )
        ids = [tc["id"] for tc in pause["pending_tool_calls"]]
        with pytest.raises(ValueError, match="missing"):
            await agent.submit_approval(
                paused.run_id,
                decisions={ids[0]: ApprovalDecision.approve(), ids[3]: ApprovalDecision.approve()},
                expected_pause_id=paused.pause_id,
            )
        result = await agent.submit_approval(
            paused.run_id,
            decisions={
                ids[0]: ApprovalDecision.approve(),
                ids[1]: ApprovalDecision.deny("skip"),
                ids[3]: ApprovalDecision.approve(),
            },
            expected_pause_id=paused.pause_id,
        )
        assert result.status == RunStatus.WAITING_CLIENT_TOOL
        assert SERVER_SIDE_EFFECTS == ["refund:1", "refund:2"]
        assert CLIENT_SIDE_EFFECTS == []


# ---------------------------------------------------------------------------
# Hook-gated pause resumed by a different Agent instance
# ---------------------------------------------------------------------------


def _email_hook(call):
    if call.tool_name == "send_email" and call.params.get("to", "").endswith("@external.com"):
        return ToolDecision.require_approval("external recipient")
    return None


class TestHookGatedResumeOnNewAgent:
    async def _pause(self, store: StateStore) -> tuple:
        calls = [
            ToolCall(name="send_email", params={"to": "boss@external.com"}),
            ToolCall(name="search", params={"query": "q"}),
        ]
        llm = MockLLM([LLMResponse(tool_calls=calls), LLMResponse(text="done")])
        agent = _agent(llm, store, require_approval=None, before_tool=_email_hook)
        paused = await agent.run("go")
        assert paused.status == RunStatus.WAITING_APPROVAL
        pause = await store.get_pause_state(paused.run_id)
        ids = [tc["id"] for tc in pause["pending_tool_calls"]]
        assert pause["approval_gates"] == {
            ids[0]: {"source": "hook", "reason": "external recipient"}
        }
        [requested] = await _events(store, paused.run_id, "approval.requested")
        assert requested["source"] == "hook"
        return llm, paused, ids

    @pytest.mark.parametrize("variant", ["no_hook", "different_hook"])
    async def test_new_agent_decides_from_snapshot(self, store, variant) -> None:
        llm, paused, ids = await self._pause(store)
        hook_calls: list[str] = []

        def other_hook(call):
            hook_calls.append(call.tool_name)
            return ToolDecision.deny("would deny everything")

        resumer = _agent(
            llm,
            store,
            require_approval=None,
            before_tool=other_hook if variant == "different_hook" else None,
        )
        # Decisions validate against the persisted snapshot, not the new hook.
        with pytest.raises(ValueError, match="unknown"):
            await resumer.submit_approval(
                paused.run_id,
                decisions={
                    ids[0]: ApprovalDecision.approve(),
                    ids[1]: ApprovalDecision.approve(),
                },
                expected_pause_id=paused.pause_id,
            )
        result = await resumer.submit_approval(
            paused.run_id,
            decisions={ids[0]: ApprovalDecision.approve()},
            expected_pause_id=paused.pause_id,
        )
        assert result.status == RunStatus.SUCCESS
        # The hook is not re-run for approved or held calls on resume.
        assert hook_calls == []
        assert SERVER_SIDE_EFFECTS == ["send_email:boss@external.com", "search:q"]

    async def test_new_agent_batch_approve(self, store) -> None:
        llm, paused, _ = await self._pause(store)
        resumer = _agent(llm, store, require_approval=None)
        result = await resumer.submit_approval(
            paused.run_id, approved=True, expected_pause_id=paused.pause_id
        )
        assert result.status == RunStatus.SUCCESS
        [decided] = await _events(store, paused.run_id, "approval.decided")
        assert [d["decision"] for d in decided["decisions"]] == ["approved", "not_gated_executed"]


# ---------------------------------------------------------------------------
# Resume barrier (cancel before approved side effects)
# ---------------------------------------------------------------------------


class TestResumeBarrier:
    async def test_cancel_after_claim_skips_execution(self, store) -> None:
        llm = MockLLM([LLMResponse(tool_calls=[ToolCall(name="refund", params={"order_id": 1})])])
        agent = _agent(llm, store)
        paused = await agent.run("go")
        assert await store.claim_paused_run(
            paused.run_id, expected_status=RunStatus.WAITING_APPROVAL.value
        )
        await store.request_cancel(paused.run_id)
        result = await resume_approval_approved_claim(
            paused.run_id, state_store=store, agent=agent, provider=llm
        )
        assert result.status == RunStatus.CANCELLED
        assert SERVER_SIDE_EFFECTS == []
        record = await store.get_run(paused.run_id)
        assert record.status == RunStatus.CANCELLED.value
        assert await _events(store, paused.run_id, "run.cancelled")
        assert len(llm.call_history) == 1

    async def test_cancel_between_serial_approved_calls(self, store) -> None:
        calls = [
            ToolCall(name="refund", params={"order_id": 1}),
            ToolCall(name="search", params={"query": "q"}),
        ]
        llm = MockLLM([LLMResponse(tool_calls=calls), LLMResponse(text="never")])
        agent = _agent(llm, store)
        paused = await agent.run("go")

        async def cancel() -> None:
            await store.request_cancel(paused.run_id)

        CANCEL_HOOK["after_refund"] = cancel
        result = await agent.submit_approval(
            paused.run_id, approved=True, expected_pause_id=paused.pause_id
        )
        assert result.status == RunStatus.CANCELLED
        assert SERVER_SIDE_EFFECTS == ["refund:1"]
        assert len(llm.call_history) == 1

    async def test_stream_barrier_yields_cancelled(self, store) -> None:
        llm = MockLLM([LLMResponse(tool_calls=[ToolCall(name="refund", params={"order_id": 1})])])
        agent = _agent(llm, store)
        paused = await agent.run("go")
        await store.request_cancel(paused.run_id)
        events = [
            e async for e in agent.resume_stream(paused.run_id, expected_pause_id=paused.pause_id)
        ]
        assert events[-1].type == RunEventType.RUN_CANCELLED
        assert SERVER_SIDE_EFFECTS == []


# ---------------------------------------------------------------------------
# before_tool persistence
# ---------------------------------------------------------------------------


class TestHookPersistence:
    async def test_substitute_row_meta_and_tenant(self, store, session_factory) -> None:
        seen: list[Any] = []

        def hook(call):
            seen.append(call.tenant_id)
            return ToolDecision.substitute({"status": "held_for_review"})

        llm = MockLLM(
            [
                LLMResponse(tool_calls=[ToolCall(name="search", params={"query": "q"})]),
                LLMResponse(text="ok"),
            ]
        )
        agent = _agent(llm, store, require_approval=None, before_tool=hook)
        result = await agent.run("go", tenant_id="tenant-a")
        assert result.status == RunStatus.SUCCESS
        assert seen == ["tenant-a"]
        assert SERVER_SIDE_EFFECTS == []
        async with session_factory() as session:
            rows = (
                (
                    await session.execute(
                        select(ToolCallRecord).where(ToolCallRecord.agent_run_id == result.run_id)
                    )
                )
                .scalars()
                .all()
            )
        [row] = rows
        meta = dict(row.meta)
        observation = meta.pop("dendrux.observation", None)
        assert meta == {"substituted": True}
        assert row.result == {"status": "held_for_review"}
        if (await store.get_run(result.run_id)).storage_format == 2:
            assert json.loads(observation["content"]) == row.result
        else:
            assert observation is None
        assert await _events(store, result.run_id, "policy.substituted")

    async def test_hook_error_event_persisted(self, store) -> None:
        def hook(call):
            raise RuntimeError("boom with secret")

        llm = MockLLM(
            [
                LLMResponse(tool_calls=[ToolCall(name="search", params={"query": "q"})]),
                LLMResponse(text="ok"),
            ]
        )
        agent = _agent(llm, store, require_approval=None, before_tool=hook)
        result = await agent.run("go")
        [err] = await _events(store, result.run_id, "policy.hook_error")
        assert err["error_type"] == "RuntimeError"
        assert "secret" not in json.dumps(err)
        [denied] = await _events(store, result.run_id, "policy.denied")
        assert denied["source"] == "hook"


class TestRejectionViaResumeStream:
    async def test_stream_rejection_decided_payload(self, store) -> None:
        llm = MockLLM([LLMResponse(tool_calls=_mixed_batch()), LLMResponse(text="ok")])
        agent = _agent(llm, store)
        paused = await agent.run("go")
        pause = await store.get_pause_state(paused.run_id)
        results = [
            ToolResult(name=tc["name"], call_id=tc["id"], payload='"no"', success=False, error="no")
            for tc in pause["pending_tool_calls"]
        ]
        events = [
            e
            async for e in agent.resume_stream(
                paused.run_id, tool_results=results, expected_pause_id=paused.pause_id
            )
        ]
        assert events[-1].run_result.status == RunStatus.SUCCESS
        assert SERVER_SIDE_EFFECTS == [] and CLIENT_SIDE_EFFECTS == []
        [decided] = await _events(store, paused.run_id, "approval.decided")
        assert decided["decision"] == "rejected"
        assert [d["decision"] for d in decided["decisions"]] == ["denied", "denied"]

    async def test_decisions_guard_on_non_approval_resume(self, store) -> None:
        from dendrux.runtime.runner import _resume_core

        llm = MockLLM([])
        agent = _agent(llm, store)
        with pytest.raises(ValueError, match="awaiting approval"):
            await _resume_core(
                "r",
                state_store=store,
                agent=agent,
                provider=llm,
                expected_status=RunStatus.WAITING_CLIENT_TOOL.value,
                decisions=[],
            )


class TestStalePauseGeneration:
    async def test_decisions_for_replaced_pause_fail_before_claim(self, store) -> None:
        """ABA: the legacy physical pause is replaced between validation and claim."""
        from tests._fixtures.storage import pin_storage_format

        await pin_storage_format(store._engine, 1)
        import copy

        llm = MockLLM(
            [
                LLMResponse(tool_calls=[ToolCall(name="refund", params={"order_id": 1})]),
                LLMResponse(text="done"),
            ]
        )
        agent = _agent(llm, store)
        paused = await agent.run("go")
        old_pause = await store.get_pause_state(paused.run_id)
        old_id = old_pause["pending_tool_calls"][0]["id"]
        new_id = "new-pause-call"
        new_pause = copy.deepcopy(old_pause)
        new_pause["pending_tool_calls"][0]["id"] = new_id
        new_pause["pending_targets"] = {new_id: "server"}
        new_pause["approval_gates"] = {new_id: {"source": "static", "reason": "requires_approval"}}

        original = store.submit_and_claim

        async def replace_pause_then_claim(run_id, **kwargs):
            await store.pause_run(
                run_id, status=RunStatus.WAITING_APPROVAL.value, pause_data=new_pause
            )
            return await original(run_id, **kwargs)

        store.submit_and_claim = replace_pause_then_claim
        try:
            with pytest.raises(ValueError, match="unknown"):
                await agent.submit_approval(
                    paused.run_id,
                    decisions={old_id: ApprovalDecision.approve()},
                    expected_pause_id=paused.pause_id,
                )
        finally:
            store.submit_and_claim = original

        record = await store.get_run(paused.run_id)
        assert record.status == RunStatus.WAITING_APPROVAL.value
        assert await store.get_pause_state(paused.run_id) == new_pause
        assert SERVER_SIDE_EFFECTS == []

        # The newer pause is still healthy and accepts its own decisions.
        result = await agent.submit_approval(
            paused.run_id,
            decisions={new_id: ApprovalDecision.approve()},
            expected_pause_id=paused.pause_id,
        )
        assert result.status == RunStatus.SUCCESS
        assert SERVER_SIDE_EFFECTS == ["refund:1"]
