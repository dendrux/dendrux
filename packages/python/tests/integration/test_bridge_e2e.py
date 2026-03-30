"""End-to-end integration test for the bridge pause/resume flow.

Tests the core bridge cycle: agent.run() → pause → POST /tool-results
via bridge HTTP → resume → poll → complete. Uses a synthetic app with
MockLLM, not the Example 03 demo app.

Note: the Example 03 browser demo path (POST /chat, SSE stream,
final-answer fetch) is not covered by automated tests. That is a
demo, not a contract — manual testing is expected.
"""

from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine

from dendrite.agent import Agent
from dendrite.bridge import bridge
from dendrite.db.models import Base
from dendrite.llm.mock import MockLLM
from dendrite.runtime.state import SQLAlchemyStateStore
from dendrite.tool import tool
from dendrite.types import LLMResponse, RunStatus, ToolCall

# ------------------------------------------------------------------
# Test tools
# ------------------------------------------------------------------


@tool()
async def server_add(a: int, b: int) -> int:
    """Server-side add."""
    return a + b


@tool(target="client")
async def read_range(sheet: str) -> str:
    """Client-side tool — reads from Excel."""
    return ""


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
async def db_store():
    """Create a fresh in-memory SQLite state store."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    store = SQLAlchemyStateStore(engine)
    yield store
    await engine.dispose()


# ------------------------------------------------------------------
# Full pause/resume cycle via bridge HTTP
# ------------------------------------------------------------------


class TestBridgeEndToEnd:
    async def test_pause_submit_resume_complete(self, db_store) -> None:
        """Full cycle: agent.run() → pause → POST /tool-results → resume → complete."""

        # MockLLM sequence:
        # 1. First call: agent calls read_range (client tool → pause)
        # 2. Second call (after resume): agent returns final answer
        tc = ToolCall(
            name="read_range",
            params={"sheet": "Sheet1"},
            provider_tool_call_id="p1",
        )
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[tc]),
                LLMResponse(text="The data shows revenue of $42.50"),
            ],
            model="mock",
        )

        agent = Agent(
            provider=llm,
            prompt="You are a spreadsheet analyst.",
            tools=[server_add, read_range],
            state_store=db_store,
        )

        # 1. Start run — agent hits client tool, pauses
        result = await agent.run("Read the spreadsheet")

        assert result.status == RunStatus.WAITING_CLIENT_TOOL
        run_id = result.run_id
        assert run_id is not None

        # Extract pending tool call info
        pause_data = await db_store.get_pause_state(run_id)
        assert pause_data is not None
        pending = pause_data["pending_tool_calls"]
        assert len(pending) == 1
        assert pending[0]["name"] == "read_range"
        tool_call_id = pending[0]["id"]

        # 2. Create bridge and mount on test app
        from fastapi import FastAPI

        app = FastAPI()
        app.mount("/dendrite", bridge(agent, allow_insecure_dev_mode=True))

        # 3. POST tool results via HTTP
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                f"/dendrite/runs/{run_id}/tool-results",
                json={
                    "tool_results": [
                        {
                            "tool_call_id": tool_call_id,
                            "tool_name": "read_range",
                            "result": '"Revenue: $42.50"',
                            "success": True,
                        }
                    ]
                },
            )

            assert resp.status_code == 200
            body = resp.json()
            assert body["run_id"] == run_id
            assert body["status"] == "accepted"

            # 4. Poll for final result (background resume is async)
            for _ in range(20):
                poll_resp = await client.get(f"/dendrite/runs/{run_id}")
                assert poll_resp.status_code == 200
                poll_body = poll_resp.json()
                if poll_body["status"] != "running":
                    break
                await asyncio.sleep(0.1)

            assert poll_body["status"] == "success"
            assert "42.50" in (poll_body.get("answer") or "")

    async def test_submit_to_nonexistent_run_returns_409(self, db_store) -> None:
        """Submitting tool results for a run that doesn't exist returns 409."""
        llm = MockLLM([], model="mock")
        agent = Agent(
            provider=llm,
            prompt="Test.",
            tools=[server_add],
            state_store=db_store,
        )

        from fastapi import FastAPI

        app = FastAPI()
        app.mount("/dendrite", bridge(agent, allow_insecure_dev_mode=True))

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/dendrite/runs/nonexistent/tool-results",
                json={
                    "tool_results": [
                        {
                            "tool_call_id": "tc1",
                            "tool_name": "read_range",
                            "result": '"data"',
                        }
                    ]
                },
            )
            assert resp.status_code == 409

    async def test_poll_nonexistent_run_returns_404(self, db_store) -> None:
        """Polling a run that doesn't exist returns 404."""
        llm = MockLLM([], model="mock")
        agent = Agent(
            provider=llm,
            prompt="Test.",
            tools=[server_add],
            state_store=db_store,
        )

        from fastapi import FastAPI

        app = FastAPI()
        app.mount("/dendrite", bridge(agent, allow_insecure_dev_mode=True))

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/dendrite/runs/nonexistent")
            assert resp.status_code == 404

    async def test_cancel_paused_run(self, db_store) -> None:
        """DELETE cancels a paused run."""
        tc = ToolCall(
            name="read_range",
            params={"sheet": "Sheet1"},
            provider_tool_call_id="p1",
        )
        llm = MockLLM([LLMResponse(tool_calls=[tc])], model="mock")
        agent = Agent(
            provider=llm,
            prompt="Test.",
            tools=[server_add, read_range],
            state_store=db_store,
        )

        result = await agent.run("Read it")
        assert result.status == RunStatus.WAITING_CLIENT_TOOL
        run_id = result.run_id

        from fastapi import FastAPI

        app = FastAPI()
        app.mount("/dendrite", bridge(agent, allow_insecure_dev_mode=True))

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.delete(f"/dendrite/runs/{run_id}")
            assert resp.status_code == 200
            body = resp.json()
            assert body["cancelled"] is True

            # Verify status is cancelled
            poll = await client.get(f"/dendrite/runs/{run_id}")
            assert poll.json()["status"] == "cancelled"
