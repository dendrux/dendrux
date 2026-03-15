"""Integration tests for Example 03 — Client Tool Bridge.

Tests the example module's import safety and the full pause/resume HTTP
flow using MockLLM (no Anthropic API key required).

The key test (test_pause_resume_via_create_demo_app) calls create_demo_app()
directly with mock overrides, exercising the real mount structure, lazy
store proxy (skipped when overrides provided), and lifespan.
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from dendrite.llm.mock import MockLLM
from dendrite.server.registry import AgentRegistry, HostedAgentConfig
from dendrite.types import LLMResponse, ToolCall

# Import the example module by file path (not on sys.path)
_EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "03_client_tools"
_SERVER_PATH = _EXAMPLE_DIR / "server.py"


def _import_server_module():  # type: ignore[no-untyped-def]
    """Import server.py by file path without requiring __init__.py."""
    spec = importlib.util.spec_from_file_location("example_03_server", _SERVER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestExampleImport:
    def test_server_module_is_importable(self) -> None:
        """server.py can be imported without side effects (no env vars, no uvicorn)."""
        mod = _import_server_module()
        assert hasattr(mod, "agent")
        assert hasattr(mod, "create_demo_app")

    def test_agent_is_defined(self) -> None:
        mod = _import_server_module()
        assert mod.agent.name == "SpreadsheetAnalyst"
        assert len(mod.agent.tools) == 2

    def test_create_demo_app_is_callable(self) -> None:
        mod = _import_server_module()
        assert callable(mod.create_demo_app)


class TestFullPauseResumeCycle:
    """End-to-end HTTP test via create_demo_app(): create -> pause -> submit -> complete.

    Calls create_demo_app() with mock overrides so the actual mount
    structure, lifespan, and route registration are exercised.
    """

    async def test_pause_resume_via_create_demo_app(self) -> None:
        mod = _import_server_module()
        agent = mod.agent

        from tests.unit.test_server import MockServerStore

        store = MockServerStore()

        # Mock LLM: first call returns client tool call, second call returns answer
        tc = ToolCall(
            name="read_excel_range",
            params={"sheet": "Sheet1", "range": "A1"},
            provider_tool_call_id="tc1",
        )
        call_count: dict[str, int] = {"n": 0}

        def _provider_factory() -> MockLLM:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return MockLLM([LLMResponse(tool_calls=[tc])])
            return MockLLM([LLMResponse(text="Based on cell A1, the revenue is $394B.")])

        registry = AgentRegistry()
        registry.register(HostedAgentConfig(agent=agent, provider_factory=_provider_factory))

        # Call create_demo_app() with overrides — exercises the real
        # mount structure and lifespan, skipping lazy store / Anthropic
        app = mod.create_demo_app(state_store=store, registry=registry)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Verify HTML client is served at /
            html_resp = await client.get("/")
            assert html_resp.status_code == 200
            assert "Dendrite" in html_resp.text

            # 2. Create run (via mounted sub-app at /dendrite)
            resp = await client.post(
                "/dendrite/runs",
                json={
                    "agent_name": "SpreadsheetAnalyst",
                    "input": "Read cell A1 from my spreadsheet",
                },
            )
            assert resp.status_code == 200
            run_id = resp.json()["run_id"]

            # 3. Wait for background task to pause
            await asyncio.sleep(0.3)

            # 4. Poll — should be paused
            status_resp = await client.get(f"/dendrite/runs/{run_id}")
            assert status_resp.status_code == 200
            data = status_resp.json()
            assert data["status"] == "waiting_client_tool"
            assert data["pending_tool_calls"] is not None
            assert len(data["pending_tool_calls"]) == 1
            assert data["pending_tool_calls"][0]["tool_name"] == "read_excel_range"

            pending_id = data["pending_tool_calls"][0]["tool_call_id"]

            # 5. Submit tool result
            resume_resp = await client.post(
                f"/dendrite/runs/{run_id}/tool-results",
                json={
                    "tool_results": [
                        {
                            "tool_call_id": pending_id,
                            "tool_name": "read_excel_range",
                            "result": "Revenue: $394B",
                        }
                    ]
                },
            )
            assert resume_resp.status_code == 200
            assert resume_resp.json()["status"] == "success"

            # 6. Poll — should be completed
            final_resp = await client.get(f"/dendrite/runs/{run_id}")
            assert final_resp.status_code == 200
            assert final_resp.json()["status"] == "success"
