"""Exercise the developer-owned routes without a provider or a running server."""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from dendrux import Agent, tool
from dendrux.llm.mock import MockLLM
from dendrux.store import RunStore
from dendrux.types import LLMResponse, ToolCall

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
REPO = Path(__file__).resolve().parents[4]


def _load(name):
    path = EXAMPLES / name / "server.py"
    spec = importlib.util.spec_from_file_location(f"demo_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@tool(target="client")
async def local_read(sheet: str) -> str:
    """Read from the authenticated actor's spreadsheet."""
    return ""


@pytest.mark.parametrize("name", ["03_client_tools", "17_full_stack_app"])
async def test_client_routes_enforce_auth_scope_and_resume(name, tmp_path, monkeypatch):
    module = _load(name)
    monkeypatch.setenv("DENDRUX_DEMO_TOKEN", "test-token")
    agents = []

    def build(url):
        agent = Agent(
            provider=MockLLM(
                [
                    LLMResponse(tool_calls=[ToolCall("local_read", {"sheet": "private"})]),
                    LLMResponse(text="done"),
                ]
            ),
            prompt="p",
            tools=[local_read],
            database_url=url,
        )
        agents.append(agent)
        return {"anthropic": agent}

    monkeypatch.setattr(module, "build_agents", build)
    app = module.create_demo_app(database_url=f"sqlite+aiosqlite:///{tmp_path / 'demo.db'}")
    headers = {"X-App-Token": "test-token"}
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://demo") as client,
    ):
        assert (
            await client.post("/chat?token=test-token", json={"input": "hi"})
        ).status_code == 401
        start = await client.post("/chat", headers=headers, json={"input": "hi"})
        assert start.status_code == 200
        body = start.json()
        assert set(body) == {"run_id", "status", "answer"}
        assert body["status"] == "waiting_client_tool"
        run_id = body["run_id"]
        pending_url = f"/runs/{run_id}/pending"
        assert (await client.get(pending_url)).status_code == 401
        assert (await client.get("/runs/unknown/pending", headers=headers)).status_code == 404
        pending = (await client.get(pending_url, headers=headers)).json()
        call = pending["pending_tool_calls"][0]
        assert call["params"] == {"sheet": "private"}
        result = {"tool_call_id": call["id"], "tool_name": call["name"], "payload": '"rows"'}
        submit_url = f"/runs/{run_id}/tool-results"
        assert (await client.post(submit_url, json={"results": [result]})).status_code == 401
        invalid = {**result, "tool_call_id": "wrong"}
        assert (
            await client.post(submit_url, headers=headers, json={"results": [invalid]})
        ).status_code == 400
        resumed = await client.post(submit_url, headers=headers, json={"results": [result]})
        assert resumed.status_code == 200
        assert resumed.json() == {"run_id": run_id, "status": "success", "answer": "done"}
        assert (await client.get(pending_url, headers=headers)).json() is None
        async with RunStore.from_database_url(
            f"sqlite+aiosqlite:///{tmp_path / 'demo.db'}"
        ) as store:
            assert await store.get_run(run_id, tenant_id=None) is None
            assert (await store.get_run(run_id, tenant_id=module.TENANT)).status == "success"


async def test_approval_route_stops_on_hidden_args_and_requires_auth(tmp_path, monkeypatch):
    module = _load("17_full_stack_app")
    monkeypatch.setenv("DENDRUX_DEMO_TOKEN", "test-token")
    executed = []

    @tool()
    async def refund(order: int) -> str:
        """Refund an order after an explicit authorized decision."""
        executed.append(order)
        return "refunded"

    def build(url):
        return {
            "anthropic": Agent(
                provider=MockLLM(
                    [
                        LLMResponse(tool_calls=[ToolCall("refund", {"order": 7})]),
                        LLMResponse(text="done"),
                    ]
                ),
                prompt="p",
                tools=[refund],
                require_approval=["refund"],
                database_url=url,
            )
        }

    monkeypatch.setattr(module, "build_agents", build)
    app = module.create_demo_app(database_url=f"sqlite+aiosqlite:///{tmp_path / 'demo.db'}")
    headers = {"X-App-Token": "test-token"}
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://demo") as client,
    ):
        body = (await client.post("/chat", headers=headers, json={"input": "refund"})).json()
        assert body["status"] == "waiting_approval"
        url = f"/runs/{body['run_id']}/approvals"
        assert (await client.post(url, json={"approved": True})).status_code == 401
        original = RunStore.get_pending

        async def hidden(self, *args, **kwargs):
            pause = await original(self, *args, **kwargs)
            return replace(
                pause,
                tool_calls=[replace(c, args_hidden=True, params=None) for c in pause.tool_calls],
            )

        with monkeypatch.context() as patch:
            patch.setattr(RunStore, "get_pending", hidden)
            assert (
                await client.post(url, headers=headers, json={"approved": True})
            ).status_code == 409
            assert executed == []
        reviewed = (await client.get(f"/runs/{body['run_id']}/pending", headers=headers)).json()
        result = await client.post(
            url, headers=headers, json={"approved": True, "expected_pause_id": reviewed["pause_id"]}
        )
        assert result.status_code == 200
        assert result.json()["status"] == "success"
        assert executed == [7]


@pytest.mark.parametrize("stall_headers", [True, False])
async def test_live_recipe_bounds_network_send_and_releases_follower(stall_headers):
    """Execute the documented response with a send that never completes."""
    path = REPO / "docs/recipes/build-your-own-read-routes.mdx"
    snippets = re.findall(r"```python\n(.*?)```", path.read_text(), re.S)
    snippet = next(code for code in snippets if "class LiveResponse" in code)
    tree = ast.parse(snippet)
    tree.body = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef))
    ]
    namespace = {}
    exec(compile(tree, str(path), "exec"), namespace)
    closed = []

    async def close():
        closed.append(True)

    async def frames():
        yield "data: test\n\n"

    async def send(message):
        if stall_headers or message["type"] == "http.response.body":
            await asyncio.Event().wait()

    response = namespace["LiveResponse"](
        frames(),
        follow=SimpleNamespace(aclose=close),
        deadline=asyncio.get_running_loop().time() + 0.02,
    )
    async with asyncio.timeout(1):
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, None, send)
    assert closed == [True]


@pytest.mark.parametrize("outcome", ["drained", "missing", "capacity", "invalid_cursor"])
async def test_live_recipe_validates_cursor_and_closes_on_startup_outcomes(outcome):
    from fastapi import FastAPI

    from dendrux.errors import RunNotFoundError
    from dendrux.store import FollowCapacityError

    app = FastAPI()
    calls = []
    closed = []

    async def owned_run(run_id: str):
        return run_id

    async def current_tenant():
        return "T"

    async def start():
        if outcome == "missing":
            raise RunNotFoundError("r")
        if outcome == "capacity":
            raise FollowCapacityError(500)

    async def close():
        closed.append(True)

    def follow(run_id, **kwargs):
        calls.append((run_id, kwargs))
        return SimpleNamespace(start=start, aclose=close, drained_terminal=True)

    path = REPO / "docs/recipes/build-your-own-read-routes.mdx"
    snippets = re.findall(r"```python\n(.*?)```", path.read_text(), re.S)
    snippet = next(code for code in snippets if "class LiveResponse" in code)
    namespace = {
        "app": app,
        "store": SimpleNamespace(follow=follow),
        "owned_run": owned_run,
        "current_tenant": current_tenant,
    }
    exec(compile(snippet, str(path), "exec"), namespace)
    headers = {"Last-Event-ID": "invalid" if outcome == "invalid_cursor" else "7"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://demo") as client:
        response = await client.get("/runs/r/live", headers=headers)
    assert (
        response.status_code
        == {"drained": 204, "missing": 404, "capacity": 503, "invalid_cursor": 400}[outcome]
    )
    if outcome == "invalid_cursor":
        assert calls == [] and closed == []
    else:
        assert closed == [True]
        assert calls[0][0] == "r"
        assert calls[0][1]["tenant_id"] == "T"
        assert calls[0][1]["after_sequence_index"] == 7
    if outcome == "capacity":
        assert response.headers["Retry-After"] == "5"
