"""Client Tool Streaming — agent with server + client tools over NDJSON.

Demonstrates Dendrux streaming with client-side tool execution using
fetch() + NDJSON instead of Dendrux's SSE transport.

Run with:
    cd packages/python
    ANTHROPIC_API_KEY=sk-... python examples/09_client_tools_streaming/server.py

Then open http://localhost:8001

Two endpoints, same NDJSON wire format:
    POST /chat          — starts a new streaming run
    POST /resume/{id}   — resumes a paused run with tool results
"""

import json
from typing import Any

from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.responses import StreamingResponse

from dendrux import Agent, tool
from dendrux.types import RunEvent  # noqa: TC001

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


# Server tool: executes on the server, no pause
@tool()
async def lookup_price(ticker: str) -> str:
    """Look up the current stock price for a ticker symbol."""
    prices = {"AAPL": 227.50, "GOOGL": 178.30, "MSFT": 445.20, "TSLA": 312.80}
    price = prices.get(ticker.upper())
    if price is None:
        return f"Unknown ticker: {ticker}"
    return f"{ticker.upper()}: ${price:.2f}"


# Client tool: the agent pauses and waits for the client to execute
@tool(target="client")
async def read_excel_range(sheet: str, range: str) -> str:
    """Read a range of cells from the user's Excel spreadsheet.

    This tool runs on the client side. The server will pause and wait
    for the client to provide the result.
    """
    return ""


# ---------------------------------------------------------------------------
# Wire-format serializer
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Ensure a value survives json.dumps. Falls back to str()."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _serialize_event(event: RunEvent) -> dict[str, Any]:
    """Convert a RunEvent to a stable wire-format dict.

    Only includes fields relevant to each event type.
    Enum values are emitted as strings, not enum objects.
    """
    d: dict[str, Any] = {"type": event.type.value}

    if event.run_id is not None:
        d["run_id"] = event.run_id
    if event.text is not None:
        d["text"] = event.text
    if event.tool_name is not None:
        d["tool_name"] = event.tool_name
    if event.tool_call_id is not None:
        d["tool_call_id"] = event.tool_call_id
    if event.error is not None:
        d["error"] = event.error

    if event.tool_result is not None:
        tr = event.tool_result
        trd: dict[str, Any] = {
            "call_id": tr.call_id,
            "name": tr.name,
            "success": tr.success,
            "payload": _json_safe(tr.payload),
        }
        if tr.error is not None:
            trd["error"] = tr.error
        d["tool_result"] = trd

    if event.run_result is not None:
        r = event.run_result
        rd: dict[str, Any] = {"status": r.status.value}
        if r.answer is not None:
            rd["answer"] = r.answer
        if r.error is not None:
            rd["error"] = r.error
        # On RUN_PAUSED: include pending tool calls for the browser
        ps = r.meta.get("pause_state")
        if ps is not None:
            rd["pending_tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "params": _json_safe(tc.params),
                }
                for tc in ps.pending_tool_calls
            ]
        d["run_result"] = rd

    return d


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def create_demo_app():
    """Create the demo FastAPI application."""
    import os
    from pathlib import Path

    from dotenv import load_dotenv
    from fastapi import FastAPI

    from dendrux.llm.anthropic import AnthropicProvider
    from dendrux.types import ToolResult

    load_dotenv(Path(__file__).resolve().parents[4] / ".env")

    db_path = Path.home() / ".dendrux" / "dendrux.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"

    agent = Agent(
        name="SpreadsheetAnalyst",
        provider=AnthropicProvider(
            model="claude-sonnet-4-6",
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        ),
        database_url=db_url,
        prompt=(
            "You are a spreadsheet analyst. You can look up stock prices with "
            "lookup_price (runs instantly on the server) and read data from the "
            "user's Excel file with read_excel_range (runs on the client — you "
            "must wait for the user to provide the data). "
            "Always use the tools when the user asks about prices or spreadsheet data."
        ),
        tools=[lookup_price, read_excel_range],
    )

    app = FastAPI(title="Dendrux Client Tools Streaming Demo")

    client_html = Path(__file__).parent / "client.html"

    # ------------------------------------------------------------------
    # Shared NDJSON streaming helper
    # ------------------------------------------------------------------

    async def _event_generator(stream):  # type: ignore[no-untyped-def]
        """Iterate a RunStream, yield NDJSON lines.

        Ensures stream.aclose() fires on client disconnect, serialization
        error, or normal stream end.
        """
        try:
            async for event in stream:
                wire = _serialize_event(event)
                yield json.dumps(wire) + "\n"
        except Exception as exc:
            # Serialization or stream error — emit one error event
            yield json.dumps({"type": "run_error", "error": str(exc)}) + "\n"
        finally:
            await stream.aclose()

    def _ndjson_response(stream) -> StreamingResponse:  # type: ignore[no-untyped-def]
        return StreamingResponse(
            _event_generator(stream),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(client_html, media_type="text/html")

    @app.post("/chat")
    async def start_run(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

        user_input = (body.get("input") or "").strip()
        if not user_input:
            return JSONResponse(status_code=400, content={"error": "input is required"})

        stream = agent.stream(user_input)
        return _ndjson_response(stream)

    @app.post("/resume/{run_id}")
    async def resume_run(run_id: str, request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

        raw_results = body.get("tool_results")
        if not isinstance(raw_results, list) or not raw_results:
            return JSONResponse(
                status_code=400,
                content={"error": "tool_results array is required"},
            )

        try:
            tool_results = [
                ToolResult(
                    name=r["name"],
                    call_id=r["call_id"],
                    payload=(
                        r["result"] if isinstance(r["result"], str) else json.dumps(r["result"])
                    ),
                )
                for r in raw_results
            ]
        except (KeyError, TypeError) as exc:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid tool_results: {exc}"},
            )

        stream = agent.resume_stream(run_id, tool_results=tool_results)
        return _ndjson_response(stream)

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_demo_app(), host="0.0.0.0", port=8000)
