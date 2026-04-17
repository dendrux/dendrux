# Example 03: Client-Side Tool Pause/Resume

Demonstrates Dendrux's core differentiator — an agent that **pauses** when it needs a client-side tool, waits for the user to provide the result, then **resumes** reasoning.

## Setup

```bash
cd packages/python
pip install -e ".[dev,db,anthropic,http]"
```

Set your API key:

```bash
export ANTHROPIC_API_KEY=sk-...
```

## Run

```bash
python examples/03_client_tools/server.py
```

Open http://localhost:8000 in your browser.

## What to type

The agent has two tools:

- **lookup_price** (server-side) — instant, no pause
- **read_excel_range** (client-side) — agent pauses and waits for you

To see the pause/resume flow, use a prompt that triggers both:

> Look up the AAPL stock price, then read cell A1 from my spreadsheet.

The agent will:

1. Call `lookup_price("AAPL")` — executes instantly on the server.
2. Call `read_excel_range(sheet="Sheet1", range="A1")` — **pauses**.
3. The UI shows a form asking you to provide the result.
4. Type any value (e.g. `Revenue: $394B`) and click Submit.
5. The agent resumes with your data and generates a final answer.

## Architecture

Dendrux ships:

- `make_read_router` → mounts reads + SSE at `/api/*`.
- `agent.submit_tool_results(...)` + `agent.cancel_run(...)` → race-safe resume methods.

The developer writes:

- `POST /chat` → calls `agent.run(...)`.
- `POST /runs/{id}/tool-results` → calls `agent.submit_tool_results(...)`.
- `DELETE /runs/{id}` → calls `agent.cancel_run(...)`.
- `client.html` → browser UI, `EventSource` on the SSE stream, dispatches on `event_type`.

```
Browser                    Server                     Agent
  |                          |                          |
  |-- POST /chat ----------> |-- agent.run() ---------> |
  |                          |                          |-- LLM call
  |                          |                          |-- lookup_price (server)
  |                          |                          |-- read_excel_range -> PAUSE
  | <-- {run_id, status} --- |                          |
  |                          |                          |
  |-- GET /api/runs/{id}/events/stream (SSE) -------->  |
  | <-- run.paused (with pending_tool_calls) ---------- |
  |                          |                          |
  |-- POST /runs/{id}/tool-results ------------------>  |
  |                          |-- agent.submit_tool_results(...) --> |
  |                          |    (persist-first + CAS claim + resume)
  | <-- 200 {status: success, answer: ...} ------------ |
  | <-- run.completed (SSE)  |                          |
```

## Where each piece lives

| Surface | Owner | File |
|---|---|---|
| `GET /api/runs/...` + SSE | Dendrux (`make_read_router`) | `dendrux/http/read_router.py` |
| `POST /chat` | You | `server.py` |
| `POST /runs/{id}/tool-results` | You | `server.py` |
| `DELETE /runs/{id}` | You | `server.py` |
| Browser UI | You | `client.html` |
| Pause/resume orchestration | Dendrux | `agent.submit_tool_results` |
| CAS + race-safety | Dendrux | `runtime/submit.py` |

## If something looks wrong

- If the agent doesn't pause, your prompt didn't trigger `read_excel_range`. Try: "read cell A1 from my spreadsheet."
- If you see connection errors, make sure the server is running on port 8000.
- This demo uses a no-op `authorize` dependency (no auth). Real apps plug in a user-session dep there.
