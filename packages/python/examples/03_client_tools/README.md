# Example 03: Client Tool Bridge

Demonstrates Dendrite's core differentiator — an agent that **pauses** when it needs a client-side tool, waits for the user to provide the result, then **resumes** reasoning.

## Setup

```bash
cd packages/python
pip install -e ".[dev,db,anthropic,bridge]"
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
1. Call `lookup_price("AAPL")` — executes instantly on the server
2. Call `read_excel_range(sheet="Sheet1", range="A1")` — **pauses**
3. The UI shows a form asking you to provide the result
4. Type any value (e.g. `Revenue: $394B`) and click Submit
5. The agent resumes with your data and generates a final answer

## Architecture

The developer owns run creation (`POST /chat` calls `agent.run()`).
The bridge handles everything after the first pause.

```
Browser                    Server                     Agent
  |                          |                          |
  |-- POST /chat ----------> |-- agent.run() ---------> |
  |                          |                          |-- LLM call
  |                          |                          |-- lookup_price (server)
  |                          |                          |-- read_excel_range -> PAUSE
  | <-- {run_id, status} --- |                          |
  |                          |                          |
  |-- GET /dendrite/runs/{id}/events (SSE) -----------> |
  | <-- snapshot (status=waiting_client_tool) --------- |
  |                          |                          |
  |-- POST /dendrite/runs/{id}/tool-results ----------> |
  |                          |-- submit_and_claim ----> |-- resume
  | <-- 200 (accepted) ----- |                          |-- LLM call (with result)
  | <-- run.completed (SSE)  | <-- RunResult ---------- |
```

## Bridge endpoints

| Endpoint | Purpose |
|---|---|
| `GET /dendrite/runs/{id}` | Poll status + pending tool calls |
| `GET /dendrite/runs/{id}/events` | SSE stream (snapshot + live events) |
| `POST /dendrite/runs/{id}/tool-results` | Submit client tool results |
| `POST /dendrite/runs/{id}/input` | Submit clarification answer |
| `DELETE /dendrite/runs/{id}` | Cancel a run |

## If something looks wrong

- If the agent doesn't pause, your prompt didn't trigger `read_excel_range`. Try: "read cell A1 from my spreadsheet."
- If you see connection errors, make sure the server is running on port 8000.
- This demo uses `allow_insecure_dev_mode=True` (no HMAC auth) — local development only.
