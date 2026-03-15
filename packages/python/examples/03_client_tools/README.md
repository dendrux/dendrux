# Example 03: Client Tool Bridge

Demonstrates Dendrite's core differentiator — an agent that **pauses** when it needs a client-side tool, waits for the user to provide the result, then **resumes** reasoning.

## Setup

```bash
cd packages/python
pip install -e ".[dev,db,anthropic,server]"
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

## What's happening under the hood

```
Browser                    Server                     Agent Loop
  │                          │                            │
  ├─ POST /dendrite/runs ──> │ ── run() ──────────────>   │
  │                          │                            ├─ LLM call
  ├─ GET  /events (SSE) ──> │                            ├─ lookup_price (server)
  │  <── run.step ────────── │                            ├─ read_excel_range → PAUSE
  │  <── run.paused ──────── │                            │
  │                          │                            │ (waiting)
  ├─ POST /tool-results ──> │ ── resume() ────────────>  │
  │                          │                            ├─ LLM call (with result)
  │  <── run.completed ───── │ <── RunResult ────────────  │
```

## If something looks wrong

- If the agent doesn't pause, your prompt didn't trigger `read_excel_range`. Try being explicit: "read cell A1 from my spreadsheet."
- If you see connection errors, make sure the server is running on port 8000.
- This demo uses `allow_insecure_dev_mode=True` (no HMAC auth) — intended for local development only.
