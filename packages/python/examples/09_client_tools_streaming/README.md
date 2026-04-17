# Example 09: Client Tool Streaming

Demonstrates Dendrux streaming with client-side tool execution using
`fetch()` + NDJSON — a single-request streaming variant of the
client-tool pattern shown in example 03.

## Why fetch() instead of SSE?

Browser `EventSource` only supports GET requests. Client tool resume
requires POST (to send tool results in the body), so SSE isn't viable
for the resume path. Using `fetch()` with a streamed response body
gives us POST + streaming on both endpoints with the same code path.

## Why NDJSON?

Newline-Delimited JSON (one JSON object per line) is the simplest
streaming format. No framing (`data:`, `event:`) needed — the same
format used by OpenAI and Anthropic streaming APIs.

## How client tools work

1. `POST /chat` starts a streaming run
2. Agent calls `read_excel_range` (a client tool) — run pauses
3. Stream emits `run_paused` with `pending_tool_calls` array
4. Browser shows the tool form, user provides the result
5. `POST /resume/{run_id}` resumes the run with tool results
6. Stream continues from where it left off

The `run_paused` event is the handoff signal. The browser reads
`pending_tool_calls` to know what the agent needs, then submits
results to the resume endpoint.

## Running

```bash
cd packages/python
ANTHROPIC_API_KEY=sk-... python examples/09_client_tools_streaming/server.py
```

Open http://localhost:8001

Try: *"Look up AAPL price, then read cell A1 from my spreadsheet"*

## Wire format

Each line is a JSON object with a `type` field:

| type | key fields |
|------|-----------|
| `run_started` | `run_id` |
| `run_resumed` | `run_id` |
| `text_delta` | `text` |
| `tool_use_start` | `tool_name`, `tool_call_id` |
| `tool_use_end` | `tool_name`, `tool_call_id` |
| `tool_result` | `tool_result.call_id`, `.name`, `.success`, `.payload` |
| `run_paused` | `run_result.status`, `run_result.pending_tool_calls` |
| `run_completed` | `run_result.status`, `run_result.answer` |
| `run_error` | `error` |

### Resume request

```
POST /resume/{run_id}
Content-Type: application/json

{
  "tool_results": [
    { "call_id": "...", "name": "read_excel_range", "result": "\"A1: Revenue\"" }
  ]
}
```

`result` is a JSON-encoded string. The response is an NDJSON stream
with the same event vocabulary — first event is `run_resumed`.
