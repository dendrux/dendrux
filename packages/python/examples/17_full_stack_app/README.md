# Example 17 — Full-Stack Dendrux App

The canonical recipe. One FastAPI process with:

- Dendrux's read router (runs, events, SSE) mounted at `/api/dendrux`.
- Dendrux's dashboard mounted at `/dashboard`.
- Four dev-written write routes (`/chat`, `/runs/{id}/tool-results`, `/runs/{id}/approvals`, `DELETE /runs/{id}`) that call the four public Agent methods.
- One agent configured with all four governance layers — **deny, approval, budget, guardrails** — all visible live in the UI.
- Two providers (Anthropic / OpenAI) selectable in the browser.

If you read only one example, read this one.

## Run

```bash
cd packages/python
pip install -e ".[dev,db,anthropic,openai,http]"
export ANTHROPIC_API_KEY=sk-...
export OPENAI_API_KEY=sk-...
python examples/17_full_stack_app/server.py
```

Open <http://localhost:8000>. Default `X-App-Token: dev-secret-abc` (override with `DENDRUX_DEMO_TOKEN=...`).

## What Dendrux ships vs. what you wrote

| Layer | Owner | Lines |
|---|---|---|
| Read router (list / detail / events / drill-downs / SSE) | Dendrux | `dendrux.http.make_read_router` |
| Dashboard API + UI | Dendrux | `dendrux.dashboard.create_dashboard_api` |
| Pause/resume orchestration, CAS-safe claim | Dendrux | `Agent.submit_*` / `Agent.cancel_run` |
| Governance (deny / approval / budget / guardrails) | Dendrux | `Agent(...)` kwargs |
| `server.py` — routes, auth dep, app factory | **You** | 222 |
| `agent_setup.py` — tools, agents, governance config | **You** | 122 |
| `client.html` — UI, SSE dispatcher, Safety panel | **You** | 307 |

Your end-to-end code: **~650 lines** across three files, every one of them in the reader's face. No hidden framework magic.

## Feature tour

### 1. Two providers, one governance config

`agent_setup.py` defines one `_governance_kwargs()` function. Both `build_anthropic_agent` and `build_openai_agent` apply it identically. Swap providers at runtime by sending `"provider": "openai"` on any write request.

### 2. All four governance layers visible

| Layer | Configured as | Try it |
|---|---|---|
| Deny | `deny=["delete_account"]` | "Delete account user-99 right now." |
| Approval | `require_approval=["issue_refund"]` | "Refund order 4421 for $50." → approval card appears |
| Budget | `budget=Budget(max_tokens=20_000)` | Fires `budget.warned` at 75% / 90% |
| Guardrails — redact | `PII(action="redact")` | "Contact jane@acme.com" → email replaced with `<<EMAIL_ADDRESS_1>>` for the LLM; `contact_customer` tool receives the real address |
| Guardrails — block | `SecretDetection(action="block")` | "My AWS key AKIA..." → run terminates before the LLM is called |
| Guardrails — warn | Custom pattern `TKT-\d{6,}` (warn) | "Follow up on TKT-041729" → audit-only, message delivered unchanged |

Four try-it buttons in the UI pre-fill each case.

### 3. Deanonymization at the tool boundary

This is the unique guardrail story. When `PII(action="redact")` fires:

```
User input:  "Contact the customer at jane@acme.com"
                       │
                       ▼ (guardrail.redact)
LLM sees:    "Contact the customer at <<EMAIL_ADDRESS_1>>"
                       │
                       ▼ (tool call arg materializes)
contact_customer("jane@acme.com")   ← real value at the tool
                       │
                       ▼
LLM sees:    "<<EMAIL_ADDRESS_1>> has been contacted"
```

The LLM never sees the real email. The tool does. The audit trail is in `run.pii_mapping`.

### 4. All three pause types handled

- **Client-side tool** (`@tool(target="client")`) → `waiting_client_tool` → `submit_tool_results`.
- **Approval-gated server tool** (`require_approval`) → `waiting_approval` → `submit_approval(approved=bool)`.
- **Human input clarification** is *not* reachable from the built-in strategy; the route is omitted here. Reachable when you use a strategy that emits `Clarification` actions.

### 5. Dashboard mounted in the same app

```python
app.mount("/dashboard", create_dashboard_api(state_store))
```

One click from the UI header. Shows every run, its timeline, governance events, tool calls, LLM payloads — all fed from the same `run_events` table the live SSE drains.

### 6. Cancellation

A Cancel button in the header calls `DELETE /runs/{id}?provider=...`. That hits `agent.cancel_run(run_id)`, which:

- Cancels any in-process submit/resume task spawned by this Agent instance.
- CAS-finalizes the DB row to `CANCELLED`.
- Broadcasts `run.cancelled` on SSE.

## What's intentionally not in this example

- **Streaming (`agent.stream()`).** Guardrails require the full LLM response to scan and so can't fire during token streaming. The example prioritizes hyper-visible governance; for the streaming client-tool pattern without governance, see [example 09](../09_client_tools_streaming/). The streaming-plus-guardrails reconciliation is tracked as future work.
- **Token-level UI.** Would need a separate NDJSON endpoint (like example 09). Out of scope here.
- **Multiple agents / delegation.** One agent teaches the seam. See `examples/04_research_agent/` for delegation.

## Production checklist

Every `TODO(production)` marker in `server.py` maps to a box here. None of these are Dendrux's responsibility — they belong to your app, but you'll need them before deploying.

- [ ] **Real auth.** Replace the `X-App-Token` header with JWT / signed session cookies. For SSE, prefer cookies over query params — URL params leak into access logs.
- [ ] **Ownership stamping.** When calling `agent.run(...)`, pass `metadata={"owner_id": user.id}`. Every write route then fetches the run, compares `owner_id` to the authenticated identity, returns `404` on mismatch (not `403` — don't leak existence).
- [ ] **Rate limiting.** Middleware on `/chat`, `/runs/{id}/*`. Consider per-user-per-run budgets.
- [ ] **Request validation beyond happy path.** Size limits on `input` and `payload`, restrict tool_call_id shapes, reject unknown fields.
- [ ] **Audit logging.** Wrap every write route in a structured log (user, run_id, outcome) for compliance.
- [ ] **Error mapping.** Catch `dendrux.errors` explicitly and map to HTTP shapes your API conventions require. The demo lets them propagate as 500.
- [ ] **Session state.** Store `run_id` / pending approval in your own session layer, not browser memory. The demo uses a single page's JS state; real UIs persist across tabs.
- [ ] **Dashboard access.** `create_dashboard_api` accepts a separate `auth_token` parameter — this example shares it with the write routes for simplicity. Production may want a separate, tighter dashboard credential.
- [ ] **Idempotency.** Pass `idempotency_key=` on `agent.run(...)` for critical actions so retries don't double-fire refunds.
- [ ] **Run migrations.** `dendrux db migrate` before first Postgres boot. This demo uses SQLite with auto-create, not a migration story.

## File layout

```
17_full_stack_app/
├── server.py        # HTTP layer: routes, auth, app factory
├── agent_setup.py   # Agent layer: tools, agents, governance
├── client.html      # UI: SSE dispatcher, Safety panel, approval cards
└── README.md
```

Two Python files by design — `server.py` is "how HTTP plumbs into Dendrux," `agent_setup.py` is "what the agent is." Read them independently.
