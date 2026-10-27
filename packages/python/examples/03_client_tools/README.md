# Example 03 — Client tools

All HTTP routes and the browser UI belong to this application. Dendrux provides
Agent methods and RunStore reads. The agent combines a server price tool and a client spreadsheet tool.

```bash
cd packages/python
pip install -e ".[dev,anthropic]" fastapi 'uvicorn[standard]'
export ANTHROPIC_API_KEY=...

python examples/03_client_tools/server.py
```

Open http://localhost:8000. The default demo token is `dev-secret-abc`; set
`DENDRUX_DEMO_TOKEN` and enter the new value in the browser to change it.
Every API request sends `X-App-Token` as a header. Credentials never enter URLs.
The token represents one actor who owns the dedicated `demo-user` tenant.

| Route | Library call |
| --- | --- |
| POST /chat | Agent.run(..., tenant_id=...) |
| POST /runs/{id}/tool-results | Agent.submit_tool_results(..., tenant_id=...) |
| GET /runs/{id}/pending | RunStore.get_pending(..., tenant_id=...) |

The start and submit requests wait until a pause or terminal result. The client
then fetches pending state to rebuild forms. It sends call IDs and names with
tool results. Only
`run_id`, `status` and the answer intended for this actor leave the result DTO;
metadata, traces, provider bodies, arbitrary output and PII mappings stay server-side.
Client-tool params may contain real personal data. Tool outputs are client assertions.

This demo omits a live route. For optional `RunStore.follow`, see
[Build your own read routes](../../../../docs/recipes/build-your-own-read-routes.mdx).
The feed's iteration deadline cannot bound blocked network sends: the HTTP
transport needs its own send deadline and `finally: await follow.aclose()`.
Do not stream `agent.stream()` directly to a browser: a disconnect cancels it.

Before production, implement real identity, user-level run ownership within each
tenant, app audit records, rate limits and request
budgets. UI callbacks are UX only. The in-memory run/provider association must move
to your application DB to survive server restarts. Persist an app request ID before
starting work; a request timeout before the first response may lose the run ID and
cancellation may cancel the run. Own any background execution and recovery lifecycle.
