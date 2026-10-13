# Changelog

## Unreleased

- MCP: `connection.tools()` gains `prefix`, `rename`, and `strict` (also on
  `MCPToolPolicy`). `prefix=False` exposes bare tool names; `rename` maps a
  server tool name to the exposed base name; both are validated when the view
  is created. `ToolDef.meta["exposed_name"]` is the Agent-visible name; calls
  and the stale-tool check still use `meta["mcp_tool_name"]`, and call
  telemetry reports the exposed name. `MCPServer` and `MCPHost` are unchanged.
- MCP: behaviour change for runtime views. A tool with an unsafe or duplicate
  exposed name, or an `allowed_tools` / `force_serial_tools` / `rename` entry
  the server does not list, is now skipped instead of failing the whole source.
  Skips are reported as `MCPSkippedTool` entries in
  `RunResult.meta["mcp_skipped_tools"]` (present whenever
  `mcp_skipped_sources` is) and in one `mcp.tools_skipped` governance event.
  A tool with an empty name is `name_unsafe` in both modes (it no longer loads
  as `namespace__`; `MCPServer` and `MCPHost` now reject it).
  `strict=True` restores raising `ValueError`. Collisions with another source,
  a local tool, or `use_skill` still raise, now as `MCPToolNameCollisionError`
  (a `ValueError` subclass, code `tool_name_collision`).
- MCP: stable `code` on every `mcp_skipped_sources` entry and on the
  `mcp.error` payload (`authorization_required` adds `reason`), derived from
  the exception class by the new `error_code_for(exc)`.

## 0.2.0a21 - 2026-10-13

- MCP: OAuth 2.1 sign-in for remote MCP servers. `MCPOAuth` runs discovery
  (RFC 9728, RFC 8414 / OpenID), client resolution (pre-registered, client ID
  metadata documents, RFC 7591 registration), PKCE S256 with the RFC 8707
  `resource` indicator and RFC 9207 `iss` checks, refresh, scope step-up, and
  RFC 7009 revocation. `begin_authorization()` and `complete_authorization()`
  may run on different workers; the flow is bound to the application's
  `MCPAuthKey`. Applications implement the four-method `MCPOAuthStore`
  (check it with `check_oauth_store`); `MCPOAuthMemoryStore` is the in-process
  reference. Tokens are delivered only to the exact canonical resource they
  were issued for, and an expired access token is never delivered.
- MCP: refresh is serialized across processes by a store refresh claim and
  exact-version compare-and-swap. A refresh token is redeemed at most once:
  interrupted, uncertain, rejected, or abandoned refreshes retire it instead of
  replaying it, and only pre-send transport failures keep it.
- MCP: `MCPRuntime.bind(oauth=)` / `rebind(oauth=)` bind an identity to a
  user's sign-in (exclusive with `credentials=`; exactly one `Authorization`
  header on the wire). 401s and 403 `insufficient_scope` rejections recover
  once with the exact rejected token; `refresh_authorization()` retires the
  transport after a step-up; `revoke_authorization()` fences, signs out, and
  evicts, and `close()` waits for it.
- MCP: new errors `MCPAuthorizationRequired` (`not_signed_in`, `expired`,
  `revoked`, `insufficient_scope`, `resource_mismatch`), `MCPOAuthError`,
  `MCPOAuthFlowError`, and `MCPOAuthRefreshError`; new events
  `MCPOAuthRequired`, `MCPOAuthRefreshed`, `MCPOAuthRefreshFailed`, and
  `MCPOAuthRevoked`. Sign-in verdicts and transient refresh errors never count
  toward the circuit breaker or mark credentials failed.
- MCP: a credential provider's `MCPCredentialError` or
  `MCPAuthorizationRequired` from `get_auth()` or `on_auth_rejected()` now
  propagates unchanged; other exceptions are still wrapped value-free.
- MCP: `MCPAuthKey`, `MCPCredentialProvider`, and
  `MCPRefreshingCredentialProvider` live in `dendrux.mcp._credentials`
  (still importable from `dendrux.mcp`). `rotate_credentials()` refuses OAuth
  bindings.
- MCP: `MCPAuthenticationError.challenge` carries the validated `Bearer`
  challenge of the rejecting 401/403 (`MCPAuthChallenge`: `scheme`, `error`,
  `scope`, `resource_metadata_url`) at connect, discovery, and tool-call time.
  Fields failing their RFC grammar are dropped, free text is never stored, and
  nothing from the header enters messages or `repr()`.
- MCP: with `reauthenticate_on_401=True` and a refreshing credential provider,
  a 401 from the source origin during connect or `tools/list` now calls
  `on_auth_rejected()` once and retries the connection once, instead of
  failing. A rejected operation notifies the provider at most once.
  Connect-time rejections also set `request_rejected`.
- MCP: a request an `auth=` flow builds itself for another origin (for example
  an OAuth token request with `Authorization: Basic`) keeps only the headers
  its flow introduced. Copied source headers, cookies, and redirects are still
  stripped.

## 0.2.0a20 - 2026-10-12

- Tools: a tool call whose arguments are not valid JSON (or are JSON but not
  an object) is now a failed tool call, not a failed run, in both
  `agent.run()` and `agent.stream()`. The call is rejected before deny,
  approval and parameter validation, with `code: "invalid_json"` and the
  decoder error in the message, so the model can retry. Previously batch runs
  raised `ValueError` (with the raw arguments in the message) and streams
  passed `{}` on to a misleading "missing required argument(s)" error.
  `ToolCall` gains `raw_arguments` (in memory only) and `parse_error`; traces,
  pause state and the `tool.completed` event record `parse_error` and
  `raw_length`, never the raw text.

## 0.2.0a19 - 2026-10-11

- Runs: `agent.run()` / `agent.stream()` accept `runs=[...]` (ordered run ids
  of the conversation branch), `replay_last=0 | N | "all"` and
  `replay_reasoning=True` to replay earlier runs into the new request: text
  only by default, or tool calls + results (+ reasoning) for the most recent
  runs. Reasoning replays only when a run's recorded prefix (provider, model,
  system prompt, tools, seed) matches what precedes it now; from the first
  mismatch onward it is dropped with a warning. Tool calls replay verbatim even
  when the tool is no longer offered (warning). `RunResult.replay`
  (`ReplaySummary`) and `agent_runs.meta["dendrux.replay"]` record what was
  replayed and why; `ReplayRunError` is raised before any model call for
  unknown, foreign-tenant or non-terminal run ids. `retry()` re-expands the
  same runs. Guardrails force reasoning replay off.
- Anthropic: thinking / redacted_thinking blocks are replayed on assistant
  turns across tool-loop steps (only when thinking is on for the request);
  replayed thinking sends `thinking.block_binding.prefix_mismatch_behavior=
  "drop_block"` with the `thinking-binding-controls-2026-08-01` beta header;
  an extra cache breakpoint lands on the last replayed message.
- OpenAI Responses: reasoning items are replayed before that turn's
  function calls (null fields dropped); `phase` ("commentary" |
  "final_answer") is resent on assistant input items. Both OpenAI providers
  take a constructor `cache_key_prefix`; `prompt_cache_key` is sent only when a
  prefix is known and is never derived from the run id.
- OpenRouter: the context-compression plugin is disabled by default (a dev
  `plugins` entry wins); `anthropic_cache=True` sends OpenRouter's top-level
  automatic `cache_control` for `anthropic/` models.
- Reasoning blocks and the producing model are persisted in
  `react_traces.meta`, so `retry()` keeps a tool loop's reasoning.
- Streams / results: assistant text is tagged `commentary` (narration in a
  step that ended in tool calls) or `final_answer`. `RunResult.commentary`
  lists the narration per step; `RunEvent` carries `iteration` and `phase`
  (`phase` is set on tool events, on `RUN_COMPLETED`, and on `TEXT_DELTA` when
  the provider labels it natively); `TraceEntry.phase` on persisted traces.
- Tools: schema generation supports `Literal` / `Enum` (→ `enum`),
  `Annotated[T, Field(description=...)]` and Google-style `Args:` docstrings
  (per-parameter descriptions), pydantic models / dataclasses / `TypedDict`
  (nested objects, `$defs` inlined), `dict[str, T]`, and nullable optionals
  (`["string", "null"]`). `@tool(parameters=...)` passes an explicit schema;
  `@tool(strict=True)` normalizes it for OpenAI strict function calling and
  OpenAI providers send `"strict": true`.
- Tools: tool-call arguments are validated against the tool's schema before
  execution (unknown names, missing required, basic types, `enum`, nullable
  type arrays; tolerant of `anyOf` / `$ref`). Invalid calls never run the tool
  and return a model-facing error (`code: "invalid_params"`), recorded as a
  failed invocation. Applies to local, MCP and approval-resumed calls.
- Tools: `Agent(tool_result_transform=...)` rewrites the result of every
  successful server-executed tool call (local `@tool` and MCP alike, including
  tools run after `submit_approval`) before it is serialized. The transformed
  value is what gets persisted, notified, and sent to the model. Receives a
  frozen `ToolResultContext` (tool name, call id, params, raw result, target,
  `ToolDef.meta`); sync or async. A raising transform fails the call. Not
  applied to failed calls or to `submit_tool_results`.
- Runs: concurrent `cancel_run()` calls on the same paused run both report
  `cancelled` (idempotent-cancel race fixed).
- Docs: "Memory across runs" recipe, "Tools" reference, commentary / phase
  contract, OpenRouter caching + compression notes; the app-database-boundary
  recipe now stores `result.answer` instead of concatenating every
  `TEXT_DELTA`.

## 0.2.0a18 - 2026-10-01

- Runs: task cancellation of `agent.run()` now CAS-finalizes a running row as
  cancelled, emits `run.cancelled`, closes run lifecycle hooks, and re-raises
  cancellation, including timeouts during startup and error handling. Durable
  cancellation cleanup survives repeated cancellation; closing lifecycle hooks
  are invoked at most once per recorder/notifier, in the original run task.
- Streams: accept `idempotency_key` with the same request fingerprint as `run()`.
  Cached outcomes emit one terminal event without replaying text or tool events;
  active matches and conflicts emit `RUN_ERROR` without modifying existing runs.
  The stream ID resolves during first iteration when a key matches an existing run.
- Docs: clarify final-turn `RunResult.answer` and the partial-answer buffer reset
  at each `TOOL_RESULT` on stream cancellation.
- Pricing: `Agent(pricing=PriceTable({...}))` fills `cost_usd` for providers
  that report tokens only (Anthropic, OpenAI). Rates are developer-owned, USD
  per million tokens, keyed by model id or glob; a provider-reported cost
  (OpenRouter) always wins. `UsageStats.cost_source` says whether a cost is
  `"provider"`, `"table"`, or `"mixed"`; it rides `llm.completed` events and
  `token_usage.meta`, no migration.
- Usage: a run total `cost_usd` is now `None` when any step is unpriced,
  instead of silently summing only the priced steps.
- Preserve unknown pricing when providers omit usage or unpriced calls report
  no fresh tokens, including across pause/resume. Idempotent retries now retain
  the final usage summary and its cost source on SQLite and PostgreSQL.

## 0.2.0a17 - 2026-09-20

- MCP: require SDK 2.2+, adopting its origin-restricted redirect handling;
  preserve per-source redirects-off behavior and report blocked targets as
  structured origins without contacting them.

- Runs: preserve discovery metadata on errors and stream-closed cancellations,
  expose `RunStream.result` after closure and `run_id`/`run_meta` on blocking
  execution errors. An empty skipped-source list means discovery completed;
  an absent key means its outcome is unavailable.

- MCP: opt-in `reauthenticate_on_401` with `MCPRefreshingCredentialProvider`
  rejection notification, shared refresh/reconnect, and one retry of a
  definitively rejected tool POST. Live 401/403 errors are typed per request.
- MCP: `MCPRuntime.rotate_credentials()` drains calls and replaces credentials
  without invalidating handles, reconnecting lazily with catalog revalidation
  or explicit same-permissions catalog preservation.
- MCP: structured `MCPOrigin` diagnostics on connection errors; URL-free
  transport diagnostics and wrapped transport tool errors.

## 0.2.0a16 - 2026-09-19

- MCP HTTP: per-source `follow_redirects` and `max_redirects` settings on
  `MCPSource.http()` and `MCPServer()`; redirects can be disabled entirely.
- MCP HTTP: configured headers and authentication-generated credential headers
  are scoped to the original scheme, hostname, and effective port before every
  request, including redirect hops and authentication retries.
- MCP HTTP: public async `MCPDestinationPolicy` and immutable `MCPDestination`
  allow application-owned IP policy to be enforced before every new TCP socket.
  Approved numeric IPs are dialed directly while preserving TLS hostname
  verification. DNS changes, redirects, and reconnects cannot bypass the check.
- MCP HTTP: `MCPDestinationDeniedError` distinguishes denied destinations during
  connect/discovery. Policy exceptions fail closed without leaking callback text.
  Policy-enabled connections disable environment proxies and TLS overrides.
- MCP runtime: destination policy identity participates in connection configuration
  checks; use `rebind()` to replace policy and retire existing sockets.
- Regression coverage includes real HTTP redirects, DNS changes between pooled
  connections, denied private IPs, SDK reconnects, and real TLS hostname verification.

## 0.2.0a15 - 2026-09-19

- MCP: `await connection.discover()` returns an `MCPDiscovery` (server
  facts plus the raw `MCPToolInfo` catalog) through the managed runtime's
  normal lease path, without an Agent. The connection stays warm for
  `idle_timeout` afterwards.
- MCP: `await runtime.rebind(...)` replaces a live connection's
  configuration in one step (validate, evict, bind). `bind()` documents that
  every `MCPSource` field except `name` — timeouts, `max_result_bytes`,
  `failure_mode` included — is configuration.
- MCP: the `mcp.error` governance event gains `error_type`; best-effort
  runs list skipped sources on `RunResult.meta["mcp_skipped_sources"]`
  (`source_name`, `namespace`, `error_type`, `error`).
- MCP: a server that rejects the source's credentials with HTTP 401/403
  now raises `MCPAuthenticationError` (with `status_code`) on SDK 2.x
  Streamable HTTP, where the SDK reports the rejected initialize as a
  generic JSON-RPC error. The adapter classifies from the HTTP response it
  observed. `transport_detail` for a task-group failure now names the
  wrapped error instead of "unhandled errors in a TaskGroup".
- MCP: `MCPRuntime(close_timeout=2.0)` bounds transport close separately
  from `shutdown_timeout`. A Streamable HTTP close sends a
  session-terminating `DELETE`, which routinely exceeded the previous fixed
  250 ms grace and was logged as "exceeded the shutdown grace".
- `agent.cancel_run()` now interrupts an in-flight model stream for runs
  driven by `stream()` / `resume_stream()` on the same `Agent` instance,
  including while waiting for the next token. The stream ends with
  `RUN_CANCELLED`; `answer` holds the text streamed so far and is persisted
  on the run row. `run.cancelled` gains `interrupted` / `partial_output`.
- Closing a `RunStream` early now persists the partial answer
  (`run.cancelled` with `reason: stream_closed`) and closes the loop
  generator deterministically in the caller's task.
- `cancel_run()` returns the state at the time of the request; the result's
  `meta["cancel_requested"]` and `RunDetail.cancel_requested` distinguish
  "requested" from "stopped".
- Loops: `run_stream()` accepts an optional `interrupt: asyncio.Event`.
  Custom loops that override `run_stream` should accept the kwarg.
- Fixed a `ValueError` ("Token was created in a different Context") logged
  when a stream was abandoned mid-provider-call.
- The interrupt race keeps the provider generator in the loop's own task
  (the same cancel-and-uncancel pattern as `asyncio.timeout()`), so
  ContextVars providers set inside their stream generators reset cleanly.
  Provider generators are never advanced from a helper task.
- Closing a `RunStream` never raises out of the loop generator's cleanup;
  cleanup failures are logged and the run is still finalized `cancelled`.

## 0.2.0a14 - 2026-08-15

- Add the production managed MCP client runtime, including lazy shared
  connections, tenant partitioning, explicit tool views, idle retirement,
  bounded connection and call capacity, eviction, recovery, and circuit
  breaking.
- Add lazy credential providers with credential-safe errors and typed MCP
  authentication failures.
- Add process-local MCP snapshots and lifecycle observer events.
- Publish the managed runtime, typed errors, and observability types from
  `dendrux.mcp`.
- Add runtime-first documentation and a multi-user MCP example.
- Build and exercise wheel and source distributions in isolated environments
  against a real MCP SDK v2 stdio server before release.
