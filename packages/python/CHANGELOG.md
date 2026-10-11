# Changelog

## Unreleased

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
