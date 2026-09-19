# Changelog

## Unreleased

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
