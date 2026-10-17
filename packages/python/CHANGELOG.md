# Changelog

## Unreleased

Run control, per-call governance and subtree reads. This release includes a
security fix (see Changed) and a database migration (see Migration): run
`dendrux db migrate` before upgrading a Postgres deployment.

### Added

- **Per-call approvals.** `agent.submit_approval(run_id, decisions={call_id:
  ApprovalDecision.approve() | ApprovalDecision.deny(reason)})` decides each
  approval-gated call on its own. Pass exactly one of `approved=` and
  `decisions=` (otherwise `ValueError`); `rejection_reason` is not allowed with
  `decisions`. Decision ids must be exactly the gated call ids, validated against
  an `approval_gates` snapshot persisted at pause time, so another `Agent`
  instance can decide. A denied call gets `{"denied": reason}`; held non-gated
  server calls execute automatically. `approval.decided` keeps its scalar
  `decision` (`approved|rejected|mixed`, `run_id`) and adds `decisions`: one
  entry per pending call with `approved|denied|not_gated_executed|client_pending`.
  For server-target calls, the batch forms (`approved=True/False`) preserve their
  legacy approval and rejection payloads; client-target routing changes as
  described under Changed. `ApprovalDecision` is exported from `dendrux`.
- **`before_tool` policy hook.** `Agent(before_tool=hook)` (sync or async) is
  called with a `ToolCallContext` (`run_id`, `call_id`, `tool_name`, deep-copied
  `params` and `meta`, `target`, `source` of `local|mcp|skill`, `iteration`,
  `tenant_id`) once for each parsed tool call that passes static `deny=` and
  skill preflight, before approval, in `run` and `stream`, and returns a `ToolDecision`: `allow()`,
  `deny(reason)`, `require_approval(reason)`, `substitute(result)` or `None`.
  `require_approval` on a client-target call is denied with a
  `policy.hook_warning`. It fails closed: a hook that raises, returns a
  non-decision, or substitutes a non-JSON value denies the call
  (`policy.hook_error`, exception type only). It is not re-run on resume and not
  supported with `SingleCall` (`ValueError`). New events: `policy.substituted`,
  `policy.hook_error`, `policy.hook_warning`; `policy.denied` and
  `approval.requested` gain `source="hook"` and a reason. A substituted call
  writes a `tool_calls` row with `meta.substituted = true`. `ToolCallContext`,
  `ToolDecision` and `BeforeToolHook` are exported from `dendrux`.
- **Durable pause.** New `RunStatus.PAUSED` (`"paused"`).
  `await agent.request_pause(run_id, *, cascade=True, tenant_id=..., reason=None)
  -> PauseOutcome` flags a running run; it pauses at the top of its next
  iteration (or between the approved calls of a resumed batch), writes
  `run.paused` with `reason="requested"`, and `run()` / streams return a paused
  result. `await agent.submit_resume(run_id)`, `agent.resume(run_id)` and
  `agent.resume_stream(run_id)` continue it. `cancel_run` finalizes a paused run;
  `sweep(abandoned_waiting=...)` includes it. `RunDetail.pause_requested` mirrors
  the flag. Cancel wins over pause. `cascade=True` is flag propagation only,
  not a resumable call stack: a paused child returns a paused result to the
  delegating tool and resuming it never rewrites the parent's recorded tool
  result. `PauseOutcome` is exported from `dendrux`.
- **Cascade cancel.** `cancel_run(run_id, *, cascade=False, tenant_id=...)`;
  `cascade=True` cancels non-terminal descendants over a point-in-time snapshot,
  deepest first, `run_id` last, with per-run outcomes in `meta["cascade"]`.
- **Inbox notes.** `await agent.submit_note(run_id, text, *, metadata=None,
  tenant_id=...) -> note_id` leaves a note for a non-terminal run. The runner
  delivers pending notes at the top of its next iteration as a user message
  `NOTE FROM USER:\n{text}` and writes `inbox.delivered` (note id only). Delivery
  is at-least-once. Text is limited to 32 KiB and metadata (never shown to the
  model) to 8 KiB. Notes a finished run never delivered are listed in
  `RunResult.meta["undelivered_notes"]` and readable with
  `RunStore.get_notes(run_id, undelivered_only=, tenant_id=)` (`RunNote`).
  `SingleCall` never delivers notes.
- **Heartbeat.** `dendrux.heartbeat()` (sync, thread-safe, fire-and-forget) and
  `await dendrux.aheartbeat()` refresh persisted progress from inside a
  long-running tool to reduce the risk of `sweep(stale_running=...)` reaping it.
  Writes are best-effort and failures are logged. Tool start now also touches
  progress. Both are no-ops outside a persisted run.
- **Dollar budget.** `Budget(max_cost_usd=..., on_exceed="warn"|"pause"|"cancel",
  include_children=True)`; `max_tokens` is now optional (at least one limit is
  required). `on_exceed="pause"` pauses durably (`pause_reason="budget"`);
  `"cancel"` ends the run `cancelled` with `failure_reason="budget_exceeded"`.
  With `include_children` the check uses the live root-inclusive subtree rollup.
  Events carry `metric` (`tokens|cost_usd`); `budget.exceeded` adds `action`; new
  `budget.unpriced` fires once per run when the cost limit meets an unpriced call
  (unpriced calls are excluded from the dollar comparison and reported by `budget.unpriced`).
- **Subtree reads.** `RunStore.get_subtree(run_id, *, tenant_id=)` returns a
  `RunSubtree` (`nodes` with depth and per-node `RunUsage`, plus rolled-up
  `usage`) computed live from `llm_interactions`, so running and paused runs are
  counted. `RunStore.get_events(event_types=, before=)`, and `before=` on
  `get_llm_calls` / `get_tool_invocations` (timezone-aware, exclusive).
- **`MCPConnection.call_tool(name, arguments=None, *, timeout=None)`** runs a
  server tool directly through the managed call path (lease, capacity, 401
  re-auth, telemetry, redaction, error mapping). Unknown names raise the new
  `MCPToolNotFoundError` (error code `tool_not_found`). It bypasses all Agent
  governance, persistence, budgets and tenant authorization; never expose it to a
  model or untrusted client.
- **`ContextLengthExceededError`** (`dendrux`, `dendrux.errors`) for
  context-window overflows from OpenAI-compatible endpoints, batch and streaming,
  with `status_code`, `provider_code` and `model`.
- New governance event types: `policy.substituted`, `policy.hook_error`,
  `policy.hook_warning`, `budget.unpriced`, `inbox.delivered` (25 in total).
- `tenant_id=` on `cancel_run`, `request_pause`, `submit_note`, `get_subtree` and
  `get_notes` scopes the call to one tenant (omitted means unscoped).
- Docs: new Tool policy hook and Inbox notes pages; rewritten Budget; Approval,
  Pause and resume, Cancellation, MCP, Models, State persistence, RunStore, Agent,
  HTTP API and Errors updated. New example `examples/36_run_control.py` (mock
  provider, no API key).

### Changed

- **Security fix: client-target calls are no longer executed server-side on
  approval resume.** Previously an approved batch executed every pending call on
  the server, including calls to tools declared `target="client"`. Each pending
  call is now routed by its persisted target: server calls (approved or
  non-gated) execute, client-target calls stay pending and the run moves to
  `waiting_client_tool`. This applies to the batch and per-call forms. If you
  relied on approval running client tools on the server, expose them as server
  tools.
- **`run()` is interruptible by `cancel_run`.** `agent.cancel_run(run_id)` on the
  same `Agent` instance and process now cancels the in-flight model call or tool;
  `run()` returns `RunResult(status=CANCELLED)` with `meta["interrupted"] = True`
  and `meta["partial_output"] = False` instead of finishing the call first.
  Cancelling the awaiting task yourself still raises `CancelledError`.
  Interrupted tools receive `CancelledError`.
- **Delegated runs inherit the parent's tenant** and reject a different one: a
  child started with a `tenant_id` other than its parent's raises `ValueError`
  before the run is created. Child runs previously stored an unset tenant as
  untenanted.
- **`StateStore` protocol additions** (all optional for custom stores):
  `get_run_control`, `request_pause`, `try_pause_run`, `pause_with_event`,
  `finalize_with_event`, `add_note`, `get_notes`, `mark_notes_delivered`,
  `get_subtree`, `subtree_usage`, `run_usage`, plus `event_types=` / `before=` /
  `total_usage=` / `failure_reason=` keywords. Pre-0.3 custom stores keep working
  for existing flows through legacy fallbacks; `request_pause`, `submit_note`,
  `get_notes` and cascade operations raise `NotImplementedError` naming the
  missing method.
- **Budget:** a fired threshold no longer fires again after a pause and resume
  (fired `(metric, fraction)` keys are persisted in the pause); pause and cancel
  act at the next top-of-iteration checkpoint, so a run can overshoot by up to
  one iteration; `SingleCall` degrades pause/cancel to warn. Budget resume
  continues only if the resuming Agent's limits are strictly above current usage
  for every exceeded metric, otherwise it pauses again immediately. Event
  payloads gained `metric`.
- **Paused runs show running totals** (`total_*` tokens and `total_cost_usd`)
  instead of zero until they finish.
- A stale `running` row with a cancel pending is swept as `cancelled`, not
  `error`. `sweep(abandoned_waiting=...)` includes `paused` runs.

### Fixed

- OpenAI-compatible gateway (`OpenAIProvider`, `OpenRouterProvider`):
  - A streaming request failing with any 400 was reported as "streaming
    unsupported". Only a 400 that actually says streaming is unsupported (or whose
    `param` is `stream`) is mapped now; other errors surface unchanged.
  - Tool calls buffered by a backend that ends a stream without a `finish_reason`
    are flushed at end of stream instead of being lost.
  - Context-window overflows are detected for batch calls, stream creation and
    in-stream errors and raise `ContextLengthExceededError`; rate-limit, quota and
    auth errors that mention tokens are never classified as overflows.

### Migration

- New Alembic revision `b5c6d7e8f9a0` (down revision `a4b5c6d7e8f9`): adds
  `agent_runs.pause_requested` (boolean, not null, default false) and the
  `run_inbox` table (foreign key to `agent_runs` with `ON DELETE CASCADE`, index
  on `run_id, delivered_at, created_at, id`). Run `dendrux db migrate` for
  Postgres and other migration-managed databases. Downgrade drops both.
- `RunStatus.PAUSED` is a new status string in an existing varchar column; no
  enum migration. Code that switches over statuses should handle `paused`.
- SQLite databases created by an earlier release are upgraded in place on
  startup. The upgrade is add-only: missing nullable or defaulted columns and
  indexes are added. A column that cannot be added safely raises an error naming
  the table and column and telling you to run `dendrux db migrate` or recreate
  the file. Postgres still requires `dendrux db migrate`.

### Known limitations

- A cancel that arrives after a run's last checkpoint does not stop a run that
  then completes successfully; the request is cleared.
- `sweep(stale_running=...)` must be longer than the longest gap between progress
  updates (LLM calls, tool starts and ends, `heartbeat()`); otherwise the event
  sequence indexes of a still-running run can collide with the sweep's event.
- Pause cascade is flag propagation, not a resumable call stack. Resuming a child
  does not rewrite the parent's recorded tool result.
- Budget pause and cancel take effect at the next top-of-iteration checkpoint,
  so a run can overshoot by up to one iteration. They degrade to warn on
  `SingleCall`.
- Inbox delivery is at-least-once, and a note accepted while a run makes its last
  LLM call may never be delivered (it is reported as undelivered).
- The MCP SDK's own DEBUG logging may echo raw tool names. Dendrux's logs, errors
  and telemetry scrub credential-shaped names.
- `RunStore.get_subtree` reads are best-effort and not one snapshot transaction.
- `cancel_run` interrupts only a run executing on the same `Agent` instance in
  the same process; elsewhere cancellation is cooperative.
- The `submit_*` methods (`submit_tool_results`, `submit_input`,
  `submit_approval`, `submit_resume`) take no `tenant_id`. HTTP routes must
  verify run ownership before calling them.

## 0.2.0a23 - 2026-10-15

- Runs started from a stream consumer's own loop body are no longer linked as delegation children of the stream run; this previously happened because delegation context leaked between yields.
- Skills v2. A skill can now carry files and scripts, be built from code, and be
  activated per run. Dendrux still never executes skill code itself.
  - `Skill` is validated and immutable, and is exported from `dendrux`. Build it
    directly, with `Skill.from_markdown(text, files=, revision=)` or
    `Skill.from_dir(path, revision=)`; `skill.to_markdown()` writes canonical
    `SKILL.md`. New fields: `files` (text as `str`, binary as `bytes`, anything
    under `scripts/` is a script), `allowed_tools`, `revision` (a non-secret
    audit label) and `digest` (`sha256:...`, content only). Invalid input
    raises `SkillValidationError` (a `ValueError`). `path` is now optional and
    keyword-only; after frontmatter parsing removes exactly one leading newline,
    the remaining body is kept without further stripping.
  - New built-in tools `read_skill_resource` and `run_skill_script` join
    `use_skill`. All three are reserved names, can be named in `deny=` and
    `require_approval=` (even on an agent with no other tools), and skip
    `tool_result_transform`. `deny=["use_skill"]` turns off model-driven
    activation.
  - `Agent(skill_executor=...)` takes a `SkillExecutor` subclass
    (`execute(request) -> SkillExecutionResult`, `timeout_seconds`) that you
    write. Without one, `run_skill_script` is not offered and a one-time
    warning names the skills whose scripts cannot run. `SkillExecutor`,
    `SkillExecutionRequest` and `SkillExecutionResult` are exported from
    `dendrux`.
  - `agent.run(..., activate_skills=[...])` and `agent.stream(...)` deliver
    chosen skill bodies in the first message. Unknown or denied names raise
    `ValueError` before a run is created. Activations are part of the
    idempotency fingerprint, and `agent.retry` keeps them.
  - Resume safety: every run records a manifest (name, revision, digest) in
    `agent_runs.meta["dendrux.skills"]`. `resume`, `resume_stream`,
    `submit_tool_results`, `submit_input` and `submit_approval` check it before
    claiming the run and raise `SkillMissingError` or `SkillMismatchError`
    (`dendrux.errors`, base `SkillContinuityError`), leaving the run paused.
    `resume_stream` ends with a `RUN_ERROR` instead. No migration.
  - **Breaking (alpha):** the governance events `skill.registered`,
    `skill.denied` and `skill.invoked` are replaced by seven events:
    `skill.catalog_resolved` (also lists denied names), `skill.activated`,
    `skill.activation_failed`, `skill.resource_read`, `skill.script_requested`,
    `skill.script_completed` and `skill.script_failed`. Payloads carry names,
    paths, counts and timings, never bodies, arguments or output. Update any
    code that filters on the old names.
  - Behaviour changes: a `use_skill` miss is now an error result (it used to be
    a successful call that still emitted `skill.invoked`); on SingleCall a skill
    with files raises `ValueError`; `agent.retry` now rejects a source run from
    another tenant with `ReplayRunError`; `skills_dir` problems surface at the
    first run, not at construction.
  - Docs: rewritten Skills page with local subprocess and E2B executor recipes,
    and the example in `examples/14_skills/` now shows files, a script executor,
    explicit activation and the skill events.

## 0.2.0a22 - 2026-10-14

- MCP: `connection.tools()` gains `prefix`, `rename`, and `strict` (also on
  `MCPToolPolicy`). `prefix=False` exposes bare tool names; `rename` maps a
  server tool name to the exposed base name; both are validated when the view
  is created. `ToolDef.meta["exposed_name"]` is the Agent-visible name; calls
  and the stale-tool check still use `meta["mcp_tool_name"]`, and call
  telemetry reports the exposed name. `MCPServer` and `MCPHost` are unchanged.
- MCP: behaviour change for runtime views. A tool with an unsafe or duplicate
  exposed name, or an `allowed_tools` / `force_serial_tools` / `rename` entry
  the server does not list, is now skipped instead of failing the whole source.
  Skips are reported as `MCPSkippedTool` entries (reason typed as
  `MCPSkipReason`) in
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
- MCP fix: a failed call through `MCPServer` or `MCPHost` now names the tool by
  its exposed, sanitized name (`srv__get_echo`), not the raw server name
  (`srv__get.echo`).
- MCP fix: internal OAuth token, token-response and rejection carriers now
  have redacting `repr`s, so a stray log line or traceback never prints a
  secret.

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
