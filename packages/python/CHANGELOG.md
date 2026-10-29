# Changelog

## Unreleased (0.6.0)

"Production", part 1: a run survives a dead worker (`agent.reclaim()`,
`sweep(on_stale="reclaim")`, `@tool(idempotent=True)`), a Postgres event
notifier for live reads, published scale numbers with a rerunnable load script,
one trace per run and a log-codes page, and the 0.5 FutureWarning flips.
Entries are added as each track merges.

- guardrails redact values inside placeholder-shaped text the run did not issue

- For format-2 runs, `RunStore.get_traces` and `dendrux traces` return the
  provider form (PII placeholders); the reverse mapping is available as before.

- State-store subclass upgrade: overrides of `SQLAlchemyStateStore` write methods
  must accept and forward the new keyword-only parameters (a `**kwargs` passthrough
  is supported), or explicitly declare only the capabilities they implement.
  Validation now runs when constructing an `Agent`; incompatible inherited
  capability declarations raise `StoreCapabilityError` before execution.
- Results from reclaimed runs include `RecoveryInfo` and an additive `recovery`
  wire field with the reclaim count, last reclaim time and bounded replayed-call ids.
  `agent_runs.strategy` retains its strategy name; recovery classifies the loop
  using `loop_state.loop`.

- Format-2 persisted runs require JSON-only provider options of at most 2 KiB
  and at most 64 pending calls per assistant batch (configurable through
  `Limits.max_pending_calls`). Checkpoints use a 32 KiB compact UTF-8 JSON bound
  on both databases. Storeless and format-1 runs retain their previous limits.

## 0.5.0 - 2026-10-28

"Trust". Pending and live reads require a tenant scope and hide payloads by default; provider
failures are typed, every queue and payload has a bound, paused runs are
fenced and bound to a pause id, and the read router and embedded dashboard are
gone in favour of read primitives on `RunStore`. Run `dendrux db migrate` once,
as a deploy job, before starting 0.5 workers. See
[Upgrading to 0.5.0](https://dendrux.dev/docs/upgrading-to-0.5.0) for an action
for every change below.

### Highlights

- **Explicit scopes for pending and live reads.** Every `RunStore` read takes
  `tenant_id=`; `get_pending` and `follow` require it. Trusted historical reads
  default to `UNSCOPED` (all tenants); pass a tenant to narrow them. `Agent(strict_tenancy=True)`
  makes every entry point require one. Idempotency keys are unique per tenant.
- **Safe live and pending reads.** `RunStore.get_pending`, `follow`,
  `get_event_view` and `redact_event` replace the router. Redacted views show
  only enumerated codes, generated ids, numbers and declared names; the active
  pause stays actionable.
- **Typed provider errors and deadlines.** Silent empty answers are errors,
  `ProviderError` subclasses carry a category and a safe message, and calls have
  deadlines.
- **Provider response limits.** `ProviderResponseError` reports empty normal
  completions and local stream-limit failures with category `rejected`.
  `max_stream_bytes` defaults to 16 MiB of retained fragments plus small event
  charges; SDK snapshots are not charged again. `max_stream_tool_calls`
  defaults to 1,000. Overflow closes upstream; tool-only responses and
  token-limit truncation remain supported.
- **Delegation bounds.** `Budget.max_subtree_nodes` defaults to 10,000;
  incomplete accounting fails closed. Cascade cancellation signals pages of
  descendant ids. Its bounded detail uses the `cascade_count` and
  `cascade_truncated` metadata keys to report how much was processed.
- **Explicit wire omission.** `answer_omitted` distinguishes an answer omitted
  by the wire display bound from a real empty answer, within wire version 1.
- **Run integrity.** Claims are fenced by an epoch, every pause has a
  `pause_id`, and approving requires `expected_pause_id`.
- **Bounded everything.** Input, metadata, notes, tool results, MCP discovery,
  followers, the tool queue and `sweep()` all have limits with overrides.
- **Scale.** Equal `database_url` Agents share one reference-counted pool,
  non-SQLite pools default to pre-ping and recycling, and a
  [scaling guide](https://dendrux.dev/docs/guides/scaling) covers PgBouncer,
  the migration job and sweep cadence.

### Breaking changes

Each item links to a before/after note on the upgrade page.

- `make_read_router`, `dendrux.http`, the embedded dashboard, the
  `dendrux dashboard` command and the `http` extra are removed (see Removed).
- Approving a pause requires `expected_pause_id=`; without it
  `submit_approval(approved=True)` raises `InputValidationError`.
- `Budget(max_cost_usd=...)` on a model with no price raises
  `BudgetConfigurationError` at construction and at run start.
- Provider failures raise `ProviderError` subclasses, not SDK, `TimeoutError`
  or `ConnectionError` instances; a run that ended `success` with an empty
  answer now ends `error`. New deadlines: `call_timeout=600` and
  `stream_idle_timeout=600`.
- Local `@tool` arguments are validated and coerced; model-typed parameters
  receive model instances.
- Metadata keys starting with `dendrux.` are rejected.
- `RunNotFoundError` takes a run id and is one class everywhere; `retry()` and
  `submit_*` raise it for missing and foreign runs.
- `RunStore` and `StateStore` `list_runs` / `count_runs`: `tenant_id=None` means
  untenanted runs only; omit it or pass `UNSCOPED` for all tenants.
- Size caps: `ValueError` for oversized input text, metadata and notes;
  oversized tool results are replaced by a `size_limit` result.
- Guardrail events and errors carry kinds, counts and locations, never matched
  text; unscannable content blocks under `block` and `redact`. A guardrail-blocked
  run ends with `run.error`, not `run.completed`.
- Custom `StateStore`s: `save_run_event` allocates and returns the index
  (`EventSequencer` is removed); claims, epochs and sweeps have new methods.
- Redacted views hide free text, past tool call ids and tool names they cannot
  prove; `get_pauses` hides past arguments and `user_input` by default.
- `Agent.get_result(include_pii=False)` leaves `pii_mapping` out;
  `RunResult.meta` has no `pause_state`.
- Read flags and control booleans must be real `bool` values; reads validate ids,
  cursors, filters and dates and raise `ValueError` with a fixed message that never echoes the value, and log `code=read_argument_invalid`.
- NUL and lone surrogates are rejected in input text, keys and metadata; JSON
  must be finite; client tool results are checked against the pending calls.
- `RunStore.follow` admits 500 followers per store and 100 per tenant scope;
  single-tenant apps with more live readers set `max_followers_per_tenant=None`.
- In-process `cancel_run` on a resumed run returns a `CANCELLED` result instead
  of raising `CancelledError`.
- MCP discovery is bounded, and tools with malformed argument schemas are
  skipped or rejected.
- Provider output is bounded before it is stored: opaque evidence is dropped
  past 1 MiB, 10,000 nodes, 32 levels or 5,000 keys; overlong ids are hashed and
  an overlong tool name is rejected as unknown. Custom providers can see shortened
  raw records.
- Migrations `d7e8f9a0b1c2` and `e8f9a0b1c2d3`.
- The OpenTelemetry run span is `ERROR` when `RunResult.error_info` is set.

### Added

- **Read primitives.** `RunStore.get_pending`, `follow`, `get_event_view`,
  `get_pause_page`, `EventFeed`, `redact_event`, `PendingPause`,
  `PendingToolCall`, `Follow`, `FollowEnd`, `FollowHeartbeat` and
  `FollowCapacityError`. `get_pauses` takes `limit` and `include_payloads`.
- **Results.** `RunResult.pending`, `RunResult.pause_id`, `RunResult.error_info`,
  `RunResult.to_dict()` (a versioned browser envelope, `wire_version` 1),
  `Agent.get_result(run_id, tenant_id=, include_pii=)`, and
  `ToolResult.from_dict` / `list_from_dict` for results coming back from a client.
- **Pause ids.** `pause_id` on `RunResult`, `PendingPause`, `PausePair` and the
  `run.paused` event; `expected_pause_id=` on every `submit_*`, `resume` and
  `resume_stream`; `PauseStatusMismatchError(reason="pause_id")`. Pause state is
  versioned (`schema_version`).
- **Tenancy and limits.** `UNSCOPED`, `Agent(strict_tenancy=, limits=,
  tool_threads=, on_governance_change=)`, `Limits`, `TenantScopeRequiredError`,
  `NoteLimitExceededError`, `GovernanceMismatchError` and the `governance.changed`
  event.
- **Provider errors.** `ProviderError` and `ProviderAuthError`,
  `ProviderRateLimitError`, `ProviderTimeoutError`, `ProviderUnavailableError`,
  `ProviderStreamIncompleteError`, `ModelRefusalError`, `ContentFilteredError`
  and `ContextLengthExceededError` with `.category`, `.retryable` and
  `.safe_message`; `call_timeout`, `stream_idle_timeout` and
  `stream_total_timeout`; `llm.failed` events; `PersistenceError` for database
  failures.
- **Budget and PII.** `Budget(count_cache_tokens=)`, dollar-budget price lookup
  (exact, then glob, response model before requested model),
  `PII(restore="allowlisted")` with `@tool(receives_pii=...)` and the
  `guardrail.restore_skipped` event.
- **Operations.** `dendrux db check [--allow-unstamped]`,
  `SchemaOutOfDateError.missing_indexes`, `sweep(batch_size=, max_runs=,
  database_options=)` with a `sweep_cap_reached` warning, `database_options=` on
  `RunStore.from_database_url`, `sweep` and `get_engine`, `dendrux.tool_pool_stats()`
  and `DENDRUX_TOOL_THREADS` / `DENDRUX_TOOL_QUEUE_LIMIT`.
- **MCP.** `MCPSource(max_discovery_pages=, max_discovery_tools=,
  discovery_timeout=, max_discovery_bytes=)`, `MCPDiscoveryLimitError` (code
  `discovery_limit`), `MCPInvalidToolSchemaError` (code `invalid_tool_schema`),
  skip reason `invalid_schema`, `MCPSessionExpiredError`, and a single reopen of an
  expired session.
- **Events.** `tools.declared` at the start of every run segment, `llm.failed`,
  `governance.changed`, and `approval.requested.params`.
- **Docs.** [Upgrading to 0.5.0](https://dendrux.dev/docs/upgrading-to-0.5.0),
  [Scaling workers](https://dendrux.dev/docs/guides/scaling), multi-tenancy, PII
  restore, handling provider errors, build your own read routes, and a
  tenant-erasure recipe. Every Python snippet in the docs runs in CI.

### Changed

- Implicit `DENDRUX_DATABASE_URL` engines warn (`DeprecationWarning`) and are
  removed in 0.6; the shared engine refuses use from a second event loop.
- `Agent(database_url=...)` Agents with equal URL and options share one pool.
  Non-SQLite pools default to `pool_pre_ping=True` and `pool_recycle=1800`.
- Event numbering is allocated inside the insert; concurrent writers no longer
  read the event log first.
- `sweep()` is bounded (`max_runs` 10,000, ceiling 100,000 per category) and ages
  waiting runs by the time of their last status change; a waiting run is
  abandoned only when both that time and `updated_at` are stale. Swept waiting
  runs carry a safe error and a failure reason.
- Guardrails scan a multi-part message, and each tool result, as one text; phone,
  card, IP and AWS-key detection is stricter.
- Resuming under changed governance emits `governance.changed`; resume re-applies
  `deny` and the per-run tool call limits.
- Sync tools run on a bounded process-wide thread pool; a call that times out
  reports `timeout_not_started` or `timeout_outcome_unknown`.
- Cache writes on OpenAI-wire providers count as cache-creation tokens, not fresh
  input.
- FutureWarnings announce the 0.6 flips: strict tenancy, `on_governance_change`,
  `count_cache_tokens` and (under strict tenancy) a missing pause id on non-approval
  submits.

### Fixed

- Large structured client results (above about 64 KiB) are validated off the
  event loop, one job per loop; `ToolResult.from_dict` stays synchronous, so async
  routes that parse large wire objects call it through `asyncio.to_thread`.
- A caller cancelled right after claiming a paused run still has the run finalised
  as `cancelled` before `CancelledError` propagates.

- Failed streams, Responses `error` / `response.failed`, refusals and content
  filters no longer return silent empty answers.
- Anthropic thinking with structured output uses `output_config.format`.
- A provider reused across event loops rebuilds its HTTP transport; streams close
  their upstream request on cancel; a custom HTTP client keeps the SDK's
  connection limits.
- Duplicate and stale submits cannot finalize another worker's run; cancellation
  cleanup survives anyio cancellation.
- Error paths, usage writes and budget reads no longer disable enforcement when
  a database write fails once.
- Pending reads on Postgres no longer fail when stored pause state holds U+0000
  or a lone surrogate, and convert the pause state once per read.
- Redacted feeds no longer expose a past approval's arguments, a person's answer,
  exception text or image descriptions.
- Hostile ids, cursors and filters given to `RunStore` raise `ValueError`
  instead of a database error.
- Tool-argument validation errors no longer echo input keys or values.
- Followers of one run share their active-pause read, so a follower's cost no
  longer grows with the run's past pauses; one tenant cannot take every follower slot.
- MCP discovery stops on a repeated cursor, a page, tool or byte cap, or a
  deadline, and a malformed tool schema no longer raises from the validator.
- Provider failures are logged with safe fields only, without tracebacks or vendor
  text; database failures never expose SQL or bound parameters.

### Removed

- `make_read_router` and the `dendrux.http` package.
- The embedded dashboard (`dendrux.dashboard`, `packages/dashboard`) and the
  `dendrux dashboard` command.
- The `http` extra (install your web framework and server directly).
- `EventSequencer`.

## 0.4.1 - 2026-10-18

Truncation is now visible, Anthropic gets strict mode, and run-control writes
take a tenant scope. No migration. No breaking API changes; see Changed for the
behaviour differences.

### Added

- **Truncated answers are flagged.** `RunResult.stop_reason` is the
  normalized reason the call that produced the final answer stopped
  (`StopReason`: `end_turn`, `tool_use`, `max_tokens`, `stop_sequence`,
  `content_filter`, `refusal`, `pause_turn`, `other`; `None` when the
  provider did not report one). `RunResult.truncated` is `True` when that is
  `max_tokens`. A final answer cut off at the output-token limit (or, on
  Anthropic, at the context window: `model_context_window_exceeded`) still
  ends `SUCCESS`, now with `truncated=True` and a logged warning.
  - `LLMResponse.stop_reason` / `raw_stop_reason` on Anthropic, OpenAI Chat,
    OpenAI Responses and OpenRouter, batch and streaming.
  - The `run.completed` event carries `stop_reason` and `truncated`; so does
    the `RUN_COMPLETED` stream event. Each recorded LLM call's
    `semantic_response` has its `stop_reason`. Idempotent cached results and
    `cancel_run()` on a finished run restore it (`None` for runs finished
    before 0.4.1).
- **Anthropic strict mode.** `@tool(strict=True)` tools are sent to Anthropic
  with `"strict": true`, and structured output (`output_type=`) uses strict
  mode on Anthropic whenever the schema allows it.
  - Dendrux sends a per-request copy of the schema adapted to Anthropic's
    strict subset: objects are closed, and constraints Anthropic rejects
    (`minimum`/`maximum`, `exclusiveMinimum`/`exclusiveMaximum`,
    `multipleOf`, `maxItems`, `minItems` above 1, `uniqueItems`, unsupported
    `format`s and regex features, ...) move into the field description.
    Pydantic still validates the full constraints.
  - Schemas Anthropic can't take strictly (free-form dicts, recursive
    models, tuples, `Any` fields, or more than 24 optional / 16 union-typed
    parameters / 20 strict tools in one request) are sent non-strict exactly
    as in 0.4.0. An explicit `@tool(strict=True)` downgraded this way logs a
    WARNING once per tool.
  - If Anthropic rejects a strict request with a 400, Dendrux retries it
    once with every strict entry sent as in 0.4.0. Rejections it recognises
    as strict-schema limits are remembered per model for the rest of the
    process, with a WARNING once.
  - `AnthropicProvider(strict=False)` turns all of this off and sends 0.4.0
    payloads. Anthropic caches strict schemas for up to 24 hours; keep PHI
    and secrets out of schemas (see the tools docs).
- **`tenant_id=` on run-control writes.** `submit_tool_results`,
  `submit_input`, `submit_approval`, `submit_resume`, `resume` and
  `resume_stream` take `tenant_id=` like `cancel_run` and `submit_note`. A
  run outside the given tenant raises `RunNotFoundError` before anything is
  written. A scoped `resume_stream` raises on first iteration, before any
  event.
- **Cost of a failed structured call.** `StructuredOutputValidationError`
  has `usage`, `model` and `stop_reason`.
- **Unknown tools.** The result of a call to a tool that doesn't exist lists
  the tools the model was offered: `Unknown tool: bash. Available tools:
  read_file, write_file` (sorted, at most 50).

### Changed

- A SingleCall structured call whose output fails validation is now recorded
  as a completed, priced LLM call (an `llm_interactions` row, budget check)
  before the run fails. **It emits `llm.completed` instead of `llm.failed`.**
  Detect validation failures with the `run.error` event or the raised
  `StructuredOutputValidationError`. As for every errored run, the run row's
  `total_*` columns stay 0; `RunStore` subtree usage includes the call.
- **One prompt-cache miss on upgrade for Anthropic structured output and
  strict tools.** Their tool payload now carries `strict` and the adapted
  schema. Non-strict Anthropic tool payloads are byte-identical to 0.4.0.
- `@tool(strict=True, parameters=...)` now also checks draft-7
  `definitions` for free-form objects and raises `TypeError` for one, as it
  already did for `$defs` (OpenAI strict rejected such schemas anyway).

### Fixed

- OpenAI Responses streaming: a truncated response (`response.incomplete`)
  lost its usage, provider payload and reasoning items, so it couldn't be
  priced.
- `cancel_run(cascade=True)` on a parent and child sharing one task (same
  Agent, in-process delegation) could record the parent as cancelled without
  `meta["interrupted"]` when store writes were slow: the child was signalled
  before the parent's interrupt was registered. All in-process signals are
  now sent together after the durable cancel requests.

## 0.4.0 - 2026-10-18

Image input across the run lifecycle. This release includes a database
migration (see Migration): run `dendrux db migrate` before upgrading a
Postgres deployment, even if you never use images. It also changes how MCP
image results reach the model (see Changed).

### Added

- **Images in user input.** `run()`, `stream()`, `submit_input()`,
  `resume(user_input=)` and `history=` accept `UserContent`: a string, or a
  list of strings and images in the order the model should see them.
  `ChatMessage.user(...)` takes the same list (`ChatMessage.parts`); images are
  supported in user turns only (`ChatMessage.assistant()` takes a string and an
  assistant `ChatMessage` with `parts` raises `ValueError`). `Image` (`from_bytes`, `from_path`,
  `from_base64`, `from_url`) and `ToolOutput` are exported from `dendrux`, with
  `ImageRef`, `ImagePart` and `TextPart`. Supported formats are PNG, JPEG, GIF
  and WebP, detected from the bytes (never from a declared type, file name or
  `Content-Type`); animated GIF and WebP are rejected unless allowed.
- **Images in tool results.** A server `@tool` returns
  `ToolOutput(result, images=[...])` (`result` is serialised exactly like a plain
  return value, and the images follow the text). A client tool attaches
  `ToolResult.images` in `submit_tool_results(...)`. `tool_result_message(result)`
  is the one TOOL-message builder. `tool_result_transform` and
  `ToolDecision.substitute(...)` accept a `ToolOutput`, and `ToolCallContext.images`
  lists the images a `before_tool` hook's arguments refer to by id.
  `ToolResult.notes` carries per-image ingest notes.
- **Images from MCP tools.** Image content blocks become image parts for the
  model in agent runs (see Changed for the payload change). An image that cannot
  be used does not fail the call: the model sees
  `[image from MCP could not be used: <reason>]` and the other images.
- **Provider support.** Anthropic (tool images native), OpenAI Responses (tool
  images native by default, confirmed by the live matrix on gpt-5-mini and
  gpt-4o-mini in batch, stream and resume;
  `OpenAIResponsesProvider(tool_result_images="portable")` selects the portable
  form), OpenAI Chat and OpenRouter (portable: a text tool
  message plus one framed user message that says the images are tool output).
  Image bytes are encoded once and never change between calls, so prompt
  caching is unaffected; cache markers, cache keys and text-only request bodies
  are unchanged. Anthropic's "more than 20 images means every image at most
  2000 px" rule is enforced per request (also for `anthropic/*` OpenRouter
  slugs).
- **Non-vision models.** Capability is resolved per model: an
  `ImagePolicy.capability_overrides` entry, then what the provider instance has
  learned, then the adapter (static table for OpenAI models, model catalogue for
  OpenRouter, `claude-*` for Anthropic). A new image for a model that cannot
  read images raises `UnsupportedModalityError` before any provider request; a
  history image becomes a placeholder and never an error. `on_unsupported` can
  be `"placeholder"` or a describe function; `unknown_capability` chooses
  between sending and failing for unknown models.
- **Rejection handling.** When a provider refuses one image, `ImageRejectedError`
  is raised; a rejected tool-result image is instead replaced by a placeholder
  and the call is retried once (only if nothing was streamed yet). Rejected
  images are remembered for the run and are not resent by `retry()` or `runs=`
  replays. An OpenAI content-policy image code counts as a rejection.
- **`ImagePolicy` and `Agent(images=, blob_store=)`.** Defaults are the vendor
  limits: 8000 px long side, 5 MiB per image, 20 images per message, 100 and
  20 MiB per request. Also: `max_estimated_image_tokens_per_request`,
  `allowed_media_types`, `allow_animated`, `urls: UrlPolicy`, `drop:
  ImageDropPolicy` and `ingest_hook` with the required `ingest_hook_version`.
  `agent.image_policy` and `agent.blobs` expose the policy and the blob accessor.
- **Opt-in dropping of old images.** `ImageDropPolicy(keep_recent=3,
  interval=25, scope="tool_results")` keeps the model's view of tool images
  bounded in steps, which keeps most of the prompt cache. It is sticky, stored
  with the run, and survives pause, resume, retry and replay. Logs a warning and
  emits `image.warning` once per run when combined with provider thinking
  (Anthropic, or OpenAI Responses with `thinking=True`).
- **Blob storage.** Images are stored once as tenant-scoped, content-addressed
  blobs in two new tables, `dendrux_blobs` and `dendrux_blob_refs`; runs, pauses,
  traces and events hold references only, and `llm_interactions.provider_request`
  holds `dendrux-image:sha256:<hex>` instead of base64. Bytes live in the
  database by default; pass `Agent(blob_store=...)` and
  `RunStore.from_database_url(url, blob_store=...)` with any object implementing
  the `BlobStore` protocol (`put`, `get`, `delete`) to keep them in object
  storage. `InMemoryBlobStore` backs runs without a database (not durable).
  `agent.blobs` and `RunStore.blobs` provide `read`, `info`, `references`,
  `purge` and `collect_garbage`; `RunStore.read_image(run_id, sha256,
  tenant_id=)` is a run-scoped read.
- **Erasure.** `blobs.purge(sha256, tenant_id=, reason=)` deletes the bytes (and
  the external object), keeps the row as a tombstone, renders
  `[image ... was deleted]` in every run that used the image, and makes
  re-ingesting the same bytes raise `InvalidImageError("purged")`.
- **Fetch-once URLs.** `Image.from_url(url)` fetches once at ingest and stores
  the bytes; the URL is never stored, logged or sent to a provider. The fetcher
  allows HTTPS on 443 to public addresses only, validates every resolved address
  and connects to the validated one (TLS is checked against the original host),
  re-checks each of at most 3 redirects, ignores proxy environment variables and
  `.netrc`, and caps size and time. `UrlPolicy` tunes it.
- **Image guardrails.** The `ImageGuardrail` protocol
  (`async scan_image(ImageScanInput) -> list[Finding]`) screens images once per
  image, guardrail id and version; verdicts are stored and reused on resume,
  retry and replay. If guardrails are configured but none scans images,
  `guardrail.image_unscanned` is written once per run.
- **Events.** Six governance event types (31 in total): `image.rejected`,
  `image.state`, `image.view_changed`, `image.degraded`, `image.warning`,
  `guardrail.image_unscanned`. `run.resumed` and `tool.completed` gain
  `image_count` when images are present. The dashboard maps all six.
- **Errors** in `dendrux.errors`: `InvalidImageError`, `ImageLimitError`,
  `ImageFetchError` (all `ImageError`, a `ValueError`), `UnsupportedModalityError`,
  `ImageRejectedError`, `ProviderError`, `BlobNotFoundError`,
  `BlobIntegrityError`, `ImagePersistenceUnsupportedError`,
  `BlobContentionError`. None of them carries image bytes, base64 or a URL's
  query, and they are raised without a chained exception.
- **Provider surface.** `LLMProvider.image_input_support(model)`,
  `tool_result_images`, `thinking_enabled`, `many_image_rule(model)` and
  `image_capability_key(model)`; `OpenRouterModel.supports_image_input`;
  `MockLLM(responses, image_support="yes" | "no" | "unknown")` and
  `MockLLM.wire_messages` for tests.
- **Token estimates.** `estimate_image_tokens(...)` and
  `llm_interactions.semantic_request.image_stats` (`count`, `bytes`,
  `estimated_tokens`, `None` when the model is unknown). Chat spans in the
  OpenTelemetry notifier gain `dendrux.images.count` and
  `dendrux.images.estimated_tokens`.
- Docs: new Images guide, resize recipe (Pillow `ingest_hook`) and S3 `BlobStore`
  recipe; MCP, Agent, RunStore, Errors, Governance, HTTP API, Guardrails, Cost
  and pricing, State persistence and API glossary updated. New examples
  `examples/37_images.py` and `examples/38_images_client_tool.py` (mock provider,
  no API key by default).

### Changed

- **MCP image results are image parts in agent runs.** Image content blocks no
  longer appear as base64 inside the tool payload. The payload's content
  envelope keeps a stub, `{"type": "image", "image_index": 0, "mime_type":
  "image/png"}`, and the bytes reach the model as an image. Affected: a
  `tool_result_transform` that read `data` from an MCP image block now receives
  a `ToolOutput` as `ctx.result` (`ctx.result.result` is the envelope or
  structured value, `ctx.result.images` the images) and must return it, or a new
  `ToolOutput`, to keep the images; anything reading base64 out of
  `tool_calls.result` or traces for MCP image tools finds the stub instead.
  `MCPConnection.call_tool(...)` is unchanged: it returns the previous envelope,
  base64 included, and its `is_error` text for image-bearing results is
  unchanged (it can contain base64). Structured-content results now also carry
  their images, which were previously discarded.
- **`ProviderError` replaces raw SDK exceptions on image-bearing requests.** An
  application that caught a vendor exception from a request with images now sees
  `ProviderError` (sanitised message, at most 500 characters), or one of the more
  specific image errors. Timeouts and connection failures on such requests are
  `ProviderError` with `error_type="timeout"` or `"connection"`. Requests
  without images keep the exact errors they had.
- **Built-in providers report `supports_multimodal=True`.** A custom
  `LLMProvider` is treated as unable to read images until it sets
  `supports_multimodal=True` and converts `Message.parts`, so an adapter that
  reads only `Message.content` can never silently drop an image.
  `MockLLM` defaults to `supports_multimodal=True`; `MockLLM(image_support="no")`
  restores the old behaviour.
- **Custom `Strategy.format_tool_result` must keep image parts.** If a result had
  images and the returned message has none, the loop raises `TypeError`. Build
  the message with `tool_result_message(result)`.
- **`ChatMessage` gains a third field, `parts`** (default `None`, user turns
  only). Text-only messages now show `parts=None` in `repr` and
  `dataclasses.asdict`, and `parts` is last in `__match_args__`; apps that
  serialise history with `asdict` see the new key. `ChatMessage` now validates
  at construction: non-`str` `content` raises `TypeError`, a role other than
  `user`/`assistant` raises `ValueError`, and a string role (`"assistant"`) is
  converted to `ChatRole`.
- **Custom loops.** A `Loop` whose `run()` (or `run_stream()`) does not accept
  `image_runtime` raises `TypeError` for image input, image replay (`runs=`) and
  `retry()`. Text-only runs are unaffected.
- **Idempotency covers images.** The same `idempotency_key` with different images
  raises `IdempotencyConflictError`. A keyed retry whose image has since been
  purged returns the existing run (the fingerprint is rebuilt from the stored
  input), so two requests that differ only in which since-purged image they send
  count as the same request. This does not apply when `history=` carries images:
  that retry raises `InvalidImageError("purged")`.
- **Image submissions re-check the pause at claim time.** `submit_tool_results` /
  `submit_input` with images verify that the run is still at the pause they
  validated against; otherwise `RunAlreadyClaimedError` and the run stays
  paused. Text-only submissions are unchanged.
- **`provider_request` scrubbing is structural.** The stored request is
  sanitised by replacing the image data fields themselves
  (`dendrux-image:sha256:<hex>`), not by pattern-matching text.
- **`Message` gains `parts`.** Text-only messages are unchanged in meaning but
  `parts=None` now appears in `repr`, `dataclasses.asdict` and as the last entry
  of `__match_args__`; `Message.content` is always a `str` (the text projection
  for multimodal messages). Text-only construction is slightly faster than in
  0.3.0.
- `PauseState` gains `image_view` and history messages gain `parts` in pause JSON,
  only for runs that have images; v0.3.0 pause data resumes unchanged. A persisted
  `agent_runs.input_data`, trace, tool-call or interaction row for a text-only run
  is byte-identical to 0.3.0.
- `GuardrailEngine` text scans skip an image-only guardrail (`scan_image`
  without `scan`); other guardrails behave as in 0.3.0.

### Fixed

- In-memory SQLite (and any engine on `StaticPool` / `SingletonThreadPool`)
  could silently lose rows when parallel tool calls, heartbeats or readers used
  the database at the same time: tool calls, run events, traces, progress
  updates, run rows, and image blob rows and references (image runs could then
  fail with `BlobNotFoundError`). The sessions shared one connection and one
  transaction, so one session's end-of-session rollback discarded another's
  pending write, with no error. The bug predates 0.4.0 and became much more
  frequent on SQLAlchemy 2.1. Dendrux's own database sessions on such engines
  (stores, `RunStore`, `get_session`, SQLite schema sync) are now serialised
  per connection pool; your own `engine.connect()` calls on the same engine are
  not. Creating a store on such an engine logs one warning that sessions are
  serialised and the data is not durable. Do not hold a Dendrux session across
  a call that runs an agent on such an engine (it waits for itself; a warning
  names the holder after 10 seconds). File SQLite and Postgres were never
  affected and are unchanged.
- SQLAlchemy 2.1 is supported (the existing `>=2.0,<3.0` range). CI tests
  both 2.0 and 2.1.

### Migration

- New Alembic revision `c6d7e8f9a0b1` (down revision `b5c6d7e8f9a0`): creates
  `dendrux_blobs` and `dendrux_blob_refs` (with indexes; refs cascade with the run
  and reference blobs with `ON DELETE RESTRICT`). Purely additive. **Every
  Postgres (and other migration-managed) deployment must run `dendrux db migrate`
  after upgrading, even if it never uses images.** Until then the first use of the
  database raises `SchemaOutOfDateError` naming both tables. Downgrade drops
  both.
- SQLite files created by an earlier release gain the two tables on startup.
- No existing table or column changes. Reading a database that has no images
  behaves exactly as in 0.3.0.
- Run `await agent.blobs.collect_garbage()` on a schedule (hourly or daily,
  `older_than` at least one hour) to remove blobs no run references. Deleting a
  run does not delete its blobs by itself.

### Notes

- Pass-through URLs and provider-hosted file ids are not supported in this
  release: `Image.from_url` always fetches and stores. Dendrux does not resize,
  re-encode or decode pixels; use `ImagePolicy.ingest_hook` (a Pillow recipe is
  in the docs; it checks the pixel count before decoding). A synchronous `ingest_hook` runs on the event loop, so do heavy
  work in an async hook that uses a thread.
- The URL fetcher verifies certificates with httpx's default bundle (certifi),
  not the operating system's trust store. A corporate CA needs
  `UrlPolicy(fetcher=...)` (an async callable).
- Image guardrails run on batch paths only; `stream()` still rejects guardrails.
- A gateway that words "this model has no image support" differently from the
  OpenAI, OpenAI Responses and OpenRouter messages does not map to
  `UnsupportedModalityError`; set `ImagePolicy(capability_overrides={model:
  "no"})`.
- Do not enable DEBUG logging for the `openai`, `anthropic` or `httpx`
  libraries when images are in use: their "Request options" lines contain the
  base64 body. Dendrux's own logs never contain image bytes.
- Images are stored in the database by default. For image-heavy workloads use a
  `BlobStore` (S3 recipe in the docs). One external store per deployment:
  every Agent and `RunStore` must use the same one.
- Image tokens are billed inside the provider's input tokens; Dendrux does not
  add anything to usage or cost, and its per-image estimate is for planning only.
- Learned "this model cannot read images" knowledge lives on the provider
  instance and is not shared between processes; a new process re-learns it with
  one extra rejected call.

## 0.3.0 - 2026-10-17

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
- **Upgrading Postgres (and other non-SQLite databases).** After upgrading the
  package, run `dendrux db migrate` with `DENDRUX_DATABASE_URL` pointing at the
  database, then restart. If you forget, the first use of the database (agent
  run, `sweep`) raises `SchemaOutOfDateError` listing the missing tables and
  columns, instead of a raw `UndefinedColumnError` mid-run. The check is
  read-only and runs once per engine; it verifies that the required tables and
  columns are present by exact name (not their types or defaults, and not
  Alembic's version table), ignores anything extra, and does not run on SQLite
  or for `RunStore`. If the database refuses the catalog read, it logs a warning
  and continues. Set `DENDRUX_SKIP_SCHEMA_CHECK=1` to skip it if you manage the
  schema yourself.

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
- `RunStore.from_database_url` on a SQLite path that does not exist creates an
  empty database on first use instead of raising, so a mistyped path reads as
  "no runs".
- The in-place SQLite upgrade needs the write lock only when the file is out of
  date; if another process holds it longer than the SQLite busy timeout, startup
  raises `SqliteSchemaSyncError` (raise the timeout with
  `connect_args={"timeout": N}`).

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
