CREATE TABLE agent_runs (
	id VARCHAR(26) NOT NULL,
	tenant_id VARCHAR(255),
	agent_name VARCHAR(255) NOT NULL,
	status VARCHAR(50) NOT NULL,
	input_data JSON,
	output_data JSON,
	iteration_count INTEGER NOT NULL,
	model VARCHAR(255),
	strategy VARCHAR(100),
	parent_run_id VARCHAR(26),
	delegation_level INTEGER NOT NULL,
	retry_of_run_id VARCHAR(26),
	total_input_tokens INTEGER NOT NULL,
	total_output_tokens INTEGER NOT NULL,
	total_cost_usd NUMERIC(10, 6),
	total_cache_read_tokens INTEGER NOT NULL,
	total_cache_creation_tokens INTEGER NOT NULL,
	total_reasoning_tokens INTEGER DEFAULT '0' NOT NULL,
	meta JSON,
	pause_data JSON,
	pii_mapping JSON,
	error TEXT,
	last_progress_at TIMESTAMP WITH TIME ZONE,
	failure_reason VARCHAR(100),
	idempotency_key VARCHAR(255),
	idempotency_fingerprint VARCHAR(64),
	cancel_requested BOOLEAN DEFAULT '0' NOT NULL,
	pause_requested BOOLEAN DEFAULT '0' NOT NULL,
	claim_epoch INTEGER DEFAULT 0 NOT NULL,
	storage_format SMALLINT DEFAULT 1 NOT NULL,
	reclaim_attempts SMALLINT DEFAULT 0 NOT NULL,
	reclaim_limit SMALLINT,
	lease_expired_at TIMESTAMP WITH TIME ZONE,
	last_reclaimed_at TIMESTAMP WITH TIME ZONE,
	loop_state JSONB,
	trace_link VARCHAR(64),
	created_at TIMESTAMP WITH TIME ZONE NOT NULL,
	updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
	status_changed_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
	PRIMARY KEY (id),
	FOREIGN KEY(parent_run_id) REFERENCES agent_runs (id) ON DELETE SET NULL,
	FOREIGN KEY(retry_of_run_id) REFERENCES agent_runs (id) ON DELETE SET NULL
);

CREATE INDEX ix_agent_runs_tenant_id_created_at ON agent_runs (tenant_id, created_at DESC);

CREATE INDEX ix_agent_runs_sweep_waiting ON agent_runs (status, status_changed_at, id);

CREATE INDEX ix_agent_runs_tenant_id ON agent_runs (tenant_id);

CREATE INDEX ix_agent_runs_parent_run_id ON agent_runs (parent_run_id);

CREATE INDEX ix_agent_runs_status ON agent_runs (status);

CREATE INDEX ix_agent_runs_created_at ON agent_runs (created_at);

CREATE INDEX ix_agent_runs_retry_of_run_id ON agent_runs (retry_of_run_id);

CREATE UNIQUE INDEX uq_agent_runs_tenant_idempotency_key ON agent_runs (coalesce(tenant_id, ''), idempotency_key) WHERE idempotency_key IS NOT NULL;

CREATE INDEX ix_agent_runs_sweep_running ON agent_runs (status, id);

CREATE TABLE dendrux_blobs (
	tenant_scope VARCHAR(260) NOT NULL,
	sha256 CHAR(64) NOT NULL,
	generation VARCHAR(26) NOT NULL,
	state VARCHAR(16) NOT NULL,
	media_type VARCHAR(32) NOT NULL,
	byte_length INTEGER NOT NULL,
	width INTEGER NOT NULL,
	height INTEGER NOT NULL,
	frame_count INTEGER DEFAULT '1' NOT NULL,
	storage VARCHAR(16) NOT NULL,
	data BYTEA,
	ingest_transform VARCHAR(100),
	created_at TIMESTAMP WITH TIME ZONE NOT NULL,
	last_used_at TIMESTAMP WITH TIME ZONE NOT NULL,
	state_changed_at TIMESTAMP WITH TIME ZONE NOT NULL,
	purged_at TIMESTAMP WITH TIME ZONE,
	purge_reason VARCHAR(200),
	PRIMARY KEY (tenant_scope, sha256)
);

CREATE INDEX ix_dendrux_blobs_gc ON dendrux_blobs (state, last_used_at);

CREATE INDEX ix_dendrux_blobs_work ON dendrux_blobs (state, state_changed_at);

CREATE TABLE dendrux_settings (
	key VARCHAR(64) NOT NULL,
	value VARCHAR(255) NOT NULL,
	updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
	PRIMARY KEY (key)
);

CREATE TABLE react_traces (
	id VARCHAR(26) NOT NULL,
	agent_run_id VARCHAR(26) NOT NULL,
	role VARCHAR(20) NOT NULL,
	content TEXT NOT NULL,
	trace_kind VARCHAR(16) DEFAULT 'turn' NOT NULL,
	content_digest CHAR(64),
	order_index INTEGER NOT NULL,
	meta JSON,
	created_at TIMESTAMP WITH TIME ZONE NOT NULL,
	updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
	PRIMARY KEY (id),
	FOREIGN KEY(agent_run_id) REFERENCES agent_runs (id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX uq_react_traces_run_order ON react_traces (agent_run_id, order_index);

CREATE TABLE tool_calls (
	id VARCHAR(26) NOT NULL,
	agent_run_id VARCHAR(26) NOT NULL,
	tool_call_id VARCHAR(26) NOT NULL,
	provider_tool_call_id VARCHAR(255),
	tool_name VARCHAR(255) NOT NULL,
	target VARCHAR(50) NOT NULL,
	params JSON,
	result JSON,
	success BOOLEAN NOT NULL,
	duration_ms INTEGER,
	iteration_index INTEGER,
	error_message TEXT,
	meta JSON,
	outcome VARCHAR(16) DEFAULT 'completed' NOT NULL,
	idempotent BOOLEAN DEFAULT false NOT NULL,
	replayed_at TIMESTAMP WITH TIME ZONE,
	content_digest CHAR(64),
	created_at TIMESTAMP WITH TIME ZONE NOT NULL,
	updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
	PRIMARY KEY (id),
	FOREIGN KEY(agent_run_id) REFERENCES agent_runs (id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX uq_tool_calls_run_call ON tool_calls (agent_run_id, tool_call_id);

CREATE INDEX ix_tool_calls_agent_run_id ON tool_calls (agent_run_id);

CREATE TABLE token_usage (
	id VARCHAR(26) NOT NULL,
	agent_run_id VARCHAR(26) NOT NULL,
	iteration_index INTEGER NOT NULL,
	model VARCHAR(255),
	provider VARCHAR(100),
	input_tokens INTEGER NOT NULL,
	output_tokens INTEGER NOT NULL,
	cost_usd NUMERIC(10, 6),
	duration_ms INTEGER,
	cache_read_input_tokens INTEGER,
	cache_creation_input_tokens INTEGER,
	reasoning_tokens INTEGER,
	meta JSON,
	created_at TIMESTAMP WITH TIME ZONE NOT NULL,
	PRIMARY KEY (id),
	FOREIGN KEY(agent_run_id) REFERENCES agent_runs (id) ON DELETE CASCADE
);

CREATE INDEX ix_token_usage_agent_run_id ON token_usage (agent_run_id);

CREATE TABLE llm_interactions (
	id VARCHAR(26) NOT NULL,
	agent_run_id VARCHAR(26) NOT NULL,
	iteration_index INTEGER NOT NULL,
	model VARCHAR(255),
	provider VARCHAR(100),
	semantic_request JSON,
	semantic_response JSON,
	provider_request JSON,
	provider_response JSON,
	input_tokens INTEGER NOT NULL,
	output_tokens INTEGER NOT NULL,
	cost_usd NUMERIC(10, 6),
	duration_ms INTEGER,
	cache_read_input_tokens INTEGER,
	cache_creation_input_tokens INTEGER,
	reasoning_tokens INTEGER,
	guardrail_findings JSON,
	created_at TIMESTAMP WITH TIME ZONE NOT NULL,
	PRIMARY KEY (id),
	FOREIGN KEY(agent_run_id) REFERENCES agent_runs (id) ON DELETE CASCADE
);

CREATE INDEX ix_llm_interactions_agent_run_id ON llm_interactions (agent_run_id);

CREATE TABLE run_events (
	id VARCHAR(26) NOT NULL,
	agent_run_id VARCHAR(26) NOT NULL,
	event_type VARCHAR(50) NOT NULL,
	sequence_index INTEGER NOT NULL,
	iteration_index INTEGER NOT NULL,
	correlation_id VARCHAR(26),
	data JSON,
	content_digest CHAR(64),
	created_at TIMESTAMP WITH TIME ZONE NOT NULL,
	PRIMARY KEY (id),
	FOREIGN KEY(agent_run_id) REFERENCES agent_runs (id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX uq_run_events_run_seq ON run_events (agent_run_id, sequence_index);

CREATE INDEX ix_run_events_event_type ON run_events (event_type);

CREATE INDEX ix_run_events_started ON run_events (agent_run_id) WHERE ((event_type)::text = 'run.started'::text);

CREATE TABLE run_inbox (
	id VARCHAR(26) NOT NULL,
	run_id VARCHAR(26) NOT NULL,
	text TEXT NOT NULL,
	metadata JSON,
	created_at TIMESTAMP WITH TIME ZONE NOT NULL,
	delivered_at TIMESTAMP WITH TIME ZONE,
	delivered_iteration INTEGER,
	PRIMARY KEY (id),
	FOREIGN KEY(run_id) REFERENCES agent_runs (id) ON DELETE CASCADE
);

CREATE INDEX ix_run_inbox_run_pending ON run_inbox (run_id, delivered_at, created_at, id);

CREATE TABLE dendrux_blob_refs (
	id VARCHAR(26) NOT NULL,
	tenant_scope VARCHAR(260) NOT NULL,
	sha256 CHAR(64) NOT NULL,
	run_id VARCHAR(26) NOT NULL,
	owner_kind VARCHAR(32) NOT NULL,
	owner_id VARCHAR(64) NOT NULL,
	ordinal INTEGER NOT NULL,
	created_at TIMESTAMP WITH TIME ZONE NOT NULL,
	PRIMARY KEY (id),
	FOREIGN KEY(tenant_scope, sha256) REFERENCES dendrux_blobs (tenant_scope, sha256) ON DELETE RESTRICT,
	CONSTRAINT uq_dendrux_blob_refs_owner UNIQUE (run_id, owner_kind, owner_id, ordinal),
	FOREIGN KEY(run_id) REFERENCES agent_runs (id) ON DELETE CASCADE
);

CREATE INDEX ix_dendrux_blob_refs_blob ON dendrux_blob_refs (tenant_scope, sha256);

CREATE INDEX ix_dendrux_blob_refs_run ON dendrux_blob_refs (run_id, tenant_scope);

