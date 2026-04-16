/** Types matching the Python normalizer output exactly. */

// -- Timeline node types --

export interface RunStartedNode {
  type: "run_started";
  sequence_index: number;
  agent_name: string;
  timestamp: string | null;
}

export interface LLMCallNode {
  type: "llm_call";
  sequence_index: number;
  iteration: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number | null;
  model: string | null;
  has_tool_calls: boolean;
  timestamp: string | null;
  assistant_text: string | null;
  // null when the provider didn't report cache stats; 0 means reported zero.
  cache_read_input_tokens: number | null;
  cache_creation_input_tokens: number | null;
}

export interface ToolCallNode {
  type: "tool_call";
  sequence_index: number;
  iteration: number;
  tool_call_id: string;
  tool_name: string;
  target: string;
  success: boolean;
  duration_ms: number | null;
  timestamp: string | null;
  params: Record<string, unknown> | null;
  result: Record<string, unknown> | null;
  error_message: string | null;
}

export interface PendingToolCallInfo {
  tool_call_id: string;
  tool_name: string;
  target: string | null;
}

export interface SubmittedResultInfo {
  call_id: string;
  tool_name: string;
  success: boolean;
}

export interface PauseSegmentNode {
  type: "pause_segment";
  sequence_index: number;
  iteration: number;
  pause_status: string;
  pending_tool_calls: PendingToolCallInfo[];
  paused_at: string | null;
  resumed_at: string | null;
  wait_duration_ms: number | null;
  submitted_results: SubmittedResultInfo[];
  user_input: string | null;
}

export interface FinishNode {
  type: "finish";
  sequence_index: number;
  status: string;
  timestamp: string | null;
}

export interface ErrorNode {
  type: "error";
  sequence_index: number;
  error: string;
  timestamp: string | null;
}

export interface CancelledNode {
  type: "cancelled";
  sequence_index: number;
  timestamp: string | null;
}

export interface GovernanceEventNode {
  type: "governance_event";
  sequence_index: number;
  iteration: number;
  event_type: string;
  severity: string;
  title: string;
  data: Record<string, unknown>;
  timestamp: string | null;
}

export type TimelineNode =
  | RunStartedNode
  | LLMCallNode
  | ToolCallNode
  | PauseSegmentNode
  | FinishNode
  | ErrorNode
  | CancelledNode
  | GovernanceEventNode;

// -- Run summary --

export interface RunSummary {
  run_id: string;
  agent_name: string;
  status: string;
  model: string | null;
  strategy: string | null;
  input_text: string | null;
  answer: string | null;
  error: string | null;
  iteration_count: number;
  total_input_tokens: number;
  total_output_tokens: number;
  total_cost_usd: number | null;
  total_cache_read_tokens: number;
  total_cache_creation_tokens: number;
  created_at: string | null;
  updated_at: string | null;
}

// -- Message (trace) --

export interface TraceMessage {
  role: string;
  content: string;
  order_index: number;
  meta: Record<string, unknown> | null;
  created_at: string | null;
}

// -- LLM Interaction (evidence layer) --

export interface LLMInteraction {
  id: string;
  iteration_index: number;
  model: string | null;
  provider: string | null;
  semantic_request: Record<string, unknown> | null;
  semantic_response: Record<string, unknown> | null;
  provider_request: Record<string, unknown> | null;
  provider_response: Record<string, unknown> | null;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number | null;
  duration_ms: number | null;
  cache_read_input_tokens: number | null;
  cache_creation_input_tokens: number | null;
  created_at: string | null;
}

export interface LLMCallsResponse {
  llm_calls: LLMInteraction[];
}

// -- Delegation types --

export interface ParentRef {
  run_id: string;
  resolved: boolean;
  agent_name: string | null;
  status: string | null;
  delegation_level: number | null;
}

export interface RunBrief {
  run_id: string;
  agent_name: string;
  status: string;
  delegation_level: number;
}

export interface SubtreeSummary {
  direct_child_count: number;
  descendant_count: number;
  max_depth: number;
  subtree_input_tokens: number;
  subtree_output_tokens: number;
  subtree_cost_usd: number | null;
  unknown_cost_count: number;
  status_counts: Record<string, number>;
}

export interface DelegationInfo {
  parent: ParentRef | null;
  children: RunBrief[];
  ancestry: RunBrief[];
  subtree_summary: SubtreeSummary;
  ancestry_complete: boolean;
}

// -- API response shapes --

export interface RunDetailResponse {
  summary: RunSummary;
  nodes: TimelineNode[];
  system_prompt: string | null;
  messages_by_iteration: Record<string, TraceMessage[]>;
  delegation: DelegationInfo | null;
}

export interface RunListItem {
  run_id: string;
  agent_name: string;
  status: string;
  iteration_count: number;
  total_input_tokens: number;
  total_output_tokens: number;
  total_cost_usd: number | null;
  total_cache_read_tokens: number;
  total_cache_creation_tokens: number;
  model: string | null;
  parent_run_id: string | null;
  delegation_level: number;
  pause_count: number;
  created_at: string | null;
  updated_at: string | null;
}

export interface RunListResponse {
  runs: RunListItem[];
  total: number;
}
