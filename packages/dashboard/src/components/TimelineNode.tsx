/** Individual timeline node — spacious cards with icon markers. */

import type { TimelineNode, LLMCallNode, ToolCallNode, FinishNode, ErrorNode, RunStartedNode, CancelledNode, GovernanceEventNode } from "@/lib/types";
import { formatDuration, formatTokens, formatCost, formatTime } from "@/lib/format";

interface TimelineNodeCardProps {
  node: TimelineNode;
  isSelected: boolean;
  onSelect: () => void;
}

export function TimelineNodeCard({ node, isSelected, onSelect }: TimelineNodeCardProps) {
  return (
    <button
      onClick={onSelect}
      className={`w-full text-left transition-all duration-150 ${
        isSelected ? "opacity-100" : "opacity-90 hover:opacity-100"
      }`}
    >
      {node.type === "run_started" && <RunStartedContent node={node} />}
      {node.type === "llm_call" && <LLMCallContent node={node} />}
      {node.type === "tool_call" && <ToolCallContent node={node} />}
      {node.type === "finish" && <FinishContent node={node} />}
      {node.type === "error" && <ErrorContent node={node} />}
      {node.type === "cancelled" && <CancelledContent node={node} />}
      {node.type === "governance_event" && <GovernanceEventContent node={node} />}
    </button>
  );
}

function RunStartedContent({ node }: { node: RunStartedNode }) {
  return (
    <div>
      <h3 className="text-lg font-bold text-text-primary">Run Started</h3>
      <div className="flex items-center gap-4 text-sm text-text-muted mt-1">
        <span className="flex items-center gap-1">
          <span className="material-symbols-outlined text-sm">smart_toy</span>
          {node.agent_name}
        </span>
        {node.timestamp && (
          <span className="flex items-center gap-1">
            <span className="material-symbols-outlined text-sm">schedule</span>
            {formatTime(node.timestamp)}
          </span>
        )}
      </div>
    </div>
  );
}

function LLMCallContent({ node }: { node: LLMCallNode }) {
  const cacheRead = node.cache_read_input_tokens ?? 0;
  return (
    <div>
      <h3 className="text-lg font-bold text-text-primary">
        LLM Call{node.model ? `: ${node.model}` : ""}
      </h3>
      <div className="flex items-center gap-4 text-sm text-text-muted mt-1">
        <span className="flex items-center gap-1">
          <span className="material-symbols-outlined text-sm">toll</span>
          {formatTokens(node.input_tokens + node.output_tokens)} tokens
        </span>
        {cacheRead > 0 && (
          <span className="flex items-center gap-1" title="Tokens read from prompt cache">
            <span className="material-symbols-outlined text-sm">bolt</span>
            {formatTokens(cacheRead)} cached
          </span>
        )}
        {node.cost_usd != null && (
          <span className="flex items-center gap-1">
            <span className="material-symbols-outlined text-sm">payments</span>
            {formatCost(node.cost_usd)}
          </span>
        )}
        <span className="text-xs text-text-muted">Iter {node.iteration}</span>
      </div>
      {node.assistant_text && (
        <div className="mt-3 p-4 rounded-xl bg-surface border border-border-soft text-text-secondary text-sm leading-relaxed line-clamp-3">
          {node.assistant_text}
        </div>
      )}
    </div>
  );
}

function ToolCallContent({ node }: { node: ToolCallNode }) {
  const isServer = node.target === "server";
  return (
    <div>
      <h3 className="text-lg font-bold text-text-primary">
        {isServer ? "Server" : "Client"} Tool: {node.tool_name}
      </h3>
      <div className="flex items-center gap-4 text-sm text-text-muted mt-1">
        <span className={`flex items-center gap-1 ${node.success ? "text-state-tool" : "text-state-error"}`}>
          <span className="material-symbols-outlined text-sm">
            {node.success ? "check_circle" : "error"}
          </span>
          {node.success ? "Success" : "Failed"}
        </span>
        {node.duration_ms != null && (
          <span className="flex items-center gap-1">
            <span className="material-symbols-outlined text-sm">schedule</span>
            {formatDuration(node.duration_ms)}
          </span>
        )}
      </div>
    </div>
  );
}

function FinishContent({ node }: { node: FinishNode }) {
  return (
    <div className="pt-6">
      <h3 className="text-lg font-bold text-text-primary">Finish: Success</h3>
      <p className="text-text-muted text-sm mt-1">Workflow execution completed.</p>
      {node.timestamp && (
        <span className="text-xs text-text-muted mt-1 block">{formatTime(node.timestamp)}</span>
      )}
    </div>
  );
}

function ErrorContent({ node }: { node: ErrorNode }) {
  return (
    <div>
      <h3 className="text-lg font-bold text-state-error">Error</h3>
      <div className="mt-3 p-4 rounded-xl bg-state-error/5 border border-state-error/20 text-state-error/80 text-sm leading-relaxed">
        {node.error}
      </div>
    </div>
  );
}

function CancelledContent({ node }: { node: CancelledNode }) {
  return (
    <div>
      <h3 className="text-lg font-bold text-text-muted">Cancelled</h3>
      {node.timestamp && (
        <span className="text-xs text-text-muted mt-1 block">{formatTime(node.timestamp)}</span>
      )}
    </div>
  );
}

const SEVERITY_STYLES: Record<string, { badge: string; text: string; icon: string }> = {
  warning: { badge: "bg-amber-500/10 text-amber-400 border-amber-500/20", text: "text-amber-400", icon: "warning" },
  error: { badge: "bg-red-500/10 text-red-400 border-red-500/20", text: "text-red-400", icon: "error" },
  pause: { badge: "bg-yellow-500/10 text-yellow-400 border-yellow-500/20", text: "text-yellow-400", icon: "pause_circle" },
  info: { badge: "bg-purple-500/10 text-purple-400 border-purple-500/20", text: "text-purple-400", icon: "shield" },
};

function GovernanceEventContent({ node }: { node: GovernanceEventNode }) {
  const style = SEVERITY_STYLES[node.severity] ?? SEVERITY_STYLES.info!;
  const data = node.data;

  // Build detail string from event data
  let detail = "";
  if (data.tool_name) detail = String(data.tool_name);
  if (data.decision) detail = String(data.decision);
  if (data.fraction != null) detail = `${Math.round(Number(data.fraction) * 100)}% (${data.used}/${data.max})`;
  if (data.used != null && data.max != null && data.fraction == null) detail = `${data.used}/${data.max} tokens`;
  if (data.direction) detail = detail ? `${data.direction} - ${detail}` : String(data.direction);
  if (data.findings_count) detail += detail ? ` (${data.findings_count} findings)` : `${data.findings_count} findings`;
  if (data.entities && Array.isArray(data.entities)) detail += ` [${(data.entities as string[]).join(", ")}]`;
  if (data.error && !data.tool_name) detail = String(data.error);

  return (
    <div>
      <div className="flex items-center gap-2">
        <span className={`material-symbols-outlined text-lg ${style.text}`}>{style.icon}</span>
        <h3 className={`text-base font-semibold ${style.text}`}>{node.title}</h3>
        <span className={`text-[10px] px-2 py-0.5 rounded-full border font-mono ${style.badge}`}>
          {node.event_type}
        </span>
      </div>
      {detail && (
        <p className="text-sm text-text-muted mt-1 ml-7">{detail}</p>
      )}
    </div>
  );
}
