/** Individual timeline node — spacious cards with icon markers. */

import type { TimelineNode, LLMCallNode, ToolCallNode, FinishNode, ErrorNode, RunStartedNode, CancelledNode } from "@/lib/types";
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
