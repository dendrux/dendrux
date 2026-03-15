/** Individual timeline node card — compact by default, detail on selection. */

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
      className={`w-full text-left rounded-[10px] px-4 py-3 transition-colors duration-120 border ${
        isSelected
          ? "bg-elevated border-border"
          : "bg-transparent border-transparent hover:bg-surface"
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

// -- Node content renderers --

function RunStartedContent({ node }: { node: RunStartedNode }) {
  return (
    <div>
      <NodeTitle label="Run Started" color="text-text-muted" />
      <NodeMeta items={[
        { label: "Agent", value: node.agent_name },
        ...(node.timestamp ? [{ label: "", value: formatTime(node.timestamp) }] : []),
      ]} />
    </div>
  );
}

function LLMCallContent({ node }: { node: LLMCallNode }) {
  return (
    <div>
      <div className="flex items-center gap-2">
        <div className="w-[3px] h-4 rounded-full bg-state-llm" />
        <NodeTitle label={`LLM Call`} color="text-state-llm" />
        <span className="text-xs text-text-muted">Iteration {node.iteration}</span>
      </div>
      <NodeMeta items={[
        { label: "Tokens", value: `${formatTokens(node.input_tokens)} in / ${formatTokens(node.output_tokens)} out`, mono: true },
        ...(node.cost_usd != null ? [{ label: "Cost", value: formatCost(node.cost_usd), mono: true }] : []),
        ...(node.model ? [{ label: "Model", value: node.model }] : []),
      ]} />
      {node.assistant_text && (
        <p className="mt-1.5 text-xs text-text-secondary line-clamp-2 leading-relaxed">
          {node.assistant_text}
        </p>
      )}
    </div>
  );
}

function ToolCallContent({ node }: { node: ToolCallNode }) {
  const isServer = node.target === "server";

  return (
    <div>
      <div className="flex items-center gap-2">
        <div className="w-[3px] h-4 rounded-full bg-state-tool" />
        <NodeTitle label={node.tool_name} color="text-state-tool" />
        <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
          isServer ? "bg-state-tool/10 text-state-tool" : "bg-state-paused/10 text-state-paused"
        }`}>
          {node.target}
        </span>
        {!node.success && (
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-state-error/10 text-state-error font-medium">
            failed
          </span>
        )}
      </div>
      <NodeMeta items={[
        ...(node.duration_ms != null ? [{ label: "", value: formatDuration(node.duration_ms), mono: true }] : []),
        { label: "Iter", value: String(node.iteration) },
      ]} />
    </div>
  );
}

function FinishContent({ node }: { node: FinishNode }) {
  return (
    <div>
      <NodeTitle label="Finished" color="text-text-primary" />
      <NodeMeta items={[
        { label: "Status", value: node.status },
        ...(node.timestamp ? [{ label: "", value: formatTime(node.timestamp) }] : []),
      ]} />
    </div>
  );
}

function ErrorContent({ node }: { node: ErrorNode }) {
  return (
    <div className="bg-state-error/5 -mx-4 -my-3 px-4 py-3 rounded-[10px]">
      <div className="flex items-center gap-2">
        <div className="w-[3px] h-4 rounded-full bg-state-error" />
        <NodeTitle label="Error" color="text-state-error" />
      </div>
      <p className="mt-1 text-xs text-state-error/80 leading-relaxed">{node.error}</p>
    </div>
  );
}

function CancelledContent({ node }: { node: CancelledNode }) {
  return (
    <div>
      <NodeTitle label="Cancelled" color="text-text-muted" />
      <NodeMeta items={[
        ...(node.timestamp ? [{ label: "", value: formatTime(node.timestamp) }] : []),
      ]} />
    </div>
  );
}

// -- Shared sub-components --

function NodeTitle({ label, color }: { label: string; color: string }) {
  return <span className={`text-sm font-medium ${color}`}>{label}</span>;
}

function NodeMeta({ items }: { items: { label: string; value: string; mono?: boolean }[] }) {
  if (items.length === 0) return null;
  return (
    <div className="flex items-center gap-3 mt-1 text-xs text-text-muted">
      {items.map((item, i) => (
        <span key={i} className={item.mono ? "font-mono" : ""}>
          {item.label && <span className="text-text-muted">{item.label} </span>}
          <span className="text-text-secondary">{item.value}</span>
        </span>
      ))}
    </div>
  );
}
