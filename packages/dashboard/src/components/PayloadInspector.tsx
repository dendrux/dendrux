/**
 * PayloadInspector — shows what went in and what came out.
 * For LLM calls: system prompt, tool schemas, full conversation, response.
 * For tool calls: params (input) and result (output).
 * For pauses: pending calls and submitted results.
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import type {
  TraceMessage,
  TimelineNode,
  LLMCallNode,
  LLMInteraction,
  ToolCallNode,
  PauseSegmentNode,
  RunStartedNode,
  ErrorNode,
  FinishNode,
} from "@/lib/types";
import { fetchLLMCalls } from "@/lib/api";
import { useDashboardStore } from "@/lib/store";
import { formatDuration, formatTokens, formatCost } from "@/lib/format";

interface PayloadInspectorProps {
  runId: string;
  messages: Record<string, TraceMessage[]>;
  systemPrompt?: string | null;
  onClose?: () => void;
}

export function PayloadInspector({ runId, messages, systemPrompt, onClose }: PayloadInspectorProps) {
  const { selectedNode, selectedIteration, clearSelection } = useDashboardStore();
  const handleClose = onClose ?? clearSelection;

  if (!selectedNode) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-text-muted px-8">
        <span className="material-symbols-outlined text-4xl text-white/10 mb-3">data_object</span>
        <p className="text-sm">Select a timeline node to inspect</p>
      </div>
    );
  }

  const iterMessages = selectedIteration != null ? messages[String(selectedIteration)] : undefined;

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-5 border-b border-white/[0.06] flex items-center justify-between flex-shrink-0">
        <h2 className="text-base font-bold text-text-primary flex items-center gap-2">
          <span className="material-symbols-outlined text-text-muted">data_object</span>
          Payload Inspector
        </h2>
        <button onClick={handleClose} className="text-text-muted hover:text-text-primary transition-colors">
          <span className="material-symbols-outlined">close</span>
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        <AnimatePresence mode="wait">
          <motion.div
            key={selectedNode.sequence_index}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
          >
            <NodeInspector
              node={selectedNode}
              runId={runId}
              systemPrompt={systemPrompt}
              iterMessages={iterMessages}
            />
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-white/[0.06] text-xs text-text-muted font-mono flex-shrink-0">
        seq #{selectedNode.sequence_index} · {selectedNode.type}
      </div>
    </div>
  );
}

/** Type-specific inspector. */
function NodeInspector({ node, runId, systemPrompt, iterMessages }: {
  node: TimelineNode;
  runId: string;
  systemPrompt?: string | null;
  iterMessages?: TraceMessage[];
}) {
  switch (node.type) {
    case "tool_call": return <ToolCallInspector node={node as ToolCallNode} iterMessages={iterMessages} />;
    case "llm_call": return <LLMCallInspector node={node as LLMCallNode} runId={runId} systemPrompt={systemPrompt} iterMessages={iterMessages} />;
    case "pause_segment": return <PauseInspector node={node as PauseSegmentNode} />;
    case "run_started": return <RunStartedInspector node={node as RunStartedNode} />;
    case "error": return <ErrorInspector node={node as ErrorNode} />;
    case "finish": return <FinishInspector node={node as FinishNode} />;
    default: return <Section title="Details"><JsonBlock data={node as unknown as Record<string, unknown>} /></Section>;
  }
}

// -------------------------------------------------------------------
// LLM Call — the most important inspector. Shows the full context.
// -------------------------------------------------------------------

function LLMCallInspector({ node, runId, systemPrompt, iterMessages }: {
  node: LLMCallNode;
  runId: string;
  systemPrompt?: string | null;
  iterMessages?: TraceMessage[];
}) {
  type ViewMode = "semantic" | "provider";
  const [viewMode, setViewMode] = useState<ViewMode>("semantic");

  // Fetch evidence layer data (llm_interactions)
  const { data: llmCallsData } = useQuery({
    queryKey: ["llm-calls", runId],
    queryFn: () => fetchLLMCalls(runId),
    staleTime: 30_000,
  });

  // Match the interaction for this iteration
  const interaction: LLMInteraction | undefined = llmCallsData?.llm_calls?.find(
    (c) => c.iteration_index === node.iteration
  );

  return (
    <>
      {/* Stats */}
      <div className="flex gap-3 p-5 border-b border-white/[0.06]">
        <StatCard label="Input" value={formatTokens(node.input_tokens)} mono />
        <StatCard label="Output" value={formatTokens(node.output_tokens)} mono />
        {node.cost_usd != null && <StatCard label="Cost" value={formatCost(node.cost_usd)} mono />}
        {node.model && <StatCard label="Model" value={node.model} />}
        {interaction?.duration_ms != null && <StatCard label="Latency" value={formatDuration(interaction.duration_ms)} mono />}
      </div>

      {/* Semantic / Provider toggle — same interaction, two views */}
      <div className="px-5 py-3 border-b border-white/[0.06] flex items-center justify-between">
        <span className="text-xs text-text-muted">
          {viewMode === "semantic" ? "Dendrux semantics" : "Vendor wire format"}
        </span>
        <div className="flex gap-1.5">
          <ToggleButton active={viewMode === "semantic"} onClick={() => setViewMode("semantic")} icon="neurology" label="Semantic" />
          <ToggleButton active={viewMode === "provider"} onClick={() => setViewMode("provider")} icon="cloud" label="Provider" />
        </div>
      </div>

      {viewMode === "provider" ? (
        /* ---- PROVIDER VIEW — exact vendor API payloads ---- */
        interaction ? (
          <>
            <Collapsible title="Provider Request" icon="upload" defaultOpen>
              {interaction.provider_request ? (
                <JsonBlock data={interaction.provider_request} />
              ) : (
                <Empty text="Provider adapter does not emit request payloads" />
              )}
            </Collapsible>
            <Collapsible title="Provider Response" icon="download" defaultOpen>
              {interaction.provider_response ? (
                <JsonBlock data={interaction.provider_response} />
              ) : (
                <Empty text="Provider adapter does not emit response payloads" />
              )}
            </Collapsible>
          </>
        ) : (
          <div className="p-8 text-center">
            <span className="material-symbols-outlined text-3xl text-white/10 mb-3 block">cloud_off</span>
            <p className="text-text-muted text-sm">No provider payload data for this call.</p>
            <p className="text-text-muted text-xs mt-1">Run the agent again to capture payloads.</p>
          </div>
        )
      ) : (
        /* ---- SEMANTIC VIEW — Dendrux's normalized payloads ---- */
        <>
          {/* Persisted semantic request (full-fidelity from llm_interactions) */}
          {interaction?.semantic_request ? (
            <>
              <Collapsible title="Semantic Request" icon="upload" defaultOpen>
                <JsonBlock data={interaction.semantic_request} />
              </Collapsible>
              <Collapsible title="Semantic Response" icon="download" defaultOpen>
                {interaction.semantic_response ? (
                  <JsonBlock data={interaction.semantic_response} />
                ) : (
                  <Empty text="No response data" />
                )}
              </Collapsible>
            </>
          ) : (
            <>
              {/* Fallback: render from traces when no evidence data exists (pre-3.5 runs) */}
              {systemPrompt && (
                <Collapsible title="System Prompt" icon="psychology" defaultOpen={node.iteration === 1}>
                  <pre className="text-text-secondary font-mono text-xs whitespace-pre-wrap leading-relaxed">
                    {systemPrompt}
                  </pre>
                </Collapsible>
              )}
              {iterMessages && iterMessages.length > 0 && (
                <Section title={`Message Flow — Iteration ${node.iteration}`} icon="forum">
                  <div className="space-y-4">
                    {iterMessages.map((msg, i) => <MessageBubble key={i} msg={msg} />)}
                  </div>
                </Section>
              )}
              {node.assistant_text && (!iterMessages || iterMessages.length === 0) && (
                <Section title="Assistant Response" icon="smart_toy">
                  <pre className="text-text-secondary font-mono text-xs whitespace-pre-wrap leading-relaxed">
                    {node.assistant_text}
                  </pre>
                </Section>
              )}
            </>
          )}

          {/* Tool calls indicator */}
          {node.has_tool_calls && (
            <div className="px-5 py-3 border-b border-white/[0.06] flex items-center gap-2 text-xs text-text-muted">
              <span className="material-symbols-outlined text-sm text-state-tool">build</span>
              This LLM call requested tool execution (see tool nodes below)
            </div>
          )}
        </>
      )}
    </>
  );
}

/** Toggle button for Semantic / Provider view. */
function ToggleButton({ active, onClick, icon, label }: {
  active: boolean;
  onClick: () => void;
  icon: string;
  label: string;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium transition-colors ${
        active
          ? "bg-state-llm/15 text-state-llm border border-state-llm/30"
          : "bg-white/5 text-text-muted border border-white/10 hover:bg-white/10"
      }`}
    >
      <span className="material-symbols-outlined text-sm">{icon}</span>
      {label}
    </button>
  );
}

// -------------------------------------------------------------------
// Tool Call
// -------------------------------------------------------------------

function ToolCallInspector({ node, iterMessages }: { node: ToolCallNode; iterMessages?: TraceMessage[] }) {
  return (
    <>
      {/* Stats */}
      <div className="flex gap-3 p-5 border-b border-white/[0.06]">
        <StatCard label="Tool" value={node.tool_name} />
        <StatCard label="Target" value={node.target} />
        <StatCard label="Duration" value={formatDuration(node.duration_ms)} mono />
        <StatCard
          label="Status"
          value={node.success ? "Success" : "Failed"}
          className={node.success ? "text-state-success" : "text-state-error"}
        />
      </div>

      {/* Input */}
      <Section title="Input (params)" icon="upload">
        {node.params ? <JsonBlock data={node.params} /> : <Empty text="No parameters" />}
      </Section>

      {/* Output */}
      <Section title="Output (result)" icon="download">
        {node.result ? (
          typeof node.result === "string"
            ? <pre className="text-text-secondary font-mono text-xs whitespace-pre-wrap leading-relaxed">{node.result}</pre>
            : <JsonBlock data={node.result} />
        ) : <Empty text="No result" />}
      </Section>

      {/* Error */}
      {node.error_message && (
        <Section title="Error" icon="error">
          <p className="text-state-error text-sm">{node.error_message}</p>
        </Section>
      )}

      {/* Context: iteration messages */}
      {iterMessages && iterMessages.length > 0 && (
        <Collapsible title={`Conversation Context — Iteration ${node.iteration}`} icon="forum">
          <div className="space-y-3">
            {iterMessages.map((msg, i) => <MessageBubble key={i} msg={msg} />)}
          </div>
        </Collapsible>
      )}
    </>
  );
}

// -------------------------------------------------------------------
// Pause
// -------------------------------------------------------------------

function PauseInspector({ node }: { node: PauseSegmentNode }) {
  return (
    <>
      <div className="flex gap-3 p-5 border-b border-white/[0.06]">
        <StatCard label="Wait" value={formatDuration(node.wait_duration_ms)} mono />
        <StatCard label="Status" value={node.pause_status.replace(/_/g, " ")} />
      </div>

      {node.pending_tool_calls.length > 0 && (
        <Section title="Pending Tool Calls" icon="hourglass_top">
          {node.pending_tool_calls.map((tc) => (
            <div key={tc.tool_call_id} className="flex items-center gap-2 text-sm mb-2">
              <span className="material-symbols-outlined text-sm text-state-paused">build</span>
              <span className="font-mono text-text-primary">{tc.tool_name}</span>
              {tc.target && <span className="text-text-muted text-xs">({tc.target})</span>}
            </div>
          ))}
        </Section>
      )}

      {node.submitted_results.length > 0 && (
        <Section title="Submitted Results" icon="task_alt">
          {node.submitted_results.map((r) => (
            <div key={r.call_id} className="flex items-center gap-2 text-sm mb-2">
              <span className={`material-symbols-outlined text-sm ${r.success ? "text-state-success" : "text-state-error"}`}>
                {r.success ? "check_circle" : "cancel"}
              </span>
              <span className="font-mono text-text-primary">{r.tool_name}</span>
            </div>
          ))}
        </Section>
      )}

      {node.user_input && (
        <Section title="User Input" icon="chat">
          <pre className="text-text-secondary font-mono text-xs whitespace-pre-wrap">{node.user_input}</pre>
        </Section>
      )}
    </>
  );
}

// -------------------------------------------------------------------
// Simple node types
// -------------------------------------------------------------------

function RunStartedInspector({ node }: { node: RunStartedNode }) {
  return (
    <div className="flex gap-3 p-5 border-b border-white/[0.06]">
      <StatCard label="Agent" value={node.agent_name} />
    </div>
  );
}

function ErrorInspector({ node }: { node: ErrorNode }) {
  return (
    <Section title="Error Message" icon="error">
      <p className="text-state-error text-sm leading-relaxed">{node.error}</p>
    </Section>
  );
}

function FinishInspector({ node }: { node: FinishNode }) {
  return (
    <div className="flex gap-3 p-5 border-b border-white/[0.06]">
      <StatCard label="Status" value={node.status} className="text-state-success" />
    </div>
  );
}

// -------------------------------------------------------------------
// Shared components
// -------------------------------------------------------------------

function Section({ title, icon, children }: { title: string; icon?: string; children: React.ReactNode }) {
  return (
    <div className="p-5 border-b border-white/[0.06]">
      <h4 className="flex items-center gap-1.5 text-[10px] uppercase text-text-muted font-bold tracking-widest mb-3">
        {icon && <span className="material-symbols-outlined text-sm">{icon}</span>}
        {title}
      </h4>
      {children}
    </div>
  );
}

function Collapsible({ title, icon, defaultOpen = false, children }: {
  title: string; icon?: string; defaultOpen?: boolean; children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border-b border-white/[0.06]">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-1.5 p-5 text-left hover:bg-white/[0.02] transition-colors"
      >
        <span className="material-symbols-outlined text-sm text-text-muted transition-transform" style={{ transform: open ? "rotate(90deg)" : "rotate(0deg)" }}>
          chevron_right
        </span>
        {icon && <span className="material-symbols-outlined text-sm text-text-muted">{icon}</span>}
        <span className="text-[10px] uppercase text-text-muted font-bold tracking-widest">{title}</span>
      </button>
      {open && <div className="px-5 pb-5">{children}</div>}
    </div>
  );
}

function MessageBubble({ msg }: { msg: TraceMessage }) {
  const roleConfig: Record<string, { color: string; icon: string }> = {
    system: { color: "text-text-muted", icon: "psychology" },
    user: { color: "text-text-primary", icon: "person" },
    assistant: { color: "text-state-llm", icon: "smart_toy" },
    tool: { color: "text-state-tool", icon: "build" },
  };
  const cfg = roleConfig[msg.role] ?? roleConfig.user!;

  return (
    <div>
      <div className="flex items-center gap-1.5 mb-1">
        <span className={`material-symbols-outlined text-sm ${cfg.color}`}>{cfg.icon}</span>
        <span className={`text-xs font-bold uppercase tracking-wider ${cfg.color}`}>{msg.role}</span>
      </div>
      <pre className="text-text-secondary font-mono text-xs whitespace-pre-wrap leading-relaxed pl-6">
        {typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content, null, 2)}
      </pre>
    </div>
  );
}

function StatCard({ label, value, mono, className }: { label: string; value: string; mono?: boolean; className?: string }) {
  return (
    <div className="flex-1 p-3 rounded-lg bg-canvas border border-white/[0.04]">
      <span className="text-[10px] uppercase text-text-muted font-bold block mb-1">{label}</span>
      <span className={`text-sm ${mono ? "font-mono" : ""} ${className ?? "text-text-primary"}`}>{value}</span>
    </div>
  );
}

function Empty({ text }: { text: string }) {
  return <p className="text-text-muted text-xs italic">{text}</p>;
}

function JsonBlock({ data }: { data: Record<string, unknown> }) {
  return (
    <pre className="font-mono text-xs leading-relaxed whitespace-pre-wrap text-text-secondary">
      {formatJsonValue(data, 0)}
    </pre>
  );
}

function formatJsonValue(value: unknown, indent: number): string {
  const pad = "  ".repeat(indent);
  const inner = "  ".repeat(indent + 1);

  if (value === null || value === undefined) return "null";
  if (typeof value === "string") return `"${value}"`;
  if (typeof value === "number" || typeof value === "boolean") return String(value);

  if (Array.isArray(value)) {
    if (value.length === 0) return "[]";
    const items = value.map((v) => `${inner}${formatJsonValue(v, indent + 1)}`);
    return `[\n${items.join(",\n")}\n${pad}]`;
  }

  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    if (entries.length === 0) return "{}";
    const lines = entries.map(([k, v]) => `${inner}"${k}": ${formatJsonValue(v, indent + 1)}`);
    return `{\n${lines.join(",\n")}\n${pad}}`;
  }

  return String(value);
}
