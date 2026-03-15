/**
 * PayloadInspector — sticky right panel.
 *
 * Shows full JSON payload for the selected timeline node.
 * Quiet background. Clear section hierarchy. Crossfade on change.
 */

import { motion, AnimatePresence } from "framer-motion";
import JsonView from "@uiw/react-json-view";
import { darkTheme } from "@uiw/react-json-view/dark";
import type { TimelineNode, TraceMessage } from "@/lib/types";
import { useDashboardStore } from "@/lib/store";
import { formatDuration, formatTokens, formatCost } from "@/lib/format";

interface PayloadInspectorProps {
  messages: Record<string, TraceMessage[]>;
}

export function PayloadInspector({ messages }: PayloadInspectorProps) {
  const { selectedNode, selectedIteration, clearSelection } = useDashboardStore();

  if (!selectedNode) {
    return (
      <div className="h-full flex items-center justify-center text-text-muted text-sm">
        Select a timeline node to inspect
      </div>
    );
  }

  const iterMessages = selectedIteration != null ? messages[String(selectedIteration)] : undefined;

  return (
    <div className="h-full flex flex-col bg-inspector">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border-soft">
        <h3 className="text-sm font-medium text-text-primary">Inspector</h3>
        <button
          onClick={clearSelection}
          className="text-text-muted hover:text-text-secondary transition-colors text-xs"
        >
          Close
        </button>
      </div>

      {/* Metadata band */}
      <InspectorMeta node={selectedNode} />

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-4 py-3">
        <AnimatePresence mode="wait">
          <motion.div
            key={selectedNode.sequence_index}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
          >
            {/* Node payload */}
            <InspectorSection title="Node Data">
              <JsonView
                value={selectedNode as unknown as Record<string, unknown>}
                style={{
                  ...dendriJsonTheme,
                  fontSize: "12px",
                  lineHeight: "1.5",
                  fontFamily: "var(--font-mono)",
                }}
                collapsed={1}
                enableClipboard={false}
              />
            </InspectorSection>

            {/* Iteration messages */}
            {iterMessages && iterMessages.length > 0 && (
              <InspectorSection title={`Messages (Iteration ${selectedIteration})`}>
                <div className="space-y-2">
                  {iterMessages.map((msg, i) => (
                    <div key={i} className="text-xs">
                      <span className={`font-medium ${
                        msg.role === "assistant" ? "text-state-llm" :
                        msg.role === "user" ? "text-text-primary" :
                        "text-text-muted"
                      }`}>
                        {msg.role}
                      </span>
                      <pre className="mt-1 text-text-secondary font-mono whitespace-pre-wrap leading-relaxed">
                        {typeof msg.content === "string"
                          ? msg.content.slice(0, 2000)
                          : JSON.stringify(msg.content, null, 2)}
                      </pre>
                    </div>
                  ))}
                </div>
              </InspectorSection>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}

function InspectorMeta({ node }: { node: TimelineNode }) {
  const items: { label: string; value: string }[] = [];

  items.push({ label: "Type", value: node.type });

  if ("iteration" in node) {
    items.push({ label: "Iteration", value: String((node as { iteration: number }).iteration) });
  }
  if ("input_tokens" in node) {
    const n = node as { input_tokens: number; output_tokens: number };
    items.push({ label: "Tokens", value: `${formatTokens(n.input_tokens)} / ${formatTokens(n.output_tokens)}` });
  }
  if ("cost_usd" in node && (node as { cost_usd: number | null }).cost_usd != null) {
    items.push({ label: "Cost", value: formatCost((node as { cost_usd: number }).cost_usd) });
  }
  if ("duration_ms" in node) {
    items.push({ label: "Duration", value: formatDuration((node as { duration_ms: number | null }).duration_ms) });
  }
  if ("wait_duration_ms" in node) {
    items.push({ label: "Wait", value: formatDuration((node as { wait_duration_ms: number | null }).wait_duration_ms) });
  }
  if ("target" in node) {
    items.push({ label: "Target", value: (node as { target: string }).target });
  }

  return (
    <div className="px-4 py-2 border-b border-border-soft flex flex-wrap gap-x-4 gap-y-1">
      {items.map((item, i) => (
        <span key={i} className="text-xs">
          <span className="text-text-muted">{item.label} </span>
          <span className="text-text-secondary font-mono">{item.value}</span>
        </span>
      ))}
    </div>
  );
}

function InspectorSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-4">
      <h4 className="text-xs font-medium text-text-muted mb-2 uppercase tracking-wider">
        {title}
      </h4>
      {children}
    </div>
  );
}

/** Custom JSON theme matching our warm-dark palette. */
const dendriJsonTheme = {
  ...darkTheme,
  "--w-rjv-background-color": "transparent",
  "--w-rjv-color": "#cdbfae",
  "--w-rjv-key-string": "#f3eadb",
  "--w-rjv-type-string-color": "#c6922b",
  "--w-rjv-type-int-color": "#4c7ff7",
  "--w-rjv-type-float-color": "#4c7ff7",
  "--w-rjv-type-boolean-color": "#4fb7c8",
  "--w-rjv-type-null-color": "#8e8172",
  "--w-rjv-info-color": "#8e8172",
  "--w-rjv-arrow-color": "#8e8172",
  "--w-rjv-curlybraces-color": "#8e8172",
  "--w-rjv-brackets-color": "#8e8172",
} as React.CSSProperties;
