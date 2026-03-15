/**
 * PauseBoundary — the signature Dendrite component.
 *
 * The flow line stops. The segment breathes. The UI acknowledges
 * waiting as a first-class state. The resumed edge reconnects
 * with calm precision.
 */

import { motion } from "framer-motion";
import type { PauseSegmentNode } from "@/lib/types";
import { formatDuration } from "@/lib/format";

interface PauseBoundaryProps {
  node: PauseSegmentNode;
  isSelected: boolean;
  onSelect: () => void;
}

export function PauseBoundary({ node, isSelected, onSelect }: PauseBoundaryProps) {
  const isLive = !node.resumed_at;
  const durationText = isLive ? "waiting..." : formatDuration(node.wait_duration_ms);

  return (
    <button
      onClick={onSelect}
      className={`w-full text-left rounded-[10px] transition-colors duration-120 border ${
        isSelected
          ? "bg-state-paused/8 border-state-paused/20"
          : "bg-state-paused/[0.04] border-transparent hover:bg-state-paused/8"
      }`}
    >
      {/* Top: Pause marker */}
      <div className="px-4 pt-4 pb-2">
        <div className="flex items-center gap-2 mb-1">
          <div className="w-[3px] h-4 rounded-full bg-state-paused" />
          <span className="text-sm font-medium text-state-paused">Paused</span>
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-state-paused/10 text-state-paused font-medium">
            {node.pause_status === "waiting_client_tool" ? "client tool" : "human input"}
          </span>
        </div>

        {/* Pending tool calls */}
        <div className="ml-[11px] pl-3 border-l border-dashed border-state-paused/30">
          {node.pending_tool_calls.map((tc) => (
            <div key={tc.tool_call_id} className="text-xs text-text-secondary py-0.5">
              <span className="text-state-paused/80 font-mono">{tc.tool_name}</span>
              {tc.target && (
                <span className="text-text-muted ml-1.5">({tc.target})</span>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Duration — the hero number */}
      <div className="flex items-center justify-center py-3 border-y border-dashed border-state-paused/15">
        {isLive ? (
          <motion.span
            className="font-mono text-xl font-medium text-state-paused"
            animate={{ opacity: [0.4, 1, 0.4] }}
            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
          >
            {durationText}
          </motion.span>
        ) : (
          <span className="font-mono text-xl font-medium text-state-paused">
            {durationText}
          </span>
        )}
      </div>

      {/* Bottom: Resume marker (if resumed) */}
      {node.resumed_at && (
        <div className="px-4 pt-2 pb-4">
          <div className="flex items-center gap-2 mb-1">
            <div className="w-[3px] h-4 rounded-full bg-state-resumed" />
            <span className="text-sm font-medium text-state-resumed">Resumed</span>
          </div>

          {node.submitted_results.length > 0 && (
            <div className="ml-[11px] pl-3 border-l border-solid border-state-resumed/30">
              {node.submitted_results.map((r) => (
                <div key={r.call_id} className="text-xs text-text-secondary py-0.5">
                  <span className="text-state-resumed/80 font-mono">{r.tool_name}</span>
                  {!r.success && (
                    <span className="text-state-error ml-1.5">(failed)</span>
                  )}
                </div>
              ))}
            </div>
          )}

          {node.user_input && (
            <p className="ml-[11px] pl-3 mt-1 text-xs text-text-secondary border-l border-solid border-state-resumed/30">
              "{node.user_input}"
            </p>
          )}
        </div>
      )}
    </button>
  );
}
