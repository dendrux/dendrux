/**
 * PauseBoundary — the signature Dendrux component.
 *
 * Centered. Dramatic. The flow line stops, the segment breathes,
 * the duration is the largest number on the page.
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
  const durationText = isLive ? "waiting" : formatDuration(node.wait_duration_ms);

  return (
    <button
      onClick={onSelect}
      className={`w-full text-left transition-all duration-150 ${
        isSelected ? "opacity-100" : "opacity-90 hover:opacity-100"
      }`}
    >
      {/* Pause card — centered, dramatic */}
      <div className="p-6 sm:p-8 rounded-2xl bg-state-paused/[0.05] border border-state-paused/10 flex flex-col items-center justify-center text-center">
        <span className="text-[10px] uppercase tracking-[0.2em] text-state-paused font-bold mb-2">
          Suspended Bridge
        </span>

        {/* The hero number */}
        {isLive ? (
          <motion.span
            className="text-4xl sm:text-5xl font-mono font-medium text-state-paused tracking-tighter"
            animate={{ opacity: [0.4, 1, 0.4] }}
            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
          >
            {durationText}
          </motion.span>
        ) : (
          <span className="text-4xl sm:text-5xl font-mono font-medium text-state-paused tracking-tighter">
            {durationText}
          </span>
        )}

        {/* Pending tool calls */}
        <div className="mt-4 text-text-muted text-sm max-w-xs">
          {node.pending_tool_calls.map((tc) => (
            <span key={tc.tool_call_id} className="block">
              Awaiting <span className="font-mono text-state-paused/80">{tc.tool_name}</span>
              {tc.target && <span className="text-text-muted"> ({tc.target})</span>}
            </span>
          ))}
        </div>
      </div>

      {/* Resume section */}
      {node.resumed_at && (
        <div className="flex items-start gap-4 mt-6">
          {/* Resume icon */}
          <div className="w-10 h-10 rounded-full bg-surface border border-state-resumed flex items-center justify-center shadow-lg flex-shrink-0">
            <span className="material-symbols-outlined text-state-resumed text-xl">play_arrow</span>
          </div>
          <div className="pt-2 min-w-0">
            <h3 className="text-lg font-bold text-text-primary">Resumed</h3>
            <div className="flex flex-wrap items-center gap-3 text-sm text-text-muted mt-1">
              {node.submitted_results.length > 0 && (
                <span className="flex items-center gap-1">
                  <span className="material-symbols-outlined text-sm">verified_user</span>
                  {node.submitted_results.length} result{node.submitted_results.length > 1 ? "s" : ""} submitted
                </span>
              )}
              {node.user_input && (
                <span className="flex items-center gap-1">
                  <span className="material-symbols-outlined text-sm">chat</span>
                  User responded
                </span>
              )}
            </div>
          </div>
        </div>
      )}
    </button>
  );
}
