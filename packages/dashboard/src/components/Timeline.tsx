/** Timeline — the hero component. Vertical spine with glowing icon markers. */

import { motion } from "framer-motion";
import type { TimelineNode } from "@/lib/types";
import { useDashboardStore } from "@/lib/store";
import { TimelineNodeCard } from "./TimelineNode";
import { PauseBoundary } from "./PauseBoundary";

interface TimelineProps {
  nodes: TimelineNode[];
}

const NODE_CONFIG: Record<string, {
  icon: string; color: string; border: string; spine: string; glow: string;
}> = {
  run_started: { icon: "play_circle", color: "text-text-muted", border: "border-white/10", spine: "bg-white/10", glow: "glow-white" },
  llm_call: { icon: "neurology", color: "text-state-llm", border: "border-state-llm/50", spine: "bg-state-llm", glow: "glow-blue" },
  tool_call: { icon: "database", color: "text-state-tool", border: "border-state-tool/50", spine: "bg-state-tool", glow: "glow-green" },
  pause_segment: { icon: "pause", color: "text-state-paused", border: "border-state-paused/40", spine: "", glow: "glow-yellow" },
  finish: { icon: "check", color: "text-canvas", border: "border-white/20", spine: "bg-white/15", glow: "glow-white" },
  error: { icon: "error", color: "text-state-error", border: "border-state-error/50", spine: "bg-state-error", glow: "glow-red" },
  cancelled: { icon: "cancel", color: "text-text-muted", border: "border-white/10", spine: "bg-white/10", glow: "glow-white" },
};

export function Timeline({ nodes }: TimelineProps) {
  const { selectedNode, selectNode } = useDashboardStore();

  return (
    <div className="relative">
      {nodes.map((node, i) => {
        const isSelected = selectedNode?.sequence_index === node.sequence_index;
        const isPause = node.type === "pause_segment";
        const isFinish = node.type === "finish";
        const cfg = NODE_CONFIG[node.type] ?? NODE_CONFIG.run_started!;

        return (
          <motion.div
            key={node.sequence_index}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.22, delay: i * 0.04, ease: "easeOut" }}
            className={`relative flex gap-4 ${isPause ? "py-4" : ""} mb-8`}
          >
            {/* Left column: icon + spine */}
            <div className="relative flex-shrink-0 w-10 flex flex-col items-center">
              {/* Spine */}
              {!isPause ? (
                <div className={`absolute left-1/2 -translate-x-1/2 top-0 w-[3px] h-full ${cfg.spine} rounded-full`} />
              ) : (
                <div className="absolute left-1/2 -translate-x-1/2 top-0 bottom-0 w-px spine-dashed" />
              )}

              {/* Icon circle with glow */}
              {isPause ? (
                <div className={`relative w-10 h-10 rounded-full bg-state-paused/10 border border-state-paused/30 flex items-center justify-center z-10 ${cfg.glow}`}>
                  <span className="material-symbols-outlined text-state-paused text-xl">pause</span>
                </div>
              ) : isFinish ? (
                <div className={`relative w-10 h-10 rounded-full bg-white/10 border border-white/20 flex items-center justify-center z-10 shadow-lg ${cfg.glow}`}>
                  <span className="material-symbols-outlined text-white text-xl font-bold">check</span>
                </div>
              ) : (
                <div className={`relative w-10 h-10 rounded-full bg-surface border ${cfg.border} flex items-center justify-center z-10 ${cfg.glow}`}>
                  <span className={`material-symbols-outlined ${cfg.color} text-xl`}>{cfg.icon}</span>
                </div>
              )}
            </div>

            {/* Right column: content */}
            <div className="flex-1 min-w-0 pt-1">
              {isPause ? (
                <PauseBoundary
                  node={node}
                  isSelected={isSelected}
                  onSelect={() => selectNode(isSelected ? null : node)}
                />
              ) : (
                <TimelineNodeCard
                  node={node}
                  isSelected={isSelected}
                  onSelect={() => selectNode(isSelected ? null : node)}
                />
              )}
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}
