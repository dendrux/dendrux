/** Timeline — the hero component. Vertical spine with branching nodes. */

import { motion } from "framer-motion";
import type { TimelineNode } from "@/lib/types";
import { useDashboardStore } from "@/lib/store";
import { TimelineNodeCard } from "./TimelineNode";
import { PauseBoundary } from "./PauseBoundary";

interface TimelineProps {
  nodes: TimelineNode[];
}

export function Timeline({ nodes }: TimelineProps) {
  const { selectedNode, selectNode } = useDashboardStore();

  return (
    <div className="relative pl-8 py-4">
      {/* Vertical spine */}
      <div className="absolute left-[15px] top-0 bottom-0 w-[2px] bg-border-soft" />

      {nodes.map((node, i) => {
        const isSelected = selectedNode?.sequence_index === node.sequence_index;
        const isPause = node.type === "pause_segment";

        return (
          <motion.div
            key={node.sequence_index}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.22, delay: i * 0.04, ease: "easeOut" }}
            className="relative mb-0.5"
          >
            {/* Node marker on the spine */}
            <NodeMarker type={node.type} isPause={isPause} />

            {/* Pause gets the special treatment */}
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
          </motion.div>
        );
      })}
    </div>
  );
}

/** Small circle on the spine — colored by node type. */
function NodeMarker({ type, isPause }: { type: string; isPause: boolean }) {
  const colorMap: Record<string, string> = {
    run_started: "bg-text-muted",
    llm_call: "bg-state-llm",
    tool_call: "bg-state-tool",
    pause_segment: "bg-state-paused",
    finish: "bg-text-primary",
    error: "bg-state-error",
    cancelled: "bg-text-muted",
  };
  const color = colorMap[type] ?? "bg-text-muted";

  return (
    <div
      className={`absolute -left-8 top-3 w-2.5 h-2.5 rounded-full border-2 border-canvas ${color} ${
        isPause ? "ring-2 ring-state-paused/20" : ""
      }`}
    />
  );
}
