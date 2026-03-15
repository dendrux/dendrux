/** Client-side state — selected node, inspector visibility. */

import { create } from "zustand";
import type { TimelineNode } from "./types";

interface DashboardState {
  /** Currently selected timeline node (drives the inspector). */
  selectedNode: TimelineNode | null;
  /** Iteration messages to show in inspector context. */
  selectedIteration: number | null;

  selectNode: (node: TimelineNode | null) => void;
  clearSelection: () => void;
}

export const useDashboardStore = create<DashboardState>((set) => ({
  selectedNode: null,
  selectedIteration: null,

  selectNode: (node) =>
    set({
      selectedNode: node,
      selectedIteration: node && "iteration" in node ? (node as { iteration: number }).iteration : null,
    }),

  clearSelection: () =>
    set({ selectedNode: null, selectedIteration: null }),
}));
