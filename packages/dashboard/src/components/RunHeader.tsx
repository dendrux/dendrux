/** Run detail header — branded, responsive, clean white accents. */

import { useNavigate } from "react-router-dom";
import type { RunSummary } from "@/lib/types";
import { formatTokens, formatCost } from "@/lib/format";
import { StatusBadge } from "./StatusBadge";

interface RunHeaderProps {
  summary: RunSummary;
  systemPrompt?: string | null;
}

export function RunHeader({ summary }: RunHeaderProps) {
  const navigate = useNavigate();

  return (
    <header className="flex flex-col sm:flex-row sm:items-center justify-between px-4 sm:px-8 py-3 sm:py-4 border-b border-white/[0.06] bg-canvas/80 backdrop-blur-md z-50 gap-3 sm:gap-0">
      {/* Left: logo + run info */}
      <div className="flex items-center gap-4 sm:gap-6 min-w-0">
        <button
          onClick={() => navigate("/")}
          className="flex items-center gap-2 hover:opacity-80 transition-opacity flex-shrink-0"
        >
          <div className="w-8 h-8 bg-white/10 rounded-lg flex items-center justify-center glow-white">
            <span className="material-symbols-outlined text-white text-xl">account_tree</span>
          </div>
          <h1 className="text-base sm:text-lg font-bold tracking-tight text-text-primary whitespace-nowrap">
            Dendrite <span className="text-text-muted font-light">Run</span>
          </h1>
        </button>

        <div className="h-4 w-px bg-white/10 hidden sm:block flex-shrink-0" />

        <div className="flex flex-col min-w-0">
          <span className="text-[10px] uppercase tracking-widest text-text-secondary font-bold truncate">
            {summary.agent_name}
          </span>
          <span className="text-xs sm:text-sm font-mono text-text-muted truncate">{summary.run_id}</span>
        </div>

        <StatusBadge status={summary.status} />
      </div>

      {/* Right: metadata */}
      <div className="flex items-center gap-3 sm:gap-4 flex-shrink-0 overflow-x-auto">
        {summary.model && (
          <MetaBlock label="Model" value={summary.model} />
        )}
        <MetaBlock label="Tokens" value={formatTokens(summary.total_input_tokens + summary.total_output_tokens)} mono />
        <MetaBlock label="Cost" value={formatCost(summary.total_cost_usd)} mono />

        {/* System prompt is now shown inside the LLM Call inspector */}
      </div>
    </header>
  );
}

function MetaBlock({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex flex-col items-end flex-shrink-0">
      <span className="text-[10px] uppercase tracking-widest text-text-muted font-bold">{label}</span>
      <span className={`text-xs sm:text-sm text-text-secondary ${mono ? "font-mono" : ""}`}>{value}</span>
    </div>
  );
}
