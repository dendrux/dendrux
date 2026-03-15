/** Run detail page header — agent, status, timing, cost summary. */

import { useNavigate } from "react-router-dom";
import type { RunSummary } from "@/lib/types";
import { formatTokens, formatCost, formatTime } from "@/lib/format";
import { StatusBadge } from "./StatusBadge";

interface RunHeaderProps {
  summary: RunSummary;
  systemPrompt: string | null;
}

export function RunHeader({ summary, systemPrompt }: RunHeaderProps) {
  const navigate = useNavigate();

  return (
    <header className="border-b border-border-soft px-6 py-4">
      {/* Back link */}
      <button
        onClick={() => navigate("/")}
        className="text-text-muted text-xs hover:text-text-secondary transition-colors mb-3 flex items-center gap-1"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M19 12H5M12 19l-7-7 7-7" />
        </svg>
        All runs
      </button>

      {/* Title row */}
      <div className="flex items-center gap-3 mb-3">
        <h1 className="text-[22px] font-semibold leading-tight text-text-primary">
          {summary.agent_name}
        </h1>
        <StatusBadge status={summary.status} />
      </div>

      {/* Metadata band */}
      <div className="flex items-center gap-6 text-xs text-text-secondary">
        <MetaItem label="Run" value={summary.run_id} mono />
        {summary.model && <MetaItem label="Model" value={summary.model} />}
        <MetaItem
          label="Tokens"
          value={`${formatTokens(summary.total_input_tokens)} in / ${formatTokens(summary.total_output_tokens)} out`}
          mono
        />
        <MetaItem label="Cost" value={formatCost(summary.total_cost_usd)} mono />
        <MetaItem label="Iterations" value={String(summary.iteration_count)} mono />
        {summary.created_at && (
          <MetaItem label="Started" value={formatTime(summary.created_at)} />
        )}
      </div>

      {/* System prompt (collapsible) */}
      {systemPrompt && (
        <details className="mt-3">
          <summary className="text-xs text-text-muted cursor-pointer hover:text-text-secondary transition-colors">
            System prompt
          </summary>
          <pre className="mt-2 p-3 bg-surface rounded-lg text-xs text-text-secondary font-mono whitespace-pre-wrap leading-relaxed max-h-48 overflow-y-auto">
            {systemPrompt}
          </pre>
        </details>
      )}
    </header>
  );
}

function MetaItem({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-text-muted">{label}</span>
      <span className={mono ? "font-mono text-text-primary" : "text-text-primary"}>
        {value}
      </span>
    </div>
  );
}
