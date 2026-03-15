/** Runs Overview — the landing page. Quiet, scannable, subordinate to detail. */

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { fetchRuns } from "@/lib/api";
import { StatusBadge } from "@/components/StatusBadge";
import { formatTokens, formatCost, formatRelativeTime } from "@/lib/format";

export function RunsList() {
  const navigate = useNavigate();
  const [statusFilter, setStatusFilter] = useState<string>("");
  const [agentFilter, setAgentFilter] = useState<string>("");

  const { data, isLoading, error } = useQuery({
    queryKey: ["runs", statusFilter, agentFilter],
    queryFn: () =>
      fetchRuns({
        limit: 50,
        status: statusFilter || undefined,
        agent: agentFilter || undefined,
      }),
    refetchInterval: 10_000,
  });

  const runs = data?.runs ?? [];
  const pausedRuns = runs.filter(
    (r) => r.status === "waiting_client_tool" || r.status === "waiting_human_input"
  );

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="flex items-center justify-between px-4 sm:px-8 py-4 border-b border-white/[0.06] bg-canvas/80 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-white/10 rounded-lg flex items-center justify-center glow-white">
            <span className="material-symbols-outlined text-white text-xl">account_tree</span>
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight text-text-primary">
              Dendrite <span className="text-text-muted font-light">Dashboard</span>
            </h1>
          </div>
        </div>
        <span className="text-sm text-text-muted font-mono">{data?.total ?? 0} runs</span>
      </header>

      {/* Active pauses strip */}
      {pausedRuns.length > 0 && (
        <div className="px-4 sm:px-6 py-3 bg-state-paused/[0.04] border-b border-state-paused/10">
          <div className="flex items-center gap-2 text-xs">
            <span className="w-1.5 h-1.5 rounded-full bg-state-paused animate-pulse" />
            <span className="text-state-paused font-medium">
              {pausedRuns.length} run{pausedRuns.length > 1 ? "s" : ""} waiting on client
            </span>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="px-4 sm:px-6 py-3 border-b border-border-soft flex items-center gap-3">
        <FilterSelect
          value={statusFilter}
          onChange={setStatusFilter}
          placeholder="All statuses"
          options={[
            { value: "", label: "All statuses" },
            { value: "success", label: "Success" },
            { value: "error", label: "Error" },
            { value: "running", label: "Running" },
            { value: "waiting_client_tool", label: "Waiting on client" },
            { value: "waiting_human_input", label: "Waiting for input" },
            { value: "cancelled", label: "Cancelled" },
          ]}
        />
        <input
          type="text"
          value={agentFilter}
          onChange={(e) => setAgentFilter(e.target.value)}
          placeholder="Filter by agent..."
          className="bg-surface border border-border-soft rounded-lg px-3 py-1.5 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-border"
        />
        <span className="text-xs text-text-muted ml-auto" />
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        {isLoading ? (
          <TableSkeleton />
        ) : error ? (
          <div className="px-6 py-12 text-center text-state-error text-sm">
            Failed to load runs: {(error as Error).message}
          </div>
        ) : runs.length === 0 ? (
          <div className="px-6 py-12 text-center text-text-muted text-sm">
            No runs yet.
          </div>
        ) : (
          <table className="w-full">
            <thead>
              <tr className="border-b border-border-soft text-xs text-text-muted">
                <th className="text-left px-6 py-2.5 font-medium">Agent</th>
                <th className="text-left px-3 py-2.5 font-medium">Status</th>
                <th className="text-right px-3 py-2.5 font-medium">Iters</th>
                <th className="text-right px-3 py-2.5 font-medium">Tokens</th>
                <th className="text-right px-3 py-2.5 font-medium">Cost</th>
                <th className="text-right px-3 py-2.5 font-medium">Pauses</th>
                <th className="text-right px-6 py-2.5 font-medium">Time</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr
                  key={run.run_id}
                  onClick={() => navigate(`/runs/${run.run_id}`)}
                  className="border-b border-border-soft/50 hover:bg-surface cursor-pointer transition-colors duration-100"
                >
                  <td className="px-4 sm:px-6 py-3">
                    <div className="text-sm text-text-primary font-medium">{run.agent_name}</div>
                    <div className="text-[10px] text-text-muted font-mono mt-0.5">{run.run_id}</div>
                  </td>
                  <td className="px-3 py-3">
                    <StatusBadge status={run.status} />
                  </td>
                  <td className="px-3 py-3 text-right text-xs text-text-secondary font-mono">
                    {run.iteration_count}
                  </td>
                  <td className="px-3 py-3 text-right text-xs text-text-secondary font-mono">
                    {formatTokens(run.total_input_tokens + run.total_output_tokens)}
                  </td>
                  <td className="px-3 py-3 text-right text-xs text-text-secondary font-mono">
                    {formatCost(run.total_cost_usd)}
                  </td>
                  <td className="px-3 py-3 text-right text-xs font-mono">
                    {run.pause_count > 0 ? (
                      <span className="text-state-paused">{run.pause_count}</span>
                    ) : (
                      <span className="text-text-muted">0</span>
                    )}
                  </td>
                  <td className="px-4 sm:px-6 py-3 text-right text-xs text-text-muted">
                    {formatRelativeTime(run.created_at)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function FilterSelect({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder: string;
  options: { value: string; label: string }[];
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="bg-surface border border-border-soft rounded-lg px-3 py-1.5 text-xs text-text-primary focus:outline-none focus:border-border appearance-none"
    >
      {options.map((opt) => (
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  );
}

function TableSkeleton() {
  return (
    <div className="px-6 py-4 space-y-3 animate-pulse">
      {[...Array(8)].map((_, i) => (
        <div key={i} className="h-12 bg-surface rounded-lg" />
      ))}
    </div>
  );
}
