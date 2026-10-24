/** Delegation panel — parent link, ancestry breadcrumb, children list, subtree summary. */

import { useNavigate } from "react-router-dom";
import type { DelegationInfo } from "@/lib/types";
import { formatTokens, formatCost } from "@/lib/format";
import { StatusBadge } from "./StatusBadge";
import { TruncationNotice } from "./TruncationNotice";

interface DelegationPanelProps {
  delegation: DelegationInfo;
}

export function DelegationPanel({ delegation }: DelegationPanelProps) {
  const { parent, children, ancestry, subtree_summary: ss, ancestry_complete } = delegation;

  // Nothing interesting to show for solo runs
  const hasDelegation = parent != null || children.length > 0;
  if (!hasDelegation) return null;

  return (
    <div className="border-b border-border-soft/50">
      {delegation.truncated && (
        <TruncationNotice>
          This delegation tree is very large, so only a bounded part is shown: the first{" "}
          {children.length} children and the nearest ancestors. Subtree totals are lower bounds.
        </TruncationNotice>
      )}

      {/* Ancestry breadcrumb */}
      {ancestry.length > 0 && (
        <div className="px-4 sm:px-8 py-2 flex items-center gap-1 text-xs text-text-muted overflow-x-auto">
          {!ancestry_complete && (
            <span className="text-state-paused mr-1" title="Ancestry chain is incomplete">...</span>
          )}
          {ancestry.map((a, i) => (
            <span key={a.run_id} className="flex items-center gap-1 flex-shrink-0">
              {i > 0 && <span className="text-text-muted/50 mx-0.5">/</span>}
              <AncestryLink runId={a.run_id} agentName={a.agent_name} />
            </span>
          ))}
          <span className="text-text-muted/50 mx-0.5">/</span>
          <span className="text-text-primary font-medium">current</span>
        </div>
      )}

      <div className="px-4 sm:px-8 py-3 flex flex-wrap items-start gap-4">
        {/* Parent link */}
        {parent && (
          <DelegationCard title="Parent">
            {parent.resolved ? (
              <RunLink runId={parent.run_id} agentName={parent.agent_name ?? ""} status={parent.status ?? "unknown"} />
            ) : (
              <div className="text-xs">
                <span className="text-text-muted font-mono">{parent.run_id}</span>
                <span className="text-state-paused ml-2 text-[10px]">unresolved</span>
              </div>
            )}
          </DelegationCard>
        )}

        {/* Children list */}
        {children.length > 0 && (
          <DelegationCard title={`Children (${children.length})`}>
            <div className="space-y-1.5">
              {children.map((c) => (
                <RunLink key={c.run_id} runId={c.run_id} agentName={c.agent_name} status={c.status} />
              ))}
            </div>
          </DelegationCard>
        )}

        {/* Subtree summary — only show when there are descendants */}
        {ss.descendant_count > 0 && (
          <DelegationCard title="Subtree">
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
              <StatRow label="Descendants" value={String(ss.descendant_count)} />
              <StatRow label="Max depth" value={String(ss.max_depth)} />
              <StatRow
                label="Tokens"
                value={formatTokens(ss.subtree_input_tokens + ss.subtree_output_tokens)}
                mono
              />
              <StatRow
                label="Cost"
                value={
                  formatCost(ss.subtree_cost_usd) +
                  (ss.unknown_cost_count > 0 ? ` (${ss.unknown_cost_count} unknown)` : "")
                }
                mono
              />
            </div>
            {Object.keys(ss.status_counts).length > 1 && (
              <div className="flex items-center gap-2 mt-2 flex-wrap">
                {Object.entries(ss.status_counts).map(([status, count]) => (
                  <span key={status} className="flex items-center gap-1">
                    <StatusBadge status={status} className="text-[10px] py-0 px-1.5" />
                    <span className="text-[10px] text-text-muted font-mono">{count}</span>
                  </span>
                ))}
              </div>
            )}
          </DelegationCard>
        )}
      </div>
    </div>
  );
}

function DelegationCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-surface/50 rounded-lg border border-border-soft/50 px-3 py-2 min-w-[160px]">
      <h4 className="text-[10px] uppercase tracking-widest text-text-muted font-bold mb-1.5">{title}</h4>
      {children}
    </div>
  );
}

function RunLink({ runId, agentName, status }: { runId: string; agentName: string; status: string }) {
  const navigate = useNavigate();
  return (
    <button
      onClick={() => navigate(`/runs/${runId}`)}
      className="flex items-center gap-2 hover:bg-white/[0.04] rounded px-1 -mx-1 py-0.5 transition-colors w-full text-left"
    >
      <span className="text-xs text-text-primary font-medium truncate">{agentName}</span>
      <StatusBadge status={status} className="text-[10px] py-0 px-1.5 flex-shrink-0" />
    </button>
  );
}

function AncestryLink({ runId, agentName }: { runId: string; agentName: string }) {
  const navigate = useNavigate();
  return (
    <button
      onClick={() => navigate(`/runs/${runId}`)}
      className="text-text-secondary hover:text-text-primary transition-colors"
    >
      {agentName}
    </button>
  );
}

function StatRow({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <>
      <span className="text-text-muted">{label}</span>
      <span className={`text-text-secondary text-right ${mono ? "font-mono" : ""}`}>{value}</span>
    </>
  );
}
