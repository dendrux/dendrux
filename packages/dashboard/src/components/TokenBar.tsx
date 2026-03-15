/** Token usage bar — per-iteration breakdown at the bottom of run detail. */

import type { TimelineNode, LLMCallNode } from "@/lib/types";
import { formatTokens, formatCost } from "@/lib/format";

interface TokenBarProps {
  nodes: TimelineNode[];
}

interface IterationUsage {
  iteration: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
}

export function TokenBar({ nodes }: TokenBarProps) {
  // Aggregate LLM calls by iteration
  const byIteration = new Map<number, IterationUsage>();

  for (const node of nodes) {
    if (node.type !== "llm_call") continue;
    const llm = node as LLMCallNode;
    const existing = byIteration.get(llm.iteration);
    if (existing) {
      existing.input_tokens += llm.input_tokens;
      existing.output_tokens += llm.output_tokens;
      existing.cost_usd += llm.cost_usd ?? 0;
    } else {
      byIteration.set(llm.iteration, {
        iteration: llm.iteration,
        input_tokens: llm.input_tokens,
        output_tokens: llm.output_tokens,
        cost_usd: llm.cost_usd ?? 0,
      });
    }
  }

  const iterations = Array.from(byIteration.values()).sort((a, b) => a.iteration - b.iteration);
  if (iterations.length === 0) return null;

  const maxTokens = Math.max(...iterations.map((it) => it.input_tokens + it.output_tokens));

  return (
    <div className="border-t border-border-soft px-6 py-4">
      <h3 className="text-xs font-medium text-text-muted uppercase tracking-wider mb-3">
        Token Usage by Iteration
      </h3>
      <div className="space-y-1.5">
        {iterations.map((it) => {
          const total = it.input_tokens + it.output_tokens;
          const widthPct = maxTokens > 0 ? (total / maxTokens) * 100 : 0;
          const inputPct = total > 0 ? (it.input_tokens / total) * 100 : 0;

          return (
            <div key={it.iteration} className="flex items-center gap-3 text-xs">
              <span className="text-text-muted w-6 text-right font-mono">{it.iteration}</span>
              <div className="flex-1 h-3 bg-surface rounded overflow-hidden">
                <div
                  className="h-full flex rounded"
                  style={{ width: `${Math.max(widthPct, 2)}%` }}
                >
                  <div
                    className="bg-state-llm/40"
                    style={{ width: `${inputPct}%` }}
                  />
                  <div
                    className="bg-state-llm"
                    style={{ width: `${100 - inputPct}%` }}
                  />
                </div>
              </div>
              <span className="text-text-secondary font-mono w-20 text-right">
                {formatTokens(total)}
              </span>
              <span className="text-text-muted font-mono w-16 text-right">
                {formatCost(it.cost_usd)}
              </span>
            </div>
          );
        })}
      </div>
      <div className="flex items-center gap-4 mt-2 text-[10px] text-text-muted">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-sm bg-state-llm/40" /> Input
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-sm bg-state-llm" /> Output
        </span>
      </div>
    </div>
  );
}
