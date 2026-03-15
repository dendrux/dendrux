/** Run Detail — the hero page. Timeline left, inspector right. */

import { useEffect } from "react";
import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { fetchRunDetail } from "@/lib/api";
import { useDashboardStore } from "@/lib/store";
import { RunHeader } from "@/components/RunHeader";
import { Timeline } from "@/components/Timeline";
import { PayloadInspector } from "@/components/PayloadInspector";
import { TokenBar } from "@/components/TokenBar";

export function RunDetail() {
  const { runId } = useParams<{ runId: string }>();
  const clearSelection = useDashboardStore((s) => s.clearSelection);
  const selectedNode = useDashboardStore((s) => s.selectedNode);

  // Clear selection on mount and unmount
  useEffect(() => {
    clearSelection();
    return () => clearSelection();
  }, [runId, clearSelection]);

  // Keyboard: Escape closes inspector
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") clearSelection();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [clearSelection]);

  const { data, isLoading, error } = useQuery({
    queryKey: ["run-detail", runId],
    queryFn: () => fetchRunDetail(runId!),
    enabled: !!runId,
  });

  if (isLoading) return <LoadingSkeleton />;
  if (error) return <ErrorState error={error} />;
  if (!data) return <ErrorState error={new Error("Run not found")} />;

  const hasInspector = !!selectedNode;

  return (
    <div className="h-screen flex flex-col">
      <RunHeader summary={data.summary} systemPrompt={data.system_prompt} />

      <div className="flex-1 flex overflow-hidden">
        {/* Timeline — gets the most space */}
        <div
          className={`flex-1 overflow-y-auto px-6 transition-all duration-200 ${
            hasInspector ? "max-w-[calc(100%-420px)]" : ""
          }`}
        >
          {data.nodes.length > 0 ? (
            <>
              <Timeline nodes={data.nodes} />
              <TokenBar nodes={data.nodes} />
            </>
          ) : (
            <EmptyTimeline summary={data.summary} />
          )}
        </div>

        {/* Inspector — sticky right panel */}
        {hasInspector && (
          <div className="w-[420px] border-l border-border-soft overflow-y-auto flex-shrink-0">
            <PayloadInspector messages={data.messages_by_iteration} />
          </div>
        )}
      </div>
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="p-6 space-y-4 animate-pulse">
      <div className="h-6 w-48 bg-surface rounded" />
      <div className="h-4 w-96 bg-surface rounded" />
      <div className="space-y-2 mt-8">
        {[...Array(5)].map((_, i) => (
          <div key={i} className="flex items-center gap-4">
            <div className="w-2.5 h-2.5 rounded-full bg-surface" />
            <div className="h-16 flex-1 bg-surface rounded-[10px]" />
          </div>
        ))}
      </div>
    </div>
  );
}

function EmptyTimeline({ summary }: { summary: import("@/lib/types").RunSummary }) {
  return (
    <div className="py-12 px-6">
      <div className="text-center mb-8">
        <p className="text-text-muted text-sm">
          This run was created before timeline events were recorded.
        </p>
        <p className="text-text-muted text-xs mt-1">
          New runs will show the full timeline here.
        </p>
      </div>

      {/* Still show what we know from the run record */}
      <div className="max-w-lg mx-auto space-y-4">
        {summary.answer && (
          <div className="bg-surface rounded-[10px] p-4 border border-border-soft">
            <h4 className="text-xs font-medium text-text-muted uppercase tracking-wider mb-2">Answer</h4>
            <p className="text-sm text-text-primary leading-relaxed">{summary.answer}</p>
          </div>
        )}
        {summary.error && (
          <div className="bg-state-error/5 rounded-[10px] p-4 border border-state-error/20">
            <h4 className="text-xs font-medium text-state-error uppercase tracking-wider mb-2">Error</h4>
            <p className="text-sm text-state-error/80 leading-relaxed">{summary.error}</p>
          </div>
        )}
        {summary.input_text && (
          <div className="bg-surface rounded-[10px] p-4 border border-border-soft">
            <h4 className="text-xs font-medium text-text-muted uppercase tracking-wider mb-2">Input</h4>
            <p className="text-sm text-text-secondary leading-relaxed">{summary.input_text}</p>
          </div>
        )}
      </div>
    </div>
  );
}

function ErrorState({ error }: { error: Error }) {
  return (
    <div className="flex items-center justify-center h-screen">
      <div className="text-center">
        <p className="text-state-error text-sm font-medium">Failed to load run</p>
        <p className="text-text-muted text-xs mt-1">{error.message}</p>
      </div>
    </div>
  );
}
