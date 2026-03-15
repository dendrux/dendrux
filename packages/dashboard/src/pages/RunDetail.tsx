/** Run Detail — the hero page. Timeline left, inspector right. Responsive. */

import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { fetchRunDetail } from "@/lib/api";
import { useDashboardStore } from "@/lib/store";
import { RunHeader } from "@/components/RunHeader";
import { Timeline } from "@/components/Timeline";
import { PayloadInspector } from "@/components/PayloadInspector";
import { TokenBar } from "@/components/TokenBar";
import type { RunSummary } from "@/lib/types";

export function RunDetail() {
  const { runId } = useParams<{ runId: string }>();
  const clearSelection = useDashboardStore((s) => s.clearSelection);
  const selectedNode = useDashboardStore((s) => s.selectedNode);
  const [inspectorOpen, setInspectorOpen] = useState(false);

  // Open inspector when a node is selected
  useEffect(() => {
    if (selectedNode) setInspectorOpen(true);
  }, [selectedNode]);

  // Clear selection on mount/unmount
  useEffect(() => {
    clearSelection();
    return () => clearSelection();
  }, [runId, clearSelection]);

  // Keyboard: Escape closes inspector
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        clearSelection();
        setInspectorOpen(false);
      }
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

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      <RunHeader summary={data.summary} systemPrompt={data.system_prompt} />

      <main className="flex flex-1 overflow-hidden relative">
        {/* Timeline — takes remaining space */}
        <section className="flex-1 min-w-0 overflow-y-auto py-10 px-4 sm:px-8 lg:px-12">
          <div className="max-w-2xl mx-auto">
            {data.nodes.length > 0 ? (
              <>
                <Timeline nodes={data.nodes} />
                <TokenBar nodes={data.nodes} />
              </>
            ) : (
              <EmptyTimeline summary={data.summary} />
            )}
          </div>
        </section>

        {/* Inspector — slides in on selection, responsive width */}
        {inspectorOpen && (
          <aside className="
            w-full sm:w-[380px] lg:w-[440px] xl:w-[480px]
            absolute sm:relative inset-0 sm:inset-auto
            bg-surface border-l border-accent-brass/10
            flex flex-col shadow-2xl z-40
          ">
            <PayloadInspector
              messages={data.messages_by_iteration}
              systemPrompt={data.system_prompt}
              onClose={() => {
                clearSelection();
                setInspectorOpen(false);
              }}
            />
          </aside>
        )}
      </main>
    </div>
  );
}

function EmptyTimeline({ summary }: { summary: RunSummary }) {
  return (
    <div className="py-12">
      <div className="text-center mb-8">
        <span className="material-symbols-outlined text-4xl text-accent-brass/30 mb-3 block">timeline</span>
        <p className="text-text-muted text-sm">
          This run was created before timeline events were recorded.
        </p>
        <p className="text-text-muted text-xs mt-1">
          New runs will show the full timeline here.
        </p>
      </div>
      <div className="max-w-lg mx-auto space-y-4">
        {summary.answer && (
          <div className="p-5 rounded-xl bg-surface border border-border-soft">
            <h4 className="text-[10px] uppercase text-accent-brass font-bold tracking-widest mb-2">Answer</h4>
            <p className="text-sm text-text-primary leading-relaxed">{summary.answer}</p>
          </div>
        )}
        {summary.error && (
          <div className="p-5 rounded-xl bg-state-error/5 border border-state-error/20">
            <h4 className="text-[10px] uppercase text-state-error font-bold tracking-widest mb-2">Error</h4>
            <p className="text-sm text-state-error/80 leading-relaxed">{summary.error}</p>
          </div>
        )}
        {summary.input_text && (
          <div className="p-5 rounded-xl bg-surface border border-border-soft">
            <h4 className="text-[10px] uppercase text-accent-brass font-bold tracking-widest mb-2">Input</h4>
            <p className="text-sm text-text-secondary leading-relaxed">{summary.input_text}</p>
          </div>
        )}
      </div>
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="p-8 sm:p-12 space-y-6 animate-pulse">
      <div className="h-8 w-48 bg-surface rounded-lg" />
      <div className="h-4 w-96 max-w-full bg-surface rounded" />
      <div className="space-y-8 mt-12">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="flex items-start gap-4">
            <div className="w-10 h-10 rounded-full bg-surface flex-shrink-0" />
            <div className="flex-1 space-y-2 min-w-0">
              <div className="h-5 w-64 max-w-full bg-surface rounded" />
              <div className="h-3 w-48 max-w-full bg-surface rounded" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ErrorState({ error }: { error: Error }) {
  return (
    <div className="flex items-center justify-center h-screen px-4">
      <div className="text-center">
        <span className="material-symbols-outlined text-4xl text-state-error/50 mb-3 block">error</span>
        <p className="text-state-error text-sm font-medium">Failed to load run</p>
        <p className="text-text-muted text-xs mt-1">{error.message}</p>
      </div>
    </div>
  );
}
