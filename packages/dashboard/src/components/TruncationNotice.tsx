/** A visible notice that part of the data is not shown (bounded reads, never silent). */

export function TruncationNotice({ children }: { children: React.ReactNode }) {
  return (
    <div
      role="status"
      className="flex items-start gap-2 px-4 sm:px-8 py-2 text-xs text-state-paused bg-state-paused/[0.06] border-b border-state-paused/15"
    >
      <span className="material-symbols-outlined text-sm leading-4">warning</span>
      <span>{children}</span>
    </div>
  );
}
