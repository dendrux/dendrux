/** Formatting utilities for the dashboard. */

/** Format a duration in ms to human-readable string. */
export function formatDuration(ms: number | null | undefined): string {
  if (ms == null) return "-";
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const mins = Math.floor(ms / 60_000);
  const secs = ((ms % 60_000) / 1000).toFixed(0);
  return `${mins}m ${secs}s`;
}

/** Format a token count with commas. */
export function formatTokens(n: number): string {
  return n.toLocaleString();
}

/** Format a cost in USD. */
export function formatCost(usd: number | null | undefined): string {
  if (usd == null) return "-";
  if (usd < 0.01) return `$${usd.toFixed(4)}`;
  return `$${usd.toFixed(3)}`;
}

/** Format a timestamp to time-only (HH:MM:SS). */
export function formatTime(ts: string | null | undefined): string {
  if (!ts) return "-";
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  } catch {
    return ts;
  }
}

/** Format a timestamp to relative "2m ago" style. */
export function formatRelativeTime(ts: string | null | undefined): string {
  if (!ts) return "-";
  try {
    const d = new Date(ts);
    const now = Date.now();
    const diff = now - d.getTime();
    if (diff < 60_000) return "just now";
    if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
    if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
    return d.toLocaleDateString();
  } catch {
    return ts;
  }
}
