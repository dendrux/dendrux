/** Status badge — muted fill, thin outline, small color accent. */

const STATUS_STYLES: Record<string, { bg: string; text: string; dot: string }> = {
  success: {
    bg: "bg-state-success/10",
    text: "text-state-success",
    dot: "bg-state-success",
  },
  error: {
    bg: "bg-state-error/10",
    text: "text-state-error",
    dot: "bg-state-error",
  },
  running: {
    bg: "bg-state-llm/10",
    text: "text-state-llm",
    dot: "bg-state-llm",
  },
  waiting_client_tool: {
    bg: "bg-state-paused/10",
    text: "text-state-paused",
    dot: "bg-state-paused",
  },
  waiting_human_input: {
    bg: "bg-state-paused/10",
    text: "text-state-paused",
    dot: "bg-state-paused",
  },
  cancelled: {
    bg: "bg-text-muted/10",
    text: "text-text-muted",
    dot: "bg-text-muted",
  },
  max_iterations: {
    bg: "bg-state-paused/10",
    text: "text-state-paused",
    dot: "bg-state-paused",
  },
};

const STATUS_LABELS: Record<string, string> = {
  success: "Success",
  error: "Error",
  running: "Running",
  waiting_client_tool: "Waiting on client",
  waiting_human_input: "Waiting for input",
  cancelled: "Cancelled",
  max_iterations: "Max iterations",
};

interface StatusBadgeProps {
  status: string;
  className?: string;
}

export function StatusBadge({ status, className = "" }: StatusBadgeProps) {
  const style = STATUS_STYLES[status] ?? STATUS_STYLES.cancelled!;
  const label = STATUS_LABELS[status] ?? status;

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-xs font-medium ${style.bg} ${style.text} ${className}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${style.dot}`} />
      {label}
    </span>
  );
}
