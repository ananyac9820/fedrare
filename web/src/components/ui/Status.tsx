import type { DataStatus, GateState, ProjectStatus } from "@/lib/data";

const STATUS: Record<ProjectStatus, { label: string; cls: string; dot: string; title: string }> = {
  verified: {
    label: "Verified",
    cls: "text-verified bg-verified-soft border-verified/25",
    dot: "bg-verified",
    title: "Real, measured, from our own data",
  },
  baseline: {
    label: "Baseline",
    cls: "text-baseline bg-baseline-soft border-baseline/25",
    dot: "bg-baseline",
    title: "A real working method we compare against - not our novelty",
  },
  "in-progress": {
    label: "In progress",
    cls: "text-progress bg-progress-soft border-progress/25",
    dot: "bg-progress",
    title: "Being built or validated - not final",
  },
  failed: {
    label: "Failed",
    cls: "text-failed bg-failed-soft border-failed/25",
    dot: "bg-failed",
    title: "Measured, and did not meet its pre-set bar",
  },
  "not-run": {
    label: "Not run yet",
    cls: "text-notrun bg-notrun-soft border-notrun/40 border-dashed",
    dot: "bg-notrun",
    title: "No result exists - no numbers are shown",
  },
  blocked: {
    label: "Blocked",
    cls: "text-notrun bg-notrun-soft border-notrun/40 border-dashed",
    dot: "bg-progress",
    title: "Waiting on something else before it can run",
  },
};

export function StatusTag({ status, className = "" }: { status: ProjectStatus; className?: string }) {
  const s = STATUS[status];
  return (
    <span
      title={s.title}
      className={`inline-flex shrink-0 items-center gap-1.5 rounded-full border px-2.5 py-1 font-mono text-[10px] font-semibold uppercase leading-none tracking-[0.12em] ${s.cls} ${className}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${s.dot}`} aria-hidden />
      {s.label}
    </span>
  );
}

export function SampleBadge({ className = "", title }: { className?: string; title?: string }) {
  return (
    <span
      title={title ?? "Placeholder values for layout only - not results"}
      className={`sample-stripes inline-flex shrink-0 items-center rounded-full border border-dashed border-notrun/50 px-2.5 py-1 font-mono text-[10px] font-semibold uppercase leading-none tracking-[0.12em] text-notrun ${className}`}
    >
      Sample data
    </span>
  );
}

/** SAMPLE DATA badge only when the data file says the numbers are placeholders. */
export function DataBadge({ status, title }: { status: DataStatus; title?: string }) {
  return status === "sample" ? <SampleBadge title={title} /> : null;
}

const GATE_STATUS: Record<GateState, ProjectStatus> = {
  passed: "verified",
  failed: "failed",
  pending: "not-run",
  "not-run": "not-run",
};
const GATE_LABEL: Record<GateState, string> = {
  passed: "Passed",
  failed: "Failed",
  pending: "Pending",
  "not-run": "Not run",
};

export function GateBadge({ state }: { state: GateState }) {
  const s = STATUS[GATE_STATUS[state]];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-mono text-[10px] font-semibold uppercase leading-none tracking-[0.12em] ${s.cls}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${s.dot}`} aria-hidden />
      {GATE_LABEL[state]}
    </span>
  );
}

/** Small accent label that sits beside headings, like the reference's pill tags. */
export function Pill({ children, tone = "sand", className = "" }: {
  children: React.ReactNode;
  tone?: "sand" | "accent" | "paper";
  className?: string;
}) {
  const tones = {
    sand: "bg-sand text-muted border-sand-deep",
    accent: "bg-accent-soft text-accent border-accent/20",
    paper: "bg-paper text-muted border-line",
  };
  return (
    <span
      className={`inline-flex items-center rounded-full border px-3 py-1 font-mono text-[10px] font-medium uppercase tracking-[0.18em] ${tones[tone]} ${className}`}
    >
      {children}
    </span>
  );
}

export function StatusLegend() {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <StatusTag status="verified" />
      <StatusTag status="baseline" />
      <StatusTag status="in-progress" />
      <StatusTag status="failed" />
      <StatusTag status="not-run" />
      <SampleBadge />
    </div>
  );
}
