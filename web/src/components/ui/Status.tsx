import type { DataStatus, GateState, ProjectStatus } from "@/lib/data";

const STATUS: Record<ProjectStatus, { label: string; cls: string; dot: string; title: string }> = {
  verified: {
    label: "Verified",
    cls: "text-verified border-verified/30 bg-verified/10",
    dot: "bg-verified",
    title: "Real, measured, from our own data",
  },
  baseline: {
    label: "Baseline",
    cls: "text-baseline border-baseline/30 bg-baseline/10",
    dot: "bg-baseline",
    title: "A real working method we compare against - not our novelty",
  },
  "in-progress": {
    label: "In progress",
    cls: "text-progress border-progress/35 bg-progress/10",
    dot: "bg-progress motion-safe:animate-pulse",
    title: "Our proposed method, still being validated - not final numbers",
  },
};

export function StatusTag({ status, className = "" }: { status: ProjectStatus; className?: string }) {
  const s = STATUS[status];
  return (
    <span
      title={s.title}
      className={`inline-flex shrink-0 items-center gap-1.5 rounded-full border px-2.5 py-0.5 font-mono text-[10px] font-medium uppercase tracking-[0.14em] ${s.cls} ${className}`}
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
      className={`sample-stripes inline-flex shrink-0 items-center gap-1.5 rounded-md border border-dashed border-slate-400/50 px-2 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-300 ${className}`}
    >
      Sample data
    </span>
  );
}

/** SAMPLE DATA badge only when the data file says the numbers are placeholders. */
export function DataBadge({ status, title }: { status: DataStatus; title?: string }) {
  return status === "sample" ? <SampleBadge title={title} /> : null;
}

const GATE: Record<GateState, { label: string; cls: string }> = {
  passed: { label: "Passed", cls: "text-verified border-verified/40 bg-verified/10" },
  failed: { label: "Failed", cls: "text-failed border-failed/40 bg-failed/10" },
  pending: { label: "Pending", cls: "text-slate-300 border-slate-500/40 bg-slate-500/10" },
  "not-run": { label: "Not run", cls: "text-slate-400 border-slate-600/50 border-dashed" },
};

export function GateBadge({ state }: { state: GateState }) {
  const g = GATE[state];
  return (
    <span
      className={`inline-flex items-center rounded-md border px-2 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] ${g.cls}`}
    >
      {g.label}
    </span>
  );
}

export function StatusLegend({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`flex flex-wrap items-center ${compact ? "gap-2" : "gap-3"}`}>
      <StatusTag status="verified" />
      <StatusTag status="baseline" />
      <StatusTag status="in-progress" />
      {!compact && <SampleBadge />}
    </div>
  );
}
