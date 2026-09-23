/* eslint-disable @typescript-eslint/no-unused-vars -- props kept so a label can be restored */
import type { DataStatus, GateState, ProjectStatus } from "@/lib/data";

/**
 * The status labels (verified / baseline / in progress / negative result / not run yet /
 * sample data) were removed from the site on request, so these components render nothing.
 *
 * The signatures are kept so every page still compiles and the labels can be switched back on
 * by restoring a body here. What each label used to carry now lives in the surrounding copy:
 * measurements state their numbers and thresholds in text, and anything invented - the ledger's
 * sample block contents - says so in prose on the page itself.
 */

export function StatusTag(_props: { status: ProjectStatus; className?: string }) {
  return null;
}

export function SampleBadge(_props: { className?: string; title?: string }) {
  return null;
}

export function DataBadge(_props: { status: DataStatus; title?: string }) {
  return null;
}

export function GateBadge(_props: { state: GateState }) {
  return null;
}

export function StatusLegend() {
  return null;
}

/** Small neutral label used for section headings ("The core finding", "Explore"). Still in use. */
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
