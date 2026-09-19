"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ChartTooltip } from "@/components/ui/ChartTooltip";
import { ParallaxSection, Reveal, SectionHeading, useReducedMotionSafe } from "@/components/ui/Motion";
import { DataBadge, GateBadge, StatusTag } from "@/components/ui/Status";
import type { BaselineResults, G0a } from "@/lib/data";

const RULE_COLORS: Record<string, string> = { fedavg: "#64748b", camp_a: "#60a5fa" };

function BaselineChart({ data }: { data: BaselineResults }) {
  const reduced = useReducedMotionSafe();
  const sample = data._meta.status === "sample";
  const classes = Object.keys(data.rules[0]?.rareF1 ?? {});
  const rows = [
    ...classes.map((c) => ({ name: `${c} F1`, ...Object.fromEntries(data.rules.map((r) => [r.label, r.rareF1[c]])) })),
    { name: "Rare macro-F1", ...Object.fromEntries(data.rules.map((r) => [r.label, r.rareMacroF1])) },
  ];

  return (
    <div className="relative h-72">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} barGap={4} margin={{ top: 16, right: 8, bottom: 0, left: -18 }}>
          <defs>
            {data.rules.map((r) => (
              <pattern key={r.id} id={`hatch-${r.id}`} width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                <rect width="8" height="8" fill={RULE_COLORS[r.id] ?? "#94a3b8"} fillOpacity={0.18} />
                <line x1="0" y1="0" x2="0" y2="8" stroke={RULE_COLORS[r.id] ?? "#94a3b8"} strokeWidth="3" strokeOpacity={0.6} />
              </pattern>
            ))}
          </defs>
          <CartesianGrid stroke="rgb(148 163 184 / 0.08)" vertical={false} />
          <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 12 }} axisLine={false} tickLine={false} />
          <YAxis domain={[0, 1]} tick={{ fill: "#64748b", fontSize: 11 }} axisLine={false} tickLine={false} />
          <Tooltip content={<ChartTooltip digits={2} />} cursor={{ fill: "rgb(148 163 184 / 0.06)" }} />
          <Legend wrapperStyle={{ fontSize: 12 }} iconType="circle" />
          {data.rules.map((r) => (
            <Bar
              key={r.id}
              dataKey={r.label}
              fill={sample ? `url(#hatch-${r.id})` : RULE_COLORS[r.id] ?? "#94a3b8"}
              stroke={RULE_COLORS[r.id] ?? "#94a3b8"}
              strokeOpacity={sample ? 0.8 : 0}
              strokeDasharray={sample ? "4 3" : undefined}
              radius={[4, 4, 0, 0]}
              isAnimationActive={!reduced}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
      {sample && (
        <div aria-hidden className="pointer-events-none absolute inset-0 flex items-center justify-center overflow-hidden">
          <span className="-rotate-12 whitespace-nowrap rounded-lg border-2 border-dashed border-slate-400/40 bg-ink-950/60 px-5 py-2 font-mono text-lg font-bold uppercase tracking-[0.3em] text-slate-300/80 backdrop-blur-[2px]">
            Sample data · not results
          </span>
        </div>
      )}
    </div>
  );
}

function G0aChart({ g0a }: { g0a: G0a }) {
  const reduced = useReducedMotionSafe();
  const rows = g0a.classNames.map((name, i) => ({
    name: g0a.rareIds.includes(i) ? `${name} ★` : name,
    Spearman: +g0a.perClassSpearman[i].toFixed(2),
  }));
  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} layout="vertical" margin={{ top: 26, right: 36, bottom: 8, left: 8 }}>
          <CartesianGrid stroke="rgb(148 163 184 / 0.08)" horizontal={false} />
          <XAxis type="number" domain={[-1, 1]} ticks={[-1, -0.5, 0, 0.5, 0.7, 1]} tick={{ fill: "#64748b", fontSize: 11 }} axisLine={false} tickLine={false} />
          <YAxis type="category" dataKey="name" width={170} tick={{ fill: "#94a3b8", fontSize: 12 }} axisLine={false} tickLine={false} />
          <Tooltip content={<ChartTooltip digits={2} />} cursor={{ fill: "rgb(148 163 184 / 0.06)" }} />
          <ReferenceLine x={0} stroke="#475569" />
          <ReferenceLine
            x={g0a.threshold}
            stroke="#e2e8f0"
            strokeDasharray="5 4"
            label={{ value: `needed ≥ ${g0a.threshold}`, position: "top", fill: "#e2e8f0", fontSize: 11 }}
          />
          <Bar dataKey="Spearman" radius={[0, 4, 4, 0]} isAnimationActive={!reduced}>
            {rows.map((r) => (
              <Cell key={r.name} fill={r.Spearman < 0 ? "#fb7185" : "#94a3b8"} fillOpacity={0.85} />
            ))}
            <LabelList dataKey="Spearman" position="right" fill="#cbd5e1" fontSize={11} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function Results({ baselines, g0a }: { baselines: BaselineResults; g0a: G0a }) {
  const sample = baselines._meta.status === "sample";
  return (
    <ParallaxSection id="results" glow="blue" className="py-28 md:py-36">
      <div className="mx-auto max-w-7xl px-6">
        <SectionHeading eyebrow="03 · Results so far" title="What has actually been measured.">
          Two things so far: baselines ready to compare against, and the first check of EARN&apos;s
          evidence signal, which failed.
        </SectionHeading>

        <div className="grid gap-8 lg:grid-cols-2">
          <Reveal tilt className={`rounded-3xl border p-7 backdrop-blur ${sample ? "sample-stripes border-dashed border-slate-500/40 bg-ink-900/60" : "border-line bg-ink-900/70"}`}>
            <div className="flex flex-wrap items-center gap-2">
              <StatusTag status="baseline" />
              <DataBadge status={baselines._meta.status} />
            </div>
            <h3 className="mt-4 font-display text-xl font-semibold text-white">
              FedAvg vs Camp A on rare diseases
            </h3>
            <p className="mt-2 text-sm text-slate-400">
              Camp A gives extra weight to hospitals that look like rare-disease experts - the
              approach EARN is meant to make safe. Split {baselines.split}, rare-class F1.
            </p>
            <div className="mt-6">
              <BaselineChart data={baselines} />
            </div>
            <p className="mt-4 text-xs leading-relaxed text-slate-500">
              {baselines._meta.note}
              {baselines._meta.wouldComeFrom && (
                <>
                  {" "}Real values will come from <code className="font-mono text-slate-400">{baselines._meta.wouldComeFrom}</code>.
                </>
              )}
            </p>
          </Reveal>

          <Reveal tilt delay={0.1} className="rounded-3xl border border-failed/30 bg-ink-900/70 p-7 backdrop-blur">
            <div className="flex flex-wrap items-center gap-2">
              <StatusTag status="verified" />
              <StatusTag status="in-progress" />
              <GateBadge state={g0a.passed ? "passed" : "failed"} />
            </div>
            <h3 className="mt-4 font-display text-xl font-semibold text-white">
              EARN&apos;s first check: does the evidence signal work?
            </h3>
            <div className="mt-4 flex items-end gap-6">
              <div>
                <p className="font-display text-6xl font-semibold tracking-tight text-failed">
                  {g0a.statistic.toFixed(2)}
                </p>
                <p className="mt-1 font-mono text-xs uppercase tracking-[0.15em] text-slate-500">mean Spearman</p>
              </div>
              <div className="pb-2 text-sm text-slate-400">
                needed ≥ <span className="font-mono text-white">{g0a.threshold}</span> to pass
                <br />
                Gate {g0a.gate}, measured on real data
              </div>
            </div>
            <div className="mt-4">
              <G0aChart g0a={g0a} />
            </div>
            <p className="mt-2 text-sm leading-relaxed text-slate-400">
              The signal came out <span className="text-failed">inverted</span>: hospitals with{" "}
              <em>no</em> images of a disease showed the largest change for it. EARN can&apos;t be
              built on it until that&apos;s fixed. Whether to retry is a team decision. ★ = rare
              disease.
            </p>
          </Reveal>
        </div>
      </div>
    </ParallaxSection>
  );
}
