"use client";

import { motion } from "framer-motion";
import { Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CountUp, ParallaxSection, Reveal, SectionHeading, useReducedMotionSafe } from "@/components/ui/Motion";
import { StatusTag } from "@/components/ui/Status";
import { ChartTooltip } from "@/components/ui/ChartTooltip";
import type { CentreMismatch } from "@/lib/data";

function GapBar({ label, value, max, color, delay }: {
  label: string;
  value: number;
  max: number;
  color: string;
  delay: number;
}) {
  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between gap-4">
        <span className="text-sm text-slate-300">{label}</span>
        <span className="font-display text-2xl font-semibold text-white">{(100 * value).toFixed(1)}%</span>
      </div>
      <div className="h-4 overflow-hidden rounded-full bg-ink-800">
        <motion.div
          className="h-full rounded-full"
          style={{ background: color, transformOrigin: "0% 50%" }}
          initial={{ scaleX: 0 }}
          whileInView={{ scaleX: value / max }}
          viewport={{ once: true, amount: 0.8 }}
          transition={{ duration: 1.3, delay, ease: [0.22, 1, 0.36, 1] }}
        />
      </div>
    </div>
  );
}

export function Problem({ mismatch, specialistId, ratio, rare, rule, coverage, sources }: {
  mismatch: CentreMismatch[];
  specialistId: number;
  ratio: number;
  rare: { name: string; sharePct: number; holders: number }[];
  rule: { headRatioDivisor: number; cutoffPct: number; headClass: string; headSharePct: number; holderMinImages: number };
  coverage: { zeroRareCentres: number[] };
  sources: string[];
}) {
  const reduced = useReducedMotionSafe();
  const s = mismatch[specialistId];
  const barMax = Math.max(s.rareShare, s.fedavgWeight) * 1.1;
  const chartData = mismatch.map((m) => ({
    name: `C${m.id}`,
    "Share of rare-disease images": +(100 * m.rareShare).toFixed(1),
    "Share of FedAvg weight": +(100 * m.fedavgWeight).toFixed(1),
    id: m.id,
  }));

  return (
    <ParallaxSection id="problem" glow="teal" className="py-28 md:py-36">
      <div className="mx-auto max-w-7xl px-6">
        <SectionHeading
          eyebrow="01 · The problem"
          title="The hospital that knows the most about rare diseases gets one of the smaller votes."
          tags={<StatusTag status="verified" />}
        >
          Standard federated averaging (FedAvg) weights each hospital by how many images it has in
          total, not by what it knows. On our data that shortchanges the rare-disease specialist.
        </SectionHeading>

        <div className="grid gap-10 lg:grid-cols-[1.05fr_1fr]">
          <Reveal tilt className="rounded-3xl border border-line bg-ink-900/70 p-8 backdrop-blur md:p-10">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="font-mono text-xs uppercase tracking-[0.2em] text-slate-400">
                Centre {specialistId} · training split
              </p>
              <StatusTag status="verified" />
            </div>
            <p className="mt-6 font-display text-[7rem] font-semibold leading-none tracking-tighter text-white md:text-[9rem]">
              <CountUp to={ratio} decimals={2} suffix="×" />
            </p>
            <p className="mt-2 max-w-md text-lg text-slate-300">
              mismatch between what Centre {specialistId} knows about rare diseases and how much
              FedAvg listens to it.
            </p>

            <div className="mt-10 space-y-6">
              <GapBar label={`Rare-disease training images held by Centre ${specialistId}`} value={s.rareShare} max={barMax} color="#2dd4bf" delay={0.1} />
              <GapBar label={`Aggregation weight FedAvg gives Centre ${specialistId}`} value={s.fedavgWeight} max={barMax} color="#64748b" delay={0.35} />
            </div>
            <p className="mt-6 text-sm text-slate-500">
              {s.rareTrainImages} of {mismatch.reduce((a, m) => a + m.rareTrainImages, 0)} rare-disease
              training images · {s.trainImages.toLocaleString("en-US")} of{" "}
              {mismatch.reduce((a, m) => a + m.trainImages, 0).toLocaleString("en-US")} training images overall
            </p>
          </Reveal>

          <div className="flex flex-col gap-6">
            <Reveal delay={0.1} from="right" className="rounded-3xl border border-line bg-ink-900/70 p-6 backdrop-blur">
              <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
                <h3 className="font-display text-lg font-semibold text-white">Every hospital, same comparison</h3>
                <StatusTag status="verified" />
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} barGap={3} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
                    <CartesianGrid stroke="rgb(148 163 184 / 0.08)" vertical={false} />
                    <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 12 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: "#64748b", fontSize: 11 }} axisLine={false} tickLine={false} unit="%" />
                    <Tooltip content={<ChartTooltip unit="%" />} cursor={{ fill: "rgb(148 163 184 / 0.06)" }} />
                    <Legend wrapperStyle={{ fontSize: 12, color: "#94a3b8" }} iconType="circle" />
                    <Bar dataKey="Share of rare-disease images" fill="#2dd4bf" radius={[4, 4, 0, 0]} isAnimationActive={!reduced}>
                      {chartData.map((d) => (
                        <Cell key={d.id} fillOpacity={d.id === specialistId ? 1 : 0.45} />
                      ))}
                    </Bar>
                    <Bar dataKey="Share of FedAvg weight" fill="#64748b" radius={[4, 4, 0, 0]} isAnimationActive={!reduced}>
                      {chartData.map((d) => (
                        <Cell key={d.id} fillOpacity={d.id === specialistId ? 1 : 0.5} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="mt-3 text-xs text-slate-500">
                Centre {specialistId} is the only hospital whose rare-disease share is far above its
                weight. Centre {coverage.zeroRareCentres.join(" and ")} holds no rare-disease images at
                all, yet still votes on them.
              </p>
            </Reveal>

            <div className="grid gap-6 sm:grid-cols-2">
              <Reveal delay={0.15} className="rounded-3xl border border-line bg-ink-900/70 p-6 backdrop-blur">
                <StatusTag status="verified" />
                <h3 className="mt-4 font-display text-lg font-semibold text-white">What counts as rare</h3>
                <p className="mt-2 text-sm leading-relaxed text-slate-400">
                  Under 1/{rule.headRatioDivisor} of the largest class ({rule.headClass},{" "}
                  {rule.headSharePct.toFixed(1)}%) - a {rule.cutoffPct.toFixed(2)}% cutoff.
                </p>
                <ul className="mt-3 space-y-1 text-sm">
                  {rare.map((r) => (
                    <li key={r.name} className="flex justify-between text-slate-300">
                      <span>{r.name}</span>
                      <span className="font-mono">{r.sharePct.toFixed(2)}%</span>
                    </li>
                  ))}
                </ul>
                <p className="mt-3 text-xs text-slate-500">
                  The rule was chosen after seeing the counts, and we say so.
                </p>
              </Reveal>
              <Reveal delay={0.25} className="rounded-3xl border border-line bg-ink-900/70 p-6 backdrop-blur">
                <StatusTag status="verified" />
                <h3 className="mt-4 font-display text-lg font-semibold text-white">Few hospitals can check</h3>
                <p className="mt-2 text-sm leading-relaxed text-slate-400">
                  Hospitals holding at least {rule.holderMinImages} training images of each rare disease:
                </p>
                <ul className="mt-3 space-y-1 text-sm">
                  {rare.map((r) => (
                    <li key={r.name} className="flex justify-between text-slate-300">
                      <span>{r.name}</span>
                      <span className="font-mono">{r.holders} of 6</span>
                    </li>
                  ))}
                </ul>
                <p className="mt-3 text-xs text-slate-500">
                  If one of them misreports, there may be nobody left to contradict it.
                </p>
              </Reveal>
            </div>
          </div>
        </div>
        <p className="mt-8 font-mono text-[11px] text-slate-600">Source: {sources.join(" · ")}</p>
      </div>
    </ParallaxSection>
  );
}
