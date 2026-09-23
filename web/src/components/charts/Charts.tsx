"use client";

import { motion } from "framer-motion";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  LabelList,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { CHART, ChartTooltip } from "@/components/ui/ChartTooltip";
import { useReducedMotionSafe } from "@/components/ui/Motion";
import type { Attack, BaselineRule, CentreMismatch, G0a, StudyRow } from "@/lib/data";
import { sd } from "@/lib/data";

const axis = { fill: CHART.axis, fontSize: 12 };

/** Two horizontal bars: share of rare-disease data vs share of FedAvg weight. */
export function GapBars({ rareShare, fedavgWeight, centre }: {
  rareShare: number;
  fedavgWeight: number;
  centre: number;
}) {
  const max = Math.max(rareShare, fedavgWeight) * 1.1;
  const bars = [
    { label: `Rare-disease training images held by Centre ${centre}`, value: rareShare, color: CHART.accent },
    { label: `Aggregation weight FedAvg gives Centre ${centre}`, value: fedavgWeight, color: CHART.neutral },
  ];
  return (
    <div className="space-y-8">
      {bars.map((b, i) => (
        <div key={b.label}>
          <div className="mb-3 flex items-baseline justify-between gap-6">
            <span className="text-sm leading-snug text-muted">{b.label}</span>
            <span className="font-display text-3xl font-medium text-ink">{(100 * b.value).toFixed(1)}%</span>
          </div>
          <div className="h-3 overflow-hidden rounded-full bg-sand">
            <motion.div
              className="h-full rounded-full"
              style={{ background: b.color, transformOrigin: "0% 50%" }}
              initial={{ scaleX: 0 }}
              whileInView={{ scaleX: b.value / max }}
              viewport={{ once: true, amount: 0.8 }}
              transition={{ duration: 1.1, delay: 0.1 + i * 0.2, ease: [0.25, 0.8, 0.3, 1] }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

export function MismatchChart({ mismatch, specialistId }: { mismatch: CentreMismatch[]; specialistId: number }) {
  const reduced = useReducedMotionSafe();
  const data = mismatch.map((m) => ({
    name: `C${m.id}`,
    id: m.id,
    "Share of rare-disease images": +(100 * m.rareShare).toFixed(1),
    "Share of FedAvg weight": +(100 * m.fedavgWeight).toFixed(1),
  }));
  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} barGap={4} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
          <CartesianGrid stroke={CHART.grid} vertical={false} />
          <XAxis dataKey="name" tick={axis} axisLine={false} tickLine={false} />
          <YAxis tick={axis} axisLine={false} tickLine={false} unit="%" />
          <Tooltip content={<ChartTooltip unit="%" />} cursor={{ fill: "rgb(29 41 41 / 0.04)" }} />
          <Legend wrapperStyle={{ fontSize: 12, paddingTop: 12 }} iconType="circle" />
          <Bar dataKey="Share of rare-disease images" fill={CHART.accent} radius={[5, 5, 0, 0]} isAnimationActive={!reduced}>
            {data.map((d) => (
              <Cell key={d.id} fillOpacity={d.id === specialistId ? 1 : 0.45} />
            ))}
          </Bar>
          <Bar dataKey="Share of FedAvg weight" fill={CHART.neutral} radius={[5, 5, 0, 0]} isAnimationActive={!reduced}>
            {data.map((d) => (
              <Cell key={d.id} fillOpacity={d.id === specialistId ? 1 : 0.6} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Rare-class F1 per rule, with the spread across seeds as error bars (population SD). */
export function RareF1Chart({ rules, sample }: { rules: BaselineRule[]; sample: boolean }) {
  const reduced = useReducedMotionSafe();
  const classes = Object.keys(rules[0]?.rareF1 ?? {});
  const colors = [CHART.baseline, CHART.accent];
  const rows = [
    ...classes.map((c) => ({ name: c, key: c })),
    { name: "Rare average", key: "__macro" },
  ].map(({ name, key }) => {
    const row: Record<string, string | number> = { name };
    rules.forEach((r) => {
      const seeds = r.perSeed ?? [];
      const value = key === "__macro" ? r.rareMacroF1 : r.rareF1[key];
      const spread = seeds.length > 1
        ? sd(seeds.map((s) => (key === "__macro" ? s.rareMacroF1 : s.rareF1[key])))
        : 0;
      row[r.label] = +value.toFixed(3);
      row[`${r.label}__sd`] = +spread.toFixed(3);
    });
    return row;
  });

  return (
    <div className="relative h-80">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} barGap={6} margin={{ top: 24, right: 8, bottom: 0, left: -16 }}>
          <defs>
            {rules.map((r, i) => (
              <pattern key={r.id} id={`hatch-${r.id}`} width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                <rect width="8" height="8" fill={colors[i % colors.length]} fillOpacity={0.12} />
                <line x1="0" y1="0" x2="0" y2="8" stroke={colors[i % colors.length]} strokeWidth="3" strokeOpacity={0.5} />
              </pattern>
            ))}
          </defs>
          <CartesianGrid stroke={CHART.grid} vertical={false} />
          <XAxis dataKey="name" tick={axis} axisLine={false} tickLine={false} />
          <YAxis domain={[0, 1]} tick={axis} axisLine={false} tickLine={false} />
          <Tooltip content={<ChartTooltip digits={3} />} cursor={{ fill: "rgb(29 41 41 / 0.04)" }} />
          {rules.length > 1 && <Legend wrapperStyle={{ fontSize: 12 }} iconType="circle" />}
          {rules.map((r, i) => (
            <Bar
              key={r.id}
              dataKey={r.label}
              fill={sample ? `url(#hatch-${r.id})` : colors[i % colors.length]}
              stroke={sample ? colors[i % colors.length] : undefined}
              strokeDasharray={sample ? "4 3" : undefined}
              radius={[5, 5, 0, 0]}
              maxBarSize={72}
              isAnimationActive={!reduced}
            >
              {!sample && <ErrorBar dataKey={`${r.label}__sd`} width={8} stroke={CHART.ink} strokeWidth={1.2} />}
              {!sample && <LabelList dataKey={r.label} position="top" offset={14} fill={CHART.label} fontSize={12} formatter={(v) => Number(v).toFixed(3)} />}
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
      {sample && (
        <div aria-hidden className="pointer-events-none absolute inset-0 flex items-center justify-center overflow-hidden">
          <span className="-rotate-12 whitespace-nowrap rounded-xl border-2 border-dashed border-notrun/40 bg-paper/80 px-5 py-2 font-mono text-base font-bold uppercase tracking-[0.3em] text-notrun">
            Sample data · not results
          </span>
        </div>
      )}
    </div>
  );
}

/** Balanced accuracy after every round (mean over seeds) against the gate's bar. */
export function CurveChart({ curve, threshold }: {
  curve: { round: number; balancedAccuracy: number }[];
  threshold: number;
}) {
  const reduced = useReducedMotionSafe();
  const data = curve.map((c) => ({ round: c.round, "Balanced accuracy": +c.balancedAccuracy.toFixed(3) }));
  return (
    <div className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 24, right: 16, bottom: 8, left: -16 }}>
          <CartesianGrid stroke={CHART.grid} vertical={false} />
          <XAxis dataKey="round" tick={axis} axisLine={false} tickLine={false} ticks={[1, 10, 20, 30, data.length]}
            label={{ value: "round", position: "insideBottomRight", offset: -4, fill: CHART.axis, fontSize: 11 }} />
          <YAxis domain={[0, 0.6]} ticks={[0, 0.15, 0.3, 0.45, 0.6]} tick={axis} axisLine={false} tickLine={false} />
          <Tooltip content={<ChartTooltip digits={3} />} />
          <ReferenceLine
            y={threshold}
            stroke={CHART.failed}
            strokeDasharray="6 4"
            label={{ value: `G0b bar ${threshold}`, position: "insideTopLeft", fill: CHART.failed, fontSize: 11 }}
          />
          <Line type="monotone" dataKey="Balanced accuracy" stroke={CHART.baseline} strokeWidth={2.4} dot={false} isAnimationActive={!reduced} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function G0aChart({ g0a }: { g0a: G0a }) {
  const reduced = useReducedMotionSafe();
  const rows = g0a.classNames.map((name, i) => ({
    name: g0a.rareIds.includes(i) ? `${name} ★` : name,
    Spearman: +g0a.perClassSpearman[i].toFixed(2),
  }));
  return (
    <div className="h-96">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} layout="vertical" margin={{ top: 28, right: 40, bottom: 8, left: 8 }}>
          <CartesianGrid stroke={CHART.grid} horizontal={false} />
          <XAxis type="number" domain={[-1, 1]} ticks={[-1, -0.5, 0, 0.5, 0.7, 1]} tick={axis} axisLine={false} tickLine={false} />
          <YAxis type="category" dataKey="name" width={168} tick={{ fill: CHART.label, fontSize: 12 }} axisLine={false} tickLine={false} />
          <Tooltip content={<ChartTooltip digits={2} />} cursor={{ fill: "rgb(29 41 41 / 0.04)" }} />
          <ReferenceLine x={0} stroke={CHART.axis} />
          <ReferenceLine
            x={g0a.threshold}
            stroke={CHART.ink}
            strokeDasharray="5 4"
            label={{ value: `needed ≥ ${g0a.threshold}`, position: "top", fill: CHART.ink, fontSize: 11 }}
          />
          <Bar dataKey="Spearman" radius={[0, 5, 5, 0]} isAnimationActive={!reduced}>
            {rows.map((r) => (
              <Cell key={r.name} fill={r.Spearman < 0 ? CHART.failed : CHART.neutral} fillOpacity={0.85} />
            ))}
            <LabelList dataKey="Spearman" position="right" fill={CHART.label} fontSize={11} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

const ATTACK_COLOR: Record<Attack, string> = {
  none: CHART.accent,
  A1: "#b5543c",
  A2: "#c9962b",
  A3: "#5b6f95",
};

/** One metric for several methods, one bar per attack, with the spread across seeds. */
export function StudyChart({ rows, methods, attacks, metric, domain = [0, 0.7] }: {
  rows: StudyRow[];
  methods: string[];
  attacks: Attack[];
  metric: "rare_macro_f1" | "balanced_accuracy";
  domain?: [number, number];
}) {
  const reduced = useReducedMotionSafe();
  const data = methods.map((m) => {
    const row: Record<string, string | number> = {};
    attacks.forEach((a) => {
      const r = rows.find((x) => x.method === m && x.attack === a);
      if (r) {
        row.name = r.label;
        row[a] = +r[metric].mean.toFixed(3);
        row[`${a}__sd`] = +r[metric].sd.toFixed(3);
      }
    });
    return row;
  });
  return (
    <div className="h-96">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} barGap={2} margin={{ top: 16, right: 8, bottom: 56, left: -16 }}>
          <CartesianGrid stroke={CHART.grid} vertical={false} />
          <XAxis dataKey="name" tick={{ ...axis, fontSize: 11 }} axisLine={false} tickLine={false}
            interval={0} angle={-30} textAnchor="end" height={70} />
          <YAxis domain={domain} allowDataOverflow tick={axis} axisLine={false} tickLine={false}
            ticks={Array.from({ length: Math.floor((domain[1] - domain[0]) / 0.1 + 1e-9) + 1 }, (_, i) => +(domain[0] + i * 0.1).toFixed(1))} />
          <Tooltip content={<ChartTooltip digits={3} />} cursor={{ fill: "rgb(29 41 41 / 0.04)" }} />
          <Legend verticalAlign="top" wrapperStyle={{ fontSize: 12, paddingBottom: 12 }} iconType="circle" />
          {attacks.map((a) => (
            <Bar key={a} dataKey={a} name={a === "none" ? "no attack" : a} fill={ATTACK_COLOR[a]}
              radius={[4, 4, 0, 0]} maxBarSize={28} isAnimationActive={!reduced}>
              <ErrorBar dataKey={`${a}__sd`} width={4} stroke={CHART.ink} strokeWidth={1} />
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Several balanced-accuracy curves against a gate's bar. */
export function CompareCurves({ series, threshold, label }: {
  series: { name: string; color: string; curve: { round: number; balancedAccuracy: number }[] }[];
  threshold: number;
  label: string;
}) {
  const reduced = useReducedMotionSafe();
  const rounds = Math.max(...series.map((s) => s.curve.length));
  const data = Array.from({ length: rounds }, (_, i) => {
    const row: Record<string, number> = { round: i + 1 };
    series.forEach((s) => {
      if (s.curve[i]) row[s.name] = +s.curve[i].balancedAccuracy.toFixed(3);
    });
    return row;
  });
  return (
    <div className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 24, right: 16, bottom: 8, left: -16 }}>
          <CartesianGrid stroke={CHART.grid} vertical={false} />
          <XAxis dataKey="round" type="number" domain={[1, rounds]} tick={axis} axisLine={false} tickLine={false}
            label={{ value: "round", position: "insideBottomRight", offset: -4, fill: CHART.axis, fontSize: 11 }} />
          <YAxis domain={[0, 0.6]} ticks={[0, 0.15, 0.3, 0.45, 0.6]} tick={axis} axisLine={false} tickLine={false} />
          <Tooltip content={<ChartTooltip digits={3} />} />
          <Legend wrapperStyle={{ fontSize: 12 }} iconType="circle" />
          <ReferenceLine y={threshold} stroke={CHART.failed} strokeDasharray="6 4"
            label={{ value: label, position: "insideTopLeft", fill: CHART.failed, fontSize: 11 }} />
          {series.map((s) => (
            <Line key={s.name} type="monotone" dataKey={s.name} stroke={s.color} strokeWidth={2.2} dot={false}
              isAnimationActive={!reduced} connectNulls />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Per-round trust (0-1) for several series, with a marker at the round the attacker turns. */
export function TrustCurves({ series, turnRound }: {
  series: { name: string; color: string; values: number[] }[];
  turnRound: number;
}) {
  const reduced = useReducedMotionSafe();
  const rounds = Math.max(...series.map((s) => s.values.length));
  const data = Array.from({ length: rounds }, (_, i) => {
    const row: Record<string, number> = { round: i + 1 };
    series.forEach((s) => { row[s.name] = +s.values[i].toFixed(3); });
    return row;
  });
  return (
    <div className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 24, right: 16, bottom: 8, left: -16 }}>
          <CartesianGrid stroke={CHART.grid} vertical={false} />
          <XAxis dataKey="round" type="number" domain={[1, rounds]} tick={axis} axisLine={false} tickLine={false}
            label={{ value: "round", position: "insideBottomRight", offset: -4, fill: CHART.axis, fontSize: 11 }} />
          <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tick={axis} axisLine={false} tickLine={false} />
          <Tooltip content={<ChartTooltip digits={2} />} />
          <Legend wrapperStyle={{ fontSize: 12 }} iconType="circle" />
          <ReferenceLine x={turnRound} stroke={CHART.failed} strokeDasharray="6 4"
            label={{ value: "specialist turns", position: "insideTopRight", fill: CHART.failed, fontSize: 11 }} />
          {series.map((s) => (
            <Line key={s.name} type="monotone" dataKey={s.name} stroke={s.color} strokeWidth={2.2} dot={false}
              isAnimationActive={!reduced} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
