"use client";

interface Entry {
  name?: string | number;
  value?: number | string;
  color?: string;
}

/** Light tooltip shared by every chart. Recharts passes active/payload/label. */
export function ChartTooltip({ active, payload, label, unit = "", digits }: {
  active?: boolean;
  payload?: Entry[];
  label?: string | number;
  unit?: string;
  digits?: number;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl border border-line bg-paper px-3 py-2 text-xs shadow-lg">
      {label !== undefined && <p className="mb-1 font-mono text-faint">{label}</p>}
      {payload.map((p) => (
        <p key={String(p.name)} className="flex items-center gap-2 text-ink">
          <span className="h-2 w-2 rounded-full" style={{ background: p.color }} />
          <span className="text-muted">{p.name}</span>
          <span className="ml-auto font-mono">
            {typeof p.value === "number" && digits !== undefined ? p.value.toFixed(digits) : p.value}
            {unit}
          </span>
        </p>
      ))}
    </div>
  );
}

/** Shared chart styling for the light theme. */
export const CHART = {
  grid: "#ebe4d4",
  axis: "#8a918d",
  label: "#56615f",
  ink: "#1d2929",
  accent: "#2c6a64",
  neutral: "#b7ad99",
  failed: "#ae2330",
  baseline: "#2c5b9a",
};
