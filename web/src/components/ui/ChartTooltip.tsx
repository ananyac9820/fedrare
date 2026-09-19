"use client";

interface Entry {
  name?: string | number;
  value?: number | string;
  color?: string;
}

/** Dark tooltip shared by every chart. Recharts passes active/payload/label. */
export function ChartTooltip({ active, payload, label, unit = "", digits }: {
  active?: boolean;
  payload?: Entry[];
  label?: string | number;
  unit?: string;
  digits?: number;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-line bg-ink-900/95 px-3 py-2 text-xs shadow-xl backdrop-blur">
      {label !== undefined && <p className="mb-1 font-mono text-slate-400">{label}</p>}
      {payload.map((p) => (
        <p key={String(p.name)} className="flex items-center gap-2 text-slate-200">
          <span className="h-2 w-2 rounded-full" style={{ background: p.color }} />
          <span className="text-slate-400">{p.name}</span>
          <span className="ml-auto font-mono">
            {typeof p.value === "number" && digits !== undefined ? p.value.toFixed(digits) : p.value}
            {unit}
          </span>
        </p>
      ))}
    </div>
  );
}
