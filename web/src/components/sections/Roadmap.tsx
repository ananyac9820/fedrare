import { ParallaxSection, Reveal, SectionHeading } from "@/components/ui/Motion";
import { GateBadge, StatusTag } from "@/components/ui/Status";
import type { MissingItem, Roadmap as RoadmapData } from "@/lib/data";

const PROGRESS_ICON = {
  done: { icon: "✓", cls: "text-verified" },
  failed: { icon: "✕", cls: "text-failed" },
  blocked: { icon: "‖", cls: "text-progress" },
  pending: { icon: "·", cls: "text-slate-500" },
} as const;

export function Roadmap({ roadmap, missing }: { roadmap: RoadmapData; missing: MissingItem[] }) {
  const asOf = new Date(`${roadmap.asOf}T00:00:00Z`).toLocaleDateString("en-GB", {
    day: "numeric",
    month: "long",
    year: "numeric",
    timeZone: "UTC",
  });

  return (
    <ParallaxSection id="status" glow="rose" className="py-28 md:py-36">
      <div className="mx-auto max-w-7xl px-6">
        <SectionHeading eyebrow="05 · Status" title="Where the project honestly stands.">
          Every decision point was fixed before any result existed, and a failed gate is reported as
          a failure. As of {asOf}.
        </SectionHeading>

        <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-4">
          {roadmap.gates.map((g, i) => (
            <Reveal key={g.id} delay={i * 0.08} tilt>
              <article
                className={`flex h-full flex-col rounded-2xl border p-5 backdrop-blur ${
                  g.state === "failed"
                    ? "border-failed/40 bg-failed/[0.06]"
                    : g.state === "passed"
                      ? "border-verified/40 bg-verified/[0.06]"
                      : "border-line bg-ink-900/70"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="font-display text-2xl font-semibold text-white">{g.id}</span>
                  <GateBadge state={g.state} />
                </div>
                <p className="mt-1 text-sm font-medium text-slate-300">{g.name}</p>
                <p className="font-mono text-[11px] text-slate-500">{g.when}</p>
                <p className="mt-3 flex-1 text-xs leading-relaxed text-slate-400">{g.condition}</p>
                {g.result && (
                  <div className="mt-3 flex flex-wrap items-center gap-2">
                    <span className="font-mono text-xs text-slate-200">{g.result}</span>
                    {g.resultStatus === "verified" && <StatusTag status="verified" />}
                  </div>
                )}
                {g.note && <p className="mt-2 text-xs leading-relaxed text-slate-500">{g.note}</p>}
                <p className="mt-3 border-t border-line pt-3 text-[11px] leading-relaxed text-slate-500">
                  If it fails: {g.ifFails}
                </p>
              </article>
            </Reveal>
          ))}
        </div>

        <div className="mt-10 overflow-x-auto pb-2 pt-4">
          <ol className="grid min-w-[1040px] grid-cols-6 gap-4">
            {roadmap.weeks.map((w, i) => {
              const current = w.week === roadmap.currentWeek;
              const past = w.week < roadmap.currentWeek;
              return (
                <Reveal as="li" key={w.week} delay={i * 0.06} className="h-full">
                  <div
                    className={`relative h-full rounded-2xl border p-4 ${
                      current ? "border-accent/60 bg-ink-800/90" : past ? "border-line bg-ink-900/40" : "border-line bg-ink-900/70"
                    }`}
                  >
                    {current && (
                      <span className="absolute -top-3 left-4 rounded-full bg-accent px-2.5 py-0.5 font-mono text-[10px] font-bold uppercase tracking-wider text-ink-950">
                        Now
                      </span>
                    )}
                    <p className="font-mono text-xs text-slate-500">
                      Week {w.week} · {w.dates}
                    </p>
                    <p className="mt-2 text-sm leading-snug text-slate-300">{w.work}</p>
                    <p className="mt-3 text-[11px] text-slate-500">→ {w.deliverable}</p>
                    {w.progress && (
                      <ul className="mt-3 space-y-1 border-t border-line pt-3">
                        {w.progress.map((p) => (
                          <li key={p.item} className="flex gap-2 text-[11px] leading-snug text-slate-400">
                            <span className={`w-3 shrink-0 font-bold ${PROGRESS_ICON[p.state].cls}`}>
                              {PROGRESS_ICON[p.state].icon}
                            </span>
                            {p.item}
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                </Reveal>
              );
            })}
          </ol>
        </div>

        <Reveal className="mt-14 rounded-3xl border border-line bg-ink-900/70 p-6 backdrop-blur">
          <h3 className="font-display text-lg font-semibold text-white">Waiting on real data</h3>
          <p className="mt-1 text-sm text-slate-400">
            Everything marked sample data on this page will be replaced from these sources.
          </p>
          <div className="mt-4 overflow-x-auto">
            <table className="w-full min-w-[720px] text-left text-sm">
              <thead>
                <tr className="border-b border-line font-mono text-[11px] uppercase tracking-[0.15em] text-slate-500">
                  <th className="py-2 pr-4">What</th>
                  <th className="py-2 pr-4">From</th>
                  <th className="py-2 pr-4">Branch</th>
                  <th className="py-2">File</th>
                </tr>
              </thead>
              <tbody>
                {missing.map((m) => (
                  <tr key={m.what} className="border-b border-line/60 align-top">
                    <td className="py-3 pr-4 text-slate-300">
                      {m.what}
                      {m.blockedBy && <span className="block text-xs text-progress">Blocked by: {m.blockedBy}</span>}
                    </td>
                    <td className="py-3 pr-4 text-slate-300">
                      {m.owner}
                      <span className="block text-xs text-slate-500">{m.track}</span>
                    </td>
                    <td className="py-3 pr-4 font-mono text-xs text-slate-400">{m.branch}</td>
                    <td className="py-3 font-mono text-xs text-slate-400">{m.file}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Reveal>
      </div>
    </ParallaxSection>
  );
}
