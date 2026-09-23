import type { Metadata } from "next";
import { Card } from "@/components/ui/Buttons";
import { Pager } from "@/components/ui/Pager";
import { HScroll } from "@/components/ui/HScroll";
import { PageHeader, Reveal, Section } from "@/components/ui/Motion";
import { GateBadge, Pill, StatusTag } from "@/components/ui/Status";
import type { ProgressState } from "@/lib/data";
import { pending, roadmap } from "@/lib/data";

export const metadata: Metadata = { title: "Status" };

const ICON: Record<ProgressState, { icon: string; cls: string; label: string }> = {
  done: { icon: "✓", cls: "text-verified", label: "done" },
  failed: { icon: "✕", cls: "text-failed", label: "under the bar" },
  blocked: { icon: "‖", cls: "text-notrun", label: "blocked" },
  pending: { icon: "·", cls: "text-faint", label: "pending" },
  "in-progress": { icon: "◐", cls: "text-progress", label: "in progress" },
};

export default function StatusPage() {
  const asOf = new Date(`${roadmap.asOf}T00:00:00Z`).toLocaleDateString("en-GB", {
    day: "numeric",
    month: "long",
    year: "numeric",
    timeZone: "UTC",
  });
  const done = roadmap.weeks.flatMap((w) => (w.progress ?? []).filter((p) => p.state === "done" || p.state === "failed"));

  return (
    <>
      <PageHeader eyebrow="06 · Status" title="Where the project honestly stands.">
        Every decision point below was fixed before any result existed. As of {asOf},{" "}
        {(() => {
          const n = (st: string) => roadmap.gates.filter((g) => g.state === st).length;
          const parts = [
            n("passed") && `${n("passed")} passed`,
            n("failed") && `${n("failed")} came in under the bar`,
            n("not-run") && `${n("not-run")} could not be evaluated`,
            n("pending") && `${n("pending")} still ahead`,
          ].filter(Boolean);
          return `of ${roadmap.gates.length} gates, ${parts.join(", ")}.`;
        })()}
      </PageHeader>

      <Section label="Gates" title="Four decision points.">
        <div className="grid grid-cols-1 [&>*]:min-w-0 gap-8 md:grid-cols-2">
          {roadmap.gates.map((g, i) => (
            <Reveal key={g.id} delay={i * 0.06}>
              <Card
                dashed={g.state === "pending" || g.state === "not-run"}
                className={`h-full ${g.state === "failed" ? "border-failed/30" : ""}`}
              >
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <span className="font-display text-4xl font-medium text-ink">{g.id}</span>
                  <GateBadge state={g.state} />
                </div>
                <p className="mt-3 text-lg text-ink">{g.name}</p>
                <p className="font-mono text-xs text-faint">{g.when}</p>
                <p className="mt-6 text-pretty leading-relaxed text-muted">{g.condition}</p>
                {g.result && (
                  <div className="mt-6 flex flex-wrap items-center gap-2 rounded-2xl bg-sand/70 px-5 py-4">
                    <span className="font-mono text-sm text-ink">{g.result}</span>
                    {g.resultStatus === "verified" && <StatusTag status="verified" />}
                  </div>
                )}
                {g.note && <p className="mt-5 text-sm leading-relaxed text-muted">{g.note}</p>}
                <p className="mt-6 border-t border-line pt-5 text-sm leading-relaxed text-faint">If the bar is not met: {g.ifFails}</p>
              </Card>
            </Reveal>
          ))}
        </div>
      </Section>

      <Section tone="sand" label="Six weeks" title="The plan, week by week.">
        <HScroll label="Six-week plan">
          {roadmap.weeks.map((w) => {
            const current = w.week === roadmap.currentWeek;
            return (
              <article
                key={w.week}
                className={`flex w-[82vw] shrink-0 snap-start flex-col rounded-[28px] border bg-paper p-8 sm:w-[340px] md:p-10 ${
                  current ? "border-accent/50" : "border-line"
                }`}
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="font-mono text-xs text-faint">
                    Week {w.week} · {w.dates}
                  </span>
                  {current && <Pill tone="accent">Now</Pill>}
                </div>
                <p className="mt-6 text-pretty leading-relaxed text-ink">{w.work}</p>
                <p className="mt-5 text-sm text-muted">→ {w.deliverable}</p>
                {w.progress && (
                  <ul className="mt-8 space-y-3 border-t border-line pt-6">
                    {w.progress.map((p) => (
                      <li key={p.item} className="flex gap-3 text-sm leading-snug text-muted">
                        <span className={`w-4 shrink-0 font-bold ${ICON[p.state].cls}`} aria-label={ICON[p.state].label}>
                          {ICON[p.state].icon}
                        </span>
                        {p.item}
                      </li>
                    ))}
                  </ul>
                )}
              </article>
            );
          })}
        </HScroll>
      </Section>

      <Section label="Run vs not run" title="What exists, and what doesn't yet.">
        <div className="grid grid-cols-1 [&>*]:min-w-0 gap-8 lg:grid-cols-[1fr_1.4fr]">
          <Reveal>
            <Card className="h-full">
              <div className="flex flex-wrap items-center gap-2">
                <Pill>Has run</Pill>
                <StatusTag status="verified" />
              </div>
              <ul className="mt-8 space-y-4">
                {done.map((p) => (
                  <li key={p.item} className="flex gap-3 leading-snug text-ink">
                    <span className={`w-4 shrink-0 font-bold ${ICON[p.state].cls}`} aria-label={ICON[p.state].label}>
                      {ICON[p.state].icon}
                    </span>
                    <span>
                      {p.item}
                      {p.state === "failed" && <span className="ml-2 text-sm text-failed">(under the bar)</span>}
                    </span>
                  </li>
                ))}
              </ul>
            </Card>
          </Reveal>
          <Reveal delay={0.08}>
            <Card dashed className="h-full overflow-x-auto">
              <div className="flex flex-wrap items-center gap-2">
                <Pill>Not run yet</Pill>
                <StatusTag status="not-run" />
              </div>
              <table className="mt-8 w-full min-w-[520px] text-left text-sm">
                <thead>
                  <tr className="border-b border-line font-mono text-[11px] uppercase tracking-[0.16em] text-faint">
                    <th className="pb-4 pr-4 font-medium">What</th>
                    <th className="pb-4 pr-4 font-medium">State</th>
                    <th className="pb-4 font-medium">Owner · branch</th>
                  </tr>
                </thead>
                <tbody>
                  {pending.items.map((p) => (
                    <tr key={p.what} className="border-b border-line/70 align-top last:border-0">
                      <td className="py-5 pr-4 text-ink">
                        {p.what}
                        {p.blockedBy && <span className="mt-1 block text-xs text-progress">Waiting on {p.blockedBy}</span>}
                      </td>
                      <td className="py-5 pr-4">
                        <StatusTag status={p.status} />
                      </td>
                      <td className="py-5 text-muted">
                        {p.owner}
                        <span className="mt-1 block font-mono text-xs text-faint">{p.branch}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          </Reveal>
        </div>
      </Section>
      <Pager current="/status" />
    </>
  );
}
