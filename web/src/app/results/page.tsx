import type { Metadata } from "next";
import { CurveChart, G0aChart, RareF1Chart } from "@/components/charts/Charts";
import { Card, PillLink } from "@/components/ui/Buttons";
import { PageHeader, Reveal, Section } from "@/components/ui/Motion";
import { DataBadge, GateBadge, Pill, StatusTag } from "@/components/ui/Status";
import { baselines, failedGates, g0a, g0aTopCentres, g0b, gate, pending, sd } from "@/lib/data";

export const metadata: Metadata = { title: "Results" };

export default function ResultsPage() {
  const sample = baselines._meta.status === "sample";
  const fedavg = baselines.rules.find((r) => r.id === "fedavg");
  const seeds = fedavg?.perSeed ?? [];
  const spread = (xs: number[]) => (xs.length > 1 ? ` ± ${sd(xs).toFixed(3)}` : "");
  const g0bGate = gate("G0b");
  const g0aGate = gate("G0a");
  const zeroTop = g0aTopCentres.filter((t) => t.images === 0);
  const rareZeroTop = zeroTop.filter((t) => t.rare);
  const curve = fedavg?.curve ?? [];
  const at = (round: number) => curve.find((c) => c.round === round)?.balancedAccuracy;
  const lastRound = curve.at(-1)?.round;
  const tenBefore = lastRound ? at(lastRound - 10) : undefined;

  return (
    <>
      <PageHeader
        eyebrow="03 · Results"
        tags={
          <>
            <StatusTag status="verified" />
            <StatusTag status="failed" />
          </>
        }
        title="Everything we've measured - including what failed."
      >
        Real numbers from real runs, with the spread across random seeds.{" "}
        {failedGates.length > 0 &&
          `${failedGates.length} pre-set check${failedGates.length === 1 ? "" : "s"} did not pass, and ${failedGates.length === 1 ? "it is" : "both are"} shown here as ${failedGates.length === 1 ? "a failure" : "failures"}. `}
        Anything that hasn&apos;t run has no number on this page.
      </PageHeader>

      {/* Summary */}
      <Section>
        <div className="grid grid-cols-1 [&>*]:min-w-0 gap-8 md:grid-cols-3">
          <Reveal>
            <Card className="h-full">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <StatusTag status="baseline" />
                {g0bGate && <GateBadge state={g0bGate.state} />}
              </div>
              <p className="mt-10 font-display text-6xl text-ink">
                {g0b.statistic.toFixed(3)}
              </p>
              <p className="mt-2 font-mono text-xs text-faint">{spread(g0b.perSeed.map((s) => s.value)).replace(" ± ", "± ")} across {g0b.perSeed.length} seeds</p>
              <p className="mt-6 text-sm leading-relaxed text-muted">
                FedAvg balanced accuracy. Gate {g0b.gate} needed ≥ {g0b.threshold}.
              </p>
            </Card>
          </Reveal>
          <Reveal delay={0.08}>
            <Card className="h-full">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <StatusTag status="in-progress" />
                {g0aGate && <GateBadge state={g0aGate.state} />}
              </div>
              <p className="mt-10 font-display text-6xl text-failed">{g0a.statistic.toFixed(2)}</p>
              <p className="mt-2 font-mono text-xs text-faint">mean Spearman, 8 diseases</p>
              <p className="mt-6 text-sm leading-relaxed text-muted">
                EARN&apos;s evidence signal. Gate {g0a.gate} needed ≥ {g0a.threshold}.
              </p>
            </Card>
          </Reveal>
          <Reveal delay={0.16}>
            <Card dashed className="h-full">
              <StatusTag status="not-run" />
              <p className="mt-10 font-display text-6xl text-notrun">{pending.items.length}</p>
              <p className="mt-2 font-mono text-xs text-faint">pieces without results yet</p>
              <p className="mt-6 text-sm leading-relaxed text-muted">
                Including Camp A, EARN itself and the attacks. Listed at the bottom of this page.
              </p>
            </Card>
          </Reveal>
        </div>
      </Section>

      {/* FedAvg */}
      <Section
        tone="sand"
        label="Baseline · FedAvg"
        title="FedAvg on the natural split."
        tags={
          <>
            <StatusTag status="baseline" />
            <DataBadge status={baselines._meta.status} />
          </>
        }
        intro={
          fedavg && !sample
            ? `Tier A: a classifier head trained on frozen DenseNet-121 features across all six hospitals, ${fedavg.rounds} rounds, ${fedavg.seeds} seeds. This is the standard method we compare against - not our contribution.`
            : "Placeholder values only - no run has produced these numbers yet."
        }
      >
        <div className="grid grid-cols-1 [&>*]:min-w-0 gap-8 lg:grid-cols-[1.35fr_1fr]">
          <Reveal>
            <Card className={`h-full ${sample ? "sample-stripes" : ""}`}>
              <h3 className="font-display text-2xl font-medium text-ink">Rare-disease F1</h3>
              <p className="mt-2 text-sm text-muted">Final round, mean over seeds. Whiskers show ± one standard deviation.</p>
              <div className="mt-10">
                <RareF1Chart rules={baselines.rules} sample={sample} />
              </div>
            </Card>
          </Reveal>
          <Reveal delay={0.08}>
            <Card className="h-full">
              <h3 className="font-display text-2xl font-medium text-ink">In numbers</h3>
              {fedavg && !sample ? (
                <dl className="mt-8 space-y-5 text-sm">
                  <div className="flex items-baseline justify-between gap-4">
                    <dt className="text-muted">Balanced accuracy</dt>
                    <dd className="font-mono text-ink">
                      {fedavg.balancedAccuracy.toFixed(3)}
                      {spread(seeds.map((s) => s.balancedAccuracy))}
                    </dd>
                  </div>
                  {Object.keys(fedavg.rareF1).map((name) => (
                    <div key={name} className="flex items-baseline justify-between gap-4">
                      <dt className="text-muted">{name} F1</dt>
                      <dd className="font-mono text-ink">
                        {fedavg.rareF1[name].toFixed(3)}
                        {spread(seeds.map((s) => s.rareF1[name]))}
                      </dd>
                    </div>
                  ))}
                  {fedavg.accuracy !== undefined && (
                    <div className="flex items-baseline justify-between gap-4">
                      <dt className="text-muted">Overall accuracy</dt>
                      <dd className="font-mono text-ink">{fedavg.accuracy.toFixed(3)}</dd>
                    </div>
                  )}
                  <div className="border-t border-line pt-5">
                    <p className="font-mono text-[11px] uppercase tracking-[0.16em] text-faint">Per seed · balanced accuracy</p>
                    <div className="mt-3 flex flex-wrap gap-2">
                      {seeds.map((s) => (
                        <span key={s.seed} className="rounded-full bg-sand px-3 py-1 font-mono text-xs text-ink">
                          {s.seed}: {s.balancedAccuracy.toFixed(3)}
                        </span>
                      ))}
                    </div>
                  </div>
                </dl>
              ) : (
                <p className="mt-6 text-sm text-muted">No real run yet.</p>
              )}
            </Card>
          </Reveal>
        </div>

        {curve.length > 0 && !sample && (
          <Reveal className="mt-8">
            <Card className="grid grid-cols-1 [&>*]:min-w-0 gap-10 lg:grid-cols-[1.35fr_1fr] lg:items-center">
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="mr-2 font-display text-2xl font-medium text-ink">Gate {g0b.gate}: below the bar</h3>
                  {g0bGate && <GateBadge state={g0bGate.state} />}
                </div>
                <p className="mt-2 text-sm text-muted">Balanced accuracy after every round, mean over seeds.</p>
                <div className="mt-8">
                  <CurveChart curve={curve} threshold={g0b.threshold} />
                </div>
              </div>
              <div>
                <p className="text-pretty text-lg leading-relaxed text-ink">
                  FedAvg finished at <span className="font-mono">{g0b.statistic.toFixed(3)}</span>, short of
                  the {g0b.threshold} it needed.
                  {tenBefore !== undefined && lastRound && (
                    <>
                      {" "}It was still climbing when the run stopped: {tenBefore.toFixed(3)} at round{" "}
                      {lastRound - 10}, {at(lastRound)!.toFixed(3)} at round {lastRound}.
                    </>
                  )}
                </p>
                {g0bGate?.note && (
                  <div className="mt-8 rounded-2xl bg-progress-soft/70 p-6">
                    <StatusTag status="in-progress" />
                    <p className="mt-4 text-sm leading-relaxed text-ink">{g0bGate.note}</p>
                  </div>
                )}
              </div>
            </Card>
          </Reveal>
        )}
      </Section>

      {/* G0a */}
      <Section
        label="Gate G0a · evidence signal"
        title="EARN's evidence signal came out inverted."
        tags={
          <>
            <StatusTag status="verified" />
            <StatusTag status="failed" />
          </>
        }
        intro={`EARN needs to read, from each hospital's update, how much of each disease it holds. We checked whether that "evidence" ranks hospitals the way their real image counts do. It needed a Spearman correlation of at least ${g0a.threshold}. It scored ${g0a.statistic.toFixed(2)}.`}
      >
        <div className="grid grid-cols-1 [&>*]:min-w-0 gap-8 lg:grid-cols-[1.35fr_1fr]">
          <Reveal>
            <Card className="h-full">
              <h3 className="font-display text-2xl font-medium text-ink">Spearman correlation, per disease</h3>
              <p className="mt-2 text-sm text-muted">★ = rare disease. Red bars point the wrong way.</p>
              <div className="mt-8">
                <G0aChart g0a={g0a} />
              </div>
            </Card>
          </Reveal>
          <Reveal delay={0.08}>
            <Card className="h-full">
              <h3 className="font-display text-2xl font-medium text-ink">What went wrong</h3>
              <p className="mt-6 text-pretty leading-relaxed text-muted">
                For {rareZeroTop.length === 2 ? "both" : rareZeroTop.length} rare disease
                {rareZeroTop.length === 1 ? "" : "s"}, the hospital with the <strong className="text-ink">highest</strong> evidence
                was one holding <strong className="text-ink">zero</strong> images of it. The likely cause: local
                training pushes down the score of a disease a hospital never sees, and that also shows up as
                a large change - the measure can&apos;t tell &quot;up&quot; from &quot;down&quot;.
              </p>
              <ul className="mt-8 space-y-3 border-t border-line pt-6 text-sm">
                {zeroTop.map((t) => (
                  <li key={t.name} className="flex items-baseline justify-between gap-4">
                    <span className={t.rare ? "font-semibold text-accent" : "text-ink"}>{t.name}</span>
                    <span className="font-mono text-xs text-muted">top: Centre {t.centre} · 0 images</span>
                  </li>
                ))}
              </ul>
              <p className="mt-8 text-sm leading-relaxed text-muted">
                Retrying is a team decision. Until the signal is fixed, Camp A and EARN cannot be run.
              </p>
            </Card>
          </Reveal>
        </div>
      </Section>

      {/* Not run */}
      <Section
        tone="sand"
        label="Not run yet"
        title="No result, no number."
        tags={<StatusTag status="not-run" />}
        intro="These have not produced results. Nothing on this site shows a figure for them."
      >
        <div className="grid grid-cols-1 [&>*]:min-w-0 gap-8 md:grid-cols-2 lg:grid-cols-3">
          {pending.items.map((p, i) => (
            <Reveal key={p.what} delay={i * 0.05}>
              <Card dashed className="h-full">
                <StatusTag status={p.status} />
                <h3 className="mt-6 font-display text-2xl font-medium leading-snug text-ink">{p.what}</h3>
                {p.blockedBy && <p className="mt-3 text-sm text-progress">Waiting on {p.blockedBy}</p>}
                <div className="mt-8 flex flex-wrap items-center gap-2 border-t border-line pt-6">
                  <Pill tone="paper">{p.owner}</Pill>
                  <span className="font-mono text-xs text-faint">{p.branch}</span>
                </div>
              </Card>
            </Reveal>
          ))}
        </div>
        <Reveal className="mt-14">
          <PillLink href="/status">Full plan and gates →</PillLink>
        </Reveal>
      </Section>
    </>
  );
}
