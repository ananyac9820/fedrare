import type { Metadata } from "next";
import Link from "next/link";
import { CompareCurves, CurveChart, G0aChart, RareF1Chart } from "@/components/charts/Charts";
import { CHART } from "@/components/ui/ChartTooltip";
import { Card, PillLink } from "@/components/ui/Buttons";
import { PageHeader, Reveal, Section } from "@/components/ui/Motion";
import { DataBadge, GateBadge, Pill, StatusTag } from "@/components/ui/Status";
import { baselines, failedGates, g0a, g0aTopCentres, g0b, gate, pending, sd, study } from "@/lib/data";

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
  const g0bRetry = g0b.retry;
  const g0aRetry = g0a.retry;
  const g0aRetryStat = g0aRetry?.definitions.find((d) => d.key.startsWith("retry"));
  const ROLE: Record<string, string> = {
    retry_bias_round1: "Bias-row change - the design doc's one retry (decides the gate)",
    avg3_weight_rounds1to3: "Weight row averaged over rounds 1-3 - reported only",
    M1_signed_bias_round1: "Sign-aware bias change - amendment M1, logged before running",
  };

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
                {(g0bRetry?.statistic ?? g0b.statistic).toFixed(3)}
              </p>
              <p className="mt-2 font-mono text-xs text-faint">
                {g0bRetry
                  ? `± ${g0bRetry.retry.sd.toFixed(3)} across ${g0bRetry.retry.perSeed.length} seeds · first attempt ${g0b.statistic.toFixed(3)}`
                  : `${spread(g0b.perSeed.map((s) => s.value)).replace(" ± ", "± ")} across ${g0b.perSeed.length} seeds`}
              </p>
              <p className="mt-6 text-sm leading-relaxed text-muted">
                FedAvg balanced accuracy{g0bRetry ? " after the retry (last dense block fine-tuned)" : ""}. Gate {g0b.gate} needed ≥ {g0b.threshold}.
              </p>
            </Card>
          </Reveal>
          <Reveal delay={0.08}>
            <Card className="h-full">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <StatusTag status="failed" />
                {g0aGate && <GateBadge state={g0aGate.state} />}
              </div>
              <p className="mt-10 font-display text-6xl text-failed">{g0a.statistic.toFixed(2)}</p>
              <p className="mt-2 font-mono text-xs text-faint">
                mean Spearman, 8 diseases{g0aRetryStat ? ` · retry ${g0aRetryStat.statistic.toFixed(2)}` : ""}
              </p>
              <p className="mt-6 text-sm leading-relaxed text-muted">
                EARN&apos;s evidence signal. Gate {g0a.gate} needed ≥ {g0a.threshold}.
              </p>
            </Card>
          </Reveal>
          <Reveal delay={0.16}>
            <Card className="h-full">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <StatusTag status="verified" />
                {gate("G1") && <GateBadge state={gate("G1")!.state} />}
              </div>
              <p className="mt-10 font-display text-6xl text-ink">{study.runs}</p>
              <p className="mt-2 font-mono text-xs text-faint">attack-study runs · {pending.items.length} items not run</p>
              <p className="mt-6 text-sm leading-relaxed text-muted">
                Attacks, Camp A, robust rules and EARN on an oracle signal. <Link href="/study" className="text-accent underline-offset-4 hover:underline">See the study →</Link>
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
            ? `The first attempt (20 Sep): a classifier head trained on frozen DenseNet-121 features across all six hospitals, ${fedavg.rounds} rounds, ${fedavg.seeds} seeds. This is the standard method we compare against - not our contribution. The retry below replaced it.`
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
                  <h3 className="mr-2 font-display text-2xl font-medium text-ink">Gate {g0b.gate}, first attempt: below the bar</h3>
                  {g0bGate && <GateBadge state={g0bRetry ? (g0b.passed ? "passed" : "failed") : g0bGate.state} />}
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
                {g0bGate?.note && !g0bRetry && (
                  <div className="mt-8 rounded-2xl bg-progress-soft/70 p-6">
                    <StatusTag status="in-progress" />
                    <p className="mt-4 text-sm leading-relaxed text-ink">{g0bGate.note}</p>
                  </div>
                )}
              </div>
            </Card>
          </Reveal>
        )}
        {g0bRetry && (
          <Reveal className="mt-8">
            <Card className="grid grid-cols-1 [&>*]:min-w-0 gap-10 lg:grid-cols-[1.35fr_1fr] lg:items-center">
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="mr-2 font-display text-2xl font-medium text-ink">The retry: passed</h3>
                  <GateBadge state={g0bRetry.passed ? "passed" : "failed"} />
                </div>
                <p className="mt-2 text-sm text-muted">Balanced accuracy after every round, mean over seeds.</p>
                <div className="mt-8">
                  <CompareCurves threshold={g0b.threshold} label={`G0b bar ${g0b.threshold}`} series={[
                    { name: "Frozen features, 100 rounds", color: CHART.neutral, curve: g0bRetry.amended.frozen.curve },
                    { name: "Last block fine-tuned, 100 rounds", color: CHART.accent, curve: g0bRetry.amended.ft4.curve },
                  ]} />
                </div>
              </div>
              <div>
                <p className="text-pretty text-lg leading-relaxed text-ink">
                  The design doc&apos;s retry - fine-tune DenseNet&apos;s last dense block (by FedAvg, on training
                  images only, {g0bRetry.fineTuningMinutes.toFixed(0)} minutes on a laptop GPU) and re-extract the
                  features - lifts FedAvg to <span className="font-mono">{g0bRetry.retry.mean.toFixed(3)}</span> at the
                  same 40 rounds. Gate {g0b.gate} passes.
                </p>
                <p className="mt-6 text-sm leading-relaxed text-muted">
                  Training for 100 rounds was an amendment, logged before running: {g0bRetry.amended.frozen.mean.toFixed(3)} on
                  frozen features, {g0bRetry.amended.ft4.mean.toFixed(3)} on fine-tuned ones (rare F1{" "}
                  {g0bRetry.amended.ft4.rareMacroF1.toFixed(3)}). Every later experiment uses the fine-tuned features
                  and 100 rounds.
                </p>
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
                {g0aRetry
                  ? "The one allowed retry failed too, so the project switched to Fallback F1."
                  : "Retrying is a team decision. Until the signal is fixed, Camp A and EARN cannot be run."}
              </p>
            </Card>
          </Reveal>
        </div>
        {g0aRetry && (
          <Reveal className="mt-8">
            <Card className="overflow-x-auto">
              <div className="flex flex-wrap items-center gap-2">
                <h3 className="mr-2 font-display text-2xl font-medium text-ink">The retry, and one amendment</h3>
                <GateBadge state={g0aRetry.passed ? "passed" : "failed"} />
              </div>
              <p className="mt-2 text-sm text-muted">
                Written down and committed before running. Mean Spearman over the 8 diseases, needs ≥ {g0a.threshold}.
                Holder AUROC: how well the signal separates hospitals holding ≥ 20 images of a disease from the rest (0.5 = chance).
              </p>
              <table className="mt-8 w-full min-w-[560px] text-left text-sm">
                <thead>
                  <tr className="border-b border-line font-mono text-[11px] uppercase tracking-[0.14em] text-faint">
                    <th className="pb-3 pr-4 font-medium">Evidence measure</th>
                    <th className="pb-3 pr-4 font-medium">Spearman</th>
                    <th className="pb-3 font-medium">Holder AUROC</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-line/70">
                    <td className="py-4 pr-4 text-ink">Weight-row change - the original check</td>
                    <td className="py-4 pr-4 font-mono text-failed">{g0a.statistic.toFixed(3)}</td>
                    <td className="py-4 font-mono text-faint">-</td>
                  </tr>
                  {g0aRetry.definitions.map((d) => (
                    <tr key={d.key} className="border-b border-line/70 last:border-0">
                      <td className="py-4 pr-4 text-ink">{ROLE[d.key] ?? d.role}</td>
                      <td className={`py-4 pr-4 font-mono ${d.passed ? "text-verified" : "text-failed"}`}>{d.statistic.toFixed(3)}</td>
                      <td className="py-4 font-mono text-ink">{d.holderAuroc.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-6 text-sm leading-relaxed text-muted">
                The sign-aware measure can tell <em>whether</em> a hospital holds a disease better than chance, but
                not <em>how much</em> - local training uses a class-balanced loss, which deliberately removes count
                information. That risk was written down before the run.
              </p>
            </Card>
          </Reveal>
        )}
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
        <Reveal className="mt-14 flex flex-wrap gap-3">
          <PillLink href="/study">The attack study →</PillLink>
          <PillLink href="/status" variant="outline">Full plan and gates</PillLink>
        </Reveal>
      </Section>
    </>
  );
}
