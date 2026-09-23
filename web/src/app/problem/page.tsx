import type { Metadata } from "next";
import { GapBars, MismatchChart } from "@/components/charts/Charts";
import { Card, PillLink } from "@/components/ui/Buttons";
import { HScroll } from "@/components/ui/HScroll";
import { CountUp, PageHeader, Reveal, Section } from "@/components/ui/Motion";
import { Pager } from "@/components/ui/Pager";
import { Pill, StatusTag } from "@/components/ui/Status";
import {
  coverage,
  dataset,
  fmt,
  mismatch,
  pct,
  rareClasses,
  rareTrainTotal,
  specialist,
  specialistRatio,
  totalTrainImages,
  zeroRareCentres,
} from "@/lib/data";

export const metadata: Metadata = { title: "The problem" };

export default function ProblemPage() {
  const rule = dataset.rareRule;
  const n = dataset.centres.length;

  return (
    <>
      <PageHeader
        eyebrow="01 · The problem"
        tags={<StatusTag status="verified" />}
        title="The hospital that knows most about rare diseases gets one of the smaller votes."
      >
        Standard federated averaging (FedAvg) weights each hospital by how many images it holds in
        total, not by what it knows. On the real Fed-ISIC2019 split that shortchanges the one
        hospital that specialises in the rare diseases.
      </PageHeader>

      {/* The gap */}
      <Section label="The gap" title={`Centre ${specialist.id}, in one number.`} tags={<StatusTag status="verified" />}>
        <Reveal>
          <Card className="grid grid-cols-1 [&>*]:min-w-0 gap-14 md:grid-cols-[1fr_1.2fr] md:items-center md:p-14">
            <div>
              <p className="font-display text-[6.5rem] font-medium leading-none tracking-tight text-ink md:text-[8.5rem]">
                <CountUp to={specialistRatio} decimals={2} suffix="×" />
              </p>
              <p className="mt-6 max-w-sm text-pretty text-lg leading-relaxed text-muted">
                more of the rare-disease data than its share of the vote.
              </p>
            </div>
            <div>
              <GapBars rareShare={specialist.rareShare} fedavgWeight={specialist.fedavgWeight} centre={specialist.id} />
              <p className="mt-8 text-sm text-faint">
                {specialist.rareTrainImages} of {rareTrainTotal} rare-disease training images ·{" "}
                {fmt(specialist.trainImages)} of {fmt(totalTrainImages)} training images overall
              </p>
            </div>
          </Card>
        </Reveal>
      </Section>

      {/* Every hospital */}
      <Section
        tone="sand"
        label="Every hospital"
        title="The same comparison, for all six."
        tags={<StatusTag status="verified" />}
        intro={`Only Centre ${specialist.id} holds far more of the rare-disease data than its weight. Centre ${zeroRareCentres.join(" and ")} holds none at all - yet under FedAvg it still votes on rare diseases.`}
      >
        <HScroll label="Per-hospital breakdown">
          {mismatch.map((m) => {
            const isSpecialist = m.id === specialist.id;
            const none = zeroRareCentres.includes(m.id);
            return (
              <article
                key={m.id}
                className={`flex w-[82vw] shrink-0 snap-start flex-col rounded-[28px] border bg-paper p-8 sm:w-[320px] md:p-10 ${
                  isSpecialist ? "border-accent/50" : "border-line"
                }`}
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h3 className="font-display text-3xl font-medium text-ink">Centre {m.id}</h3>
                  {isSpecialist && <Pill tone="accent">Specialist</Pill>}
                  {none && <Pill>No rare images</Pill>}
                </div>
                <p className="mt-2 font-mono text-xs text-faint">{fmt(m.totalImages)} images in total</p>
                <dl className="mt-10 space-y-5 text-sm">
                  <div className="flex items-baseline justify-between gap-4">
                    <dt className="text-muted">Rare-disease training images</dt>
                    <dd className="font-mono text-ink">{m.rareTrainImages}</dd>
                  </div>
                  <div className="flex items-baseline justify-between gap-4">
                    <dt className="text-muted">Share of all rare-disease data</dt>
                    <dd className="font-mono text-ink">{pct(m.rareShare)}</dd>
                  </div>
                  <div className="flex items-baseline justify-between gap-4">
                    <dt className="text-muted">Share of FedAvg weight</dt>
                    <dd className="font-mono text-ink">{pct(m.fedavgWeight)}</dd>
                  </div>
                  <div className="flex items-baseline justify-between gap-4 border-t border-line pt-5">
                    <dt className="text-muted">Knowledge vs weight</dt>
                    <dd className={`font-display text-2xl ${isSpecialist ? "text-accent" : "text-ink"}`}>
                      {(m.rareShare / m.fedavgWeight).toFixed(2)}×
                    </dd>
                  </div>
                </dl>
              </article>
            );
          })}
        </HScroll>
        <Reveal className="mt-16">
          <Card>
            <h3 className="font-display text-2xl font-medium text-ink">Side by side</h3>
            <p className="mt-2 text-sm text-muted">Training split. Centre {specialist.id} highlighted.</p>
            <div className="mt-10">
              <MismatchChart mismatch={mismatch} specialistId={specialist.id} />
            </div>
          </Card>
        </Reveal>
      </Section>

      {/* What counts as rare */}
      <Section
        label="What counts as rare"
        title="Two diseases, about one percent each."
        tags={<StatusTag status="verified" />}
        intro={`A disease is rare here if it holds under 1/${rule.headRatioDivisor} of the largest class's share (${rule.headClass}, ${rule.headSharePct.toFixed(1)}%) - a ${rule.cutoffPct.toFixed(2)}% cutoff. The rule was chosen after seeing the counts, and we say so.`}
      >
        <div className="grid grid-cols-1 [&>*]:min-w-0 gap-8 md:grid-cols-2">
          {rareClasses.map((c, i) => (
            <Reveal key={c.id} delay={i * 0.08}>
              <Card className="h-full">
                <Pill tone="accent">Rare</Pill>
                <h3 className="mt-8 font-display text-4xl font-medium tracking-tight text-ink">{c.name}</h3>
                <div className="mt-10 grid grid-cols-2 gap-6 border-t border-line pt-8">
                  <div>
                    <p className="font-display text-4xl text-ink">{c.sharePct.toFixed(2)}%</p>
                    <p className="mt-2 text-sm text-muted">of all images</p>
                  </div>
                  <div>
                    <p className="font-display text-4xl text-ink">
                      {coverage[c.id]} <span className="text-2xl text-faint">of {n}</span>
                    </p>
                    <p className="mt-2 text-sm text-muted">hospitals hold ≥ {rule.holderMinImages} training images</p>
                  </div>
                </div>
              </Card>
            </Reveal>
          ))}
        </div>
      </Section>

      {/* Who can check */}
      <Section
        tone="sand"
        label="Who can check"
        title="Few hospitals can vouch for a rare disease."
        tags={<StatusTag status="verified" />}
        intro={`Counting hospitals with at least ${rule.holderMinImages} training images of each disease. If one of only three holders misreports, there may be nobody left to contradict it - and it isn't only the rare diseases that are thinly held.`}
      >
        <Reveal>
          <Card className="overflow-x-auto p-0 md:p-0">
            <table className="w-full min-w-[560px] text-left">
              <thead>
                <tr className="border-b border-line font-mono text-[11px] uppercase tracking-[0.16em] text-faint">
                  <th className="px-8 py-6 font-medium md:px-10">Disease</th>
                  <th className="px-4 py-6 font-medium">Share of data</th>
                  <th className="px-8 py-6 font-medium md:px-10">Hospitals holding it</th>
                </tr>
              </thead>
              <tbody>
                {dataset.classes.map((c) => (
                  <tr key={c.id} className="border-b border-line/70 last:border-0">
                    <td className="px-8 py-5 md:px-10">
                      <span className={c.rare ? "font-semibold text-accent" : "text-ink"}>{c.name}</span>
                      {c.rare && <Pill tone="accent" className="ml-3">Rare</Pill>}
                    </td>
                    <td className="px-4 py-5 font-mono text-sm text-muted">{c.sharePct.toFixed(2)}%</td>
                    <td className="px-8 py-5 md:px-10">
                      <span className="flex items-center gap-4">
                        <span className="flex gap-1.5" aria-hidden>
                          {Array.from({ length: n }, (_, k) => (
                            <span
                              key={k}
                              className={`h-3 w-3 rounded-full ${k < coverage[c.id] ? (c.rare ? "bg-accent" : "bg-ink/60") : "bg-sand-deep"}`}
                            />
                          ))}
                        </span>
                        <span className="font-mono text-sm text-ink">
                          {coverage[c.id]} of {n}
                        </span>
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </Reveal>
        <Reveal className="mt-14">
          <PillLink href="/method">How EARN would use this →</PillLink>
        </Reveal>
      </Section>
      <Pager current="/problem" />
    </>
  );
}
