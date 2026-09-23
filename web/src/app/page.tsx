import Link from "next/link";
import { GapBars } from "@/components/charts/Charts";
import { Pager } from "@/components/ui/Pager";
import { HomeHero } from "@/components/home/HomeHero";
import { Ticker } from "@/components/home/Ticker";
import { Card, PillLink } from "@/components/ui/Buttons";
import { HScroll } from "@/components/ui/HScroll";
import { CountUp, Reveal, Section } from "@/components/ui/Motion";
import { GateBadge, Pill, StatusTag } from "@/components/ui/Status";
import type { ProjectStatus } from "@/lib/data";
import {
  dataset,
  failedGates,
  fmt,
  g0a,
  g0b,
  gate,
  pending,
  rareClasses,
  rareTrainTotal,
  specialist,
  specialistRatio,
  study,
  totalImages,
} from "@/lib/data";

export default function Home() {
  const notRun = pending.items.filter((i) => i.status !== "in-progress").length;
  const pages: { href: string; n: string; title: string; body: string; tags: ProjectStatus[] }[] = [
    {
      href: "/problem",
      n: "01",
      title: "The problem",
      body: `Centre ${specialist.id} holds ${(100 * specialist.rareShare).toFixed(1)}% of rare-disease training images but gets ${(100 * specialist.fedavgWeight).toFixed(1)}% of FedAvg's say.`,
      tags: ["verified"],
    },
    {
      href: "/method",
      n: "02",
      title: "The method",
      body: "EARN's six-step round and the coverage blend - built and tested, running on an oracle signal while the measured one is reworked.",
      tags: ["in-progress", "failed"],
    },
    {
      href: "/results",
      n: "03",
      title: "The results",
      body: `Everything measured so far, including the ${failedGates.length} check${failedGates.length === 1 ? "" : "s"} that came in under the bar.`,
      tags: ["baseline", "failed"],
    },
    {
      href: "/study",
      n: "04",
      title: "The attack study",
      body: `${study.runs} runs: what happens when a hospital lies about rare diseases, across eight aggregation methods.`,
      tags: ["verified"],
    },
    {
      href: "/ledger",
      n: "05",
      title: "The ledger",
      body: "The trust record on a real (local) chain: every EARN round committed, forged boosts rejected.",
      tags: ["verified"],
    },
    {
      href: "/status",
      n: "06",
      title: "The status",
      body: `The six-week plan, every gate's state, and the ${notRun} pieces that haven't run yet.`,
      tags: ["verified"],
    },
  ];

  return (
    <>
      <HomeHero
        nodes={dataset.centres.map((c) => ({ id: c.id, images: c.totalImages, highlight: c.id === specialist.id }))}
      />
      <Ticker
        items={[
          `${fmt(totalImages)} real dermoscopy images`,
          `${dataset.centres.length} real hospitals`,
          `${dataset.classes.length} skin-disease classes`,
          `${rareClasses.length} rare classes, about 1% each`,
          "Images never leave the hospital",
        ]}
      />

      {/* The headline finding */}
      <section className="py-28 md:py-44">
        <div className="mx-auto max-w-6xl px-6">
          <Reveal className="mx-auto max-w-3xl text-center">
            <div className="flex flex-wrap items-center justify-center gap-3">
              <Pill>The core finding</Pill>
              <StatusTag status="verified" />
            </div>
            <p className="mt-10 font-display text-[7rem] font-medium leading-none tracking-tight text-ink md:text-[11rem]">
              <CountUp to={specialistRatio} decimals={2} suffix="×" />
            </p>
            <p className="mx-auto mt-8 max-w-2xl text-pretty text-xl leading-relaxed text-muted">
              The gap between what Centre {specialist.id} knows about rare diseases and how much
              standard federated averaging listens to it.
            </p>
          </Reveal>
          <Reveal delay={0.1} className="mx-auto mt-20 max-w-3xl">
            <Card>
              <GapBars rareShare={specialist.rareShare} fedavgWeight={specialist.fedavgWeight} centre={specialist.id} />
              <p className="mt-8 text-sm text-faint">
                {specialist.rareTrainImages} of {rareTrainTotal} rare-disease training images ·{" "}
                {fmt(specialist.trainImages)} of {fmt(dataset.centres.reduce((a, c) => a + c.trainImages, 0))} training
                images overall
              </p>
            </Card>
          </Reveal>
          <Reveal className="mt-14 flex justify-center">
            <PillLink href="/problem" variant="outline">
              How this breaks down per hospital →
            </PillLink>
          </Reveal>
        </div>
      </section>

      {/* Explore */}
      <Section
        tone="sand"
        label="Explore"
        title="Six pages, one honest picture."
        intro="Each page carries its own status labels, so you always know whether you're looking at a measurement, a baseline, or a proposal."
      >
        <HScroll label="Project pages">
          {pages.map((p) => (
            <Link
              key={p.href}
              href={p.href}
              className="group flex w-[82vw] shrink-0 snap-start flex-col rounded-[28px] border border-line bg-paper p-8 transition hover:border-ink/30 sm:w-[340px] md:p-10"
            >
              <span className="font-mono text-xs text-faint">{p.n}</span>
              <h3 className="mt-6 font-display text-3xl font-medium tracking-tight text-ink">{p.title}</h3>
              <p className="mt-4 flex-1 text-pretty leading-relaxed text-muted">{p.body}</p>
              <div className="mt-8 flex flex-wrap gap-2">
                {p.tags.map((t) => (
                  <StatusTag key={t} status={t} />
                ))}
              </div>
              <span className="mt-8 text-sm font-semibold text-accent group-hover:underline">Open →</span>
            </Link>
          ))}
        </HScroll>
      </Section>

      {/* Where it stands */}
      <Section
        label="Where it stands"
        title="Measured first, claimed second."
        intro="Every decision point was fixed before any result existed, and each check is reported exactly as it came out - never tuned until it agreed. One retry was allowed per check: the baseline cleared its bar on the retry; the evidence signal came back negative and redirected the project to the study it had planned for that outcome."
      >
        <div className="grid grid-cols-1 [&>*]:min-w-0 gap-8 md:grid-cols-3">
          <Reveal>
            <Card className="h-full">
              <div className="flex items-center justify-between gap-2">
                <span className="font-display text-2xl text-ink">Gate {g0a.gate}</span>
                <GateBadge state={gate("G0a")?.state ?? "failed"} />
              </div>
              <p className="mt-8 font-display text-5xl text-failed">{g0a.statistic.toFixed(2)}</p>
              <p className="mt-3 text-sm leading-relaxed text-muted">
                Evidence signal. Needed a Spearman correlation of at least {g0a.threshold}.
              </p>
            </Card>
          </Reveal>
          <Reveal delay={0.08}>
            <Card className="h-full">
              <div className="flex items-center justify-between gap-2">
                <span className="font-display text-2xl text-ink">Gate {g0b.gate}</span>
                <GateBadge state={gate("G0b")?.state ?? "failed"} />
              </div>
              <p className={`mt-8 font-display text-5xl ${g0b.retry?.passed ? "text-verified" : "text-failed"}`}>
                {(g0b.retry?.statistic ?? g0b.statistic).toFixed(3)}
              </p>
              <p className="mt-3 text-sm leading-relaxed text-muted">
                FedAvg balanced accuracy{g0b.retry ? ` after the retry (first attempt ${g0b.statistic.toFixed(3)})` : ""}. Needed at least {g0b.threshold}.
              </p>
            </Card>
          </Reveal>
          <Reveal delay={0.16}>
            <Card className="h-full">
              <div className="flex items-center justify-between gap-2">
                <span className="font-display text-2xl text-ink">Gate G1</span>
                <GateBadge state={gate("G1")?.state ?? "pending"} />
              </div>
              <p className="mt-8 font-display text-5xl text-failed">
                {study.gateG1.s2.balanced_accuracy_drop.toFixed(3)}
              </p>
              <p className="mt-3 text-sm leading-relaxed text-muted">
                The attack works on the specialist split, but costs this much balanced accuracy - more than the
                0.03 that would keep it hidden. On the natural split it barely dents rare F1.
              </p>
            </Card>
          </Reveal>
        </div>
        <Reveal className="mt-14 flex flex-wrap gap-3">
          <PillLink href="/results">See all results</PillLink>
          <PillLink href="/status" variant="outline">
            Full status
          </PillLink>
        </Reveal>
      </Section>
      <Pager current="/" />
    </>
  );
}
