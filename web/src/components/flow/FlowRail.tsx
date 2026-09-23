"use client";

import { motion, useInView, useScroll, useSpring, useTransform } from "framer-motion";
import Link from "next/link";
import { useRef } from "react";
import { useReducedMotionSafe } from "@/components/ui/Motion";
import { Pill, StatusTag } from "@/components/ui/Status";
import type { ProjectStatus } from "@/lib/data";
import {
  coverage,
  dataset,
  fmt,
  g0a,
  ledger,
  mismatch,
  pct,
  peerConfidence,
  rareClasses,
  specialist,
  study,
  studyRow,
  totalImages,
} from "@/lib/data";

type Viz = "hospitals" | "evidence" | "coverage" | "weights" | "chain" | null;

interface Stage {
  title: string;
  lead: string;
  detail: string;
  chips: { k: string; v: string }[];
  status: ProjectStatus[];
  href?: string;
  hrefLabel?: string;
  viz: Viz;
}

const f1 = (m: string, a: "none" | "A1" | "A2" | "A3") => studyRow("s1", m, a);
const rare5 = rareClasses[0];
const rare6 = rareClasses[1];
const chain = ledger.onChain;

const fedavgNone = f1("fedavg", "none");
const campNone = f1("camp_a", "none");
const fedavgA2 = f1("fedavg", "A2");
const clippedA2 = f1("fedavg_clipped", "A2");
const g0aRetry = g0a.retry?.definitions?.[0];

const STAGES: Stage[] = [
  {
    title: "Six hospitals hold the data",
    lead: "Nothing leaves the building.",
    detail:
      `Fed-ISIC2019 is six real hospitals, not a simulated split. They differ wildly in size - ` +
      `from ${fmt(mismatch[0].trainImages)} training images down to ${fmt(mismatch[5].trainImages)} - ` +
      `and that size is what standard federated averaging turns into influence.`,
    chips: [
      { k: "Images", v: fmt(totalImages) },
      { k: "Hospitals", v: String(dataset.centres.length) },
      { k: "Diseases", v: String(dataset.classes.length) },
    ],
    status: ["verified"],
    href: "/problem",
    hrefLabel: "Why that is a problem",
    viz: "hospitals",
  },
  {
    title: "One pass through DenseNet-121",
    lead: "Turn every image into 1024 numbers, once.",
    detail:
      "The backbone runs over all images a single time and the features are saved. Every experiment " +
      "after this trains only the small classifier head on those features, which is what makes 408 " +
      "runs affordable on a laptop.",
    chips: [
      { k: "Per image", v: "1024 features" },
      { k: "Cost", v: "~35 min once" },
      { k: "Then", v: "minutes per run" },
    ],
    status: ["verified"],
    viz: null,
  },
  {
    title: "Each hospital trains on its own images",
    lead: "The only thing sent back is a model update.",
    detail:
      "Every round, all six hospitals train the head locally and return the change they made. " +
      "This part is ordinary federated learning and it is the baseline everything is measured against.",
    chips: [
      { k: "Rounds", v: String(study.rounds) },
      { k: "Seeds", v: study.seeds.join(", ") },
      { k: "Sent", v: "updates only" },
    ],
    status: ["baseline"],
    viz: null,
  },
  {
    title: "Size check: clip to the median",
    lead: "Nobody buys influence by shouting.",
    detail:
      `Each update is scaled down to the median update size. Measured effect on the sleeper attack ` +
      `(A2): rare-disease F1 falls to ${fedavgA2?.rare_macro_f1.mean.toFixed(3) ?? "-"} under plain ` +
      `FedAvg, and clipping recovers most of it at ${clippedA2?.rare_macro_f1.mean.toFixed(3) ?? "-"}.`,
    chips: [
      { k: "FedAvg under A2", v: fedavgA2?.rare_macro_f1.mean.toFixed(3) ?? "-" },
      { k: "With clipping", v: clippedA2?.rare_macro_f1.mean.toFixed(3) ?? "-" },
    ],
    status: ["baseline", "verified"],
    href: "/study",
    hrefLabel: "All attacks and methods",
    viz: null,
  },
  {
    title: "Read the evidence",
    lead: "This is the step that broke.",
    detail:
      `EARN needs to infer, from the update alone, how much of each disease a hospital holds. Gate ` +
      `G0a checked that and it came out inverted: ${g0a.statistic.toFixed(2)} where ${g0a.threshold} ` +
      `was needed, and the one allowed retry scored ${g0aRetry ? g0aRetry.statistic.toFixed(3) : "-"}. ` +
      `A hospital that has never seen a disease moves that disease's row the most.`,
    chips: [
      { k: "G0a", v: g0a.statistic.toFixed(2) },
      { k: "Retry", v: g0aRetry ? g0aRetry.statistic.toFixed(3) : "-" },
      { k: "Needed", v: `>= ${g0a.threshold}` },
    ],
    status: ["failed"],
    href: "/results",
    hrefLabel: "The failed gate in full",
    viz: "evidence",
  },
  {
    title: "Count who can vouch",
    lead: "Coverage decides how the claim is checked.",
    detail:
      `Counting hospitals with at least ${dataset.rareRule.holderMinImages} training images: ` +
      `${rare5.name} has ${coverage[rare5.id]}, ${rare6.name} has ${coverage[rare6.id]}. EARN would ` +
      `blend peer checking with a hospital's own history in that proportion - peers matter less when ` +
      `fewer peers can contradict.`,
    chips: [
      { k: rare5.name, v: `${coverage[rare5.id]} of 6 · p ${peerConfidence(coverage[rare5.id]).toFixed(2)}` },
      { k: rare6.name, v: `${coverage[rare6.id]} of 6 · p ${peerConfidence(coverage[rare6.id]).toFixed(2)}` },
    ],
    status: ["verified", "in-progress"],
    href: "/method",
    hrefLabel: "The coverage blend",
    viz: "coverage",
  },
  {
    title: "Update trust, slowly",
    lead: "Earned in ten rounds, halved in one.",
    detail:
      "Agreement raises a hospital's trust for that disease by a small fixed step; disagreement halves " +
      "it. EARN is fully built and unit-tested, but because the evidence signal failed, it could only " +
      "be run on an oracle signal - labelled exploratory, and it still did not meet its G2 conditions.",
    chips: [
      { k: "Up", v: "+0.1" },
      { k: "Down", v: "x0.5" },
      { k: "Run on", v: "oracle signal" },
    ],
    status: ["in-progress"],
    href: "/study",
    hrefLabel: "Exploratory EARN results",
    viz: null,
  },
  {
    title: "Aggregate the round",
    lead: "Whose update counts, and by how much.",
    detail:
      `FedAvg weights Centre ${specialist.id} at ${pct(specialist.fedavgWeight)} - its share of images. ` +
      `Weighting by update-evidence (Camp A) gives it ` +
      `${campNone ? pct(campNone.specialist_weight_5.mean) : "-"} instead, less than FedAvg, and ` +
      `rare-disease F1 drops from ${fedavgNone?.rare_macro_f1.mean.toFixed(3) ?? "-"} to ` +
      `${campNone?.rare_macro_f1.mean.toFixed(3) ?? "-"}. The signal that was supposed to help hurts.`,
    chips: [
      { k: "FedAvg rare F1", v: fedavgNone?.rare_macro_f1.mean.toFixed(3) ?? "-" },
      { k: "Camp A rare F1", v: campNone?.rare_macro_f1.mean.toFixed(3) ?? "-" },
    ],
    status: ["baseline", "verified"],
    href: "/study",
    hrefLabel: "Every method compared",
    viz: "weights",
  },
  {
    title: "Commit the round to the chain",
    lead: "A record the coordinator cannot bend.",
    detail:
      chain
        ? `The trust table, coverage and each hospital's history hash go into one block that points at ` +
          `the last. ${fmt(chain.roundsCommitted)} real rounds were replayed on the Hardhat contract: ` +
          `about ${Math.round(chain.gasPerRound.mean / 1000)}k gas and ` +
          `${chain.latencyMsPerRound.mean.toFixed(1)} ms per round, and every forged trust boost was ` +
          `rejected.`
        : "The trust table, coverage and each hospital's history hash go into one block that points at the last.",
    chips: chain
      ? [
          { k: "Rounds on chain", v: fmt(chain.roundsCommitted) },
          { k: "Gas / round", v: `~${Math.round(chain.gasPerRound.mean / 1000)}k` },
          { k: "Latency", v: `${chain.latencyMsPerRound.mean.toFixed(1)} ms` },
          { k: "Forged boosts", v: chain.allTamperRejected ? "all rejected" : "see results" },
        ]
      : [],
    status: ["verified", "in-progress"],
    href: "/ledger",
    hrefLabel: "Try breaking the chain",
    viz: "chain",
  },
  {
    title: "What the whole thing showed",
    lead: "The honest outcome, not the hoped-for one.",
    detail:
      `${study.runs} runs across methods, attacks and both splits. The pre-registered attack gate G1 ` +
      `did not pass on either split, so the paper follows the plan's own fallback: an empirical study ` +
      `of how these methods behave on a real hospital split, with the failures reported as failures.`,
    chips: [
      { k: "Runs", v: String(study.runs) },
      { k: "G1", v: "failed on S1 and S2" },
      { k: "Framing", v: "Fallback F1" },
    ],
    status: ["verified", "failed"],
    href: "/status",
    hrefLabel: "Where the project stands",
    viz: null,
  },
];

// ---------------------------------------------------------------- mini visuals

function Hospitals() {
  const max = Math.max(...mismatch.map((m) => m.trainImages));
  return (
    <div className="flex flex-wrap items-end gap-3">
      {mismatch.map((m) => {
        const size = 14 + 30 * Math.sqrt(m.trainImages / max);
        const isSpec = m.id === specialist.id;
        return (
          <span key={m.id} className="flex flex-col items-center gap-2">
            <span
              className={`block rounded-full ${isSpec ? "bg-accent" : "bg-sand-deep"}`}
              style={{ width: size, height: size }}
              aria-hidden
            />
            <span className={`font-mono text-[10px] ${isSpec ? "text-accent" : "text-faint"}`}>C{m.id}</span>
          </span>
        );
      })}
    </div>
  );
}

function Evidence() {
  return (
    <div className="flex flex-wrap items-end gap-2">
      {g0a.perClassSpearman.map((rho, i) => (
        <span key={i} className="flex flex-col items-center gap-1.5" title={`${g0a.classNames[i]}: ${rho.toFixed(2)}`}>
          <span className="flex h-12 items-center">
            <span
              className={`block w-3 rounded-sm ${rho < 0 ? "bg-failed/70" : "bg-sand-deep"}`}
              style={{ height: `${Math.max(6, Math.abs(rho) * 44)}px` }}
              aria-hidden
            />
          </span>
          <span className={`font-mono text-[9px] ${rho < 0 ? "text-failed" : "text-faint"}`}>
            {rho < 0 ? "-" : "+"}
          </span>
        </span>
      ))}
      <span className="w-full text-xs text-muted sm:ml-2 sm:w-auto sm:self-center">
        red = points the wrong way ({g0a.perClassSpearman.filter((r) => r < 0).length} of{" "}
        {g0a.perClassSpearman.length})
      </span>
    </div>
  );
}

function Coverage() {
  return (
    <div className="space-y-3">
      {rareClasses.map((c) => (
        <div key={c.id} className="flex items-center gap-3">
          <span className="w-24 shrink-0 text-xs text-muted sm:w-32">{c.name}</span>
          <span className="flex gap-1.5" aria-hidden>
            {dataset.centres.map((_, k) => (
              <span
                key={k}
                className={`h-2.5 w-2.5 rounded-full ${k < coverage[c.id] ? "bg-accent" : "bg-sand-deep"}`}
              />
            ))}
          </span>
          <span className="font-mono text-xs text-ink">{coverage[c.id]} of {dataset.centres.length}</span>
        </div>
      ))}
    </div>
  );
}

function Weights() {
  const rows = [
    { label: "FedAvg", value: specialist.fedavgWeight, tone: "bg-sand-deep" },
    { label: "Camp A", value: campNone?.specialist_weight_5.mean ?? 0, tone: "bg-failed/60" },
  ];
  const max = Math.max(...rows.map((r) => r.value)) * 1.25;
  return (
    <div className="space-y-3">
      {rows.map((r) => (
        <div key={r.label} className="flex items-center gap-3">
          <span className="w-20 text-xs text-muted">{r.label}</span>
          <span className="h-2.5 flex-1 overflow-hidden rounded-full bg-sand">
            <span className={`block h-full rounded-full ${r.tone}`} style={{ width: `${(r.value / max) * 100}%` }} />
          </span>
          <span className="font-mono text-xs text-ink">{pct(r.value)}</span>
        </div>
      ))}
      <p className="text-xs text-faint">Centre {specialist.id}&apos;s weight on the {rare5.name} row.</p>
    </div>
  );
}

function ChainViz() {
  return (
    <div className="flex flex-wrap items-center gap-1.5" aria-hidden>
      {Array.from({ length: 5 }, (_, i) => (
        <span key={i} className="flex items-center gap-1.5">
          <span className="h-6 w-7 rounded-md border border-accent/50 bg-accent-soft/60 sm:h-7 sm:w-9" />
          {i < 4 && <span className="h-px w-3 bg-accent/50" />}
        </span>
      ))}
    </div>
  );
}

const VIZ: Record<Exclude<Viz, null>, () => React.ReactElement> = {
  hospitals: Hospitals,
  evidence: Evidence,
  coverage: Coverage,
  weights: Weights,
  chain: ChainViz,
};

// ---------------------------------------------------------------- the rail

function StageCard({ stage, index, reduced }: { stage: Stage; index: number; reduced: boolean }) {
  const ref = useRef<HTMLLIElement>(null);
  const active = useInView(ref, { amount: 0.55, margin: "-15% 0px -25% 0px" });
  const Visual = stage.viz ? VIZ[stage.viz] : null;

  return (
    <li ref={ref} className="relative pl-16 md:pl-24">
      {/* number on the rail */}
      <span
        className={`absolute left-0 top-1 flex h-12 w-12 items-center justify-center rounded-full border font-mono text-sm transition-colors duration-500 md:h-14 md:w-14 ${
          active && !reduced
            ? "border-accent bg-accent text-cream"
            : "border-line bg-paper text-muted"
        }`}
      >
        {String(index + 1).padStart(2, "0")}
      </span>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.2 }}
        transition={{ duration: 0.65, ease: [0.25, 0.8, 0.3, 1] }}
        className={`rounded-[28px] border bg-paper p-8 transition-shadow duration-500 md:p-10 ${
          active && !reduced ? "border-accent/40 shadow-[0_24px_60px_-45px_rgba(29,41,41,0.5)]" : "border-line"
        }`}
      >
        <div className="flex flex-wrap items-center gap-2">
          {stage.status.map((s) => (
            <StatusTag key={s} status={s} />
          ))}
        </div>
        <h3 className="mt-6 font-display text-3xl font-medium leading-tight tracking-tight text-ink md:text-4xl">
          {stage.title}
        </h3>
        <p className="mt-3 text-lg text-accent">{stage.lead}</p>
        <p className="mt-5 max-w-2xl text-pretty leading-relaxed text-muted">{stage.detail}</p>

        {Visual && (
          <div className="mt-8 overflow-x-auto rounded-2xl bg-sand/50 p-6">
            <Visual />
          </div>
        )}

        {stage.chips.length > 0 && (
          <dl className="mt-8 flex flex-wrap gap-x-10 gap-y-4 border-t border-line pt-6">
            {stage.chips.map((c) => (
              <div key={c.k}>
                <dt className="font-mono text-[10px] uppercase tracking-[0.16em] text-faint">{c.k}</dt>
                <dd className="mt-1 font-display text-xl text-ink">{c.v}</dd>
              </div>
            ))}
          </dl>
        )}

        {stage.href && (
          <Link
            href={stage.href}
            className="mt-7 inline-flex items-center gap-2 text-sm font-semibold text-accent hover:underline"
          >
            {stage.hrefLabel} &rarr;
          </Link>
        )}
      </motion.div>
    </li>
  );
}

export function FlowRail() {
  const wrap = useRef<HTMLDivElement>(null);
  const reduced = useReducedMotionSafe();
  const { scrollYProgress } = useScroll({ target: wrap, offset: ["start 0.55", "end 0.85"] });
  const smooth = useSpring(scrollYProgress, { stiffness: 90, damping: 28, mass: 0.35 });
  const fillHeight = useTransform(smooth, (v) => `${Math.min(100, Math.max(0, v * 100))}%`);

  return (
    <div ref={wrap} className="relative mx-auto max-w-4xl px-6 pb-8">
      {/* the rail itself */}
      <div aria-hidden className="absolute bottom-0 left-[calc(1.5rem+1.5rem)] top-2 w-px bg-line md:left-[calc(1.5rem+1.75rem)]">
        <motion.div
          className="w-px bg-accent"
          style={{ height: reduced ? "100%" : fillHeight }}
        />
        {!reduced && (
          <motion.span
            className="absolute -left-[3px] block h-[7px] w-[7px] rounded-full bg-accent shadow-[0_0_0_4px_rgba(44,106,100,0.15)]"
            style={{ top: fillHeight }}
          />
        )}
      </div>

      <ol className="space-y-12 md:space-y-16">
        {STAGES.map((s, i) => (
          <StageCard key={s.title} stage={s} index={i} reduced={reduced} />
        ))}
      </ol>

      <div className="mt-16 pl-16 md:pl-24">
        <Pill tone="accent">End of the round</Pill>
        <p className="mt-4 max-w-2xl text-pretty leading-relaxed text-muted">
          {study.framing.decision}
        </p>
      </div>
    </div>
  );
}
