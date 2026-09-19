"use client";

import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { ParallaxSection, Reveal, SectionHeading, useOnScreen, useReducedMotionSafe } from "@/components/ui/Motion";
import { StatusTag } from "@/components/ui/Status";
import type { ProjectStatus } from "@/lib/data";

interface Step {
  title: string;
  symbol: string;
  body: string;
  statuses: ProjectStatus[];
  warning?: string;
}

const STEPS: Step[] = [
  {
    title: "Local training",
    symbol: "U[k]",
    body: "Each hospital trains on its own images and returns only a model update. Standard federated learning.",
    statuses: ["baseline"],
  },
  {
    title: "Evidence",
    symbol: "e(k,c)",
    body: "How strongly each hospital changed the model's row for each disease. No data shared, no self-reported counts.",
    statuses: ["in-progress"],
    warning: "First check (G0a) failed - see Results",
  },
  {
    title: "Coverage check",
    symbol: "n(c) → p(c)",
    body: "Count how many hospitals can vouch for each disease, then blend peer checking with each hospital's own locked history.",
    statuses: ["in-progress"],
  },
  {
    title: "Trust update",
    symbol: "T(k,c)",
    body: "Slow up, fast down: +0.1 when the check agrees, halved when it does not. Full trust takes about ten consistent rounds.",
    statuses: ["in-progress"],
  },
  {
    title: "Aggregation",
    symbol: "w(k,c)",
    body: "Shared layers use FedAvg. Rare-disease rows add trust × evidence share, capped at 50% per hospital. FedAvg and Camp A run today; EARN's weighting does not yet.",
    statuses: ["baseline", "in-progress"],
  },
  {
    title: "Ledger commit",
    symbol: "block t",
    body: "Trust table, coverage and history hashes go on an append-only chain. The contract rejects any trust rise above the allowed step.",
    statuses: ["in-progress"],
  },
];

export function HowItWorks({ blend }: {
  blend: { name: string; holders: number; rare: boolean }[];
}) {
  const reduced = useReducedMotionSafe();
  const [box, onScreen] = useOnScreen<HTMLDivElement>("0px");
  const [active, setActive] = useState(0);

  useEffect(() => {
    if (reduced || !onScreen) return;
    const id = setInterval(() => setActive((a) => (a + 1) % STEPS.length), 2600);
    return () => clearInterval(id);
  }, [reduced, onScreen]);

  const rows = [1, 2, 3, 4, 5].map((n) => ({
    n,
    p: Math.min(1, Math.max(0, (n - 1) / 4)),
    how: ["Own locked history only", "Mostly history", "Half peers, half history", "Mostly peers", "Peers only"][n - 1],
    ours: blend.filter((b) => b.holders === n || (n === 5 && b.holders >= 5)),
  }));

  return (
    <ParallaxSection id="how" glow="amber" className="py-28 md:py-36">
      <div className="mx-auto max-w-7xl px-6">
        <SectionHeading
          eyebrow="02 · How it works"
          title="One training round, six steps."
          tags={
            <>
              <StatusTag status="baseline" />
              <StatusTag status="in-progress" />
            </>
          }
        >
          Steps 2-4 and 6 are EARN, our proposed method. They are designed and specified but not yet
          validated, so they are marked in progress.
        </SectionHeading>

        <div ref={box} className="relative">
          <div aria-hidden className="absolute left-0 right-0 top-[3.25rem] hidden h-px bg-line xl:block">
            {!reduced && (
              <motion.div
                className="h-px w-40 bg-gradient-to-r from-transparent via-accent to-transparent"
                animate={{ x: ["-10%", "1100%"] }}
                transition={{ duration: 5, repeat: Infinity, ease: "linear" }}
              />
            )}
          </div>
          <ol className="grid gap-5 md:grid-cols-3 xl:grid-cols-6">
            {STEPS.map((step, i) => {
              const isActive = !reduced && i === active;
              const earn = step.statuses.includes("in-progress");
              return (
                <Reveal as="li" key={step.title} delay={i * 0.08} tilt className="h-full">
                  <div
                    className={`relative flex h-full flex-col rounded-2xl border p-5 backdrop-blur transition duration-500 ${
                      isActive
                        ? "border-accent/60 bg-ink-800/90 shadow-[0_0_40px_-10px] shadow-accent/40"
                        : earn
                          ? "border-dashed border-progress/30 bg-ink-900/70"
                          : "border-line bg-ink-900/70"
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <span
                        className={`relative z-10 flex h-9 w-9 items-center justify-center rounded-full border font-mono text-sm ${
                          isActive ? "border-accent bg-accent text-ink-950" : "border-line bg-ink-950 text-slate-300"
                        }`}
                      >
                        {i + 1}
                      </span>
                      <span className="font-mono text-xs text-slate-500">{step.symbol}</span>
                    </div>
                    <h3 className="mt-4 font-display text-lg font-semibold text-white">{step.title}</h3>
                    <p className="mt-2 flex-1 text-sm leading-relaxed text-slate-400">{step.body}</p>
                    {step.warning && (
                      <a href="#results" className="mt-3 text-xs text-failed underline-offset-2 hover:underline">
                        ⚠ {step.warning}
                      </a>
                    )}
                    <div className="mt-4 flex flex-wrap gap-1.5">
                      {step.statuses.map((s) => (
                        <StatusTag key={s} status={s} />
                      ))}
                    </div>
                  </div>
                </Reveal>
              );
            })}
          </ol>
        </div>

        <Reveal tilt className="mt-14 grid gap-8 rounded-3xl border border-dashed border-progress/30 bg-ink-900/70 p-8 backdrop-blur lg:grid-cols-[1fr_1.3fr]">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <StatusTag status="in-progress" />
              <span className="font-mono text-xs uppercase tracking-[0.2em] text-slate-500">The core idea</span>
            </div>
            <h3 className="mt-4 font-display text-2xl font-semibold text-white">The coverage blend</h3>
            <p className="mt-3 text-sm leading-relaxed text-slate-400">
              Checking a claim needs someone able to check it. When many hospitals hold a disease,
              peers can contradict a false update. When few do, the only evidence left is the
              hospital&apos;s own track record - which is why that record must be tamper-proof.
              Confidence in peers falls smoothly with coverage, with no hard cutoff.
            </p>
            <pre className="mt-6 overflow-x-auto rounded-xl border border-line bg-ink-950/80 p-4 font-mono text-[13px] leading-relaxed text-slate-300">
{`p(c)   = clip((n(c) - 1) / 4, 0, 1)
a(k,c) = p(c)     · agree_with_peers(k,c)
       + (1-p(c)) · agree_with_own_history(k,c)`}
            </pre>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-line font-mono text-[11px] uppercase tracking-[0.15em] text-slate-500">
                  <th className="py-2 pr-4">Holders n(c)</th>
                  <th className="py-2 pr-4">p(c)</th>
                  <th className="py-2 pr-4">How the claim is checked</th>
                  <th className="py-2">Our diseases (S1)</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.n} className="border-b border-line/60">
                    <td className="py-3 pr-4 font-mono text-slate-300">{r.n === 5 ? "5-6" : r.n}</td>
                    <td className="py-3 pr-4 font-mono text-white">{r.p.toFixed(2)}</td>
                    <td className="py-3 pr-4 text-slate-400">{r.how}</td>
                    <td className="py-3">
                      {r.ours.length ? (
                        <span className="leading-relaxed">
                          {r.ours.map((o, j) => (
                            <span key={o.name} className={o.rare ? "font-semibold text-accent" : "text-slate-400"}>
                              {o.name}
                              {j < r.ours.length - 1 && <span className="text-slate-600">, </span>}
                            </span>
                          ))}
                        </span>
                      ) : (
                        <span className="text-slate-600">{r.n === 1 ? "constructed split S2" : "-"}</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-4 flex flex-wrap items-center gap-2 text-xs text-slate-500">
              <StatusTag status="verified" /> holder counts (≥ 20 training images) are measured;
              rare diseases highlighted. The blend itself is the proposal. Split S2 pushes the rare
              diseases down to 1 holder.
            </p>
          </div>
        </Reveal>
      </div>
    </ParallaxSection>
  );
}
