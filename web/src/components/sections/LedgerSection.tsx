"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import { ParallaxSection, Reveal, SectionHeading, useOnScreen, useReducedMotionSafe } from "@/components/ui/Motion";
import { DataBadge, SampleBadge, StatusTag } from "@/components/ui/Status";
import type { Ledger } from "@/lib/data";

const LedgerChain = dynamic(() => import("@/components/three/LedgerChain"), {
  ssr: false,
  loading: () => <div className="h-full w-full" />,
});

const START = 6;
const STATIC_BLOCKS = 12;
const short = (h: string) => `${h.slice(0, 10)}…${h.slice(-4)}`;

function Field({ label, value, bad }: { label: string; value: React.ReactNode; bad?: boolean }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-line/60 py-2 text-sm last:border-0">
      <span className="text-slate-500">{label}</span>
      <span className={`text-right font-mono ${bad ? "text-failed" : "text-slate-200"}`}>{value}</span>
    </div>
  );
}

export function LedgerSection({ ledger, classShort }: { ledger: Ledger; classShort: string[] }) {
  const reduced = useReducedMotionSafe();
  const [box, onScreen] = useOnScreen<HTMLDivElement>("0px");
  const total = ledger.rounds.length;
  const [visible, setVisible] = useState(START);
  const [tampered, setTampered] = useState<number | null>(null);
  const [pinned, setPinned] = useState<number | null>(null);

  // Rounds keep getting committed while the chain is on screen; under reduced motion the
  // chain is shown already built.
  useEffect(() => {
    if (reduced || !onScreen || tampered !== null) return;
    const id = setInterval(() => setVisible((v) => (v >= total ? START : v + 1)), 2600);
    return () => clearInterval(id);
  }, [reduced, onScreen, tampered, total]);

  const shown = reduced ? Math.min(STATIC_BLOCKS, total) : Math.min(visible, total);
  const latest = ledger.rounds[shown - 1].round;
  const selected = pinned !== null && pinned <= latest ? pinned : latest;
  const rec = ledger.rounds.find((r) => r.round === selected)!;
  const tamperTarget = Math.max(1, latest - 3);
  const isEdited = tampered !== null && selected === tampered;
  const isBroken = tampered !== null && selected > tampered;

  return (
    <ParallaxSection id="ledger" glow="teal" className="py-28 md:py-36">
      <div className="mx-auto max-w-7xl px-6">
        <SectionHeading
          eyebrow="04 · The ledger"
          title="A record nobody can quietly rewrite."
          tags={
            <>
              <StatusTag status="in-progress" />
              <DataBadge status={ledger._meta.status} title={ledger._meta.note} />
            </>
          }
        >
          Every round, the trust table, coverage and a hash of each hospital&apos;s history are
          committed to an append-only chain. Each block stores the hash of the one before it, so
          changing any past round breaks every link after it.
        </SectionHeading>

        <div className="grid gap-8 lg:grid-cols-[1.5fr_1fr]">
          <Reveal tilt className="relative overflow-hidden rounded-3xl border border-line bg-ink-900/70 backdrop-blur">
            <div ref={box} className="h-[360px] md:h-[440px]">
              <LedgerChain
                rounds={ledger.rounds.map((r) => r.round)}
                visible={shown}
                tamperedFrom={tampered}
                selected={selected}
                onSelect={setPinned}
                active={onScreen}
                reduced={reduced}
              />
            </div>
            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line px-5 py-4">
              <p className="text-xs text-slate-500">
                {reduced ? "Showing a static chain." : "A new round is committed every few seconds."} Click a
                block to inspect it.
              </p>
              {tampered === null ? (
                <button
                  onClick={() => {
                    setTampered(tamperTarget);
                    setPinned(tamperTarget);
                  }}
                  className="rounded-full border border-failed/50 px-4 py-2 text-xs font-semibold text-failed transition hover:bg-failed/10"
                >
                  Tamper with round {tamperTarget}
                </button>
              ) : (
                <button
                  onClick={() => {
                    setTampered(null);
                    setPinned(null);
                  }}
                  className="rounded-full border border-line px-4 py-2 text-xs font-semibold text-slate-200 transition hover:border-slate-400"
                >
                  Restore the chain
                </button>
              )}
            </div>
          </Reveal>

          <Reveal tilt delay={0.1} className="rounded-3xl border border-line bg-ink-900/70 p-6 backdrop-blur" >
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="font-display text-lg font-semibold text-white">Block {rec.round}</h3>
              <DataBadge status={ledger._meta.status} title={ledger._meta.note} />
            </div>
            <div className="mt-3" aria-live="polite">
              <Field label="Previous block hash" value={short(rec.prevHash)} bad={isBroken} />
              <Field label="This block hash" value={isEdited ? "no longer matches" : short(rec.blockHash)} bad={isEdited} />
              <Field label="Trust table T(k,c)" value={short(rec.trustTableHash)} bad={isEdited} />
              <Field label="History hashes" value={short(rec.historyHash)} />
              <Field
                label="Coverage n(c)"
                value={
                  <span className="text-xs">
                    {rec.coverage.map((n, i) => `${classShort[i]} ${n}`).join(" · ")}
                  </span>
                }
              />
              <Field
                label="Largest trust rise"
                value={`${rec.maxTrustRise.toFixed(3)} ≤ ${ledger.rules.maxTrustStep} ✓`}
              />
            </div>
            {isEdited && (
              <p className="mt-4 rounded-xl border border-failed/40 bg-failed/10 p-3 text-sm text-rose-200">
                This round was edited after it was committed. Its hash changed, so it no longer matches
                what round {rec.round + 1} recorded as its predecessor.
              </p>
            )}
            {isBroken && (
              <p className="mt-4 rounded-xl border border-failed/40 bg-failed/10 p-3 text-sm text-rose-200">
                Link broken: this block points to the original round {tampered}, which no longer exists.
                Every later block fails the same check.
              </p>
            )}
            {!isEdited && !isBroken && (
              <p className="mt-4 text-xs leading-relaxed text-slate-500">{ledger._meta.note}</p>
            )}
          </Reveal>
        </div>

        <div className="mt-8 grid gap-6 md:grid-cols-3">
          <Reveal delay={0.05} className="rounded-3xl border border-dashed border-progress/30 bg-ink-900/70 p-6 backdrop-blur">
            <StatusTag status="in-progress" />
            <h3 className="mt-4 font-display text-lg font-semibold text-white">It locks the history</h3>
            <p className="mt-2 text-sm leading-relaxed text-slate-400">
              When only one or two hospitals hold a disease, the only evidence is a hospital&apos;s own
              past. If that record could be rewritten, an attacker would fabricate a consistent past
              and pass the check. The check only means something on an append-only record.
            </p>
          </Reveal>
          <Reveal delay={0.15} className="rounded-3xl border border-dashed border-progress/30 bg-ink-900/70 p-6 backdrop-blur">
            <div className="flex flex-wrap gap-2">
              <StatusTag status="in-progress" />
              <SampleBadge title="Illustration of the contract rule, not a recorded event" />
            </div>
            <h3 className="mt-4 font-display text-lg font-semibold text-white">It binds the coordinator</h3>
            <p className="mt-2 text-sm leading-relaxed text-slate-400">
              The contract rejects any trust value rising faster than the fixed step, so even whoever
              runs aggregation can&apos;t quietly boost a hospital.
            </p>
            <div className="mt-4 rounded-xl border border-line bg-ink-950/80 p-3 font-mono text-xs leading-relaxed">
              <p className="text-slate-400">submit T(centre 5, vascular) 0.30 → 0.60</p>
              <p className="text-failed">
                ✕ rejected: rise 0.30 &gt; max step {ledger.rules.maxTrustStep.toFixed(2)}
              </p>
            </div>
          </Reveal>
          <Reveal delay={0.25} className="rounded-3xl border border-line bg-ink-900/70 p-6 backdrop-blur">
            <p className="font-mono text-xs uppercase tracking-[0.2em] text-slate-500">Fair criticism</p>
            <h3 className="mt-3 font-display text-lg font-semibold text-white">Why not just a database?</h3>
            <p className="mt-2 text-sm leading-relaxed text-slate-400">
              A signed append-only log kept by a trusted third party would achieve much of this. Our
              answer: in a federation of competing hospitals there is no mutually trusted third party.
              We state this plainly rather than overclaim.
            </p>
          </Reveal>
        </div>
      </div>
    </ParallaxSection>
  );
}
