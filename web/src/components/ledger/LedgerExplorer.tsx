"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import { useOnScreen, useReducedMotionSafe } from "@/components/ui/Motion";
import { DataBadge } from "@/components/ui/Status";
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
    <div className="flex items-start justify-between gap-6 border-b border-line/70 py-4 text-sm last:border-0">
      <span className="text-muted">{label}</span>
      <span className={`text-right font-mono ${bad ? "text-failed" : "text-ink"}`}>{value}</span>
    </div>
  );
}

export function LedgerExplorer({ ledger, classShort }: { ledger: Ledger; classShort: string[] }) {
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
    const id = setInterval(() => setVisible((v) => (v >= total ? START : v + 1)), 3000);
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
    <div className="grid gap-8 lg:grid-cols-[1.5fr_1fr]">
      <div className="overflow-hidden rounded-[28px] border border-line bg-paper">
        <div ref={box} className="paper-grid h-[360px] md:h-[460px]">
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
        <div className="flex flex-wrap items-center justify-between gap-4 border-t border-line px-8 py-6">
          <p className="text-sm text-muted">
            {reduced ? "Showing a static chain." : "A new round is committed every few seconds."} Click a
            block to inspect it.
          </p>
          {tampered === null ? (
            <button
              type="button"
              onClick={() => {
                setTampered(tamperTarget);
                setPinned(tamperTarget);
              }}
              className="rounded-full border border-failed/40 bg-failed-soft px-5 py-2.5 text-sm font-semibold text-failed transition hover:border-failed"
            >
              Tamper with round {tamperTarget}
            </button>
          ) : (
            <button
              type="button"
              onClick={() => {
                setTampered(null);
                setPinned(null);
              }}
              className="rounded-full border border-ink/20 bg-paper px-5 py-2.5 text-sm font-semibold text-ink transition hover:border-ink/50"
            >
              Restore the chain
            </button>
          )}
        </div>
      </div>

      <div className="rounded-[28px] border border-line bg-paper p-8 md:p-10">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h3 className="font-display text-3xl font-medium text-ink">Block {rec.round}</h3>
          <DataBadge status={ledger._meta.status} title={ledger._meta.note} />
        </div>
        <div className="mt-6" aria-live="polite">
          <Field label="Previous block hash" value={short(rec.prevHash)} bad={isBroken} />
          <Field label="This block hash" value={isEdited ? "no longer matches" : short(rec.blockHash)} bad={isEdited} />
          <Field label="Trust table T(k,c)" value={short(rec.trustTableHash)} bad={isEdited} />
          <Field label="History hashes" value={short(rec.historyHash)} />
          <Field
            label="Coverage n(c)"
            value={<span className="text-xs leading-relaxed">{rec.coverage.map((n, i) => `${classShort[i]} ${n}`).join(" · ")}</span>}
          />
          <Field label="Largest trust rise" value={`${rec.maxTrustRise.toFixed(3)} ≤ ${ledger.rules.maxTrustStep} ✓`} />
          {rec.chainBlockHash && <Field label="On-chain block hash (keccak)" value={short(rec.chainBlockHash)} />}
          {rec.gas !== undefined && <Field label="Gas to commit" value={rec.gas.toLocaleString("en-US")} />}
        </div>
        {isEdited && (
          <p className="mt-6 rounded-2xl bg-failed-soft p-5 text-sm leading-relaxed text-failed">
            This round was edited after it was committed. Its hash changed, so it no longer matches what
            round {rec.round + 1} recorded as its predecessor.
          </p>
        )}
        {isBroken && (
          <p className="mt-6 rounded-2xl bg-failed-soft p-5 text-sm leading-relaxed text-failed">
            Link broken: this block points to the original round {tampered}, which no longer exists. Every
            later block trips the same check.
          </p>
        )}
        {!isEdited && !isBroken && <p className="mt-6 text-sm leading-relaxed text-faint">{ledger._meta.note}</p>}
      </div>
    </div>
  );
}
