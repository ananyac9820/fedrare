import type { Metadata } from "next";
import { LedgerExplorer } from "@/components/ledger/LedgerExplorer";
import { Card, PillLink } from "@/components/ui/Buttons";
import { HScroll } from "@/components/ui/HScroll";
import { PageHeader, Reveal, Section } from "@/components/ui/Motion";
import { DataBadge, Pill, SampleBadge, StatusTag } from "@/components/ui/Status";
import { TrustCurves } from "@/components/charts/Charts";
import { CHART } from "@/components/ui/ChartTooltip";
import { SHORT, dataset, fmt, ledger, study } from "@/lib/data";

export const metadata: Metadata = { title: "The ledger" };

const COMMITTED = [
  { title: "Round number", body: "Which training round this block records." },
  { title: "Trust table T(k,c)", body: "A hash of every hospital's trust for every disease after this round." },
  { title: "Coverage n(c)", body: "How many hospitals showed real evidence for each disease this round." },
  { title: "History hashes", body: "One per hospital - the record its own-history check reads from." },
  { title: "Previous block hash", body: "The link that makes rewriting any earlier round detectable." },
];

export default function LedgerPage() {
  const chain = ledger.onChain;
  const sample = ledger._meta.status === "sample";
  const t = chain?.tamperExample;
  const overhead = study.overheadMs;
  const fu = study.ledgerFollowup;
  const revert = chain?.tamperError?.match(/custom error '([^']+)'/)?.[1] ?? chain?.tamperError;

  return (
    <>
      <PageHeader
        eyebrow="05 · The ledger"
        tags={
          <>
            <StatusTag status={sample ? "in-progress" : "verified"} />
            <DataBadge status={ledger._meta.status} title={ledger._meta.note} />
          </>
        }
        title="A record nobody can quietly rewrite."
      >
        Every round, EARN commits its trust table, coverage and a hash of every hospital&apos;s history to
        an append-only chain. The contract is built and tested on a local Hardhat chain; every round below
        is a real round EARN produced. EARN itself ran on an oracle evidence signal, because the real
        signal failed Gate G0a - so these are real mechanics, not a validated method.
      </PageHeader>

      <Section
        label="Try it"
        title="Tamper with a past round."
        tags={sample ? <SampleBadge title={ledger._meta.note} /> : <StatusTag status="verified" />}
        intro={sample
          ? "The hash links below are real SHA-256; the block contents are sample data until EARN and the contract exist."
          : `${ledger.rounds.length} real rounds from the exploratory EARN run (natural split, no attack, seed 42). Hashes are the Python ledger's SHA-256 chain; each round was also committed to the EarnLedger contract.`}
      >
        <Reveal>
          <LedgerExplorer ledger={ledger} classShort={dataset.classes.map((c) => SHORT[c.name] ?? c.name)} />
        </Reveal>
      </Section>

      <Section tone="sand" label="Each block" title="What gets committed every round.">
        <HScroll label="What each ledger block contains">
          {COMMITTED.map((c, i) => (
            <article key={c.title} className="flex w-[82vw] shrink-0 snap-start flex-col rounded-[28px] border border-line bg-paper p-8 sm:w-[300px] md:p-10">
              <span className="font-mono text-xs text-faint">{String(i + 1).padStart(2, "0")}</span>
              <h3 className="mt-8 font-display text-2xl font-medium tracking-tight text-ink">{c.title}</h3>
              <p className="mt-4 text-pretty leading-relaxed text-muted">{c.body}</p>
            </article>
          ))}
        </HScroll>
      </Section>

      <Section label="Why a blockchain" title="Two jobs a plain database can't do on its own." tags={<StatusTag status="in-progress" />}>
        <div className="grid grid-cols-1 [&>*]:min-w-0 gap-8 md:grid-cols-3">
          <Reveal>
            <Card dashed className="h-full">
              <Pill>Locks the history</Pill>
              <p className="mt-8 text-pretty leading-relaxed text-muted">
                When only one or two hospitals hold a disease, the only evidence is a hospital&apos;s own
                past. If that record could be rewritten, an attacker would fabricate a consistent past
                and pass the check. The check only means something on an append-only record.
              </p>
            </Card>
          </Reveal>
          <Reveal delay={0.08}>
            <Card dashed className="h-full">
              <div className="flex flex-wrap gap-2">
                <Pill>Binds the coordinator</Pill>
                {t ? <StatusTag status="verified" /> : <SampleBadge title="Illustration of the contract rule, not a recorded event" />}
              </div>
              <p className="mt-8 text-pretty leading-relaxed text-muted">
                The contract rejects any trust value rising faster than the fixed step, so even whoever
                runs aggregation can&apos;t quietly boost a hospital.
              </p>
              <div className="mt-6 rounded-2xl bg-sand/70 p-5 font-mono text-xs leading-relaxed">
                {t ? (
                  <>
                    <p className="text-muted">
                      submit T(centre {t.client}, {SHORT[dataset.classes[t.class].name]}) {(t.from_bps / 1e4).toFixed(2)} →{" "}
                      {(t.to_bps / 1e4).toFixed(2)}
                    </p>
                    <p className="mt-1 text-failed">✕ reverted: {revert} (max step {ledger.rules.maxTrustStep.toFixed(2)})</p>
                    <p className="mt-2 text-faint">Tried on the real final table of all {chain?.runs} runs - rejected every time.</p>
                  </>
                ) : (
                  <>
                    <p className="text-muted">submit T(centre 5, vascular) 0.30 → 0.60</p>
                    <p className="mt-1 text-failed">✕ rejected: rise 0.30 &gt; max step {ledger.rules.maxTrustStep.toFixed(2)}</p>
                  </>
                )}
              </div>
            </Card>
          </Reveal>
          <Reveal delay={0.16}>
            <Card className="h-full">
              <Pill tone="paper">Fair criticism</Pill>
              <p className="mt-8 text-pretty leading-relaxed text-muted">
                A signed append-only log kept by a trusted third party would achieve much of this. Our
                answer: in a federation of competing hospitals there is no mutually trusted third party.
                We state this plainly rather than overclaim.
              </p>
            </Card>
          </Reveal>
        </div>
      </Section>

      <Section
        tone="sand"
        label="Cost"
        title={chain ? "What the chain costs per round." : "Gas and latency: not measured yet."}
        tags={<StatusTag status={chain ? "verified" : "not-run"} />}
        intro={chain ? `${fmt(chain.roundsCommitted)} real EARN rounds from ${chain.runs} runs (both splits, every attack, three seeds) replayed on ${chain.network}; Solidity ${chain.solc}.` : undefined}
      >
        {chain && (
          <div className="grid grid-cols-1 [&>*]:min-w-0 gap-8 md:grid-cols-3">
            <Reveal>
              <Card className="h-full">
                <Pill>Gas per round</Pill>
                <p className="mt-8 font-display text-5xl text-ink">{fmt(Math.round(chain.gasPerRound.mean))}</p>
                <p className="mt-3 font-mono text-xs text-faint">
                  median {fmt(chain.gasPerRound.median)} · max {fmt(chain.gasPerRound.max)} · deploy {fmt(chain.deployGas)}
                </p>
                <p className="mt-6 text-sm leading-relaxed text-muted">
                  One transaction stores 48 trust values, 8 coverage counts and the history hash. Rounds where
                  trust moves a lot cost more storage writes.
                </p>
              </Card>
            </Reveal>
            <Reveal delay={0.08}>
              <Card className="h-full">
                <Pill>Latency per round</Pill>
                <p className="mt-8 font-display text-5xl text-ink">{chain.latencyMsPerRound.mean.toFixed(2)} ms</p>
                <p className="mt-3 font-mono text-xs text-faint">
                  median {chain.latencyMsPerRound.median.toFixed(2)} · max {chain.latencyMsPerRound.max.toFixed(2)} ms
                </p>
                <p className="mt-6 text-sm leading-relaxed text-muted">
                  Send to receipt on a local, instantly-mining chain. A public chain adds block time
                  (seconds) - still small next to a training round.
                </p>
              </Card>
            </Reveal>
            <Reveal delay={0.16}>
              <Card className="h-full">
                <Pill>Checks on real data</Pill>
                <ul className="mt-8 space-y-4 text-sm">
                  <li className="flex justify-between gap-4"><span className="text-muted">On-chain trust equals Python&apos;s</span><span className="font-mono">{chain.allFinalTrustMatch ? "✓ all runs" : "✕"}</span></li>
                  <li className="flex justify-between gap-4"><span className="text-muted">History hash verifies</span><span className="font-mono">{chain.allHistoryVerified ? "✓ all runs" : "✕"}</span></li>
                  <li className="flex justify-between gap-4"><span className="text-muted">Forged boost rejected</span><span className="font-mono">{chain.allTamperRejected ? "✓ all runs" : "✕"}</span></li>
                  {overhead.earn && (
                    <li className="flex justify-between gap-4 border-t border-line pt-4"><span className="text-muted">EARN aggregation, per round</span><span className="font-mono">{overhead.earn.mean.toFixed(1)} ms</span></li>
                  )}
                  {overhead.fedavg && (
                    <li className="flex justify-between gap-4"><span className="text-muted">FedAvg, per round</span><span className="font-mono">{overhead.fedavg.mean.toFixed(1)} ms</span></li>
                  )}
                </ul>
              </Card>
            </Reveal>
          </div>
        )}
        <Reveal className="mt-10">
          <Card>
            <Pill tone="paper">What the ablation found</Pill>
            <p className="mt-6 max-w-3xl text-pretty leading-relaxed text-muted">
              Against attacks A1-A3, making the history editable (the &quot;no ledger&quot; ablation) changed
              nothing measurable: those attackers never needed to rewrite their past. The chain&apos;s other
              job - stopping a coordinator from boosting trust - held on every real round.
            </p>
          </Card>
        </Reveal>
      </Section>

      {fu && (
        <Section
          label="Follow-up · when the specialist turns"
          title="The lock catches the insider - but can't stop the damage."
          tags={
            <>
              <StatusTag status="verified" />
              <Pill tone="paper">Exploratory · oracle evidence</Pill>
            </>
          }
          intro={`Pre-registered after the main study (${fu.entry}): the only hospital holding the rare diseases on the specialist split (centre ${fu.attacker}) is honest for 15 rounds, then attacks. With one holder, EARN can only check a hospital against its own locked history. Mean of seeds ${fu.seeds.join(", ")}.`}
        >
          <div className="grid grid-cols-1 [&>*]:min-w-0 gap-8 lg:grid-cols-[1.35fr_1fr]">
            <Reveal>
              <Card className="h-full">
                <h3 className="font-display text-2xl font-medium text-ink">The turned specialist&apos;s trust</h3>
                <p className="mt-2 text-sm text-muted">Vascular-lesion row, per round.</p>
                <div className="mt-8">
                  <TrustCurves turnRound={16} series={[
                    { name: "Locked history (ledger)", color: CHART.accent, values: fu.trajectory.earn.trust },
                    { name: "Editable history (no ledger)", color: CHART.failed, values: fu.trajectory.noLedger.trust },
                  ]} />
                </div>
              </Card>
            </Reveal>
            <Reveal delay={0.08}>
              <Card className="h-full">
                <div className="flex flex-wrap items-center gap-2">
                  <Pill>Detected</Pill>
                  <StatusTag status="verified" />
                </div>
                <p className="mt-6 text-pretty leading-relaxed text-ink">
                  With the locked history, trust falls to <span className="font-mono">{fu.trajectory.earn.minTrust.toFixed(2)}</span>{" "}
                  by round {fu.trajectory.earn.roundOfMin}. With an editable history it never leaves{" "}
                  <span className="font-mono">{fu.trajectory.noLedger.minTrust.toFixed(2)}</span>.
                </p>
                <div className="mt-8 flex flex-wrap items-center gap-2">
                  <Pill>Not protected</Pill>
                  <StatusTag status="failed" />
                </div>
                <p className="mt-6 text-pretty leading-relaxed text-muted">
                  Rare-disease F1 still falls from {fu.splits.s2.rareF1.earn.none.toFixed(3)} to{" "}
                  {fu.splits.s2.rareF1.earn.attacked.toFixed(3)} - as it does under FedAvg - and trust climbs back
                  to {fu.trajectory.earn.trustRound30.toFixed(2)} by round 30. EARN&apos;s trust only removes a bonus, so the
                  attacker keeps its normal share; and a running-average history slowly absorbs the new behaviour.
                  The pre-registered bar (a {fu.criterion.f1_margin} rare-F1 gain from the lock) was not met.
                </p>
              </Card>
            </Reveal>
          </div>
        </Section>
      )}

      <Section>
        <Reveal className="mt-14">
          <PillLink href="/status">Where this fits in the plan →</PillLink>
        </Reveal>
      </Section>
    </>
  );
}
