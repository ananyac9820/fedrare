import type { Metadata } from "next";
import { LedgerExplorer } from "@/components/ledger/LedgerExplorer";
import { Card, PillLink } from "@/components/ui/Buttons";
import { HScroll } from "@/components/ui/HScroll";
import { PageHeader, Reveal, Section } from "@/components/ui/Motion";
import { DataBadge, Pill, SampleBadge, StatusTag } from "@/components/ui/Status";
import { SHORT, dataset, ledger, pending } from "@/lib/data";

export const metadata: Metadata = { title: "The ledger" };

const COMMITTED = [
  { title: "Round number", body: "Which training round this block records." },
  { title: "Trust table T(k,c)", body: "A hash of every hospital's trust for every disease after this round." },
  { title: "Coverage n(c)", body: "How many hospitals showed real evidence for each disease this round." },
  { title: "History hashes", body: "One per hospital - the record its own-history check reads from." },
  { title: "Previous block hash", body: "The link that makes rewriting any earlier round detectable." },
];

export default function LedgerPage() {
  const cost = pending.items.filter((p) => p.owner === "Aditya");

  return (
    <>
      <PageHeader
        eyebrow="04 · The ledger"
        tags={
          <>
            <StatusTag status="in-progress" />
            <DataBadge status={ledger._meta.status} title={ledger._meta.note} />
          </>
        }
        title="A record nobody can quietly rewrite."
      >
        Every round, EARN would commit its trust table, coverage and each hospital&apos;s history hash to
        an append-only chain. Each block stores the hash of the one before it, so changing any past
        round breaks every link after it.
      </PageHeader>

      <Section
        label="Try it"
        title="Tamper with a past round."
        tags={<SampleBadge title={ledger._meta.note} />}
        intro="The hash links below are real SHA-256; the block contents are sample data until EARN and the contract exist."
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
                <SampleBadge title="Illustration of the contract rule, not a recorded event" />
              </div>
              <p className="mt-8 text-pretty leading-relaxed text-muted">
                The contract rejects any trust value rising faster than the fixed step, so even whoever
                runs aggregation can&apos;t quietly boost a hospital.
              </p>
              <div className="mt-6 rounded-2xl bg-sand/70 p-5 font-mono text-xs leading-relaxed">
                <p className="text-muted">submit T(centre 5, vascular) 0.30 → 0.60</p>
                <p className="mt-1 text-failed">✕ rejected: rise 0.30 &gt; max step {ledger.rules.maxTrustStep.toFixed(2)}</p>
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

      <Section tone="sand" label="Cost" title="Gas and latency: not measured yet." tags={<StatusTag status="not-run" />}>
        <div className="grid grid-cols-1 [&>*]:min-w-0 gap-8 md:grid-cols-2">
          {cost.map((p, i) => (
            <Reveal key={p.what} delay={i * 0.08}>
              <Card dashed className="h-full">
                <StatusTag status={p.status} />
                <h3 className="mt-6 font-display text-2xl font-medium text-ink">{p.what}</h3>
                <p className="mt-3 text-sm text-muted">{p.producedBy}</p>
                <div className="mt-8 flex flex-wrap items-center gap-2 border-t border-line pt-6">
                  <Pill tone="paper">{p.owner}</Pill>
                  <span className="font-mono text-xs text-faint">{p.branch}</span>
                </div>
              </Card>
            </Reveal>
          ))}
        </div>
        <Reveal className="mt-14">
          <PillLink href="/status">Where this fits in the plan →</PillLink>
        </Reveal>
      </Section>
    </>
  );
}
