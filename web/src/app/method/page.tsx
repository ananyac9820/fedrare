import type { Metadata } from "next";
import Link from "next/link";
import { StudyChart } from "@/components/charts/Charts";
import { Pager } from "@/components/ui/Pager";
import { Card, PillLink } from "@/components/ui/Buttons";
import { HScroll } from "@/components/ui/HScroll";
import { PageHeader, Reveal, Section } from "@/components/ui/Motion";
import { GateBadge, Pill, StatusTag } from "@/components/ui/Status";
import type { ProjectStatus } from "@/lib/data";
import { coverage, dataset, g0a, gate, peerConfidence, study, studyRow } from "@/lib/data";

export const metadata: Metadata = { title: "The method" };

const STEPS: { title: string; symbol: string; body: string; statuses: ProjectStatus[]; warning?: string }[] = [
  {
    title: "Local training",
    symbol: "U[k]",
    body: "Each hospital trains on its own images and returns only a model update. Standard federated learning - runs today.",
    statuses: ["baseline"],
  },
  {
    title: "Evidence",
    symbol: "e(k,c)",
    body: "How strongly each hospital changed the model's row for each disease. No data shared, no self-reported counts.",
    statuses: ["failed"],
    warning: "Failed G0a, and its one retry",
  },
  {
    title: "Coverage check",
    symbol: "n(c) → p(c)",
    body: "Count how many hospitals can vouch for each disease, then blend peer checking with each hospital's own locked history.",
    statuses: ["in-progress"],
    warning: "Built; tested only on an oracle signal",
  },
  {
    title: "Trust update",
    symbol: "T(k,c)",
    body: "Slow up, fast down: +0.1 when the check agrees, halved when it doesn't. Full trust takes about ten consistent rounds.",
    statuses: ["in-progress"],
  },
  {
    title: "Aggregation",
    symbol: "w(k,c)",
    body: "Shared layers use FedAvg. Rare-disease rows add a trust × evidence bonus, capped at 50% per hospital.",
    statuses: ["baseline", "in-progress"],
  },
  {
    title: "Ledger commit",
    symbol: "block t",
    body: "Trust table, coverage and history hashes go on an append-only chain. The contract rejects any trust rise above the allowed step.",
    statuses: ["verified"],
  },
];

const HOW = ["Own locked history only", "Mostly history", "Half peers, half history", "Mostly peers", "Peers only"];

export default function MethodPage() {
  const n = dataset.centres.length;
  const rows = [1, 2, 3, 4, 5].map((k) => ({
    n: k,
    p: peerConfidence(k),
    how: HOW[k - 1],
    diseases: dataset.classes.filter((c) => (k === 5 ? coverage[c.id] >= 5 : coverage[c.id] === k)),
  }));
  const g0aGate = gate("G0a");
  const retry = g0a.retry?.definitions.find((d) => d.key.startsWith("retry"));
  const g2 = study.g2Oracle;
  const e = (split: "s1" | "s2", method: string, attack: "none" | "A1" | "A2" | "A3") =>
    studyRow(split, method, attack, "earn");
  const CHECK_LABEL: Record<string, string> = {
    "A1_within_0.05": "Under A1, rare F1 within 0.05 of no attack",
    "A2_within_0.05": "Under A2, rare F1 within 0.05 of no attack",
    "within_0.02_of_camp_a_reported": "No attack: within 0.02 of Camp A",
    above_fedavg: "No attack: above FedAvg",
    "specialist_trust_r15_ge_0.8": "Centre 2 trust ≥ 0.8 by round 15",
  };

  return (
    <>
      <PageHeader
        eyebrow="02 · The method"
        tags={
          <>
            <StatusTag status="in-progress" />
            <StatusTag status="failed" />
          </>
        }
        title="How EARN decides whose update counts."
      >
        EARN - Earned, Audited, Rare-class-aware aggregation - was our proposal. It is now built and
        unit-tested step by step, but its foundation - reading from each update which hospitals hold a
        disease - failed its check and its one retry. So EARN is not validated. We ran it once more on an
        oracle signal, clearly labelled, to see whether the rest of the mechanism would have worked.
      </PageHeader>

      <Section>
        <Reveal>
          <div className="flex flex-col gap-6 rounded-[28px] border border-failed/25 bg-failed-soft/60 p-8 md:flex-row md:items-center md:justify-between md:p-10">
            <div className="max-w-2xl">
              <div className="flex flex-wrap items-center gap-2">
                <StatusTag status="failed" />
                {g0aGate && <GateBadge state={g0aGate.state} />}
              </div>
              <p className="mt-5 text-pretty text-lg leading-relaxed text-ink">
                EARN reads each hospital&apos;s &quot;evidence&quot; for each disease. Our check of that
                signal failed: mean Spearman {g0a.statistic.toFixed(2)} against a bar of {g0a.threshold}
                {retry ? `, and ${retry.statistic.toFixed(2)} on the one allowed retry` : ""}. The project
                therefore switched to Fallback F1, the attack study.
              </p>
            </div>
            <PillLink href="/results" variant="outline">
              See the failed check
            </PillLink>
          </div>
        </Reveal>
      </Section>

      <Section
        label="One round"
        title="Six steps, every training round."
        intro="Every step is implemented in src/federated/earn.py and unit-tested. The evidence step is the one that failed."
      >
        <HScroll label="The six steps of one round">
          {STEPS.map((s, i) => (
            <article
              key={s.title}
              className={`flex w-[82vw] shrink-0 snap-start flex-col rounded-[28px] border bg-paper p-8 sm:w-[320px] md:p-10 ${
                s.statuses.includes("in-progress") ? "border-dashed border-progress/40" : "border-line"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="flex h-11 w-11 items-center justify-center rounded-full bg-sand font-mono text-sm text-ink">
                  {i + 1}
                </span>
                <span className="font-mono text-xs text-faint">{s.symbol}</span>
              </div>
              <h3 className="mt-8 font-display text-3xl font-medium tracking-tight text-ink">{s.title}</h3>
              <p className="mt-4 flex-1 text-pretty leading-relaxed text-muted">{s.body}</p>
              {s.warning && (
                <Link href="/results" className="mt-5 text-sm font-medium text-failed underline-offset-4 hover:underline">
                  {s.warning} →
                </Link>
              )}
              <div className="mt-8 flex flex-wrap gap-2">
                {s.statuses.map((t) => (
                  <StatusTag key={t} status={t} />
                ))}
              </div>
            </article>
          ))}
        </HScroll>
      </Section>

      <Section
        tone="sand"
        label="The core idea"
        title="Trust peers less when fewer peers can check."
        tags={<StatusTag status="in-progress" />}
        intro="Checking a claim needs someone able to check it. When many hospitals hold a disease, peers can contradict a false update. When few do, the only evidence left is the hospital's own track record. EARN blends the two smoothly, with no hard cutoff."
      >
        <div className="grid grid-cols-1 [&>*]:min-w-0 gap-8 lg:grid-cols-[1fr_1.35fr]">
          <Reveal>
            <Card className="h-full">
              <Pill>The blend</Pill>
              <pre className="mt-8 overflow-x-auto rounded-2xl bg-sand/70 p-6 font-mono text-[13px] leading-loose text-ink">
{`p(c)   = clip((n(c) - 1) / 4, 0, 1)

a(k,c) = p(c)     · agree_with_peers
       + (1-p(c)) · agree_with_own_history`}
              </pre>
              <p className="mt-6 text-sm leading-relaxed text-muted">
                n(c) is how many hospitals hold disease c. p(c) is how much the peer check can be
                trusted for it.
              </p>
            </Card>
          </Reveal>
          <Reveal delay={0.08}>
            <Card className="h-full overflow-x-auto">
              <table className="w-full min-w-[460px] text-left text-sm">
                <thead>
                  <tr className="border-b border-line font-mono text-[11px] uppercase tracking-[0.16em] text-faint">
                    <th className="pb-4 pr-4 font-medium">n(c)</th>
                    <th className="pb-4 pr-4 font-medium">p(c)</th>
                    <th className="pb-4 pr-4 font-medium">Checked by</th>
                    <th className="pb-4 font-medium">Our diseases</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.n} className="border-b border-line/70 align-top last:border-0">
                      <td className="py-5 pr-4 font-mono text-ink">{r.n === 5 ? `5-${n}` : r.n}</td>
                      <td className="py-5 pr-4 font-mono text-ink">{r.p.toFixed(2)}</td>
                      <td className="py-5 pr-4 text-muted">{r.how}</td>
                      <td className="py-5 leading-relaxed">
                        {r.diseases.length ? (
                          r.diseases.map((d, j) => (
                            <span key={d.id} className={d.rare ? "font-semibold text-accent" : "text-ink"}>
                              {d.name}
                              {j < r.diseases.length - 1 && <span className="text-faint">, </span>}
                            </span>
                          ))
                        ) : (
                          <span className="text-faint">{r.n === 1 ? "constructed split S2" : "-"}</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-6 flex flex-wrap items-center gap-2 text-xs leading-relaxed text-faint">
                <StatusTag status="verified" /> holder counts are measured (≥ {dataset.rareRule.holderMinImages}{" "}
                training images); rare diseases highlighted. The blend itself is the proposal.
              </p>
            </Card>
          </Reveal>
        </div>
      </Section>

      <Section label="Trust and weighting" title="Slow to earn, quick to lose." tags={<StatusTag status="in-progress" />}>
        <div className="grid grid-cols-1 [&>*]:min-w-0 gap-8 md:grid-cols-3">
          <Reveal>
            <Card dashed className="h-full">
              <Pill>Trust update</Pill>
              <p className="mt-8 font-display text-3xl text-ink">+0.1 or ×0.5</p>
              <p className="mt-4 text-pretty leading-relaxed text-muted">
                Agree with the check and trust rises by a small fixed step. Disagree once and it halves.
              </p>
            </Card>
          </Reveal>
          <Reveal delay={0.08}>
            <Card dashed className="h-full">
              <Pill>Weighting</Pill>
              <p className="mt-8 font-display text-3xl text-ink">≤ 50% per row</p>
              <p className="mt-4 text-pretty leading-relaxed text-muted">
                Each rare-disease row gets FedAvg weight plus trust × evidence share. No hospital may
                hold more than half of any row.
              </p>
            </Card>
          </Reveal>
          <Reveal delay={0.16}>
            <Card dashed className="h-full">
              <Pill>Safe default</Pill>
              <p className="mt-8 font-display text-3xl text-ink">T = 0 → FedAvg</p>
              <p className="mt-4 text-pretty leading-relaxed text-muted">
                With all trust at zero, EARN is clipped FedAvg - except that the 50% cap still applies,
                which trims centre 0&apos;s 53% share. The design doc&apos;s &quot;exactly FedAvg&quot; could
                not hold alongside the cap; we kept the cap and say so.
              </p>
            </Card>
          </Reveal>
        </div>
        <Reveal className="mt-14">
          <PillLink href="/ledger">Why the history must be locked →</PillLink>
        </Reveal>
      </Section>

      <Section
        label="Exploratory · oracle evidence"
        title="If the signal had worked, would EARN?"
        tags={
          <>
            <StatusTag status="in-progress" />
            <Pill tone="paper">Not a validated result</Pill>
          </>
        }
        intro="We replaced the failed signal with an oracle: every honest hospital's true image counts, which an attacker can fake exactly as it fakes a reported count. Everything else is EARN as designed, with three ablations and four sensitivity settings. Gate G2 cannot be evaluated after G0a failed; below is the same test on the oracle, reported as exploratory."
      >
        <div className="grid grid-cols-1 [&>*]:min-w-0 gap-8 md:grid-cols-2">
          {(["s1", "s2"] as const).map((sp, i) => (
            <Reveal key={sp} delay={i * 0.08}>
              <Card className="h-full">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h3 className="font-display text-2xl font-medium text-ink">{sp === "s1" ? "S1 · natural" : "S2 · specialist"}</h3>
                  <StatusTag status={g2[sp].passed ? "verified" : "failed"} />
                </div>
                <ul className="mt-8 space-y-4 text-sm">
                  {Object.entries(g2[sp].checks).map(([k, ok]) => (
                    <li key={k} className="flex items-baseline justify-between gap-4 border-b border-line/70 pb-3 last:border-0">
                      <span className="text-muted">{CHECK_LABEL[k] ?? k}</span>
                      <span className={`font-mono ${ok ? "text-verified" : "text-failed"}`}>{ok ? "✓" : "✕"}</span>
                    </li>
                  ))}
                </ul>
                <p className="mt-6 font-mono text-xs text-faint">
                  rare F1: no attack {g2[sp].values.earn_none.toFixed(3)} · A1 {g2[sp].values.earn_A1.toFixed(3)} · A2{" "}
                  {g2[sp].values.earn_A2.toFixed(3)} · FedAvg {g2[sp].values.fedavg_none.toFixed(3)}
                </p>
              </Card>
            </Reveal>
          ))}
        </div>
        <Reveal className="mt-8">
          <Card>
            <h3 className="font-display text-2xl font-medium text-ink">EARN and its ablations, specialist split S2</h3>
            <p className="mt-2 text-sm text-muted">Rare macro-F1, mean ± sd over 3 seeds. Oracle evidence throughout.</p>
            <div className="mt-8">
              <StudyChart rows={study.earn.s2} attacks={["none", "A1", "A2", "A3"]} metric="rare_macro_f1"
                methods={["fedavg", "camp_a_reported", "earn", "earn_no_blend", "earn_no_ramp", "earn_no_ledger", "earn_real_signal"]} />
            </div>
          </Card>
        </Reveal>
        <div className="mt-8 grid grid-cols-1 [&>*]:min-w-0 gap-8 md:grid-cols-2">
          <Reveal>
            <Card className="h-full">
              <Pill>Why it fails on S1</Pill>
              <p className="mt-8 text-pretty leading-relaxed text-muted">
                Honest hospitals&apos; rare-disease updates barely point the same way as each other in a round, so
                the peer check fails them too. Centre 2&apos;s trust at round 15 is only{" "}
                {e("s1", "earn", "none")?.specialist_trust_5_r15?.mean.toFixed(2)} (dermatofibroma) and{" "}
                {e("s1", "earn", "none")?.specialist_trust_6_r15?.mean.toFixed(2)} (vascular lesion) - EARN ends up
                close to clipped FedAvg.
              </p>
            </Card>
          </Reveal>
          <Reveal delay={0.08}>
            <Card className="h-full">
              <Pill>Why it fails on S2</Pill>
              <p className="mt-8 text-pretty leading-relaxed text-muted">
                With one holder, only the history check runs - and it rewards being <em>consistent</em>, not
                being right. The A1 attacker is consistent from round 1, earns full trust and captures{" "}
                {e("s2", "earn", "A1")?.attacker_ratio_6?.mean.toFixed(1)}x its fair weight. Peer-only checking
                (&quot;no blend&quot;) holds it to {e("s2", "earn_no_blend", "A1")?.attacker_ratio_6?.mean.toFixed(2)}x,
                but then zeroes the honest specialist&apos;s trust too.
              </p>
            </Card>
          </Reveal>
        </div>
      </Section>
      <Pager current="/method" />
    </>
  );
}
