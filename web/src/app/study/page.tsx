import type { Metadata } from "next";
import { StudyChart } from "@/components/charts/Charts";
import { Card, PillLink } from "@/components/ui/Buttons";
import { HScroll } from "@/components/ui/HScroll";
import { PageHeader, Reveal, Section } from "@/components/ui/Motion";
import { GateBadge, Pill, StatusTag } from "@/components/ui/Status";
import type { Attack, Split, StudyRow } from "@/lib/data";
import { ATTACKS, gate, pct, study, studyRow } from "@/lib/data";

export const metadata: Metadata = { title: "The attack study" };

const SPLITS: { id: Split; name: string; blurb: string }[] = [
  { id: "s1", name: "S1 · natural split", blurb: "Fed-ISIC2019's real six hospitals, unchanged. Each rare disease has 3-4 holders." },
  { id: "s2", name: "S2 · specialist split", blurb: "Constructed and disclosed: 90% of every other hospital's rare images moved to centre 2, so one hospital holds each rare disease." },
];
const F1_METHODS = ["fedavg", "fedavg_clipped", "krum", "multi_krum", "trimmed_mean", "coordinate_wise_median", "camp_a", "camp_a_reported"];
const ALL_ATTACKS: Attack[] = ["none", "A1", "A2", "A3"];
const RARE = [
  { id: 5, name: "Dermatofibroma", short: "DF" },
  { id: 6, name: "Vascular lesion", short: "VL" },
];

const f3 = (x: number) => x.toFixed(3);
const ms = (x?: { mean: number; sd: number }) => (x ? `${x.mean.toFixed(3)} ± ${x.sd.toFixed(3)}` : "-");
const tick = (ok: boolean) => (
  <span className={ok ? "text-verified" : "text-failed"} aria-label={ok ? "met" : "not met"}>{ok ? "✓" : "✕"}</span>
);

function row(split: Split, method: string, attack: Attack): StudyRow {
  const r = studyRow(split, method, attack);
  if (!r) throw new Error(`missing study row ${split}/${method}/${attack}`);
  return r;
}

export default function StudyPage() {
  const g1 = study.gateG1;
  const g1Gate = gate("G1");
  const fed = (s: Split) => row(s, "fedavg", "none");
  const campA = row("s1", "camp_a", "none");
  const reported = row("s1", "camp_a_reported", "none");
  const repA1s2 = row("s2", "camp_a_reported", "A1");
  const sleeper = row("s1", "fedavg", "A2");
  const sleeperClip = row("s1", "fedavg_clipped", "A2");
  const mkS2 = row("s2", "multi_krum", "none");
  const krumS1 = row("s1", "krum", "none");
  const bestS2Sleeper = F1_METHODS.map((m) => row("s2", m, "A2"))
    .reduce((b, r) => (r.rare_macro_f1.mean > b.rare_macro_f1.mean ? r : b));
  const specShare = (s: Split, c: string) => study.shares[s].rare_share[c][2];

  return (
    <>
      <PageHeader
        eyebrow="04 · The study"
        tags={
          <>
            <StatusTag status="verified" />
            {g1Gate && <GateBadge state={g1Gate.state} />}
          </>
        }
        title="What happens when a hospital lies about rare diseases."
      >
        {study.runs} federated runs on real hospital data: eight aggregation methods, three attacks and no
        attack, two splits, three seeds each. Every setting, attacker and pass mark was written down before
        the first run. This is now the project&apos;s main result - Fallback F1 in the design doc.
      </PageHeader>

      {/* G1 */}
      <Section
        label="Gate G1 · is the problem real?"
        title="The attack works - but not quietly enough to pass the gate."
        tags={<StatusTag status="failed" />}
        intro="G1 asked for three things at once under Camp A (weighting hospitals by the evidence in their updates): the attacker gains at least 2x its FedAvg weight on a rare disease, that disease's F1 falls by at least 0.15, and overall balanced accuracy falls by less than 0.03 - so nobody would notice."
      >
        <div className="grid grid-cols-1 [&>*]:min-w-0 gap-8 md:grid-cols-2">
          {SPLITS.map((s, i) => (
            <Reveal key={s.id} delay={i * 0.08}>
              <Card className="h-full">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h3 className="font-display text-2xl font-medium text-ink">{s.name}</h3>
                  <GateBadge state={g1[s.id].passed ? "passed" : "failed"} />
                </div>
                <table className="mt-8 w-full text-left text-sm">
                  <thead>
                    <tr className="border-b border-line font-mono text-[11px] uppercase tracking-[0.14em] text-faint">
                      <th className="pb-3 pr-3 font-medium">Disease</th>
                      <th className="pb-3 pr-3 font-medium">Weight ≥ 2x</th>
                      <th className="pb-3 font-medium">F1 drop ≥ 0.15</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(g1[s.id].per_class).map(([name, v]) => (
                      <tr key={name} className="border-b border-line/70 last:border-0">
                        <td className="py-4 pr-3 text-ink">{name}</td>
                        <td className="py-4 pr-3 font-mono">{v.ratio.toFixed(2)}x {tick(v.weight_condition)}</td>
                        <td className="py-4 font-mono">{v.f1_drop.toFixed(3)} {tick(v.f1_condition)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="mt-6 flex flex-wrap items-baseline justify-between gap-3 rounded-2xl bg-sand/70 px-5 py-4 text-sm">
                  <span className="text-muted">Balanced-accuracy drop (must be &lt; 0.03)</span>
                  <span className="font-mono text-ink">
                    {g1[s.id].balanced_accuracy_drop.toFixed(3)} {tick(g1[s.id].balanced_accuracy_condition)}
                  </span>
                </p>
              </Card>
            </Reveal>
          ))}
        </div>
        <Reveal className="mt-8">
          <div className="rounded-[28px] border border-line bg-paper p-8 md:p-10">
            <div className="flex flex-wrap items-center gap-2">
              <Pill>Decision, fixed in advance</Pill>
              <StatusTag status="verified" />
            </div>
            <p className="mt-6 max-w-3xl text-pretty text-lg leading-relaxed text-ink">{study.framing.decision}</p>
            <p className="mt-4 max-w-3xl text-pretty leading-relaxed text-muted">
              On S1 the attacker grabs about five times its fair share of the rare-disease rows, but three or
              four honest holders pull the model back. On S2 the attack does real damage, but it also costs{" "}
              {f3(g1.s2.balanced_accuracy_drop)} balanced accuracy - visible, so it fails the &quot;quiet&quot;
              condition. Both results are reported as they came out.
            </p>
          </div>
        </Reveal>
      </Section>

      {/* The attacks */}
      <Section tone="sand" label="The attacks" title="Three ways to cheat, fixed before any run.">
        <HScroll label="The attacks">
          {ATTACKS.map((a) => (
            <article key={a.id} className="flex w-[82vw] shrink-0 snap-start flex-col rounded-[28px] border border-line bg-paper p-8 sm:w-[320px] md:p-10">
              <span className="font-mono text-xs text-faint">{a.id}</span>
              <h3 className="mt-6 font-display text-2xl font-medium tracking-tight text-ink">{a.name}</h3>
              <p className="mt-2 text-sm text-accent">{a.who}</p>
              <p className="mt-4 flex-1 text-pretty leading-relaxed text-muted">{a.what}</p>
            </article>
          ))}
        </HScroll>
      </Section>

      {/* Results chart */}
      <Section
        label="Rare-disease F1"
        title="Every method, every attack."
        tags={<StatusTag status="verified" />}
        intro={`Mean F1 over the two rare diseases on the pooled test set after ${study.rounds} rounds; whiskers are ± one standard deviation over seeds ${study.seeds.join(", ")}.`}
      >
        <div className="space-y-8">
          {SPLITS.map((s, i) => (
            <Reveal key={s.id} delay={i * 0.06}>
              <Card>
                <h3 className="font-display text-2xl font-medium text-ink">{s.name}</h3>
                <p className="mt-2 text-sm text-muted">{s.blurb}</p>
                <div className="mt-8">
                  <StudyChart rows={study.f1[s.id]} methods={F1_METHODS} attacks={ALL_ATTACKS} metric="rare_macro_f1" />
                </div>
              </Card>
            </Reveal>
          ))}
        </div>
      </Section>

      {/* Findings */}
      <Section tone="sand" label="What we learned" title="Four findings, each measured.">
        <div className="grid grid-cols-1 [&>*]:min-w-0 gap-8 md:grid-cols-2">
          <Reveal>
            <Card className="h-full">
              <Pill>1 · Evidence weighting backfires</Pill>
              <p className="mt-8 text-pretty leading-relaxed text-muted">
                Camp A reading evidence from the updates gives the specialist <strong className="text-ink">less</strong>{" "}
                say than plain FedAvg ({pct(campA.specialist_weight_5.mean)} vs {pct(fed("s1").specialist_weight_5.mean)} on
                the dermatofibroma row) and lower rare F1 ({f3(campA.rare_macro_f1.mean)} vs {f3(fed("s1").rare_macro_f1.mean)}).
                It is the inverted G0a signal at work: the hospitals that never see a disease move its row the most.
              </p>
            </Card>
          </Reveal>
          <Reveal delay={0.06}>
            <Card className="h-full">
              <Pill>2 · Honest counts help, lying counts hurt</Pill>
              <p className="mt-8 text-pretty leading-relaxed text-muted">
                Weighting by reported class counts is the best method with no attack on S1 (rare F1{" "}
                {f3(reported.rare_macro_f1.mean)}). But a hospital that simply claims to be a big holder captures{" "}
                {repA1s2.attacker_ratio_6?.mean.toFixed(1)}x its fair weight on S2, and rare F1 falls to{" "}
                {f3(repA1s2.rare_macro_f1.mean)}.
              </p>
            </Card>
          </Reveal>
          <Reveal delay={0.12}>
            <Card className="h-full">
              <Pill>3 · The sleeper is the worst attack</Pill>
              <p className="mt-8 text-pretty leading-relaxed text-muted">
                A genuine holder that turns after 15 honest rounds cuts FedAvg&apos;s rare F1 on S1 from{" "}
                {f3(fed("s1").rare_macro_f1.mean)} to {f3(sleeper.rare_macro_f1.mean)}. Clipping updates to the median
                size recovers most of it ({f3(sleeperClip.rare_macro_f1.mean)}). On S2 the best method,{" "}
                {bestS2Sleeper.label}, reaches only {ms(bestS2Sleeper.rare_macro_f1)}; most fall close to zero.
              </p>
            </Card>
          </Reveal>
          <Reveal delay={0.18}>
            <Card className="h-full">
              <Pill>4 · Robust filters can erase a lone specialist</Pill>
              <p className="mt-8 text-pretty leading-relaxed text-muted">
                On the natural split Krum actually favours centre 2 (it is the one update Krum keeps in
                {" "}{pct(krumS1.specialist_weight_5.mean)} of rounds). But when one hospital holds a disease (S2),
                Multi-Krum gives it {pct(mkS2.specialist_weight_5.mean)} of the rare rows and rare F1 is{" "}
                {f3(mkS2.rare_macro_f1.mean)} - the &quot;odd one out&quot; is exactly the expert.
              </p>
            </Card>
          </Reveal>
        </div>
      </Section>

      {/* Fairness */}
      <Section
        label="Fairness to the specialist"
        title="How much say centre 2 gets on rare diseases."
        tags={<StatusTag status="verified" />}
        intro="Mean weight on the rare-disease rows over all rounds, no attack, compared with the share of that disease's images centre 2 actually holds."
      >
        <Reveal>
          <Card className="overflow-x-auto">
            <table className="w-full min-w-[640px] text-left text-sm">
              <thead>
                <tr className="border-b border-line font-mono text-[11px] uppercase tracking-[0.14em] text-faint">
                  <th className="pb-4 pr-4 font-medium">Method</th>
                  {SPLITS.map((s) => RARE.map((c) => (
                    <th key={`${s.id}${c.id}`} className="pb-4 pr-4 font-medium">{s.id.toUpperCase()} · {c.short}</th>
                  )))}
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-line/70 bg-sand/50">
                  <td className="py-4 pr-4 text-muted">Centre 2&apos;s share of the images</td>
                  {SPLITS.map((s) => RARE.map((c) => (
                    <td key={`${s.id}${c.id}`} className="py-4 pr-4 font-mono text-accent">{pct(specShare(s.id, c.name))}</td>
                  )))}
                </tr>
                {F1_METHODS.map((m) => (
                  <tr key={m} className="border-b border-line/70 last:border-0">
                    <td className="py-4 pr-4 text-ink">{row("s1", m, "none").label}</td>
                    {SPLITS.map((s) => RARE.map((c) => {
                      const r = row(s.id, m, "none");
                      const w = c.id === 5 ? r.specialist_weight_5.mean : r.specialist_weight_6.mean;
                      return <td key={`${s.id}${c.id}`} className="py-4 pr-4 font-mono text-ink">{pct(w)}</td>;
                    }))}
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </Reveal>
      </Section>

      {/* Full tables */}
      <Section tone="sand" label="All numbers" title="The full tables.">
        {SPLITS.map((s) => (
          <details key={s.id} className="mb-6 rounded-[28px] border border-line bg-paper p-6 md:p-8">
            <summary className="cursor-pointer font-display text-2xl text-ink">{s.name}</summary>
            <div className="mt-6 overflow-x-auto">
              <table className="w-full min-w-[820px] text-left text-sm">
                <thead>
                  <tr className="border-b border-line font-mono text-[11px] uppercase tracking-[0.14em] text-faint">
                    {["Method", "Attack", "Balanced acc", "Rare F1", "DF F1", "VL F1", "Rare → nevus", "Attacker weight"].map((h) => (
                      <th key={h} className="pb-3 pr-4 font-medium">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {study.f1[s.id].map((r) => (
                    <tr key={`${r.method}${r.attack}`} className="border-b border-line/60 last:border-0">
                      <td className="py-3 pr-4 text-ink">{r.attack === "none" ? r.label : ""}</td>
                      <td className="py-3 pr-4 font-mono text-muted">{r.attack}</td>
                      <td className="py-3 pr-4 font-mono">{ms(r.balanced_accuracy)}</td>
                      <td className="py-3 pr-4 font-mono">{ms(r.rare_macro_f1)}</td>
                      <td className="py-3 pr-4 font-mono">{f3(r.f1_5.mean)}</td>
                      <td className="py-3 pr-4 font-mono">{f3(r.f1_6.mean)}</td>
                      <td className="py-3 pr-4 font-mono">{pct(r.to_target_rare.mean)}</td>
                      <td className="py-3 font-mono">
                        {r.attacker_ratio_5 ? `${r.attacker_ratio_5.mean.toFixed(2)}x / ${r.attacker_ratio_6!.mean.toFixed(2)}x` : "-"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-4 text-xs text-faint">
                Attacker weight = its mean weight on the DF / VL rows while attacking, divided by its FedAvg weight.
                Rare → nevus = share of rare-disease test images classified as the attack&apos;s target.
              </p>
            </div>
          </details>
        ))}
        <Reveal className="mt-10 flex flex-wrap gap-3">
          <PillLink href="/method">EARN, tested on an oracle signal →</PillLink>
          <PillLink href="/status" variant="outline">Gates and plan</PillLink>
        </Reveal>
      </Section>
    </>
  );
}
