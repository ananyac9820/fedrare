import type { Metadata } from "next";
import { DermaconCurves } from "@/components/charts/Charts";
import { Card, PillLink } from "@/components/ui/Buttons";
import { Pager } from "@/components/ui/Pager";
import { PageHeader, Reveal, Section } from "@/components/ui/Motion";
import { Pill } from "@/components/ui/Status";
import { dermacon, dermaconHolders, dermaconSpecialist, fmt, pct, REPO } from "@/lib/data";

export const metadata: Metadata = {
  title: "Indian data",
  description:
    "DermaCon-IN as a second dataset: what transferred, what did not, and why.",
};

const d = dermacon.dataset;
const dg0 = dermacon.gates.dg0;
const dg2 = dermacon.gates.dg2;
const spec = dermaconSpecialist;

export default function DermaconPage() {
  const fst = d.fitzpatrick.filter((f) => f.type.startsWith("FST"));
  const fstTotal = fst.reduce((a, f) => a + f.count, 0);
  const trainTotal = dermacon.splits.d1.reduce((a, c) => a + c.images, 0);
  // FST 4-6 is the darker end; 1-2 is where ISIC 2019 sits. Both are stated, because
  // "100% are 3 to 6" is true here and reads like a rounding error.
  const share = (types: string[]) =>
    fst.filter((f) => types.includes(f.type)).reduce((a, f) => a + f.count, 0) / fstTotal;
  const darkShare = share(["FST 4", "FST 5", "FST 6"]);
  const lightCount = fst
    .filter((f) => ["FST 1", "FST 2"].includes(f.type))
    .reduce((a, f) => a + f.count, 0);

  return (
    <>
      <PageHeader eyebrow="Indian data" title="A second dataset, and what it exposed.">
        Everything else on this site is one dataset from six European and American hospitals.
        This page is {d.name}: {fmt(d.images)} clinical photographs from {fmt(d.patients)}{" "}
        patients in {d.origin}. It was added to test whether the method travels — and the
        answer, measured, is that half of it does.
      </PageHeader>

      {/* The caveat first, because every number below depends on it */}
      <Section>
        <Reveal>
          <Card className="border-ink/20">
            <Pill tone="accent">Read this before the numbers</Pill>
            <h2 className="mt-6 font-display text-3xl leading-tight text-ink md:text-4xl">
              The six centres on this page are constructed, not real.
            </h2>
            <p className="mt-6 max-w-3xl text-pretty leading-relaxed text-muted">
              {d.name} was collected across three hospitals, but the public release does not
              record which hospital took which photograph. So a real hospital split cannot be
              recovered from it. The six centres below are age bands — a split we built, and one
              we label as built everywhere it appears. Fed-ISIC2019 remains the only dataset in
              this project with real hospitals, and the two are reported side by side, never
              pooled.
            </p>
          </Card>
        </Reveal>
      </Section>

      {/* What the dataset is */}
      <Section
        tone="sand"
        label="The dataset"
        title="Indian skin, and a much longer tail."
        intro={`${d.subClasses} sub-classes and ${d.diseaseLabels} disease labels sit under the ${d.classes.length} classes used here.`}
      >
        <div className="grid grid-cols-1 [&>*]:min-w-0 gap-8 md:grid-cols-3">
          <Reveal>
            <Card className="h-full">
              <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-faint">
                Skin tone
              </p>
              <p className="mt-6 font-display text-6xl text-ink">{pct(darkShare, 0)}</p>
              <p className="mt-4 text-sm leading-relaxed text-muted">
                of its images are Fitzpatrick 4 to 6, and{" "}
                {lightCount === 0 ? "not one is Fitzpatrick 1 or 2" : `${lightCount} are 1 or 2`}.
                ISIC 2019, which every other number on this site rests on, sits at the opposite end.
                That is the reason a second dataset was worth the work rather than a box to tick.
              </p>
              <ul className="mt-6 space-y-1.5">
                {fst.map((f) => (
                  <li key={f.type} className="flex items-baseline justify-between gap-3">
                    <span className="text-xs text-muted">{f.type}</span>
                    <span className="font-mono text-xs text-faint">
                      {pct(f.count / fstTotal, 0)}
                    </span>
                  </li>
                ))}
              </ul>
            </Card>
          </Reveal>
          <Reveal delay={0.08}>
            <Card className="h-full">
              <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-faint">
                The rare pair
              </p>
              <ul className="mt-6 space-y-4">
                {d.rareClasses.map((c) => (
                  <li key={c.name}>
                    <p className="font-display text-3xl text-ink">{c.count}</p>
                    <p className="text-sm text-muted">{c.name}</p>
                  </li>
                ))}
              </ul>
              <p className="mt-6 text-sm leading-relaxed text-muted">
                Picked by the same rule used on Fed-ISIC2019 — below the head class&apos;s share
                divided by 25 — not chosen by hand.
              </p>
            </Card>
          </Reveal>
          <Reveal delay={0.16}>
            <Card className="h-full">
              <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-faint">
                Classes used
              </p>
              <ul className="mt-6 space-y-2.5">
                {d.classes.map((c) => (
                  <li key={c.name} className="flex items-baseline justify-between gap-3">
                    <span className={`truncate text-sm ${c.rare ? "font-semibold text-accent" : "text-muted"}`}>
                      {c.name}
                    </span>
                    <span className="font-mono text-xs text-faint">{fmt(c.count)}</span>
                  </li>
                ))}
              </ul>
              <p className="mt-6 text-xs leading-relaxed text-faint">
                {fmt(d.usable)} images used. The {d.images - d.usable} labelled
                &ldquo;{d.droppedClass}&rdquo; are dropped: it is not a disease.
              </p>
            </Card>
          </Reveal>
        </div>
      </Section>

      {/* The constructed split */}
      <Section
        label="The constructed split"
        title="Six age bands, one of them lopsided."
        intro="Rare-disease share divided by size share, on the images the models actually trained on. A band above 1.00 holds more rare cases than its size would suggest."
      >
        <Reveal>
          <Card>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[34rem] text-left text-sm">
                <thead className="font-mono text-[10px] uppercase tracking-[0.16em] text-faint">
                  <tr className="border-b border-line">
                    <th className="pb-3 pr-4 font-medium">Constructed centre</th>
                    <th className="pb-3 pr-4 font-medium">Training images</th>
                    <th className="pb-3 pr-4 font-medium">Rare cases</th>
                    <th className="pb-3 font-medium">Rare share ÷ size share</th>
                  </tr>
                </thead>
                <tbody>
                  {dermacon.splits.d1.map((c) => {
                    const isSpec = c.centre === spec.id;
                    return (
                      <tr key={c.centre} className="border-b border-line/60 last:border-0">
                        <td className={`py-3 pr-4 ${isSpec ? "font-semibold text-accent" : "text-ink"}`}>
                          age {c.band}
                          {isSpec && (
                            <span className="ml-2 font-mono text-[10px] uppercase tracking-[0.16em] text-accent">
                              the specialist
                            </span>
                          )}
                        </td>
                        <td className="py-3 pr-4 font-mono text-xs text-muted">{fmt(c.images)}</td>
                        <td className="py-3 pr-4 font-mono text-xs text-muted">{c.rare}</td>
                        <td className={`py-3 font-mono text-xs ${isSpec ? "text-accent" : "text-muted"}`}>
                          {c.ratio.toFixed(2)}×
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="mt-6 max-w-3xl text-sm leading-relaxed text-muted">
              The {spec.band?.band} band is the specialist: {pct((spec.band?.images ?? 0) / trainTotal, 0)}{" "}
              of the training images, {spec.band?.ratio.toFixed(2)}× its share of the rare cases. An
              older cohort carrying more neoplasms is clinically coherent rather than an artefact we
              arranged. The last band reaches a far higher ratio on only{" "}
              {dermacon.splits.d1.at(-1)?.images} training images, which is too few to mean
              anything — it is shown rather than quietly dropped.
            </p>
            <p className="mt-4 max-w-3xl text-sm leading-relaxed text-muted">
              A second split, {dermacon.splits.d0Note}, is run alongside as a control: with no
              specialist to find, a method that claims to find one is misfiring.
            </p>
          </Card>
        </Reveal>
      </Section>

      {/* The gates */}
      <Section tone="sand" label="The checks" title="One cleared, one did not.">
        <div className="grid grid-cols-1 [&>*]:min-w-0 gap-8 md:grid-cols-2">
          <Reveal>
            <Card className="h-full">
              <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-faint">
                DG0 · is the baseline worth studying?
              </p>
              <p className="mt-8 font-display text-6xl text-ink">{dg0.statistic.toFixed(3)}</p>
              <p className="mt-2 font-mono text-xs text-faint">
                balanced accuracy over {dg0.nClasses} classes · bar was {dg0.threshold.toFixed(2)}
              </p>
              <p className="mt-6 text-sm leading-relaxed text-muted">
                Cleared — by {dg0.margin.toFixed(3)}. That is thin enough to say out loud rather
                than round away. {dg0.rule}, {dg0.rounds} rounds, seed {dg0.seed}, with the
                threshold written down before the run.
              </p>
            </Card>
          </Reveal>
          <Reveal delay={0.08}>
            <Card className="h-full">
              <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-faint">
                DG2 · does EARN help here?
              </p>
              <p className="mt-8 font-display text-5xl text-ink">
                {dg2.conditions.filter((c) => c.met).length} of {dg2.conditions.length}
              </p>
              <p className="mt-2 font-mono text-xs text-faint">conditions met — so it did not clear</p>
              <ul className="mt-6 space-y-3">
                {dg2.conditions.map((c) => (
                  <li key={c.label} className="flex gap-3 text-sm leading-relaxed">
                    <span className={`mt-0.5 font-mono text-xs ${c.met ? "text-verified" : "text-failed"}`}>
                      {c.met ? "met" : "not met"}
                    </span>
                    <span className="text-muted">{c.label}</span>
                  </li>
                ))}
              </ul>
            </Card>
          </Reveal>
        </div>
      </Section>

      {/* The finding */}
      <Section
        label="What it exposed"
        title="The threshold was calibrated to the first dataset."
        intro="This is the result worth having, and it only appears when you run on something you did not design against."
      >
        <div className="grid grid-cols-1 [&>*]:min-w-0 gap-8 lg:grid-cols-[1.1fr_1fr]">
          <Reveal>
            <Card className="h-full">
              <p className="max-w-2xl leading-relaxed text-muted">
                EARN only listens to a centre about a disease once that centre holds at least 20
                images of it. On Fed-ISIC2019, four of six hospitals clear that for the rare
                classes. Here, {dermaconHolders === 0 ? "no constructed centre does" : `only ${dermaconHolders} of six does`} — the
                specialist holds 11 of one rare class and 8 of the other, so it counts as a holder
                of neither. With no holders, trust and evidence never engage and EARN quietly
                becomes capped FedAvg.
              </p>
              <p className="mt-5 max-w-2xl leading-relaxed text-muted">
                An absolute image count is the wrong unit when the rarest class has{" "}
                {d.rareClasses[0]?.count} images instead of hundreds. Swapping it for a threshold
                relative to the round&apos;s own evidence restores the intended behaviour: the
                specialist goes from {spec.earn?.specialistWeight?.toFixed(3)} of the rare rows —
                below its {spec.earn?.sizeShare?.toFixed(3)} size share — to{" "}
                {spec.earnTauMean?.specialistWeight?.toFixed(3)}.
              </p>
              <p className="mt-5 max-w-2xl text-sm leading-relaxed text-faint">
                That change was made after seeing the failure and knowing its cause, so it is
                post-hoc, and it does not change the DG2 verdict. It has since been tested on
                Fed-ISIC2019 under a pre-registered adversarial check — and rejected. A threshold
                set at &ldquo;above this round&apos;s average evidence&rdquo; is one a liar can
                clear by lying, because the claim is the evidence: under the sudden-expert attack,
                a centre holding no rare images at all went from 4.8% of the rare rows to 20.7%.
                EARN keeps the absolute threshold, and this page&apos;s finding stands as one about
                units rather than a defect with a known fix.
              </p>
            </Card>
          </Reveal>
          <Reveal delay={0.08}>
            <Card className="h-full">
              <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-faint">
                Rare-class F1, mean of {dg2.seeds.length} seeds
              </p>
              <DermaconCurves curves={dermacon.curves} methods={dermacon.methods.d1} />
            </Card>
          </Reveal>
        </div>

        <Reveal className="mt-8">
          <Card>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[40rem] text-left text-sm">
                <thead className="font-mono text-[10px] uppercase tracking-[0.16em] text-faint">
                  <tr className="border-b border-line">
                    <th className="pb-3 pr-4 font-medium">Method</th>
                    <th className="pb-3 pr-4 font-medium">Balanced accuracy</th>
                    <th className="pb-3 pr-4 font-medium">Rare-class F1</th>
                    <th className="pb-3 font-medium">Specialist&apos;s rare-row weight</th>
                  </tr>
                </thead>
                <tbody>
                  {dermacon.methods.d1.map((m) => (
                    <tr key={m.id} className="border-b border-line/60 last:border-0">
                      <td className="py-3 pr-4 text-ink">
                        {m.label}
                        {!m.preRegistered && (
                          <span className="ml-2 font-mono text-[10px] uppercase tracking-[0.16em] text-faint">
                            post-hoc
                          </span>
                        )}
                      </td>
                      <td className="py-3 pr-4 font-mono text-xs text-muted">
                        {m.balancedAccuracy.toFixed(4)}
                        {m.balancedAccuracySd !== null && ` ± ${m.balancedAccuracySd.toFixed(4)}`}
                      </td>
                      <td className="py-3 pr-4 font-mono text-xs text-muted">
                        {m.rareMacroF1.toFixed(4)}
                        {m.rareMacroF1Sd !== null && ` ± ${m.rareMacroF1Sd.toFixed(4)}`}
                      </td>
                      <td className="py-3 font-mono text-xs text-muted">
                        {m.specialistWeight !== null ? m.specialistWeight.toFixed(3) : "—"}
                        {m.sizeShare !== null && (
                          <span className="text-faint"> / {m.sizeShare.toFixed(3)} size</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-6 max-w-3xl text-sm leading-relaxed text-muted">
              Two claims have to be kept apart here. The weighting change is solid: ±
              {spec.earnTauMean?.specialistWeightSd?.toFixed(3)} across seeds, because the weight
              comes from the evidence and the trust schedule rather than from training noise. The
              accuracy gain is not: {spec.earnTauMean?.rareMacroF1.toFixed(4)} against FedAvg&apos;s{" "}
              {spec.fedavg?.rareMacroF1.toFixed(4)} is a gap smaller than either standard
              deviation. On three seeds that cannot be separated from noise, and we do not claim it.
            </p>
          </Card>
        </Reveal>
      </Section>

      {/* Provenance */}
      <Section tone="sand" label="Provenance" title="Where this data came from.">
        <Reveal>
          <Card>
            <p className="leading-relaxed text-muted">
              {d.name} — {d.citation}. {d.origin}. Licensed {d.licence}: non-commercial,
              share-alike, attribution required. Images are smartphone clinical photographs, not
              dermoscopy, so the model on the live demo does not apply to them.
            </p>
            <p className="mt-4 font-mono text-xs text-faint">doi:{d.doi}</p>
            <div className="mt-8 flex flex-wrap gap-3">
              <PillLink href={`https://doi.org/${d.doi}`} external>
                The dataset ↗
              </PillLink>
              <PillLink href={`${REPO}/blob/main/docs/DERMACON_PLAN.md`} variant="outline" external>
                What we fixed before running ↗
              </PillLink>
              <PillLink href={`${REPO}/blob/main/docs/DEVIATIONS.md`} variant="outline" external>
                Results log (R6, R7) ↗
              </PillLink>
            </div>
          </Card>
        </Reveal>
      </Section>

      <Pager current="/dermacon" />
    </>
  );
}
