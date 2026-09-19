import { Footer, Nav } from "@/components/Chrome";
import { Hero } from "@/components/sections/Hero";
import { HowItWorks } from "@/components/sections/HowItWorks";
import { LedgerSection } from "@/components/sections/LedgerSection";
import { Problem } from "@/components/sections/Problem";
import { Results } from "@/components/sections/Results";
import { Roadmap } from "@/components/sections/Roadmap";
import {
  baselines,
  coverage,
  dataset,
  fmt,
  g0a,
  ledger,
  mismatch,
  missing,
  rareClasses,
  roadmap,
  specialist,
  specialistRatio,
  totalImages,
} from "@/lib/data";

const REPO = "https://github.com/ananyac9820/fedrare";
const SHORT: Record<string, string> = {
  Melanoma: "MEL",
  "Melanocytic nevus": "NV",
  "Basal cell carcinoma": "BCC",
  "Actinic keratosis": "AK",
  "Benign keratosis": "BKL",
  Dermatofibroma: "DF",
  "Vascular lesion": "VASC",
  "Squamous cell carcinoma": "SCC",
};

export default function Home() {
  const rareIds = rareClasses.map((c) => c.id);
  const zeroRareCentres = dataset.centres
    .filter((c) => rareIds.every((id) => c.totalCounts[id] === 0))
    .map((c) => c.id);

  return (
    <>
      <Nav />
      <main>
        <Hero
          nodes={dataset.centres.map((c) => ({
            id: c.id,
            images: c.totalImages,
            highlight: c.id === specialist.id,
          }))}
          stats={[
            { value: fmt(totalImages), label: "real dermoscopy images" },
            { value: String(dataset.centres.length), label: "real hospitals" },
            { value: String(dataset.classes.length), label: "skin-disease classes" },
            { value: String(rareClasses.length), label: "rare classes, about 1% each" },
          ]}
        />
        <Problem
          mismatch={mismatch}
          specialistId={specialist.id}
          ratio={specialistRatio}
          rare={rareClasses.map((c) => ({ name: c.name, sharePct: c.sharePct, holders: coverage[c.id] }))}
          rule={dataset.rareRule}
          coverage={{ zeroRareCentres }}
          sources={dataset._meta.sources}
        />
        <HowItWorks
          blend={dataset.classes.map((c) => ({ name: c.name, holders: coverage[c.id], rare: c.rare }))}
        />
        <Results baselines={baselines} g0a={g0a} />
        <LedgerSection ledger={ledger} classShort={dataset.classes.map((c) => SHORT[c.name] ?? c.name)} />
        <Roadmap roadmap={roadmap} missing={missing.items} />
      </main>
      <Footer syncedAt={dataset._meta.syncedAt} repo={REPO} />
    </>
  );
}
