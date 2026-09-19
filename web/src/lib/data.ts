/**
 * The only place the UI gets numbers from. Everything is read from web/data/*.json, which
 * `npm run sync-data` regenerates from the research repo's results/ folder. To swap in real
 * results, change the data files - not the components.
 */
import baselineJson from "../../data/baseline_results.json";
import datasetJson from "../../data/dataset.json";
import g0aJson from "../../data/g0a.json";
import ledgerJson from "../../data/ledger.json";
import missingJson from "../../data/missing.json";
import roadmapJson from "../../data/roadmap.json";

/** Where a number came from. "sample" is always shown with a SAMPLE DATA badge. */
export type DataStatus = "verified" | "sample";
/** What a thing is, in the project. */
export type ProjectStatus = "verified" | "baseline" | "in-progress";

export interface Meta {
  status: DataStatus;
  sources: string[];
  note?: string;
  producedBy?: string;
  wouldComeFrom?: string;
  syncedAt: string;
}

export interface DiseaseClass {
  id: number;
  name: string;
  sharePct: number;
  rare: boolean;
}

export interface Centre {
  id: number;
  trainCounts: number[];
  totalCounts: number[];
  trainImages: number;
  totalImages: number;
}

export interface Dataset {
  _meta: Meta;
  name: string;
  source: string;
  classes: DiseaseClass[];
  rareRule: {
    headRatioDivisor: number;
    cutoffPct: number;
    headClass: string;
    headSharePct: number;
    holderMinImages: number;
  };
  centres: Centre[];
}

export interface G0a {
  _meta: Meta;
  gate: string;
  passed: boolean;
  statistic: number;
  threshold: number;
  definition: string;
  classNames: string[];
  rareIds: number[];
  perClassSpearman: number[];
  evidence: number[][];
  trainCounts: number[][];
}

export interface BaselineRule {
  id: string;
  label: string;
  status: ProjectStatus;
  seeds: number;
  balancedAccuracy: number;
  rareMacroF1: number;
  rareF1: Record<string, number>;
}

export interface BaselineResults {
  _meta: Meta;
  split: string;
  rules: BaselineRule[];
}

export type GateState = "passed" | "failed" | "pending" | "not-run";

export interface Gate {
  id: string;
  name: string;
  when: string;
  condition: string;
  ifFails: string;
  state: GateState;
  result?: string;
  resultStatus?: DataStatus;
  note?: string;
}

export interface Week {
  week: number;
  dates: string;
  work: string;
  deliverable: string;
  progress?: { item: string; state: "done" | "failed" | "blocked" | "pending" }[];
}

export interface Roadmap {
  _meta: Meta;
  asOf: string;
  currentWeek: number;
  weeks: Week[];
  gates: Gate[];
}

export interface LedgerRound {
  round: number;
  trustTableHash: string;
  coverage: number[];
  historyHash: string;
  maxTrustRise: number;
  prevHash: string;
  blockHash: string;
}

export interface Ledger {
  _meta: Meta;
  rules: { maxTrustStep: number; halvingFactor: number };
  rounds: LedgerRound[];
}

export interface MissingItem {
  what: string;
  owner: string;
  track: string;
  branch: string;
  file: string;
  producedBy: string;
  blockedBy?: string;
}

export const dataset = datasetJson as Dataset;
export const g0a = g0aJson as G0a;
export const baselines = baselineJson as BaselineResults;
export const roadmap = roadmapJson as Roadmap;
export const ledger = ledgerJson as Ledger;
export const missing = missingJson as { _meta: Meta; items: MissingItem[] };

// ------------------------------------------------------------------ derived, never hard-coded

export const rareClasses = dataset.classes.filter((c) => c.rare);
export const totalImages = dataset.centres.reduce((a, c) => a + c.totalImages, 0);
export const totalTrainImages = dataset.centres.reduce((a, c) => a + c.trainImages, 0);

export interface CentreMismatch {
  id: number;
  totalImages: number;
  trainImages: number;
  rareTrainImages: number;
  /** share of all rare-class training images this centre holds */
  rareShare: number;
  /** FedAvg aggregation weight = share of all training images */
  fedavgWeight: number;
}

/** Per centre: how much of the rare-disease data it holds vs how much say FedAvg gives it. */
export const mismatch: CentreMismatch[] = (() => {
  const rareTrain = dataset.centres.map((c) =>
    rareClasses.reduce((a, cls) => a + c.trainCounts[cls.id], 0),
  );
  const rareTotal = rareTrain.reduce((a, b) => a + b, 0);
  return dataset.centres.map((c, k) => ({
    id: c.id,
    totalImages: c.totalImages,
    trainImages: c.trainImages,
    rareTrainImages: rareTrain[k],
    rareShare: rareTrain[k] / rareTotal,
    fedavgWeight: c.trainImages / totalTrainImages,
  }));
})();

/** The centre whose rare-data share most exceeds its FedAvg weight (the hero stat). */
export const specialist = mismatch.reduce((best, m) =>
  m.rareShare / m.fedavgWeight > best.rareShare / best.fedavgWeight ? m : best,
);
export const specialistRatio = specialist.rareShare / specialist.fedavgWeight;

/** Coverage n(c): centres holding at least holderMinImages training images of class c. */
export const coverage = dataset.classes.map(
  (cls) =>
    dataset.centres.filter((c) => c.trainCounts[cls.id] >= dataset.rareRule.holderMinImages).length,
);

/** EARN's proposed peer confidence p(c) = clip((n(c) - 1) / 4, 0, 1). Design, not a result. */
export const peerConfidence = (n: number) => Math.min(1, Math.max(0, (n - 1) / 4));

export const pct = (x: number, digits = 1) => `${(100 * x).toFixed(digits)}%`;
export const fmt = (n: number) => n.toLocaleString("en-US");
