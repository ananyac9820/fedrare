/**
 * The only place the UI gets numbers from. Everything is read from web/data/*.json, which
 * `npm run sync-data` regenerates from the research repo's results/ folder. To swap in real
 * results, change the data files - not the components.
 */
import baselineJson from "../../data/baseline_results.json";
import datasetJson from "../../data/dataset.json";
import g0aJson from "../../data/g0a.json";
import g0bJson from "../../data/g0b.json";
import ledgerJson from "../../data/ledger.json";
import missingJson from "../../data/missing.json";
import roadmapJson from "../../data/roadmap.json";
import studyJson from "../../data/study.json";

/** Where a number came from. "sample" is always shown with a SAMPLE DATA badge. */
export type DataStatus = "verified" | "sample";
/** What a thing is, in the project. */
export type ProjectStatus = "verified" | "baseline" | "in-progress" | "failed" | "not-run" | "blocked";

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
  retry?: {
    source: string;
    verdict: string;
    passed: boolean;
    next: string;
    definitions: { key: string; role: string; statistic: number; holderAuroc: number; passed: boolean;
      perClassSpearman: number[] }[];
  };
}

export interface G0bRun {
  rounds: number;
  variant: string;
  mean: number;
  sd: number;
  rareMacroF1: number;
  perSeed: { seed: number; value: number }[];
  curve: { round: number; balancedAccuracy: number }[];
}

export interface G0b {
  _meta: Meta;
  gate: string;
  passed: boolean;
  statistic: number;
  threshold: number;
  definition: string;
  perSeed: { seed: number; value: number }[];
  config: { rounds: number; local_steps: number; lr: number; class_balanced_loss: boolean };
  retry?: {
    source: string;
    passed: boolean;
    statistic: number;
    threshold: number;
    definition: string;
    retry: G0bRun;
    amended: { frozen: G0bRun; ft4: G0bRun };
    fineTuningMinutes: number;
  };
}

export interface SeedResult {
  seed: number;
  balancedAccuracy: number;
  accuracy: number;
  rareMacroF1: number;
  rareF1: Record<string, number>;
}

export interface BaselineRule {
  id: string;
  label: string;
  status: ProjectStatus;
  seeds: number;
  rounds?: number;
  balancedAccuracy: number;
  accuracy?: number;
  rareMacroF1: number;
  rareF1: Record<string, number>;
  perSeed?: SeedResult[];
  curve?: { round: number; balancedAccuracy: number; rareMacroF1: number }[];
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

export type ProgressState = "done" | "failed" | "blocked" | "pending" | "in-progress";

export interface Week {
  week: number;
  dates: string;
  work: string;
  deliverable: string;
  progress?: { item: string; state: ProgressState }[];
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
  chainBlockHash?: string;
  gas?: number;
  trustBps?: number[];
}

export interface Stat { n: number; mean: number; median: number; min: number; max: number }

export interface Ledger {
  _meta: Meta;
  rules: { maxTrustStep: number; halvingFactor: number };
  rounds: LedgerRound[];
  onChain?: {
    network: string;
    solc: string;
    runs: number;
    roundsCommitted: number;
    gasPerRound: Stat;
    latencyMsPerRound: Stat;
    deployGas: number;
    allFinalTrustMatch: boolean;
    allHistoryVerified: boolean;
    allTamperRejected: boolean;
    tamperExample?: { client: number; class: number; from_bps: number; to_bps: number };
    tamperError?: string;
  };
}

// ------------------------------------------------------------------ the attack study (F1)

export interface MS { mean: number; sd: number }
export type Split = "s1" | "s2";
export type Attack = "none" | "A1" | "A2" | "A3";

export interface StudyRow {
  method: string;
  label: string;
  attack: Attack;
  balanced_accuracy: MS;
  rare_macro_f1: MS;
  macro_f1: MS;
  f1_5: MS;
  f1_6: MS;
  to_target_rare: MS;
  specialist_weight_5: MS;
  specialist_weight_6: MS;
  attacker_ratio_5?: MS;
  attacker_ratio_6?: MS;
  specialist_trust_5_r15?: MS;
  specialist_trust_6_r15?: MS;
  specialist_trust_5_final?: MS;
  specialist_trust_6_final?: MS;
  attacker_trust_5_max?: MS;
  attacker_trust_6_max?: MS;
}

export interface G1Class {
  attacker_weight: number;
  attacker_fedavg_weight: number;
  ratio: number;
  f1_no_attack: number;
  f1_under_A1: number;
  f1_drop: number;
  weight_condition: boolean;
  f1_condition: boolean;
}

export interface G1Split {
  passed: boolean;
  balanced_accuracy_drop: number;
  balanced_accuracy_condition: boolean;
  per_class: Record<string, G1Class>;
}

export interface Study {
  _meta: Meta;
  runs: number;
  rounds: number;
  seeds: number[];
  shares: Record<Split, { size_share: number[]; rare_share: Record<string, number[]>;
    coverage_ge20: Record<string, number>; counts: number[][] }>;
  gateG1: Record<Split, G1Split>;
  gateG1Reported: Record<Split, G1Split>;
  framing: { s1: boolean; s2: boolean; decision: string };
  f1: Record<Split, StudyRow[]>;
  earn: Record<Split, StudyRow[]>;
  g2Oracle: Record<Split, { passed: boolean; checks: Record<string, boolean>; values: Record<string, number> }>;
  overheadMs: Record<string, { mean: number; sd: number; n: number }>;
  ledgerFollowup?: {
    entry: string;
    attacker: number;
    seeds: number[];
    criterion: { f1_margin: number; weight_factor: number };
    splits: Record<Split, { lockMatters: boolean; rareF1Gain: number;
      rareF1: Record<string, { none: number; attacked: number }> }>;
    trajectory: { split: Split; earn: TrustTrajectory; noLedger: TrustTrajectory };
  };
}

export interface TrustTrajectory {
  trust: number[];
  rareF1: number[];
  minTrust: number;
  roundOfMin: number;
  trustRound30: number;
}

export interface PendingItem {
  what: string;
  status: "not-run" | "blocked" | "in-progress";
  owner: string;
  track: string;
  branch: string;
  file: string;
  producedBy: string;
  blockedBy?: string;
}

export const dataset = datasetJson as Dataset;
export const g0a = g0aJson as G0a;
export const g0b = g0bJson as G0b;
export const baselines = baselineJson as BaselineResults;
export const roadmap = roadmapJson as Roadmap;
export const ledger = ledgerJson as Ledger;
export const pending = missingJson as { _meta: Meta; items: PendingItem[] };
export const study = studyJson as unknown as Study;

export const ATTACKS: { id: Attack; name: string; who: string; what: string }[] = [
  { id: "none", name: "No attack", who: "-", what: "Every hospital trains honestly." },
  { id: "A1", name: "A1 · Sudden expert", who: "Centre 4 (holds no rare images)",
    what: "Flips rare labels to nevus and inflates its rare-disease rows to look like a big holder, every round." },
  { id: "A2", name: "A2 · Sleeper", who: "Centre 1 (a real holder on S1)",
    what: "Honest for 15 rounds to build a record, then behaves exactly like A1." },
  { id: "A3", name: "A3 · Scaling", who: "Centre 4",
    what: "Flips rare labels and multiplies its whole update by 10 to dominate the average." },
];

/** One study row for (split, method, attack). */
export const studyRow = (split: Split, method: string, attack: Attack, table: "f1" | "earn" = "f1") =>
  study[table][split].find((r) => r.method === method && r.attack === attack);

// ------------------------------------------------------------------ small stats helpers

export const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
/** Population standard deviation over seeds (matches the research scripts). */
export const sd = (xs: number[]) => {
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) ** 2)));
};

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

export const rareTrainTotal = mismatch.reduce((a, m) => a + m.rareTrainImages, 0);

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

/** Centres holding no image of any rare class. */
export const zeroRareCentres = dataset.centres
  .filter((c) => rareClasses.every((cls) => c.totalCounts[cls.id] === 0))
  .map((c) => c.id);

/** EARN's proposed peer confidence p(c) = clip((n(c) - 1) / 4, 0, 1). Design, not a result. */
export const peerConfidence = (n: number) => Math.min(1, Math.max(0, (n - 1) / 4));

/** For each class, the centre with the highest G0a evidence, and whether it holds zero images. */
export const g0aTopCentres = g0a.classNames.map((name, c) => {
  const col = g0a.evidence.map((row) => row[c]);
  const top = col.indexOf(Math.max(...col));
  return { name, centre: top, images: g0a.trainCounts[top][c], rare: g0a.rareIds.includes(c) };
});

export const failedGates = roadmap.gates.filter((g) => g.state === "failed");
export const gate = (id: string) => roadmap.gates.find((g) => g.id === id);

export const pct = (x: number, digits = 1) => `${(100 * x).toFixed(digits)}%`;
export const fmt = (n: number) => n.toLocaleString("en-US");

export const REPO = "https://github.com/ananyac9820/fedrare";

export const SHORT: Record<string, string> = {
  Melanoma: "MEL",
  "Melanocytic nevus": "NV",
  "Basal cell carcinoma": "BCC",
  "Actinic keratosis": "AK",
  "Benign keratosis": "BKL",
  Dermatofibroma: "DF",
  "Vascular lesion": "VASC",
  "Squamous cell carcinoma": "SCC",
};
