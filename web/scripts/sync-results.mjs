#!/usr/bin/env node
/**
 * Regenerates web/data/*.json from the research repo's results/ folder.
 *
 * The UI only ever reads web/data/. Every file there carries _meta.status:
 *   "verified" - computed from a real results/ file (named in _meta.sources)
 *   "sample"   - placeholder, shown in the UI with a SAMPLE DATA badge
 *
 * A file is only upgraded from sample to verified when its source exists in results/.
 * Missing sources leave the current file untouched, so this is safe to run any time:
 *
 *     npm run sync-data
 *
 * results/ is gitignored in the research repo, so the generated JSON is committed here -
 * the site works from a fresh clone, and a re-sync after new results is a data-only diff.
 */
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { load as loadYaml } from "js-yaml";

const WEB = path.resolve(import.meta.dirname, "..");
const RESULTS = path.resolve(WEB, "..", "results");
const DATA = path.join(WEB, "data");
const today = new Date().toISOString().slice(0, 10);

const exists = (f) => fs.existsSync(path.join(RESULTS, f));
const readText = (f) => fs.readFileSync(path.join(RESULTS, f), "utf8");
const readJson = (f) => JSON.parse(readText(f));
const readData = (f) =>
  fs.existsSync(path.join(DATA, f)) ? JSON.parse(fs.readFileSync(path.join(DATA, f), "utf8")) : null;
const writeData = (f, obj) => {
  fs.writeFileSync(path.join(DATA, f), JSON.stringify(obj, null, 2) + "\n");
  console.log(`  wrote data/${f} (${obj._meta.status})`);
};
const sha256 = (s) => crypto.createHash("sha256").update(s).digest("hex");

function parseCsv(text) {
  const [header, ...rows] = text.trim().split(/\r?\n/).map((l) => l.split(","));
  return rows.map((r) => Object.fromEntries(header.map((h, i) => [h, r[i]])));
}

fs.mkdirSync(DATA, { recursive: true });
console.log(`Syncing from ${RESULTS}`);

// ------------------------------------------------------------------ dataset.json
if (exists("class_distribution.csv") && exists("rare_classes.yaml") && exists("gate_g0a.json")) {
  const dist = parseCsv(readText("class_distribution.csv"));
  const rare = loadYaml(readText("rare_classes.yaml"));
  const g0a = readJson("gate_g0a.json");
  const names = Object.keys(dist[0]).filter((k) => k !== "");
  const shares = rare.class_share_pct;
  const headShare = Math.max(...Object.values(shares));

  writeData("dataset.json", {
    _meta: {
      status: "verified",
      sources: ["results/class_distribution.csv", "results/rare_classes.yaml",
        "results/gate_g0a.json (train_counts)"],
      producedBy: "scripts/02_explore_data.py, scripts/06_gate_g0a.py",
      syncedAt: today,
    },
    name: "Fed-ISIC2019",
    source: "huggingface.co/datasets/flwrlabs/fed-isic2019",
    classes: names.map((name, id) => ({
      id, name, sharePct: shares[name], rare: rare.rare_class_ids.includes(id),
    })),
    rareRule: {
      headRatioDivisor: rare.head_ratio_divisor,
      cutoffPct: rare.rare_cutoff_pct,
      headClass: names.find((n) => shares[n] === headShare),
      headSharePct: headShare,
      holderMinImages: rare.specialist_min_images,
    },
    centres: dist.map((row, k) => {
      const total = names.map((n) => Number(row[n]));
      const train = g0a.train_counts[k];
      return {
        id: k,
        trainCounts: train,
        totalCounts: total,
        trainImages: train.reduce((a, b) => a + b, 0),
        totalImages: total.reduce((a, b) => a + b, 0),
      };
    }),
  });
} else {
  console.log("  dataset.json: sources missing, left unchanged");
}

// ------------------------------------------------------------------ g0a.json
if (exists("gate_g0a.json")) {
  const g = readJson("gate_g0a.json");
  writeData("g0a.json", {
    _meta: { status: "verified", sources: ["results/gate_g0a.json"],
      producedBy: "scripts/06_gate_g0a.py (feature/pipeline)", syncedAt: today },
    gate: "G0a", passed: g.passed, statistic: g.statistic, threshold: g.threshold,
    definition: g.definition, classNames: g.class_names, rareIds: g.rare_ids,
    perClassSpearman: g.per_class_spearman, evidence: g.evidence, trainCounts: g.train_counts,
  });
}

// ------------------------------------------------------------------ baseline_results.json
if (exists("tier_a_s1_metrics.csv")) {
  const rows = parseCsv(readText("tier_a_s1_metrics.csv"));
  const lastRound = Math.max(...rows.map((r) => Number(r.round)));
  const final = rows.filter((r) => Number(r.round) === lastRound);
  const mean = (rs, key) => rs.reduce((a, r) => a + Number(r[key]), 0) / rs.length;
  const labels = { fedavg: "FedAvg", camp_a: "Camp A (evidence weighting)" };
  const rules = [...new Set(final.map((r) => r.rule))].map((id) => {
    const rs = final.filter((r) => r.rule === id);
    return {
      id, label: labels[id] ?? id, status: "baseline", seeds: rs.length,
      balancedAccuracy: mean(rs, "balanced_accuracy"),
      rareMacroF1: mean(rs, "rare_macro_f1"),
      rareF1: { Dermatofibroma: mean(rs, "f1_5_Dermatofibroma"),
        "Vascular lesion": mean(rs, "f1_6_Vascular_lesion") },
    };
  });
  writeData("baseline_results.json", {
    _meta: { status: "verified", sources: ["results/tier_a_s1_metrics.csv"],
      producedBy: "scripts/07_tier_a_s1_baselines.py", syncedAt: today,
      note: `Tier A, natural split S1, final round (${lastRound}), mean over seeds.` },
    split: "S1", rules,
  });
} else {
  console.log("  baseline_results.json: results/tier_a_s1_metrics.csv missing, sample kept");
}

// ------------------------------------------------------------------ roadmap.json gates
const roadmap = readData("roadmap.json");
if (roadmap) {
  const setGate = (id, patch) => {
    const gate = roadmap.gates.find((g) => g.id === id);
    if (gate) Object.assign(gate, patch);
  };
  if (exists("gate_g0a.json")) {
    const g = readJson("gate_g0a.json");
    setGate("G0a", { state: g.passed ? "passed" : "failed",
      result: `Mean Spearman ${g.statistic.toFixed(2)} (needs >= ${g.threshold})`,
      resultStatus: "verified" });
  }
  if (exists("gate_g0b.json")) {
    const g = readJson("gate_g0b.json");
    setGate("G0b", { state: g.passed ? "passed" : "failed",
      result: `FedAvg balanced accuracy ${g.statistic.toFixed(3)} (needs >= ${g.threshold})`,
      resultStatus: "verified" });
  }
  roadmap._meta.syncedAt = today;
  writeData("roadmap.json", roadmap);
}

// ------------------------------------------------------------------ ledger.json
const ledger = readData("ledger.json");
if (exists("ledger_rounds.json")) {
  const real = readJson("ledger_rounds.json");
  writeData("ledger.json", { ...ledger, _meta: { status: "verified",
    sources: ["results/ledger_rounds.json"], syncedAt: today }, rounds: real.rounds });
} else if (!ledger || ledger._meta.status === "sample") {
  // Deterministic sample chain: real SHA-256 linking, made-up contents.
  const coverage = [6, 6, 3, 3, 5, 4, 3, 2];
  let prev = "0".repeat(64);
  const rounds = Array.from({ length: 40 }, (_, i) => {
    const round = i + 1;
    const record = {
      round,
      trustTableHash: sha256(`sample-trust-table-${round}`),
      coverage,
      historyHash: sha256(`sample-history-${round}`),
      maxTrustRise: Math.min(0.1, Math.round((0.1 - 0.0025 * (round % 7)) * 1000) / 1000),
      prevHash: prev,
    };
    record.blockHash = sha256(JSON.stringify(record));
    prev = record.blockHash;
    return record;
  });
  writeData("ledger.json", {
    ...(ledger ?? {}),
    _meta: { status: "sample", sources: [],
      note: "Sample chain. Hash links are real SHA-256, contents are made up. Real rounds " +
        "come from EARN (Week 3) committed through the Hardhat contract (Week 5).",
      wouldComeFrom: "results/ledger_rounds.json", syncedAt: today },
    rules: { maxTrustStep: 0.1, halvingFactor: 0.5 },
    rounds,
  });
}

console.log("Done.");
