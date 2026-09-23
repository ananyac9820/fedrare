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
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;

function parseCsv(text) {
  const [header, ...rows] = text.trim().split(/\r?\n/).map((l) => l.split(","));
  return rows.map((r) => Object.fromEntries(header.map((h, i) => [h, r[i]])));
}

fs.mkdirSync(DATA, { recursive: true });
console.log(`Syncing from ${RESULTS}`);

// Rare classes (ids + names) - needed to find the per-class F1 columns below.
const rareYaml = exists("rare_classes.yaml") ? loadYaml(readText("rare_classes.yaml")) : null;

// ------------------------------------------------------------------ dataset.json
if (exists("class_distribution.csv") && rareYaml && exists("gate_g0a.json")) {
  const dist = parseCsv(readText("class_distribution.csv"));
  const g0a = readJson("gate_g0a.json");
  const names = Object.keys(dist[0]).filter((k) => k !== "");
  const shares = rareYaml.class_share_pct;
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
      id, name, sharePct: shares[name], rare: rareYaml.rare_class_ids.includes(id),
    })),
    rareRule: {
      headRatioDivisor: rareYaml.head_ratio_divisor,
      cutoffPct: rareYaml.rare_cutoff_pct,
      headClass: names.find((n) => shares[n] === headShare),
      headSharePct: headShare,
      holderMinImages: rareYaml.specialist_min_images,
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

// ------------------------------------------------------------------ g0b.json
if (exists("gate_g0b.json")) {
  const g = readJson("gate_g0b.json");
  writeData("g0b.json", {
    _meta: { status: "verified", sources: ["results/gate_g0b.json"],
      producedBy: "scripts/07_tier_a_s1_baselines.py (feature/pipeline)", syncedAt: today },
    gate: "G0b", passed: g.passed, statistic: g.statistic, threshold: g.threshold,
    definition: g.definition,
    perSeed: Object.entries(g.per_seed).map(([seed, value]) => ({ seed: Number(seed), value })),
    config: g.tier_a_config,
  });
}

// ------------------------------------------------------------------ baseline_results.json
// One metrics file per rule (scripts/07 writes tier_a_s1_<rule>.csv); the older combined
// tier_a_s1_metrics.csv is used only if no per-rule file exists.
const perRule = ["tier_a_s1_fedavg.csv", "tier_a_s1_camp_a.csv"].filter(exists);
const baselineFiles = perRule.length ? perRule : ["tier_a_s1_metrics.csv"].filter(exists);
if (baselineFiles.length && rareYaml) {
  const rows = baselineFiles.flatMap((f) => parseCsv(readText(f)));
  const rare = rareYaml.rare_class_ids.map((id, i) => ({
    name: rareYaml.rare_class_names[i],
    column: `f1_${id}_${rareYaml.rare_class_names[i].replaceAll(" ", "_")}`,
  }));
  const labels = { fedavg: "FedAvg", camp_a: "Camp A (evidence weighting)" };
  const num = (r, k) => Number(r[k]);

  const rules = [...new Set(rows.map((r) => r.rule))].map((id) => {
    const rs = rows.filter((r) => r.rule === id);
    const lastRound = Math.max(...rs.map((r) => Number(r.round)));
    const final = rs.filter((r) => Number(r.round) === lastRound);
    const perSeed = final.map((r) => ({
      seed: Number(r.seed),
      balancedAccuracy: num(r, "balanced_accuracy"),
      accuracy: num(r, "accuracy"),
      rareMacroF1: num(r, "rare_macro_f1"),
      rareF1: Object.fromEntries(rare.map((c) => [c.name, num(r, c.column)])),
    }));
    const rounds = [...new Set(rs.map((r) => Number(r.round)))].sort((a, b) => a - b);
    return {
      id, label: labels[id] ?? id, status: "baseline", seeds: perSeed.length, rounds: lastRound,
      // means over seeds, final round
      balancedAccuracy: mean(perSeed.map((s) => s.balancedAccuracy)),
      accuracy: mean(perSeed.map((s) => s.accuracy)),
      rareMacroF1: mean(perSeed.map((s) => s.rareMacroF1)),
      rareF1: Object.fromEntries(rare.map((c) => [c.name, mean(perSeed.map((s) => s.rareF1[c.name]))])),
      perSeed,
      // mean over seeds after every round - shows whether the run had finished improving
      curve: rounds.map((round) => {
        const at = rs.filter((r) => Number(r.round) === round);
        return { round, balancedAccuracy: mean(at.map((r) => num(r, "balanced_accuracy"))),
          rareMacroF1: mean(at.map((r) => num(r, "rare_macro_f1"))) };
      }),
    };
  });
  writeData("baseline_results.json", {
    _meta: { status: "verified", sources: baselineFiles.map((f) => `results/${f}`),
      producedBy: "scripts/07_tier_a_s1_baselines.py", syncedAt: today,
      note: "Tier A, natural split S1. Final-round values, mean over seeds." },
    split: "S1", rules,
  });
} else {
  console.log("  baseline_results.json: no results/tier_a_s1_<rule>.csv yet, sample kept");
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
