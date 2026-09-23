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
// results/ is gitignored, so a fresh clone has none: the committed copies in docs/results/ are
// the fallback, both at the same relative path and flattened (docs/results/runs.csv stands in
// for results/grid/runs.csv). When both exist the NEWER file wins, so a fresh run is picked up
// but a stale leftover in results/ cannot silently downgrade the published data.
const FALLBACK = path.resolve(WEB, "..", "docs", "results");
const DATA = path.join(WEB, "data");
const today = new Date().toISOString().slice(0, 10);

const find = (f) =>
  [path.join(RESULTS, f), path.join(FALLBACK, f), path.join(FALLBACK, path.basename(f))]
    .filter((c) => fs.existsSync(c))
    .sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs)[0] ?? null;
const exists = (f) => find(f) !== null;
// Newest file in docs/results - anything older in results/ is probably a leftover run.
const snapshotMs = fs.existsSync(FALLBACK)
  ? Math.max(...fs.readdirSync(FALLBACK).map((n) => {
      const s = fs.statSync(path.join(FALLBACK, n));
      return s.isFile() ? s.mtimeMs : 0;
    }), 0)
  : 0;
const stale = [];
const readText = (f) => {
  const chosen = find(f);
  if (chosen.startsWith(RESULTS) && snapshotMs && fs.statSync(chosen).mtimeMs < snapshotMs - 864e5) {
    stale.push(f);
  }
  return fs.readFileSync(chosen, "utf8");
};
const readJson = (f) => JSON.parse(readText(f));
const readData = (f) =>
  fs.existsSync(path.join(DATA, f)) ? JSON.parse(fs.readFileSync(path.join(DATA, f), "utf8")) : null;
/** Writes only when something other than the sync date changed, so re-running is a no-op. */
const writeData = (f, obj) => {
  const target = path.join(DATA, f);
  const next = JSON.stringify(obj, null, 2) + "\n";
  if (fs.existsSync(target)) {
    const strip = (s) => s.replace(/"syncedAt": "[^"]*"/g, '"syncedAt": ""');
    if (strip(fs.readFileSync(target, "utf8")) === strip(next)) {
      console.log(`  data/${f} unchanged`);
      return;
    }
  }
  fs.writeFileSync(target, next);
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
    ...(exists("gate_g0a_retry.json") ? { retry: g0aRetry(readJson("gate_g0a_retry.json")) } : {}),
  });
}

function g0aRetry(r) {
  return {
    source: "results/gate_g0a_retry.json (scripts/06b_gate_g0a_retry.py)",
    verdict: r.verdict, passed: r.passed, next: r.next,
    definitions: Object.entries(r.definitions).map(([key, d]) => ({
      key, role: d.role, statistic: d.statistic, holderAuroc: d.holder_auroc, passed: d.passed,
      perClassSpearman: d.per_class_spearman,
    })),
  };
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
    ...(exists("gate_g0b_retry.json") ? { retry: g0bRetry(readJson("gate_g0b_retry.json")) } : {}),
  });
}

function g0bRetry(r) {
  const run = (x) => ({ rounds: x.rounds, variant: x.variant, mean: x.balanced_accuracy_mean,
    sd: x.balanced_accuracy_std, rareMacroF1: x.rare_macro_f1_mean,
    perSeed: Object.entries(x.per_seed).map(([seed, value]) => ({ seed: Number(seed), value })),
    curve: x.curve_mean.map((v, i) => ({ round: i + 1, balancedAccuracy: v })) });
  return {
    source: "results/gate_g0b_retry.json (scripts/08_g0b_retry.py)",
    passed: r.passed, statistic: r.statistic, threshold: r.threshold, definition: r.definition,
    retry: run(r.retry),
    amended: { frozen: run(r.amendment_D4_100_rounds.frozen), ft4: run(r.amendment_D4_100_rounds.ft4) },
    fineTuningMinutes: r.fine_tuning.fine_tuning.seconds / 60,
  };
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
    const r = exists("gate_g0a_retry.json") ? readJson("gate_g0a_retry.json") : null;
    const retry = r?.definitions.retry_bias_round1;
    setGate("G0a", { state: (r ? r.passed : g.passed) ? "passed" : "failed",
      result: `Mean Spearman ${g.statistic.toFixed(2)}` +
        (retry ? `; one retry (bias row) ${retry.statistic.toFixed(2)}` : "") + ` (needs >= ${g.threshold})`,
      resultStatus: "verified",
      note: r ? "Failed after its one allowed retry, so the project took Fallback F1. A sign-aware " +
        "amendment (M1) scored " + r.definitions.M1_signed_bias_round1.statistic.toFixed(2) +
        " and failed too." : undefined });
  }
  if (exists("gate_g0b.json")) {
    const g = readJson("gate_g0b.json");
    const r = exists("gate_g0b_retry.json") ? readJson("gate_g0b_retry.json") : null;
    setGate("G0b", { state: (r ? r.passed : g.passed) ? "passed" : "failed",
      result: r ? `Retry ${r.statistic.toFixed(3)}; first attempt ${g.statistic.toFixed(3)} (needs >= ${g.threshold})`
        : `FedAvg balanced accuracy ${g.statistic.toFixed(3)} (needs >= ${g.threshold})`,
      resultStatus: "verified",
      note: r ? "Passed on the design doc's retry: the last dense block fine-tuned by FedAvg, " +
        "features re-extracted." : undefined });
  }
  if (exists("analysis.json")) {
    const a = readJson("analysis.json");
    const g1 = a.gate_g1;
    const worst = (split) => Object.values(g1[split].per_class).reduce((b, v) => (v.f1_drop > b.f1_drop ? v : b));
    setGate("G1", { state: g1.s1.passed || g1.s2.passed ? "passed" : "failed",
      result: `S1: F1 drop ${worst("s1").f1_drop.toFixed(2)} (needs >= 0.15) · S2: F1 drop ` +
        `${worst("s2").f1_drop.toFixed(2)} but balanced accuracy fell ${g1.s2.balanced_accuracy_drop.toFixed(3)} (must be < 0.03)`,
      resultStatus: "verified",
      note: a.framing.decision });
    const g2 = a.g2_oracle;
    setGate("G2", { state: "not-run",
      result: "Not evaluable - G0a failed, so EARN has no working signal",
      note: `Exploratory re-run on an oracle evidence signal: conditions ${g2.s1.passed ? "met" : "not met"} ` +
        `on S1, ${g2.s2.passed ? "met" : "not met"} on S2. Reported as exploratory, never as the gate.` });
  }
  roadmap._meta.syncedAt = today;
  writeData("roadmap.json", roadmap);
}

// ------------------------------------------------------------------ study.json
if (exists("analysis.json")) {
  const a = readJson("analysis.json");
  const slim = (rows) => rows.map((r) => Object.fromEntries(Object.entries(r).map(([k, v]) =>
    [k, v && typeof v === "object" && "mean" in v ? { mean: v.mean, sd: v.sd } : v])));
  writeData("study.json", {
    _meta: { status: "verified", sources: ["results/analysis.json", "results/grid/runs.csv"],
      producedBy: "scripts/09_run_grid.py, scripts/10_analyse.py", syncedAt: today,
      note: `${a._meta.runs} runs, pre-registered in docs/DEVIATIONS.md D6. Tier A on the ` +
        `fine-tuned features, ${a._meta.rounds} rounds, seeds ${a._meta.seeds.join(", ")}.` },
    runs: a._meta.runs, rounds: a._meta.rounds, seeds: a._meta.seeds,
    shares: a.shares, gateG1: a.gate_g1, gateG1Reported: a.gate_g1_camp_a_reported, framing: a.framing,
    f1: { s1: slim(a.f1_tables.s1), s2: slim(a.f1_tables.s2) },
    earn: { s1: slim(a.earn_tables.s1), s2: slim(a.earn_tables.s2) },
    g2Oracle: a.g2_oracle, overheadMs: a.overhead_ms_per_round,
    ...(exists("followup_ledger.json") && exists("followup_ledger_trajectory.json") ? {
      ledgerFollowup: (() => {
        const f = readJson("followup_ledger.json");
        const t = readJson("followup_ledger_trajectory.json");
        const pick = (m) => ({ trust: t.methods[m].trust_6, rareF1: t.methods[m].rare_f1,
          minTrust: t.methods[m].min_trust_6_after_turn, roundOfMin: t.methods[m].round_of_min,
          trustRound30: t.methods[m].trust_6_round_30 });
        return { entry: f.entry, attacker: f.attacker, seeds: f.seeds, criterion: f.criterion,
          splits: Object.fromEntries(Object.entries(f.splits).map(([sp, v]) => [sp, {
            lockMatters: v.lock_matters, rareF1Gain: v.rare_f1_gain_from_lock,
            rareF1: Object.fromEntries(Object.entries(v.methods).map(([m, r]) => [m, { none: r.rare_f1_no_attack, attacked: r.rare_f1_A2s }])) }])),
          trajectory: { split: t.split, earn: pick("earn"), noLedger: pick("earn_no_ledger") } };
      })() } : {}),
  });
} else {
  console.log("  study.json: results/analysis.json missing, left unchanged");
}

// ------------------------------------------------------------------ ledger.json
const ledger = readData("ledger.json");
const exampleRun = "ledger/earn_rounds_s1_none_seed42.json";
if (exists("ledger/ledger_rounds.json") && exists(exampleRun)) {
  const chain = readJson("ledger/ledger_rounds.json");
  const run = readJson(exampleRun);
  const onChain = new Map(chain.example_per_round.map((p) => [p.round, p]));
  let prevTrust = Array(run.rounds[0].trust_bps.length).fill(0);
  let prev = "0".repeat(64);
  const rounds = run.rounds.map((b) => {
    const rise = Math.max(0, ...b.trust_bps.map((t, i) => t - prevTrust[i])) / 10000;
    const rec = { round: b.round, trustTableHash: sha256(JSON.stringify(b.trust_bps)),
      coverage: b.coverage, historyHash: b.history_hash, maxTrustRise: rise, prevHash: prev,
      blockHash: b.block_hash, chainBlockHash: onChain.get(b.round)?.block_hash,
      gas: onChain.get(b.round)?.gas, trustBps: b.trust_bps };
    prevTrust = b.trust_bps;
    prev = b.block_hash;
    return rec;
  });
  const sm = chain.summary;
  writeData("ledger.json", {
    _meta: { status: "verified", sources: ["results/ledger/ledger_rounds.json", `results/${exampleRun}`],
      producedBy: "EARN (oracle evidence, exploratory) via scripts/09_run_grid.py; ledger/scripts/measure.js",
      note: "Real EARN rounds (S1, no attack, seed 42) from the exploratory oracle-evidence run, " +
        "replayed on the EarnLedger contract on a local Hardhat chain. Hashes are the Python ledger's " +
        "SHA-256 chain; the on-chain keccak block hash is shown too.",
      syncedAt: today },
    rules: { maxTrustStep: run.max_step, halvingFactor: 0.5 },
    rounds,
    onChain: { network: chain._meta.network, solc: chain._meta.solc, runs: sm.runs,
      roundsCommitted: sm.rounds_committed, gasPerRound: sm.gas_per_round,
      latencyMsPerRound: sm.latency_ms_per_round, deployGas: sm.deploy_gas,
      allFinalTrustMatch: sm.all_final_trust_match, allHistoryVerified: sm.all_history_verified,
      allTamperRejected: sm.all_tamper_rejected,
      tamperExample: chain.runs.find((r) => r.split === "s1" && r.attack === "none" && r.seed === 42)?.tamper_entry,
      tamperError: chain.runs[0].tamper_error },
  });
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

const NL = String.fromCharCode(10);
if (stale.length) {
  console.log(
    NL + "  WARNING: used " + stale.length + " file(s) from results/ that predate the " +
    "committed docs/results snapshot by more than a day:" + NL + "    " +
    stale.join(NL + "    ") + NL +
    "  Re-run the pipeline, or delete them so the committed copies are used." + NL,
  );
}
console.log("Done.");
