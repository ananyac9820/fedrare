// Replay EARN's real per-round trust tables (results/ledger/earn_rounds_*.json, written by
// scripts/09_run_grid.py) onto a fresh EarnLedger on the local Hardhat chain, and measure gas and
// latency per round. Also re-checks, on the real data, that the contract accepts every honest
// EARN round and rejects a coordinator that boosts a hospital faster than the step.
//
//   cd ledger && npx hardhat run scripts/measure.js
//
// Writes results/ledger/ledger_rounds.json.

const fs = require("fs");
const path = require("path");
const { ethers } = require("hardhat");

const RESULTS = path.resolve(__dirname, "..", "..", "results", "ledger");

const stats = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const mean = s.reduce((a, b) => a + b, 0) / s.length;
  return { n: s.length, mean, median: s[Math.floor(s.length / 2)], min: s[0], max: s[s.length - 1] };
};

async function replay(file) {
  const run = JSON.parse(fs.readFileSync(path.join(RESULTS, file), "utf8"));
  const K = run.n_clients, C = run.n_classes;
  const step = Math.round(run.max_step * 10000);
  const Ledger = await ethers.getContractFactory("EarnLedger");
  const ledger = await Ledger.deploy(K, C, step);
  const deployReceipt = await ledger.deploymentTransaction().wait();

  const gas = [], latencyMs = [], perRound = [];
  for (const block of run.rounds) {
    const t0 = process.hrtime.bigint();
    const tx = await ledger.commitRound(block.round, block.trust_bps, block.coverage, "0x" + block.history_hash);
    const receipt = await tx.wait();
    const ms = Number(process.hrtime.bigint() - t0) / 1e6;
    gas.push(Number(receipt.gasUsed));
    latencyMs.push(ms);
    perRound.push({ round: block.round, gas: Number(receipt.gasUsed), latency_ms: +ms.toFixed(3),
                    block_hash: await ledger.blockHash(block.round) });
  }

  const onChain = (await ledger.getTrust()).map(Number);
  const expected = run.rounds[run.rounds.length - 1].trust_bps;
  const finalMatches = JSON.stringify(onChain) === JSON.stringify(expected);
  const lastHist = "0x" + run.rounds[run.rounds.length - 1].history_hash;
  const historyVerified = await ledger.verifyHistory(run.rounds.length, lastHist);

  // Tamper test on the real table: the coordinator tries to lift one hospital's trust on one
  // disease by 0.3 in a single round. The entry forged is the lowest-trust one in the final
  // table, so the boost is a genuine rise (an entry already at 1.0 cannot rise at all).
  const forged = [...expected];
  const idx = expected.indexOf(Math.min(...expected));
  forged[idx] = Math.min(10000, forged[idx] + 3000);
  let tamperRejected = false, tamperError = null;
  try {
    await (await ledger.commitRound(run.rounds.length + 1, forged, run.rounds[0].coverage, lastHist)).wait();
  } catch (e) {
    tamperRejected = true;
    tamperError = (e.errorName || e.shortMessage || String(e)).slice(0, 120);
  }

  return {
    file, split: run.split, attack: run.attack, seed: run.seed, rounds: run.rounds.length,
    deploy_gas: Number(deployReceipt.gasUsed), gas_per_round: stats(gas),
    latency_ms_per_round: stats(latencyMs), total_gas: gas.reduce((a, b) => a + b, 0),
    final_trust_matches_python: finalMatches, history_hash_verified: historyVerified,
    tamper_rejected: tamperRejected, tamper_error: tamperError,
    tamper_entry: { client: Math.floor(idx / C), class: idx % C, from_bps: expected[idx], to_bps: forged[idx] },
    per_round: perRound,
  };
}

async function main() {
  const files = fs.readdirSync(RESULTS).filter((f) => /^earn_rounds_.*\.json$/.test(f)).sort();
  if (!files.length) throw new Error(`no earn_rounds_*.json in ${RESULTS}; run scripts/09_run_grid.py first`);
  const runs = [];
  for (const f of files) {
    const r = await replay(f);
    console.log(`${f}: gas/round mean ${r.gas_per_round.mean.toFixed(0)} (max ${r.gas_per_round.max}), ` +
      `latency mean ${r.latency_ms_per_round.mean.toFixed(2)} ms, final match ${r.final_trust_matches_python}, ` +
      `tamper rejected ${r.tamper_rejected}`);
    runs.push(r);
  }
  const allGas = runs.flatMap((r) => r.per_round.map((p) => p.gas));
  const allLat = runs.flatMap((r) => r.per_round.map((p) => p.latency_ms));
  const out = {
    _meta: { network: "Hardhat in-process chain (automine), chainId 31337", solc: "0.8.24, optimizer 200 runs",
             created_utc: new Date().toISOString(), contract: "ledger/contracts/EarnLedger.sol" },
    summary: {
      runs: runs.length, rounds_committed: allGas.length,
      gas_per_round: stats(allGas), latency_ms_per_round: stats(allLat),
      deploy_gas: runs[0].deploy_gas,
      all_final_trust_match: runs.every((r) => r.final_trust_matches_python),
      all_history_verified: runs.every((r) => r.history_hash_verified),
      all_tamper_rejected: runs.every((r) => r.tamper_rejected),
    },
    runs: runs.map(({ per_round, ...rest }) => rest),
    example_per_round: runs.find((r) => r.split === "s1" && r.attack === "none" && r.seed === 42)?.per_round ?? runs[0].per_round,
  };
  fs.writeFileSync(path.join(RESULTS, "ledger_rounds.json"), JSON.stringify(out, null, 2));
  console.log(`wrote ${path.join(RESULTS, "ledger_rounds.json")}`);
}

main().catch((e) => { console.error(e); process.exitCode = 1; });
