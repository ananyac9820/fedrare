# Results file format (status report item C7)

The shape of every experiment result, shared by the Pipeline & EARN and Attacks & Eval tracks.
Produced by `scripts/09_run_grid.py`, read by `scripts/10_analyse.py` and (via
`results/analysis.json`) by the website's `npm run sync-data`.

A **run** is one `(split, method, attack, seed)`: `split` in `s1 | s2`, `attack` in
`none | A1 | A2 | A3` (`src/attacks/attacks.py`), `method` from `src/experiments/grid.py`.

## `results/grid/runs.csv` - one row per run

| Column | Meaning |
|---|---|
| `split`, `method`, `attack`, `seed` | the run key |
| `attacker` | attacking centre id, `-1` for `none` |
| `rounds`, `seconds` | federated rounds; wall time including local training |
| `balanced_accuracy`, `accuracy`, `macro_f1`, `rare_macro_f1` | final-round global head, pooled test split |
| `f1_0` ... `f1_7` | per-class F1 (5 = Dermatofibroma, 6 = Vascular lesion) |
| `to_target_5`, `to_target_6`, `to_target_rare` | **attack success**: share of that class's (or both rare classes') test images predicted as the target class, nevus |
| `attacker_weight_5`, `attacker_weight_6` | attacker's weight on that rare head row, mean over the rounds its attack is active |
| `attacker_fedavg_weight` | attacker's training-size share - its weight under FedAvg |
| `specialist_weight_5`, `specialist_weight_6`, `specialist_fedavg_weight` | the same for centre 2, over all rounds |
| `specialist_trust_{c}_r15`, `specialist_trust_{c}_final` | EARN only: centre 2's trust at round 15 and at the end |
| `attacker_trust_{c}_final`, `attacker_trust_{c}_max` | EARN only |

## `results/grid/rounds.csv.gz` - one row per run and round

`split, method, attack, seed, round, balanced_accuracy, rare_macro_f1, f1_5, f1_6,
to_target_5, to_target_6, to_target_rare`

## `results/grid/weights.csv.gz` - one row per run, round, client and rare class

`split, method, attack, seed, round, client, class, weight` and, where the rule has them,
`evidence`, `trust`, `holder`, `coverage`. `weight` is `round_info["head_row_weights"][client, class]`
(`docs/aggregation_interface.md`).

## `results/ledger/earn_rounds_<split>_<attack>_seed<seed>.json` - EARN's chain input

`{split, attack, seed, n_clients, n_classes, max_step, rounds: [{round, trust (K x C floats),
trust_bps (K*C ints), coverage (C), history_hash (hex), block_hash (hex, Python chain)}]}`.
Written for the full `earn` method only. `ledger/scripts/measure.js` replays these on-chain and
writes `results/ledger/ledger_rounds.json` (gas and latency per round, final-state match,
tamper test).

## Gate records

`results/gate_g0a.json`, `gate_g0a_retry.json`, `gate_g0b.json`, `gate_g0b_retry.json`, and
G1 / G2-oracle inside `results/analysis.json` (`gate_g1`, `framing`, `g2_oracle`).
