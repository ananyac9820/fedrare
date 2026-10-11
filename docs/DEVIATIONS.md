# Deviations and amendments log

Every change to the pre-registered plan (EARN Project Design v2, 16 Sep 2026) is recorded here
**before** the run it affects, with the reason, and is committed before any result exists.
A result produced under an amendment is always reported next to the pre-registered result,
never in place of it. Entries are append-only: a later entry may supersede an earlier one but
never edits it.

Format: ID, date, what changes, why, which gate / result it touches, status.

---

## D1 - G0a retry: which of the two listed options is the retry (23 Sep 2026)

**What.** The design doc (Section 6.6, G0a "If it fails") lists two retries - the bias-row
change instead of the weight row, *or* evidence averaged over 3 rounds - and allows one retry.
We fix the **bias-row change** (listed first) as *the* retry that decides the gate. The
3-round average of the weight-row evidence is computed in the same script and reported, but
it does not decide the gate.

**Why.** Running both and letting the better one decide would be two retries.

**Measured exactly like G0a** (scripts/06_gate_g0a.py): round 1 of the Tier A loop, seed 42,
TierAConfig defaults, updates clipped to the median norm, Spearman across the 6 centres
between evidence and true training counts, per class; gate statistic = mean over the 8 classes;
pass at >= 0.7. Only the evidence definition changes: e(k,c) = |change of classifier.bias[c]|.

**Status.** Pre-registered in `scripts/06b_gate_g0a_retry.py` before its first run.

---

## D2 - Amendment M1: sign-aware evidence (23 Sep 2026) - METHOD CHANGE

**What.** A third evidence definition, run in the same script and reported separately as an
*amendment*, not as the design doc's retry:

    e_M1(k,c) = max(0, change of classifier.bias[c] in client k's clipped update)

i.e. the **signed** bias change, with negative values (the model was pushed *away* from
class c) set to zero. Same round, seed, clipping, statistic and 0.7 threshold as G0a.

**Why.** G0a came out *inverted* (mean Spearman -0.21): for both rare classes the centre with
the highest evidence (centre 4) holds zero images of them. Any magnitude-only measure (the
weight row, the bias row, or a multi-round average of either) cannot tell "pushed up" from
"pushed down". A centre that never sees class c pushes that class's score *down* on every
single step, so under Adam - which normalises step size per parameter - its row moves the
most, consistently. That is the mechanism behind the inversion, and it predicts that both
listed retries fail too.

The label-inference work the design doc cites for this signal (Section 4.1: "known signal
from label-inference work"; Ramakrishna & Dan 2022; also iDLG and LLG) uses the **sign** of
the output-layer gradient: a class present in the batch makes its bias gradient negative, so
its bias rises. The magnitude-only definition in G0a was therefore a mis-transcription of the
cited signal rather than the signal itself. We still treat the correction as a method change,
because it was made after seeing G0a fail.

**Risk stated in advance.** The local loss is class-balanced (effective-number weights), which
deliberately removes most count information: a centre holding 4 images of a class is weighted
as if the class were balanced. So e_M1 may detect *presence* well and *rank counts* poorly,
and can still fail the Spearman-vs-counts gate. That outcome will be reported as is.

**Decision rule, fixed now.**
- If D1 (the retry) passes: G0a passes; M1 is reported only.
- If D1 fails and M1 passes: by the design doc's letter G0a has **failed** and the project
  goes to Fallback F1. We will report that verdict, and in addition carry EARN forward on M1
  as a clearly labelled amended track, so the team can choose the framing with both sets of
  numbers in hand. Every EARN/Camp A number produced on M1 is labelled "amended evidence (M1)".
- If both fail: G0a has failed; Fallback F1 only. No further evidence variants will be tried.

**Status.** Pre-registered in `scripts/06b_gate_g0a_retry.py` before its first run.

---

## R1 - Result: G0a retry and M1 (23 Sep 2026)

`scripts/06b_gate_g0a_retry.py` -> `results/gate_g0a_retry.json`. Mean Spearman, threshold 0.7:

| Evidence | Role | Statistic | Holder AUROC |
|---|---|---|---|
| weight row, round 1 (original G0a) | failed 20 Sep | -0.210 | - |
| bias row, round 1 | D1 - the retry | **-0.463** | 0.137 |
| weight row, rounds 1-3 averaged | reported only | -0.240 | 0.145 |
| signed bias, round 1 | D2 - amendment M1 | 0.110 | 0.738 |

**Verdict under the rule fixed in D2: G0a FAILED; M1 failed too -> Fallback F1 only.** No
further evidence variants will be tried. Both magnitude retries came out inverted, as the D2
mechanism predicted. M1 separates holders from non-holders better than chance (AUROC 0.74)
but does not rank counts, the risk stated in advance.

---

## D3 - G0b retry: unfreeze the last dense block (23 Sep 2026)

**What.** The design doc's G0b retry ("unfreeze the last dense block and re-extract features.
One retry"), made concrete before running:

- Backbone: ImageNet DenseNet-121. Everything up to and including `transition3` stays frozen;
  `denseblock4` + `norm5` are trained, with a fresh 1024->8 classifier.
- Trained **federated** (FedAvg through the same aggregation interface, BatchNorm running
  statistics averaged, `num_batches_tracked` copied), on S1 **training** images only, all 6
  centres: 20 rounds x 50 local steps, batch 32, Adam (block lr 1e-4, classifier lr 5e-4),
  class-balanced loss, seed 42. No augmentation (inputs are cached `transition3` outputs).
  The test split is never touched.
- The trained block (eval mode) then re-extracts the 1024-d features for train and test
  (`data/features/fed_isic2019_densenet121_ft4_*.npz`); the classifier used during
  fine-tuning is discarded.
- G0b is then re-measured exactly as before: Tier A FedAvg on S1, TierAConfig defaults (40
  rounds), seeds 42/43/44, final-round balanced accuracy, mean over seeds, pass at >= 0.45.

**Why federated.** Pooling all hospitals' images to fine-tune would contradict the setting.

**Status.** Pre-registered in `scripts/08_g0b_retry.py` before its first run.

## D4 - Longer training, "more rounds" (23 Sep 2026) - AMENDMENT (status report C3)

**What.** All Tier A experiments from here on use **100 rounds** instead of 40, on the D3
features, whatever the D3 outcome. G0b is additionally reported at 100 rounds on both feature
sets, labelled as the amendment; the 40-round D3 number is the one that decides G0b.

**Why.** The 20 Sep baseline was still climbing at round 40 (0.397 at round 30, 0.415 at 40).
A comparison against an unconverged baseline mostly measures convergence speed. This was
decided after seeing that result, so it is an amendment, not part of the pre-registered
fallback.

**Status.** Fixed before any 100-round run exists.

---

## R2 - Result: G0b retry (23 Sep 2026)

`scripts/08_g0b_retry.py` -> `results/gate_g0b_retry.json`. Federated fine-tuning of
denseblock4 + norm5 took 12.8 min on Apple MPS.

| Features | Rounds | Role | Balanced acc (mean ± sd, 3 seeds) | Rare macro-F1 |
|---|---|---|---|---|
| frozen | 40 | G0b, failed 20 Sep | 0.415 ± 0.005 | 0.332 |
| **ft4** | **40** | **D3 - the retry** | **0.535 ± 0.002** | - |
| frozen | 100 | D4 amendment | 0.450 ± 0.007 | 0.348 |
| ft4 | 100 | D4 amendment (used from here on) | 0.536 ± 0.003 | 0.592 |

**Verdict: G0b PASSES on its pre-registered retry.** Disclosed limitations: the block was
fine-tuned once (seed 42), without attackers, on the S1 assignment of the training images; S2
reuses the same features (same images, different centre assignment). Attacks act on the
classifier head only.

---

## D5 - Scope after G0a: Fallback F1, plus EARN as a labelled exploratory analysis (23 Sep 2026)

**Decision (Arya, 23 Sep).** The paper's main result follows the design doc: **Fallback F1**,
an empirical study of how rare-class-aware aggregation behaves on a real hospital split,
including the never-cut items (the G1 attack experiment on S1 and S2, honest negative
results). In addition, EARN is built in full (aggregator, ablations, ledger, contract, gas and
latency) and run on an **oracle evidence signal** (D6). Every EARN number is labelled
"exploratory - oracle evidence" and none is presented as validating EARN, because the signal
EARN needs does not exist on our data (R1). Tier B and BOBA are cut (design doc cut lines 1
and 2); the real chain is kept (cut line 3 not needed).

## D6 - Pre-registration of the F1 study, G1, and the exploratory EARN runs (23 Sep 2026)

Fixed before any of these runs exists. Code: `src/attacks/attacks.py`,
`src/federated/earn.py`, `src/experiments/grid.py`, `scripts/09_run_grid.py`,
`scripts/10_analyse.py`.

**Common.** Features `ft4` (D3), Tier A, 100 rounds (D4), all other TierAConfig defaults;
seeds 42, 43, 44; splits S1 (natural) and S2 (`src/data/splits.py`, fixed seed); rare classes
5 (Dermatofibroma) and 6 (Vascular lesion); final-round global head on the pooled test split.

**Attacks** (constants in `src/attacks/attacks.py`). Target class: Melanocytic nevus (1).
- A1 sudden expert - attacker **centre 4** (no rare images on S1 or S2), every round: trains
  on rare->nevus flipped labels, then rewrites both rare head rows to norm 3x its largest
  honest row norm along minus its unit mean feature; rare biases move by -3x its largest
  honest bias change.
- A2 sleeper - attacker **centre 1** (a real holder of both rare classes on S1), honest for
  rounds 1-15, A1 behaviour from round 16.
- A3 scaling - attacker **centre 4**, flipped-label update multiplied by 10, every round.
- none.

**F1 methods** (main study): `fedavg`, `fedavg_clipped`, `krum` (f=1), `multi_krum` (f=1),
`trimmed_mean` (trim 1), `coordinate_wise_median`, `camp_a` (the design doc's Camp A:
evidence = head weight-row norm), and **`camp_a_reported`** - an addition: Camp A whose
evidence is each hospital's *reported* class counts (the cwFedAvg / FedSat / MLFCIL style,
design doc Section 3 row 1). An attacker that claims to be a holder (A1 always, A2 from
round 16) reports, for each rare class, the largest honest count of that class on that split;
otherwise every hospital reports its true training counts.

**Measured per run.** Balanced accuracy, macro-F1, F1 of each rare class, rare macro-F1;
attack success = share of rare-class test images predicted as nevus; the attacker's weight on
each rare head row averaged over the rounds its attack is active; centre 2's weight on each
rare row averaged over all rounds; the attacker's FedAvg weight (its training-size share).

**G1 (design doc 6.6, operationalised).** For split s and rare class c, under `camp_a`:
ratio = attacker's mean rare-row-c weight under A1 / its FedAvg weight;
dF1 = F1_c(no attack) - F1_c(A1); dBA = balanced acc(no attack) - balanced acc(A1); all means
over the 3 seeds. **G1 passes on s if some rare c has ratio >= 2 and dF1 >= 0.15, with
dBA < 0.03.** The same numbers for `camp_a_reported` are reported, not gating. The S1/S2
framing then follows the design doc's Section 6.2 table.

**Exploratory EARN (oracle evidence; not a gate).** Evidence = the reported counts used by
`camp_a_reported` (so an attacker can spoof it exactly as there); holder = at least 20 images
(tau = 19.5); EARNConfig defaults (step 0.1, halving, agreement 0.5, cap 0.5, divisor 4,
lam 1). Variants:
- `earn` (full), ablations `earn_no_blend`, `earn_no_ramp`, `earn_no_ledger` (the attacker's
  history is editable);
- sensitivity: `earn_step0.05`, `earn_step0.2`, `earn_tau10` (>= 10 images), `earn_tau40`
  (>= 40 images);
- `earn_real_signal`: EARN on the design doc's own evidence (weight-row norm, holder = above
  the class's mean evidence that round) - shows what EARN does with the signal that failed G0a.

All with the same attacks, splits and seeds. **G2-oracle** (reported with the design doc's G2
wording, explicitly not the gate, which is not evaluable after G0a failed): under A1 and A2,
`earn` rare macro-F1 within 0.05 of its no-attack value; with no attack, within 0.02 of
`camp_a_reported` and above `fedavg`; centre 2's trust on both rare classes >= 0.8 at round 15
with no attack. Evaluated per split.

**Output format (status report C7)** - `docs/results_format.md`.

---

## R3 - Results: F1 study, G1, exploratory EARN, ledger (23 Sep 2026)

All 408 pre-registered runs completed (`results/grid/`, analysis in `docs/results/analysis.md`).

**G1 fails on both splits, for different reasons.** Camp A (update evidence) under A1:
- S1: the attacker captures 5.0x / 5.5x its FedAvg weight on the Dermatofibroma / Vascular
  lesion rows, but those classes' F1 drops only 0.086 / 0.017 (needs >= 0.15).
- S2: 4.8x / 4.7x weight and F1 drops of 0.207 / 0.237 - both met - but balanced accuracy
  falls 0.051, above the 0.03 stealth bar. The attack works on S2; it is not stealthy.
Framing (design doc 6.2): "succeeds on neither" -> **Fallback F1**, consistent with R1.

**One correction to a measurement script, disclosed.** The ledger tamper check first forged
+0.3 on centre 4 / vascular lesion. In the three S2 + A1 runs that entry was already at 1.0
(the A1 attacker had reached full trust), so the "forgery" was not a rise and was correctly
accepted. The check now forges the lowest-trust entry; all 24 runs reject it. No experiment
result changed.

**Diagnostic (explains, does not change, a result).** Why EARN's honest specialist earns
little trust on S1: honest holders' per-round rare-row updates agree weakly with each other
(centre 2's mean agreement 0.14 / 0.12, other holders 0.00 / 0.43 for DF / VL), so the peer
check at the pre-registered 0.5 bar fails honest hospitals once training settles. On S2
(coverage 1) only the history check runs, and it passes any *consistent* client - the honest
specialist (agreement 0.85-0.89) and the A1 attacker alike.

---

## D7 - Follow-up: does the locked history matter? Specialist sleeper A2s (23 Sep 2026, after merge)

**Why.** In the D6 grid the "no ledger" ablation (attacker may rewrite its stored history) changed
nothing measurable, because no tested attacker needed to rewrite its past: A1 is consistent from
round 1, and A2's attacker (centre 1) is not the sole holder anywhere. The design doc's argument for
the ledger (Section 5) is about the case where **the only holder of a disease turns**: at coverage 1
only the history check runs, so a locked history should expose the turn and an editable one should
hide it. That case was not in the grid. EARN remains exploratory (oracle evidence).

**What.** New attack setting **A2s** = the A2 sleeper behaviour (honest rounds 1-15, A1 from round 16:
rare labels flipped to nevus, rare rows inflated 3x against the rare classes, claims to be the top
holder) with attacker **centre 2**, the specialist - the sole rare holder on S2 (coverage 1) and one
of 4 / 3 holders on S1. Methods: `fedavg`, `camp_a_reported`, `earn`, `earn_no_ledger` (attacker's
history editable). Splits S1 and S2; seeds 42, 43, 44; everything else exactly as D6 (ft4
features, 100 rounds, oracle evidence, holder >= 20 images). The no-attack rows are the existing D6
runs. Script: `scripts/11_ledger_followup.py` -> `results/followup_ledger.json`.

**Pre-registered criterion ("the lock matters"), per split:** mean over seeds of
rare macro-F1(earn, A2s) - rare macro-F1(earn_no_ledger, A2s) >= 0.05 **and** the attacker's mean
rare-row weight while attacking under `earn` is at most half of that under `earn_no_ledger`.
Prediction written now: met on S2 (history is the only check), not met on S1 (peers dominate).
Either outcome is reported. No other variant will be run under this entry.

---

## R4 - Result: D7 specialist sleeper (23 Sep 2026)

`scripts/11_ledger_followup.py` -> `results/followup_ledger.json` (copy in `docs/results/`).
Mean of seeds 42-44; rare macro-F1 no attack -> under A2s:

| Split | FedAvg | Camp A (reported) | EARN (oracle) | EARN, no ledger | Lock criterion |
|---|---|---|---|---|---|
| S1 | 0.592 -> 0.484 | 0.605 -> 0.505 | 0.596 -> 0.552 | 0.596 -> 0.552 | not met (gain +0.000) |
| S2 | 0.490 -> 0.000 | 0.560 -> 0.000 | 0.555 -> 0.000 | 0.555 -> 0.000 | not met (gain +0.000) |

**Verdict: the pre-registered criterion is not met on either split; the prediction for S2 was
wrong.** Reported as is.

**Diagnostic (explains, does not change, the result; seed 42, S2, vascular-lesion row).**
- *With the ledger*, the history check **does detect the turn**: agreement goes from +0.75 (round 15)
  to -0.27 (round 16) and trust falls 1.00 -> 0.50 -> 0.25 -> 0.12 -> 0.06 (round 19).
  *Without the ledger* (history editable) trust stays at 1.00 throughout. This is the first
  measured effect of the history lock in this project.
- Detection does not become protection, for two design reasons:
  1. **Trust only removes a bonus.** EARN's row weight is size share + T x evidence share; at T = 0
     the attacker keeps its FedAvg-size share (15.8% on S2), and as the sole holder nobody outweighs
     it. Rare F1 falls 0.605 -> 0.313 (round 16) -> 0.000 (round 20) even while trust is near zero.
  2. **A running-mean history is locked against rewriting, not against drift.** The attacker's new
     malicious updates enter its history, agreement recovers (+0.94 by round 30) and trust returns to
     1.00 by round 30.
- At coverage 1 there is also no other source of the disease's knowledge: once the only holder
  turns, no aggregation rule can recover the class (FedAvg and Camp A also reach 0.000). The best a
  defence can do there is to detect and flag - which the lock did.

No further variants will be run under D7. Design implications (future work, not tested): weight
floors that let trust go *below* the FedAvg share; a history that is frozen after it is established
(or decays towards the pre-turn record) rather than a running mean.

---

## D8 - Tier B confirmation, pre-registered and ready (23 Sep 2026) - NOT YET RUN

**What.** Full DenseNet-121 (ImageNet init, all layers trained) federated on S1, 6 centres, 20 rounds
x 50 local Adam steps (batch 32, lr 1e-4, class-balanced loss, train-time augmentation), seed 42,
same aggregation interface and the same sleeper attack as Tier A (centre 1, honest rounds 1-15).
Three runs: FedAvg no attack; FedAvg under A2; FedAvg + clipping under A2. Test metrics after rounds
5, 10, 15, 20. Script: `scripts/12_tier_b.py` (smoke-tested; resumable per round).

**Criterion (fixed now):** Tier B *confirms* the Tier A sleeper finding if the A2 drop in FedAvg's
final rare macro-F1 is >= 0.10 **and** clipping recovers at least half of that drop.

**Why only this.** Measured cost: ~1.24 s per training step on the Apple GPU, ~2.1 h per run,
~6.5 h for the three. Tier B was the design doc's first cut line; this is the smallest run that can
confirm one headline finding with the real model. One seed only - stated as a limitation.

**Status.** Not run: it needs the laptop for ~6.5 h, which is the team's decision.

**R4 addendum (same day).** The diagnostic was repeated over all three seeds
(`scripts/11b_ledger_trajectory.py` -> `results/followup_ledger_trajectory.json`): with the locked
history the attacker's vascular-lesion trust goes 1.00 (round 15) -> 0.06 (minimum, round 19) ->
1.00 (round 30); with the editable history it stays at 1.00 throughout. Mean rare F1 is 0.589 at
round 15 and 0.018 at round 20 (locked) vs 0.586 -> 0.000 (editable). Same conclusion as above.

## R5 - Literature re-check done; BOBA not applicable as specified (23 Sep 2026)

See `docs/LITERATURE_RECHECK.md`. CELM already uses "class coverage" as a normalising term (paper
wording narrowed); CALM (Sep 2026) does per-class peer agreement without trust, coverage adaptation
or attacks; BOBA needs a (c-1)-dimensional subspace (7) from n-f = 5 gradients and server data for
every class, so it cannot be run as published with 6 clients and 8 classes. Our cut of BOBA
(design doc cut line 2) therefore stands on substance. CARE-FL's full text could not be accessed.

## R6 - Result: DermaCon-IN, gates DG0 and DG2 (29 Sep 2026)

Pre-registered in `docs/DERMACON_PLAN.md`, committed before any run on this dataset.

**DG0 cleared, narrowly.** FedAvg on D1 (age bands), 40 rounds, seed 42: balanced accuracy
**0.4082** against a bar of 0.40, over 7 classes. The margin is 0.008 and is reported as such;
the near-IID control D0 reached 0.4569. Rare-class macro-F1 is 0.1317 on D1 and 0.1929 on D0.
`docs/results/gate_dg0.json`, `docs/results/dermacon_baselines_fedavg.csv`.

**DG2 not cleared.** Of the three conditions, two were met and one was not:

| condition | result |
| --- | --- |
| D1 rare macro-F1 within 0.05 of FedAvg | met - EARN 0.1240 vs FedAvg 0.1317 |
| D1 specialist weight share above its size share | **not met** - centre 4 got 0.068 of the rare rows against a size share of 0.079 |
| D0 within 0.05 of FedAvg (no effect expected) | met - 0.1778 vs 0.1929 |

**Why it failed, measured rather than guessed.** EARN's holder threshold is an absolute count:
tau = HOLDER_MIN_IMAGES - 0.5 = 19.5, i.e. a centre must hold 20 images of a class before it can
be a holder of it. That was calibrated on Fed-ISIC2019, where 4 of 6 centres clear it for the rare
classes. On DermaCon-IN's D1, only centre 3 clears it, and only for Keratinisation (24 images); the
designated specialist, centre 4, holds 11 Keratinisation and 8 Neoplasms and so is a holder of
neither. With no holders, EARN's trust and evidence terms never engage and it reduces to capped
FedAvg - the small shortfall against the size share is the cap and renormalisation, not a decision
about the specialist.

Camp A, which uses the same oracle evidence proportionally and has no holder threshold, does lift
the specialist on the same split: rare-row weight 0.137 against a size share of 0.079, with rare
macro-F1 0.1470 and balanced accuracy 0.4315, both above FedAvg.

**What this is worth.** This is what a second dataset is for. The finding is that the mechanism's
holder threshold does not transfer across dataset scales - an absolute image count is the wrong
unit when the rarest class has 52 images instead of 216 - while the evidence-proportional part of
the idea does transfer. It is reported as a failure of the gate as written, not softened.

`docs/results/gate_dg2.json`, `docs/results/dermacon_earn.csv`, `scripts/18_dermacon_earn.py`.

---

## D9 - Amendment M2: a scale-relative holder threshold, for DermaCon-IN only (29 Sep 2026) - POST-HOC

**What.** Re-run the DG2 comparison on DermaCon-IN with `EARNConfig(tau="mean")` - a centre is a
holder of class c when its evidence for c is above this round's mean across centres - in place of
the absolute tau = 19.5. Nothing else changes: same splits, same oracle evidence, same 40 rounds,
same seed 42, same three DG2 conditions.

**Why.** R6 shows the absolute threshold is the reason EARN does nothing on this dataset. A
threshold relative to the round's own evidence is scale-free, is already implemented in
`src/federated/earn.py` (it exists for evidence with no natural unit), and needs no new code.

**Status when written.** Not yet run. This is **post-hoc**: it was chosen after seeing DG2 fail
and knowing why. Its result is therefore exploratory and is reported next to the pre-registered
DG2 result, never in place of it, and the DG2 verdict stands as recorded in R6 whatever M2 shows.
It does not change any Fed-ISIC2019 result, where the absolute threshold stays as pre-registered.

## R7 - Result: amendment M2 on DermaCon-IN, three seeds (29 Sep 2026) - POST-HOC

M2 (D9) run on D0 and D1, seeds 42/43/44, 40 rounds, oracle evidence. DG2 remains decided by
seed 42 under the pre-registered threshold and remains NOT CLEARED (R6); nothing here changes it.

**D1 (age bands), mean +- sd over 3 seeds**

| method | balanced accuracy | rare macro-F1 | specialist's rare-row weight |
| --- | --- | --- | --- |
| FedAvg | 0.4217 +- 0.0127 | 0.1557 +- 0.0208 | 0.079 = its size share |
| Camp A (oracle evidence) | 0.4377 +- 0.0097 | 0.1633 +- 0.0161 | 0.137 |
| EARN, tau = 19.5 (pre-registered) | 0.4181 +- 0.0165 | 0.1478 +- 0.0261 | 0.068 |
| EARN, tau = mean (M2) | 0.4307 +- 0.0085 | 0.1714 +- 0.0125 | 0.142 +- 0.003 |

**D0 (near-IID control)**: FedAvg 0.1992 +- 0.0213 rare macro-F1, Camp A 0.1879 +- 0.0174,
EARN 0.1883 +- 0.0207, M2 0.1730 +- 0.0119. All within the 0.05 margin of FedAvg.

**What holds and what does not.** Two different claims have to be separated here.

*The weighting behaves as intended, and that part is solid.* Under M2 the specialist centre
receives 0.142 +- 0.003 of the rare rows against a size share of 0.079 - 1.8x - while under the
pre-registered threshold it receives 0.068, below its size share. The spread across seeds is
0.003 because the weight is a function of the evidence and the trust schedule, not of training
noise. The diagnosis in R6 is confirmed: an absolute holder threshold of 20 images is what stops
the mechanism engaging at this dataset's scale.

*The downstream benefit is not established.* M2's rare macro-F1 on D1 is 0.1714 +- 0.0125 against
FedAvg's 0.1557 +- 0.0208. The gap of 0.016 is smaller than either standard deviation, so on three
seeds it cannot be separated from noise. On seed 42 alone the same comparison reads 0.1642 vs
0.1317, a gap of 0.033, because seed 42 happens to be a low draw for FedAvg - which is exactly the
reason a single-seed claim was not good enough to report. On D0, M2 is consistently slightly below
FedAvg (0.1730 vs 0.1992), inside the margin but in the wrong direction.

**Therefore.** The honest statement is: the holder threshold does not transfer across dataset
scales, a scale-relative threshold restores the intended weighting, and whether that converts into
better rare-class accuracy on DermaCon-IN is not answered by three seeds. Establishing it would
need more seeds and a pre-registered threshold on data not yet looked at. Nothing here is claimed
as EARN working on Indian data.

`docs/results/dermacon_earn.csv`, `docs/results/gate_dg2.json` (M2 carried under
`amendment_reported_alongside`, with `decides_the_gate: false`).

## D10 - Amendment M2 tested on Fed-ISIC2019, pre-registered (10 Oct 2026) - NOT YET RUN

**What.** Run EARN twice on Fed-ISIC2019, identical in every respect except the holder threshold:

  - *absolute* (as pre-registered everywhere so far): a centre is a holder of class c once it holds
    at least HOLDER_MIN_IMAGES = 20 images of c, i.e. `EARNConfig(tau=19.5)`;
  - *relative* (M2): a centre is a holder of c when its evidence for c is above this round's mean
    across centres, i.e. `EARNConfig(tau="mean")`.

Splits S1 and S2, attacks `none` and `A1`, seeds 42/43/44, 100 rounds, oracle evidence. A1 is
included deliberately: lowering the bar for who counts as a holder could make a *claiming* attacker
easier to admit, and that risk has to be measured, not assumed away.

**Why.** On DermaCon-IN the absolute threshold admitted almost nobody and EARN silently became
capped FedAvg (R6); a relative threshold restored the intended weighting (R7). That was post-hoc
on that dataset. This entry asks the opposite question on the dataset where the absolute threshold
does work: does the relative rule break anything, and does it make spoofing easier? Answering it
before changing the method is the difference between a fix and a rescue.

**Feature set - a limitation stated up front.** The published 408-run grid uses the `ft4` features
(D3). Those files live in a gitignored `data/` and are not on the machine doing this run, and
re-deriving them means fine-tuning a dense block, which is not feasible here. This amendment
therefore runs on the **frozen** features for both arms. The comparison is like for like and
answers the question asked; it does **not** restate any grid number, and the two arms must never be
quoted next to ft4 results. `scripts/19_earn_threshold.py --variant ft4` re-runs the whole thing
unchanged if those features become available.

**Criteria, fixed now.** Mean over the three seeds, per split:

1. *No harm without attack.* Under `none`, the relative arm's rare macro-F1 is within 0.05 of the
   absolute arm's, in either direction, on both S1 and S2.
2. *No easier to spoof.* Under A1, the attacker (centre 4, which holds no rare images on either
   split) gains no more than 0.05 additional weight share on the rare rows under the relative rule
   than under the absolute one, on both splits.
3. *The specialist is still served.* On S2, where centre 2 is the sole rare holder, its rare-row
   weight under the relative rule is at least its weight under the absolute rule minus 0.05.

**What each outcome means, decided now.** All three hold -> the relative threshold is adopted as
EARN's default and that change is logged as a method change, with the DermaCon-IN result as its
motivation. Any one fails -> the absolute threshold stays, and R6/R7 are reported as a
dataset-specific finding about units rather than a defect to be fixed. Either way both arms are
reported.

**Status.** Written before the first run. Script: `scripts/19_earn_threshold.py`.

## R8 - Result: M2 on Fed-ISIC2019 rejected (11 Oct 2026)

Run as pre-registered in D10: both arms, S1 and S2, attacks none and A1, seeds 42/43/44, 100
rounds, frozen features. `docs/results/amendment_m2_fedisic.json`,
`docs/results/earn_threshold_amendment.csv`.

| criterion (D10) | absolute | relative | met |
| --- | --- | --- | --- |
| S1 no harm without attack (rare macro-F1) | 0.3538 | 0.3592 | yes |
| S2 no harm without attack (rare macro-F1) | 0.3084 | 0.2995 | yes |
| S1 attacker gains no more weight under A1 | 0.048 | **0.207** | **no** |
| S2 attacker gains no more weight under A1 | 0.263 | 0.263 | yes |
| S2 specialist still served | 0.500 | 0.500 | yes |

**Rejected, on the condition that was put in to catch it.** Without an attack the relative
threshold is harmless on both splits. Under A1 on S1 it is not: centre 4, which holds no image of
either rare class, goes from 4.8% of the rare rows to **20.7%** - it captures 4.3x the influence
the absolute rule allowed it. Rare macro-F1 under the same attack falls from 0.336 to 0.194.

The mechanism is the obvious one in hindsight. A threshold of "above this round's mean evidence"
is a threshold a liar can clear by claiming, because the claim itself is the evidence. The
absolute rule demands 20 real images, which a centre with none cannot fabricate without also
clearing a bar that has a unit attached to it. On S2 the two rules coincide (coverage 1: the
attacker's claim makes it a holder either way, and both arms give it 0.263), so S1 is where the
difference shows.

**Decision, as fixed in D10: the absolute threshold stays.** EARN is unchanged. R6 and R7 stand as
a dataset-specific finding about units - on DermaCon-IN's scale the absolute threshold admits
almost nobody and the mechanism idles - and *not* as a defect with a known fix. Anything that
replaces it has to be scale-free **and** unspoofable, which the round-mean rule is not. A
threshold in units a claimant does not control - a fraction of the dataset's own class count, say -
is the direction, and it would need its own pre-registration and its own A1 check.

**Two notes on the run.** The laptop suspended mid-run, so two of the 24 rows carry absurd wall
times (298 s and 47,513 s); the metrics are unaffected, since every run is seeded and
deterministic, but the `seconds` column should not be quoted. And both arms use the frozen
features, as D10 stated: the comparison is like for like, but these numbers are not grid numbers.

**What this entry is worth.** The relative threshold looked right on the dataset it was invented
on, and a pre-registered adversarial check rejected it on the dataset it was not. That is the
pre-registration earning its keep rather than decorating the process.

## D11 - A pre-registered search for a replacement evidence signal (11 Oct 2026) - NOT YET RUN

**What this is not.** G0a is decided and stays failed: the design doc's signal scored -0.21, its one
allowed retry -0.46, and amendment M1 0.11, all against a bar of 0.7 (R1). That verdict is closed.
This entry is a **search for a different signal**, run as an exploratory study with every candidate
declared before any of them is computed, so that reporting the best one afterwards is not a choice
made with hindsight.

**Why the current signal is inverted, as diagnosed from the existing results.** A client's class-c
classifier-row update under cross-entropy is a pull up from its own class-c samples and a push down
from every other sample it holds. The class-balanced loss gives a class the client does not hold a
weight of zero, so for such a client the row receives only downward pressure, in proportion to its
total images. Magnitude therefore tracks *size*, not holdings - hospitals that never see a disease
move its row the most.

**Candidates, fixed now.** All are computed from the same round-1 updates the gate used, with no
extra information from the clients:

| id | evidence e(k,c) | the thought |
| --- | --- | --- |
| C1 | signed bias change, per-client normalised: dB_c / sum_j abs(dB_j) | removes the per-client step-size effect that makes small centres look loud |
| C2 | signed weight-row change, per-client normalised by that client's total row movement | the same correction on the weight row the design doc used |
| C3 | cosine between a client's row-c update and the size-weighted mean row-c update of all clients | a peer-consensus direction, with no server data |
| C4 | probe-based, CELM-lite: synthesise one probe per class by gradient ascent on the global head's logit c, then e(k,c) = client k's logit c on that probe minus the global model's | evidence from behaviour rather than from update geometry - the family CELM reports working |
| C5 | C1 restricted to the positive part: max(0, C1) | holdings are non-negative; a push down is not evidence of absence-of-evidence |

**Measured exactly as G0a measures.** Round 1 of the Tier A loop, seed 42, TierAConfig defaults,
natural split S1, frozen features, updates clipped to the median norm. Per class, Spearman between
e(., c) and the true per-class training counts across the six centres; statistic = mean over the 8
classes. Reported alongside: Spearman against centre *size*, which is what the broken signal was
really tracking.

**Rules, fixed now.**
1. All five candidates are reported, in this order, pass or fail. No candidate is added afterwards.
2. A candidate reaching the 0.7 bar does **not** revive G0a and does not become EARN's signal. It
   would be a finding that a faithful signal exists, and nothing more, until it has cleared a fresh
   pre-registration that includes an A1 spoofability check - because R8 has just shown that a rule
   can be faithful on honest data and still hand influence to a liar.
3. If none reaches 0.7, that is reported as the result and Fallback F1 stands.

**Status.** Written before the first candidate was computed. Script: `scripts/20_evidence_search.py`.

## R9 - Result: the evidence search (11 Oct 2026) - no candidate clears the bar

All five candidates from D11, scored exactly as G0a scores, with the design doc's own signal
recomputed alongside for comparison. `docs/results/evidence_search.json`,
`scripts/20_evidence_search.py`.

| signal | mean Spearman vs true counts | vs centre size |
| --- | --- | --- |
| G0a, the design doc's signal | **-0.210** | -0.150 |
| C1 signed bias, per-client normalised | **+0.426** | +0.143 |
| C2 signed weight row, per-client normalised | +0.272 | +0.043 |
| C3 cosine with the consensus row direction | **+0.588** | **+0.493** |
| C4 probe-based, CELM-lite | -0.204 | +0.007 |
| C5 positive part of C1 | +0.281 | +0.058 |

**None reaches 0.7, so Fallback F1 stands and G0a stays failed.** What the search does settle is
*why* the original signal is inverted, which until now was a hypothesis.

**The inversion is a per-client scale artefact.** Dividing each client's class-c bias change by its
own total bias movement takes the statistic from -0.210 to +0.426 - a sign flip, from "hospitals
that never see a disease move its row the most" to a signal that ranks holdings the right way
round, on nothing but the information the gate already had. The absolute size of an update says
more about how far a client stepped than about what it holds; the shape of it says more.

**The best-scoring candidate is the least trustworthy.** C3 scores highest at +0.588, but its
correlation with centre *size* is +0.493 - most of its apparent skill is the thing we are trying
not to use, since size is what FedAvg already weights by. C1 scores lower and tracks size at only
+0.143, so on the project's own terms it is the better signal despite the smaller number. Ranking
candidates by the headline statistic alone would have picked the wrong one, which is the argument
for having declared the size column in advance.

**C4 is not a test of CELM.** C4 is a cheap stand-in - one probe per class, obtained by pushing a
feature vector up the *global* head's logit, then asking how much each client's head raises it.
CELM probes each client model with class-wise logit maximisation, 200 Adam steps per class-client,
and reports it working on Fed-ISIC2019. C4 scoring -0.204 says our shortcut does not work; it says
nothing about CELM, and must not be cited as evidence against it.

**Where this leaves the project.** A faithful-enough signal has not been found, but the search has
converted an unexplained failure into an understood one with a measured direction of travel: remove
per-client scale, and prefer a candidate whose skill does not come from size. Under D11 rule 2,
even a candidate that had cleared 0.7 would still need a fresh pre-registration with an A1
spoofability check before becoming EARN's signal - and C1, being a per-client *normalised* quantity,
is precisely the shape of rule a claiming attacker can game, which R8 showed is a separate failure
mode from unfaithfulness.

